#!/usr/bin/env python3
"""Zero-motion, fail-closed preflight for SO-101 phase-target hardware runs.

This entry point intentionally uses only the Python standard library.  It can
therefore reject missing devices, permissions, calibration, and planner assets
before the realtime server imports LeRobot or initializes a model on CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import pickletools
import stat
import sys
from typing import Any, Callable
import zipfile


REPO = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE = REPO / (
    "outputs/lerobot/so101/"
    "my_first_task_clean_v1_fcpg_context_bridge_val2_seed6126_2k/latest.pt"
)
DEFAULT_MODEL = REPO / (
    "outputs/lerobot/so101/so101_v9_local_multiscale_seed5131/latest.pt"
)
DEFAULT_GLOBAL_TIP = REPO / (
    "outputs/lerobot/so101/"
    "my_first_task_clean_v1_global_tip_seed6131_2k/best.pt"
)
DEFAULT_CALIBRATION = REPO / (
    "outputs/lerobot/so101/"
    "my_first_task_clean_v1_dual_camera_calibration/calibration.json"
)
DEFAULT_SECONDARY = REPO / (
    "data/lerobot/so101/cache/my_first_task_clean_v1_column_224"
)
DEFAULT_OUTPUT = REPO / "outputs/lerobot/so101/realtime_fcpg_lewm_mpc"
DEFAULT_ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115543-if00"
DEFAULT_WRIST_DEVICE = (
    "/dev/v4l/by-id/usb-XHH-260202-H_Integrated_Camera-video-index0"
)
DEFAULT_COLUMN_DEVICE = (
    "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"
)
CAMERA_ALIAS_ROOTS = (Path("/dev/v4l/by-id"),)
SERIAL_ALIAS_ROOTS = (Path("/dev/serial/by-id"), Path("/dev/serial/by-path"))
SECONDARY_CACHE_FILES = (
    "actions.npy",
    "episodes.npy",
    "frame_indices.npy",
    "frames.npy",
    "manifest.json",
    "states.npy",
    "tasks.npy",
)
LIVE_LEWM_CHECKPOINT_TOKENS = frozenset(
    {
        "model",
        "args",
        "run_config",
        "contract",
        "protocol",
        "frameskip",
        "history",
    }
)
INCOMPATIBLE_V10_FORMAT = "so101-v10-goal-isolated-v2"

#: Second RECOGNIZED schema, added rather than substituted.
#:
#: The frozen B2 export carries no ``run_config["contract"]`` block at all -- its
#: protocol lives at ``run_config["protocol"]`` -- while the legacy
#: StateResidualLeWM checkpoints carry ``run_config["contract"]["protocol"]``.
#: The production loader closes that gap at runtime by merging a sidecar
#: ``deployment_contract.json`` into ``run_config["contract"]``
#: (``scripts/visualize_so101_lewm.py:539-553``), which is why this checkpoint
#: loads, grounds and plans correctly on the real rig even though the raw ``.pt``
#: lacks the token this screen demands.
#:
#: So the fix is to teach this screen the second schema, NOT to drop ``contract``
#: from the shared requirement: the legacy branch below still requires it
#: exactly as before, and a legacy checkpoint missing it is still NO_GO.
B2_EXPORT_PROTOCOL = "so101-v10-dynamics-b2head-lodo-v1"
B2_RUNTIME_PROTOCOL = "so101-state-residual-lewm-b2head-runtime-v1"
DEPLOYMENT_CONTRACT_FILENAME = "deployment_contract.json"
DEPLOYMENT_CONTRACT_FORMAT = "so101-b2head-deployment-contract-v1"

#: The frozen-B2 branch requires every legacy token EXCEPT ``contract``, which
#: that generation genuinely does not contain.  Everything else still applies.
B2_CHECKPOINT_TOKENS = frozenset(LIVE_LEWM_CHECKPOINT_TOKENS - {"contract"})


def _frozen_b2_sidecar_contract(checkpoint: Path) -> dict[str, Any]:
    """Validate the sidecar that supplies the frozen B2 runtime contract.

    Deliberately computes NO digest.  ``load_deployment_contract`` binds the
    sidecar to its checkpoint by SHA-256 and that binding is load-bearing, but it
    belongs to the production loader that actually applies the contract.
    Recomputing it here would add checksum machinery to a cheap pre-screen and
    duplicate the authority for the same invariant in two places.  This screen
    only answers "is this the recognized frozen-B2 schema, and does its sidecar
    declare the runtime protocol the deployment path will resolve to".

    Raises ``ValueError`` -- fail closed -- on anything unrecognized.
    """

    sidecar = checkpoint.parent / DEPLOYMENT_CONTRACT_FILENAME
    if not sidecar.is_file():
        raise ValueError(
            "checkpoint identifies as the frozen B2 export but carries no "
            f"{DEPLOYMENT_CONTRACT_FILENAME} sidecar; the frozen .pt holds no "
            "runtime contract of its own, so it cannot be deployed without it"
        )
    try:
        with sidecar.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{DEPLOYMENT_CONTRACT_FILENAME} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{DEPLOYMENT_CONTRACT_FILENAME} must be a JSON object")
    if payload.get("format") != DEPLOYMENT_CONTRACT_FORMAT:
        raise ValueError(
            "deployment contract format is not "
            f"{DEPLOYMENT_CONTRACT_FORMAT!r}: {payload.get('format')!r}"
        )
    if payload.get("weights_unmodified") is not True:
        raise ValueError(
            "deployment contract does not assert weights_unmodified; the frozen "
            "scientific artifact must be deployed unmodified"
        )
    if payload.get("source_checkpoint_parent_protocol") != B2_EXPORT_PROTOCOL:
        raise ValueError(
            "deployment contract parent protocol does not match the checkpoint's "
            f"own export protocol {B2_EXPORT_PROTOCOL!r}: "
            f"{payload.get('source_checkpoint_parent_protocol')!r}"
        )
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("deployment contract has no contract block")
    # The DERIVED live runtime protocol, deliberately different from the
    # training/export protocol above.  Conflating the two is the whole hazard
    # this branch exists to avoid: the export protocol identifies the artifact,
    # the runtime protocol determines how the planner routes it -- including
    # whether causal memory activates.
    if contract.get("protocol") != B2_RUNTIME_PROTOCOL:
        raise ValueError(
            "deployment contract does not declare the frozen B2 live runtime "
            f"protocol {B2_RUNTIME_PROTOCOL!r}: {contract.get('protocol')!r}"
        )
    if contract.get("parent_protocol") != B2_EXPORT_PROTOCOL:
        raise ValueError(
            "deployment contract parent_protocol does not match "
            f"{B2_EXPORT_PROTOCOL!r}: {contract.get('parent_protocol')!r}"
        )
    for field in ("frameskip", "history", "state_dim"):
        value = contract.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"deployment contract {field!r} must be a positive integer, got "
                f"{value!r}"
            )
    return {
        "sidecar": str(sidecar),
        "format": DEPLOYMENT_CONTRACT_FORMAT,
        "export_protocol": B2_EXPORT_PROTOCOL,
        "runtime_protocol": B2_RUNTIME_PROTOCOL,
        "frameskip": contract["frameskip"],
        "history": contract["history"],
        "state_dim": contract["state_dim"],
        "weights_unmodified": True,
        # Stated rather than implied: this screen does not verify the sha binding.
        "sha256_binding_enforced_by": (
            "scripts/visualize_so101_lewm.py:load_deployment_contract "
            "(production loader)"
        ),
    }


class HardwarePreflightError(RuntimeError):
    """Raised when a hardware server start is rejected before model loading."""


def _path_under(path: Path, roots: tuple[Path, ...]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _stable_device(
    role: str,
    value: str | Path,
    *,
    alias_roots: tuple[Path, ...],
    access_mode: int,
    require_character_device: bool,
) -> dict[str, Any]:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or not _path_under(requested, alias_roots):
        roots = " or ".join(str(root) for root in alias_roots)
        raise ValueError(f"{role} must use a stable alias below {roots}: {requested}")
    if not requested.is_symlink():
        if not requested.exists():
            raise FileNotFoundError(f"{role} alias is missing or disconnected: {requested}")
        raise ValueError(f"{role} stable alias is not a symlink: {requested}")

    resolved = requested.resolve(strict=True)
    mode = resolved.stat().st_mode
    if require_character_device and not stat.S_ISCHR(mode):
        raise ValueError(f"{role} does not resolve to a character device: {resolved}")
    if not os.access(resolved, access_mode):
        permission = "read/write" if access_mode & os.W_OK else "read"
        raise PermissionError(f"{role} lacks {permission} permission: {resolved}")
    return {
        "requested_alias": str(requested),
        "resolved_device": str(resolved),
        "character_device": stat.S_ISCHR(mode),
    }


def _camera_device(
    role: str,
    value: str | Path,
    *,
    sysfs_root: Path,
    alias_roots: tuple[Path, ...],
    require_character_device: bool,
) -> dict[str, Any]:
    requested = Path(value).expanduser()
    if len(alias_roots) != 1 or alias_roots[0].name != "by-id":
        raise ValueError(
            "camera alias contract requires exactly one /dev/v4l/by-id root"
        )
    alias_root = alias_roots[0]
    if not requested.is_absolute() or requested.parent != alias_root:
        raise ValueError(
            f"{role} camera must use a direct alias below {alias_root}: "
            f"{requested}"
        )
    result = _stable_device(
        f"{role} camera",
        requested,
        alias_roots=alias_roots,
        access_mode=os.R_OK,
        require_character_device=require_character_device,
    )
    resolved = Path(result["resolved_device"])
    sysfs_device = sysfs_root / resolved.name / "device"
    if not sysfs_device.exists():
        raise FileNotFoundError(
            f"{role} camera has no sysfs physical-device binding: {sysfs_device}"
        )
    physical = sysfs_device.resolve(strict=True)
    usb_parent: Path | None = None
    for candidate in (physical, *physical.parents):
        if (candidate / "idVendor").is_file() and (candidate / "idProduct").is_file():
            usb_parent = candidate
            break
    if usb_parent is None:
        raise ValueError(
            f"{role} camera sysfs path has no USB vendor/product identity: {physical}"
        )
    result.update(
        {
            "physical_device": str(usb_parent),
            "usb_vendor": (usb_parent / "idVendor").read_text(encoding="ascii").strip(),
            "usb_product": (usb_parent / "idProduct").read_text(encoding="ascii").strip(),
        }
    )
    return result


def _file_record(path: str | Path, *, role: str) -> dict[str, Any]:
    requested = Path(path).expanduser()
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{role} is not a regular file: {resolved}")
    size = resolved.stat().st_size
    if size <= 0:
        raise ValueError(f"{role} is empty: {resolved}")
    if not os.access(resolved, os.R_OK):
        raise PermissionError(f"{role} is not readable: {resolved}")
    return {"path": str(resolved), "bytes": size}


def _live_lewm_checkpoint(path: str | Path) -> dict[str, Any]:
    """Screen the live model schema without importing torch or model code."""

    record = _file_record(path, role="LeWM checkpoint")
    resolved = Path(record["path"])
    try:
        with zipfile.ZipFile(resolved) as archive:
            payloads = [
                item
                for item in archive.infolist()
                if item.filename.endswith("data.pkl")
            ]
            if len(payloads) != 1:
                raise ValueError("checkpoint archive must contain exactly one data.pkl")
            payload = payloads[0]
            if payload.file_size > 32 * 1024 * 1024:
                raise ValueError("checkpoint metadata exceeds the preflight size limit")
            strings = {
                argument
                for opcode, argument, _ in pickletools.genops(archive.read(payload))
                if opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}
                and isinstance(argument, str)
            }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError("checkpoint is not a readable torch ZIP archive") from exc
    if INCOMPATIBLE_V10_FORMAT in strings:
        raise ValueError(
            "goal-isolated V10 checkpoints are incompatible with the phase-target "
            "StateResidualLeWM planner"
        )
    # TWO recognized schemas.  The branch is chosen by a POSITIVE identifier --
    # the frozen B2 export protocol string -- not by the absence of `contract`,
    # so an unknown checkpoint that merely happens to omit `contract` still
    # falls through to the legacy branch and is rejected there.
    if B2_EXPORT_PROTOCOL in strings:
        missing = sorted(B2_CHECKPOINT_TOKENS.difference(strings))
        if missing:
            raise ValueError(
                "frozen B2 checkpoint lacks required tokens: " + ", ".join(missing)
            )
        sidecar = _frozen_b2_sidecar_contract(resolved)
        return {
            **record,
            "lightweight_schema": "so101_b2head_frozen_export",
            "recognized_schema": "frozen_b2",
            "required_contract_tokens": sorted(B2_CHECKPOINT_TOKENS),
            "deployment_contract": sidecar,
            "torch_imported": False,
        }

    missing = sorted(LIVE_LEWM_CHECKPOINT_TOKENS.difference(strings))
    if missing:
        raise ValueError(
            "checkpoint lacks the live StateResidualLeWM contract tokens: "
            + ", ".join(missing)
        )
    return {
        **record,
        "lightweight_schema": "train_so101_lewm_state_residual",
        "recognized_schema": "legacy_state_residual",
        "required_contract_tokens": sorted(LIVE_LEWM_CHECKPOINT_TOKENS),
        "torch_imported": False,
    }


def _json_record(
    path: str | Path,
    *,
    role: str,
    validate: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    result = _file_record(path, role=role)
    payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    if validate is not None:
        validate(payload)
    result["json_type"] = type(payload).__name__
    return result


def _require_object(payload: Any) -> None:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("JSON contract must be a non-empty object")


def _validate_home_state(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("home-state report must be a JSON object")
    state = payload.get("current_state")
    if (
        not isinstance(state, list)
        or len(state) != 6
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in state)
        or any(not math.isfinite(float(value)) for value in state)
    ):
        raise ValueError("home-state report must contain six finite current_state values")


def _validate_live_anchor(payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("live-anchor calibration must use version 1")
    if payload.get("view") != "column":
        raise ValueError("live-anchor calibration must bind the column view")


def _secondary_cache(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"secondary cache is not a directory: {resolved}")
    files: dict[str, int] = {}
    for name in SECONDARY_CACHE_FILES:
        record = _file_record(resolved / name, role=f"secondary cache {name}")
        files[name] = int(record["bytes"])
    _json_record(
        resolved / "manifest.json",
        role="secondary cache manifest",
        validate=_require_object,
    )
    return {"path": str(resolved), "files": files}


def _output_location(path: str | Path) -> dict[str, Any]:
    requested = Path(path).expanduser().absolute()
    if requested.exists() and not requested.is_dir():
        raise ValueError(f"output path exists but is not a directory: {requested}")
    ancestor = requested
    while not ancestor.exists():
        if ancestor.parent == ancestor:
            raise FileNotFoundError(f"output path has no existing parent: {requested}")
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK | os.X_OK):
        raise PermissionError(f"output parent is not writable: {ancestor}")
    return {"path": str(requested), "writable_parent": str(ancestor)}


def _default_robot_calibration_dir() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    lerobot_home = Path(
        os.environ.get("HF_LEROBOT_HOME", str(hf_home / "lerobot"))
    ).expanduser()
    calibration_root = Path(
        os.environ.get(
            "HF_LEROBOT_CALIBRATION", str(lerobot_home / "calibration")
        )
    ).expanduser()
    return calibration_root / "robots" / "so101_follower"


def build_hardware_preflight_report(
    *,
    bridge_checkpoint: str | Path,
    model_checkpoint: str | Path,
    global_tip_checkpoint: str | Path | None,
    disable_global_tip_locator: bool,
    calibration: str | Path,
    secondary_cache: str | Path,
    live_anchor_calibration: str | Path | None,
    home_state_report: str | Path | None,
    output: str | Path,
    robot_port: str | Path,
    robot_id: str,
    calibration_dir: str | Path | None,
    wrist_device: str | Path,
    column_device: str | Path,
    execute_enabled: bool,
    acknowledge_workspace_clear: bool,
    gripper_calibration_enabled: bool = False,
    control_mode: str = "phase_target",
    sysfs_root: Path = Path("/sys/class/video4linux"),
    camera_alias_roots: tuple[Path, ...] = CAMERA_ALIAS_ROOTS,
    serial_alias_roots: tuple[Path, ...] = SERIAL_ALIAS_ROOTS,
    require_character_devices: bool = True,
) -> dict[str, Any]:
    """Return a JSON-safe report without opening a camera or robot port."""

    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, operation: Callable[[], dict[str, Any]]) -> None:
        try:
            checks[name] = {"status": "pass", "details": operation()}
        except Exception as exc:
            checks[name] = {
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    def execution_contract() -> dict[str, Any]:
        motion_enabled = bool(execute_enabled or gripper_calibration_enabled)
        if control_mode != "phase_target":
            raise ValueError("real hardware requires --control-mode phase_target")
        if motion_enabled and not acknowledge_workspace_clear:
            raise ValueError(
                "hardware motion permission requires --acknowledge-workspace-clear"
            )
        return {
            "control_mode": control_mode,
            "execute_enabled": bool(execute_enabled),
            "gripper_calibration_enabled": bool(gripper_calibration_enabled),
            "motion_permission_enabled": motion_enabled,
            "workspace_clear_acknowledged": bool(acknowledge_workspace_clear),
            "motion_sent": False,
        }

    check("execution_contract", execution_contract)
    check(
        "wrist_camera",
        lambda: _camera_device(
            "wrist",
            wrist_device,
            sysfs_root=sysfs_root,
            alias_roots=camera_alias_roots,
            require_character_device=require_character_devices,
        ),
    )
    check(
        "column_camera",
        lambda: _camera_device(
            "column",
            column_device,
            sysfs_root=sysfs_root,
            alias_roots=camera_alias_roots,
            require_character_device=require_character_devices,
        ),
    )

    def camera_roles() -> dict[str, Any]:
        wrist = checks.get("wrist_camera", {})
        column = checks.get("column_camera", {})
        if wrist.get("status") != "pass" or column.get("status") != "pass":
            raise ValueError("both camera device checks must pass before role binding")
        wrist_identity = wrist["details"]["physical_device"]
        column_identity = column["details"]["physical_device"]
        if wrist_identity == column_identity:
            raise ValueError(
                "wrist and column aliases resolve to the same physical USB device: "
                f"{wrist_identity}"
            )
        return {
            "schema": "so101-camera-role-manifest-v1",
            "wrist_physical_device": wrist_identity,
            "column_physical_device": column_identity,
        }

    check("camera_role_binding", camera_roles)
    check(
        "robot_port",
        lambda: _stable_device(
            "SO-101 robot port",
            robot_port,
            alias_roots=serial_alias_roots,
            access_mode=os.R_OK | os.W_OK,
            require_character_device=require_character_devices,
        ),
    )
    robot_calibration_dir = (
        Path(calibration_dir).expanduser()
        if calibration_dir is not None
        else _default_robot_calibration_dir()
    )
    check(
        "robot_calibration",
        lambda: _json_record(
            robot_calibration_dir / f"{robot_id}.json",
            role="LeRobot SO-101 calibration",
            validate=_require_object,
        ),
    )
    check(
        "bridge_checkpoint",
        lambda: _file_record(bridge_checkpoint, role="FCPG bridge checkpoint"),
    )
    check(
        "model_checkpoint",
        lambda: _live_lewm_checkpoint(model_checkpoint),
    )
    if disable_global_tip_locator:
        checks["global_tip_checkpoint"] = {
            "status": "pass",
            "details": {"disabled": True},
        }
    elif global_tip_checkpoint is None:
        checks["global_tip_checkpoint"] = {
            "status": "fail",
            "error_type": "ValueError",
            "error": "global tip locator is enabled but its checkpoint is unset",
        }
    else:
        check(
            "global_tip_checkpoint",
            lambda: _file_record(
                global_tip_checkpoint, role="global-tip checkpoint"
            ),
        )
    check(
        "camera_calibration",
        lambda: _json_record(
            calibration,
            role="dual-camera calibration",
            validate=_require_object,
        ),
    )
    check("secondary_cache", lambda: _secondary_cache(secondary_cache))
    if live_anchor_calibration is None:
        checks["live_anchor_calibration"] = {
            "status": "pass",
            "details": {"provided": False},
        }
    else:
        check(
            "live_anchor_calibration",
            lambda: _json_record(
                live_anchor_calibration,
                role="live-anchor calibration",
                validate=_validate_live_anchor,
            ),
        )
    if home_state_report is None:
        checks["home_state_report"] = {
            "status": "pass",
            "details": {"provided": False},
        }
    else:
        check(
            "home_state_report",
            lambda: _json_record(
                home_state_report,
                role="home-state report",
                validate=_validate_home_state,
            ),
        )
    check("output", lambda: _output_location(output))

    failures = [name for name, result in checks.items() if result["status"] != "pass"]
    return {
        "schema": "so101-phase-target-hardware-preflight-v1",
        "passed": not failures,
        "decision": "GO" if not failures else "NO_GO",
        "zero_motion": True,
        "heavy_runtime_imported": False,
        "failed_checks": failures,
        "checks": checks,
    }


def require_hardware_preflight(report: dict[str, Any]) -> None:
    if report.get("passed") is True:
        return
    failures = report.get("failed_checks", [])
    details = []
    for name in failures:
        result = report.get("checks", {}).get(name, {})
        details.append(f"{name}: {result.get('error', 'failed')}")
    raise HardwarePreflightError(
        "SO-101 hardware preflight NO_GO; " + "; ".join(details)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--model-checkpoint", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--global-tip-checkpoint", type=Path, default=DEFAULT_GLOBAL_TIP)
    parser.add_argument("--disable-global-tip-locator", action="store_true")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--secondary-cache", type=Path, default=DEFAULT_SECONDARY)
    parser.add_argument("--live-anchor-calibration", type=Path)
    parser.add_argument(
        "--home-state-report", "--object-prior-report", dest="home_state_report", type=Path
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default="myfollower01")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--wrist-device", default=DEFAULT_WRIST_DEVICE)
    parser.add_argument("--column-device", default=DEFAULT_COLUMN_DEVICE)
    parser.add_argument("--control-mode", default="phase_target")
    parser.add_argument("--execute-enabled", action="store_true")
    parser.add_argument("--gripper-calibration-enabled", action="store_true")
    parser.add_argument("--acknowledge-workspace-clear", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        help="Optionally persist the JSON report; stdout is always populated.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_hardware_preflight_report(
        bridge_checkpoint=args.bridge_checkpoint,
        model_checkpoint=args.model_checkpoint,
        global_tip_checkpoint=args.global_tip_checkpoint,
        disable_global_tip_locator=args.disable_global_tip_locator,
        calibration=args.calibration,
        secondary_cache=args.secondary_cache,
        live_anchor_calibration=args.live_anchor_calibration,
        home_state_report=args.home_state_report,
        output=args.output,
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        calibration_dir=args.calibration_dir,
        wrist_device=args.wrist_device,
        column_device=args.column_device,
        execute_enabled=args.execute_enabled,
        gripper_calibration_enabled=args.gripper_calibration_enabled,
        acknowledge_workspace_clear=args.acknowledge_workspace_clear,
        control_mode=args.control_mode,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    print(rendered)
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
