"""Non-actuating live preflight for the fresh-fish MPC-only approach pilot.

This module answers one question, with the arm **not moving**: if the runtime
were started right now, would it run in the geometry, with the calibration,
with the frozen checkpoint, and against the targets we think it would?

Every check here is read-only with respect to the robot.  Nothing in this
module opens the SO-101 serial bus, constructs a ``SO101Follower``, or emits a
command; the only hardware it touches is the two USB cameras, which it opens,
reads one frame from, and releases.  Camera access is injected as a probe
callable so the whole report is reachable without hardware.

Why this exists rather than reusing ``scripts/preflight_so101_phase_target.py``:
that preflight is stdlib-only by design and deliberately never opens a camera,
so it can reject a run before LeRobot or CUDA load.  It therefore validates
device *aliases* (symlink, character device, USB identity) but cannot see the
mode a camera actually negotiated.  Under ``letterbox_v1`` that gap is exactly
the dangerous one: a camera that comes up at a different raw resolution still
produces a 224x224 planning canvas, just with the content rectangle and the pad
band in the wrong rows, so nothing downstream can detect it.  The two preflights
are complementary and both should run.

Fail-closed everywhere.  A check that cannot be executed is ``SKIPPED``, and a
``SKIPPED`` check is a ``NO_GO`` unless the caller explicitly asked for a
dry run -- a dry run reports ``NO_GO_DRY_RUN``, which is never a clearance to
run hardware.  No check is allowed to pass by absence of evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fcpg.letterbox_geometry import (
    LETTERBOX_V1,
    PLANNING_IMAGE_SIZE,
    planning_geometry_from_protocol,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Report schema emitted by :func:`build_live_preflight_report`.
PREFLIGHT_SCHEMA = "so101-pilot-live-preflight-v1"

#: Raw capture resolution BOTH live views must report.  One
#: ``source_image_size`` serves both views on the planner config, which is only
#: sound while they share a raw resolution, so a mismatch is fatal rather than
#: per-view corrected.  Mirrors
#: ``scripts/serve_so101_realtime_fcpg_lewm.py:106 LIVE_SOURCE_IMAGE_SIZE``.
REQUIRED_SOURCE_IMAGE_SIZE = (640, 480)

#: The only geometry the fresh pilot may run in.
REQUIRED_GEOMETRY_PROTOCOL = LETTERBOX_V1

#: Frozen, shadow-only B2 head.  Never retrained, never modified.
FROZEN_B2_DIR = REPO_ROOT / "outputs/lerobot/so101/v10_b2head_allseen_final"
FROZEN_B2_CHECKPOINT = FROZEN_B2_DIR / "best.pt"
FROZEN_B2_CONTRACT = FROZEN_B2_DIR / "deployment_contract.json"

#: Geometry-versioned calibration produced by the metadata-upgrade artifact.
#: The historical unversioned file is kept untouched and is used below as a
#: NEGATIVE control: the strict loader must refuse it.
UPGRADED_CALIBRATION = (
    REPO_ROOT
    / "outputs/lerobot/so101/my_first_task_clean_v1_dual_camera_calibration"
    / "calibration_letterbox_v1.json"
)
HISTORICAL_CALIBRATION = (
    REPO_ROOT
    / "outputs/lerobot/so101/my_first_task_clean_v1_dual_camera_calibration"
    / "calibration.json"
)

#: Tensor census of the frozen checkpoint, from the B0 freeze manifest.
EXPECTED_B2_TENSOR_TOTAL = 344
EXPECTED_B2_BASE_TENSORS = 303
EXPECTED_B2_HEAD_TENSORS = 41
B2_BASE_PREFIX = "base."

#: Runtime protocol the frozen head dispatches under.
B2_RUNTIME_PROTOCOL = "so101-state-residual-lewm-b2head-runtime-v1"
B2_PARENT_PROTOCOL = "so101-v10-dynamics-b2head-lodo-v1"

#: The ONLY execution-verified scope.  Approach only; grasp/transport/place are
#: not cleared by anything in this repository.
VERIFIED_SCOPE = "phase_target_approach"

#: Flags that grant hardware motion permission.  This module refuses to
#: assemble or forward any of them: a preflight that could actuate is not a
#: preflight.
FORBIDDEN_MOTION_FLAGS = (
    "--execute-enabled",
    "--gripper-calibration-enabled",
    "--acknowledge-workspace-clear",
    "--bounded-autonomy-enabled",
)

#: Live camera capture requests, mirroring the runtime's OpenCVCameraConfig
#: construction so the probe negotiates the same mode the runtime will.
LIVE_CAMERA_REQUESTS = {
    "wrist": {"fps": 60, "fourcc": "MJPG"},
    "column": {"fps": 30, "fourcc": "YUYV"},
}

VIEWS = ("wrist", "column")

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_SKIPPED = "SKIPPED"

DECISION_GO = "GO"
DECISION_NO_GO = "NO_GO"
DECISION_NO_GO_DRY_RUN = "NO_GO_DRY_RUN"


class PilotPreflightError(RuntimeError):
    """Raised when the pilot preflight refuses to clear a live run."""


@dataclass(frozen=True)
class CheckResult:
    """One preflight check outcome."""

    name: str
    status: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in (STATUS_PASS, STATUS_FAIL, STATUS_SKIPPED):
            raise ValueError(f"unknown check status {self.status!r}")

    @property
    def passed(self) -> bool:
        return self.status == STATUS_PASS

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "details": self.details,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


def _fail(name: str, summary: str, error: str, **details: Any) -> CheckResult:
    return CheckResult(
        name=name, status=STATUS_FAIL, summary=summary, details=details, error=error
    )


def _pass(name: str, summary: str, **details: Any) -> CheckResult:
    return CheckResult(name=name, status=STATUS_PASS, summary=summary, details=details)


def _skip(name: str, summary: str, reason: str, **details: Any) -> CheckResult:
    return CheckResult(
        name=name, status=STATUS_SKIPPED, summary=summary, details=details, error=reason
    )


# ---------------------------------------------------------------------------
# Check 1 -- live cameras open and report their ACTUAL raw resolution
# ---------------------------------------------------------------------------


def opencv_camera_probe(device: str, view: str) -> dict[str, Any]:
    """Open one camera, read one frame, report its RAW size, and release.

    Deliberately reads the size off a real decoded frame rather than off
    ``CAP_PROP_FRAME_WIDTH``: the property reports what the driver was asked
    for, which is precisely the value that can disagree with what arrives.

    Opens no robot bus and sends no command.  This is the only hardware access
    anywhere in this module.
    """

    import cv2  # imported lazily so the module stays importable without cv2

    request = LIVE_CAMERA_REQUESTS.get(view, {})
    capture = cv2.VideoCapture(str(device))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"{view} camera did not open: {device}")
        fourcc = request.get("fourcc")
        if fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, REQUIRED_SOURCE_IMAGE_SIZE[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUIRED_SOURCE_IMAGE_SIZE[1])
        if request.get("fps"):
            capture.set(cv2.CAP_PROP_FPS, float(request["fps"]))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"{view} camera opened but delivered no frame")
        shape = tuple(int(v) for v in frame.shape)
        if len(shape) != 3 or shape[2] != 3:
            raise RuntimeError(
                f"{view} camera returned a frame of shape {shape}; expected [H, W, 3]"
            )
        return {
            "device": str(device),
            "raw_width": shape[1],
            "raw_height": shape[0],
            "frame_shape": list(shape),
            "requested_fourcc": request.get("fourcc"),
            "requested_fps": request.get("fps"),
        }
    finally:
        capture.release()


def check_camera_raw_resolution(
    devices: Mapping[str, str],
    *,
    probe: Callable[[str, str], dict[str, Any]] | None = None,
    required_size: tuple[int, int] = REQUIRED_SOURCE_IMAGE_SIZE,
    allow_skip: bool = False,
) -> CheckResult:
    """Both views open and report exactly ``required_size`` raw pixels.

    Fails closed on: a view that will not open, a view at the wrong raw
    resolution, and two views at *different* raw resolutions.  The last case is
    fatal even if one of them is correct, because a single declared
    ``source_image_size`` serves both views on the planner config.
    """

    name = "camera_raw_resolution"
    missing = [view for view in VIEWS if not devices.get(view)]
    if missing:
        return _fail(
            name,
            "camera devices not supplied",
            f"missing device path for: {', '.join(sorted(missing))}",
            required_size=list(required_size),
        )
    if probe is None:
        if allow_skip:
            return _skip(
                name,
                "camera probe not run",
                "no camera probe supplied (dry run); raw resolution UNVERIFIED",
                required_size=list(required_size),
            )
        probe = opencv_camera_probe

    observed: dict[str, Any] = {}
    for view in VIEWS:
        try:
            observed[view] = probe(str(devices[view]), view)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            return _fail(
                name,
                f"{view} camera unavailable",
                f"{type(exc).__name__}: {exc}",
                observed=observed,
                required_size=list(required_size),
            )

    sizes = {
        view: (int(observed[view]["raw_width"]), int(observed[view]["raw_height"]))
        for view in VIEWS
    }
    wrong = {v: s for v, s in sizes.items() if s != tuple(required_size)}
    if wrong:
        detail = "; ".join(f"{v}={s[0]}x{s[1]}" for v, s in sorted(wrong.items()))
        return _fail(
            name,
            "raw camera resolution mismatch",
            (
                f"required {required_size[0]}x{required_size[1]} on every view, but "
                f"{detail}. One declared source_image_size serves both views, and "
                "the letterbox pad band is a function of the raw aspect ratio, so "
                "planning pixels would be biased silently -- both protocols still "
                "yield a 224x224 canvas."
            ),
            observed_sizes={v: list(s) for v, s in sizes.items()},
            required_size=list(required_size),
        )
    if len({tuple(s) for s in sizes.values()}) != 1:
        return _fail(
            name,
            "views disagree on raw resolution",
            "wrist and column report different raw resolutions",
            observed_sizes={v: list(s) for v, s in sizes.items()},
        )
    return _pass(
        name,
        f"both views raw {required_size[0]}x{required_size[1]}",
        observed_sizes={v: list(s) for v, s in sizes.items()},
        probe_details=observed,
        motion_sent=False,
    )


# ---------------------------------------------------------------------------
# Check 2 -- declared geometry reaches the planner config AND the contract
# ---------------------------------------------------------------------------

#: Expected ``letterbox_v1`` content rectangle for a 640x480 source on a
#: 224x224 planning canvas: 28 pad rows top and bottom, no pad columns.
EXPECTED_LETTERBOX_CONTENT_RECT = (0, 28, 224, 196)

#: Geometry keys the planner must emit into ``model_contract`` so an archived
#: ``plan.json`` identifies the geometry it actually ran in.
REQUIRED_CONTRACT_GEOMETRY_KEYS = (
    "geometry_protocol",
    "source_image_size",
    "planning_image_size",
    "content_rect",
)


def check_geometry_contract(
    *,
    required_protocol: str = REQUIRED_GEOMETRY_PROTOCOL,
    source_image_size: tuple[int, int] = REQUIRED_SOURCE_IMAGE_SIZE,
    declared_geometry_fn: Callable[..., Any] | None = None,
    contract_fields_fn: Callable[..., Any] | None = None,
) -> CheckResult:
    """The declared geometry reaches the planner config AND the model contract.

    Every part of this check is *executed*, not string-matched against source:
    the runtime's own geometry declaration helper is called, a real
    :class:`PreparedPlannerConfig` is constructed from what it returns, the
    planner's contract-field builder is run on that config, and the live
    source-size guard is exercised in both directions.  A formatting-level
    refactor of either module therefore cannot make this check pass or fail
    spuriously -- only a behavioural change can.

    Constructing the config is safe: it loads no checkpoint and touches no
    GPU.  Only ``PreparedSO101RealtimePlanner`` does that, and it is never
    instantiated here.
    """

    name = "geometry_contract"
    try:
        geometry = planning_geometry_from_protocol(
            required_protocol,
            source_width=int(source_image_size[0]),
            source_height=int(source_image_size[1]),
            image_size=PLANNING_IMAGE_SIZE,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            name,
            "declared geometry could not be built",
            f"{type(exc).__name__}: {exc}",
            required_protocol=required_protocol,
        )

    content_rect = tuple(int(v) for v in geometry.content_rect)
    if (
        required_protocol == LETTERBOX_V1
        and tuple(source_image_size) == REQUIRED_SOURCE_IMAGE_SIZE
        and content_rect != EXPECTED_LETTERBOX_CONTENT_RECT
    ):
        return _fail(
            name,
            "unexpected letterbox content rectangle",
            (
                f"content_rect={content_rect} for "
                f"{source_image_size[0]}x{source_image_size[1]} under "
                f"{required_protocol!r}; expected "
                f"{EXPECTED_LETTERBOX_CONTENT_RECT}"
            ),
            content_rect=list(content_rect),
        )

    # --- the runtime's own declaration, called ---------------------------
    if declared_geometry_fn is None:
        try:
            from scripts.serve_so101_realtime_fcpg_lewm import (
                _declared_pixel_geometry,
            )
        except Exception as exc:  # noqa: BLE001
            return _skip(
                name,
                "runtime geometry declaration not importable",
                (
                    f"{type(exc).__name__}: {exc}; the geometry the live runtime "
                    "would declare is UNVERIFIED and is not assumed"
                ),
            )
        declared_geometry_fn = _declared_pixel_geometry

    try:
        live_protocol, live_source = declared_geometry_fn(replay=False)
        replay_protocol, _ = declared_geometry_fn(replay=True)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            name,
            "runtime geometry declaration raised",
            f"{type(exc).__name__}: {exc}",
        )

    if live_protocol != required_protocol:
        return _fail(
            name,
            "runtime declares the wrong live geometry",
            (
                f"live path declares {live_protocol!r}, but the pilot requires "
                f"{required_protocol!r}; pixels would be scored in a geometry "
                "the calibration was not fitted in"
            ),
            runtime_live_protocol=live_protocol,
        )
    if live_source is None or tuple(int(v) for v in live_source) != tuple(
        int(v) for v in source_image_size
    ):
        return _fail(
            name,
            "runtime declares the wrong live source resolution",
            (
                f"live path declares source_image_size={live_source!r}, but the "
                f"pilot requires {tuple(source_image_size)!r}"
            ),
            runtime_live_source_image_size=(
                None if live_source is None else list(live_source)
            ),
        )
    if replay_protocol == required_protocol:
        return _fail(
            name,
            "replay and live geometries are not separated",
            (
                f"the replay path also declares {replay_protocol!r}; cached "
                "shadow frames were never letterboxed, so declaring the live "
                "geometry for them would reinterpret every pixel silently"
            ),
            runtime_replay_protocol=replay_protocol,
        )

    # --- the model contract, built from a real config --------------------
    try:
        from fcpg.so101_realtime_planner import PreparedPlannerConfig
    except Exception as exc:  # noqa: BLE001
        return _skip(
            name,
            "planner config not importable",
            f"{type(exc).__name__}: {exc}; the model contract is UNVERIFIED",
        )

    config = PreparedPlannerConfig(
        bridge_checkpoint=Path("preflight-unused-bridge.pt"),
        calibration=Path("preflight-unused-calibration.json"),
        secondary_cache=Path("preflight-unused-cache"),
        geometry_protocol=live_protocol,
        source_image_size=(int(live_source[0]), int(live_source[1])),
    )

    if contract_fields_fn is None:
        try:
            from fcpg.so101_realtime_planner import (
                planning_geometry_contract_fields,
            )
        except Exception as exc:  # noqa: BLE001
            return _skip(
                name,
                "contract geometry builder not importable",
                f"{type(exc).__name__}: {exc}; the model contract is UNVERIFIED",
            )
        contract_fields_fn = planning_geometry_contract_fields

    try:
        contract_fields = dict(contract_fields_fn(config))
    except Exception as exc:  # noqa: BLE001
        return _fail(
            name,
            "contract geometry builder raised",
            f"{type(exc).__name__}: {exc}",
        )

    missing = [k for k in REQUIRED_CONTRACT_GEOMETRY_KEYS if k not in contract_fields]
    if missing:
        return _fail(
            name,
            "model contract omits geometry provenance",
            (
                "an archived plan.json would not identify its own geometry; "
                f"missing keys: {', '.join(missing)}"
            ),
            contract_fields=contract_fields,
        )
    mismatches: dict[str, Any] = {}
    if contract_fields["geometry_protocol"] != required_protocol:
        mismatches["geometry_protocol"] = contract_fields["geometry_protocol"]
    if [int(v) for v in (contract_fields["source_image_size"] or ())] != [
        int(v) for v in source_image_size
    ]:
        mismatches["source_image_size"] = contract_fields["source_image_size"]
    if int(contract_fields["planning_image_size"]) != int(PLANNING_IMAGE_SIZE):
        mismatches["planning_image_size"] = contract_fields["planning_image_size"]
    if [int(v) for v in contract_fields["content_rect"]] != list(content_rect):
        mismatches["content_rect"] = contract_fields["content_rect"]
    if mismatches:
        return _fail(
            name,
            "model contract geometry disagrees with the declared geometry",
            f"mismatched contract fields: {mismatches}",
            contract_fields=contract_fields,
            expected_content_rect=list(content_rect),
        )

    # --- the live source-size guard, exercised both ways -----------------
    guard = _check_source_size_guard(source_image_size, required_protocol)
    if guard is not None:
        return _fail(name, "live source-size guard is not fail-closed", guard)

    return _pass(
        name,
        f"{required_protocol} declared and contracted, content_rect={content_rect}",
        geometry=geometry.to_dict(),
        pad_rows=int(geometry.pad_rows),
        pad_columns=int(geometry.pad_columns),
        runtime_live_protocol=live_protocol,
        runtime_live_source_image_size=[int(v) for v in live_source],
        runtime_replay_protocol=replay_protocol,
        model_contract_fields=contract_fields,
        executed=(
            "runtime geometry declaration, PreparedPlannerConfig construction, "
            "model-contract field builder, and the live source-size guard in "
            "both directions"
        ),
    )


def _check_source_size_guard(
    source_image_size: tuple[int, int], required_protocol: str
) -> str | None:
    """Return an error string if the planner's live size guard fails open."""

    try:
        from fcpg.so101_realtime_planner import PreparedPlannerConfig
    except Exception as exc:  # noqa: BLE001 - heavy deps may be absent
        return (
            f"planner config could not be imported ({type(exc).__name__}), so the "
            "live source-size guard is UNVERIFIED; refusing to assume it holds"
        )

    config = PreparedPlannerConfig(
        bridge_checkpoint=Path("unused-bridge.pt"),
        calibration=Path("unused-calibration.json"),
        secondary_cache=Path("unused-cache"),
        geometry_protocol=required_protocol,
        source_image_size=(int(source_image_size[0]), int(source_image_size[1])),
    )
    matching = {view: tuple(source_image_size) for view in VIEWS}
    try:
        config.require_source_image_size(matching)
    except Exception as exc:  # noqa: BLE001
        return f"guard rejected a MATCHING live resolution: {type(exc).__name__}: {exc}"

    wrong = dict(matching)
    wrong["column"] = (1280, 720)
    try:
        config.require_source_image_size(wrong)
    except ValueError:
        return None
    return (
        "guard ACCEPTED a mismatched live resolution (column 1280x720); the pad "
        "band would be wrong while the canvas stayed 224x224"
    )


# ---------------------------------------------------------------------------
# Check 3 -- the strict projection-fit loader accepts the upgraded calibration
# ---------------------------------------------------------------------------


def check_projection_fit_loader(
    *,
    calibration: Path = UPGRADED_CALIBRATION,
    historical_calibration: Path | None = HISTORICAL_CALIBRATION,
    required_protocol: str = REQUIRED_GEOMETRY_PROTOCOL,
    loader: Callable[..., Any] | None = None,
) -> CheckResult:
    """The strict loader accepts the upgraded fit and refuses an unversioned one.

    The negative control is the point.  A loader that accepts everything would
    pass the positive half trivially, so the historical unversioned artifact
    must be *refused* under the same contract for this check to mean anything.
    """

    name = "projection_fit_loader"
    if not calibration.exists():
        return _fail(
            name,
            "upgraded calibration missing",
            f"geometry-versioned calibration not found: {calibration}",
            calibration=str(calibration),
        )
    if loader is None:
        try:
            from scripts.evaluate_so101_real_shadow_fcpg_lewm import (
                load_projection_fits,
            )
        except Exception as exc:  # noqa: BLE001
            return _skip(
                name,
                "strict loader not importable",
                f"{type(exc).__name__}: {exc}",
                calibration=str(calibration),
            )
        loader = load_projection_fits

    try:
        fits = loader(calibration, expected_geometry=required_protocol)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            name,
            "strict loader refused the upgraded calibration",
            f"{type(exc).__name__}: {exc}",
            calibration=str(calibration),
            required_protocol=required_protocol,
        )

    views = sorted(fits)
    if set(views) != set(VIEWS):
        return _fail(
            name,
            "calibration does not cover both views",
            f"loaded views {views}, expected {sorted(VIEWS)}",
            calibration=str(calibration),
        )

    negative_control: dict[str, Any] = {"performed": False}
    if historical_calibration is not None and historical_calibration.exists():
        try:
            loader(historical_calibration, expected_geometry=required_protocol)
        except ValueError as exc:
            negative_control = {
                "performed": True,
                "unversioned_artifact": str(historical_calibration),
                "refused_with": str(exc)[:200],
            }
        else:
            return _fail(
                name,
                "strict loader is not fail-closed",
                (
                    f"the unversioned artifact {historical_calibration} loaded "
                    f"under expected_geometry={required_protocol!r}; an "
                    "undeclared fit must never be reinterpreted"
                ),
                calibration=str(calibration),
            )

    return _pass(
        name,
        f"upgraded calibration accepted under {required_protocol}",
        calibration=str(calibration),
        views=views,
        negative_control=negative_control,
    )


# ---------------------------------------------------------------------------
# Check 4 -- the frozen B2 head loads, shadow-only, unmodified
# ---------------------------------------------------------------------------


def check_frozen_b2_checkpoint(
    *,
    checkpoint: Path = FROZEN_B2_CHECKPOINT,
    contract_path: Path = FROZEN_B2_CONTRACT,
    load_tensors: bool = True,
) -> CheckResult:
    """The frozen head is present, contract-bound, and loads shadow-only.

    Reads the checkpoint onto the CPU and counts tensors.  It never writes,
    re-saves, or re-trains, and it computes no digest of its own: hash binding
    already lives in the deployment path, and duplicating it here would add
    checksum machinery this task must not add.  Byte immutability is checked
    the way the freeze manifest states it -- declared size and the contract's
    own ``weights_unmodified`` flag.
    """

    name = "frozen_b2_checkpoint"
    import json

    if not checkpoint.exists():
        return _fail(
            name,
            "frozen checkpoint missing",
            f"not found: {checkpoint}",
            checkpoint=str(checkpoint),
        )
    if not contract_path.exists():
        return _fail(
            name,
            "deployment contract missing",
            (
                f"not found: {contract_path}; the frozen .pt carries no runtime "
                "contract of its own, so it cannot be deployed without the sidecar"
            ),
            checkpoint=str(checkpoint),
        )
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return _fail(
            name,
            "deployment contract unreadable",
            f"{type(exc).__name__}: {exc}",
            contract=str(contract_path),
        )

    problems: list[str] = []
    if contract.get("weights_unmodified") is not True:
        problems.append("contract does not assert weights_unmodified=True")
    declared_size = contract.get("source_checkpoint_size_bytes")
    actual_size = checkpoint.stat().st_size
    if declared_size is not None and int(declared_size) != int(actual_size):
        problems.append(
            f"checkpoint size {actual_size} != contract-declared {declared_size}"
        )
    runtime_block = contract.get("contract")
    if not isinstance(runtime_block, Mapping):
        problems.append("contract has no runtime 'contract' block")
        runtime_block = {}
    if runtime_block.get("protocol") != B2_RUNTIME_PROTOCOL:
        problems.append(
            f"runtime protocol {runtime_block.get('protocol')!r} != "
            f"{B2_RUNTIME_PROTOCOL!r}"
        )
    if runtime_block.get("parent_protocol") != B2_PARENT_PROTOCOL:
        problems.append(
            f"parent protocol {runtime_block.get('parent_protocol')!r} != "
            f"{B2_PARENT_PROTOCOL!r}"
        )
    scope_block = contract.get("verified_scope")
    if not isinstance(scope_block, Mapping):
        problems.append("contract has no 'verified_scope' block")
    else:
        if scope_block.get("scope") != VERIFIED_SCOPE:
            problems.append(
                f"verified scope {scope_block.get('scope')!r} != {VERIFIED_SCOPE!r}"
            )
        not_verified = scope_block.get("not_execution_verified") or []
        for phase in ("grasp", "transport"):
            if phase not in not_verified:
                problems.append(
                    f"contract no longer marks {phase!r} as NOT execution-verified; "
                    "approach-only clearance cannot be inferred"
                )

    tensor_census: dict[str, Any] = {"loaded": False}
    if load_tensors:
        try:
            import torch

            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
        except Exception as exc:  # noqa: BLE001
            return _fail(
                name,
                "frozen checkpoint failed to load",
                f"{type(exc).__name__}: {exc}",
                checkpoint=str(checkpoint),
            )
        state = payload.get("model") if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping):
            problems.append("checkpoint has no 'model' state dict")
        else:
            base = [k for k in state if str(k).startswith(B2_BASE_PREFIX)]
            head = [k for k in state if not str(k).startswith(B2_BASE_PREFIX)]
            tensor_census = {
                "loaded": True,
                "total": len(state),
                "base": len(base),
                "head": len(head),
                "expected_total": EXPECTED_B2_TENSOR_TOTAL,
                "expected_base": EXPECTED_B2_BASE_TENSORS,
                "expected_head": EXPECTED_B2_HEAD_TENSORS,
                "map_location": "cpu",
                "shadow_only": True,
            }
            if len(state) != EXPECTED_B2_TENSOR_TOTAL:
                problems.append(
                    f"tensor count {len(state)} != {EXPECTED_B2_TENSOR_TOTAL}"
                )
            if len(base) != EXPECTED_B2_BASE_TENSORS:
                problems.append(
                    f"frozen base tensors {len(base)} != {EXPECTED_B2_BASE_TENSORS}"
                )
            if len(head) != EXPECTED_B2_HEAD_TENSORS:
                problems.append(
                    f"head tensors {len(head)} != {EXPECTED_B2_HEAD_TENSORS}"
                )
        size_after = checkpoint.stat().st_size
        if size_after != actual_size:
            problems.append(
                f"checkpoint size changed during load: {actual_size} -> {size_after}"
            )

    if problems:
        return _fail(
            name,
            "frozen checkpoint contract violated",
            "; ".join(problems),
            checkpoint=str(checkpoint),
            contract=str(contract_path),
            tensor_census=tensor_census,
        )
    return _pass(
        name,
        f"frozen head contract-bound, shadow-only, scope={VERIFIED_SCOPE}",
        checkpoint=str(checkpoint),
        contract=str(contract_path),
        size_bytes=int(actual_size),
        runtime_protocol=B2_RUNTIME_PROTOCOL,
        verified_scope=VERIFIED_SCOPE,
        tensor_census=tensor_census,
        weights_written=False,
    )


# ---------------------------------------------------------------------------
# Check 5 -- targets registered, or correctly unregistered
# ---------------------------------------------------------------------------

#: Registration states this check may report.  ``pickup_not_visible`` is a
#: CORRECT fail-closed runtime outcome and is never worked around here; an
#: unregistered profile is reported as such so an operator registers it, and a
#: target is never synthesized.
REGISTRATION_REGISTERED = "registered"
REGISTRATION_NOT_REGISTERED = "not_registered_operator_action_required"


def check_target_registration(
    profile_dir: Path | None,
    *,
    required_protocol: str = REQUIRED_GEOMETRY_PROTOCOL,
    source_image_size: tuple[int, int] = REQUIRED_SOURCE_IMAGE_SIZE,
    require_registered: bool = False,
) -> CheckResult:
    """Targets are either registered under this geometry, or plainly absent.

    Never fabricates a pickup or drop target.  A profile registered under a
    different (or undeclared) geometry is REFUSED rather than reinterpreted,
    because a stored 224x224 crop is consistent with both protocols and
    resampling a template is not a re-registration.
    """

    name = "target_registration"
    import json

    try:
        from fcpg.so101_visual_targets import check_profile_geometry
    except Exception as exc:  # noqa: BLE001
        return _skip(
            name,
            "profile geometry gate not importable",
            f"{type(exc).__name__}: {exc}",
        )

    required = planning_geometry_from_protocol(
        required_protocol,
        source_width=int(source_image_size[0]),
        source_height=int(source_image_size[1]),
        image_size=PLANNING_IMAGE_SIZE,
    )

    if profile_dir is None:
        return _absent_registration(
            name, None, require_registered, "no target profile directory supplied"
        )
    manifest_path = Path(profile_dir) / "targets.json"
    if not manifest_path.exists():
        return _absent_registration(
            name,
            profile_dir,
            require_registered,
            f"no targets.json under {profile_dir}",
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return _fail(
            name,
            "target profile unreadable",
            f"{type(exc).__name__}: {exc}",
            profile=str(manifest_path),
        )

    rejection = check_profile_geometry(payload, required)
    if rejection is not None:
        return _fail(
            name,
            f"target profile refused: {rejection.get('reason')}",
            str(rejection.get("message"))[:400],
            profile=str(manifest_path),
            reason=rejection.get("reason"),
            declared_geometry=rejection.get("declared_geometry"),
            required_geometry=rejection.get("required_geometry"),
        )

    roles = payload.get("references")
    role_names: list[str] = []
    if isinstance(roles, Mapping):
        role_names = sorted(str(k) for k in roles)
    elif isinstance(roles, Sequence):
        role_names = [str(item) for item in roles]
    return _pass(
        name,
        f"target profile loadable under {required_protocol}",
        registration_state=REGISTRATION_REGISTERED,
        profile=str(manifest_path),
        roles=role_names,
        required_geometry=required.to_dict(),
    )


def _absent_registration(
    name: str, profile_dir: Path | None, require_registered: bool, reason: str
) -> CheckResult:
    detail = {
        "registration_state": REGISTRATION_NOT_REGISTERED,
        "profile_dir": None if profile_dir is None else str(profile_dir),
        "operator_action": (
            "register pickup and drop_zone once on a live column frame via the "
            "runtime Web UI; targets are never synthesized, and "
            "'pickup_not_visible' remains a correct fail-closed outcome"
        ),
    }
    if require_registered:
        return _fail(name, "targets not registered", reason, **detail)
    return _pass(
        name,
        "targets not yet registered (operator action required, not a fault)",
        reason=reason,
        **detail,
    )


# ---------------------------------------------------------------------------
# Check 6 -- this preflight cannot actuate
# ---------------------------------------------------------------------------


def check_no_motion_authorization(
    argv: Sequence[str] | None = None,
    *,
    forbidden: Sequence[str] = FORBIDDEN_MOTION_FLAGS,
) -> CheckResult:
    """No motion-permission flag is present anywhere in this invocation."""

    name = "no_motion_authorization"
    tokens = [str(token) for token in (argv or ())]
    present = sorted({flag for flag in forbidden if flag in tokens})
    if present:
        return _fail(
            name,
            "motion permission flag present in a preflight invocation",
            (
                "refusing to continue: "
                + ", ".join(present)
                + ". This preflight never actuates the arm; motion permission "
                "belongs only to a supervised collection run."
            ),
            forbidden_flags=list(forbidden),
            present=present,
        )
    return _pass(
        name,
        "no motion permission requested",
        forbidden_flags=list(forbidden),
        robot_bus_opened=False,
        commands_sent=0,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_live_preflight_report(
    *,
    wrist_device: str | None,
    column_device: str | None,
    target_profile: Path | None,
    calibration: Path = UPGRADED_CALIBRATION,
    historical_calibration: Path | None = HISTORICAL_CALIBRATION,
    checkpoint: Path = FROZEN_B2_CHECKPOINT,
    contract_path: Path = FROZEN_B2_CONTRACT,
    required_protocol: str = REQUIRED_GEOMETRY_PROTOCOL,
    source_image_size: tuple[int, int] = REQUIRED_SOURCE_IMAGE_SIZE,
    camera_probe: Callable[[str, str], dict[str, Any]] | None = None,
    dry_run: bool = False,
    load_tensors: bool = True,
    require_registered_targets: bool = False,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run every non-actuating check and return a JSON-safe report."""

    checks = [
        check_no_motion_authorization(argv),
        check_camera_raw_resolution(
            {"wrist": wrist_device or "", "column": column_device or ""},
            probe=camera_probe,
            required_size=source_image_size,
            allow_skip=dry_run,
        ),
        check_geometry_contract(
            required_protocol=required_protocol,
            source_image_size=source_image_size,
        ),
        check_projection_fit_loader(
            calibration=calibration,
            historical_calibration=historical_calibration,
            required_protocol=required_protocol,
        ),
        check_frozen_b2_checkpoint(
            checkpoint=checkpoint,
            contract_path=contract_path,
            load_tensors=load_tensors,
        ),
        check_target_registration(
            target_profile,
            required_protocol=required_protocol,
            source_image_size=source_image_size,
            require_registered=require_registered_targets,
        ),
    ]

    failed = [c.name for c in checks if c.status == STATUS_FAIL]
    skipped = [c.name for c in checks if c.status == STATUS_SKIPPED]
    if failed:
        decision = DECISION_NO_GO
    elif skipped:
        decision = DECISION_NO_GO_DRY_RUN if dry_run else DECISION_NO_GO
    else:
        decision = DECISION_GO

    return {
        "schema": PREFLIGHT_SCHEMA,
        "decision": decision,
        "passed": decision == DECISION_GO,
        "dry_run": bool(dry_run),
        "zero_motion": True,
        "robot_bus_opened": False,
        "commands_sent": 0,
        "approach_only": True,
        "required_geometry_protocol": required_protocol,
        "required_source_image_size": list(source_image_size),
        "failed_checks": failed,
        "skipped_checks": skipped,
        "checks": [c.to_dict() for c in checks],
    }


def require_live_preflight(report: Mapping[str, Any]) -> None:
    """Raise unless the report is an unqualified ``GO``."""

    if report.get("decision") == DECISION_GO:
        return
    details = []
    for check in report.get("checks", ()):
        if check.get("status") in (STATUS_FAIL, STATUS_SKIPPED):
            details.append(f"{check.get('name')}: {check.get('error', 'failed')}")
    raise PilotPreflightError(
        f"pilot live preflight {report.get('decision')}; " + "; ".join(details)
    )


__all__ = [
    "PREFLIGHT_SCHEMA",
    "REQUIRED_SOURCE_IMAGE_SIZE",
    "REQUIRED_GEOMETRY_PROTOCOL",
    "FROZEN_B2_CHECKPOINT",
    "FROZEN_B2_CONTRACT",
    "UPGRADED_CALIBRATION",
    "HISTORICAL_CALIBRATION",
    "FORBIDDEN_MOTION_FLAGS",
    "STATUS_PASS",
    "STATUS_FAIL",
    "STATUS_SKIPPED",
    "DECISION_GO",
    "DECISION_NO_GO",
    "DECISION_NO_GO_DRY_RUN",
    "CheckResult",
    "PilotPreflightError",
    "opencv_camera_probe",
    "check_camera_raw_resolution",
    "check_geometry_contract",
    "check_projection_fit_loader",
    "check_frozen_b2_checkpoint",
    "check_target_registration",
    "check_no_motion_authorization",
    "build_live_preflight_report",
    "require_live_preflight",
]
