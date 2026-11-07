"""Production-shadow-only DICG candidate geometry.

Projects a frozen training hypothesis bank through CAD/FK and one explicitly
selected observer-bundle extrinsic.  The pixel math is the same as
``scripts/build_dicg_motion_dataset.py::_project_candidates``.

This module does not resample hypotheses, does not emit teacher targets or
physical tip ground truth, and does not treat a score as a tip position.
It is not live calibration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from fcpg.so101_kinematics import SO101MuJoCoKinematics


GEOMETRY_PROTOCOL = "letterbox_v1"
DATASET_SCHEMA = "fcpg.dicg_motion_dataset.v1"
N_HYPOTHESES = 64
N_TIPS = 2
HYPOTHESIS_XYZ_DIM = 3
JOINT_DIM = 6
ARM_JOINT_COUNT = 5
N_XY = 2
TIP_INDICES = (1, 2)
CANVAS_HW = (224, 224)
DEFAULT_MIN_DEPTH_M = 0.05
DEFAULT_EDGE_MARGIN_PX = 1.0
HYPOTHESES_KEY = "hypotheses_tool_offsets_m"

FORBIDDEN_CONTRACT_KEYS = frozenset(
    {
        "goal",
        "structured_goal",
        "goal_delta",
        "pred_delta",
        "gt_delta",
        "action_delta_causal",
        "dynamics_latent",
        "lewm_target",
        "lewm_future",
        "actor_action",
        "q_true_px",
        "absolute_tip_px",
        "physical_tip_m",
    }
)
_FORBIDDEN_EXTRINSIC_ALIASES = frozenset(
    {"", "latest", "default", "last", "newest", "auto", "none"}
)

__all__ = [
    "ARM_JOINT_COUNT",
    "CANVAS_HW",
    "DATASET_SCHEMA",
    "DEFAULT_EDGE_MARGIN_PX",
    "DEFAULT_MIN_DEPTH_M",
    "DICGShadowGeometry",
    "DICGShadowGeometryError",
    "DICGShadowGeometryResult",
    "FORBIDDEN_CONTRACT_KEYS",
    "GEOMETRY_PROTOCOL",
    "HYPOTHESES_KEY",
    "N_HYPOTHESES",
    "TIP_INDICES",
]


class DICGShadowGeometryError(ValueError):
    """Dataset, bundle, extrinsic, or projection contract failed.  Fail closed."""


@dataclass(frozen=True)
class DICGShadowGeometryResult:
    """One ``(q_t, q_tp1)`` CAD projection.  Not a physical tip label."""

    hypotheses_tool_offsets_m: np.ndarray
    candidate_t_px: np.ndarray
    candidate_tp1_px: np.ndarray
    depth_t_m: np.ndarray
    depth_tp1_m: np.ndarray
    depth_valid_t: np.ndarray
    depth_valid_tp1: np.ndarray
    inside_t: np.ndarray
    inside_tp1: np.ndarray
    depth_valid: np.ndarray
    inside_valid: np.ndarray
    tip_valid: np.ndarray
    hypothesis_valid: np.ndarray
    metadata: dict[str, Any]

    @property
    def hypotheses(self) -> np.ndarray:
        return self.hypotheses_tool_offsets_m

    @property
    def valid(self) -> np.ndarray:
        return self.hypothesis_valid


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_existing_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise DICGShadowGeometryError(f"{label} is not a file: {resolved}")
    return resolved


def _canonical_group_key(value: object) -> str:
    payload: object = value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise DICGShadowGeometryError(
                "extrinsic group_key must be a JSON object or its canonical encoding"
            ) from exc
    if not isinstance(payload, dict):
        raise DICGShadowGeometryError("extrinsic group_key must encode a JSON object")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _require_finite(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise DICGShadowGeometryError(f"{name} must be numeric, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise DICGShadowGeometryError(f"{name} contains non-finite values")
    return array


def _require_shape(name: str, values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape:
        raise DICGShadowGeometryError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _inside(points: np.ndarray, *, margin: float) -> np.ndarray:
    height, width = CANVAS_HW
    x = points[..., 0]
    y = points[..., 1]
    return (
        (x >= float(margin))
        & (x <= float(width - 1) - float(margin))
        & (y >= float(margin))
        & (y <= float(height - 1) - float(margin))
    )


def _load_hypotheses(path: Path) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as handle:
            keys = set(handle.files)
            leaked = FORBIDDEN_CONTRACT_KEYS.intersection(keys)
            if leaked:
                raise DICGShadowGeometryError(
                    "dataset-contract NPZ carries forbidden goal/LeWM/"
                    f"absolute-tip keys: {sorted(leaked)}"
                )
            if HYPOTHESES_KEY not in keys:
                raise DICGShadowGeometryError(
                    f"dataset-contract NPZ missing {HYPOTHESES_KEY!r}"
                )
            if "schema" in keys:
                schema = np.asarray(handle["schema"]).reshape(-1)
                if schema.size < 1 or str(schema[0]) != DATASET_SCHEMA:
                    raise DICGShadowGeometryError(
                        f"dataset-contract schema must be {DATASET_SCHEMA!r}"
                    )
            hypotheses = np.asarray(handle[HYPOTHESES_KEY])
    except DICGShadowGeometryError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise DICGShadowGeometryError(
            f"dataset-contract NPZ is unreadable with allow_pickle=False: {exc}"
        ) from exc
    hypotheses = _require_finite(HYPOTHESES_KEY, hypotheses)
    hypotheses = _require_shape(
        HYPOTHESES_KEY,
        hypotheses,
        (N_HYPOTHESES, N_TIPS, HYPOTHESIS_XYZ_DIM),
    )
    return np.ascontiguousarray(hypotheses, dtype=np.float32)


def _load_observer_bundle(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DICGShadowGeometryError(f"observer bundle JSON is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise DICGShadowGeometryError("observer bundle must be a JSON object")
    declared = payload.get("geometry_protocol")
    if declared is not None and declared != GEOMETRY_PROTOCOL:
        raise DICGShadowGeometryError(
            f"observer bundle geometry_protocol must be {GEOMETRY_PROTOCOL!r}, "
            f"got {declared!r}"
        )
    fit = payload.get("fit")
    if not isinstance(fit, dict):
        raise DICGShadowGeometryError("observer bundle missing fit block")
    intrinsics = fit.get("letterbox_intrinsics")
    extrinsics = fit.get("extrinsics_cam_from_base")
    joint_offsets = fit.get("joint_offsets_normalized")
    if not isinstance(intrinsics, dict):
        raise DICGShadowGeometryError("observer bundle missing letterbox_intrinsics")
    if not isinstance(extrinsics, list) or not extrinsics:
        raise DICGShadowGeometryError(
            "observer bundle missing non-empty extrinsics_cam_from_base"
        )
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DICGShadowGeometryError(
            "letterbox_intrinsics must contain finite fx/fy/cx/cy"
        ) from exc
    if not all(np.isfinite((fx, fy, cx, cy))):
        raise DICGShadowGeometryError("letterbox_intrinsics must be finite")
    if fx <= 0.0 or fy <= 0.0:
        raise DICGShadowGeometryError("letterbox fx/fy must be positive")

    matrices: list[np.ndarray] = []
    group_keys: list[dict[str, Any]] = []
    group_map: dict[str, int] = {}
    for index, item in enumerate(extrinsics):
        if not isinstance(item, dict):
            raise DICGShadowGeometryError(f"bundle extrinsic {index} must be an object")
        if "group_index" in item and int(item["group_index"]) != index:
            raise DICGShadowGeometryError(
                f"bundle extrinsic {index} group_index {item['group_index']!r} "
                "does not match list position"
            )
        if "group_key" not in item:
            raise DICGShadowGeometryError(f"bundle extrinsic {index} missing group_key")
        raw_key = item["group_key"]
        if not isinstance(raw_key, dict):
            raise DICGShadowGeometryError(
                f"bundle extrinsic {index} group_key must be a JSON object"
            )
        canonical = _canonical_group_key(raw_key)
        if canonical in group_map:
            raise DICGShadowGeometryError(
                f"bundle extrinsic group_key is not unique: {canonical}"
            )
        matrix = np.asarray(item.get("matrix"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise DICGShadowGeometryError(f"bundle extrinsic {index} is invalid")
        group_map[canonical] = index
        group_keys.append(dict(raw_key))
        matrices.append(matrix)

    offsets = np.asarray(joint_offsets, dtype=np.float64)
    if offsets.shape != (ARM_JOINT_COUNT,) or not np.all(np.isfinite(offsets)):
        raise DICGShadowGeometryError("joint_offsets_normalized must be finite [5]")
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "extrinsics": np.stack(matrices),
        "group_keys": tuple(group_keys),
        "group_map": group_map,
        "joint_offsets": offsets,
    }


def _reject_implicit_extrinsic_alias(value: object) -> None:
    if isinstance(value, str) and value.strip().lower() in _FORBIDDEN_EXTRINSIC_ALIASES:
        raise DICGShadowGeometryError(
            "extrinsic identity cannot be latest/default/auto; "
            "pass an explicit index or unique group key"
        )


def _resolve_extrinsic(
    bundle: Mapping[str, Any],
    *,
    extrinsic_index: object,
    extrinsic_group_key: object,
) -> tuple[int, dict[str, Any], str, np.ndarray]:
    if extrinsic_index is None and extrinsic_group_key is None:
        raise DICGShadowGeometryError(
            "caller must explicitly select a bundle extrinsic index "
            "or a unique auditable group key; silent latest/default is forbidden"
        )
    _reject_implicit_extrinsic_alias(extrinsic_index)
    _reject_implicit_extrinsic_alias(extrinsic_group_key)

    n_extrinsics = int(np.asarray(bundle["extrinsics"]).shape[0])
    resolved_from_index: int | None = None
    if extrinsic_index is not None:
        if isinstance(extrinsic_index, bool) or not isinstance(
            extrinsic_index, (int, np.integer)
        ):
            raise DICGShadowGeometryError(
                "extrinsic_index must be an explicit non-negative integer"
            )
        index = int(extrinsic_index)
        if index < 0 or index >= n_extrinsics:
            raise DICGShadowGeometryError(
                f"extrinsic_index {index} is outside [0, {n_extrinsics})"
            )
        resolved_from_index = index

    resolved_from_key: int | None = None
    if extrinsic_group_key is not None:
        canonical = _canonical_group_key(extrinsic_group_key)
        matches = [
            index
            for index, key in enumerate(bundle["group_keys"])
            if _canonical_group_key(key) == canonical
        ]
        if len(matches) != 1:
            raise DICGShadowGeometryError(
                "extrinsic_group_key must uniquely identify one bundle extrinsic"
            )
        resolved_from_key = int(matches[0])

    if (
        resolved_from_index is not None
        and resolved_from_key is not None
        and resolved_from_index != resolved_from_key
    ):
        raise DICGShadowGeometryError(
            "extrinsic_index and extrinsic_group_key select different extrinsics"
        )
    index = int(
        resolved_from_index if resolved_from_index is not None else resolved_from_key
    )
    group_key = dict(bundle["group_keys"][index])
    canonical = _canonical_group_key(group_key)
    matrix = np.asarray(bundle["extrinsics"][index], dtype=np.float64)
    return index, group_key, canonical, matrix


def _require_joint_vector(name: str, values: object) -> np.ndarray:
    array = _require_finite(name, np.asarray(values, dtype=np.float64))
    if array.shape != (JOINT_DIM,):
        raise DICGShadowGeometryError(f"{name} must have shape {(JOINT_DIM,)}, got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_kinematics_outputs(
    poses: np.ndarray,
    local: np.ndarray,
    *,
    n_states: int,
) -> tuple[np.ndarray, np.ndarray]:
    poses = _require_finite("forward_gripper_poses", np.asarray(poses, dtype=np.float64))
    poses = _require_shape("forward_gripper_poses", poses, (n_states, 4, 4))
    local = _require_finite("forward_anchor_frames local", np.asarray(local, dtype=np.float64))
    if local.ndim != 3 or local.shape[0] != n_states or local.shape[-1] != 3:
        raise DICGShadowGeometryError(
            "forward_anchor_frames local must be [T, n_anchors, 3], "
            f"got {local.shape}"
        )
    if local.shape[1] <= max(TIP_INDICES):
        raise DICGShadowGeometryError(
            f"forward_anchor_frames must include tip indices {TIP_INDICES}, "
            f"got n_anchors={local.shape[1]}"
        )
    return poses, local


class DICGShadowGeometry:
    """Frozen hypothesis bank + explicit letterbox bundle projection."""

    def __init__(
        self,
        *,
        hypotheses_tool_offsets_m: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        joint_offsets: np.ndarray,
        extrinsic: np.ndarray,
        kinematics: Any,
        metadata: Mapping[str, Any],
        min_depth_m: float = DEFAULT_MIN_DEPTH_M,
        edge_margin_px: float = DEFAULT_EDGE_MARGIN_PX,
    ) -> None:
        hypotheses = _require_finite(
            HYPOTHESES_KEY, np.asarray(hypotheses_tool_offsets_m)
        )
        hypotheses = _require_shape(
            HYPOTHESES_KEY,
            hypotheses,
            (N_HYPOTHESES, N_TIPS, HYPOTHESIS_XYZ_DIM),
        )
        offsets = _require_finite(
            "joint_offsets_normalized", np.asarray(joint_offsets, dtype=np.float64)
        )
        offsets = _require_shape(
            "joint_offsets_normalized", offsets, (ARM_JOINT_COUNT,)
        )
        matrix = _require_finite("extrinsic", np.asarray(extrinsic, dtype=np.float64))
        matrix = _require_shape("extrinsic", matrix, (4, 4))
        if not np.isfinite(float(min_depth_m)) or float(min_depth_m) <= 0.0:
            raise DICGShadowGeometryError("min_depth_m must be a positive finite value")
        if not np.isfinite(float(edge_margin_px)) or float(edge_margin_px) < 0.0:
            raise DICGShadowGeometryError("edge_margin_px must be a non-negative finite value")
        for name, value in (("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy)):
            if not np.isfinite(float(value)):
                raise DICGShadowGeometryError(f"{name} must be finite")
        if float(fx) <= 0.0 or float(fy) <= 0.0:
            raise DICGShadowGeometryError("fx/fy must be positive")
        if kinematics is None or not callable(
            getattr(kinematics, "forward_gripper_poses", None)
        ):
            raise DICGShadowGeometryError(
                "kinematics must provide forward_gripper_poses and forward_anchor_frames"
            )
        if not callable(getattr(kinematics, "forward_anchor_frames", None)):
            raise DICGShadowGeometryError(
                "kinematics must provide forward_gripper_poses and forward_anchor_frames"
            )
        self.hypotheses_tool_offsets_m = np.ascontiguousarray(hypotheses, dtype=np.float32)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.joint_offsets = np.ascontiguousarray(offsets, dtype=np.float64)
        self.extrinsic = np.ascontiguousarray(matrix, dtype=np.float64)
        self.kinematics = kinematics
        self.min_depth_m = float(min_depth_m)
        self.edge_margin_px = float(edge_margin_px)
        self.metadata = dict(metadata)

    @classmethod
    def from_files(
        cls,
        dataset_contract: str | Path,
        observer_bundle: str | Path,
        mujoco_model: str | Path,
        *,
        extrinsic_index: int | None = None,
        extrinsic_group_key: Mapping[str, Any] | str | None = None,
        kinematics: Any | None = None,
        min_depth_m: float = DEFAULT_MIN_DEPTH_M,
        edge_margin_px: float = DEFAULT_EDGE_MARGIN_PX,
    ) -> DICGShadowGeometry:
        """Load the frozen bank, bundle, and explicit MuJoCo model.

        ``kinematics`` is an injection point for tests.  Production callers omit
        it so ``SO101MuJoCoKinematics`` is constructed from ``mujoco_model``.
        """

        contract_path = _require_existing_file(dataset_contract, label="dataset-contract")
        bundle_path = _require_existing_file(observer_bundle, label="observer bundle")
        model_path = _require_existing_file(mujoco_model, label="mujoco model")
        hypotheses = _load_hypotheses(contract_path)
        bundle = _load_observer_bundle(bundle_path)
        index, group_key, canonical, extrinsic = _resolve_extrinsic(
            bundle,
            extrinsic_index=extrinsic_index,
            extrinsic_group_key=extrinsic_group_key,
        )
        if kinematics is None:
            kinematics = SO101MuJoCoKinematics(model_path)
        metadata = {
            "dataset_contract_path": str(contract_path),
            "dataset_contract_sha256": sha256_file(contract_path),
            "observer_bundle_path": str(bundle_path),
            "observer_bundle_sha256": sha256_file(bundle_path),
            "mujoco_model_path": str(model_path),
            "mujoco_model_sha256": sha256_file(model_path),
            "extrinsic_index": index,
            "extrinsic_group_key": group_key,
            "extrinsic_group_key_canonical": canonical,
            "extrinsic_identity": {
                "index": index,
                "group_key": group_key,
                "group_key_canonical": canonical,
            },
            "geometry_protocol": GEOMETRY_PROTOCOL,
            "not_live_calibrated": True,
            "shadow_only": True,
            "live_ready": False,
            "has_physical_labels": False,
            "absolute_tip_accuracy_claim": False,
            "uncertainty_calibrated": False,
            "not_physical_tip_ground_truth": True,
            "bundle_is_detector_relative": True,
            "hypothesis_bank_resampled": False,
            "n_hypotheses": N_HYPOTHESES,
            "tip_indices": list(TIP_INDICES),
            "canvas_hw": list(CANVAS_HW),
            "min_depth_m": float(min_depth_m),
            "edge_margin_px": float(edge_margin_px),
            "fx": float(bundle["fx"]),
            "fy": float(bundle["fy"]),
            "cx": float(bundle["cx"]),
            "cy": float(bundle["cy"]),
        }
        return cls(
            hypotheses_tool_offsets_m=hypotheses,
            fx=float(bundle["fx"]),
            fy=float(bundle["fy"]),
            cx=float(bundle["cx"]),
            cy=float(bundle["cy"]),
            joint_offsets=bundle["joint_offsets"],
            extrinsic=extrinsic,
            kinematics=kinematics,
            metadata=metadata,
            min_depth_m=min_depth_m,
            edge_margin_px=edge_margin_px,
        )

    def project_pair(
        self, q_t: object, q_tp1: object
    ) -> DICGShadowGeometryResult:
        """Project ``q_t, q_tp1`` to candidate pixels ``[64, 2, 2]``."""

        q_t_arr = _require_joint_vector("q_t", q_t)
        q_tp1_arr = _require_joint_vector("q_tp1", q_tp1)
        q = np.stack((q_t_arr, q_tp1_arr), axis=0)
        q_corrected = q.copy()
        q_corrected[:, :ARM_JOINT_COUNT] += self.joint_offsets[None, :]
        poses = np.asarray(self.kinematics.forward_gripper_poses(q_corrected), dtype=np.float64)
        _base, local = self.kinematics.forward_anchor_frames(q_corrected)
        poses, local = _validate_kinematics_outputs(poses, local, n_states=2)
        local_tips = np.asarray(local, dtype=np.float64)[:, TIP_INDICES, :]
        hypotheses = self.hypotheses_tool_offsets_m.astype(np.float64, copy=False)
        tool_points = local_tips[:, None, :, :] + hypotheses[None, :, :, :]
        points_base = (
            np.einsum("fij,fnkj->fnki", poses[:, :3, :3], tool_points)
            + poses[:, None, None, :3, 3]
        )
        cam_from_base = np.stack((self.extrinsic, self.extrinsic), axis=0)
        camera = (
            np.einsum("fij,fnkj->fnki", cam_from_base[:, :3, :3], points_base)
            + cam_from_base[:, None, None, :3, 3]
        )
        depth = camera[..., 2]
        _require_finite("projected_depth", depth)
        safe_depth = np.maximum(depth, self.min_depth_m)
        pixels = np.stack(
            (
                self.fx * camera[..., 0] / safe_depth + self.cx,
                self.fy * camera[..., 1] / safe_depth + self.cy,
            ),
            axis=-1,
        )
        _require_finite("projected_pixels", pixels)
        if pixels.shape != (2, N_HYPOTHESES, N_TIPS, N_XY):
            raise DICGShadowGeometryError(
                f"projected pixels must have shape {(2, N_HYPOTHESES, N_TIPS, N_XY)}, "
                f"got {pixels.shape}"
            )
        candidate_t = np.ascontiguousarray(pixels[0], dtype=np.float32)
        candidate_tp1 = np.ascontiguousarray(pixels[1], dtype=np.float32)
        depth_t = np.ascontiguousarray(depth[0], dtype=np.float32)
        depth_tp1 = np.ascontiguousarray(depth[1], dtype=np.float32)
        depth_valid_t = depth[0] > self.min_depth_m
        depth_valid_tp1 = depth[1] > self.min_depth_m
        inside_t = _inside(pixels[0], margin=self.edge_margin_px)
        inside_tp1 = _inside(pixels[1], margin=self.edge_margin_px)
        depth_valid = depth_valid_t & depth_valid_tp1
        inside_valid = inside_t & inside_tp1
        tip_valid = depth_valid & inside_valid
        hypothesis_valid = tip_valid.all(axis=-1)
        return DICGShadowGeometryResult(
            hypotheses_tool_offsets_m=self.hypotheses_tool_offsets_m.copy(),
            candidate_t_px=candidate_t,
            candidate_tp1_px=candidate_tp1,
            depth_t_m=depth_t,
            depth_tp1_m=depth_tp1,
            depth_valid_t=np.ascontiguousarray(depth_valid_t),
            depth_valid_tp1=np.ascontiguousarray(depth_valid_tp1),
            inside_t=np.ascontiguousarray(inside_t),
            inside_tp1=np.ascontiguousarray(inside_tp1),
            depth_valid=np.ascontiguousarray(depth_valid),
            inside_valid=np.ascontiguousarray(inside_valid),
            tip_valid=np.ascontiguousarray(tip_valid),
            hypothesis_valid=np.ascontiguousarray(hypothesis_valid),
            metadata=dict(self.metadata),
        )
