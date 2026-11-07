"""Pure schema/contract for the formal ``self_contact_v1`` screen protocol.

Offline data only.  Does not open a serial port, camera, or live service.
Does not import the planner, live serve, LeWM, or printed-datum observer.

The retired ``printed_datum`` / world-witness / ``center_h32mm`` seat route
stays in the collector and analyzer for audit compatibility.  Formal
self-contact records must not require or forge those fields.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

PROTOCOL_PRINTED_DATUM = "printed_datum"
PROTOCOL_SELF_CONTACT_V1 = "self_contact_v1"
PRINTED_DATUM_SCHEMA_ID = "minimal_physical_grounding_screen.collection.v1"
SELF_CONTACT_SCHEMA_ID = "minimal_physical_grounding_screen.self_contact.v1"
SELF_CONTACT_ANCHOR_ID = "wrist_triangular_jaws"
RETIRED_PROTOCOL_STATUS = "retired_audit_only"
FORMAL_PROTOCOL_STATUS = "formal"

N_CONFIGS = 8
N_NEAR_BLOCKS = 8
N_DELAYED_BLOCKS = 1
N_REVISITS = N_NEAR_BLOCKS + N_DELAYED_BLOCKS
N_PULSES = N_CONFIGS * N_REVISITS
N_ADJACENT_SAME_CONFIG_PAIRS = N_CONFIGS * (N_NEAR_BLOCKS - 1)
T_CLOSE_S = 300.0
T_DISTANT_S = 3600.0
MIN_N_SAME_CONFIG_T_CLOSE = 48
MIN_N_CROSS_CONFIG_T_CLOSE = 30
ARM_DQ_JOINT_COUNT = 5
CROSS_CONFIG_DQ_INF_LO_DEG = 10.0
CROSS_CONFIG_DQ_INF_HI_DEG = 30.0
PLANNING_CANVAS_HW = (224, 224)
PLANNING_CANVAS_NAME = "planning_224"
SESSION_KIND_DRY_RUN = "dry_run_no_contact"

# Primary estimand for formal self_contact_v1.  Knife-edge image is GT;
# nominal FK is a predictor only.  Camera/cable/config coupling stays in r_raw.
ESTIMAND_B_RAW = "deployment_chain_B_raw_pixel_grounding"
ESTIMAND_A_REGISTERED = "intrinsic_fk_A_registered_sensitivity"
CLAIM_C_WRIST_OFF = "wrist_off_policy_improvement"
SELF_CONTACT_CONDITION_7_KEY = "7_true_contact_prediction_and_camera_identity"

FORBIDDEN_A_CLAIM_PHRASES = (
    "纯机械臂内禀",
    "内禀 FK",
    "内禀 sag",
    "纯内禀 FK/sag",
    "证实 11.14",
    "证实了 11.14",
    "11.14 px 已被本筛查独立测出",
    "本筛查证实了 11.14 px 场",
    "CLAIM 2 已证明",
    "wrist-off 已证明",
    "授权声明 A",
    "授权声明 C",
)

# Copied from the collector interleaved-block table.  Tests assert equality.
INTERLEAVED_BLOCK_PERMUTATIONS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (2, 0, 7, 4, 1, 6, 3, 5),
    (7, 3, 1, 5, 0, 4, 2, 6),
    (4, 6, 2, 5, 1, 7, 0, 3),
    (6, 7, 5, 1, 3, 2, 4, 0),
    (3, 4, 6, 0, 7, 1, 5, 2),
    (5, 2, 3, 7, 6, 0, 1, 4),
    (1, 5, 4, 6, 2, 3, 7, 0),
    (7, 6, 0, 2, 5, 4, 3, 1),
)

CONFIG_LABELS: tuple[str, ...] = tuple(f"C{index + 1}" for index in range(N_CONFIGS))

IMAGE_ONLY_KNIFE_EDGE_SOURCES = frozenset(
    {
        "image_only_knife_edge",
        "image_knife_edge_intersection",
        "image_knife_edge_midpoint",
    }
)

FORBIDDEN_PIXEL_TRUTH_KEYS = frozenset(
    {
        "q_true",
        "q_true_px",
        "valid_true",
        "u_from_fk",
        "u_hat_from_fk",
        "u_from_m0",
        "u_hat_from_m0",
        "u_from_m1",
        "u_hat_from_m1",
        "p_m1_as_u_hat",
        "u_from_fcpg",
        "u_hat_from_fcpg",
        "fcpg_residual_as_u_hat",
        "real_warp",
        "affine_label",
        "synthetic_q_true",
        "snapshot_metal",
        "metal_as_u_hat",
        "u_from_metal",
        "u_hat_from_metal",
    }
)

# Printed-datum / seat / witness fields must not appear on formal pulses.
FORBIDDEN_PRINTED_FIELDS = frozenset(
    {
        "D",
        "datum_corner_d",
        "feature_layout",
        "world_witness_px",
        "world_witness_features_px",
        "world_witness_corners_px",
        "world_witness_feature_ids",
        "seat",
        "seat_name",
        "seat_xyz_m",
        "seat_quality",
        "seat_occluded",
        "printed_datum_id",
        "center_h32mm",
        "near_h32mm",
    }
)

RETIRED_SEAT_IDS = frozenset({"center_h32mm", "near_h32mm"})

REQUIRED_PULSE_FIELDS = (
    "config_label",
    "revisit_index",
    "block_index",
    "block_kind",
    "timestamp",
    "q_arm_deg",
    "goal_position",
    "present_position",
    "velocity_available",
    "moving_available",
    "load_current",
    "gripper_command_readback",
    "visual_contact",
    "position_plateau",
    "true_contact_valid",
    "u_hat_px224",
    "u_hat_source",
    "background_registration",
    "column_frame",
    "wrist_frame",
)

LOAD_READ_STATUSES = frozenset(
    {"not_opened", "ok", "error", "all_zero", "constant", "unavailable"}
)


class SelfContactSchemaError(ValueError):
    """Fail-closed schema refusal.  Not a screen PASS/FAIL."""


def is_planning_224_hw(shape: Sequence[int] | None) -> bool:
    """True only for the locked planning canvas ``(224, 224)``."""

    if shape is None:
        return False
    if len(shape) < 2:
        return False
    return int(shape[0]) == PLANNING_CANVAS_HW[0] and int(shape[1]) == PLANNING_CANVAS_HW[1]


def _as_mapping(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SelfContactSchemaError(f"{where} must be an object")
    return dict(value)


def refuse_fabricated_pixel_truth(payload: Mapping[str, Any], *, where: str) -> None:
    hit = sorted(set(payload) & FORBIDDEN_PIXEL_TRUTH_KEYS)
    if hit:
        raise SelfContactSchemaError(
            f"{where} contains forbidden pixel-truth key(s) {hit}; "
            "synthetic q_true / FK / M0 / M1 / FCPG / metal cannot be pixel GT"
        )


def refuse_printed_fields(payload: Mapping[str, Any], *, where: str) -> None:
    hit = sorted(set(payload) & FORBIDDEN_PRINTED_FIELDS)
    if hit:
        raise SelfContactSchemaError(
            f"{where} contains retired printed-datum / seat / witness "
            f"field(s) {hit}; formal self_contact_v1 must not require or forge them"
        )


def resolve_protocol(payload: Mapping[str, Any]) -> str:
    protocol = str(payload.get("protocol") or "").strip()
    schema = str(payload.get("schema") or "").strip()
    if protocol == PROTOCOL_SELF_CONTACT_V1 or schema == SELF_CONTACT_SCHEMA_ID:
        return PROTOCOL_SELF_CONTACT_V1
    if protocol == PROTOCOL_PRINTED_DATUM or schema == PRINTED_DATUM_SCHEMA_ID:
        return PROTOCOL_PRINTED_DATUM
    if protocol:
        raise SelfContactSchemaError(f"unsupported protocol {protocol!r}")
    return PROTOCOL_PRINTED_DATUM


def empty_frame_provenance(source: str) -> dict[str, Any]:
    if source not in {"column", "wrist"}:
        raise SelfContactSchemaError("frame source must be column or wrist")
    return {
        "source": source,
        "path": None,
        "timestamp": None,
        "hardware_opened": False,
    }


def empty_load_current() -> dict[str, Any]:
    return {
        "load": None,
        "current": None,
        "read_status": "not_opened",
        "physical_witness_valid": False,
        "role": "corroborative_only",
    }


def empty_gripper_sequence() -> dict[str, Any]:
    return {
        "command_sequence_summary": {
            "n_steps": 0,
            "goal_start": None,
            "goal_end": None,
        },
        "readback_sequence_summary": {
            "n_samples": 0,
            "present_min": None,
            "present_max": None,
            "closed_state_is_not_physical_contact": True,
        },
    }


def empty_visual_contact() -> dict[str, Any]:
    return {
        "valid": False,
        "metrics": {
            "gap_px224": None,
            "edge_fit_ok": False,
            "reason": "dry_run_no_contact",
        },
    }


def empty_position_plateau() -> dict[str, Any]:
    return {
        "valid": False,
        "metrics": {
            "present_velocity": None,
            "plateau_samples": 0,
            "reason": "dry_run_no_contact",
        },
    }


def empty_background_registration() -> dict[str, Any]:
    return {
        "valid": False,
        "n_inliers": 0,
        "n_inliers_sufficient": False,
        "spatial_coverage": 0.0,
        "spatial_coverage_sufficient": False,
        "residual_px224": None,
        "transform_2x3": None,
        "camera_break_frame": False,
        "camera_identity_break": False,
        "movable_objects_dominated": False,
        "inliers_in_exclude_rois": 0,
        "reason": "dry_run_no_hardware",
        "diagnostics": {"reason": "dry_run_no_hardware"},
    }


def _finite_len(value: Any, n: int, *, where: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise SelfContactSchemaError(f"{where} must be a length-{n} vector")
    out = tuple(float(item) for item in value)
    if not all(item == item and item != float("inf") and item != float("-inf") for item in out):
        raise SelfContactSchemaError(f"{where} must be finite")
    return out


def planned_q_arm_deg(config_index: int) -> list[float]:
    """Abstract 8-config arm descriptors.  Gripper is excluded from ``dq``.

    Pairwise ``|dq|_inf`` on the first five joints is ``12° × |i-j|``, so
    same-block pairs with ``|i-j| ∈ {1, 2}`` sit in the 10–30° cell.
    """

    index = int(config_index)
    if index < 0 or index >= N_CONFIGS:
        raise SelfContactSchemaError("config_index must be in 0..7")
    return [0.0, 12.0 * index, 0.0, 0.0, 0.0]


def dq_inf_arm_deg(q_a: Sequence[float], q_b: Sequence[float]) -> float:
    left = [float(v) for v in q_a][:ARM_DQ_JOINT_COUNT]
    right = [float(v) for v in q_b][:ARM_DQ_JOINT_COUNT]
    n = min(len(left), len(right), ARM_DQ_JOINT_COUNT)
    if n == 0:
        raise SelfContactSchemaError("q_arm_deg must include the first five joints")
    return max(abs(left[i] - right[i]) for i in range(n))


def planned_samples() -> list[dict[str, Any]]:
    if len(INTERLEAVED_BLOCK_PERMUTATIONS) != N_REVISITS:
        raise SelfContactSchemaError("permutation table must have one order per block")
    if len(set(INTERLEAVED_BLOCK_PERMUTATIONS)) != N_REVISITS:
        raise SelfContactSchemaError("block permutations must be pairwise distinct")
    samples: list[dict[str, Any]] = []
    for block, perm in enumerate(INTERLEAVED_BLOCK_PERMUTATIONS):
        if tuple(sorted(perm)) != tuple(range(N_CONFIGS)):
            raise SelfContactSchemaError(f"block {block} permutation is not 0..7")
        kind = "delayed" if block >= N_NEAR_BLOCKS else "near"
        revisit = block + 1
        for slot in perm:
            label = CONFIG_LABELS[int(slot)]
            samples.append(
                {
                    "sample_id": f"{label}:r{revisit}",
                    "config_label": label,
                    "config_index": int(slot),
                    "revisit_index": int(revisit),
                    "block_index": int(block),
                    "block_kind": kind,
                    "q_arm_deg": planned_q_arm_deg(int(slot)),
                }
            )
    return samples


def count_adjacent_same_config_pairs(samples: Sequence[Mapping[str, Any]]) -> int:
    blocks_by_label: dict[str, list[int]] = {}
    for row in samples:
        if str(row.get("block_kind") or "near") != "near":
            continue
        label = str(row["config_label"])
        blocks_by_label.setdefault(label, []).append(int(row["block_index"]))
    n_pairs = 0
    for blocks in blocks_by_label.values():
        ordered = sorted(blocks)
        n_pairs += sum(
            1 for left, right in zip(ordered, ordered[1:]) if int(right) - int(left) == 1
        )
    return n_pairs


def _require_flag(payload: Mapping[str, Any], key: str, *, where: str) -> bool:
    if key not in payload:
        raise SelfContactSchemaError(f"{where} missing {key}")
    value = payload[key]
    if not isinstance(value, bool):
        raise SelfContactSchemaError(f"{where}.{key} must be a bool")
    return value


def _validate_frame(
    raw: Any,
    *,
    source: str,
    where: str,
    dry_run: bool,
) -> dict[str, Any]:
    frame = _as_mapping(raw, where=where)
    if frame.get("source") not in {None, source} and frame.get("source") != source:
        raise SelfContactSchemaError(f"{where}.source must be {source!r}")
    opened = bool(frame.get("hardware_opened", False))
    path = frame.get("path")
    stamp = frame.get("timestamp")
    if dry_run:
        if opened:
            raise SelfContactSchemaError(
                f"{where} dry-run must set hardware_opened=false"
            )
        if path not in {None, ""}:
            raise SelfContactSchemaError(f"{where} dry-run path must be empty")
        if stamp not in {None, ""}:
            raise SelfContactSchemaError(f"{where} dry-run timestamp must be empty")
    return {
        "source": source,
        "path": None if path in {None, ""} else str(path),
        "timestamp": None if stamp in {None, ""} else str(stamp),
        "hardware_opened": opened,
    }


def _validate_metric_block(raw: Any, *, where: str) -> dict[str, Any]:
    block = _as_mapping(raw, where=where)
    valid = _require_flag(block, "valid", where=where)
    metrics = block.get("metrics", {})
    if metrics is None:
        metrics = {}
    if not isinstance(metrics, Mapping):
        raise SelfContactSchemaError(f"{where}.metrics must be an object")
    return {"valid": valid, "metrics": dict(metrics)}


def _validate_load_current(raw: Any, *, where: str, dry_run: bool) -> dict[str, Any]:
    block = _as_mapping(raw, where=where)
    status = str(block.get("read_status") or "unavailable")
    if status not in LOAD_READ_STATUSES:
        raise SelfContactSchemaError(f"{where}.read_status={status!r} is not licensed")
    if dry_run and status != "not_opened":
        raise SelfContactSchemaError(f"{where} dry-run read_status must be not_opened")
    role = str(block.get("role") or "corroborative_only")
    if role != "corroborative_only":
        raise SelfContactSchemaError(
            f"{where}.role must be corroborative_only; Load/Current never define contact"
        )
    witness = bool(block.get("physical_witness_valid", False))
    if dry_run and witness:
        raise SelfContactSchemaError(f"{where} dry-run cannot claim a physical witness")
    payload = {
        "load": None if block.get("load") is None else list(block["load"]),
        "current": None if block.get("current") is None else list(block["current"]),
        "read_status": status,
        "physical_witness_valid": witness,
        "role": role,
    }
    if block.get("load_decoded") is not None:
        payload["load_decoded"] = list(block["load_decoded"])
    if block.get("current_decoded") is not None:
        payload["current_decoded"] = list(block["current_decoded"])
    if isinstance(block.get("encoding"), Mapping):
        payload["encoding"] = dict(block["encoding"])
    if block.get("disk_values") not in {None, ""}:
        payload["disk_values"] = str(block["disk_values"])
    return payload


def _optional_u_hat_pair(raw: Any, *, where: str) -> tuple[float, float] | None:
    if raw is None or raw == "":
        return None
    parsed = _finite_len(raw, 2, where=where)
    return (parsed[0], parsed[1])


def _validate_u_hat(
    raw: Mapping[str, Any],
    *,
    where: str,
    dry_run: bool,
    true_contact_valid: bool,
) -> tuple[
    tuple[float, float] | None,
    str | None,
    tuple[float, float] | None,
    bool,
    tuple[float, float] | None,
]:
    """B primary is always the image-only raw knife-edge.

    ``u_hat_px224`` is the B analysis alias and equals raw when raw is
    present.  Canonical / background transform is optional secondary and
    must not blank or replace raw.
    """

    source = raw.get("u_hat_source")
    legacy_pixel = raw.get("u_hat_px224", raw.get("u_hat_px"))
    raw_pixel = _optional_u_hat_pair(
        raw.get("u_hat_px224_raw"), where=f"{where}.u_hat_px224_raw"
    )
    explicit_canonical = _optional_u_hat_pair(
        raw.get("u_hat_px224_canonical"), where=f"{where}.u_hat_px224_canonical"
    )
    canonicalized = bool(raw.get("u_hat_canonicalized", False))
    legacy_pair = _optional_u_hat_pair(legacy_pixel, where=f"{where}.u_hat_px224")
    if raw_pixel is None and legacy_pair is not None and not canonicalized:
        raw_pixel = legacy_pair
    if explicit_canonical is not None:
        canonical = explicit_canonical
    elif (
        canonicalized
        and legacy_pair is not None
        and raw_pixel is not None
        and legacy_pair != raw_pixel
    ):
        canonical = legacy_pair
    else:
        canonical = None
    if dry_run or not true_contact_valid:
        if raw_pixel is None and (legacy_pixel is None or legacy_pixel == ""):
            if source not in {None, ""} and str(source) not in IMAGE_ONLY_KNIFE_EDGE_SOURCES:
                raise SelfContactSchemaError(
                    f"{where}.u_hat_source must be empty on a non-contact pulse "
                    f"or one of {sorted(IMAGE_ONLY_KNIFE_EDGE_SOURCES)}"
                )
            if canonicalized and canonical is not None:
                raise SelfContactSchemaError(
                    f"{where} cannot claim u_hat_canonicalized on a non-contact pulse"
                )
            return (
                None,
                None if source in {None, ""} else str(source),
                raw_pixel,
                False,
                None,
            )
        if source not in {None, ""} and str(source) not in IMAGE_ONLY_KNIFE_EDGE_SOURCES:
            raise SelfContactSchemaError(
                f"{where}.u_hat_source must be empty on a non-contact pulse "
                f"or one of {sorted(IMAGE_ONLY_KNIFE_EDGE_SOURCES)}"
            )
        primary = raw_pixel if raw_pixel is not None else legacy_pair
        return (
            primary,
            None if source in {None, ""} else str(source),
            raw_pixel if raw_pixel is not None else primary,
            False,
            None,
        )
    if str(source) not in IMAGE_ONLY_KNIFE_EDGE_SOURCES:
        raise SelfContactSchemaError(
            f"{where}.u_hat_source must be image-only knife-edge; "
            f"got {source!r}"
        )
    if raw_pixel is None:
        raise SelfContactSchemaError(
            f"{where}.u_hat_px224_raw is required on a true-contact pulse; "
            "background invalid must not blank the raw knife-edge"
        )
    return raw_pixel, str(source), raw_pixel, bool(canonical is not None), canonical


def _validate_background(
    raw: Any,
    *,
    where: str,
    dry_run: bool,
) -> dict[str, Any]:
    block = _as_mapping(raw, where=where)
    valid = bool(block.get("valid", False))
    n_inliers = int(block.get("n_inliers", 0))
    sufficient = bool(block.get("n_inliers_sufficient", False))
    camera_break = bool(
        block.get("camera_identity_break", block.get("camera_break_frame", False))
    )
    movable = bool(block.get("movable_objects_dominated", False))
    if dry_run:
        if valid or sufficient or camera_break:
            raise SelfContactSchemaError(
                f"{where} dry-run cannot claim a valid background registration"
            )
        if n_inliers != 0:
            raise SelfContactSchemaError(f"{where} dry-run n_inliers must be 0")
    if valid and (not sufficient or camera_break or movable):
        raise SelfContactSchemaError(
            f"{where}.valid cannot be true when inliers are insufficient, "
            "the camera broke frame, or movable objects dominated"
        )
    residual = block.get("residual_px224")
    coverage = block.get("spatial_coverage")
    coverage_ok = bool(block.get("spatial_coverage_sufficient", False))
    exclude_hits = int(block.get("inliers_in_exclude_rois", 0) or 0)
    transform = block.get("transform_2x3")
    if valid and (not coverage_ok or exclude_hits > 0):
        raise SelfContactSchemaError(
            f"{where}.valid cannot be true when spatial coverage is "
            "insufficient or exclude-ROI features became inliers"
        )
    if transform is not None and transform != "":
        if not isinstance(transform, (list, tuple)) or len(transform) != 2:
            raise SelfContactSchemaError(f"{where}.transform_2x3 must be a 2x3 matrix")
        transform = [
            list(_finite_len(row, 3, where=f"{where}.transform_2x3"))
            for row in transform
        ]
    else:
        transform = None
    if valid and transform is None:
        raise SelfContactSchemaError(f"{where}.valid requires transform_2x3")
    reason = block.get("reason")
    diagnostics = dict(block.get("diagnostics") or {})
    if reason is None:
        reason = diagnostics.get("reason")
    return {
        "valid": valid,
        "n_inliers": n_inliers,
        "n_inliers_sufficient": sufficient,
        "spatial_coverage": None if coverage is None else float(coverage),
        "spatial_coverage_sufficient": coverage_ok,
        "residual_px224": None if residual is None else float(residual),
        "transform_2x3": transform,
        "camera_break_frame": camera_break,
        "camera_identity_break": camera_break,
        "movable_objects_dominated": movable,
        "inliers_in_exclude_rois": exclude_hits,
        "reason": None if reason in {None, ""} else str(reason),
        "diagnostics": diagnostics,
    }


def validate_self_contact_record(
    raw: Mapping[str, Any],
    *,
    where: str = "self_contact_record",
    dry_run: bool = False,
    hardware_opened: bool = False,
) -> dict[str, Any]:
    """Validate one pulse.  Dry-run rows may be empty and must not claim contact."""

    record = _as_mapping(raw, where=where)
    refuse_fabricated_pixel_truth(record, where=where)
    refuse_printed_fields(record, where=where)
    missing = [key for key in REQUIRED_PULSE_FIELDS if key not in record]
    if missing:
        raise SelfContactSchemaError(f"{where} missing required field(s) {missing}")
    label = str(record["config_label"])
    if label not in CONFIG_LABELS:
        raise SelfContactSchemaError(f"{where}.config_label={label!r} is not C1..C8")
    revisit = int(record["revisit_index"])
    block_index = int(record["block_index"])
    block_kind = str(record["block_kind"])
    if block_kind not in {"near", "delayed"}:
        raise SelfContactSchemaError(f"{where}.block_kind must be near|delayed")
    q_arm = list(_finite_len(record["q_arm_deg"], ARM_DQ_JOINT_COUNT, where=f"{where}.q_arm_deg"))
    visual = _validate_metric_block(record["visual_contact"], where=f"{where}.visual_contact")
    plateau = _validate_metric_block(
        record["position_plateau"], where=f"{where}.position_plateau"
    )
    true_contact = _require_flag(record, "true_contact_valid", where=where)
    if true_contact and not (visual["valid"] and plateau["valid"]):
        raise SelfContactSchemaError(
            f"{where}.true_contact_valid requires visual_contact.valid "
            "and position_plateau.valid; Load/Current are corroborative only"
        )
    # A record that itself denies hardware cannot be validated as a live
    # contact even if the caller passed hardware_opened=True.
    opened = bool(hardware_opened) and not dry_run
    if record.get("hardware_opened") is False:
        opened = False
    if dry_run or not opened:
        if true_contact or visual["valid"] or plateau["valid"]:
            raise SelfContactSchemaError(
                f"{where} dry-run / hardware_opened=false cannot claim contact"
            )
    canvas = record.get("u_hat_canvas")
    frame_hw = record.get("u_hat_frame_hw")
    if frame_hw is not None and frame_hw != "":
        if not isinstance(frame_hw, (list, tuple)) or len(frame_hw) < 2:
            raise SelfContactSchemaError(f"{where}.u_hat_frame_hw must be [H, W]")
        frame_hw = (int(frame_hw[0]), int(frame_hw[1]))
        if not is_planning_224_hw(frame_hw) and (
            true_contact or bool(record.get("u_hat_canonicalized"))
        ):
            raise SelfContactSchemaError(
                f"{where} cannot label native/non-224 pixels as u_hat_px224"
            )
    if canvas not in {None, "", PLANNING_CANVAS_NAME} and (
        true_contact or bool(record.get("u_hat_canonicalized"))
    ):
        raise SelfContactSchemaError(
            f"{where}.u_hat_canvas={canvas!r} is not {PLANNING_CANVAS_NAME}"
        )
    u_hat, u_source, u_hat_raw, u_hat_canonicalized, u_hat_canonical = _validate_u_hat(
        record,
        where=where,
        dry_run=dry_run or not opened,
        true_contact_valid=true_contact,
    )
    background = _validate_background(
        record["background_registration"],
        where=f"{where}.background_registration",
        dry_run=dry_run or not opened,
    )
    column = _validate_frame(
        record["column_frame"],
        source="column",
        where=f"{where}.column_frame",
        dry_run=dry_run or not opened,
    )
    wrist = _validate_frame(
        record["wrist_frame"],
        source="wrist",
        where=f"{where}.wrist_frame",
        dry_run=dry_run or not opened,
    )
    load_current = _validate_load_current(
        record["load_current"],
        where=f"{where}.load_current",
        dry_run=dry_run or not opened,
    )
    gripper = _as_mapping(
        record["gripper_command_readback"],
        where=f"{where}.gripper_command_readback",
    )
    if dry_run or not opened:
        summary = _as_mapping(
            gripper.get("command_sequence_summary") or {},
            where=f"{where}.gripper_command_readback.command_sequence_summary",
        )
        if int(summary.get("n_steps") or 0) != 0:
            raise SelfContactSchemaError(
                f"{where} dry-run cannot claim a gripper close sequence"
            )
    velocity_available = _require_flag(record, "velocity_available", where=where)
    moving_available = _require_flag(record, "moving_available", where=where)
    if dry_run or not opened:
        if velocity_available or moving_available:
            raise SelfContactSchemaError(
                f"{where} dry-run must report Velocity/Moving as unavailable"
            )
        if record.get("goal_position") not in {None, ""} or record.get(
            "present_position"
        ) not in {None, ""}:
            raise SelfContactSchemaError(
                f"{where} dry-run Goal/Present must be empty"
            )
    stamp = record.get("timestamp")
    predictor: dict[str, Any] = {}
    if record.get("fk_tip_xyz_m") is not None:
        predictor["fk_tip_xyz_m"] = list(
            _finite_len(record["fk_tip_xyz_m"], 3, where=f"{where}.fk_tip_xyz_m")
        )
    if record.get("fk_joint_jacobian_m_per_deg") is not None:
        jac = record["fk_joint_jacobian_m_per_deg"]
        if not isinstance(jac, (list, tuple)) or len(jac) != 3:
            raise SelfContactSchemaError(f"{where}.fk_joint_jacobian_m_per_deg must be 3x5")
        predictor["fk_joint_jacobian_m_per_deg"] = [
            list(_finite_len(row, ARM_DQ_JOINT_COUNT, where=f"{where}.fk_joint_jacobian_m_per_deg"))
            for row in jac
        ]
    if record.get("prediction_provider") is not None:
        provider = record["prediction_provider"]
        if isinstance(provider, Mapping):
            predictor["prediction_provider"] = dict(provider)
        else:
            predictor["prediction_provider"] = {"name": str(provider)}
    if "prediction_valid" in record:
        predictor["prediction_valid"] = bool(record.get("prediction_valid"))
    if record.get("prediction_failure_reason") is not None:
        predictor["prediction_failure_reason"] = str(record["prediction_failure_reason"])
    roi_xyxy = record.get("gripper_roi_xyxy")
    if roi_xyxy is not None and roi_xyxy != "":
        if not isinstance(roi_xyxy, (list, tuple)) or len(roi_xyxy) != 4:
            raise SelfContactSchemaError(f"{where}.gripper_roi_xyxy must be [x0,y0,x1,y1]")
        predictor["gripper_roi_xyxy"] = [int(round(float(v))) for v in roi_xyxy]
    if record.get("gripper_roi_provenance") not in {None, ""}:
        predictor["gripper_roi_provenance"] = str(record["gripper_roi_provenance"])
    if isinstance(record.get("gripper_roi_diagnostics"), Mapping):
        predictor["gripper_roi_diagnostics"] = dict(record["gripper_roi_diagnostics"])
    return {
        "sample_id": str(record.get("sample_id") or f"{label}:r{revisit}"),
        "config_label": label,
        "revisit_index": revisit,
        "block_index": block_index,
        "block_kind": block_kind,
        "timestamp": None if stamp in {None, ""} else str(stamp),
        "q_arm_deg": q_arm,
        "goal_position": record.get("goal_position"),
        "present_position": record.get("present_position"),
        "velocity_available": velocity_available,
        "moving_available": moving_available,
        "velocity": record.get("velocity"),
        "moving": record.get("moving"),
        "load_current": load_current,
        "gripper_command_readback": dict(gripper),
        "visual_contact": visual,
        "position_plateau": plateau,
        "true_contact_valid": true_contact,
        "u_hat_px224": None if u_hat_raw is None else [u_hat_raw[0], u_hat_raw[1]],
        "u_hat_px224_raw": None if u_hat_raw is None else [u_hat_raw[0], u_hat_raw[1]],
        "u_hat_px224_canonical": (
            None if u_hat_canonical is None else [u_hat_canonical[0], u_hat_canonical[1]]
        ),
        "u_hat_canonicalized": bool(u_hat_canonicalized and u_hat_canonical is not None),
        "u_hat_source": u_source,
        "u_hat_canvas": (
            None
            if canvas in {None, ""}
            else str(canvas)
        ),
        "u_hat_frame_hw": (
            None
            if frame_hw is None or frame_hw == ""
            else [int(frame_hw[0]), int(frame_hw[1])]
        ),
        "timestamps": record.get("timestamps"),
        "commands": record.get("commands"),
        "background_registration": background,
        "background_registration_valid": bool(background["valid"]),
        "camera_break_frame": bool(
            background.get("camera_identity_break", background["camera_break_frame"])
        ),
        "camera_identity_break": bool(
            background.get("camera_identity_break", background["camera_break_frame"])
        ),
        "background_inliers_insufficient": not bool(background["n_inliers_sufficient"]),
        "background_sensitivity_available": bool(background["valid"]),
        "estimand": ESTIMAND_B_RAW,
        "claim_A_intrinsic_fk_unauthorized": True,
        "column_frame": column,
        "wrist_frame": wrist,
        "failure_reason": record.get("failure_reason"),
        "adjudication": record.get("adjudication"),
        "restored": record.get("restored"),
        "restore_target": record.get("restore_target"),
        "restore_final": record.get("restore_final"),
        "restore_error": record.get("restore_error"),
        "abort_reason": record.get("abort_reason"),
        "pre_motion_identity_valid": record.get("pre_motion_identity_valid"),
        "pre_motion_identity": record.get("pre_motion_identity"),
        "seek_identity_break": record.get("seek_identity_break"),
        "visual_truth": record.get("visual_truth"),
        "q_arm_source": record.get("q_arm_source"),
        "seek_schema": record.get("seek_schema"),
        "evidence_schema_version": record.get("evidence_schema_version"),
        "images_persisted_complete": record.get("images_persisted_complete"),
        "gripper_roi_by_view": record.get("gripper_roi_by_view"),
        "pre_motion_handshake": record.get("pre_motion_handshake")
        or record.get("pre_motion_identity"),
        "visual_contact_per_view": record.get("visual_contact_per_view"),
        "hardware_opened": bool(opened),
        "contact_claimed": bool(record.get("contact_claimed", true_contact)) and bool(opened),
        **predictor,
    }


def planned_dry_run_pulse(sample: Mapping[str, Any]) -> dict[str, Any]:
    pulse = {
        "sample_id": sample["sample_id"],
        "config_label": sample["config_label"],
        "revisit_index": sample["revisit_index"],
        "block_index": sample["block_index"],
        "block_kind": sample["block_kind"],
        "timestamp": None,
        "q_arm_deg": list(sample["q_arm_deg"]),
        "goal_position": None,
        "present_position": None,
        "velocity_available": False,
        "moving_available": False,
        "velocity": None,
        "moving": None,
        "load_current": empty_load_current(),
        "gripper_command_readback": empty_gripper_sequence(),
        "visual_contact": empty_visual_contact(),
        "position_plateau": empty_position_plateau(),
        "true_contact_valid": False,
        "u_hat_px224": None,
        "u_hat_px224_raw": None,
        "u_hat_px224_canonical": None,
        "u_hat_canonicalized": False,
        "u_hat_source": None,
        "u_hat_canvas": None,
        "u_hat_frame_hw": None,
        "timestamps": None,
        "commands": None,
        "background_registration": empty_background_registration(),
        "column_frame": empty_frame_provenance("column"),
        "wrist_frame": empty_frame_provenance("wrist"),
        "hardware_opened": False,
        "contact_claimed": False,
        "seek_schema": None,
        "q_arm_source": "planned",
        "abort_reason": None,
        "restored": None,
        "restore_target": None,
        "restore_final": None,
        "restore_error": None,
        "failure_reason": None,
        "adjudication": None,
    }
    return validate_self_contact_record(
        pulse, where=str(pulse["sample_id"]), dry_run=True, hardware_opened=False
    )


def plan_self_contact_pulses() -> dict[str, Any]:
    samples = planned_samples()
    pulses = [planned_dry_run_pulse(sample) for sample in samples]
    labels = [row["config_label"] for row in samples]
    n_pairs = count_adjacent_same_config_pairs(samples)
    if len(pulses) != N_PULSES:
        raise SelfContactSchemaError(f"planned pulses must be {N_PULSES}, got {len(pulses)}")
    if n_pairs != N_ADJACENT_SAME_CONFIG_PAIRS:
        raise SelfContactSchemaError(
            f"adjacent same-config pairs must be {N_ADJACENT_SAME_CONFIG_PAIRS}, got {n_pairs}"
        )
    return {
        "schema": SELF_CONTACT_SCHEMA_ID,
        "protocol": PROTOCOL_SELF_CONTACT_V1,
        "protocol_status": FORMAL_PROTOCOL_STATUS,
        "protocol_formal": True,
        "printed_datum_protocol": {
            "protocol": PROTOCOL_PRINTED_DATUM,
            "status": RETIRED_PROTOCOL_STATUS,
            "formal": False,
            "note": (
                "printed_datum / world_witness / center_h32mm is retired "
                "audit-only and is not the formal self-contact route"
            ),
        },
        "mode": "dry_run_no_contact",
        "session_kind": "formal",
        "anchor_id": SELF_CONTACT_ANCHOR_ID,
        "camera_id": "column",
        "hardware_opened": False,
        "contact_armed": False,
        "contact_claimed": False,
        "n_configs": N_CONFIGS,
        "n_revisits": N_REVISITS,
        "n_near_blocks": N_NEAR_BLOCKS,
        "n_delayed_blocks": N_DELAYED_BLOCKS,
        "n_distinct_block_permutations": N_REVISITS,
        "n_pulses_planned": N_PULSES,
        "n_adjacent_same_config_pairs": n_pairs,
        "t_close_s": T_CLOSE_S,
        "t_distant_s": T_DISTANT_S,
        "min_n_same_config_t_close": MIN_N_SAME_CONFIG_T_CLOSE,
        "min_n_cross_config_t_close": MIN_N_CROSS_CONFIG_T_CLOSE,
        "arm_dq_joint_count": ARM_DQ_JOINT_COUNT,
        "visit_order": "interleaved_blocks",
        "forbidden_visit_order": "blocked_by_config",
        "config_labels": list(CONFIG_LABELS),
        "visit_labels": labels,
        "pulses": pulses,
        "contacts": pulses,
        "notes": {
            "closed_state_is_not_physical_contact": True,
            "p1_closed_state_is_not_physical_contact": True,
            "load_current_is_corroborative_only": True,
            "u_hat_must_be_image_only_knife_edge": True,
            "dry_run_plans_pulses_only": True,
        },
    }


def validate_self_contact_dry_run_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    document = _as_mapping(payload, where="self_contact_dry_run")
    refuse_fabricated_pixel_truth(document, where="self_contact_dry_run")
    if resolve_protocol(document) != PROTOCOL_SELF_CONTACT_V1:
        raise SelfContactSchemaError("dry-run document protocol must be self_contact_v1")
    if document.get("schema") not in {SELF_CONTACT_SCHEMA_ID, None}:
        if document.get("schema") == PRINTED_DATUM_SCHEMA_ID:
            raise SelfContactSchemaError(
                "self_contact dry-run must not use the retired printed_datum schema id"
            )
    if document.get("hardware_opened") is not False:
        raise SelfContactSchemaError("dry-run hardware_opened must be false")
    if document.get("contact_armed") is not False:
        raise SelfContactSchemaError("dry-run contact_armed must be false")
    if document.get("contact_claimed") is not False:
        raise SelfContactSchemaError("dry-run cannot claim contact")
    if str(document.get("anchor_id") or SELF_CONTACT_ANCHOR_ID) in RETIRED_SEAT_IDS:
        raise SelfContactSchemaError("self_contact must not use a retired printed seat id")
    pulses = document.get("pulses")
    if not isinstance(pulses, list) or len(pulses) != N_PULSES:
        raise SelfContactSchemaError(f"dry-run must plan exactly {N_PULSES} pulses")
    validated = [
        validate_self_contact_record(
            row, where=f"pulses[{index}]", dry_run=True, hardware_opened=False
        )
        for index, row in enumerate(pulses)
    ]
    n_pairs = count_adjacent_same_config_pairs(validated)
    if n_pairs != N_ADJACENT_SAME_CONFIG_PAIRS:
        raise SelfContactSchemaError(
            f"dry-run adjacent same-config pairs must be {N_ADJACENT_SAME_CONFIG_PAIRS}"
        )
    if int(document.get("n_distinct_block_permutations") or 0) != N_REVISITS:
        raise SelfContactSchemaError("dry-run must record 9 distinct permutations")
    return document


def self_contact_collection_input_schema() -> dict[str, Any]:
    return {
        "schema": SELF_CONTACT_SCHEMA_ID,
        "protocol": PROTOCOL_SELF_CONTACT_V1,
        "protocol_status": FORMAL_PROTOCOL_STATUS,
        "protocol_formal": True,
        "required_top_level": [
            "schema",
            "protocol",
            "anchor_id",
            "camera_id",
            "contacts",
        ],
        "required_per_contact": list(REQUIRED_PULSE_FIELDS),
        "forbidden_per_contact": sorted(
            FORBIDDEN_PRINTED_FIELDS | FORBIDDEN_PIXEL_TRUTH_KEYS
        ),
        "retired_printed_datum": {
            "protocol": PROTOCOL_PRINTED_DATUM,
            "status": RETIRED_PROTOCOL_STATUS,
            "formal": False,
            "fields_not_required": sorted(FORBIDDEN_PRINTED_FIELDS),
        },
        "rules": {
            "n_config_labels": N_CONFIGS,
            "revisits_per_config": N_REVISITS,
            "n_near_blocks": N_NEAR_BLOCKS,
            "n_delayed_blocks": N_DELAYED_BLOCKS,
            "n_distinct_block_permutations": N_REVISITS,
            "n_contacts_planned": N_PULSES,
            "n_adjacent_same_config_pairs": N_ADJACENT_SAME_CONFIG_PAIRS,
            "t_close_s": T_CLOSE_S,
            "t_distant_s": T_DISTANT_S,
            "min_n_same_config_t_close": MIN_N_SAME_CONFIG_T_CLOSE,
            "min_n_cross_config_t_close": MIN_N_CROSS_CONFIG_T_CLOSE,
            "arm_dq_joint_count": ARM_DQ_JOINT_COUNT,
            "q_arm_deg": (
                "length-5 arm readback in degrees.  Gripper is excluded "
                "from configuration dq."
            ),
            "estimand": ESTIMAND_B_RAW,
            "u_hat_px224": (
                "B primary alias of u_hat_px224_raw on the live planning "
                "224 canvas.  Never FK, M0, M1, FCPG, synthetic q_true, "
                "or snapshot:metal.  Never replaced by a background affine."
            ),
            "u_hat_px224_raw": (
                "Image-only knife-edge observation.  Drives B adjudication "
                "and FCPG label export.  Background invalid must not blank it."
            ),
            "u_hat_px224_canonical": (
                "Optional A-sensitivity point after a valid background "
                "transform.  Secondary only.  Never the default training label."
            ),
            "u_hat_source": (
                "Must be one of: " + ", ".join(sorted(IMAGE_ONLY_KNIFE_EDGE_SOURCES))
            ),
            "true_contact_valid": (
                "Joint predicate: visual_contact.valid AND "
                "position_plateau.valid.  Load/Current never suffice."
            ),
            "background_registration_valid": (
                "Optional A diagnostic.  Dual-ROI / ORB success does not "
                "gate B usable, PASS, or FAIL."
            ),
            "condition_7": (
                "printed_datum uhat_stability_gate does not apply.  "
                "B PASS-blocker is hardware_opened && true_contact_valid && "
                "prediction_valid && image-only source && planning224 && "
                "!camera_identity_break.  Missing background is A "
                "unavailable, not B INCONCLUSIVE."
            ),
            "dry_run": (
                "Plans 72 pulses only.  hardware_opened=false.  "
                "Frames/Goal/Present/Load empty.  Must not claim contact."
            ),
            "closed_state": (
                "Current p1 closed_state is not physical contact."
            ),
            "load_current": "Corroborative only; never defines contact.",
        },
    }


def wrap_legacy_contact_as_self_contact(
    legacy: Mapping[str, Any],
    *,
    true_contact_valid: bool = True,
    background_valid: bool = True,
    camera_break_frame: bool = False,
    where: str = "legacy_contact",
) -> dict[str, Any]:
    """Copy analysis predictors from a printed synthetic row without seat/D/witness."""

    refuse_fabricated_pixel_truth(legacy, where=where)
    q_raw = legacy.get("q_arm_deg", legacy.get("q_readback_deg"))
    q_arm = list(_finite_len(q_raw, ARM_DQ_JOINT_COUNT, where=f"{where}.q_arm_deg"))
    u_raw = legacy.get("u_hat_px224", legacy.get("u_hat_px"))
    pulse = {
        "sample_id": legacy.get("sample_id")
        or f"{legacy.get('config_label')}:r{legacy.get('revisit_index')}",
        "config_label": legacy["config_label"],
        "revisit_index": int(legacy["revisit_index"]),
        "block_index": int(legacy.get("block_index") or max(int(legacy["revisit_index"]) - 1, 0)),
        "block_kind": str(legacy.get("block_kind") or ("delayed" if int(legacy.get("revisit_index", 1)) >= 9 else "near")),
        "timestamp": legacy.get("timestamp"),
        "q_arm_deg": q_arm,
        "goal_position": legacy.get("goal_position") or [0.0] * 6,
        "present_position": legacy.get("present_position") or [0.0] * 6,
        "velocity_available": bool(legacy.get("velocity_available", False)),
        "moving_available": bool(legacy.get("moving_available", False)),
        "load_current": {
            "load": None,
            "current": None,
            "read_status": "unavailable",
            "physical_witness_valid": False,
            "role": "corroborative_only",
        },
        "gripper_command_readback": empty_gripper_sequence(),
        "visual_contact": {
            "valid": bool(true_contact_valid),
            "metrics": {"source": "synthetic_analysis_fixture"},
        },
        "position_plateau": {
            "valid": bool(true_contact_valid),
            "metrics": {"source": "synthetic_analysis_fixture"},
        },
        "true_contact_valid": bool(true_contact_valid),
        "u_hat_px224": None if u_raw is None else list(u_raw),
        "u_hat_px224_raw": None if u_raw is None else list(u_raw),
        "u_hat_px224_canonical": (
            None
            if u_raw is None or not background_valid or camera_break_frame
            else list(u_raw)
        ),
        "u_hat_canonicalized": (
            bool(true_contact_valid)
            and bool(background_valid)
            and not camera_break_frame
            and u_raw is not None
        ),
        "u_hat_source": (
            "image_only_knife_edge" if true_contact_valid and u_raw is not None else None
        ),
        "background_registration": {
            "valid": bool(background_valid) and not camera_break_frame,
            "n_inliers": 12 if background_valid and not camera_break_frame else 1,
            "n_inliers_sufficient": bool(background_valid) and not camera_break_frame,
            "spatial_coverage": 0.6 if background_valid and not camera_break_frame else 0.0,
            "spatial_coverage_sufficient": bool(background_valid) and not camera_break_frame,
            "residual_px224": 0.2 if background_valid else 4.0,
            "transform_2x3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            if background_valid and not camera_break_frame
            else None,
            "camera_break_frame": bool(camera_break_frame),
            "movable_objects_dominated": False,
            "inliers_in_exclude_rois": 0,
            "reason": None if background_valid and not camera_break_frame else "synthetic_invalid",
            "diagnostics": {"source": "synthetic_analysis_fixture"},
        },
        "column_frame": {
            "source": "column",
            "path": legacy.get("frame_path") or "synthetic/self_contact/column.png",
            "timestamp": legacy.get("timestamp") or "synthetic",
            "hardware_opened": True,
        },
        "wrist_frame": {
            "source": "wrist",
            "path": "synthetic/self_contact/wrist.png",
            "timestamp": legacy.get("timestamp") or "synthetic",
            "hardware_opened": True,
        },
        "u_hat_canvas": PLANNING_CANVAS_NAME if u_raw is not None else None,
        "u_hat_frame_hw": list(PLANNING_CANVAS_HW) if u_raw is not None else None,
        "hardware_opened": True,
    }
    if legacy.get("fk_tip_xyz_m") is not None:
        pulse["fk_tip_xyz_m"] = legacy["fk_tip_xyz_m"]
    else:
        pulse["fk_tip_xyz_m"] = [0.0, 0.0, 0.0]
    if legacy.get("fk_joint_jacobian_m_per_deg") is not None:
        pulse["fk_joint_jacobian_m_per_deg"] = legacy["fk_joint_jacobian_m_per_deg"]
    else:
        pulse["fk_joint_jacobian_m_per_deg"] = [[0.0] * ARM_DQ_JOINT_COUNT] * 3
    pulse["prediction_valid"] = True
    pulse["prediction_provider"] = legacy.get("prediction_provider") or {
        "name": "synthetic_analysis_fixture.copied_printed_nominal_fk",
        "role": "nominal_m1_predictor_input_only",
        "not_pixel_gt": True,
        "q_true_used": False,
        "tip_units": "m",
        "jacobian_units": "m_per_mechanical_deg",
        "allow_zero_synthetic_fixture": True,
    }
    return validate_self_contact_record(
        pulse, where=where, dry_run=False, hardware_opened=True
    )


def as_self_contact_collection(
    payload: Mapping[str, Any],
    *,
    true_contact_valid: bool = True,
    background_valid: bool = True,
    camera_break_frame: bool = False,
    session_kind: str | None = None,
) -> dict[str, Any]:
    """Strip printed seat/D/witness fields from a collection for protocol tests."""

    raw = _as_mapping(payload, where="collection")
    refuse_fabricated_pixel_truth(raw, where="collection")
    contacts_raw = raw.get("contacts")
    if not isinstance(contacts_raw, list) or not contacts_raw:
        raise SelfContactSchemaError("collection.contacts must be a non-empty list")
    contacts = [
        wrap_legacy_contact_as_self_contact(
            row,
            true_contact_valid=true_contact_valid,
            background_valid=background_valid,
            camera_break_frame=camera_break_frame,
            where=f"contacts[{index}]",
        )
        for index, row in enumerate(contacts_raw)
    ]
    leaked = sorted(FORBIDDEN_PRINTED_FIELDS & set().union(*(set(row) for row in contacts)))
    if leaked:
        raise SelfContactSchemaError(f"self_contact contacts leaked printed fields {leaked}")
    return {
        "schema": SELF_CONTACT_SCHEMA_ID,
        "protocol": PROTOCOL_SELF_CONTACT_V1,
        "protocol_status": FORMAL_PROTOCOL_STATUS,
        "protocol_formal": True,
        "anchor_id": SELF_CONTACT_ANCHOR_ID,
        "camera_id": str(raw.get("camera_id") or "column"),
        "session_kind": session_kind or raw.get("session_kind") or "synthetic_fixture",
        "branch": raw.get("branch"),
        "contacts": contacts,
        "estimand": ESTIMAND_B_RAW,
        "claim_A_intrinsic_fk_unauthorized": True,
        "truth_note": (
            "B primary is u_hat_px224_raw minus p_nominal(q).  "
            "FK tip/jacobian, if present, are M1 predictor inputs only.  "
            "Canonical / background transform is A-sensitivity secondary."
        ),
        "notes": {
            "closed_state_is_not_physical_contact": True,
            "load_current_is_corroborative_only": True,
        },
    }


def camera_identity_break_of(row: Mapping[str, Any]) -> bool:
    """Coarse framing / hash / large break-frame only.  Not ORB failure."""

    background = row.get("background_registration")
    if isinstance(background, Mapping):
        if bool(background.get("camera_identity_break", background.get("camera_break_frame"))):
            return True
    return bool(row.get("camera_identity_break") or row.get("camera_break_frame"))


def _planning224_ok(row: Mapping[str, Any]) -> bool:
    canvas = row.get("u_hat_canvas")
    frame_hw = row.get("u_hat_frame_hw")
    if canvas not in {None, "", PLANNING_CANVAS_NAME}:
        return False
    if frame_hw is None or frame_hw == "":
        return True
    if not isinstance(frame_hw, (list, tuple)) or len(frame_hw) < 2:
        return False
    return is_planning_224_hw((int(frame_hw[0]), int(frame_hw[1])))


def pulse_usable_for_estimand_b(
    row: Mapping[str, Any],
    *,
    hardware_opened: bool | None = None,
) -> bool:
    """B usable gate.  Background registration is not a B requirement."""

    opened = bool(row.get("hardware_opened") if hardware_opened is None else hardware_opened)
    source = row.get("u_hat_source")
    raw = row.get("u_hat_px224_raw")
    if raw is None and not bool(row.get("u_hat_canonicalized")):
        raw = row.get("u_hat_px224")
    prediction = row.get("prediction_valid")
    if prediction is None:
        prediction = (
            row.get("fk_tip_xyz_m") is not None
            and row.get("fk_joint_jacobian_m_per_deg") is not None
        )
    return (
        opened
        and bool(row.get("true_contact_valid"))
        and bool(prediction)
        and str(source) in IMAGE_ONLY_KNIFE_EDGE_SOURCES
        and _planning224_ok(row)
        and raw is not None
        and not camera_identity_break_of(row)
    )


def export_fcpg_training_labels(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Export B labels from raw knife-edge.  Canonical never defaults in."""

    labels: list[dict[str, Any]] = []
    n_skipped_canonical_only = 0
    for row in rows:
        if not pulse_usable_for_estimand_b(row):
            continue
        raw = row.get("u_hat_px224_raw") or row.get("u_hat_px224")
        if raw is None:
            continue
        if bool(row.get("u_hat_canonicalized")) and row.get("u_hat_px224_raw") is None:
            n_skipped_canonical_only += 1
            continue
        labels.append(
            {
                "sample_id": row.get("sample_id"),
                "config_label": row.get("config_label"),
                "u_hat_px224": list(raw),
                "u_hat_source": "image_only_raw",
                "q_arm_deg": row.get("q_arm_deg"),
                "fk_tip_xyz_m": row.get("fk_tip_xyz_m"),
                "label_kind": "r_raw_minus_nominal_predictor",
                "canonical_used": False,
                "matches_live_planning_canvas": True,
            }
        )
    return {
        "estimand": ESTIMAND_B_RAW,
        "u_hat_field": "u_hat_px224_raw",
        "matches_live_planning_canvas": True,
        "canonical_default_forbidden": True,
        "claim_A_intrinsic_fk_unauthorized": True,
        "n_labels": len(labels),
        "n_skipped_canonical_only": n_skipped_canonical_only,
        "labels": labels,
    }


def refuse_unauthorized_a_wording(text: str, *, where: str = "claim") -> None:
    hit = [phrase for phrase in FORBIDDEN_A_CLAIM_PHRASES if phrase in text]
    if hit:
        raise SelfContactSchemaError(
            f"{where} used unauthorized A/C/11.14 wording {hit}; "
            "self_contact_v1 may only authorize "
            f"{ESTIMAND_B_RAW}"
        )


def self_contact_claim_block(
    *,
    verdict: str,
    background_sensitivity_available: bool,
    authorized: bool,
) -> dict[str, Any]:
    """PASS authorizes B only.  A is unauthorized.  C is not this screen."""

    return {
        "estimand": ESTIMAND_B_RAW,
        "claim_A_intrinsic_fk_unauthorized": True,
        "background_sensitivity_available": bool(background_sensitivity_available),
        "authorized_claim": ESTIMAND_B_RAW if verdict == "PASS" and authorized else None,
        "claim_C_wrist_off_unauthorized": True,
        "does_not_confirm_11_14": True,
        "secondary_estimand_A": ESTIMAND_A_REGISTERED,
        "secondary_estimand_C": CLAIM_C_WRIST_OFF,
    }


def self_contact_validity_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """B usable gate.  Replaces printed ``uhat_stability_gate``.

    Invalid contact, missing prediction, non-image source, non-224
    canvas, or a camera identity break is INCONCLUSIVE.  Invalid
    background / missing dual ROI is A unavailable, not B
    INCONCLUSIVE.  This is a PASS-blocker and also blocks FAIL when
    the B usable set cannot adjudicate leftover.
    """

    n = len(rows)
    n_contact_invalid = 0
    n_prediction_invalid = 0
    n_source_invalid = 0
    n_planning_invalid = 0
    n_hardware_closed = 0
    n_break = 0
    n_reg_invalid = 0
    n_inliers_short = 0
    n_b_usable = 0
    for row in rows:
        if pulse_usable_for_estimand_b(row):
            n_b_usable += 1
        if not bool(row.get("true_contact_valid")):
            n_contact_invalid += 1
        prediction = row.get("prediction_valid")
        if prediction is None:
            prediction = (
                row.get("fk_tip_xyz_m") is not None
                and row.get("fk_joint_jacobian_m_per_deg") is not None
            )
        if not bool(prediction):
            n_prediction_invalid += 1
        if str(row.get("u_hat_source") or "") not in IMAGE_ONLY_KNIFE_EDGE_SOURCES:
            n_source_invalid += 1
        if not _planning224_ok(row):
            n_planning_invalid += 1
        if row.get("hardware_opened") is False:
            n_hardware_closed += 1
        if camera_identity_break_of(row):
            n_break += 1
        background = row.get("background_registration")
        if isinstance(background, Mapping):
            valid = bool(background.get("valid"))
            sufficient = bool(background.get("n_inliers_sufficient"))
        else:
            valid = bool(row.get("background_registration_valid"))
            sufficient = not bool(row.get("background_inliers_insufficient"))
        if not valid:
            n_reg_invalid += 1
        if not sufficient:
            n_inliers_short += 1
    reasons: list[str] = []
    if n_hardware_closed:
        reasons.append(
            f"hardware_opened is false on {n_hardware_closed}/{n or 0} "
            "pulses; B cannot adjudicate"
        )
    if n_contact_invalid:
        reasons.append(
            f"true_contact_valid is false on {n_contact_invalid}/{n or 0} "
            "pulses; invalid self-contact is INCONCLUSIVE"
        )
    if n_prediction_invalid:
        reasons.append(
            f"prediction_valid is false on {n_prediction_invalid}/{n or 0} "
            "pulses; missing nominal predictor is INCONCLUSIVE"
        )
    if n_source_invalid:
        reasons.append(
            f"u_hat_source is not image-only knife-edge on "
            f"{n_source_invalid}/{n or 0} pulses; INCONCLUSIVE"
        )
    if n_planning_invalid:
        reasons.append(
            f"u_hat is not planning224 on {n_planning_invalid}/{n or 0} "
            "pulses; INCONCLUSIVE"
        )
    if n_break:
        reasons.append(
            f"camera identity break on {n_break}/{n or 0} pulses; INCONCLUSIVE"
        )
    passed = not reasons
    a_unavailable = n_reg_invalid > 0 or n_inliers_short > 0
    return {
        "pass": passed,
        "replaces_uhat_stability_gate": True,
        "failure_is_inconclusive": True,
        "estimand": ESTIMAND_B_RAW,
        "claim_A_intrinsic_fk_unauthorized": True,
        "background_is_not_a_b_blocker": True,
        "n_pulses": n,
        "n_b_usable": n_b_usable,
        "n_true_contact_invalid": n_contact_invalid,
        "n_prediction_invalid": n_prediction_invalid,
        "n_source_invalid": n_source_invalid,
        "n_planning224_invalid": n_planning_invalid,
        "n_hardware_closed": n_hardware_closed,
        "n_background_registration_invalid": n_reg_invalid,
        "n_background_inliers_insufficient": n_inliers_short,
        "n_camera_break_frame": n_break,
        "n_camera_identity_break": n_break,
        "background_sensitivity_available": (not a_unavailable) and n > 0,
        "a_unavailable_diagnostic": (
            None
            if not a_unavailable
            else (
                f"background_registration invalid on {n_reg_invalid}/{n or 0} "
                "or inliers insufficient; A unauthorized / unavailable, "
                "not a B INCONCLUSIVE"
            )
        ),
        "reason": None if passed else "; ".join(reasons),
        "reasons": reasons,
    }
