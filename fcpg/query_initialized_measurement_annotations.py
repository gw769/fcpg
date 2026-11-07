"""Blind annotation validation and adjudication for measurement development."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fcpg.offline_pipeline_manifest import sha256_json
from fcpg.query_initialized_displacement_panel import (
    QUERY_DISPLACEMENT_PANEL_FORMAT,
    QUERY_DISPLACEMENT_PROTOCOL_ID,
)


QUERY_DISPLACEMENT_ANNOTATION_FORMAT = (
    "so101-fcpg-h-query-initialized-displacement-annotations-v2"
)
QUERY_DISPLACEMENT_TRUTH_FORMAT = (
    "so101-fcpg-h-query-initialized-displacement-truth-v2"
)
CLICK_FIELDS = (
    "query_fixed_tip_px",
    "query_moving_tip_px",
    "target_fixed_tip_px",
    "target_moving_tip_px",
)
VISIBILITY_VALUES = frozenset({"visible", "not_visible"})
NOT_VISIBLE_REASONS = frozenset(
    {
        "out_of_frame",
        "occluded",
        "motion_blur",
        "robot_absent",
        "identity_ambiguous",
        "other_unannotatable",
    }
)


@dataclass(frozen=True)
class PointAnnotation:
    visibility: str
    reason: str | None
    px: np.ndarray | None


def _point_annotation(row: dict[str, Any], field: str) -> PointAnnotation:
    status = row.get("point_status", {}).get(field)
    if not isinstance(status, dict):
        raise ValueError(f"annotation {field} has no point_status")
    visibility = str(status.get("visibility", ""))
    if visibility not in VISIBILITY_VALUES:
        raise ValueError(f"annotation {field} has invalid visibility")
    reason_value = status.get("reason")
    reason = None if reason_value is None else str(reason_value)
    value = row.get(field)
    if visibility == "visible":
        if reason not in {None, ""}:
            raise ValueError(f"visible annotation {field} cannot have a reason")
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError(f"visible annotation {field} must be finite [2]")
        if np.any(point < 0.0) or np.any(point > 224.0):
            raise ValueError(f"annotation {field} lies outside 224-pixel image")
        return PointAnnotation("visible", None, point)
    if value is not None:
        raise ValueError(f"non-visible annotation {field} must have null pixels")
    if reason not in NOT_VISIBLE_REASONS:
        raise ValueError(f"non-visible annotation {field} has invalid reason")
    return PointAnnotation("not_visible", reason, None)


def _expected_keys(panel: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    if (
        panel.get("format") != QUERY_DISPLACEMENT_PANEL_FORMAT
        or panel.get("protocol_id") != QUERY_DISPLACEMENT_PROTOCOL_ID
        or panel.get("split") != "development"
        or not panel.get("prediction_blind")
    ):
        raise ValueError("unexpected/non-blind query displacement panel")
    keys = tuple(
        (str(record["entry_id"]), view_index)
        for record in panel["records"]
        for view_index, _ in enumerate(record["views"])
    )
    if len(keys) != len(set(keys)):
        raise ValueError("panel entry/view keys are not unique")
    return keys


def _annotation_rows(
    payload: dict[str, Any],
    *,
    panel_sha256: str,
    expected: tuple[tuple[str, int], ...],
) -> tuple[str, dict[tuple[str, int], tuple[PointAnnotation, ...]]]:
    if (
        payload.get("format") != QUERY_DISPLACEMENT_ANNOTATION_FORMAT
        or payload.get("protocol_id") != QUERY_DISPLACEMENT_PROTOCOL_ID
        or payload.get("panel_sha256") != panel_sha256
        or payload.get("complete") is not True
    ):
        raise ValueError("annotation payload does not bind to the panel")
    result: dict[tuple[str, int], tuple[PointAnnotation, ...]] = {}
    annotators: set[str] = set()
    for row in payload.get("annotations", []):
        key = (str(row["entry_id"]), int(row["view_index"]))
        if key in result:
            raise ValueError("annotation payload repeats an entry/view")
        annotator = str(row.get("annotator", "")).strip()
        if not annotator:
            raise ValueError("annotation row has no annotator identity")
        annotators.add(annotator)
        if set(row.get("point_status", {})) != set(CLICK_FIELDS):
            raise ValueError("annotation point_status fields differ from click contract")
        result[key] = tuple(_point_annotation(row, field) for field in CLICK_FIELDS)
    if len(annotators) != 1:
        raise ValueError("one annotation file must belong to one annotator")
    if set(result) != set(expected):
        raise ValueError("annotation file is incomplete or contains foreign keys")
    return next(iter(annotators)), result


def validate_query_displacement_annotation(
    panel: dict[str, Any],
    *,
    panel_sha256: str,
    annotation_payload: dict[str, Any],
) -> tuple[
    str,
    dict[tuple[str, int], tuple[PointAnnotation, ...]],
]:
    """Validate one immutable annotation file without implying label semantics.

    This is intentionally weaker than :func:`merge_query_displacement_annotations`:
    it proves only the file/panel binding, key coverage and point structure.  In
    particular, the legacy field names do not prove that a human could observe
    the physical fixed/moving jaw identity.
    """

    expected = _expected_keys(panel)
    return _annotation_rows(
        annotation_payload,
        panel_sha256=panel_sha256,
        expected=expected,
    )


def merge_query_displacement_annotations(
    panel: dict[str, Any],
    *,
    panel_sha256: str,
    primary_payloads: tuple[dict[str, Any], dict[str, Any]],
    adjudication_payload: dict[str, Any] | None = None,
    disagreement_threshold_px: float = 6.0,
) -> dict[str, Any]:
    """Merge two blind annotations without inventing coordinates for hidden tips.

    A third annotation is required when the two primary annotators disagree on
    visibility or when two visible clicks differ by more than six pixels.
    """

    if disagreement_threshold_px != 6.0:
        raise ValueError("the annotation protocol fixes 6 px adjudication")
    expected = _expected_keys(panel)
    first_name, first = _annotation_rows(
        primary_payloads[0], panel_sha256=panel_sha256, expected=expected
    )
    second_name, second = _annotation_rows(
        primary_payloads[1], panel_sha256=panel_sha256, expected=expected
    )
    if first_name == second_name:
        raise ValueError("primary annotations must be independent annotators")
    third_name = None
    third = None
    if adjudication_payload is not None:
        third_name, third = _annotation_rows(
            adjudication_payload, panel_sha256=panel_sha256, expected=expected
        )
        if third_name in {first_name, second_name}:
            raise ValueError("adjudicator must be independent of both primaries")

    records = []
    disputed_points = 0
    visibility_disagreements = 0
    visible_consensus_points = 0
    not_visible_consensus_points = 0
    max_disagreement = 0.0
    for entry_id, view_index in expected:
        key = (entry_id, view_index)
        a = first[key]
        b = second[key]
        consensus_points: list[list[float] | None] = []
        consensus_visible: list[bool] = []
        consensus_reasons: list[str | None] = []
        disagreement_px: list[float | None] = []
        adjudicated: list[bool] = []
        for point_index, (first_point, second_point) in enumerate(zip(a, b)):
            visibility_disputed = (
                first_point.visibility != second_point.visibility
            )
            point_disagreement = None
            coordinate_disputed = False
            if (
                first_point.visibility == "visible"
                and second_point.visibility == "visible"
            ):
                assert first_point.px is not None and second_point.px is not None
                point_disagreement = float(
                    np.linalg.norm(first_point.px - second_point.px)
                )
                max_disagreement = max(max_disagreement, point_disagreement)
                coordinate_disputed = point_disagreement > disagreement_threshold_px
            needs_third = visibility_disputed or coordinate_disputed
            if needs_third and third is None:
                kind = "visibility" if visibility_disputed else "coordinate"
                raise RuntimeError(
                    f"annotation point {point_index} has disputed {kind}; "
                    "third adjudication required"
                )
            if visibility_disputed:
                visibility_disagreements += 1
            if coordinate_disputed:
                disputed_points += 1

            if needs_third:
                assert third is not None
                third_point = third[key][point_index]
                if visibility_disputed:
                    chosen = third_point
                    if chosen.visibility == "visible":
                        assert chosen.px is not None
                        consensus_points.append(chosen.px.tolist())
                        consensus_visible.append(True)
                        consensus_reasons.append(None)
                        visible_consensus_points += 1
                    else:
                        consensus_points.append(None)
                        consensus_visible.append(False)
                        consensus_reasons.append(chosen.reason)
                        not_visible_consensus_points += 1
                else:
                    if third_point.visibility != "visible" or third_point.px is None:
                        raise ValueError(
                            "coordinate adjudication requires a visible third click"
                        )
                    assert first_point.px is not None and second_point.px is not None
                    point = np.median(
                        np.stack((first_point.px, second_point.px, third_point.px)),
                        axis=0,
                    )
                    consensus_points.append(point.tolist())
                    consensus_visible.append(True)
                    consensus_reasons.append(None)
                    visible_consensus_points += 1
            elif first_point.visibility == "visible":
                assert first_point.px is not None and second_point.px is not None
                point = (first_point.px + second_point.px) / 2.0
                consensus_points.append(point.tolist())
                consensus_visible.append(True)
                consensus_reasons.append(None)
                visible_consensus_points += 1
            else:
                consensus_points.append(None)
                consensus_visible.append(False)
                reason = (
                    first_point.reason
                    if first_point.reason == second_point.reason
                    else "annotator_reason_disagreement"
                )
                consensus_reasons.append(reason)
                not_visible_consensus_points += 1
            disagreement_px.append(point_disagreement)
            adjudicated.append(needs_third)
        records.append(
            {
                "entry_id": entry_id,
                "view_index": view_index,
                "query_tip_px": consensus_points[:2],
                "target_tip_px": consensus_points[2:],
                "query_tip_visible": consensus_visible[:2],
                "target_tip_visible": consensus_visible[2:],
                "unavailable_reason": consensus_reasons,
                "primary_disagreement_px": disagreement_px,
                "adjudicated": adjudicated,
            }
        )
    payload = {
        "format": QUERY_DISPLACEMENT_TRUTH_FORMAT,
        "protocol_id": QUERY_DISPLACEMENT_PROTOCOL_ID,
        "panel_sha256": panel_sha256,
        "primary_annotators": [first_name, second_name],
        "adjudicator": third_name,
        "disagreement_threshold_px": disagreement_threshold_px,
        "max_primary_disagreement_px": max_disagreement,
        "adjudicated_point_count": disputed_points,
        "visibility_disagreement_count": visibility_disagreements,
        "visible_consensus_point_count": visible_consensus_points,
        "not_visible_consensus_point_count": not_visible_consensus_points,
        "coordinate_coverage": visible_consensus_points / (len(expected) * 4),
        "center": "strict_midpoint_not_clicked",
        "records": records,
    }
    payload["content_sha256"] = sha256_json(payload)
    return payload


__all__ = [
    "CLICK_FIELDS",
    "NOT_VISIBLE_REASONS",
    "QUERY_DISPLACEMENT_ANNOTATION_FORMAT",
    "QUERY_DISPLACEMENT_TRUTH_FORMAT",
    "VISIBILITY_VALUES",
    "merge_query_displacement_annotations",
    "validate_query_displacement_annotation",
]
