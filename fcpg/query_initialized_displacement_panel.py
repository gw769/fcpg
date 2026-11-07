"""Prediction-blind development panel for displacement-clocked measurement."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from fcpg.causal_displacement_observer import (
    DISPLACEMENT_SEGMENT_MM,
    causal_displacement_events_for_windows,
)
from fcpg.offline_pipeline_manifest import sha256_json, stable_fraction
from fcpg.temporal_grounding_protocol import select_roleless_views


QUERY_DISPLACEMENT_PANEL_FORMAT = (
    "so101-fcpg-h-query-initialized-displacement-development-panel-v1"
)
QUERY_DISPLACEMENT_PROTOCOL_ID = (
    "so101-fcpg-h-query-initialized-displacement-measurement-v1"
)
QUERY_DISPLACEMENT_PANEL_SEED = 12_680


def excluded_contract_sources(contract: dict[str, Any]) -> set[int]:
    """Collect every source exposed by the retired development contract."""

    result = {int(value) for value in contract.get("retired_source_indices", [])}
    for row in contract.get("records", []):
        if "source_index" not in row:
            raise ValueError("exclusion contract record has no source_index")
        result.add(int(row["source_index"]))
    if not result:
        raise ValueError("exclusion contract names no previously exposed source")
    return result


def _stable_seed(seed: int, *parts: object) -> int:
    digest = sha256_json({"seed": int(seed), "parts": list(parts)})
    return int(digest[:15], 16)


def _rank(seed: int, label: str, *identity: object) -> tuple[float, str]:
    text = ":".join(map(str, identity))
    return stable_fraction(seed, label, text), text


def _candidate_windows(
    index: Any,
    *,
    excluded_sources: set[int],
    seed: int,
    allowed_sources: set[int] | None = None,
    required_views: tuple[str, ...] | None = None,
    required_view_role: str | None = None,
    required_views_by_source: dict[int, tuple[str, str]] | None = None,
) -> dict[int, dict[int, list[int]]]:
    grouped: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    split = np.asarray(index.arrays["window_split_index"])
    source = np.asarray(index.arrays["window_source_index"])
    episode = np.asarray(index.arrays["window_episode"])
    for raw_window in np.flatnonzero(split == 1):
        window = int(raw_window)
        source_index = int(source[window])
        if source_index in excluded_sources:
            continue
        if allowed_sources is not None and source_index not in allowed_sources:
            continue
        episode_index = int(episode[window])
        record = index.episodes[episode_index]
        row = record["manifest_row"]
        views = row.get("view_keys") or []
        source_views = (
            None
            if required_views_by_source is None
            else required_views_by_source.get(source_index)
        )
        frozen_views = source_views if source_views is not None else required_views
        if frozen_views is None and len(views) < 2:
            continue
        if frozen_views is not None and not set(frozen_views).issubset(views):
            continue
        if required_view_role == "column":
            slots = tuple(map(str, row.get("lewm_v12_slot_views") or ()))
            if len(slots) != 2 or slots[0] != required_views[0]:
                continue
        if required_view_role == "external_wrist_by_source":
            slots = tuple(map(str, row.get("lewm_v12_slot_views") or ()))
            if slots != frozen_views:
                continue
        grouped[source_index][episode_index].append(window)
    for source_index, episodes in grouped.items():
        for episode_index, windows in episodes.items():
            windows.sort(
                key=lambda window: _rank(
                    seed, "query-displacement-window", source_index, episode_index, window
                )
            )
    return grouped


def select_query_initialized_displacement_entries(
    index: Any,
    *,
    kinematics: Any,
    excluded_sources: set[int],
    seed: int = QUERY_DISPLACEMENT_PANEL_SEED,
    source_count: int = 8,
    episodes_per_source: int = 8,
    max_window_attempts_per_episode: int = 8,
    allowed_sources: set[int] | None = None,
    required_views: tuple[str, ...] | None = None,
    required_view_role: str | None = None,
    required_views_by_source: dict[int, tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select development entries before any RGB or prediction is read."""

    if source_count <= 0 or episodes_per_source <= 0:
        raise ValueError("panel source and episode counts must be positive")
    if max_window_attempts_per_episode <= 0:
        raise ValueError("panel window-attempt count must be positive")
    if allowed_sources is not None:
        allowed_sources = {int(value) for value in allowed_sources}
        if len(allowed_sources) != source_count:
            raise ValueError("allowed source count must equal the panel source count")
        overlap = allowed_sources & excluded_sources
        if overlap:
            raise ValueError(
                "allowed sources overlap the exclusion contract: "
                f"{sorted(overlap)}"
            )
    if required_views is not None:
        required_views = tuple(map(str, required_views))
        if len(required_views) not in {1, 2} or len(set(required_views)) != len(
            required_views
        ):
            raise ValueError("required views must name one or two distinct view keys")
    if required_views_by_source is not None:
        required_views_by_source = {
            int(source): tuple(map(str, views))
            for source, views in required_views_by_source.items()
        }
        if allowed_sources is None or set(required_views_by_source) != allowed_sources:
            raise ValueError("source-view map must exactly cover allowed sources")
        if required_views is not None:
            raise ValueError("global views and source-view map are mutually exclusive")
        if any(
            len(views) != 2 or len(set(views)) != 2
            for views in required_views_by_source.values()
        ):
            raise ValueError("each source-view map row needs external and wrist views")
    if required_view_role not in {
        None,
        "column",
        "external_wrist_by_source",
    }:
        raise ValueError("required view role is unsupported")
    if required_view_role is not None and (
        required_view_role == "column"
        and (required_views is None or len(required_views) != 1)
    ):
        raise ValueError("a required view role needs exactly one frozen view")
    if (
        required_view_role == "external_wrist_by_source"
        and required_views_by_source is None
    ):
        raise ValueError("external/wrist role binding needs a source-view map")
    grouped = _candidate_windows(
        index,
        excluded_sources=excluded_sources,
        seed=seed,
        allowed_sources=allowed_sources,
        required_views=required_views,
        required_view_role=required_view_role,
        required_views_by_source=required_views_by_source,
    )
    source_order = sorted(
        grouped,
        key=lambda source_index: _rank(
            seed,
            "query-displacement-source",
            source_index,
            index.episodes[next(iter(grouped[source_index]))]["manifest_row"].get(
                "source_group"
            ),
        ),
    )
    selected_sources: list[int] = []
    entries: list[dict[str, Any]] = []
    considered_windows = 0
    accepted_windows = 0
    for source_index in source_order:
        episodes = grouped[source_index]
        episode_order = sorted(
            episodes,
            key=lambda episode_index: _rank(
                seed,
                "query-displacement-episode",
                source_index,
                index.episodes[episode_index]["canonical_episode_id"],
            ),
        )
        accepted: dict[int, dict[str, Any]] = {}
        for attempt in range(max_window_attempts_per_episode):
            candidates = [
                (episode_index, episodes[episode_index][attempt])
                for episode_index in episode_order
                if episode_index not in accepted and len(episodes[episode_index]) > attempt
            ]
            if not candidates:
                break
            considered_windows += len(candidates)
            event = causal_displacement_events_for_windows(
                index,
                np.asarray([window for _, window in candidates], dtype=np.int64),
                kinematics=kinematics,
                segment_mm=DISPLACEMENT_SEGMENT_MM,
            )
            accepted_windows += len(event["window_index"])
            for event_row, input_row in enumerate(
                event["input_occurrence_index"].tolist()
            ):
                episode_index, window = candidates[int(input_row)]
                if episode_index in accepted:
                    continue
                accepted[episode_index] = {
                    name: value[event_row]
                    for name, value in event.items()
                    if name != "input_occurrence_index"
                }
            if len(accepted) >= episodes_per_source:
                break
        accepted_order = [episode for episode in episode_order if episode in accepted]
        if len(accepted_order) < episodes_per_source:
            continue
        source_ordinal = len(selected_sources)
        selected_sources.append(source_index)
        for episode_ordinal, episode_index in enumerate(
            accepted_order[:episodes_per_source]
        ):
            event = accepted[episode_index]
            window = int(event["window_index"])
            acquisition = int(event["acquisition_endpoint_index"])
            prediction = int(event["prediction_endpoint_index"])
            local = np.asarray(
                index.arrays["window_image_indices"][window], dtype=np.int64
            )
            record = index.episodes[episode_index]
            row = record["manifest_row"]
            view_seed = _stable_seed(
                seed, "query-displacement-view", source_index, episode_index, window
            )
            source_views = (
                None
                if required_views_by_source is None
                else required_views_by_source[source_index]
            )
            frozen_views = source_views if source_views is not None else required_views
            views = (
                frozen_views
                if frozen_views is not None
                else select_roleless_views(
                    record, maximum_views=2, selection_seed=view_seed
                )
            )
            expected_views = 2 if frozen_views is None else len(frozen_views)
            if len(views) != expected_views:
                raise AssertionError("selected development entry view count drifted")
            identity = {
                "source_index": source_index,
                "episode_index": episode_index,
                "window_index": window,
                "acquisition_endpoint_index": acquisition,
                "prediction_endpoint_index": prediction,
                "views": list(views),
            }
            entries.append(
                {
                    "entry_id": sha256_json(identity)[:20],
                    "source_ordinal": source_ordinal,
                    "episode_ordinal": episode_ordinal,
                    "source_index": source_index,
                    "source_group": str(row["source_group"]),
                    "task_index": int(index.arrays["window_task_index"][window]),
                    "episode_index": episode_index,
                    "canonical_episode_id": str(record["canonical_episode_id"]),
                    "window_index": window,
                    "view_selection_seed": view_seed,
                    "views": list(views),
                    "view_roles": (
                        ["column"]
                        if required_view_role == "column"
                        else ["column", "wrist"]
                        if required_view_role == "external_wrist_by_source"
                        else [f"view{index}" for index in range(len(views))]
                    ),
                    "window_image_offsets": local.tolist(),
                    "query_context_offsets": local[: acquisition + 1].tolist(),
                    "transition_context_offsets": local[acquisition : prediction + 1].tolist(),
                    "query_offset": int(local[acquisition]),
                    "target_offset": int(local[prediction]),
                    "acquisition_endpoint_index": acquisition,
                    "prediction_endpoint_index": prediction,
                    "prediction_interval_frames": prediction - acquisition,
                    "action_progress_indices": np.asarray(
                        event["action_progress_index"], dtype=np.int64
                    ).tolist(),
                    "command_tip_endpoint_delta_mm": (
                        np.asarray(event["command_motion_endpoint"], dtype=np.float32)[
                            62:64
                        ]
                        * 1000.0
                    ).tolist(),
                    "v12_4_exposure": str(row.get("v12_4_exposure", "unknown")),
                }
            )
        if len(selected_sources) == source_count:
            break
    if len(selected_sources) != source_count:
        raise RuntimeError(
            f"only {len(selected_sources)} unexposed development sources provide "
            f"{episodes_per_source} distinct two-event episodes"
        )
    if len(entries) != source_count * episodes_per_source:
        raise AssertionError("development panel entry count drifted")
    episode_pairs = {(row["source_index"], row["episode_index"]) for row in entries}
    if len(episode_pairs) != len(entries):
        raise AssertionError("development panel reused an episode within a source")
    interval = np.asarray([row["prediction_interval_frames"] for row in entries])
    return entries, {
        "candidate_source_count": len(grouped),
        "selected_source_indices": selected_sources,
        "allowed_source_indices": (
            None if allowed_sources is None else sorted(allowed_sources)
        ),
        "required_views": None if required_views is None else list(required_views),
        "required_view_role": required_view_role,
        "required_views_by_source": (
            None
            if required_views_by_source is None
            else {
                str(source): list(required_views_by_source[source])
                for source in sorted(required_views_by_source)
            }
        ),
        "considered_windows": considered_windows,
        "accepted_event_windows": accepted_windows,
        "prediction_interval_frame_quantiles": np.quantile(
            interval, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        ).tolist(),
    }


__all__ = [
    "QUERY_DISPLACEMENT_PANEL_FORMAT",
    "QUERY_DISPLACEMENT_PANEL_SEED",
    "QUERY_DISPLACEMENT_PROTOCOL_ID",
    "excluded_contract_sources",
    "select_query_initialized_displacement_entries",
]
