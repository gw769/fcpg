"""Zero-parameter query-family-constrained decoder for frozen FCPG-H v4.2.

The frozen PartGraph score matrix is unchanged.  Local proposals carry the
identity of the supplied query that spawned them; the exact six-part decoder
may rank within that observed family but may not move a fingertip to unrelated
global support.  Owner is proposal provenance, not a geometric estimate.  No
target, state, CAD value, camera parameter, or coordinate regression enters it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fcpg.fcpg_h_candidate_first import PART_COUNT, TIP_PART_INDICES
from fcpg.part_graph_track_association import (
    LOCAL_PERSISTENCE_SOURCE,
    PartGraphAssociationOutput,
    gather_candidate_tracks,
    maximum_weight_unique_assignment_with_forced_and_allowed,
)


FCPGH_V4_3_METHOD_REVISION = (
    "fcpg-h-query-family-constrained-part-graph-v4.3"
)
FCPGH_V4_3_PROTOCOL_ID = FCPGH_V4_3_METHOD_REVISION
FCPGH_V4_3_DATASET_FORMAT = (
    "so101-fcpg-h-query-family-confirmation-dataset-v1"
)
FCPGH_V4_3_REPORT_FORMAT = (
    "so101-fcpg-h-query-family-confirmation-report-v1"
)
FCPGH_V4_3_GATE_FORMAT = (
    "so101-fcpg-h-query-family-confirmation-gate-v1"
)
FCPGH_V4_3_RECEIPT_FORMAT = (
    "so101-fcpg-h-query-family-confirmation-receipt-v1"
)
FROZEN_V4_2_FCPGH_SHA256 = (
    "e3ea0ac53bcaa63e01cb9644b5c66889efb121f177add29d11bb6321a30ee67b"
)
FROZEN_V4_2_H0_SHA256 = (
    "7043d975c3b33dd743cf0874f5889cf1e79f699d02c312d58645f5e786cd137d"
)
FROZEN_V4_2_GIT_HEAD = "8561b79e8ccd46efc7757f3e257dac2d7c257ea4"


@dataclass(frozen=True)
class QueryFamilyConstrainedOutput:
    """Observed-family assignment and exact current candidate gathers."""

    logits: torch.Tensor
    candidate_query_owner: torch.Tensor
    allowed_candidate: torch.Tensor
    selected_candidate_index: torch.Tensor
    forced_candidate_index: torch.Tensor
    protected_primary: torch.Tensor
    selected_visible: torch.Tensor
    selected_part_tracks_px: torch.Tensor
    tip_px: torch.Tensor
    center_px: torch.Tensor


def validate_candidate_query_owner(
    candidate_query_owner: torch.Tensor,
    candidate_source: torch.Tensor,
) -> torch.Tensor:
    """Validate immutable owner emitted when each proposal is created."""

    if candidate_query_owner.ndim != 2:
        raise ValueError("candidate_query_owner must be [B,M]")
    batch, candidates = candidate_query_owner.shape
    if candidate_query_owner.dtype != torch.long:
        raise ValueError("candidate_query_owner must be int64")
    if bool(((candidate_query_owner < -1) | (candidate_query_owner > 1)).any()):
        raise ValueError("query owner must lie in {-1,0,1}")
    if candidate_source.shape != (batch, candidates):
        raise ValueError("candidate_source must be [B,M]")
    if candidate_source.dtype != torch.long:
        raise ValueError("candidate_source must be int64")
    local = candidate_source.eq(LOCAL_PERSISTENCE_SOURCE)
    if not bool(candidate_query_owner.ge(0).eq(local).all()):
        raise ValueError("query owner must exist exactly for local proposals")
    return candidate_query_owner


def build_query_family_support(
    candidate_query_owner: torch.Tensor,
    endpoint_valid: torch.Tensor,
) -> torch.Tensor:
    """Return the frozen six-row feasibility matrix."""

    if candidate_query_owner.ndim != 2:
        raise ValueError("candidate_query_owner must be [B,M]")
    batch, candidates = candidate_query_owner.shape
    if candidate_query_owner.dtype != torch.long:
        raise ValueError("candidate_query_owner must be int64")
    if bool(((candidate_query_owner < -1) | (candidate_query_owner > 1)).any()):
        raise ValueError("query owner must lie in {-1,0,1}")
    if endpoint_valid.shape != (batch, candidates) or endpoint_valid.dtype != torch.bool:
        raise ValueError("endpoint_valid must be bool [B,M]")
    allowed = endpoint_valid[:, None].expand(-1, PART_COUNT, -1).clone()
    for part in range(PART_COUNT):
        owner = TIP_PART_INDICES.index(part) if part in TIP_PART_INDICES else -1
        allowed[:, part] &= candidate_query_owner.eq(owner)
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("a CAD part has no endpoint-valid query-family support")
    return allowed


def query_family_constrained_six_part_decode(
    association: PartGraphAssociationOutput,
    candidate_tracks_px: torch.Tensor,
    candidate_valid: torch.Tensor,
    candidate_source: torch.Tensor,
    candidate_query_owner: torch.Tensor,
    primary_candidate_index: torch.Tensor,
) -> QueryFamilyConstrainedOutput:
    """Protect valid primaries and solve the complete family-constrained graph."""

    if candidate_tracks_px.ndim != 4 or candidate_tracks_px.shape[-1] != 2:
        raise ValueError("candidate_tracks_px must be [B,M,T,2]")
    batch, candidates, time, _ = candidate_tracks_px.shape
    if association.logits.shape != (batch, PART_COUNT, candidates):
        raise ValueError("association logits and candidate bank do not align")
    if candidate_valid.shape != (batch, candidates, time):
        raise ValueError("candidate_valid must be [B,M,T]")
    if candidate_valid.dtype != torch.bool:
        raise ValueError("candidate_valid must be boolean")
    if primary_candidate_index.shape != (batch, len(TIP_PART_INDICES)):
        raise ValueError("primary_candidate_index must be [B,2]")
    if primary_candidate_index.dtype != torch.long:
        raise ValueError("primary_candidate_index must be int64")
    if bool(((primary_candidate_index < 0) | (primary_candidate_index >= candidates)).any()):
        raise ValueError("primary candidate index is outside the bank")
    if bool(primary_candidate_index[:, 0].eq(primary_candidate_index[:, 1]).any()):
        raise ValueError("fixed and moving queries require distinct primaries")

    owner = validate_candidate_query_owner(
        candidate_query_owner,
        candidate_source,
    )
    endpoint_valid = candidate_valid[:, :, -1]
    allowed = build_query_family_support(owner, endpoint_valid)
    protected = endpoint_valid.gather(1, primary_candidate_index)
    expected_primary_owner = owner.gather(1, primary_candidate_index)
    query_identity = torch.arange(
        len(TIP_PART_INDICES), device=owner.device, dtype=torch.long
    )[None]
    expected_protected_owner = query_identity.expand_as(protected)[protected]
    if not bool(expected_primary_owner[protected].eq(expected_protected_owner).all()):
        raise ValueError("a valid primary is outside its supplied query family")
    forced = torch.full(
        (batch, PART_COUNT),
        -1,
        dtype=torch.long,
        device=association.logits.device,
    )
    for query_row, part in enumerate(TIP_PART_INDICES):
        forced[:, part] = torch.where(
            protected[:, query_row],
            primary_candidate_index[:, query_row],
            torch.full_like(primary_candidate_index[:, query_row], -1),
        )
    selected = maximum_weight_unique_assignment_with_forced_and_allowed(
        association.logits,
        endpoint_valid,
        allowed,
        forced,
    )
    selected_tracks = gather_candidate_tracks(candidate_tracks_px, selected)
    tip_px = selected_tracks[:, list(TIP_PART_INDICES), -1]
    selected_visible = endpoint_valid.gather(1, selected)
    if not bool(allowed.gather(2, selected[:, :, None]).all()):
        raise AssertionError("query-family decoder escaped allowed support")
    for query_row, part in enumerate(TIP_PART_INDICES):
        if not bool(
            selected[:, part][protected[:, query_row]].eq(
                primary_candidate_index[:, query_row][protected[:, query_row]]
            ).all()
        ):
            raise AssertionError("query-family decoder displaced a valid primary")
    return QueryFamilyConstrainedOutput(
        logits=association.logits,
        candidate_query_owner=owner,
        allowed_candidate=allowed,
        selected_candidate_index=selected,
        forced_candidate_index=forced,
        protected_primary=protected,
        selected_visible=selected_visible,
        selected_part_tracks_px=selected_tracks,
        tip_px=tip_px,
        center_px=tip_px.mean(dim=1),
    )


__all__ = [
    "FCPGH_V4_3_DATASET_FORMAT",
    "FCPGH_V4_3_GATE_FORMAT",
    "FCPGH_V4_3_METHOD_REVISION",
    "FCPGH_V4_3_PROTOCOL_ID",
    "FCPGH_V4_3_RECEIPT_FORMAT",
    "FCPGH_V4_3_REPORT_FORMAT",
    "FROZEN_V4_2_FCPGH_SHA256",
    "FROZEN_V4_2_GIT_HEAD",
    "FROZEN_V4_2_H0_SHA256",
    "QueryFamilyConstrainedOutput",
    "build_query_family_support",
    "query_family_constrained_six_part_decode",
    "validate_candidate_query_owner",
]
