"""Content-canonical exact assignment for permutation-exact FCPG-H v4.6."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fcpg.fcpg_h_candidate_first import PART_COUNT, TIP_PART_INDICES
from fcpg.part_graph_track_association import (
    PartGraphAssociationOutput,
    gather_candidate_tracks,
)
from fcpg.query_family_constrained_part_graph_v4_3 import (
    query_family_constrained_six_part_decode,
    validate_candidate_query_owner,
)


FCPGH_V4_6_METHOD_REVISION = "fcpg-h-canonical-bundle-part-graph-v4.6"
FCPGH_V4_6_PROTOCOL_ID = FCPGH_V4_6_METHOD_REVISION
FCPGH_V4_6_DATASET_FORMAT = "so101-fcpg-h-canonical-bundle-dataset-v1"
FCPGH_V4_6_CHECKPOINT_FORMAT = "so101-fcpg-h-canonical-bundle-checkpoint-v1"
FCPGH_V4_6_REPORT_FORMAT = "so101-fcpg-h-canonical-bundle-report-v1"
FCPGH_V4_6_GATE_FORMAT = "so101-fcpg-h-canonical-bundle-gate-v1"
FCPGH_V4_6_RECEIPT_FORMAT = "so101-fcpg-h-canonical-bundle-receipt-v1"
FROZEN_V4_5_FCPGH_SHA256 = (
    "f9c24add2ef26425327dfb3db7cbc20dc85e720e7285060fb1cd4b43676c5d83"
)
FROZEN_V4_5_H0_SHA256 = (
    "5d2fc1d788e4f95732a4886a729a20e0fd5ba7c0a118e78b1551ab539bf539c3"
)


@dataclass(frozen=True)
class CanonicalQueryFamilyOutput:
    """Exact query-family result mapped back to the caller's candidate order."""

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
    canonical_candidate_order: torch.Tensor


@dataclass(frozen=True)
class _LogitsOnlyAssociation:
    logits: torch.Tensor


def content_canonical_candidate_order(
    candidate_tracks_px: torch.Tensor,
    candidate_valid: torch.Tensor,
    candidate_source: torch.Tensor,
    candidate_query_owner: torch.Tensor,
) -> torch.Tensor:
    """Lexicographically order candidates by immutable observed content.

    Stable sorts are applied from the least to most significant key.  Exact
    duplicate content is rejected because no content-only solver can recover a
    distinct original identity for two observationally identical columns.
    """

    if candidate_tracks_px.ndim != 4 or candidate_tracks_px.shape[-1] != 2:
        raise ValueError("candidate_tracks_px must be [B,M,T,2]")
    batch, candidates, time, _ = candidate_tracks_px.shape
    if candidate_valid.shape != (batch, candidates, time):
        raise ValueError("candidate_valid must be [B,M,T]")
    if candidate_valid.dtype != torch.bool:
        raise ValueError("candidate_valid must be boolean")
    if candidate_source.shape != (batch, candidates):
        raise ValueError("candidate_source must be [B,M]")
    owner = validate_candidate_query_owner(
        candidate_query_owner, candidate_source
    )
    if not bool(torch.isfinite(candidate_tracks_px).all()):
        raise ValueError("candidate tracks contain non-finite values")

    flattened_tracks = candidate_tracks_px.reshape(batch, candidates, -1)
    keys = [candidate_source, owner]
    keys.extend(candidate_valid.to(torch.uint8).unbind(dim=-1))
    keys.extend(flattened_tracks.unbind(dim=-1))
    order = torch.arange(candidates, device=candidate_tracks_px.device)[None]
    order = order.expand(batch, -1).clone()
    for key in reversed(keys):
        presented = key.gather(1, order)
        rank = torch.argsort(presented, dim=1, stable=True)
        order = order.gather(1, rank)

    canonical_tracks = candidate_tracks_px.gather(
        1, order[:, :, None, None].expand(-1, -1, time, 2)
    )
    canonical_valid = candidate_valid.gather(
        1, order[:, :, None].expand(-1, -1, time)
    )
    canonical_source = candidate_source.gather(1, order)
    canonical_owner = owner.gather(1, order)
    if candidates > 1:
        duplicate = (
            canonical_tracks[:, 1:].eq(canonical_tracks[:, :-1]).all(dim=(2, 3))
            & canonical_valid[:, 1:].eq(canonical_valid[:, :-1]).all(dim=2)
            & canonical_source[:, 1:].eq(canonical_source[:, :-1])
            & canonical_owner[:, 1:].eq(canonical_owner[:, :-1])
        )
        if bool(duplicate.any()):
            raise ValueError("content-canonical solve rejects duplicate candidates")
    return order


def _canonical_index_to_original(
    canonical_index: torch.Tensor,
    canonical_order: torch.Tensor,
) -> torch.Tensor:
    result = canonical_index.clone()
    valid = result.ge(0)
    if bool(valid.any()):
        safe = result.clamp_min(0)
        mapped = canonical_order.gather(1, safe)
        result[valid] = mapped[valid]
    return result


def canonical_query_family_six_part_decode(
    association: PartGraphAssociationOutput,
    candidate_tracks_px: torch.Tensor,
    candidate_valid: torch.Tensor,
    candidate_source: torch.Tensor,
    candidate_query_owner: torch.Tensor,
    primary_candidate_index: torch.Tensor,
) -> CanonicalQueryFamilyOutput:
    """Solve in content-canonical order, then return original-bank indices."""

    batch, candidates, time, _ = candidate_tracks_px.shape
    if association.logits.shape != (batch, PART_COUNT, candidates):
        raise ValueError("association logits and candidate bank do not align")
    if primary_candidate_index.shape != (batch, len(TIP_PART_INDICES)):
        raise ValueError("primary_candidate_index must be [B,2]")
    order = content_canonical_candidate_order(
        candidate_tracks_px,
        candidate_valid,
        candidate_source,
        candidate_query_owner,
    )
    inverse = torch.empty_like(order)
    inverse.scatter_(
        1,
        order,
        torch.arange(candidates, device=order.device)[None].expand_as(order),
    )
    canonical_tracks = candidate_tracks_px.gather(
        1, order[:, :, None, None].expand(-1, -1, time, 2)
    )
    canonical_valid = candidate_valid.gather(
        1, order[:, :, None].expand(-1, -1, time)
    )
    canonical_source = candidate_source.gather(1, order)
    canonical_owner = candidate_query_owner.gather(1, order)
    canonical_primary = inverse.gather(1, primary_candidate_index)
    canonical_logits = association.logits.gather(
        2, order[:, None].expand(-1, PART_COUNT, -1)
    )
    solved = query_family_constrained_six_part_decode(
        _LogitsOnlyAssociation(logits=canonical_logits),
        canonical_tracks,
        canonical_valid,
        canonical_source,
        canonical_owner,
        canonical_primary,
    )
    selected = order.gather(1, solved.selected_candidate_index)
    forced = _canonical_index_to_original(
        solved.forced_candidate_index, order
    )
    selected_tracks = gather_candidate_tracks(candidate_tracks_px, selected)
    tip_px = selected_tracks[:, list(TIP_PART_INDICES), -1]
    allowed = solved.allowed_candidate.gather(
        2, inverse[:, None].expand(-1, PART_COUNT, -1)
    )
    endpoint_valid = candidate_valid[:, :, -1]
    return CanonicalQueryFamilyOutput(
        logits=association.logits,
        candidate_query_owner=candidate_query_owner,
        allowed_candidate=allowed,
        selected_candidate_index=selected,
        forced_candidate_index=forced,
        protected_primary=solved.protected_primary,
        selected_visible=endpoint_valid.gather(1, selected),
        selected_part_tracks_px=selected_tracks,
        tip_px=tip_px,
        center_px=tip_px.mean(dim=1),
        canonical_candidate_order=order,
    )


__all__ = [
    "CanonicalQueryFamilyOutput",
    "FCPGH_V4_6_CHECKPOINT_FORMAT",
    "FCPGH_V4_6_DATASET_FORMAT",
    "FCPGH_V4_6_GATE_FORMAT",
    "FCPGH_V4_6_METHOD_REVISION",
    "FCPGH_V4_6_PROTOCOL_ID",
    "FCPGH_V4_6_RECEIPT_FORMAT",
    "FCPGH_V4_6_REPORT_FORMAT",
    "FROZEN_V4_5_FCPGH_SHA256",
    "FROZEN_V4_5_H0_SHA256",
    "canonical_query_family_six_part_decode",
    "content_canonical_candidate_order",
]
