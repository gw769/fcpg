"""Permutation-exact visual tower for the historical FCPG-H PartGraph.

The CAD tower, parameterization and terminal cosine association are unchanged.
Only candidate-set statistics are replaced by order-independent value
statistics, and candidate encoding is kept in fp32 under an outer autocast.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fcpg.part_graph_track_association import (
    CANDIDATE_SOURCE_COUNT,
    DEFAULT_IMAGE_SIZE,
    CandidateTubeEncoder,
    PartGraphTrackAssociation,
    _validate_candidate_inputs,
    assert_part_graph_parameter_budget,
)


FCPGH_V4_5_METHOD_REVISION = (
    "fcpg-h-permutation-exact-multipath-part-graph-v4.5"
)
FCPGH_V4_5_PROTOCOL_ID = FCPGH_V4_5_METHOD_REVISION
FCPGH_V4_5_DATASET_FORMAT = "so101-fcpg-h-multipath-part-graph-dataset-v1"
FCPGH_V4_5_CHECKPOINT_FORMAT = "so101-fcpg-h-multipath-part-graph-checkpoint-v1"
FCPGH_V4_5_REPORT_FORMAT = "so101-fcpg-h-multipath-part-graph-report-v1"
FCPGH_V4_5_GATE_FORMAT = "so101-fcpg-h-multipath-part-graph-gate-v1"
FCPGH_V4_5_RECEIPT_FORMAT = "so101-fcpg-h-multipath-part-graph-receipt-v1"
FROZEN_V4_2_FCPGH_SHA256 = (
    "e3ea0ac53bcaa63e01cb9644b5c66889efb121f177add29d11bb6321a30ee67b"
)


def masked_lower_median_dim1(
    values: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return an order-independent lower median over candidate dimension 1.

    ``values`` is ``[B,M,...,D]`` and ``valid`` is ``[B,M,...]``.  Invalid
    values sort to the end.  A position with no valid candidates returns zero.
    Only median values are used; sort indices never enter the model.
    """

    if values.ndim < 3 or valid.shape != values.shape[:-1]:
        raise ValueError("values/valid must be [B,M,...,D]/[B,M,...]")
    if values.shape[1] < 1 or valid.dtype != torch.bool:
        raise ValueError("candidate dimension must be nonempty and valid boolean")
    if not values.dtype.is_floating_point or not bool(torch.isfinite(values).all()):
        raise ValueError("median values must be finite floating point")
    promoted = values.float()
    masked = promoted.masked_fill(~valid[..., None], torch.inf)
    ordered = masked.sort(dim=1).values
    count = valid.sum(dim=1)
    lower = ((count - 1).clamp_min(0) // 2).long()
    gather_index = lower[:, None, ..., None].expand(
        lower.shape[0],
        1,
        *lower.shape[1:],
        promoted.shape[-1],
    )
    median = ordered.gather(1, gather_index).squeeze(1)
    return torch.where(count[..., None].gt(0), median, torch.zeros_like(median))


class PermutationExactCandidateTubeEncoder(CandidateTubeEncoder):
    """State-dict-compatible candidate tower with exact set-value statistics."""

    def forward(
        self,
        candidate_tracks_px: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_source: torch.Tensor,
        *,
        temporal_enabled: bool,
    ) -> torch.Tensor:
        batch, candidates, time = _validate_candidate_inputs(
            candidate_tracks_px, candidate_valid, candidate_source
        )
        device_type = candidate_tracks_px.device.type
        if device_type not in {"cpu", "cuda"}:
            raise ValueError("v4.5 candidate encoding supports CPU/CUDA only")

        # The historical order drift happened inside bf16 candidate reductions.
        # The tower is small, so keep its entire candidate path in fp32.
        with torch.autocast(device_type=device_type, enabled=False):
            tracks = candidate_tracks_px.float()
            valid = candidate_valid
            if not temporal_enabled:
                tracks = tracks[..., -1:, :].expand(-1, -1, time, -1)
                valid = valid[..., -1:].expand(-1, -1, time)

            coordinate = 2.0 * tracks / float(self.image_size - 1) - 1.0
            delta = torch.zeros_like(coordinate)
            delta[:, :, 1:] = coordinate[:, :, 1:] - coordinate[:, :, :-1]
            pair_valid = valid.clone()
            pair_valid[:, :, 1:] &= valid[:, :, :-1]
            common_delta = masked_lower_median_dim1(delta, pair_valid)
            stabilized_delta = delta - common_delta[:, None]
            acceleration = torch.zeros_like(stabilized_delta)
            acceleration[:, :, 1:] = (
                stabilized_delta[:, :, 1:] - stabilized_delta[:, :, :-1]
            )
            source = F.one_hot(
                candidate_source, num_classes=CANDIDATE_SOURCE_COUNT
            ).float()
            source = source[:, :, None].expand(-1, -1, time, -1)
            features = torch.cat(
                (
                    coordinate,
                    stabilized_delta,
                    acceleration,
                    valid[..., None].float(),
                    source,
                ),
                dim=-1,
            )
            if features.shape[-1] != self.feature_dim:
                raise AssertionError("candidate feature layout drifted")
            encoded = self.input_projection(features)
            encoded = encoded * valid[..., None].to(dtype=encoded.dtype)
            rows = encoded.reshape(batch * candidates, time, self.hidden_dim)
            _, final = self.temporal(rows)
            tube = final[0].reshape(batch, candidates, self.hidden_dim)

            tube_valid = valid.any(dim=-1)
            set_median = masked_lower_median_dim1(tube, tube_valid)
            set_median = set_median[:, None].expand_as(tube)
            masked_tube = tube.masked_fill(~tube_valid[..., None], -torch.inf)
            set_max = masked_tube.amax(dim=1)
            set_max = torch.where(
                tube_valid.any(dim=1, keepdim=False)[:, None],
                set_max,
                torch.zeros_like(set_max),
            )
            set_max = set_max[:, None].expand_as(tube)
            return self.set_projection(torch.cat((tube, set_median, set_max), dim=-1))


class PermutationExactPartGraphTrackAssociation(PartGraphTrackAssociation):
    """Historical PartGraph with a state-dict-compatible exact visual tower."""

    def __init__(
        self,
        hidden_dim: int = 64,
        *,
        image_size: int = DEFAULT_IMAGE_SIZE,
        temperature: float = 0.1,
        temporal_enabled: bool = True,
    ) -> None:
        super().__init__(
            hidden_dim,
            image_size=image_size,
            temperature=temperature,
            temporal_enabled=temporal_enabled,
        )
        self.candidate_tower = PermutationExactCandidateTubeEncoder(
            hidden_dim,
            image_size=image_size,
        )
        assert_part_graph_parameter_budget(self)


__all__ = [
    "FCPGH_V4_5_CHECKPOINT_FORMAT",
    "FCPGH_V4_5_DATASET_FORMAT",
    "FCPGH_V4_5_GATE_FORMAT",
    "FCPGH_V4_5_METHOD_REVISION",
    "FCPGH_V4_5_PROTOCOL_ID",
    "FCPGH_V4_5_RECEIPT_FORMAT",
    "FCPGH_V4_5_REPORT_FORMAT",
    "FROZEN_V4_2_FCPGH_SHA256",
    "PermutationExactCandidateTubeEncoder",
    "PermutationExactPartGraphTrackAssociation",
    "masked_lower_median_dim1",
]
