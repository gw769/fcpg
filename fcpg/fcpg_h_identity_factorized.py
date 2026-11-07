"""Identity-factorized joint matching for FCPG-H V6.

V5 produced one generic visual descriptor space for all CAD parts.  Its
whole-pair score could therefore permute the six part queries without changing
the objective.  V6 keeps the same two isolated towers and the same two loss
families, but factorizes the shared 96-D image-only descriptor into six fixed
16-D CAD identity subspaces.  Each motion row can match only the corresponding
visual subspace.  A learned-free overlap term lives inside the joint NCE score
and discourages multiple parts from explaining the same candidate.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from fcpg.fcpg_h import FCPGH_NCE_TEMPERATURE, heatmap_soft_argmax_px
from fcpg.fcpg_h_candidate_first import (
    PART_COUNT,
    TIP_PART_INDICES,
    CandidateFirstFCPGH,
    CandidateFirstLocalization,
    CandidateFirstOutput,
    CandidateFirstTemporalGrounder,
)


FCPGH_V6_METHOD_REVISION = "fcpg-h-v6-identity-factorized-joint-matching"
FCPGH_V6_CHECKPOINT_FORMAT = "so101-fcpg-h-identity-factorized-checkpoint-v1"
FCPGH_V6_RECEIPT_FORMAT = "so101-fcpg-h-v6-training-receipt-v1"
FCPGH_V6_COLLISION_WEIGHT = 1.0


def _repulsive_probabilities(
    logits: torch.Tensor,
    *,
    iterations: int = 3,
) -> torch.Tensor:
    """Return row-normalized part assignments with learned-free repulsion."""

    if logits.ndim < 2 or logits.shape[-2] != PART_COUNT:
        raise ValueError("joint logits must end in [six_parts,pixels]")
    probability = logits.softmax(dim=-1)
    for _ in range(iterations):
        other_occupancy = probability.sum(dim=-2, keepdim=True) - probability
        probability = (logits - other_occupancy).softmax(dim=-1)
    return probability


def _mean_pairwise_overlap(probability: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal spatial overlap for the six part distributions."""

    summed = probability.sum(dim=-2)
    squared_sum = summed.square().sum(dim=-1)
    self_overlap = probability.square().sum(dim=(-2, -1))
    return (squared_sum - self_overlap) / (PART_COUNT * (PART_COUNT - 1))


class IdentityFactorizedTemporalGrounder(CandidateFirstTemporalGrounder):
    """Candidate-first grounder with fixed CAD-identity feature subspaces."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.model_dim % self.parts:
            raise ValueError("V6 model dimension must split evenly across CAD parts")
        self.part_dim = self.model_dim // self.parts

    def visual_candidates(self, local_patch_maps: torch.Tensor) -> torch.Tensor:
        """Return image-only candidates `[B,2,V,M,256,D/M]`."""

        shared = super().visual_candidates(local_patch_maps)
        batch, endpoints, views, pixels, _ = shared.shape
        return shared.reshape(
            batch, endpoints, views, pixels, self.parts, self.part_dim
        ).permute(0, 1, 2, 4, 3, 5)

    def part_motion_queries(self, part_motion: torch.Tensor) -> torch.Tensor:
        """Return the identity row's fixed feature block `[B,2,M,D/M]`."""

        shared = super().part_motion_queries(part_motion)
        batch, endpoints = shared.shape[:2]
        blocks = shared.reshape(
            batch, endpoints, self.parts, self.parts, self.part_dim
        )
        part = torch.arange(self.parts, device=shared.device)
        return blocks[:, :, part, part]

    def localize(
        self,
        image_candidates: torch.Tensor,
        part_queries: torch.Tensor,
        *,
        view_valid: torch.Tensor,
        image_size: tuple[int, int] = (224, 224),
    ) -> CandidateFirstLocalization:
        batch, endpoints, views, parts, pixels, dimension = image_candidates.shape
        if (endpoints, parts, pixels, dimension) != (
            2,
            self.parts,
            256,
            self.part_dim,
        ):
            raise ValueError("V6 image candidate bank violates the fixed contract")
        if part_queries.shape != (batch, 2, self.parts, self.part_dim):
            raise ValueError("V6 part query shape mismatch")
        if view_valid.shape != (batch, views) or view_valid.dtype != torch.bool:
            raise ValueError("view_valid must be bool [B,V]")
        visual = F.normalize(image_candidates, dim=-1)
        motion = F.normalize(part_queries, dim=-1)
        logits = torch.einsum("bevmqd,bemd->bevmq", visual, motion)
        logits = logits / self.temperature
        probability = _repulsive_probabilities(logits)
        probability = probability * view_valid[:, None, :, None, None]
        part_heatmaps = probability.reshape(
            batch, 2, views, self.parts, 16, 16
        )
        tip_heatmaps = part_heatmaps[:, :, :, list(TIP_PART_INDICES)]
        tip_px = heatmap_soft_argmax_px(tip_heatmaps, image_size=image_size)
        tip_px = tip_px * view_valid[:, None, :, None, None]
        return CandidateFirstLocalization(
            part_heatmaps=part_heatmaps,
            tip_heatmaps=tip_heatmaps,
            tip_px=tip_px,
            part_logits=logits.reshape(batch, 2, views, self.parts, 16, 16),
            image_candidates=image_candidates,
            part_queries=part_queries,
        )


class IdentityFactorizedFCPGH(CandidateFirstFCPGH):
    """Frozen V12 feature extractor plus the trainable V6 core."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        core: IdentityFactorizedTemporalGrounder | None = None,
    ) -> None:
        super().__init__(
            checkpoint,
            core=core or IdentityFactorizedTemporalGrounder(),
        )


def identity_factorized_pair_scores(
    image_candidates: torch.Tensor,
    part_queries: torch.Tensor,
    view_valid: torch.Tensor,
    *,
    temperature: float,
    collision_weight: float = FCPGH_V6_COLLISION_WEIGHT,
) -> torch.Tensor:
    """Return identity-sensitive collision-aware `[image,motion]` scores."""

    if image_candidates.ndim != 6 or image_candidates.shape[-3] != PART_COUNT:
        raise ValueError("V6 candidates must be [B,2,V,M,256,D]")
    if part_queries.ndim != 4 or part_queries.shape[-2] != PART_COUNT:
        raise ValueError("V6 queries must be [B,2,M,D]")
    visual = F.normalize(image_candidates, dim=-1)
    motion = F.normalize(part_queries, dim=-1)
    similarity = torch.einsum("ievmqd,jemd->ijevmq", visual, motion)
    logits = similarity / temperature
    evidence = torch.logsumexp(logits, dim=-1) - math.log(
        image_candidates.shape[-2]
    )
    probability = _repulsive_probabilities(logits)
    collision = _mean_pairwise_overlap(probability)
    per_view = evidence.mean(dim=-1) - collision_weight * collision
    valid = view_valid[:, None, None, :]
    return (per_view * valid).sum(dim=(2, 3)) / valid.expand_as(per_view).sum(
        dim=(2, 3)
    ).clamp_min(1)


def identity_factorized_sensorimotor_nce(
    output: CandidateFirstOutput | CandidateFirstLocalization,
    *,
    view_valid: torch.Tensor,
    motion_cluster: torch.Tensor,
    source_index: torch.Tensor,
    task_index: torch.Tensor,
    high_motion: torch.Tensor,
    wrong_part_queries: torch.Tensor | None,
    temperature: float = FCPGH_NCE_TEMPERATURE,
) -> torch.Tensor:
    """The same pair-level NCE family using the V6 joint matching score."""

    batch = output.image_candidates.shape[0]
    for name, value in {
        "motion_cluster": motion_cluster,
        "source_index": source_index,
        "task_index": task_index,
        "high_motion": high_motion,
    }.items():
        if value.shape != (batch,):
            raise ValueError(f"{name} must have shape [B]")
    if view_valid.shape != (batch, output.image_candidates.shape[2]):
        raise ValueError("view_valid must have shape [B,V]")
    active = high_motion.bool() & view_valid.any(dim=1)
    factual = identity_factorized_pair_scores(
        output.image_candidates,
        output.part_queries,
        view_valid,
        temperature=temperature,
    )
    columns = factual
    column_valid = active
    if wrong_part_queries is not None:
        if wrong_part_queries.ndim == output.part_queries.ndim:
            wrong_part_queries = wrong_part_queries[None]
        if (
            wrong_part_queries.ndim != output.part_queries.ndim + 1
            or wrong_part_queries.shape[1:] != output.part_queries.shape
        ):
            raise ValueError("wrong_part_queries must be [B,2,M,D] or [K,B,2,M,D]")
        negative_sets = wrong_part_queries.shape[0]
        flat_wrong = wrong_part_queries.reshape(
            negative_sets * batch, *output.part_queries.shape[1:]
        )
        wrong = identity_factorized_pair_scores(
            output.image_candidates,
            flat_wrong,
            view_valid,
            temperature=temperature,
        )
        columns = torch.cat((factual, wrong), dim=1)
        column_valid = torch.cat((active, active.repeat(negative_sets)))

    diagonal = torch.eye(batch, device=columns.device, dtype=torch.bool)
    cluster_positive = (
        (motion_cluster[:, None] == motion_cluster[None])
        & (
            (source_index[:, None] != source_index[None])
            | (task_index[:, None] != task_index[None])
        )
    )
    positive_factual = (diagonal | cluster_positive) & active[:, None] & active[None]
    positive = torch.zeros_like(columns, dtype=torch.bool)
    positive[:, :batch] = positive_factual
    masked = columns.masked_fill(~column_valid[None], -torch.inf)
    numerator = torch.logsumexp(masked.masked_fill(~positive, -torch.inf), dim=1)
    denominator = torch.logsumexp(masked, dim=1)
    row_valid = active & positive.any(dim=1) & torch.isfinite(numerator)
    if not bool(row_valid.any()):
        return columns.sum() * 0.0
    forward = (denominator[row_valid] - numerator[row_valid]).mean()

    backward_masked = factual.T.masked_fill(~active[None], -torch.inf)
    backward_positive = positive_factual.T
    backward_numerator = torch.logsumexp(
        backward_masked.masked_fill(~backward_positive, -torch.inf), dim=1
    )
    backward_denominator = torch.logsumexp(backward_masked, dim=1)
    backward_valid = active & backward_positive.any(dim=1) & torch.isfinite(
        backward_numerator
    )
    backward = (
        backward_denominator[backward_valid] - backward_numerator[backward_valid]
    ).mean()
    return 0.5 * (forward + backward)
