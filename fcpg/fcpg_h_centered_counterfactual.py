"""Centered-evidence and paired-counterfactual NCE for FCPG-H V7.

V6 made part identities representable, but a nearly uniform spatial similarity
field could still score well and avoid the collision energy.  V7 removes the
per-part spatial mean from the existing joint score, so only localized evidence
above the scene-wide offset is rewarded.  It also evaluates each factual pair
directly against its own wrong-history and wrong-part queries inside the same
sensorimotor InfoNCE family; those hard negatives are no longer diluted among
all cross-batch columns.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from fcpg.fcpg_h import FCPGH_NCE_TEMPERATURE
from fcpg.fcpg_h_candidate_first import (
    PART_COUNT,
    CandidateFirstLocalization,
    CandidateFirstOutput,
)
from fcpg.fcpg_h_identity_factorized import (
    FCPGH_V6_COLLISION_WEIGHT,
    IdentityFactorizedFCPGH,
    IdentityFactorizedTemporalGrounder,
    _mean_pairwise_overlap,
    _repulsive_probabilities,
)


FCPGH_V7_METHOD_REVISION = "fcpg-h-v7-centered-paired-counterfactual"
FCPGH_V7_CHECKPOINT_FORMAT = "so101-fcpg-h-centered-counterfactual-checkpoint-v1"
FCPGH_V7_RECEIPT_FORMAT = "so101-fcpg-h-v7-training-receipt-v1"


class CenteredCounterfactualTemporalGrounder(IdentityFactorizedTemporalGrounder):
    """The unchanged 211,488-parameter V6 core under the V7 score contract."""


class CenteredCounterfactualFCPGH(IdentityFactorizedFCPGH):
    """Frozen V12 feature extractor plus the unchanged trainable temporal core."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        core: CenteredCounterfactualTemporalGrounder | None = None,
    ) -> None:
        super().__init__(
            checkpoint,
            core=core or CenteredCounterfactualTemporalGrounder(),
        )


def centered_identity_pair_scores(
    image_candidates: torch.Tensor,
    part_queries: torch.Tensor,
    view_valid: torch.Tensor,
    *,
    temperature: float,
    collision_weight: float = FCPGH_V6_COLLISION_WEIGHT,
) -> torch.Tensor:
    """Return localized-evidence `[image,motion]` scores.

    `logmeanexp(logits) - mean(logits)` is zero for a spatially uniform field
    regardless of its absolute cross-modal similarity.  A pair must therefore
    create a localized above-background peak to receive positive evidence.
    """

    if image_candidates.ndim != 6 or image_candidates.shape[-3] != PART_COUNT:
        raise ValueError("V7 candidates must be [B,2,V,M,256,D]")
    if part_queries.ndim != 4 or part_queries.shape[-2] != PART_COUNT:
        raise ValueError("V7 queries must be [B,2,M,D]")
    visual = F.normalize(image_candidates, dim=-1)
    motion = F.normalize(part_queries, dim=-1)
    similarity = torch.einsum("ievmqd,jemd->ijevmq", visual, motion)
    logits = similarity / temperature
    localized_evidence = (
        torch.logsumexp(logits, dim=-1)
        - math.log(image_candidates.shape[-2])
        - logits.mean(dim=-1)
    )
    probability = _repulsive_probabilities(logits)
    collision = _mean_pairwise_overlap(probability)
    per_view = localized_evidence.mean(dim=-1) - collision_weight * collision
    valid = view_valid[:, None, None, :]
    return (per_view * valid).sum(dim=(2, 3)) / valid.expand_as(per_view).sum(
        dim=(2, 3)
    ).clamp_min(1)


def centered_counterfactual_sensorimotor_nce(
    output: CandidateFirstOutput | CandidateFirstLocalization,
    *,
    view_valid: torch.Tensor,
    motion_cluster: torch.Tensor,
    source_index: torch.Tensor,
    task_index: torch.Tensor,
    high_motion: torch.Tensor,
    wrong_part_queries: torch.Tensor | None,
    temperature: float = FCPGH_NCE_TEMPERATURE,
    pair_score_function: object = centered_identity_pair_scores,
) -> torch.Tensor:
    """Global multi-positive and exact paired-counterfactual InfoNCE.

    A score function may return either `[image,motion]` or a factorized
    `[image,motion,factor]` tensor. Factors receive separate contrastive
    denominators before their losses are averaged, so an easy identity cannot
    hide a hard identity. The default singleton path is exactly V7.
    """

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
    if wrong_part_queries is None:
        raise ValueError("V7 requires paired wrong-history and wrong-part queries")
    if wrong_part_queries.ndim == output.part_queries.ndim:
        wrong_part_queries = wrong_part_queries[None]
    if (
        wrong_part_queries.ndim != output.part_queries.ndim + 1
        or wrong_part_queries.shape[1:] != output.part_queries.shape
    ):
        raise ValueError("wrong_part_queries must be [K,B,2,M,D]")

    active = high_motion.bool() & view_valid.any(dim=1)
    factual = pair_score_function(
        output.image_candidates,
        output.part_queries,
        view_valid,
        temperature=temperature,
    )
    if factual.ndim == 2:
        factual = factual[..., None]
    if factual.ndim != 3 or factual.shape[:2] != (batch, batch):
        raise ValueError("pair scores must be [B,B] or [B,B,factors]")
    factors = factual.shape[-1]
    diagonal = torch.eye(batch, device=factual.device, dtype=torch.bool)
    cluster_positive = (
        (motion_cluster[:, None] == motion_cluster[None])
        & (
            (source_index[:, None] != source_index[None])
            | (task_index[:, None] != task_index[None])
        )
    )
    positive = (diagonal | cluster_positive) & active[:, None] & active[None]

    global_logits = factual.masked_fill(~active[None, :, None], -torch.inf)
    global_numerator = torch.logsumexp(
        global_logits.masked_fill(~positive[:, :, None], -torch.inf), dim=1
    )
    global_denominator = torch.logsumexp(global_logits, dim=1)
    forward_valid = (
        active[:, None]
        & positive.any(dim=1)[:, None]
        & torch.isfinite(global_numerator)
    )
    if not bool(forward_valid.any()):
        return factual.sum() * 0.0
    forward = (
        global_denominator[forward_valid] - global_numerator[forward_valid]
    ).mean()

    backward_logits = factual.transpose(0, 1).masked_fill(
        ~active[None, :, None], -torch.inf
    )
    backward_positive = positive.T[:, :, None]
    backward_numerator = torch.logsumexp(
        backward_logits.masked_fill(~backward_positive, -torch.inf), dim=1
    )
    backward_denominator = torch.logsumexp(backward_logits, dim=1)
    backward_valid = (
        active[:, None]
        & positive.T.any(dim=1)[:, None]
        & torch.isfinite(backward_numerator)
    )
    backward = (
        backward_denominator[backward_valid] - backward_numerator[backward_valid]
    ).mean()

    negative_sets = wrong_part_queries.shape[0]
    flat_wrong = wrong_part_queries.reshape(
        negative_sets * batch, *output.part_queries.shape[1:]
    )
    wrong = pair_score_function(
        output.image_candidates,
        flat_wrong,
        view_valid,
        temperature=temperature,
    )
    if wrong.ndim == 2:
        wrong = wrong[..., None]
    if wrong.shape != (batch, negative_sets * batch, factors):
        raise ValueError("wrong pair-score factors do not match factual scores")
    wrong = wrong.reshape(batch, negative_sets, batch, factors)
    row = torch.arange(batch, device=factual.device)
    paired_wrong = wrong[row, :, row]
    factual_diagonal = factual[row, row]
    conditional_logits = torch.cat(
        (factual_diagonal[:, None, :], paired_wrong), dim=1
    )
    conditional = (
        torch.logsumexp(conditional_logits[active], dim=1)
        - factual_diagonal[active]
    ).mean()

    return torch.stack((forward, backward, conditional)).mean()
