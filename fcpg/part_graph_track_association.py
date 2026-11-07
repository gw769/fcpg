"""Calibration-free CAD part identity assignment over pre-existing track tubes.

This module deliberately has no RGB encoder and no coordinate regressor.  An
image-only tracker must first propose a candidate set.  A camera-free SO-101
motion tower can then assign the six fixed CAD identities to that set, but the
only pixel values returned by the module are values gathered from candidates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fcpg.fcpg_h_candidate_first import PART_COUNT, PART_MOTION_DIM, TIP_PART_INDICES
from fcpg.fcpg_h_unit_normalized import normalize_part_motion_units


PART_GRAPH_ASSOCIATION_METHOD_REVISION = (
    "fcpg-h-dual-source-part-graph-association-v1"
)
PART_GRAPH_ASSOCIATION_CHECKPOINT_FORMAT = (
    "so101-fcpg-h-part-graph-association-checkpoint-v1"
)
PART_GRAPH_ASSOCIATION_RECEIPT_FORMAT = (
    "so101-fcpg-h-part-graph-association-receipt-v1"
)

GLOBAL_ACQUISITION_SOURCE = 0
LOCAL_PERSISTENCE_SOURCE = 1
DISTRACTOR_SOURCE = 2
CANDIDATE_SOURCE_COUNT = 3
DEFAULT_CANDIDATE_COUNT = 32
DEFAULT_IMAGE_SIZE = 224
MAXIMUM_TRAINABLE_PARAMETERS = 500_000


@dataclass(frozen=True)
class PartGraphAssociationOutput:
    """Candidate assignments and candidate-gathered physical part tracks."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    selected_candidate_index: torch.Tensor
    selected_part_tracks_px: torch.Tensor
    tip_px: torch.Tensor
    center_px: torch.Tensor
    candidate_embeddings: torch.Tensor
    part_embeddings: torch.Tensor


@dataclass(frozen=True)
class SyntheticAssociationBatch:
    """One synthetic, camera-hidden association batch."""

    candidate_tracks_px: torch.Tensor
    candidate_valid: torch.Tensor
    candidate_source: torch.Tensor
    part_motion: torch.Tensor
    target_candidate_index: torch.Tensor
    true_part_tracks_px: torch.Tensor
    corruption_id: torch.Tensor
    persistence: torch.Tensor


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def assert_part_graph_parameter_budget(module: nn.Module) -> int:
    count = trainable_parameter_count(module)
    if count > MAXIMUM_TRAINABLE_PARAMETERS:
        raise ValueError(
            f"part-graph association has {count:,} trainable parameters; "
            f"budget is {MAXIMUM_TRAINABLE_PARAMETERS:,}"
        )
    return count


def _validate_candidate_inputs(
    candidate_tracks_px: torch.Tensor,
    candidate_valid: torch.Tensor,
    candidate_source: torch.Tensor,
) -> tuple[int, int, int]:
    if candidate_tracks_px.ndim != 4 or candidate_tracks_px.shape[-1] != 2:
        raise ValueError("candidate_tracks_px must be [B,M,T,2]")
    batch, candidates, time, _ = candidate_tracks_px.shape
    if time < 2 or candidates < PART_COUNT:
        raise ValueError("association requires T>=2 and at least six candidates")
    if candidate_valid.shape != (batch, candidates, time):
        raise ValueError("candidate_valid must be [B,M,T]")
    if candidate_valid.dtype != torch.bool:
        raise ValueError("candidate_valid must be boolean")
    if candidate_source.shape != (batch, candidates):
        raise ValueError("candidate_source must be [B,M]")
    if candidate_source.dtype != torch.long:
        raise ValueError("candidate_source must be int64")
    if bool(((candidate_source < 0) | (candidate_source >= CANDIDATE_SOURCE_COUNT)).any()):
        raise ValueError("candidate source id is outside the frozen three-source contract")
    if not bool(torch.isfinite(candidate_tracks_px).all()):
        raise ValueError("candidate tracks contain non-finite coordinates")
    return batch, candidates, time


def gather_candidate_tracks(
    candidate_tracks_px: torch.Tensor,
    candidate_index: torch.Tensor,
) -> torch.Tensor:
    """Gather tracks without creating or transporting a coordinate."""

    batch, candidates, time, xy = candidate_tracks_px.shape
    if candidate_index.shape != (batch, PART_COUNT):
        raise ValueError("candidate_index must be [B,6]")
    if bool(((candidate_index < 0) | (candidate_index >= candidates)).any()):
        raise ValueError("candidate_index is outside the candidate bank")
    bank = candidate_tracks_px[:, None].expand(-1, PART_COUNT, -1, -1, -1)
    gather_index = candidate_index[:, :, None, None, None].expand(
        -1, -1, 1, time, xy
    )
    return bank.gather(2, gather_index).squeeze(2)


def maximum_weight_unique_assignment(logits: torch.Tensor) -> torch.Tensor:
    """Decode a rectangular part/candidate score as exact one-to-one matching.

    The score network and its CE remain unchanged.  This learned-free decoder
    only states the set contract that each requested physical CAD part consumes
    a different image proposal. SciPy runs on a detached KxM matrix; assignment
    is a measurement selection operation and is never differentiated.
    """

    if logits.ndim != 3 or logits.shape[1] < 1:
        raise ValueError("logits must have shape [B,K,M] with K>=1")
    parts = logits.shape[1]
    if logits.shape[2] < parts:
        raise ValueError("unique assignment requires at least as many candidates as parts")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("assignment logits contain non-finite values")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("exact joint candidate decoding requires SciPy") from exc
    scores = logits.detach().double().cpu().numpy()
    rows = []
    for score in scores:
        part_index, candidate_index = linear_sum_assignment(-score)
        if tuple(part_index.tolist()) != tuple(range(parts)):
            raise AssertionError("linear assignment did not return every requested part")
        rows.append(torch.from_numpy(candidate_index.astype("int64", copy=False)))
    return torch.stack(rows).to(device=logits.device)


def maximum_weight_unique_assignment_with_forced(
    logits: torch.Tensor,
    endpoint_valid: torch.Tensor,
    forced_candidate_index: torch.Tensor,
) -> torch.Tensor:
    """Exact valid-support assignment with optional measured rows forced.

    ``forced_candidate_index`` is ``-1`` for an unforced part and otherwise
    names an endpoint-valid measured candidate.  Forced rows/candidates are
    removed before solving the remaining rectangular assignment.  This is a
    decoder contract, not a differentiable model component.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B,K,M]")
    allowed = torch.ones_like(logits, dtype=torch.bool)
    return maximum_weight_unique_assignment_with_forced_and_allowed(
        logits,
        endpoint_valid,
        allowed,
        forced_candidate_index,
    )


def maximum_weight_unique_assignment_with_forced_and_allowed(
    logits: torch.Tensor,
    endpoint_valid: torch.Tensor,
    allowed_candidate: torch.Tensor,
    forced_candidate_index: torch.Tensor,
) -> torch.Tensor:
    """Exact one-to-one assignment under validity, support, and forced rows.

    ``allowed_candidate[b,k,m]`` is an observed proposal-provenance constraint,
    not a learned score.  Endpoint validity is intersected with that matrix.
    SciPy solves only the remaining feasible rectangle after valid forced rows
    and their candidates have been removed.  Infeasible support fails closed.
    """

    if logits.ndim != 3 or logits.shape[1] < 1:
        raise ValueError("logits must have shape [B,K,M] with K>=1")
    batch, parts, candidates = logits.shape
    if candidates < parts:
        raise ValueError("unique assignment requires M>=K")
    if endpoint_valid.shape != (batch, candidates) or endpoint_valid.dtype != torch.bool:
        raise ValueError("endpoint_valid must be bool [B,M]")
    if allowed_candidate.shape != (batch, parts, candidates):
        raise ValueError("allowed_candidate must be [B,K,M]")
    if allowed_candidate.dtype != torch.bool:
        raise ValueError("allowed_candidate must be boolean")
    if forced_candidate_index.shape != (batch, parts):
        raise ValueError("forced_candidate_index must be [B,K]")
    if forced_candidate_index.dtype != torch.long:
        raise ValueError("forced_candidate_index must be int64")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("assignment logits contain non-finite values")
    if bool(
        ((forced_candidate_index < -1) | (forced_candidate_index >= candidates)).any()
    ):
        raise ValueError("forced candidate index is outside {-1,0,...,M-1}")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("exact joint candidate decoding requires SciPy") from exc

    score_rows = logits.detach().double().cpu().numpy()
    valid_rows = endpoint_valid.detach().cpu().numpy()
    allowed_rows = allowed_candidate.detach().cpu().numpy()
    forced_rows = forced_candidate_index.detach().cpu().numpy()
    decoded: list[torch.Tensor] = []
    for score, valid, allowed, forced in zip(
        score_rows, valid_rows, allowed_rows, forced_rows, strict=True
    ):
        feasible = allowed & valid[None]
        result = torch.full((parts,), -1, dtype=torch.long)
        forced_part = [index for index, candidate in enumerate(forced) if candidate >= 0]
        forced_candidates = [int(forced[index]) for index in forced_part]
        if len(set(forced_candidates)) != len(forced_candidates):
            raise ValueError("two forced parts consume the same measured candidate")
        if any(not bool(valid[candidate]) for candidate in forced_candidates):
            raise ValueError("a forced candidate is not endpoint-valid")
        for part, candidate in zip(forced_part, forced_candidates, strict=True):
            if not bool(feasible[part, candidate]):
                raise ValueError("a forced candidate is outside its allowed support")
            result[part] = candidate

        remaining_part = [index for index in range(parts) if index not in forced_part]
        remaining_candidate = [
            index
            for index in range(candidates)
            if bool(valid[index]) and index not in forced_candidates
        ]
        if len(remaining_candidate) < len(remaining_part):
            raise ValueError("not enough endpoint-valid candidates for exact assignment")
        if remaining_part:
            subscore = score[remaining_part][:, remaining_candidate]
            suballowed = feasible[remaining_part][:, remaining_candidate]
            if not bool(suballowed.any(axis=1).all()):
                raise ValueError("an assignment row has no allowed endpoint-valid candidate")
            cost = -subscore
            cost[~suballowed] = math.inf
            try:
                part_index, candidate_index = linear_sum_assignment(cost)
            except ValueError as exc:
                raise ValueError("allowed support has no complete unique assignment") from exc
            if tuple(part_index.tolist()) != tuple(range(len(remaining_part))):
                raise AssertionError("linear assignment omitted an unforced part")
            for row, column in zip(part_index, candidate_index, strict=True):
                part = remaining_part[int(row)]
                candidate = remaining_candidate[int(column)]
                if not bool(feasible[part, candidate]):
                    raise AssertionError("linear assignment selected forbidden support")
                result[part] = candidate
        if bool(result.lt(0).any()) or len(torch.unique(result)) != parts:
            raise AssertionError("forced exact assignment is incomplete or non-unique")
        decoded.append(result)
    selected = torch.stack(decoded).to(device=logits.device)
    if not bool(endpoint_valid.gather(1, selected).all()):
        raise AssertionError("forced exact assignment selected invalid support")
    selected_allowed = allowed_candidate.gather(2, selected[:, :, None]).squeeze(2)
    if not bool(selected_allowed.all()):
        raise AssertionError("forced exact assignment selected forbidden support")
    return selected


class CandidateTubeEncoder(nn.Module):
    """Image-only causal encoder over a permutation-invariant tube set."""

    feature_dim = 10

    def __init__(self, hidden_dim: int = 64, *, image_size: int = DEFAULT_IMAGE_SIZE) -> None:
        super().__init__()
        if hidden_dim <= 0 or image_size <= 1:
            raise ValueError("hidden_dim and image_size must be positive")
        self.hidden_dim = int(hidden_dim)
        self.image_size = int(image_size)
        self.input_projection = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.set_projection = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

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
        tracks = candidate_tracks_px
        valid = candidate_valid
        if not temporal_enabled:
            tracks = tracks[..., -1:, :].expand(-1, -1, time, -1)
            valid = valid[..., -1:].expand(-1, -1, time)

        coordinate = 2.0 * tracks / float(self.image_size - 1) - 1.0
        delta = torch.zeros_like(coordinate)
        delta[:, :, 1:] = coordinate[:, :, 1:] - coordinate[:, :, :-1]
        pair_valid = valid.clone()
        pair_valid[:, :, 1:] &= valid[:, :, :-1]
        weights = pair_valid.to(dtype=coordinate.dtype)
        common_delta = (delta * weights[..., None]).sum(dim=1) / (
            weights.sum(dim=1, keepdim=False)[..., None].clamp_min(1.0)
        )
        stabilized_delta = delta - common_delta[:, None]
        acceleration = torch.zeros_like(stabilized_delta)
        acceleration[:, :, 1:] = (
            stabilized_delta[:, :, 1:] - stabilized_delta[:, :, :-1]
        )
        source = F.one_hot(
            candidate_source, num_classes=CANDIDATE_SOURCE_COUNT
        ).to(dtype=coordinate.dtype)
        source = source[:, :, None].expand(-1, -1, time, -1)
        features = torch.cat(
            (
                coordinate,
                stabilized_delta,
                acceleration,
                valid[..., None].to(dtype=coordinate.dtype),
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

        set_mean = tube.mean(dim=1, keepdim=True).expand_as(tube)
        set_max = tube.amax(dim=1, keepdim=True).expand_as(tube)
        return self.set_projection(torch.cat((tube, set_mean, set_max), dim=-1))


class CADPartGraphEncoder(nn.Module):
    """Camera-free causal encoder with complete interaction over six CAD nodes."""

    def __init__(self, hidden_dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by attention heads")
        self.hidden_dim = int(hidden_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(PART_MOTION_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.part_identity = nn.Embedding(PART_COUNT, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=2 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.complete_graph = nn.TransformerEncoder(layer, num_layers=1)
        self.output_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _current_only_motion(part_motion: torch.Tensor) -> torch.Tensor:
        current = torch.zeros_like(part_motion[:, -1])
        current[..., 0:6] = part_motion[:, -1, :, 0:6]
        current[..., 24:27] = part_motion[:, -1, :, 24:27]
        current[..., 36:42] = part_motion[:, -1, :, 36:42]
        return current[:, None].expand_as(part_motion)

    def forward(self, part_motion: torch.Tensor, *, temporal_enabled: bool) -> torch.Tensor:
        if part_motion.ndim != 4 or part_motion.shape[2:] != (
            PART_COUNT,
            PART_MOTION_DIM,
        ):
            raise ValueError("part_motion must be [B,T,6,44]")
        if part_motion.shape[1] < 2 or not bool(torch.isfinite(part_motion).all()):
            raise ValueError("part_motion must be finite with T>=2")
        motion = part_motion if temporal_enabled else self._current_only_motion(part_motion)
        motion = normalize_part_motion_units(motion)
        batch, time = motion.shape[:2]
        encoded = self.input_projection(motion)
        rows = encoded.permute(0, 2, 1, 3).reshape(
            batch * PART_COUNT, time, self.hidden_dim
        )
        _, final = self.temporal(rows)
        parts = final[0].reshape(batch, PART_COUNT, self.hidden_dim)
        identity = self.part_identity.weight[None].expand(batch, -1, -1)
        parts = self.complete_graph(parts + identity)
        return self.output_norm(parts)


class PartGraphTrackAssociation(nn.Module):
    """Assign fixed SO-101 CAD identities to image-proposed candidate tubes."""

    def __init__(
        self,
        hidden_dim: int = 64,
        *,
        image_size: int = DEFAULT_IMAGE_SIZE,
        temperature: float = 0.1,
        temporal_enabled: bool = True,
    ) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.candidate_tower = CandidateTubeEncoder(hidden_dim, image_size=image_size)
        self.part_tower = CADPartGraphEncoder(hidden_dim)
        self.temperature = float(temperature)
        self.temporal_enabled = bool(temporal_enabled)
        assert_part_graph_parameter_budget(self)

    def encode_candidates(
        self,
        candidate_tracks_px: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_source: torch.Tensor,
    ) -> torch.Tensor:
        return self.candidate_tower(
            candidate_tracks_px,
            candidate_valid,
            candidate_source,
            temporal_enabled=self.temporal_enabled,
        )

    def encode_parts(self, part_motion: torch.Tensor) -> torch.Tensor:
        return self.part_tower(part_motion, temporal_enabled=self.temporal_enabled)

    def forward(
        self,
        candidate_tracks_px: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_source: torch.Tensor,
        part_motion: torch.Tensor,
    ) -> PartGraphAssociationOutput:
        batch, candidates, time = _validate_candidate_inputs(
            candidate_tracks_px, candidate_valid, candidate_source
        )
        if part_motion.shape[:2] != (batch, time):
            raise ValueError("candidate and CAD histories must align in batch/time only")
        candidate_embeddings = self.encode_candidates(
            candidate_tracks_px, candidate_valid, candidate_source
        )
        part_embeddings = self.encode_parts(part_motion)
        candidate_unit = F.normalize(candidate_embeddings.float(), dim=-1)
        part_unit = F.normalize(part_embeddings.float(), dim=-1)
        logits = torch.einsum("bkd,bmd->bkm", part_unit, candidate_unit)
        logits = logits / self.temperature
        probabilities = logits.softmax(dim=-1)
        selected_index = logits.argmax(dim=-1)
        selected_tracks = gather_candidate_tracks(candidate_tracks_px, selected_index)
        tip_tracks = selected_tracks[:, list(TIP_PART_INDICES)]
        tip_px = tip_tracks[:, :, -1]
        center_px = tip_px.mean(dim=1)
        return PartGraphAssociationOutput(
            logits=logits,
            probabilities=probabilities,
            selected_candidate_index=selected_index,
            selected_part_tracks_px=selected_tracks,
            tip_px=tip_px,
            center_px=center_px,
            candidate_embeddings=candidate_embeddings,
            part_embeddings=part_embeddings,
        )


def part_assignment_cross_entropy(
    output: PartGraphAssociationOutput,
    target_candidate_index: torch.Tensor,
) -> torch.Tensor:
    """The canary's single optimization objective."""

    if target_candidate_index.shape != output.logits.shape[:2]:
        raise ValueError("target_candidate_index must be [B,6]")
    if target_candidate_index.dtype != torch.long:
        raise ValueError("target_candidate_index must be int64")
    candidates = output.logits.shape[-1]
    if bool(((target_candidate_index < 0) | (target_candidate_index >= candidates)).any()):
        raise ValueError("assignment target is outside the candidate bank")
    return F.cross_entropy(
        output.logits.reshape(-1, candidates), target_candidate_index.reshape(-1)
    )


def random_pinhole_project(
    part_xyz: torch.Tensor,
    *,
    generator: torch.Generator,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project CAD trajectories through hidden random pinhole cameras.

    Camera parameters are synthetic nuisance variables returned nowhere.  They
    are used to build a held-out data distribution, never as model inputs.
    """

    if part_xyz.ndim != 4 or part_xyz.shape[2:] != (PART_COUNT, 3):
        raise ValueError("part_xyz must be [B,T,6,3]")
    if part_xyz.device.type != "cpu":
        raise ValueError("synthetic camera generation is CPU-only and deterministic")
    if not bool(torch.isfinite(part_xyz).all()):
        raise ValueError("part_xyz contains non-finite values")
    batch = part_xyz.shape[0]
    dtype = part_xyz.dtype
    target = part_xyz.mean(dim=(1, 2))
    azimuth = 2.0 * math.pi * torch.rand(batch, generator=generator, dtype=dtype)
    elevation = 0.30 + 0.65 * torch.rand(batch, generator=generator, dtype=dtype)
    radius = 0.72 + 0.28 * torch.rand(batch, generator=generator, dtype=dtype)
    camera = target + torch.stack(
        (
            radius * elevation.cos() * azimuth.cos(),
            radius * elevation.cos() * azimuth.sin(),
            radius * elevation.sin(),
        ),
        dim=-1,
    )
    forward = F.normalize(target - camera, dim=-1)
    world_up = torch.tensor((0.0, 0.0, 1.0), dtype=dtype).expand_as(forward)
    right = F.normalize(torch.linalg.cross(forward, world_up, dim=-1), dim=-1)
    up = F.normalize(torch.linalg.cross(right, forward, dim=-1), dim=-1)
    relative = part_xyz - camera[:, None, None]
    x = torch.einsum("btkd,bd->btk", relative, right)
    y = torch.einsum("btkd,bd->btk", relative, up)
    depth = torch.einsum("btkd,bd->btk", relative, forward).clamp_min(0.1)
    focal = 180.0 + 70.0 * torch.rand(batch, generator=generator, dtype=dtype)
    principal = image_size / 2.0 + (
        torch.rand(batch, 2, generator=generator, dtype=dtype) - 0.5
    ) * 16.0
    pixels = torch.stack(
        (
            focal[:, None, None] * x / depth + principal[:, None, None, 0],
            -focal[:, None, None] * y / depth + principal[:, None, None, 1],
        ),
        dim=-1,
    )
    margin = 4.0
    visible = (
        (pixels[..., 0] >= margin)
        & (pixels[..., 0] <= image_size - 1 - margin)
        & (pixels[..., 1] >= margin)
        & (pixels[..., 1] <= image_size - 1 - margin)
        & (depth > 0.1)
    )
    return pixels, visible


def _apply_common_shake(
    points: torch.Tensor,
    corruption_id: torch.Tensor,
    *,
    generator: torch.Generator,
    image_size: int,
) -> torch.Tensor:
    """Apply one causal smooth camera shake to every point in a sample."""

    batch, _, time, _ = points.shape
    increments = torch.randn(batch, time, 2, generator=generator, dtype=points.dtype)
    translation = increments.cumsum(dim=1)
    translation -= translation[:, :1]
    scale = translation.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
    translation = 12.0 * translation / scale
    angle_increment = torch.randn(batch, time, generator=generator, dtype=points.dtype)
    angle = angle_increment.cumsum(dim=1)
    angle -= angle[:, :1]
    angle_scale = angle.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    angle = math.radians(3.0) * angle / angle_scale
    shake = ((corruption_id == 1) | (corruption_id == 3)).to(points.dtype)
    translation = translation * shake[:, None, None]
    angle = angle * shake[:, None]
    center = torch.tensor(
        ((image_size - 1) / 2.0, (image_size - 1) / 2.0), dtype=points.dtype
    )
    centered = points - center
    cosine = angle.cos()[:, None, :, None]
    sine = angle.sin()[:, None, :, None]
    x = cosine * centered[..., 0:1] - sine * centered[..., 1:2]
    y = sine * centered[..., 0:1] + cosine * centered[..., 1:2]
    return torch.cat((x, y), dim=-1) + center + translation[:, None]


def build_synthetic_candidate_bank(
    true_part_tracks_px: torch.Tensor,
    part_motion: torch.Tensor,
    *,
    generator: torch.Generator,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    image_size: int = DEFAULT_IMAGE_SIZE,
    corruption_id: torch.Tensor | None = None,
    persistence: torch.Tensor | None = None,
) -> SyntheticAssociationBatch:
    """Create an order-randomized global/local candidate union.

    The first six unpermuted rows are global part candidates. Rows six and
    seven are the two valid local fingertip candidates only on persistence
    samples. The permutation and target indices are returned for CE training;
    no pixel loss is constructed.
    """

    if true_part_tracks_px.ndim != 4 or true_part_tracks_px.shape[2:] != (
        PART_COUNT,
        2,
    ):
        raise ValueError("true_part_tracks_px must be [B,T,6,2]")
    if true_part_tracks_px.device.type != "cpu" or part_motion.device.type != "cpu":
        raise ValueError("synthetic candidate construction is CPU-only")
    batch, time = true_part_tracks_px.shape[:2]
    if part_motion.shape != (batch, time, PART_COUNT, PART_MOTION_DIM):
        raise ValueError("part_motion must be [B,T,6,44]")
    if candidate_count < 16:
        raise ValueError("candidate_count must be at least 16")
    if corruption_id is None:
        corruption_id = torch.randint(0, 4, (batch,), generator=generator)
    if persistence is None:
        persistence = torch.rand(batch, generator=generator) < 0.5
    if corruption_id.shape != (batch,) or corruption_id.dtype != torch.long:
        raise ValueError("corruption_id must be int64 [B]")
    if persistence.shape != (batch,) or persistence.dtype != torch.bool:
        raise ValueError("persistence must be bool [B]")

    true_tracks = true_part_tracks_px.permute(0, 2, 1, 3).contiguous()
    dtype = true_tracks.dtype
    candidates = torch.empty(batch, candidate_count, time, 2, dtype=dtype)
    source = torch.full(
        (batch, candidate_count), DISTRACTOR_SOURCE, dtype=torch.long
    )
    # Global acquisition tubes: visible everywhere but quantized/noisy.
    global_bias = (torch.rand(batch, PART_COUNT, 1, 2, generator=generator, dtype=dtype) - 0.5) * 8.0
    global_noise = torch.randn(
        batch, PART_COUNT, time, 2, generator=generator, dtype=dtype
    )
    candidates[:, :PART_COUNT] = true_tracks + global_bias + global_noise
    source[:, :PART_COUNT] = GLOBAL_ACQUISITION_SOURCE

    # Two accurate posterior/prior tubes exist only in persistence rows.
    local_bias = (torch.rand(batch, 2, 1, 2, generator=generator, dtype=dtype) - 0.5) * 3.0
    local_noise = 0.45 * torch.randn(batch, 2, time, 2, generator=generator, dtype=dtype)
    local_true = true_tracks[:, list(TIP_PART_INDICES)] + local_bias + local_noise
    false_local = torch.rand(batch, 4, time, 2, generator=generator, dtype=dtype)
    false_local *= float(image_size - 1)
    candidates[:, 6:8] = torch.where(
        persistence[:, None, None, None], local_true, false_local[:, :2]
    )
    candidates[:, 8:10] = false_local[:, 2:4]
    source[:, 6:10] = LOCAL_PERSISTENCE_SOURCE

    # Cross-window shuffled part tubes are hard distractors with plausible motion.
    shuffled_batch = true_tracks.roll(1, dims=0)
    for row in range(10, min(16, candidate_count)):
        part = (row - 10) % PART_COUNT
        offset = (torch.rand(batch, 1, 1, generator=generator, dtype=dtype) - 0.5) * 18.0
        candidates[:, row] = shuffled_batch[:, part] + torch.cat((offset, -offset), dim=-1)

    # Remaining tubes mix static points and smooth random walks.
    remaining = candidate_count - 16
    if remaining > 0:
        initial = torch.rand(batch, remaining, 1, 2, generator=generator, dtype=dtype)
        initial *= float(image_size - 1)
        steps = 0.8 * torch.randn(
            batch, remaining, time, 2, generator=generator, dtype=dtype
        )
        steps[:, ::2] = 0.0
        candidates[:, 16:] = initial + steps.cumsum(dim=2)

    # Transform true points and every proposal by the same hidden camera shake.
    combined = torch.cat((true_tracks, candidates), dim=1)
    combined = _apply_common_shake(
        combined,
        corruption_id,
        generator=generator,
        image_size=image_size,
    )
    true_tracks = combined[:, :PART_COUNT]
    candidates = combined[:, PART_COUNT:]
    valid = torch.ones(batch, candidate_count, time, dtype=torch.bool)
    occluded = (corruption_id == 2) | (corruption_id == 3)
    if bool(occluded.any()):
        dropout = torch.rand(batch, candidate_count, time - 1, generator=generator) < 0.22
        valid[:, :, :-1] &= ~(dropout & occluded[:, None, None])

    target = torch.arange(PART_COUNT, dtype=torch.long)[None].expand(batch, -1).clone()
    target[:, TIP_PART_INDICES[0]] = torch.where(
        persistence, torch.full_like(target[:, 0], 6), target[:, TIP_PART_INDICES[0]]
    )
    target[:, TIP_PART_INDICES[1]] = torch.where(
        persistence, torch.full_like(target[:, 0], 7), target[:, TIP_PART_INDICES[1]]
    )

    permutation = torch.rand(batch, candidate_count, generator=generator).argsort(dim=1)
    gather_tracks = permutation[:, :, None, None].expand(-1, -1, time, 2)
    candidates = candidates.gather(1, gather_tracks)
    valid = valid.gather(1, permutation[:, :, None].expand(-1, -1, time))
    source = source.gather(1, permutation)
    inverse = torch.empty_like(permutation)
    inverse.scatter_(
        1,
        permutation,
        torch.arange(candidate_count, dtype=torch.long)[None].expand(batch, -1),
    )
    target = inverse.gather(1, target)
    true_part_tracks = true_tracks.permute(0, 2, 1, 3).contiguous()
    return SyntheticAssociationBatch(
        candidate_tracks_px=candidates,
        candidate_valid=valid,
        candidate_source=source,
        part_motion=part_motion,
        target_candidate_index=target,
        true_part_tracks_px=true_part_tracks,
        corruption_id=corruption_id,
        persistence=persistence,
    )
