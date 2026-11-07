"""Prod-V17.8 task-conditioned historical belief controller.

V17.8 preserves V17.7's factorized measurement, physics-conditioned
association and camera-aware belief transport.  It changes two contracts:

* exchangeable visual hypotheses are trained with one *hard* two-permutation
  set match instead of marginalizing both permutations; and
* the Actor consumes the current scene, paired grounding/innovation tracks,
  three completed real transitions, current joint state and a frozen text
  embedding at the same time.

The Actor emits five unconstrained joint increments.  Their cumulative sum is
clamped only to the absolute joint limits.  There is deliberately no fixed
per-command step cap in the learned policy.  Any deployment-only mechanical
protection remains outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.prod_v17_6 import (
    ABSENT_INDEX,
    ANCHORS,
    BELIEF_CLASSES,
    COMMAND_STEPS,
    COMPACT_DIM,
    FEATURE_DIM,
    GRID_CELLS,
    GRID_SIZE,
    HYPOTHESES,
    IMAGE_SIZE,
    PATH_STEPS,
    STATE_DIM,
    VIEWS,
    VIEW_ORDER,
    VIT_DIM,
    VIT_PATCHES,
    V176ActorOutput,
    V176AssignedMeasurement,
    V176FilterUpdate,
    V176GroundedState,
    V176GroundingEncoder,
    V176Innovation,
    V176InnovationEncoder,
    V176LeWMPrediction,
    V176Measurement,
    _ActorLayer,
    _EncoderLayer,
    _action_contract,
    _bool,
    _finite,
    _normalized_log_belief,
    _shape,
    belief_prediction_nll,
    belief_statistics,
    compute_innovation,
    fuse_measurement_prior,
    identity_or_track_assignment,
    normalize_actor_scene_tokens,
    posterior_weighted_pool,
    state_path_smooth_l1,
    uniform_spatial_prior,
)
from fcpg.prod_v17_7 import (
    HISTORY_STEPS,
    HISTORY_TOKENS_PER_STEP,
    SOURCE_UNKNOWN,
    V177BeliefLeWM,
    V177Config,
    V177ContractError,
    V177Core,
    V177ExchangeableMeasurement,
    V177RawHistory,
    analytic_shift_from_pixels,
    append_raw_history,
    broaden_geometry_prior,
    credible_coverage,
    empty_raw_history,
    pool_scene_tokens,
    propagate_belief_v177,
)


ACTOR_DIM: int = 192
ACTOR_HEADS: int = 6
ACTOR_CONTEXT_LAYERS: int = 2
TASK_EMBEDDING_DIM: int = 384
TASK_ENCODER: str = "sentence-transformers/all-MiniLM-L6-v2"


class V178ContractError(V177ContractError):
    """Raised when a tensor violates the public V17.8 contract."""


# V17.8 deliberately leaves these modules and value types structurally equal
# to V17.7.  Training is from scratch; aliases are an interface guarantee, not
# permission to silently load an older checkpoint.
V178ExchangeableMeasurement = V177ExchangeableMeasurement
V178BeliefLeWM = V177BeliefLeWM
V178RawHistory = V177RawHistory
V178Measurement = V176Measurement
V178AssignedMeasurement = V176AssignedMeasurement
V178GroundedState = V176GroundedState
V178GroundingEncoder = V176GroundingEncoder
V178LeWMPrediction = V176LeWMPrediction
V178Innovation = V176Innovation
V178InnovationEncoder = V176InnovationEncoder
V178FilterUpdate = V176FilterUpdate
V178ActorOutput = V176ActorOutput


@dataclass(frozen=True)
class V178Config(V177Config):
    """Complete V17.8 model configuration.

    ``heads`` continues to configure Measurement/LeWM.  ``actor_heads`` is
    separate so the 192-dimensional Actor uses six heads without changing the
    established 384-dimensional belief stack.
    """

    actor_dim: int = ACTOR_DIM
    actor_layers: int = 4
    actor_heads: int = ACTOR_HEADS
    actor_context_layers: int = ACTOR_CONTEXT_LAYERS
    task_embedding_dim: int = TASK_EMBEDDING_DIM
    actor_mode: str = "full"
    raw_history_steps: int = HISTORY_STEPS
    separation_weight: float = 0.0

    def validate(self) -> None:
        super().validate()
        if int(self.actor_heads) <= 0 or self.actor_dim % int(self.actor_heads):
            raise V178ContractError("actor_heads must divide actor_dim")
        if int(self.actor_context_layers) <= 0:
            raise V178ContractError("actor_context_layers must be positive")
        if int(self.task_embedding_dim) != TASK_EMBEDDING_DIM:
            raise V178ContractError(
                f"task_embedding_dim is fixed at {TASK_EMBEDDING_DIM}"
            )
        if self.actor_mode != "full":
            raise V178ContractError("V17.8 has one Actor mode: full")
        if int(self.raw_history_steps) != HISTORY_STEPS:
            raise V178ContractError(
                f"V17.8 Actor history is fixed at {HISTORY_STEPS} transitions"
            )
        if not math.isclose(float(self.separation_weight), 0.0, abs_tol=0.0):
            raise V178ContractError(
                "V17.8 removes the hypothesis-overlap penalty"
            )


def hard_permutation_pair_cross_entropy(
    anchor_hypothesis_log_likelihood: torch.Tensor,
    target_belief: torch.Tensor,
    view_valid: torch.Tensor,
) -> torch.Tensor:
    """Hard two-permutation set matching for exchangeable hypotheses.

    For every valid view the loss chooses exactly one of the direct and swapped
    assignments.  Swapping the two predicted hypotheses therefore leaves the
    scalar loss invariant, while both hypotheses no longer receive gradients
    from both target anchors as they do under permutation marginalization.

    Targets may be one-hot or normalized soft spatial labels.  This remains a
    categorical cross-entropy objective, not a claim that the network output is
    a generative image likelihood.
    """

    hypotheses = _normalized_log_belief(
        "anchor_hypothesis_log_likelihood",
        anchor_hypothesis_log_likelihood,
    )
    if hypotheses.ndim != 4 or hypotheses.shape[1:3] != (VIEWS, HYPOTHESES):
        raise V178ContractError(
            "anchor_hypothesis_log_likelihood must be [B,2,2,3137]"
        )
    batch = len(hypotheses)
    target = _finite("target_belief", target_belief).float().to(hypotheses)
    target = _shape(
        "target_belief",
        target,
        (batch, VIEWS, ANCHORS, BELIEF_CLASSES),
    )
    non_negative = (target >= 0.0).all()
    if target.device.type == "cuda":
        torch._assert_async(non_negative, "target_belief must be non-negative")
    elif not bool(non_negative):
        raise V178ContractError("target_belief must be non-negative")
    target_mass = target.sum(dim=-1, keepdim=True)
    positive_mass = (target_mass > 0.0).all()
    if target.device.type == "cuda":
        torch._assert_async(
            positive_mass, "target_belief must have positive mass"
        )
    elif not bool(positive_mass):
        raise V178ContractError("target_belief must have positive mass")
    target = target / target_mass
    valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(
        hypotheses.device
    )

    # [B,V,target anchor,predicted hypothesis]
    cost = -torch.einsum("bvac,bvsc->bvas", target, hypotheses)
    direct = cost[:, :, 0, 0] + cost[:, :, 1, 1]
    swapped = cost[:, :, 0, 1] + cost[:, :, 1, 0]
    # Detach only the discrete assignment decision.  ``torch.minimum`` splits
    # gradients at an exact tie, which would recreate the two-permutation
    # averaging that this hard objective is intended to remove.  A tie is
    # deterministically assigned direct; swapping slots still leaves the
    # scalar objective unchanged.
    choose_direct = direct.detach() <= swapped.detach()
    per_view = torch.where(choose_direct, direct, swapped)
    return (per_view * valid).sum() / valid.sum().clamp_min(1)


def project_absolute_joint_limits(
    current_q: torch.Tensor,
    increments: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate five raw joint deltas and clamp only absolute limits.

    No per-command clipping, tanh bound or sequential step cap is applied.
    ``desired`` is returned before absolute-limit projection so training and
    diagnostics can distinguish saturation from the learned raw output.
    """

    # Keep limit projection and cumulative calibrated-position arithmetic in FP32 even
    # under bf16 autocast.  The casts remain differentiable, so gradients flow
    # back to the mixed-precision action head while hardware-facing commands
    # retain stable command resolution.
    state = _finite("current_q", current_q).float()
    batch = len(state)
    state = _shape("current_q", state, (batch, STATE_DIM))
    delta = _finite("increments", increments).float().to(state)
    delta = _shape(
        "increments", delta, (batch, COMMAND_STEPS, STATE_DIM)
    )
    lower_bound = _finite("lower", lower).float().to(state)
    lower_bound = _shape("lower", lower_bound, (STATE_DIM,))
    upper_bound = _finite("upper", upper).float().to(state)
    upper_bound = _shape("upper", upper_bound, (STATE_DIM,))
    ordered = (lower_bound < upper_bound).all()
    if state.device.type == "cuda":
        torch._assert_async(ordered, "absolute joint limits are invalid")
    elif not bool(ordered):
        raise V178ContractError("absolute joint limits are invalid")
    desired = state[:, None] + torch.cumsum(delta, dim=1)
    commands = desired.clamp(
        min=lower_bound.reshape(1, 1, STATE_DIM),
        max=upper_bound.reshape(1, 1, STATE_DIM),
    )
    return commands, desired


class V178Actor(nn.Module):
    """Task-conditioned five-command Actor with real execution history.

    Dense current-scene tokens stay outside the compact two-layer context
    encoder.  That encoder jointly processes paired belief/innovation tracks,
    three completed transition summaries, current q and the frozen task-text
    embedding.  Its output and the scene tokens form the memory read by four
    causal action-decoder layers.
    """

    _SCENE_KIND = 0
    _TRACK_KIND = 1
    _HISTORY_KIND = 2
    _Q_KIND = 3
    _TASK_KIND = 4

    def __init__(self, config: V178Config = V178Config()) -> None:
        super().__init__()
        if not isinstance(config, V178Config):
            raise V178ContractError("V178Actor requires a V178Config")
        config.validate()
        self.config = config
        dimension = int(config.actor_dim)

        self.scene_projection = nn.Linear(VIT_DIM, dimension)
        self.track_projection = nn.Sequential(
            nn.LayerNorm(2 * config.compact_dim),
            nn.Linear(2 * config.compact_dim, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.appearance_projection = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.q_projection = nn.Sequential(
            nn.Linear(STATE_DIM, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.task_projection = nn.Sequential(
            nn.LayerNorm(config.task_embedding_dim),
            nn.Linear(config.task_embedding_dim, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        history_motion_dim = (COMMAND_STEPS + PATH_STEPS) * STATE_DIM
        self.history_scene_projection = nn.Linear(VIT_DIM, dimension)
        self.history_motion_projection = nn.Sequential(
            nn.LayerNorm(history_motion_dim),
            nn.Linear(history_motion_dim, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )

        self.scene_view_embedding = nn.Parameter(
            torch.empty(1, VIEWS, 1, dimension)
        )
        self.track_view_embedding = nn.Parameter(
            torch.empty(1, VIEWS, 1, dimension)
        )
        self.appearance_status_embedding = nn.Parameter(
            torch.empty(2, dimension)
        )
        self.kind_embedding = nn.Parameter(torch.empty(5, dimension))
        self.history_slot_embedding = nn.Parameter(
            torch.empty(1, HISTORY_STEPS, 1, dimension)
        )
        self.history_kind_embedding = nn.Parameter(
            torch.empty(1, 1, HISTORY_TOKENS_PER_STEP, dimension)
        )
        self.history_null_token = nn.Parameter(
            torch.empty(1, 1, 1, dimension)
        )
        self.context_layers = nn.ModuleList(
            _EncoderLayer(
                dimension,
                int(config.actor_heads),
                config.ffn_multiplier * dimension,
            )
            for _ in range(config.actor_context_layers)
        )
        self.context_norm = nn.LayerNorm(dimension)
        self.action_queries = nn.Parameter(
            torch.empty(1, COMMAND_STEPS, dimension)
        )
        self.action_time_embedding = nn.Parameter(
            torch.empty(1, COMMAND_STEPS, dimension)
        )
        self.layers = nn.ModuleList(
            _ActorLayer(
                dimension,
                int(config.actor_heads),
                config.ffn_multiplier * dimension,
            )
            for _ in range(config.actor_layers)
        )
        self.output_norm = nn.LayerNorm(dimension)
        self.action_head = nn.Linear(dimension, STATE_DIM)

        _, lower, upper = _action_contract()
        self.register_buffer("action_lower", lower.float(), persistent=True)
        self.register_buffer("action_upper", upper.float(), persistent=True)
        self.register_buffer(
            "joint_center", ((lower + upper) * 0.5).float(), persistent=True
        )
        self.register_buffer(
            "joint_half_range", ((upper - lower) * 0.5).float(), persistent=True
        )

        for value in (
            self.scene_view_embedding,
            self.track_view_embedding,
            self.appearance_status_embedding,
            self.kind_embedding,
            self.history_slot_embedding,
            self.history_kind_embedding,
            self.history_null_token,
            self.action_queries,
            self.action_time_embedding,
        ):
            nn.init.normal_(value, std=0.02)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def _history_tokens(
        self,
        history: V177RawHistory,
        batch: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        commands = _shape(
            "history.commands",
            history.commands,
            (batch, HISTORY_STEPS, COMMAND_STEPS, STATE_DIM),
        ).to(reference)
        paths = _shape(
            "history.paths",
            history.paths,
            (batch, HISTORY_STEPS, PATH_STEPS, STATE_DIM),
        ).to(reference)
        pooled = _shape(
            "history.scene_pooled",
            history.scene_pooled,
            (batch, HISTORY_STEPS, VIEWS, VIT_DIM),
        ).to(reference)
        valid = _bool(
            "history.valid", history.valid, (batch, HISTORY_STEPS)
        ).to(reference.device)

        center = self.joint_center.reshape(1, 1, 1, STATE_DIM).to(reference)
        half_range = self.joint_half_range.reshape(
            1, 1, 1, STATE_DIM
        ).to(reference)
        normalized_commands = (commands - center) / half_range
        normalized_paths = (paths - center) / half_range
        motion = self.history_motion_projection(
            torch.cat((normalized_commands, normalized_paths), dim=2).reshape(
                batch,
                HISTORY_STEPS,
                (COMMAND_STEPS + PATH_STEPS) * STATE_DIM,
            )
        )
        pooled_scene = self.history_scene_projection(
            F.layer_norm(pooled, (VIT_DIM,))
        )
        tokens = torch.cat((pooled_scene, motion[:, :, None]), dim=2)
        tokens = (
            tokens
            + self.history_slot_embedding
            + self.history_kind_embedding
            + self.kind_embedding[self._HISTORY_KIND]
        )
        null = (
            self.history_null_token
            + self.history_slot_embedding
            + self.history_kind_embedding
            + self.kind_embedding[self._HISTORY_KIND]
        )
        tokens = torch.where(
            valid[:, :, None, None], tokens, null.expand_as(tokens)
        )
        return tokens.reshape(
            batch, HISTORY_STEPS * HISTORY_TOKENS_PER_STEP, self.config.actor_dim
        )

    def forward(
        self,
        scene: torch.Tensor,
        dynamics_grounding: torch.Tensor,
        ordered_tokens: torch.Tensor,
        innovation: torch.Tensor,
        current_q: torch.Tensor,
        appearance_available: torch.Tensor,
        *,
        history: V177RawHistory,
        task_embedding: torch.Tensor,
    ) -> V176ActorOutput:
        scene = _finite("scene", scene)
        batch = len(scene)
        scene = _shape(
            "scene", scene, (batch, VIEWS, VIT_PATCHES, VIT_DIM)
        ).to(dtype=self.scene_projection.weight.dtype)
        grounded = _shape(
            "dynamics_grounding",
            dynamics_grounding,
            (batch, VIEWS, ANCHORS, self.config.compact_dim),
        ).to(scene)
        appearance = _shape(
            "ordered_tokens",
            ordered_tokens,
            (batch, VIEWS, ANCHORS, self.config.feature_dim),
        ).to(scene)
        error = _shape(
            "innovation",
            innovation,
            (batch, VIEWS, ANCHORS, self.config.compact_dim),
        ).to(scene)
        state = _shape("current_q", current_q, (batch, STATE_DIM)).to(scene)
        available = _bool(
            "appearance_available", appearance_available, (batch, VIEWS)
        ).to(scene.device)
        task = _finite("task_embedding", task_embedding).to(scene)
        task = _shape(
            "task_embedding", task, (batch, self.config.task_embedding_dim)
        )
        if not isinstance(history, V177RawHistory):
            raise V178ContractError(
                "history must be a V177RawHistory; use empty_raw_history at H0"
            )

        scene_group = (
            self.scene_projection(scene)
            + self.scene_view_embedding
            + self.kind_embedding[self._SCENE_KIND]
        ).reshape(batch, VIEWS * VIT_PATCHES, self.config.actor_dim)

        # Grounding and the innovation belonging to it are concatenated before
        # any set processing.  A joint track permutation is invariant, while
        # permuting innovation alone changes the represented physical pair.
        available_tracks = available[:, :, None]
        track_group = (
            self.track_projection(torch.cat((grounded, error), dim=-1))
            + self.appearance_projection(appearance)
            * available_tracks[..., None].to(appearance)
            + self.appearance_status_embedding[available_tracks.long()]
            + self.track_view_embedding
            + self.kind_embedding[self._TRACK_KIND]
        ).reshape(batch, VIEWS * ANCHORS, self.config.actor_dim)

        history_group = self._history_tokens(history, batch, scene)
        q_group = (
            self.q_projection(
                (state - self.joint_center.to(state))
                / self.joint_half_range.to(state)
            )
            + self.kind_embedding[self._Q_KIND]
        )[:, None]
        task_group = (
            self.task_projection(task) + self.kind_embedding[self._TASK_KIND]
        )[:, None]
        compact = torch.cat(
            (track_group, history_group, q_group, task_group), dim=1
        )
        for layer in self.context_layers:
            compact = layer(compact)
        compact = self.context_norm(compact)
        memory = torch.cat((scene_group, compact), dim=1)

        query = self.action_queries.expand(batch, -1, -1)
        query = query + self.action_time_embedding
        causal = torch.triu(
            torch.ones(
                COMMAND_STEPS,
                COMMAND_STEPS,
                dtype=torch.bool,
                device=query.device,
            ),
            diagonal=1,
        )
        traces = [query] if self.config.trace_level == "full" else None
        for layer in self.layers:
            query = layer(query, memory, causal)
            if traces is not None:
                traces.append(query)
        query = self.output_norm(query)
        increments = self.action_head(query)
        commands, desired = project_absolute_joint_limits(
            state,
            increments,
            self.action_lower,
            self.action_upper,
        )
        return V176ActorOutput(
            commands=commands,
            desired_commands=desired,
            increments=increments,
            action_queries_by_layer=(
                torch.stack(traces, dim=1) if traces is not None else None
            ),
        )


class V178Core(V177Core):
    """V17.7 belief stack with the unified V17.8 historical task Actor."""

    def __init__(self, config: V178Config = V178Config()) -> None:
        if not isinstance(config, V178Config):
            raise V178ContractError("V178Core requires a V178Config")
        nn.Module.__init__(self)
        config.validate()
        self.config = config
        self.measurement = V177ExchangeableMeasurement(config)
        self.grounding_encoder = V176GroundingEncoder(config)
        self.lewm = V177BeliefLeWM(config)
        self.innovation_encoder = V176InnovationEncoder(config)
        self.actor = V178Actor(config)

    def act(
        self,
        scene_tokens: torch.Tensor,
        grounded: V176GroundedState,
        innovation: torch.Tensor,
        current_q: torch.Tensor,
        *,
        history: V177RawHistory,
        task_embedding: torch.Tensor,
    ) -> V176ActorOutput:
        return act_with_v178(
            self.actor,
            scene_tokens,
            grounded,
            innovation,
            current_q,
            history=history,
            task_embedding=task_embedding,
        )


def act_with_v178(
    actor: V178Actor,
    scene_tokens: torch.Tensor,
    grounded: V176GroundedState,
    innovation: torch.Tensor,
    current_q: torch.Tensor,
    *,
    history: V177RawHistory,
    task_embedding: torch.Tensor,
) -> V176ActorOutput:
    """Gate invalid current pixels, then run the unified V17.8 Actor."""

    if not isinstance(actor, V178Actor):
        raise V178ContractError("act_with_v178 requires a V178Actor")
    batch = len(grounded.ordered_tokens)
    valid = _bool(
        "grounded.view_valid", grounded.view_valid, (batch, VIEWS)
    ).to(grounded.ordered_tokens.device)
    appearance_available = _bool(
        "grounded.appearance_available",
        grounded.appearance_available,
        (batch, VIEWS),
    ).to(grounded.ordered_tokens.device)
    actor_tokens = grounded.ordered_tokens * appearance_available[
        :, :, None, None
    ].to(grounded.ordered_tokens)
    actor_scene = scene_tokens * valid[:, :, None, None].to(scene_tokens)
    return actor(
        actor_scene,
        grounded.dynamics_grounding,
        actor_tokens,
        innovation,
        current_q,
        appearance_available,
        history=history,
        task_embedding=task_embedding,
    )


V178TaskConditionedHistoricalBeliefController = V178Core


__all__ = [
    "ABSENT_INDEX",
    "ACTOR_CONTEXT_LAYERS",
    "ACTOR_DIM",
    "ACTOR_HEADS",
    "ANCHORS",
    "BELIEF_CLASSES",
    "COMMAND_STEPS",
    "COMPACT_DIM",
    "FEATURE_DIM",
    "GRID_CELLS",
    "GRID_SIZE",
    "HISTORY_STEPS",
    "HISTORY_TOKENS_PER_STEP",
    "HYPOTHESES",
    "IMAGE_SIZE",
    "PATH_STEPS",
    "SOURCE_UNKNOWN",
    "STATE_DIM",
    "TASK_EMBEDDING_DIM",
    "TASK_ENCODER",
    "VIEWS",
    "VIEW_ORDER",
    "VIT_DIM",
    "VIT_PATCHES",
    "V178Actor",
    "V178ActorOutput",
    "V178AssignedMeasurement",
    "V178BeliefLeWM",
    "V178Config",
    "V178ContractError",
    "V178Core",
    "V178ExchangeableMeasurement",
    "V178FilterUpdate",
    "V178GroundedState",
    "V178GroundingEncoder",
    "V178Innovation",
    "V178InnovationEncoder",
    "V178LeWMPrediction",
    "V178Measurement",
    "V178RawHistory",
    "V178TaskConditionedHistoricalBeliefController",
    "act_with_v178",
    "analytic_shift_from_pixels",
    "append_raw_history",
    "belief_prediction_nll",
    "belief_statistics",
    "broaden_geometry_prior",
    "compute_innovation",
    "credible_coverage",
    "empty_raw_history",
    "fuse_measurement_prior",
    "hard_permutation_pair_cross_entropy",
    "identity_or_track_assignment",
    "normalize_actor_scene_tokens",
    "pool_scene_tokens",
    "posterior_weighted_pool",
    "project_absolute_joint_limits",
    "propagate_belief_v177",
    "state_path_smooth_l1",
    "uniform_spatial_prior",
]
