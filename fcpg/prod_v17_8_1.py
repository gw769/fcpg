"""Prod-V17.8.1 incremental task/clock correction.

V17.8.1 deliberately keeps the trained V17.8 Measurement, belief LeWM and
four-layer action decoder.  It adds only two pieces of information that were
missing from the failed live run:

* task-conditioned soft DINO readouts for the pickup and drop roles; and
* an explicit runtime clock for post-command settling, separate from the timed
  q0-q5 path stored by LeWM and raw history.

Both additions are zero-gated at construction.  A V17.8 checkpoint can
therefore be migrated without changing a single command before the new
parameters have been trained.  No hard object coordinate, phase label, IK or
FK result is accepted by the inference-facing Actor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from fcpg.prod_v17_6 import V176ActorOutput, _ActorLayer, _bool, _finite, _shape
from fcpg.prod_v17_8 import (
    ACTOR_DIM,
    ANCHORS,
    COMMAND_STEPS,
    STATE_DIM,
    TASK_EMBEDDING_DIM,
    VIEWS,
    VIT_DIM,
    VIT_PATCHES,
    V178Actor,
    V178Config,
    V178ContractError,
    V178Core,
    V178GroundedState,
    V178RawHistory,
    project_absolute_joint_limits,
)
from fcpg.prod_v17_8_1_clock import V1781SettledHistory
from fcpg.prod_v17_8_1_dino import (
    OBJECT_ROLE_COUNT,
    V1781DinoObjectOutput,
    V1781TaskConditionedDinoObjectHead,
)


VERSION = "17.8.1"
ARCHITECTURE_CONTRACT = "task-conditioned-settled-belief-actor-v1"


class V1781ContractError(V178ContractError):
    """Raised when a V17.8.1 tensor violates the incremental contract."""


@dataclass(frozen=True)
class V1781Config(V178Config):
    """V17.8 configuration plus the compact DINO object readout."""

    object_query_dim: int = ACTOR_DIM
    object_attention_temperature: float = 0.25
    correction_layers: int = 2

    def validate(self) -> None:
        super().validate()
        if int(self.object_query_dim) <= 0:
            raise V1781ContractError("object_query_dim must be positive")
        if int(self.correction_layers) <= 0:
            raise V1781ContractError("correction_layers must be positive")
        temperature = float(self.object_attention_temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise V1781ContractError(
                "object_attention_temperature must be finite and positive"
            )


@dataclass(frozen=True)
class V1781ActResult:
    """One Actor output and the exact pickup/drop maps used to produce it."""

    actor: V176ActorOutput
    objects: V1781DinoObjectOutput


class V1781Actor(V178Actor):
    """V17.8 Actor with a paired soft-object residual branch.

    Settling remains a runtime clock correction, not an Actor feature: neither
    the 320K archive nor the fish demonstrations contain a distinct post-q5
    wait interval.  Feeding a live-only settle token would create a new
    train/deploy shift.  The next Actor call already receives the settled
    current q, while its raw history preserves the command-time path.
    """

    def __init__(self, config: V1781Config = V1781Config()) -> None:
        if not isinstance(config, V1781Config):
            raise V1781ContractError("V1781Actor requires a V1781Config")
        super().__init__(config)
        dimension = int(config.actor_dim)
        self.object_projection = nn.Sequential(
            nn.LayerNorm(VIT_DIM),
            nn.Linear(VIT_DIM, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.object_view_embedding = nn.Parameter(
            torch.empty(1, VIEWS, 1, dimension)
        )
        self.object_role_embedding = nn.Parameter(
            torch.empty(1, 1, OBJECT_ROLE_COUNT, dimension)
        )
        self.object_valid_embedding = nn.Parameter(torch.empty(2, dimension))
        self.correction_queries = nn.Parameter(
            torch.empty(1, COMMAND_STEPS, dimension)
        )
        self.correction_time_embedding = nn.Parameter(
            torch.empty(1, COMMAND_STEPS, dimension)
        )
        self.correction_layers = nn.ModuleList(
            _ActorLayer(
                dimension,
                int(config.actor_heads),
                config.ffn_multiplier * dimension,
            )
            for _ in range(int(config.correction_layers))
        )
        self.correction_norm = nn.LayerNorm(dimension)
        self.correction_head = nn.Linear(dimension, STATE_DIM)
        for value in (
            self.object_view_embedding,
            self.object_role_embedding,
            self.object_valid_embedding,
            self.correction_queries,
            self.correction_time_embedding,
        ):
            nn.init.normal_(value, std=0.02)
        # Exact warm-start parity: every new context module may be initialized,
        # but its action residual is identically zero until V17.8.1 training.
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    @staticmethod
    def _split_history(
        history: V178RawHistory | V1781SettledHistory,
    ) -> tuple[V178RawHistory, V1781SettledHistory | None]:
        if isinstance(history, V1781SettledHistory):
            return history.raw, history
        if isinstance(history, V178RawHistory):
            return history, None
        raise V1781ContractError(
            "history must be V178RawHistory or V1781SettledHistory"
        )

    def _object_tokens(
        self,
        objects: V1781DinoObjectOutput,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(reference)
        role = _shape(
            "objects.role_tokens",
            objects.role_tokens,
            (batch, VIEWS, OBJECT_ROLE_COUNT, VIT_DIM),
        ).to(reference)
        valid = _bool(
            "objects.view_valid", objects.view_valid, (batch, VIEWS)
        ).to(reference.device)
        token = (
            self.object_projection(role)
            + self.object_view_embedding
            + self.object_role_embedding
            + self.object_valid_embedding[valid.long()][:, :, None]
        )
        return token.reshape(
            batch, VIEWS * OBJECT_ROLE_COUNT, self.config.actor_dim
        )

    def forward(  # type: ignore[override]
        self,
        scene: torch.Tensor,
        dynamics_grounding: torch.Tensor,
        ordered_tokens: torch.Tensor,
        innovation: torch.Tensor,
        current_q: torch.Tensor,
        appearance_available: torch.Tensor,
        *,
        history: V178RawHistory | V1781SettledHistory,
        task_embedding: torch.Tensor,
        objects: V1781DinoObjectOutput,
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
        task = _shape(
            "task_embedding",
            _finite("task_embedding", task_embedding).to(scene),
            (batch, TASK_EMBEDDING_DIM),
        )
        raw_history, _settled_history = self._split_history(history)

        base = super().forward(
            scene,
            grounded,
            appearance,
            error,
            state,
            available,
            history=raw_history,
            task_embedding=task,
        )
        object_group = self._object_tokens(objects, scene)
        # The residual branch has no q/task bypass.  Task and q already enter
        # the inherited V17.8 Actor; every learned V17.8.1 correction must pass
        # through the task-conditioned spatial DINO readout.
        memory = object_group
        query = self.correction_queries.expand(batch, -1, -1)
        query = query + self.correction_time_embedding
        causal = torch.triu(
            torch.ones(
                COMMAND_STEPS,
                COMMAND_STEPS,
                dtype=torch.bool,
                device=query.device,
            ),
            diagonal=1,
        )
        for layer in self.correction_layers:
            query = layer(query, memory, causal)
        correction = self.correction_head(self.correction_norm(query))
        increments = base.increments.to(correction) + correction
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
            action_queries_by_layer=base.action_queries_by_layer,
        )


class V1781Core(V178Core):
    """V17.8 belief stack with an incrementally trainable V17.8.1 Actor."""

    def __init__(self, config: V1781Config = V1781Config()) -> None:
        if not isinstance(config, V1781Config):
            raise V1781ContractError("V1781Core requires a V1781Config")
        super().__init__(config)
        self.actor = V1781Actor(config)
        self.object_head = V1781TaskConditionedDinoObjectHead(
            query_dim=int(config.object_query_dim),
            temperature=float(config.object_attention_temperature),
        )

    def act_with_diagnostics(
        self,
        scene_tokens: torch.Tensor,
        grounded: V178GroundedState,
        innovation: torch.Tensor,
        current_q: torch.Tensor,
        *,
        history: V178RawHistory | V1781SettledHistory,
        task_embedding: torch.Tensor,
    ) -> V1781ActResult:
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
        objects = self.object_head(actor_scene, task_embedding, valid)
        actor_output = self.actor(
            actor_scene,
            grounded.dynamics_grounding,
            actor_tokens,
            innovation,
            current_q,
            appearance_available,
            history=history,
            task_embedding=task_embedding,
            objects=objects,
        )
        return V1781ActResult(actor=actor_output, objects=objects)

    def act(  # type: ignore[override]
        self,
        scene_tokens: torch.Tensor,
        grounded: V178GroundedState,
        innovation: torch.Tensor,
        current_q: torch.Tensor,
        *,
        history: V178RawHistory | V1781SettledHistory,
        task_embedding: torch.Tensor,
    ) -> V176ActorOutput:
        return self.act_with_diagnostics(
            scene_tokens,
            grounded,
            innovation,
            current_q,
            history=history,
            task_embedding=task_embedding,
        ).actor


__all__ = [
    "ARCHITECTURE_CONTRACT",
    "VERSION",
    "V1781ActResult",
    "V1781Actor",
    "V1781Config",
    "V1781ContractError",
    "V1781Core",
]
