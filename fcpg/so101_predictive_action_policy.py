"""Joint action and anchor-dynamics policy for grounded SO-101 control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


STRUCTURED_GOAL_SOURCES = ("target_future_state_h4", "episode_terminal_state")


@dataclass(frozen=True)
class SO101PredictiveActionOutput:
    """Actions and their predicted embodied consequences."""

    action: torch.Tensor
    delta: torch.Tensor
    action_chunks: torch.Tensor
    dynamics_action_chunks: torch.Tensor
    action_latent: torch.Tensor
    dynamics_latent: torch.Tensor
    future_context: torch.Tensor
    future_state: torch.Tensor
    future_anchor_xyz: torch.Tensor
    future_anchor_px: torch.Tensor


class SO101PredictiveActionPolicy(nn.Module):
    """Predict actions while preserving grounded three-anchor dynamics.

    The model is intentionally task-agnostic. Task intent remains in the global
    visual history; no object class, pickup point, or placement point is part of
    the input contract. FCPG contributes corrected per-anchor appearance and
    image coordinates, while FK contributes deployable base-frame geometry.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 192,
        state_dim: int = 6,
        num_views: int = 2,
        num_anchors: int = 3,
        obs_horizon: int = 3,
        action_horizon: int = 20,
        raw_steps_per_chunk: int = 5,
        image_size: int = 224,
        model_dim: int = 192,
        encoder_layers: int = 2,
        decoder_layers: int = 4,
        dynamics_layers: int = 2,
        heads: int = 6,
        ffn_dim: int = 768,
        dropout: float = 0.0,
        state_residual_cap: float = 0.1,
        state_residual_mode: str = "uniform",
        state_residual_steps: float = 1.0,
        anchor_residual_cap_m: float = 0.25,
        pixel_residual_cap_px: float = 80.0,
        context_residual_cap: float = 2.0,
        dynamics_conditioning: str = "actor_latent",
        anchor_baseline_mode: str = "current",
        structured_goal: bool = False,
        goal_query_residual: bool = True,
        structured_goal_source: str | None = None,
        task_class_conditioning: bool = False,
        max_task_class: int = 8,
    ) -> None:
        super().__init__()
        if state_dim != 6:
            raise ValueError("SO-101 requires six joint positions")
        if latent_dim <= 0 or model_dim <= 0 or model_dim % heads:
            raise ValueError("invalid latent/model/head dimensions")
        if action_horizon <= 0 or action_horizon % raw_steps_per_chunk:
            raise ValueError("action_horizon must be divisible by raw_steps_per_chunk")
        if min(num_views, num_anchors, obs_horizon, image_size) <= 0:
            raise ValueError("view, anchor, history, and image dimensions must be positive")
        if min(
            state_residual_cap,
            anchor_residual_cap_m,
            pixel_residual_cap_px,
            context_residual_cap,
        ) <= 0:
            raise ValueError("prediction caps must be positive")
        if state_residual_mode not in {"uniform", "step_cap"}:
            raise ValueError("state_residual_mode must be 'uniform' or 'step_cap'")
        if state_residual_steps <= 0:
            raise ValueError("state_residual_steps must be positive")
        if dynamics_conditioning not in {"actor_latent", "history_action"}:
            raise ValueError(
                "dynamics_conditioning must be 'actor_latent' or 'history_action'"
            )
        if anchor_baseline_mode not in {"current", "external_fk"}:
            raise ValueError(
                "anchor_baseline_mode must be 'current' or 'external_fk'"
            )

        self.latent_dim = int(latent_dim)
        self.state_dim = int(state_dim)
        self.num_views = int(num_views)
        self.num_anchors = int(num_anchors)
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.raw_steps_per_chunk = int(raw_steps_per_chunk)
        self.future_horizon = self.action_horizon // self.raw_steps_per_chunk
        self.image_size = int(image_size)
        self.model_dim = int(model_dim)
        self.state_residual_cap = float(state_residual_cap)
        self.state_residual_mode = str(state_residual_mode)
        self.state_residual_steps = float(state_residual_steps)
        self.anchor_residual_cap_m = float(anchor_residual_cap_m)
        self.pixel_residual_cap_px = float(pixel_residual_cap_px)
        self.context_residual_cap = float(context_residual_cap)
        self.dynamics_conditioning = str(dynamics_conditioning)
        self.anchor_baseline_mode = str(anchor_baseline_mode)
        self.structured_goal = bool(structured_goal)
        self.goal_query_residual = bool(goal_query_residual)
        if structured_goal_source is not None and structured_goal_source not in STRUCTURED_GOAL_SOURCES:
            raise ValueError(
                f"unknown structured_goal_source {structured_goal_source!r}; "
                f"expected one of {STRUCTURED_GOAL_SOURCES}"
            )
        if self.structured_goal and structured_goal_source is None:
            structured_goal_source = "target_future_state_h4"
        if not self.structured_goal and structured_goal_source is not None:
            raise ValueError("structured_goal_source requires structured_goal=True")
        if (
            structured_goal_source == "episode_terminal_state"
            and self.goal_query_residual
        ):
            raise ValueError(
                "episode_terminal_state is V11.3 memory-only; "
                "query residual is the frozen V11.2 leak"
            )
        if (
            structured_goal_source == "target_future_state_h4"
            and not self.goal_query_residual
        ):
            raise ValueError(
                "target_future_state_h4 is the frozen V11.2 path and keeps "
                "the query residual"
            )
        self.structured_goal_source = structured_goal_source
        if max_task_class < 1:
            raise ValueError("max_task_class must be at least 1")
        self.task_class_conditioning = bool(task_class_conditioning)
        self.max_task_class = int(max_task_class)

        self.frame_embedding = nn.Parameter(torch.empty(obs_horizon, model_dim))
        self.view_embedding = nn.Parameter(torch.empty(num_views, model_dim))
        self.anchor_embedding = nn.Parameter(torch.empty(num_anchors, model_dim))
        self.horizon_embedding = nn.Parameter(
            torch.empty(1, self.future_horizon, model_dim)
        )
        self.context_type_embedding = nn.Parameter(torch.empty(model_dim))
        self.visual_anchor_type_embedding = nn.Parameter(torch.empty(model_dim))
        self.geometry_anchor_type_embedding = nn.Parameter(torch.empty(model_dim))
        self.state_type_embedding = nn.Parameter(torch.empty(model_dim))
        self.past_action_type_embedding = nn.Parameter(torch.empty(model_dim))
        self.action_queries = nn.Parameter(torch.empty(1, action_horizon, model_dim))
        self.action_position = nn.Parameter(torch.empty(1, action_horizon, model_dim))

        self.register_buffer("step_cap", torch.ones(state_dim))
        self.register_buffer("action_lower", -torch.ones(state_dim))
        self.register_buffer("action_upper", torch.ones(state_dim))
        self.register_buffer(
            "action_contract_set", torch.zeros((), dtype=torch.bool), persistent=False
        )

        self.context_projection = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, model_dim)
        )
        self.visual_anchor_projection = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, model_dim)
        )
        self.pixel_projection = nn.Sequential(
            nn.Linear(2, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(3, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(state_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.past_action_projection = nn.Sequential(
            nn.Linear(raw_steps_per_chunk * state_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.chunk_action_projection = nn.Sequential(
            nn.Linear(raw_steps_per_chunk * state_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        if self.structured_goal:
            self.goal_projection = nn.Sequential(
                nn.Linear(state_dim, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
            )
            self.goal_type_embedding = nn.Parameter(torch.zeros(model_dim))
            nn.init.zeros_(self.goal_projection[-1].weight)
            nn.init.zeros_(self.goal_projection[-1].bias)
        if self.task_class_conditioning:
            self.task_class_embedding = nn.Embedding(
                self.max_task_class + 1, model_dim
            )
            self.task_class_type_embedding = nn.Parameter(torch.zeros(model_dim))
            nn.init.normal_(self.task_class_embedding.weight, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        dynamics_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.dynamics_input = nn.Sequential(
            nn.LayerNorm(model_dim * 2), nn.Linear(model_dim * 2, model_dim)
        )
        self.dynamics = nn.TransformerEncoder(
            dynamics_layer,
            num_layers=dynamics_layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, state_dim),
        )
        self.state_residual_head = nn.Linear(model_dim, state_dim)
        self.context_residual_head = nn.Linear(
            model_dim, num_views * latent_dim
        )
        self.anchor_residual_head = nn.Linear(
            model_dim, num_anchors * 3
        )
        self.pixel_residual_head = nn.Linear(
            model_dim, num_views * num_anchors * 2
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        embeddings = (
            self.frame_embedding,
            self.view_embedding,
            self.anchor_embedding,
            self.horizon_embedding,
            self.context_type_embedding,
            self.visual_anchor_type_embedding,
            self.geometry_anchor_type_embedding,
            self.state_type_embedding,
            self.past_action_type_embedding,
            self.action_queries,
            self.action_position,
        )
        for value in embeddings:
            nn.init.normal_(value, std=0.02)
        # Start from the safe hold/current-geometry baseline.
        for head in (
            self.delta_head[-1],
            self.state_residual_head,
            self.context_residual_head,
            self.anchor_residual_head,
            self.pixel_residual_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def set_action_contract(
        self,
        step_cap: torch.Tensor | np.ndarray,
        lower: torch.Tensor | np.ndarray,
        upper: torch.Tensor | np.ndarray,
    ) -> None:
        cap = torch.as_tensor(step_cap, dtype=self.step_cap.dtype)
        low = torch.as_tensor(lower, dtype=self.action_lower.dtype)
        high = torch.as_tensor(upper, dtype=self.action_upper.dtype)
        expected = (self.state_dim,)
        if tuple(cap.shape) != expected or tuple(low.shape) != expected or tuple(high.shape) != expected:
            raise ValueError(f"action contract tensors must have shape {expected}")
        if torch.any(cap <= 0) or torch.any(low >= high):
            raise ValueError("action contract is invalid")
        self.step_cap.copy_(cap)
        self.action_lower.copy_(low)
        self.action_upper.copy_(high)
        self.action_contract_set.fill_(True)

    def step_scaled_residual(
        self, value: torch.Tensor, current: torch.Tensor
    ) -> torch.Tensor:
        """Motion relative to ``current``, measured in commandable steps.

        The absolute action is ``current`` plus the integrated delta, so a loss
        taken in absolute space is already a delta loss -- just divided by
        JOINT_SCALE. Over a 20-step horizon the real motion is roughly one
        ``step_cap`` (2.3-5.3 deg), i.e. ~0.03 normalized, which sits deep in the
        quadratic region of smooth_l1 and produces almost no gradient: the
        deployed checkpoint ended up worse than repeating ``current``. Dividing
        by ``step_cap`` puts the residual near unity and weights each joint by
        its own commandable range rather than by degrees, so the gripper
        (cap 6.2 deg) and the roll (cap 2.1 deg) are no longer scored alike.
        """

        if value.ndim != 3 or value.shape[-1] != self.state_dim:
            raise ValueError("value must be [batch, horizon, state_dim]")
        if current.ndim != 3 or current.shape[1] != 1:
            raise ValueError("current must be [batch, 1, state_dim]")
        return (value - current) / self.step_cap.view(1, 1, -1)

    def _validate_inputs(
        self,
        per_view_context: torch.Tensor,
        per_view_anchor_tokens: torch.Tensor,
        anchor_valid: torch.Tensor,
        view_valid: torch.Tensor,
        q_corrected_px: torch.Tensor,
        anchor_xyz_history: torch.Tensor,
        state_history: torch.Tensor,
        past_action_chunks: torch.Tensor,
    ) -> int:
        batch = int(state_history.shape[0])
        if tuple(per_view_context.shape) != (
            batch, self.obs_horizon, self.num_views, self.latent_dim
        ):
            raise ValueError("per_view_context has an invalid shape")
        if tuple(per_view_anchor_tokens.shape) != (
            batch,
            self.obs_horizon,
            self.num_views,
            self.num_anchors,
            self.latent_dim,
        ):
            raise ValueError("per_view_anchor_tokens has an invalid shape")
        if tuple(anchor_valid.shape) != (
            batch, self.obs_horizon, self.num_views, self.num_anchors
        ):
            raise ValueError("anchor_valid has an invalid shape")
        if tuple(view_valid.shape) != (
            batch, self.obs_horizon, self.num_views
        ):
            raise ValueError("view_valid has an invalid shape")
        if tuple(q_corrected_px.shape) != (
            batch, self.obs_horizon, self.num_views, self.num_anchors, 2
        ):
            raise ValueError("q_corrected_px has an invalid shape")
        if tuple(anchor_xyz_history.shape) != (
            batch, self.obs_horizon, self.num_anchors, 3
        ):
            raise ValueError("anchor_xyz_history has an invalid shape")
        if tuple(state_history.shape) != (
            batch, self.obs_horizon, self.state_dim
        ):
            raise ValueError("state_history has an invalid shape")
        if tuple(past_action_chunks.shape) != (
            batch,
            self.obs_horizon - 1,
            self.raw_steps_per_chunk * self.state_dim,
        ):
            raise ValueError("past_action_chunks has an invalid shape")
        if not torch.all(view_valid.any(dim=(1, 2))):
            raise ValueError("every sample requires at least one valid global view")
        if not torch.all(torch.isfinite(anchor_xyz_history)):
            raise ValueError("anchor_xyz_history must be finite")
        return batch

    def _encode_memory(
        self,
        per_view_context: torch.Tensor,
        per_view_anchor_tokens: torch.Tensor,
        anchor_valid: torch.Tensor,
        view_valid: torch.Tensor,
        q_corrected_px: torch.Tensor,
        anchor_xyz_history: torch.Tensor,
        state_history: torch.Tensor,
        past_action_chunks: torch.Tensor,
        goal_state: torch.Tensor | None = None,
        task_class: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame = self.frame_embedding[None, :, None, :]
        view = self.view_embedding[None, None, :, :]
        valid_context = torch.where(
            view_valid[..., None], per_view_context, torch.zeros_like(per_view_context)
        )
        context = (
            self.context_projection(valid_context)
            + frame
            + view
            + self.context_type_embedding
        )

        visual_valid = anchor_valid & view_valid[..., None]
        finite_visual = torch.where(
            visual_valid[..., None],
            per_view_anchor_tokens,
            torch.zeros_like(per_view_anchor_tokens),
        )
        finite_q = torch.where(
            visual_valid[..., None], q_corrected_px, torch.zeros_like(q_corrected_px)
        )
        q_normalized = 2.0 * finite_q / float(self.image_size - 1) - 1.0
        visual_anchors = (
            self.visual_anchor_projection(finite_visual)
            + self.pixel_projection(q_normalized)
            + frame[:, :, :, None, :]
            + view[:, :, :, None, :]
            + self.anchor_embedding[None, None, None, :, :]
            + self.visual_anchor_type_embedding
        )

        geometry_anchors = (
            self.geometry_projection(anchor_xyz_history)
            + self.frame_embedding[None, :, None, :]
            + self.anchor_embedding[None, None, :, :]
            + self.geometry_anchor_type_embedding
        )
        state = (
            self.state_projection(state_history)
            + self.frame_embedding[None, :, :]
            + self.state_type_embedding
        )
        past_action = (
            self.past_action_projection(past_action_chunks)
            + self.frame_embedding[None, 1:, :]
            + self.past_action_type_embedding
        )

        tokens = [
            context.flatten(start_dim=1, end_dim=2),
            visual_anchors.flatten(start_dim=1, end_dim=3),
            geometry_anchors.flatten(start_dim=1, end_dim=2),
            state,
            past_action,
        ]
        padding_parts = [
            ~view_valid.flatten(start_dim=1),
            ~visual_valid.flatten(start_dim=1),
            torch.zeros(
                state_history.shape[0],
                self.obs_horizon * self.num_anchors,
                dtype=torch.bool,
                device=state_history.device,
            ),
            torch.zeros(
                state_history.shape[0],
                self.obs_horizon + self.obs_horizon - 1,
                dtype=torch.bool,
                device=state_history.device,
            ),
        ]
        if self.structured_goal:
            if goal_state is None:
                raise ValueError("structured_goal requires goal_state [B, state_dim]")
            batch = int(state_history.shape[0])
            if tuple(goal_state.shape) != (batch, self.state_dim):
                raise ValueError("goal_state must have shape [B, state_dim]")
            if not torch.isfinite(goal_state).all():
                raise ValueError("goal_state must be finite")
            goal = self.goal_projection(goal_state) + self.goal_type_embedding
            tokens.append(goal.unsqueeze(1))
            padding_parts.append(
                torch.zeros(batch, 1, dtype=torch.bool, device=goal_state.device)
            )
        elif goal_state is not None:
            raise ValueError("goal_state requires structured_goal=True")
        if self.task_class_conditioning and task_class is not None:
            batch = int(state_history.shape[0])
            if tuple(task_class.shape) != (batch,):
                raise ValueError("task_class must have shape [B]")
            if task_class.dtype not in (torch.int32, torch.int64):
                raise ValueError("task_class must be an integer tensor")
            if torch.any(task_class < 1) or torch.any(task_class > self.max_task_class):
                raise ValueError(
                    f"task_class must lie in [1, {self.max_task_class}]"
                )
            tag = self.task_class_embedding(task_class) + self.task_class_type_embedding
            tokens.append(tag.unsqueeze(1))
            padding_parts.append(
                torch.zeros(batch, 1, dtype=torch.bool, device=task_class.device)
            )
        elif task_class is not None:
            raise ValueError("task_class requires task_class_conditioning=True")
        memory = torch.cat(tokens, dim=1)
        padding = torch.cat(padding_parts, dim=1)
        return self.memory_encoder(memory, src_key_padding_mask=padding), padding

    def forward(
        self,
        per_view_context: torch.Tensor,
        per_view_anchor_tokens: torch.Tensor,
        anchor_valid: torch.Tensor,
        view_valid: torch.Tensor,
        q_corrected_px: torch.Tensor,
        anchor_xyz_history: torch.Tensor,
        state_history: torch.Tensor,
        past_action_chunks: torch.Tensor,
        forced_dynamics_action: torch.Tensor | None = None,
        future_anchor_xyz_baseline: torch.Tensor | None = None,
        goal_state: torch.Tensor | None = None,
        task_class: torch.Tensor | None = None,
    ) -> SO101PredictiveActionOutput:
        batch = self._validate_inputs(
            per_view_context,
            per_view_anchor_tokens,
            anchor_valid,
            view_valid,
            q_corrected_px,
            anchor_xyz_history,
            state_history,
            past_action_chunks,
        )
        memory, padding = self._encode_memory(
            per_view_context,
            per_view_anchor_tokens,
            anchor_valid,
            view_valid,
            q_corrected_px,
            anchor_xyz_history,
            state_history,
            past_action_chunks,
            goal_state=goal_state,
            task_class=task_class,
        )
        query = self.action_queries.expand(batch, -1, -1) + self.action_position
        if self.structured_goal and self.goal_query_residual:
            if goal_state is None:
                raise ValueError("structured_goal requires goal_state [B, state_dim]")
            goal_query = self.goal_projection(goal_state) + self.goal_type_embedding
            query = query + goal_query.unsqueeze(1)
        causal = nn.Transformer.generate_square_subsequent_mask(
            self.action_horizon, device=query.device, dtype=query.dtype
        )
        action_latent = self.action_decoder(
            query,
            memory,
            tgt_mask=causal,
            memory_key_padding_mask=padding,
        )
        delta = torch.tanh(self.delta_head(action_latent)) * self.step_cap
        current = state_history[:, -1]
        actions = []
        for step in range(self.action_horizon):
            current = torch.maximum(
                torch.minimum(current + delta[:, step], self.action_upper),
                self.action_lower,
            )
            actions.append(current)
        action = torch.stack(actions, dim=1)
        action_chunks = action.reshape(
            batch,
            self.future_horizon,
            self.raw_steps_per_chunk * self.state_dim,
        )
        if forced_dynamics_action is None:
            dynamics_action_chunks = action_chunks
        else:
            expected_forced = (batch, self.action_horizon, self.state_dim)
            if tuple(forced_dynamics_action.shape) != expected_forced:
                raise ValueError(
                    f"forced_dynamics_action must have shape {expected_forced}"
                )
            dynamics_action_chunks = forced_dynamics_action.reshape(
                batch,
                self.future_horizon,
                self.raw_steps_per_chunk * self.state_dim,
            )

        if self.dynamics_conditioning == "actor_latent":
            dynamics_state_condition = action_latent[
                :, self.raw_steps_per_chunk - 1 :: self.raw_steps_per_chunk
            ]
        else:
            memory_valid = (~padding).to(dtype=memory.dtype).unsqueeze(-1)
            history_condition = torch.sum(memory * memory_valid, dim=1) / torch.sum(
                memory_valid, dim=1
            ).clamp_min(1.0)
            dynamics_state_condition = history_condition[:, None, :].expand(
                -1, self.future_horizon, -1
            )
        dynamics_input = self.dynamics_input(
            torch.cat(
                (
                    dynamics_state_condition,
                    self.chunk_action_projection(dynamics_action_chunks),
                ),
                dim=-1,
            )
        ) + self.horizon_embedding
        dynamics_mask = nn.Transformer.generate_square_subsequent_mask(
            self.future_horizon,
            device=dynamics_input.device,
            dtype=dynamics_input.dtype,
        )
        dynamics_latent = self.dynamics(dynamics_input, mask=dynamics_mask)

        command_state = dynamics_action_chunks.reshape(
            batch,
            self.future_horizon,
            self.raw_steps_per_chunk,
            self.state_dim,
        )[:, :, -1]
        # A uniform cap gives every joint the same licence to disagree with the
        # command -- 0.1 normalized is 10 degrees, which on the jaw is the whole
        # range that separates closed-on-the-fish from closed-on-nothing. MPC
        # queries this head with jaw ramps no demonstration contains, and on
        # those the residual is free to ignore the command and return the mean
        # aperture, which is exactly what it does. Bounding the disagreement by
        # each joint's own commandable step keeps the head expressing tracking
        # error, the only thing it can honestly know, instead of overwriting the
        # action it was asked about.
        state_residual_cap: torch.Tensor | float = self.state_residual_cap
        if self.state_residual_mode == "step_cap":
            if not bool(self.action_contract_set):
                raise ValueError(
                    "state_residual_mode='step_cap' requires set_action_contract; "
                    "the default unit step_cap would bound the residual at 100 deg"
                )
            state_residual_cap = self.state_residual_steps * self.step_cap.view(
                1, 1, -1
            ).to(dynamics_latent.dtype)
        future_state = command_state + state_residual_cap * torch.tanh(
            self.state_residual_head(dynamics_latent)
        )
        future_state = torch.maximum(
            torch.minimum(future_state, self.action_upper), self.action_lower
        )
        context_delta = self.context_residual_cap * torch.tanh(
            self.context_residual_head(dynamics_latent)
        ).reshape(batch, self.future_horizon, self.num_views, self.latent_dim)
        future_context = per_view_context[:, -1:, :, :] + context_delta
        anchor_delta = self.anchor_residual_cap_m * torch.tanh(
            self.anchor_residual_head(dynamics_latent)
        ).reshape(batch, self.future_horizon, self.num_anchors, 3)
        if self.anchor_baseline_mode == "current":
            if future_anchor_xyz_baseline is not None:
                raise ValueError(
                    "future_anchor_xyz_baseline requires anchor_baseline_mode='external_fk'"
                )
            anchor_base = anchor_xyz_history[:, -1:, :, :].expand(
                -1, self.future_horizon, -1, -1
            )
        else:
            expected_baseline = (
                batch,
                self.future_horizon,
                self.num_anchors,
                3,
            )
            if future_anchor_xyz_baseline is None:
                raise ValueError(
                    "external_fk anchor mode requires future_anchor_xyz_baseline"
                )
            if tuple(future_anchor_xyz_baseline.shape) != expected_baseline:
                raise ValueError(
                    "future_anchor_xyz_baseline must have shape "
                    f"{expected_baseline}"
                )
            if not torch.isfinite(future_anchor_xyz_baseline).all():
                raise ValueError("future_anchor_xyz_baseline must be finite")
            anchor_base = future_anchor_xyz_baseline
        future_anchor_xyz = anchor_base + anchor_delta
        pixel_delta = self.pixel_residual_cap_px * torch.tanh(
            self.pixel_residual_head(dynamics_latent)
        ).reshape(
            batch,
            self.future_horizon,
            self.num_views,
            self.num_anchors,
            2,
        )
        future_anchor_px = q_corrected_px[:, -1:, :, :, :] + pixel_delta
        return SO101PredictiveActionOutput(
            action=action,
            delta=delta,
            action_chunks=action_chunks,
            dynamics_action_chunks=dynamics_action_chunks,
            action_latent=action_latent,
            dynamics_latent=dynamics_latent,
            future_context=future_context,
            future_state=future_state,
            future_anchor_xyz=future_anchor_xyz,
            future_anchor_px=future_anchor_px,
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=values.dtype)
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        weights = weights.expand_as(values)
        return torch.sum(values * weights) / weights.sum().clamp_min(1.0)

    def compute_loss(
        self,
        *,
        per_view_context: torch.Tensor,
        per_view_anchor_tokens: torch.Tensor,
        anchor_valid: torch.Tensor,
        view_valid: torch.Tensor,
        q_corrected_px: torch.Tensor,
        anchor_xyz_history: torch.Tensor,
        state_history: torch.Tensor,
        past_action_chunks: torch.Tensor,
        target_action: torch.Tensor,
        target_action_mask: torch.Tensor,
        target_future_context: torch.Tensor,
        target_future_view_valid: torch.Tensor,
        target_future_state: torch.Tensor,
        target_future_anchor_xyz: torch.Tensor,
        target_future_anchor_px: torch.Tensor,
        target_future_anchor_valid: torch.Tensor,
        velocity_weight: float = 0.2,
        state_weight: float = 1.0,
        anchor_weight: float = 1.0,
        pixel_weight: float = 1.0,
        action_weight: float = 1.0,
        train_action_head: bool = True,
        teacher_force_dynamics: bool = False,
        future_anchor_xyz_baseline: torch.Tensor | None = None,
        goal_state: torch.Tensor | None = None,
        task_class: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], SO101PredictiveActionOutput]:
        output = self(
            per_view_context,
            per_view_anchor_tokens,
            anchor_valid,
            view_valid,
            q_corrected_px,
            anchor_xyz_history,
            state_history,
            past_action_chunks,
            forced_dynamics_action=(
                target_action if teacher_force_dynamics else None
            ),
            future_anchor_xyz_baseline=future_anchor_xyz_baseline,
            goal_state=goal_state,
            task_class=task_class,
        )
        if target_action.shape != output.action.shape:
            raise ValueError("target_action must match output.action")
        if target_action_mask.shape != output.action.shape[:2]:
            raise ValueError("target_action_mask has an invalid shape")
        if target_future_state.shape != output.future_state.shape:
            raise ValueError("target_future_state has an invalid shape")
        if target_future_context.shape != output.future_context.shape:
            raise ValueError("target_future_context has an invalid shape")
        if target_future_view_valid.shape != output.future_context.shape[:-1]:
            raise ValueError("target_future_view_valid has an invalid shape")
        if target_future_anchor_xyz.shape != output.future_anchor_xyz.shape:
            raise ValueError("target_future_anchor_xyz has an invalid shape")
        if target_future_anchor_px.shape != output.future_anchor_px.shape:
            raise ValueError("target_future_anchor_px has an invalid shape")
        if target_future_anchor_valid.shape != output.future_anchor_px.shape[:-1]:
            raise ValueError("target_future_anchor_valid has an invalid shape")

        action_error = F.smooth_l1_loss(
            output.action, target_action, reduction="none"
        )
        action_loss = self._masked_mean(action_error, target_action_mask)
        predicted_velocity = torch.diff(
            torch.cat((state_history[:, -1:, :], output.action), dim=1), dim=1
        )
        target_velocity = torch.diff(
            torch.cat((state_history[:, -1:, :], target_action), dim=1), dim=1
        )
        velocity_loss = self._masked_mean(
            F.smooth_l1_loss(
                predicted_velocity, target_velocity, reduction="none"
            ),
            target_action_mask,
        )
        state_loss = F.smooth_l1_loss(output.future_state, target_future_state)
        context_loss = self._masked_mean(
            F.mse_loss(
                output.future_context,
                target_future_context,
                reduction="none",
            ),
            target_future_view_valid,
        )
        anchor_loss = F.smooth_l1_loss(
            output.future_anchor_xyz / 0.1,
            target_future_anchor_xyz / 0.1,
        )
        pixel_loss = self._masked_mean(
            F.smooth_l1_loss(
                output.future_anchor_px / 32.0,
                target_future_anchor_px / 32.0,
                reduction="none",
            ),
            target_future_anchor_valid,
        )
        action_objective = (
            float(action_weight)
            * (action_loss + float(velocity_weight) * velocity_loss)
            if train_action_head
            else action_loss.new_zeros(())
        )
        loss = (
            action_objective
            + context_loss
            + float(state_weight) * state_loss
            + float(anchor_weight) * anchor_loss
            + float(pixel_weight) * pixel_loss
        )
        parts = {
            "action_loss": action_loss,
            "velocity_loss": velocity_loss,
            "state_loss": state_loss,
            "context_loss": context_loss,
            "anchor_loss": anchor_loss,
            "pixel_loss": pixel_loss,
            "action_objective": action_objective,
        }
        return loss, parts, output


__all__ = ["SO101PredictiveActionOutput", "SO101PredictiveActionPolicy"]
