"""Task-conditioned soft object tokens for Prod-V17.8.1.

This module is deliberately only a compact readout head.  It consumes the
frozen dual-view DINO patch tokens already produced by the visual backbone and
a frozen task-text embedding.  It does not consume robot state, pixel labels,
manually supplied regions, or controller outputs.

The readout contains the ordered ``pickup`` and ``drop`` task roles.  Earlier
drafts also exposed anonymous object hypotheses, but they had no downstream
loss or control consumer and were removed rather than retaining dead weights.

Every token is a soft weighted average of the original DINO patch tokens.  The
attention tensor returned to the caller is also the tensor intended for the
dashboard; no second localization path is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.prod_v17_8 import (
    TASK_EMBEDDING_DIM,
    VIEWS,
    VIT_DIM,
    VIT_PATCHES,
)


OBJECT_ROLES: tuple[str, str] = ("pickup", "drop")
OBJECT_ROLE_COUNT: int = len(OBJECT_ROLES)


class V1781DinoContractError(ValueError):
    """Raised when an input violates the soft-object-token contract."""


@dataclass(frozen=True)
class V1781DinoObjectOutput:
    """Soft DINO object readouts.

    Valid-view attention is normalized across all 256 patches.  Invalid-view
    attention and tokens are exactly zero.  ``dashboard_attention_maps`` is a
    shape-only view of ``role_attention``, so visualization and control cannot
    silently diverge.
    """

    role_tokens: torch.Tensor
    role_attention: torch.Tensor
    view_valid: torch.Tensor

    @property
    def dashboard_attention_maps(self) -> torch.Tensor:
        """Return pickup/drop attention as ``[B,2,2,16,16]``."""

        side = math.isqrt(VIT_PATCHES)
        if side * side != VIT_PATCHES:  # pragma: no cover - fixed contract
            raise V1781DinoContractError(
                "DINO patch count must form a square dashboard grid"
            )
        return self.role_attention.reshape(
            self.role_attention.shape[0],
            VIEWS,
            OBJECT_ROLE_COUNT,
            side,
            side,
        )

    @property
    def pickup_tokens(self) -> torch.Tensor:
        """Pickup-role tokens with shape ``[B,2,384]``."""

        return self.role_tokens[:, :, 0]

    @property
    def drop_tokens(self) -> torch.Tensor:
        """Drop-role tokens with shape ``[B,2,384]``."""

        return self.role_tokens[:, :, 1]


class V1781TaskConditionedDinoObjectHead(nn.Module):
    """Pool dual-view DINO patches into task-role object tokens.

    The spatial readout is fully differentiable.  There is no maximum-cell
    selection, box proposal, or hard spatial value in either the input or the
    output contract.
    """

    def __init__(
        self,
        *,
        query_dim: int = 192,
        temperature: float = 0.25,
    ) -> None:
        super().__init__()
        if query_dim <= 0:
            raise V1781DinoContractError("query_dim must be positive")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise V1781DinoContractError(
                "temperature must be finite and positive"
            )

        self.query_dim = int(query_dim)
        self.temperature = float(temperature)
        self.scene_norm = nn.LayerNorm(VIT_DIM)
        self.task_norm = nn.LayerNorm(TASK_EMBEDDING_DIM)
        self.key_projection = nn.Linear(VIT_DIM, self.query_dim, bias=False)
        self.task_projection = nn.Linear(
            TASK_EMBEDDING_DIM, self.query_dim, bias=False
        )
        self.role_queries = nn.Parameter(
            torch.empty(OBJECT_ROLE_COUNT, self.query_dim)
        )
        nn.init.normal_(self.role_queries, std=0.02)

    @staticmethod
    def _validate_inputs(
        scene_tokens: torch.Tensor,
        task_embedding: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        if scene_tokens.ndim != 4:
            raise V1781DinoContractError(
                "scene_tokens must have shape [B,2,256,384]"
            )
        batch = int(scene_tokens.shape[0])
        if tuple(scene_tokens.shape) != (
            batch,
            VIEWS,
            VIT_PATCHES,
            VIT_DIM,
        ):
            raise V1781DinoContractError(
                "scene_tokens must have shape [B,2,256,384]"
            )
        if tuple(task_embedding.shape) != (batch, TASK_EMBEDDING_DIM):
            raise V1781DinoContractError(
                "task_embedding must have shape [B,384]"
            )
        if tuple(view_valid.shape) != (batch, VIEWS):
            raise V1781DinoContractError(
                "view_valid must have shape [B,2]"
            )
        if view_valid.dtype != torch.bool:
            raise V1781DinoContractError("view_valid must be boolean")
        if not scene_tokens.is_floating_point():
            raise V1781DinoContractError("scene_tokens must be floating point")
        if not task_embedding.is_floating_point():
            raise V1781DinoContractError(
                "task_embedding must be floating point"
            )
        if scene_tokens.device != task_embedding.device:
            raise V1781DinoContractError(
                "scene_tokens and task_embedding must share a device"
            )
        if scene_tokens.device != view_valid.device:
            raise V1781DinoContractError(
                "scene_tokens and view_valid must share a device"
            )
        if not torch.isfinite(task_embedding).all():
            raise V1781DinoContractError("task_embedding must be finite")

        valid_mask = view_valid[:, :, None, None]
        sanitized_scene = torch.where(
            valid_mask, scene_tokens, torch.zeros_like(scene_tokens)
        )
        if not torch.isfinite(sanitized_scene).all():
            raise V1781DinoContractError(
                "scene_tokens must be finite in every valid view"
            )
        return batch, sanitized_scene

    def _attention(
        self,
        keys: torch.Tensor,
        task_query: torch.Tensor,
        learned_queries: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> torch.Tensor:
        queries = F.normalize(
            task_query[:, None, :] + learned_queries[None, :, :],
            dim=-1,
            eps=1.0e-6,
        )
        logits = torch.einsum("bvpd,bhd->bvhp", keys, queries)
        attention = torch.softmax(logits / self.temperature, dim=-1)
        attention = torch.where(
            view_valid[:, :, None, None],
            attention,
            torch.zeros_like(attention),
        )
        return attention

    def forward(
        self,
        scene_tokens: torch.Tensor,
        task_embedding: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> V1781DinoObjectOutput:
        """Return soft pickup/drop readouts for both views."""

        _batch, sanitized_scene = self._validate_inputs(
            scene_tokens, task_embedding, view_valid
        )
        keys = F.normalize(
            self.key_projection(self.scene_norm(sanitized_scene)),
            dim=-1,
            eps=1.0e-6,
        )
        task_query = self.task_projection(self.task_norm(task_embedding))

        role_attention = self._attention(
            keys, task_query, self.role_queries, view_valid
        )
        role_tokens = torch.einsum(
            "bvrp,bvpd->bvrd", role_attention, sanitized_scene
        )

        # Keep the invalid-view contract exact even under unusual reduced
        # precision kernels.  Valid outputs are finite because both inputs and
        # softmax logits were checked or constructed from finite tensors.
        token_mask = view_valid[:, :, None, None]
        role_tokens = torch.where(
            token_mask, role_tokens, torch.zeros_like(role_tokens)
        )

        outputs = (
            role_attention,
            role_tokens,
        )
        if not all(torch.isfinite(value).all() for value in outputs):
            raise V1781DinoContractError("soft object output must be finite")

        return V1781DinoObjectOutput(
            role_tokens=role_tokens,
            role_attention=role_attention,
            view_valid=view_valid,
        )


__all__ = [
    "OBJECT_ROLES",
    "OBJECT_ROLE_COUNT",
    "V1781DinoContractError",
    "V1781DinoObjectOutput",
    "V1781TaskConditionedDinoObjectHead",
]
