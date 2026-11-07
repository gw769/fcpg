"""Protected visual evidence memory for query-initialized FCPG-H v3.

The immutable first-frame query and the current frozen V12.4 patch map always
form the base measurement.  Past visual frames can only add a bounded logit
correction.  They never read state, action, CAD, camera, source, task, or a
LeWM posterior.  Historical evidence is extracted independently per frame and
retained as a set.  Each current patch receives the strongest compatible
historical score, so permuting the causal-past set is exactly invariant.

The matched H0 intervention executes the same projections, sparse matching,
set reduction, and bounded gain, but replaces every historical frame with a
query-only dummy frame.  Thus both arms have identical parameters and tensor
shapes; only factual past visual evidence differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fcpg.query_conditioned_observer import (
    MAX_EVENT_FRAMES,
    MEASUREMENT_TEMPERATURE,
    PATCH_GRID,
    TIP_COUNT,
    TOKEN_DIM,
    QueryConditionedMeasurement,
    _sample_current_anchors,
    assert_parameter_limit,
    decode_unique_tip_pixels,
)


METHOD_REVISION = "fcpg-h-protected-visual-evidence-memory-v3.1"
CHECKPOINT_FORMAT = "so101-fcpg-h-protected-memory-checkpoint-v3.1"
H0_CHECKPOINT_FORMAT = "so101-fcpg-h-protected-memory-h0-checkpoint-v3.1"
PAIR_RECEIPT_FORMAT = "so101-fcpg-h-protected-memory-paired-canary-receipt-v3.1.1"
HISTORY_GRID = 8
HISTORY_TOPK = 4
HISTORY_LOGIT_BOUND = 0.25
PROTECTED_MEMORY_PARAMETER_LIMIT = 500_000


@dataclass(frozen=True)
class ProtectedMemoryOutput:
    """Endpoint measurement plus native-grid prefix diagnostics."""

    measurement: QueryConditionedMeasurement
    intermediate_logits: torch.Tensor
    base_endpoint_logits: torch.Tensor
    endpoint_history_correction: torch.Tensor
    endpoint_evidence_count: torch.Tensor


def _endpoint_rows(time_valid: torch.Tensor) -> torch.Tensor:
    return time_valid.long().sum(dim=1) - 1


def _grid_centres(
    indices: torch.Tensor,
    *,
    grid: int,
    image_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    width, height = image_size
    y = torch.div(indices, grid, rounding_mode="floor")
    x = indices.remainder(grid)
    px = (x.float() + 0.5) * (float(width) / grid) - 0.5
    py = (y.float() + 0.5) * (float(height) / grid) - 0.5
    return torch.stack((px, py), dim=-1)


def decode_independent_tip_pixels(
    logits: torch.Tensor,
    *,
    image_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Decode each query independently for slot-level canary metrics."""

    if logits.ndim != 5 or logits.shape[2:] != (
        TIP_COUNT,
        PATCH_GRID,
        PATCH_GRID,
    ):
        raise ValueError("endpoint logits must be [B,V,2,16,16]")
    indices = logits.float().flatten(-2).argmax(dim=-1)
    return _grid_centres(indices, grid=PATCH_GRID, image_size=image_size).to(
        logits.dtype
    )


class ProtectedVisualEvidenceMemory(nn.Module):
    """Current-image measurement with a bounded visual-history correction."""

    def __init__(
        self,
        *,
        dimension: int = TOKEN_DIM,
        temperature: float = MEASUREMENT_TEMPERATURE,
        history_topk: int = HISTORY_TOPK,
        history_logit_bound: float = HISTORY_LOGIT_BOUND,
    ) -> None:
        super().__init__()
        if dimension != TOKEN_DIM:
            raise ValueError("protected memory is locked to V12.4 192-D tokens")
        if temperature != MEASUREMENT_TEMPERATURE:
            raise ValueError("measurement temperature is frozen at 0.10")
        if not 1 <= history_topk <= HISTORY_GRID * HISTORY_GRID:
            raise ValueError("history_topk must lie in [1,64]")
        if history_logit_bound <= 0.0:
            raise ValueError("history_logit_bound must be positive")
        self.dimension = int(dimension)
        self.temperature = float(temperature)
        self.history_topk = int(history_topk)
        self.history_logit_bound = float(history_logit_bound)

        # Current and history projections are deliberately separate.  The H0
        # dummy-history gradient therefore cannot alter the current base path
        # through a shared history projection.
        self.current_projection = nn.Linear(TOKEN_DIM, dimension)
        self.history_projection = nn.Linear(TOKEN_DIM, dimension)
        self.query_projection = nn.Linear(TOKEN_DIM, dimension)
        self.tip_identity = nn.Embedding(TIP_COUNT, dimension)
        self.current_norm = nn.LayerNorm(dimension)
        self.history_norm = nn.LayerNorm(dimension)
        self.query_norm = nn.LayerNorm(dimension)
        # A single per-identity scalar controls a bounded set-max correction.
        # Zero initialization makes both interventions exactly current-only at
        # step zero without introducing another learned head.
        self.history_gain = nn.Parameter(torch.zeros(TIP_COUNT))
        assert_parameter_limit(
            self,
            maximum=PROTECTED_MEMORY_PARAMETER_LIMIT,
            name="protected visual evidence memory",
        )

    @staticmethod
    def _validate(
        *,
        pooled_history: torch.Tensor,
        current_patch_maps: torch.Tensor,
        initial_query_tokens: torch.Tensor,
        time_valid: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> tuple[int, int, int, torch.Tensor]:
        if pooled_history.ndim != 6 or pooled_history.shape[3:] != (
            TOKEN_DIM,
            HISTORY_GRID,
            HISTORY_GRID,
        ):
            raise ValueError("pooled_history must be [B,T,V,192,8,8]")
        batch, time, views = pooled_history.shape[:3]
        if not 2 <= time <= MAX_EVENT_FRAMES:
            raise ValueError("protected memory requires 2..16 causal frames")
        if current_patch_maps.shape != (
            batch,
            views,
            TOKEN_DIM,
            PATCH_GRID,
            PATCH_GRID,
        ):
            raise ValueError("current_patch_maps must be [B,V,192,16,16]")
        if initial_query_tokens.shape != (batch, views, TIP_COUNT, TOKEN_DIM):
            raise ValueError("initial_query_tokens must be [B,V,2,192]")
        if time_valid.shape != (batch, time) or time_valid.dtype != torch.bool:
            raise ValueError("time_valid must be bool [B,T]")
        if view_valid.shape != (batch, views) or view_valid.dtype != torch.bool:
            raise ValueError("view_valid must be bool [B,V]")
        device = current_patch_maps.device
        values = (pooled_history, initial_query_tokens)
        if any(value.device != device for value in (*values, time_valid, view_valid)):
            raise ValueError("all protected-memory inputs must share one device")
        if not all(value.is_floating_point() for value in values):
            raise TypeError("history/query inputs must be floating point")
        if not current_patch_maps.is_floating_point():
            raise TypeError("current patch maps must be floating point")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (*values, current_patch_maps)
        ):
            raise ValueError("visual inputs must be finite")
        if not bool(time_valid[:, 0].all()) or bool(
            (time_valid[:, 1:] & ~time_valid[:, :-1]).any()
        ):
            raise ValueError("time_valid must be a contiguous valid prefix")
        endpoint = _endpoint_rows(time_valid)
        if bool((endpoint < 1).any()):
            raise ValueError("query and endpoint frames must be distinct")
        return batch, time, views, endpoint

    def _query_slots(self, initial_query_tokens: torch.Tensor) -> torch.Tensor:
        slots = self.query_projection(initial_query_tokens.detach())
        slots = slots + self.tip_identity.weight[None, None]
        return self.query_norm(slots)

    def _raw_history(
        self,
        pooled_history: torch.Tensor,
        initial_query_tokens: torch.Tensor,
        *,
        h0: bool,
    ) -> torch.Tensor:
        batch, time, views = pooled_history.shape[:3]
        if not h0:
            return pooled_history.detach().permute(0, 1, 2, 4, 5, 3).reshape(
                batch,
                time,
                views,
                HISTORY_GRID * HISTORY_GRID,
                TOKEN_DIM,
            )
        # Same [B,T,V,64,192] work, but no factual history value.  Alternating
        # copies preserve both supplied queries without creating a zero-token
        # shortcut.
        query = initial_query_tokens.detach()
        tip_index = torch.arange(
            HISTORY_GRID * HISTORY_GRID,
            device=query.device,
        ).remainder(TIP_COUNT)
        dummy = query[:, :, tip_index]
        return dummy[:, None].expand(-1, time, -1, -1, -1)

    def _frame_evidence(
        self,
        pooled_history: torch.Tensor,
        initial_query_tokens: torch.Tensor,
        query_slots: torch.Tensor,
        *,
        h0: bool,
    ) -> torch.Tensor:
        raw = self._raw_history(
            pooled_history,
            initial_query_tokens,
            h0=h0,
        )
        history = self.history_norm(self.history_projection(raw))
        query = F.normalize(query_slots.float(), dim=-1)
        key = F.normalize(history.float(), dim=-1)
        similarity = torch.einsum("bvkd,btvpd->btvkp", query, key)
        top_value, top_index = similarity.topk(self.history_topk, dim=-1)
        values: list[torch.Tensor] = []
        for tip in range(TIP_COUNT):
            index = top_index[:, :, :, tip]
            gathered = torch.gather(
                history,
                3,
                index[..., None].expand(-1, -1, -1, -1, self.dimension),
            )
            weight = (top_value[:, :, :, tip] / self.temperature).softmax(dim=-1)
            values.append((gathered.float() * weight[..., None]).sum(dim=-2))
        return torch.stack(values, dim=3)

    @staticmethod
    def _legal_evidence(
        *,
        time_valid: torch.Tensor,
        view_valid: torch.Tensor,
        endpoint: torch.Tensor,
    ) -> torch.Tensor:
        time = time_valid.shape[1]
        step = torch.arange(time, device=time_valid.device)[None]
        # Query frame, endpoint, and suffix padding can never be evidence.
        legal = (
            time_valid
            & (step >= 1)
            & (step < endpoint[:, None])
        )[:, :, None]
        return legal & view_valid[:, None]

    def _project_current(
        self,
        current_maps: torch.Tensor,
        query_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], int, int]:
        grid = current_maps.shape[-1]
        if current_maps.shape[-2] != grid or grid not in (HISTORY_GRID, PATCH_GRID):
            raise ValueError("current maps must use an 8x8 or 16x16 square grid")
        prefix = current_maps.shape[:-4]
        views = current_maps.shape[-4]
        raw = current_maps.detach().permute(*range(len(prefix)), -4, -2, -1, -3)
        current = self.current_norm(self.current_projection(raw))
        current = current.reshape(
            *prefix,
            views,
            grid * grid,
            self.dimension,
        )
        current_norm = F.normalize(current.float(), dim=-1)
        query_norm = F.normalize(query_slots.float(), dim=-1)
        base = torch.einsum("...vpd,...vkd->...vkp", current_norm, query_norm)
        base = base / self.temperature
        return current_norm, base, prefix, views, grid

    def _intermediate_scores(
        self,
        current_maps: torch.Tensor,
        query_slots: torch.Tensor,
        frame_evidence: torch.Tensor,
        legal_evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current, base, prefix, views, grid = self._project_current(
            current_maps,
            query_slots,
        )
        if len(prefix) != 2:
            raise ValueError("intermediate current maps must be [B,T,V,192,G,G]")
        batch, time = prefix
        evidence = F.normalize(frame_evidence.float(), dim=-1)
        similarity = torch.einsum(
            "btvpd,bsvkd->btsvkp",
            current,
            evidence,
        ) / self.temperature
        current_step = torch.arange(time, device=current.device)[None, :, None]
        memory_step = torch.arange(time, device=current.device)[None, None, :]
        prefix_legal = memory_step < current_step
        mask = prefix_legal[:, :, :, None] & legal_evidence[:, None]
        minimum = -torch.finfo(similarity.dtype).max
        history_score = similarity.masked_fill(
            ~mask[..., None, None],
            minimum,
        ).amax(dim=2)
        has_history = mask.any(dim=2)
        raw = torch.tanh(history_score - base)
        gain = torch.tanh(self.history_gain)[None, None, None, :, None]
        correction = self.history_logit_bound * gain * raw
        correction = correction * has_history[..., None, None]
        logits = base + correction
        return (
            logits.reshape(batch, time, views, TIP_COUNT, grid, grid),
            correction.reshape(batch, time, views, TIP_COUNT, grid, grid),
        )

    def _endpoint_scores(
        self,
        current_maps: torch.Tensor,
        query_slots: torch.Tensor,
        frame_evidence: torch.Tensor,
        legal_evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current, base, prefix, views, grid = self._project_current(
            current_maps,
            query_slots,
        )
        if len(prefix) != 1:
            raise ValueError("endpoint current maps must be [B,V,192,G,G]")
        batch = prefix[0]
        evidence = F.normalize(frame_evidence.float(), dim=-1)
        similarity = torch.einsum(
            "bvpd,bsvkd->bsvkp",
            current,
            evidence,
        ) / self.temperature
        minimum = -torch.finfo(similarity.dtype).max
        history_score = similarity.masked_fill(
            ~legal_evidence[..., None, None],
            minimum,
        ).amax(dim=1)
        has_history = legal_evidence.any(dim=1)
        raw = torch.tanh(history_score - base)
        gain = torch.tanh(self.history_gain)[None, None, :, None]
        correction = self.history_logit_bound * gain * raw
        correction = correction * has_history[..., None, None]
        logits = base + correction
        return (
            logits.reshape(batch, views, TIP_COUNT, grid, grid),
            base.reshape(batch, views, TIP_COUNT, grid, grid),
            correction.reshape(batch, views, TIP_COUNT, grid, grid),
        )

    def forward_cached(
        self,
        *,
        pooled_history: torch.Tensor,
        current_patch_maps: torch.Tensor,
        initial_query_tokens: torch.Tensor,
        time_valid: torch.Tensor,
        view_valid: torch.Tensor,
        displacement_progress: torch.Tensor | None = None,
        h0: bool = False,
        image_size: tuple[int, int] = (224, 224),
    ) -> ProtectedMemoryOutput:
        # Compatibility-only scalar: visual memory never reads physical motion.
        del displacement_progress
        batch, time, views, endpoint = self._validate(
            pooled_history=pooled_history,
            current_patch_maps=current_patch_maps,
            initial_query_tokens=initial_query_tokens,
            time_valid=time_valid,
            view_valid=view_valid,
        )
        query_slots = self._query_slots(initial_query_tokens)
        frame_evidence = self._frame_evidence(
            pooled_history,
            initial_query_tokens,
            query_slots,
            h0=h0,
        )
        legal_evidence = self._legal_evidence(
            time_valid=time_valid,
            view_valid=view_valid,
            endpoint=endpoint,
        )

        query_at_time = query_slots[:, None].expand(-1, time, -1, -1, -1)
        intermediate_logits, _ = self._intermediate_scores(
            pooled_history,
            query_at_time,
            frame_evidence,
            legal_evidence,
        )

        endpoint_logits, base_logits, correction = self._endpoint_scores(
            current_patch_maps,
            query_slots,
            frame_evidence,
            legal_evidence,
        )
        endpoint_count = legal_evidence.long().sum(dim=1)
        view_mask = view_valid[:, :, None, None, None]
        # Autocast may leave base/intermediate logits in bf16 while the
        # correction promotes endpoint logits to float32.  Negative infinity
        # is the exact invalid-candidate value and is representable in both.
        minimum = float("-inf")
        endpoint_logits = endpoint_logits.masked_fill(~view_mask, minimum)
        base_logits = base_logits.masked_fill(~view_mask, minimum)
        correction = correction * view_mask
        intermediate_view_mask = view_mask[:, None]
        intermediate_logits = intermediate_logits.masked_fill(
            ~intermediate_view_mask,
            minimum,
        )
        heatmaps = endpoint_logits.flatten(-2).softmax(dim=-1).reshape_as(
            endpoint_logits
        )
        heatmaps = heatmaps * view_mask
        tip_px = decode_unique_tip_pixels(endpoint_logits, image_size=image_size)
        tip_px = tip_px * view_valid[:, :, None, None]
        anchor_px, part_tokens, anchor_tokens = _sample_current_anchors(
            current_patch_maps.detach(),
            tip_px,
            image_size=image_size,
        )
        token_mask = view_valid[:, :, None, None]
        measurement = QueryConditionedMeasurement(
            measurement_logits=endpoint_logits,
            heatmaps=heatmaps,
            tip_px=tip_px,
            anchor_px=anchor_px * token_mask,
            part_tokens=part_tokens * token_mask,
            anchor_tokens=anchor_tokens * token_mask,
            visual_slots=query_slots * token_mask,
            endpoint_index=endpoint,
        )
        return ProtectedMemoryOutput(
            measurement=measurement,
            intermediate_logits=intermediate_logits,
            base_endpoint_logits=base_logits,
            endpoint_history_correction=correction,
            endpoint_evidence_count=endpoint_count,
        )


def forward_h0_cached(
    model: ProtectedVisualEvidenceMemory,
    **kwargs: torch.Tensor | tuple[int, int] | None,
) -> ProtectedMemoryOutput:
    """Matched no-factual-history intervention."""

    return model.forward_cached(**kwargs, h0=True)


__all__ = [
    "CHECKPOINT_FORMAT",
    "H0_CHECKPOINT_FORMAT",
    "HISTORY_GRID",
    "HISTORY_LOGIT_BOUND",
    "HISTORY_TOPK",
    "METHOD_REVISION",
    "PAIR_RECEIPT_FORMAT",
    "PROTECTED_MEMORY_PARAMETER_LIMIT",
    "ProtectedMemoryOutput",
    "ProtectedVisualEvidenceMemory",
    "decode_independent_tip_pixels",
    "forward_h0_cached",
]
