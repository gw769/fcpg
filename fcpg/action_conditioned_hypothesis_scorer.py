"""Action-conditioned hypothesis scorer for DICG candidate screening.

The student sees real consecutive frames, the executed ``q`` / ``Δq``, and CAD
projected dual-tip trajectories.  It predicts a *motion innovation* (observed
flow minus the CAD displacement), tip observability logits, and a per-hypothesis
consistency score used only to rank CAD hypotheses.

This is not an absolute tip localizer and is not a physical residual estimator.
Outputs must not be reported as calibrated uncertainty, live-ready grounding, or
millimetre tip accuracy.  LeWM / actor / structured-goal fields never enter the
loss.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from fcpg.dicg_test_seal import (
    DicgTestSealError,
    refuse_associated_motion_session,
    refuse_sealed_test_identity,
)
from fcpg.modules import sample_feature_map_at_pixels
from v10.supervision_segregation import (
    FORBIDDEN_LEWM_SUPERVISION,
    ACTOR_CAUSAL_PART_KEYS,
    lewm_actor_supervision_contract,
)


SCHEMA_NAME = "fcpg.dicg_motion_dataset.v1"
CHECKPOINT_FORMAT = "fcpg.action_conditioned_hypothesis_scorer.v1"
TRAINING_TARGET = "real_motion_hypothesis_consistency"

IMAGE_SIZE_PX = (224, 224)
VIT_TOKEN_DIM = 192
VIT_PATCH_GRID = 16
VIT_TOKEN_COUNT = 1 + VIT_PATCH_GRID * VIT_PATCH_GRID  # CLS + 16x16
JOINT_DIM = 6
N_TIPS = 2
N_XY = 2
HYPOTHESIS_XYZ_DIM = 3

SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2
SPLIT_NAMES = {SPLIT_TRAIN: "train", SPLIT_VAL: "val", SPLIT_TEST: "test"}
ALL_MOTION_SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)
SEALED_TEST_TOKEN = "test_sealed"

TIP_INDEX_LEFT = 0
TIP_INDEX_RIGHT = 1

DEFAULT_MIN_VALID_RATIO = 0.05
DEFAULT_MIN_EXCITATION_PX = 1.0
DEFAULT_ACCEPT_THRESHOLD = 0.5
DEFAULT_RANKING_TEMPERATURE = 4.0
DEFAULT_PAIRWISE_TIE_TOL_PX = 1.0e-3
RANKING_LISTWISE = "listwise"
RANKING_PAIRWISE = "pairwise"
RANKING_PAIRWISE_TOP1_VS_REST = "pairwise_top1_vs_rest"
RANKING_MODES = (
    RANKING_LISTWISE,
    RANKING_PAIRWISE,
    RANKING_PAIRWISE_TOP1_VS_REST,
)
PAIRWISE_RANKING_MODES = frozenset(
    {RANKING_PAIRWISE, RANKING_PAIRWISE_TOP1_VS_REST}
)
MIN_RANKABLE = 2

REQUIRED_ARRAY_KEYS = (
    "image_t_path",
    "image_tp1_path",
    "q_t",
    "q_tp1",
    "delta_q",
    "hypotheses_tool_offsets_m",
    "candidate_t_px",
    "candidate_tp1_px",
    "observed_flow_px",
    "motion_residual_target_px",
    "tip_valid",
    "hypothesis_valid",
    "excitation_px",
    "target_error_px",
    "split",
)
REQUIRED_UNICODE_KEYS = ("schema", "provenance")
OPTIONAL_UNICODE_KEYS = (
    "episode_id",
    "round_id",
    "source_id",
    "camera_id",
    "session_id",
    "session",
    "epoch_id",
    "build_id",
    "notes",
)
ROW_UNICODE_KEYS = frozenset(
    {
        "episode_id",
        "round_id",
        "source_id",
        "camera_id",
        "session_id",
        "session",
        "epoch_id",
    }
)

FORBIDDEN_DATASET_KEYS = frozenset(
    {
        "goal",
        "structured_goal",
        "goal_delta",
        "pred_delta",
        "gt_delta",
        "action_delta_causal",
        "dynamics_latent",
        "lewm_target",
        "lewm_future",
        "actor_action",
        "q_true_px",
        "absolute_tip_px",
        "physical_tip_m",
    }
)
FORBIDDEN_LOSS_KEYS = frozenset(FORBIDDEN_LEWM_SUPERVISION) | frozenset(
    ACTOR_CAUSAL_PART_KEYS
) | frozenset(
    {
        "goal",
        "structured_goal",
        "goal_sensitive",
        "actor_action_delta",
        "pred_delta",
        "gt_delta",
        "lewm_dynamics",
        "state_loss",
        "context_loss",
        "anchor_loss",
        "pixel_loss",
        "sigreg",
        "lewm_regression",
    }
)
FORBIDDEN_OUTPUT_FIELDS = frozenset(
    {
        "q_corrected_px",
        "p_hat_px",
        "absolute_tip_px",
        "tip_xy_px",
        "physical_residual_m",
        "local_physical_residual",
    }
)

ALLOWED_LOSS_KEYS = frozenset(
    {
        "total",
        "motion_residual",
        "tip_validity",
        "ranking",
        "n_rankable_samples",
        "n_motion_elements",
        "ranking_fail_closed",
    }
)


class DICGMotionDatasetError(ValueError):
    """NPZ contract, split, or path check failed.  Fail closed."""


class HypothesisScorerError(ValueError):
    """Model, loss, or checkpoint contract failed.  Fail closed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vit_tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
    """Drop CLS and reshape FrozenV9ViT tokens to ``[B, 192, 16, 16]``."""

    if tokens.ndim != 3 or tokens.shape[1] != VIT_TOKEN_COUNT or tokens.shape[2] != VIT_TOKEN_DIM:
        raise HypothesisScorerError(
            "FrozenV9ViT must return CLS + 16x16 tokens "
            f"[B,{VIT_TOKEN_COUNT},{VIT_TOKEN_DIM}], got {tuple(tokens.shape)}"
        )
    batch = int(tokens.shape[0])
    patches = tokens[:, 1:, :]
    return (
        patches.reshape(batch, VIT_PATCH_GRID, VIT_PATCH_GRID, VIT_TOKEN_DIM)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def select_amp_dtype(
    device: torch.device | str,
    *,
    capability: tuple[int, int] | None = None,
    device_name: str | None = None,
) -> torch.dtype | None:
    """CUDA mixed precision is FP16 + GradScaler only.  Never bf16, including V100."""

    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    name = device_name
    cap = capability
    if cap is None or name is None:
        if torch.cuda.is_available():
            index = dev.index if dev.index is not None else 0
            if cap is None:
                cap = tuple(torch.cuda.get_device_capability(index))
            if name is None:
                name = torch.cuda.get_device_name(index)
    name_l = str(name or "").lower()
    major, minor = cap if cap is not None else (7, 0)
    # V100 is sm_70.  bf16 is not a Tensor-Core type there and is forbidden here
    # on every CUDA device, including later GPUs.
    _ = (major, minor, name_l)
    return torch.float16


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    module.requires_grad_(False)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def assert_module_frozen(module: nn.Module, *, name: str = "encoder") -> None:
    if module.training:
        raise HypothesisScorerError(f"{name} must stay in eval mode")
    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    if trainable:
        raise HypothesisScorerError(f"{name} must stay frozen, trainable={trainable[:4]}")


def gradients_are_finite(module: nn.Module) -> bool:
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            return False
    return True


@dataclass(frozen=True)
class HypothesisScoreOutput:
    """Per-candidate motion innovation and uncalibrated consistency scores.

    ``uncalibrated_scale`` is a free positive magnitude.  It is not an ECE /
    NLL-calibrated sigma and must not be advertised as one.
    """

    motion_innovation_px: torch.Tensor
    tip_validity_logits: torch.Tensor
    hypothesis_logits: torch.Tensor
    uncalibrated_scale: torch.Tensor

    @property
    def consistency_score(self) -> torch.Tensor:
        return self.hypothesis_logits

    @property
    def tip_observability_logits(self) -> torch.Tensor:
        return self.tip_validity_logits

    def float(self) -> "HypothesisScoreOutput":
        return HypothesisScoreOutput(
            motion_innovation_px=self.motion_innovation_px.float(),
            tip_validity_logits=self.tip_validity_logits.float(),
            hypothesis_logits=self.hypothesis_logits.float(),
            uncalibrated_scale=self.uncalibrated_scale.float(),
        )

    def field_names(self) -> tuple[str, ...]:
        return (
            "motion_innovation_px",
            "tip_validity_logits",
            "hypothesis_logits",
            "uncalibrated_scale",
        )


def assert_output_has_no_absolute_tip(output: HypothesisScoreOutput) -> None:
    leaked = FORBIDDEN_OUTPUT_FIELDS.intersection(output.field_names())
    if leaked:
        raise HypothesisScorerError(
            f"absolute tip / physical residual fields are forbidden: {sorted(leaked)}"
        )


@dataclass
class ScorerLoss:
    total: torch.Tensor
    motion_residual: torch.Tensor
    tip_validity: torch.Tensor
    ranking: torch.Tensor
    n_rankable_samples: torch.Tensor
    n_motion_elements: torch.Tensor
    ranking_fail_closed: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "total": self.total,
            "motion_residual": self.motion_residual,
            "tip_validity": self.tip_validity,
            "ranking": self.ranking,
            "n_rankable_samples": self.n_rankable_samples,
            "n_motion_elements": self.n_motion_elements,
            "ranking_fail_closed": self.ranking_fail_closed,
        }


def assert_loss_parts_isolated(parts: Mapping[str, object]) -> None:
    leaked = FORBIDDEN_LOSS_KEYS.intersection(parts)
    if leaked:
        raise HypothesisScorerError(
            "goal / actor action-delta / LeWM dynamics keys are forbidden in "
            f"scorer loss: {sorted(leaked)}"
        )
    unknown = set(parts) - ALLOWED_LOSS_KEYS
    if unknown:
        raise HypothesisScorerError(f"unexpected scorer loss keys: {sorted(unknown)}")


def sample_tip_local_features(
    feature_map: torch.Tensor,
    pixels_xy: torch.Tensor,
    *,
    image_size: tuple[int, int] = IMAGE_SIZE_PX,
    grid_size: int = 3,
    radius_px: float = 14.0,
) -> torch.Tensor:
    """Bilinear-sample a centre or odd ``grid_size`` neighbourhood per tip.

    Returns ``[B, N, 2, K, C]``.
    """

    if feature_map.ndim != 4:
        raise HypothesisScorerError(
            f"feature_map must be [B,C,H,W], got {tuple(feature_map.shape)}"
        )
    if pixels_xy.ndim != 4 or pixels_xy.shape[-2:] != (N_TIPS, N_XY):
        raise HypothesisScorerError(
            f"pixels_xy must be [B,N,2,2], got {tuple(pixels_xy.shape)}"
        )
    if int(grid_size) != 1 and (int(grid_size) < 1 or int(grid_size) % 2 == 0):
        raise HypothesisScorerError("grid_size must be 1 or a positive odd integer")
    batch, n_hyp, n_tips, _ = pixels_xy.shape
    flat = pixels_xy.reshape(batch, n_hyp * n_tips, 2)
    channels = int(feature_map.shape[1])
    if int(grid_size) == 1:
        sampled = sample_feature_map_at_pixels(
            feature_map, flat, image_size=image_size, padding_mode="border"
        )
        tokens = sampled.unsqueeze(2)
    else:
        offsets_1d = torch.linspace(
            -float(radius_px),
            float(radius_px),
            int(grid_size),
            device=pixels_xy.device,
            dtype=pixels_xy.dtype,
        )
        offset_y, offset_x = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
        offsets = torch.stack((offset_x, offset_y), dim=-1).reshape(1, 1, -1, 2)
        points = flat.unsqueeze(2) + offsets
        n_local = int(points.shape[2])
        sampled = sample_feature_map_at_pixels(
            feature_map,
            points.reshape(batch, n_hyp * n_tips * n_local, 2),
            image_size=image_size,
            padding_mode="border",
        )
        tokens = sampled.reshape(batch, n_hyp * n_tips, n_local, channels)
    n_local = int(tokens.shape[2])
    return tokens.reshape(batch, n_hyp, n_tips, n_local, channels)


def _expand_hypotheses(
    hypotheses: torch.Tensor,
    batch: int,
    n_hyp: int,
) -> torch.Tensor:
    if hypotheses.ndim == 3:
        if hypotheses.shape != (n_hyp, N_TIPS, HYPOTHESIS_XYZ_DIM):
            raise HypothesisScorerError(
                "hypotheses must be [N,2,3], "
                f"got {tuple(hypotheses.shape)} for N={n_hyp}"
            )
        return hypotheses.to(dtype=torch.float32).unsqueeze(0).expand(batch, -1, -1, -1)
    if hypotheses.ndim == 4:
        if hypotheses.shape != (batch, n_hyp, N_TIPS, HYPOTHESIS_XYZ_DIM):
            raise HypothesisScorerError(
                "hypotheses must be [B,N,2,3] or [N,2,3], "
                f"got {tuple(hypotheses.shape)}"
            )
        return hypotheses
    raise HypothesisScorerError(
        f"hypotheses must be [N,2,3] or [B,N,2,3], got {tuple(hypotheses.shape)}"
    )


def _zero_last_linear(module: nn.Module) -> None:
    last = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            last = child
    if last is not None:
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)


class ActionConditionedHypothesisScorerCore(nn.Module):
    """Trainable scorer over already-extracted dense features.

    Inputs are feature maps, CAD candidate pixels, executed ``Δq``, and the
    global 3-D hypothesis offsets.  There is no absolute-tip decoder.
    """

    def __init__(
        self,
        visual_dim: int = VIT_TOKEN_DIM,
        model_dim: int = 128,
        local_grid_size: int = 3,
        local_radius_px: float = 14.0,
        pair_layers: int = 1,
        set_layers: int = 1,
        heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        max_innovation_px: float = 32.0,
        image_size: tuple[int, int] = IMAGE_SIZE_PX,
        joint_dim: int = JOINT_DIM,
    ) -> None:
        super().__init__()
        if int(model_dim) % int(heads) != 0:
            raise HypothesisScorerError(
                f"model_dim={model_dim} must be divisible by heads={heads}"
            )
        if int(local_grid_size) != 1 and int(local_grid_size) % 2 == 0:
            raise HypothesisScorerError("local_grid_size must be 1 or odd")
        self.visual_dim = int(visual_dim)
        self.model_dim = int(model_dim)
        self.local_grid_size = int(local_grid_size)
        self.local_radius_px = float(local_radius_px)
        self.max_innovation_px = float(max_innovation_px)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.joint_dim = int(joint_dim)
        fused_dim = self.visual_dim * 6 + 2 + 2 + 2 + self.joint_dim + HYPOTHESIS_XYZ_DIM
        self.tip_encoder = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        pair_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(heads),
            dim_feedforward=self.model_dim * int(mlp_ratio),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        set_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(heads),
            dim_feedforward=self.model_dim * int(mlp_ratio),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pair_transformer = nn.TransformerEncoder(
            pair_layer,
            num_layers=int(pair_layers),
            norm=nn.LayerNorm(self.model_dim),
            enable_nested_tensor=False,
        )
        self.set_transformer = nn.TransformerEncoder(
            set_layer,
            num_layers=int(set_layers),
            norm=nn.LayerNorm(self.model_dim),
            enable_nested_tensor=False,
        )
        self.motion_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, N_XY),
        )
        self.validity_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, 1),
        )
        self.hypothesis_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, 1),
        )
        self.scale_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, 1),
        )
        _zero_last_linear(self.motion_head)
        _zero_last_linear(self.hypothesis_head)

    def _fuse_tips(
        self,
        feature_t: torch.Tensor,
        feature_tp1: torch.Tensor,
        candidate_t_px: torch.Tensor,
        candidate_tp1_px: torch.Tensor,
        delta_q: torch.Tensor,
        hypotheses: torch.Tensor,
    ) -> torch.Tensor:
        tokens_t = sample_tip_local_features(
            feature_t,
            candidate_t_px,
            image_size=self.image_size,
            grid_size=self.local_grid_size,
            radius_px=self.local_radius_px,
        )
        tokens_tp1 = sample_tip_local_features(
            feature_tp1,
            candidate_tp1_px,
            image_size=self.image_size,
            grid_size=self.local_grid_size,
            radius_px=self.local_radius_px,
        )
        center_index = tokens_t.shape[3] // 2
        center_t = tokens_t[:, :, :, center_index, :]
        center_tp1 = tokens_tp1[:, :, :, center_index, :]
        mean_t = tokens_t.mean(dim=3)
        mean_tp1 = tokens_tp1.mean(dim=3)
        width = float(self.image_size[0])
        height = float(self.image_size[1])
        pos_t = torch.stack(
            (candidate_t_px[..., 0] / width, candidate_t_px[..., 1] / height),
            dim=-1,
        )
        pos_tp1 = torch.stack(
            (candidate_tp1_px[..., 0] / width, candidate_tp1_px[..., 1] / height),
            dim=-1,
        )
        cad_disp = candidate_tp1_px - candidate_t_px
        cad_norm = torch.stack((cad_disp[..., 0] / width, cad_disp[..., 1] / height), dim=-1)
        batch, n_hyp, n_tips, _ = candidate_t_px.shape
        if delta_q.shape != (batch, self.joint_dim):
            raise HypothesisScorerError(
                f"delta_q must be [B,{self.joint_dim}], got {tuple(delta_q.shape)}"
            )
        delta_exp = delta_q[:, None, None, :].expand(batch, n_hyp, n_tips, self.joint_dim)
        xyz = _expand_hypotheses(hypotheses, batch, n_hyp).to(
            device=candidate_t_px.device, dtype=candidate_t_px.dtype
        )
        fused = torch.cat(
            (
                center_t,
                mean_t,
                center_tp1,
                mean_tp1,
                center_tp1 - center_t,
                center_t * center_tp1,
                pos_t,
                pos_tp1,
                cad_norm,
                delta_exp,
                xyz,
            ),
            dim=-1,
        )
        return self.tip_encoder(fused)

    def forward(
        self,
        feature_t: torch.Tensor,
        feature_tp1: torch.Tensor,
        candidate_t_px: torch.Tensor,
        candidate_tp1_px: torch.Tensor,
        delta_q: torch.Tensor,
        hypotheses: torch.Tensor,
    ) -> HypothesisScoreOutput:
        pair_out, hyp_tokens = self.encode_hypothesis_tokens(
            feature_t,
            feature_tp1,
            candidate_t_px,
            candidate_tp1_px,
            delta_q,
            hypotheses,
        )
        motion = torch.tanh(self.motion_head(pair_out)) * self.max_innovation_px
        tip_logits = self.validity_head(pair_out).squeeze(-1)
        hyp_logits = self.hypothesis_head(hyp_tokens).squeeze(-1)
        scale = F.softplus(self.scale_head(hyp_tokens)).squeeze(-1) + 1.0e-3
        output = HypothesisScoreOutput(
            motion_innovation_px=motion,
            tip_validity_logits=tip_logits,
            hypothesis_logits=hyp_logits,
            uncalibrated_scale=scale,
        )
        assert_output_has_no_absolute_tip(output)
        return output

    def encode_hypothesis_tokens(
        self,
        feature_t: torch.Tensor,
        feature_tp1: torch.Tensor,
        candidate_t_px: torch.Tensor,
        candidate_tp1_px: torch.Tensor,
        delta_q: torch.Tensor,
        hypotheses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return left/right pair tokens and set-contextualized hypothesis tokens."""

        if feature_t.ndim != 4 or feature_tp1.ndim != 4:
            raise HypothesisScorerError("feature_t/feature_tp1 must be [B,C,H,W]")
        if feature_t.shape != feature_tp1.shape:
            raise HypothesisScorerError("feature_t and feature_tp1 shapes must match")
        if feature_t.shape[1] != self.visual_dim:
            raise HypothesisScorerError(
                f"expected visual_dim={self.visual_dim}, got {feature_t.shape[1]}"
            )
        if candidate_t_px.shape != candidate_tp1_px.shape:
            raise HypothesisScorerError("candidate_t_px and candidate_tp1_px shapes must match")
        if candidate_t_px.ndim != 4 or candidate_t_px.shape[-2:] != (N_TIPS, N_XY):
            raise HypothesisScorerError(
                f"candidates must be [B,N,2,2], got {tuple(candidate_t_px.shape)}"
            )
        batch, n_hyp, _, _ = candidate_t_px.shape
        tip_tokens = self._fuse_tips(
            feature_t,
            feature_tp1,
            candidate_t_px,
            candidate_tp1_px,
            delta_q,
            hypotheses,
        )
        pair_in = tip_tokens.reshape(batch * n_hyp, N_TIPS, self.model_dim)
        pair_out = self.pair_transformer(pair_in).reshape(batch, n_hyp, N_TIPS, self.model_dim)
        pair_pool = pair_out.mean(dim=2)
        hyp_tokens = self.set_transformer(pair_pool)
        return pair_out, hyp_tokens


class FrozenV9ViTPatchEncoder(nn.Module):
    """Strict frozen FrozenV9ViT → ``[B, 192, 16, 16]`` patch grid."""

    def __init__(self, checkpoint: str | Path) -> None:
        super().__init__()
        from v10.model import FrozenV9ViT

        self.vit = FrozenV9ViT(checkpoint)
        freeze_module(self.vit)
        self.source_checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.source_sha256 = sha256_file(self.source_checkpoint)

    def train(self, mode: bool = True) -> "FrozenV9ViTPatchEncoder":
        super().train(False)
        freeze_module(self.vit)
        return self

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise HypothesisScorerError(f"rgb must be [B,3,H,W], got {tuple(rgb.shape)}")
        self.vit.eval()
        with torch.no_grad():
            tokens = self.vit(rgb)
        return vit_tokens_to_feature_map(tokens)


class ActionConditionedHypothesisScorer(nn.Module):
    """Frozen ViT wrapper plus the trainable hypothesis-scoring core."""

    def __init__(
        self,
        vit_checkpoint: str | Path,
        core: ActionConditionedHypothesisScorerCore | None = None,
        **core_kwargs: Any,
    ) -> None:
        super().__init__()
        self.encoder = FrozenV9ViTPatchEncoder(vit_checkpoint)
        self.core = core if core is not None else ActionConditionedHypothesisScorerCore(
            **core_kwargs
        )
        if self.core.visual_dim != VIT_TOKEN_DIM:
            raise HypothesisScorerError(
                f"core.visual_dim must be {VIT_TOKEN_DIM} when wrapping FrozenV9ViT"
            )
        freeze_module(self.encoder)

    def train(self, mode: bool = True) -> "ActionConditionedHypothesisScorer":
        super().train(mode)
        freeze_module(self.encoder)
        self.core.train(mode)
        return self

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        return self.encoder(rgb)

    def encode_pair(
        self, image_t: torch.Tensor, image_tp1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(image_t), self.encode(image_tp1)

    def forward(
        self,
        image_t: torch.Tensor,
        image_tp1: torch.Tensor,
        candidate_t_px: torch.Tensor,
        candidate_tp1_px: torch.Tensor,
        delta_q: torch.Tensor,
        hypotheses: torch.Tensor,
    ) -> HypothesisScoreOutput:
        feature_t, feature_tp1 = self.encode_pair(image_t, image_tp1)
        return self.core(
            feature_t,
            feature_tp1,
            candidate_t_px,
            candidate_tp1_px,
            delta_q,
            hypotheses,
        )

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.core.parameters()


def trainable_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    if isinstance(model, ActionConditionedHypothesisScorer):
        return model.core.parameters()
    return model.parameters()


def _as_bool_mask(value: torch.Tensor) -> torch.Tensor:
    return value.to(dtype=torch.bool)


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(loss).to(dtype=loss.dtype)
    weight = mask.sum()
    if float(weight.detach()) <= 0.0:
        return loss.sum() * 0.0, weight
    return (loss * mask).sum() / weight.clamp_min(1.0), weight


def _listwise_ranking_loss(
    logits: torch.Tensor,
    target_error_px: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rankable = mask.to(dtype=torch.bool)
    n_valid = rankable.sum(dim=-1)
    eligible = n_valid >= MIN_RANKABLE
    if not bool(eligible.any()):
        zero = logits.sum() * 0.0
        return zero, eligible.to(dtype=logits.dtype).sum(), eligible
    fill = torch.finfo(logits.dtype).min / 4.0
    teacher = (-target_error_px / float(temperature)).masked_fill(~rankable, fill)
    student = logits.masked_fill(~rankable, fill)
    target = torch.softmax(teacher, dim=-1)
    log_pred = torch.log_softmax(student, dim=-1)
    per = -(target * log_pred).sum(dim=-1)
    per = per.masked_fill(~eligible, 0.0)
    denom = eligible.to(dtype=logits.dtype).sum().clamp_min(1.0)
    return per.sum() / denom, eligible.to(dtype=logits.dtype).sum(), eligible


def require_pairwise_tie_tol_px(value: object) -> float:
    """Flow-teacher px gap.  Not a physical tip / millimetre tolerance."""

    try:
        tol = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise HypothesisScorerError(
            "pairwise_tie_tol_px must be a finite non-negative flow-teacher px"
        ) from exc
    if not math.isfinite(tol) or tol < 0.0:
        raise HypothesisScorerError(
            "pairwise_tie_tol_px must be finite and >= 0 "
            "(flow-teacher px, not physical tip)"
        )
    return tol


def _pairwise_ranking_loss(
    logits: torch.Tensor,
    target_error_px: torch.Tensor,
    mask: torch.Tensor,
    *,
    tie_tol: float = DEFAULT_PAIRWISE_TIE_TOL_PX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rankable = mask.to(dtype=torch.bool)
    delta_logit = logits.unsqueeze(-1) - logits.unsqueeze(-2)
    delta_err = target_error_px.unsqueeze(-1) - target_error_px.unsqueeze(-2)
    pair_mask = (
        rankable.unsqueeze(-1)
        & rankable.unsqueeze(-2)
        & (delta_err.abs() > float(tie_tol))
    )
    # Two valid candidates are insufficient after applying the tie filter: a
    # sample is rankable only if at least one ordered pair survives.
    eligible = pair_mask.flatten(start_dim=1).any(dim=-1)
    if not bool(eligible.any()):
        zero = logits.sum() * 0.0
        return zero, eligible.to(dtype=logits.dtype).sum(), eligible
    # Prefer the lower teacher error: positive when j is worse than i.
    labels = torch.sign(-delta_err)
    pair_loss = F.softplus(-delta_logit * labels)
    pair_loss = pair_loss.masked_fill(~pair_mask, 0.0)
    pair_w = pair_mask.to(dtype=logits.dtype)
    pair_w = pair_w.masked_fill(~eligible[:, None, None], 0.0)
    weight = pair_w.sum()
    if float(weight.detach()) <= 0.0:
        return logits.sum() * 0.0, eligible.to(dtype=logits.dtype).sum(), eligible
    return pair_loss.sum() / weight.clamp_min(1.0), eligible.to(dtype=logits.dtype).sum(), eligible


def _pairwise_top1_vs_rest_ranking_loss(
    logits: torch.Tensor,
    target_error_px: torch.Tensor,
    mask: torch.Tensor,
    *,
    tie_tol: float = DEFAULT_PAIRWISE_TIE_TOL_PX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teacher-best vs rest pairs, sample-macro.  Flow-teacher px only.

    Each rankable sample contributes equally.  Pair count must not dominate.
    """

    rankable = mask.to(dtype=torch.bool)
    n_rankable = rankable.sum(dim=-1)
    can_rank = n_rankable >= MIN_RANKABLE
    filled_err = target_error_px.masked_fill(~rankable, torch.finfo(target_error_px.dtype).max)
    best_idx = filled_err.argmin(dim=-1)
    best_err = filled_err.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
    best_logit = logits.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
    is_best = torch.zeros_like(rankable)
    is_best.scatter_(-1, best_idx.unsqueeze(-1), True)
    rest_gap = (target_error_px - best_err.unsqueeze(-1)).abs()
    pair_mask = (
        can_rank.unsqueeze(-1)
        & rankable
        & (~is_best)
        & (rest_gap > float(tie_tol))
    )
    eligible = pair_mask.any(dim=-1)
    if not bool(eligible.any()):
        zero = logits.sum() * 0.0
        return zero, eligible.to(dtype=logits.dtype).sum(), eligible
    labels = torch.sign(-(best_err.unsqueeze(-1) - target_error_px))
    pair_loss = F.softplus(-(best_logit.unsqueeze(-1) - logits) * labels)
    pair_w = pair_mask.to(dtype=logits.dtype)
    pair_w = pair_w.masked_fill(~eligible.unsqueeze(-1), 0.0)
    pair_n = pair_w.sum(dim=-1)
    per_sample = (pair_loss * pair_w).sum(dim=-1) / pair_n.clamp_min(1.0)
    per_sample = per_sample.masked_fill(~eligible, 0.0)
    denom = eligible.to(dtype=logits.dtype).sum().clamp_min(1.0)
    return per_sample.sum() / denom, eligible.to(dtype=logits.dtype).sum(), eligible


def compute_scorer_losses(
    output: HypothesisScoreOutput,
    *,
    motion_residual_target_px: torch.Tensor,
    tip_valid: torch.Tensor,
    hypothesis_valid: torch.Tensor,
    excitation_px: torch.Tensor,
    target_error_px: torch.Tensor,
    min_excitation_px: float = DEFAULT_MIN_EXCITATION_PX,
    ranking: str = "listwise",
    ranking_temperature: float = DEFAULT_RANKING_TEMPERATURE,
    motion_weight: float = 1.0,
    tip_weight: float = 1.0,
    ranking_weight: float = 1.0,
    pairwise_tie_tol_px: float = DEFAULT_PAIRWISE_TIE_TOL_PX,
) -> ScorerLoss:
    """Masked motion SmoothL1, tip BCE, and fail-closed candidate ranking.

    ``pairwise_tie_tol_px`` is consumed when ``ranking`` is ``pairwise`` or
    ``pairwise_top1_vs_rest``.  It is a flow-teacher pixel gap, not
    physical-tip / millimetre accuracy.  ``pairwise`` stays pair-micro;
    ``pairwise_top1_vs_rest`` is teacher-best vs rest with sample-macro.
    """

    assert_output_has_no_absolute_tip(output)
    pred = output.float()
    target = motion_residual_target_px.to(dtype=pred.motion_innovation_px.dtype)
    if pred.motion_innovation_px.shape != target.shape:
        raise HypothesisScorerError(
            "motion_innovation_px / motion_residual_target_px shape mismatch: "
            f"{tuple(pred.motion_innovation_px.shape)} vs {tuple(target.shape)}"
        )
    tip_mask = _as_bool_mask(tip_valid)
    hyp_mask = _as_bool_mask(hypothesis_valid)
    if tip_mask.shape != pred.tip_validity_logits.shape:
        raise HypothesisScorerError("tip_valid shape must match tip_validity_logits")
    if hyp_mask.shape != pred.hypothesis_logits.shape:
        raise HypothesisScorerError("hypothesis_valid shape must match hypothesis_logits")

    motion_elem = F.smooth_l1_loss(pred.motion_innovation_px, target, reduction="none")
    motion_loss, n_motion = _masked_mean(motion_elem, tip_mask)

    tip_target = tip_mask.to(dtype=pred.tip_validity_logits.dtype)
    tip_elem = F.binary_cross_entropy_with_logits(
        pred.tip_validity_logits, tip_target, reduction="none"
    )
    # Learn observability on every labelled tip.  Invalid hypotheses still carry
    # a defined 0/1 tip label.
    tip_loss, _ = _masked_mean(tip_elem, torch.ones_like(tip_elem, dtype=torch.bool))

    excited = excitation_px.to(dtype=pred.hypothesis_logits.dtype) >= float(min_excitation_px)
    rank_mask = hyp_mask & excited
    if ranking == RANKING_LISTWISE:
        rank_loss, n_rankable, eligible = _listwise_ranking_loss(
            pred.hypothesis_logits,
            target_error_px.to(dtype=pred.hypothesis_logits.dtype),
            rank_mask,
            temperature=float(ranking_temperature),
        )
    elif ranking == RANKING_PAIRWISE:
        rank_loss, n_rankable, eligible = _pairwise_ranking_loss(
            pred.hypothesis_logits,
            target_error_px.to(dtype=pred.hypothesis_logits.dtype),
            rank_mask,
            tie_tol=require_pairwise_tie_tol_px(pairwise_tie_tol_px),
        )
    elif ranking == RANKING_PAIRWISE_TOP1_VS_REST:
        rank_loss, n_rankable, eligible = _pairwise_top1_vs_rest_ranking_loss(
            pred.hypothesis_logits,
            target_error_px.to(dtype=pred.hypothesis_logits.dtype),
            rank_mask,
            tie_tol=require_pairwise_tie_tol_px(pairwise_tie_tol_px),
        )
    else:
        raise HypothesisScorerError(f"unknown ranking loss {ranking!r}")
    fail_closed = ~eligible.any()
    total = (
        float(motion_weight) * motion_loss
        + float(tip_weight) * tip_loss
        + float(ranking_weight) * rank_loss
    )
    if not torch.isfinite(total):
        raise HypothesisScorerError("scorer loss is not finite")
    parts = ScorerLoss(
        total=total,
        motion_residual=motion_loss,
        tip_validity=tip_loss,
        ranking=rank_loss,
        n_rankable_samples=n_rankable,
        n_motion_elements=n_motion,
        ranking_fail_closed=fail_closed,
    )
    assert_loss_parts_isolated(parts.as_dict())
    return parts


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def zero_hypothesis_index(hypotheses: np.ndarray | torch.Tensor) -> int:
    offsets = _to_numpy(hypotheses)
    if offsets.ndim == 4:
        offsets = offsets[0]
    if offsets.shape[-2:] != (N_TIPS, HYPOTHESIS_XYZ_DIM):
        raise HypothesisScorerError(
            f"hypotheses must end with [2,3], got {tuple(offsets.shape)}"
        )
    norms = np.linalg.norm(offsets.astype(np.float64), axis=-1).sum(axis=-1)
    return int(np.argmin(norms))


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    y = labels.astype(np.float64).reshape(-1)
    s = scores.astype(np.float64).reshape(-1)
    if y.size < 2 or y.min() == y.max():
        return None
    order = np.argsort(s, kind="mergesort")
    y_sorted = y[order]
    n_pos = float(y.sum())
    n_neg = float(y.size) - n_pos
    if n_pos <= 0.0 or n_neg <= 0.0:
        return None
    ranks = np.empty(y.size, dtype=np.float64)
    index = 0
    s_ord = s[order]
    while index < y.size:
        j = index
        while j + 1 < y.size and s_ord[j + 1] == s_ord[index]:
            j += 1
        ranks[index : j + 1] = 0.5 * (index + j) + 1.0
        index = j + 1
    pos_rank_sum = float(ranks[y_sorted > 0.5].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def compute_scorer_metrics(
    output: HypothesisScoreOutput,
    *,
    motion_residual_target_px: torch.Tensor | np.ndarray,
    tip_valid: torch.Tensor | np.ndarray,
    hypothesis_valid: torch.Tensor | np.ndarray,
    excitation_px: torch.Tensor | np.ndarray,
    target_error_px: torch.Tensor | np.ndarray,
    hypotheses: torch.Tensor | np.ndarray,
    min_excitation_px: float = DEFAULT_MIN_EXCITATION_PX,
    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
) -> dict[str, float | None]:
    """Teacher-error ranking metrics.  Not absolute pixel / physical tip GT."""

    motion = _to_numpy(output.motion_innovation_px)
    target = _to_numpy(motion_residual_target_px)
    tip_logits = _to_numpy(output.tip_validity_logits)
    hyp_logits = _to_numpy(output.hypothesis_logits)
    tip_mask = _to_numpy(tip_valid) > 0.5
    hyp_mask = _to_numpy(hypothesis_valid) > 0.5
    errors = _to_numpy(target_error_px)
    excitation = _to_numpy(excitation_px)
    if motion.shape != target.shape:
        raise HypothesisScorerError("metric motion shapes must match")

    abs_err = np.abs(motion - target)
    if tip_mask.any():
        masked = abs_err[np.broadcast_to(tip_mask[..., None], abs_err.shape)]
        mae = float(masked.mean())
        rmse = float(np.sqrt(np.mean(np.square(masked))))
    else:
        mae = float("nan")
        rmse = float("nan")

    tip_prob = 1.0 / (1.0 + np.exp(-np.clip(tip_logits, -40.0, 40.0)))
    tip_pred = tip_prob >= 0.5
    tp = float(np.logical_and(tip_pred, tip_mask).sum())
    fp = float(np.logical_and(tip_pred, np.logical_not(tip_mask)).sum())
    fn = float(np.logical_and(np.logical_not(tip_pred), tip_mask).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    tip_auc = _binary_auc(tip_logits, tip_mask.astype(np.float64))

    zero_idx = zero_hypothesis_index(hypotheses)
    top1_errors: list[float] = []
    oracle_errors: list[float] = []
    zero_errors: list[float] = []
    regrets: list[float] = []
    accepted = 0
    false_accept = 0
    n_samples = int(hyp_logits.shape[0])
    n_fail_closed = 0
    n_zero_rankable = 0
    for sample in range(n_samples):
        valid = hyp_mask[sample]
        rankable = valid & (
            excitation[sample] >= float(min_excitation_px)
        )
        if int(rankable.sum()) < MIN_RANKABLE:
            n_fail_closed += 1
            continue
        sample_err = errors[sample].astype(np.float64)
        sample_logits = hyp_logits[sample].astype(np.float64)
        filled = np.where(rankable, sample_logits, -np.inf)
        top1 = int(np.argmax(filled))
        oracle = int(np.argmin(np.where(rankable, sample_err, np.inf)))
        top1_err = float(sample_err[top1])
        oracle_err = float(sample_err[oracle])
        top1_errors.append(top1_err)
        oracle_errors.append(oracle_err)
        regrets.append(top1_err - oracle_err)
        if rankable[zero_idx]:
            zero_errors.append(float(sample_err[zero_idx]))
            n_zero_rankable += 1
        prob = 1.0 / (1.0 + np.exp(-np.clip(sample_logits, -40.0, 40.0)))
        take = rankable & (prob >= float(accept_threshold))
        accepted += int(take.sum())
        false_accept += int(np.logical_and(take, np.arange(valid.shape[0]) != oracle).sum())

    def _med(values: Sequence[float]) -> float:
        return float(np.median(values)) if values else float("nan")

    def _p90(values: Sequence[float]) -> float:
        return float(np.percentile(values, 90)) if values else float("nan")

    threshold_non_oracle_fraction = (
        float(false_accept / accepted) if accepted else float("nan")
    )
    return {
        "motion_residual_mae": mae,
        "motion_residual_rmse": rmse,
        "tip_validity_precision": float(precision),
        "tip_validity_recall": float(recall),
        "tip_validity_auc": tip_auc,
        "top1_target_error_median": _med(top1_errors),
        "top1_target_error_p90": _p90(top1_errors),
        "oracle_best_target_error_median": _med(oracle_errors),
        "zero_hypothesis_target_error_median": _med(zero_errors),
        "top1_regret_median": _med(regrets),
        # Pairwise/listwise logits have no calibrated acceptance semantics.
        # Keep this as a plainly named threshold diagnostic rather than the
        # misleading historical "false_accept_rate".
        "ranking_logit_threshold_non_oracle_fraction": threshold_non_oracle_fraction,
        "ranking_logit_threshold": float(accept_threshold),
        "n_scored_samples": float(len(top1_errors)),
        "n_fail_closed_samples": float(n_fail_closed),
        "decision_coverage": float(len(top1_errors) / max(n_samples, 1)),
        "zero_hypothesis_coverage": float(n_zero_rankable / max(n_samples, 1)),
        "n_ranking_logits_above_threshold": float(accepted),
    }


def selection_tuple(metrics: Mapping[str, float | None]) -> tuple[float, float]:
    """Best-checkpoint key: val top-1 teacher error, then regret.  Lower better."""

    top1 = metrics.get("top1_target_error_median")
    regret = metrics.get("top1_regret_median")
    top1_v = float(top1) if top1 is not None and math.isfinite(float(top1)) else float("inf")
    regret_v = (
        float(regret) if regret is not None and math.isfinite(float(regret)) else float("inf")
    )
    return (top1_v, regret_v)


def build_checkpoint_metadata(
    *,
    vit_checkpoint: str | Path | None,
    overfit_one_batch: bool = False,
    live_ready: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if live_ready:
        raise HypothesisScorerError(
            "mechanism checkpoints cannot be marked live_ready"
        )
    contract = lewm_actor_supervision_contract()
    path = None
    digest = None
    if vit_checkpoint is not None:
        resolved = Path(vit_checkpoint).expanduser().resolve()
        path = str(resolved)
        if resolved.is_file():
            digest = sha256_file(resolved)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "schema": SCHEMA_NAME,
        "training_target": TRAINING_TARGET,
        "has_physical_labels": False,
        "absolute_tip_accuracy_claim": False,
        "uncertainty_calibrated": False,
        "live_ready": False,
        "overfit_one_batch": bool(overfit_one_batch),
        "frozen_vit_source_path": path,
        "frozen_vit_source_sha256": digest,
        "frozen_vit_class": "v10.model.FrozenV9ViT",
        "lewm_dynamics": contract["lewm_dynamics"],
        "lewm_isolation": True,
        "lewm_supervision": "not_used",
        "structured_goal_enters_scorer": False,
        "goal_sensitive_loss_enters_scorer": False,
        "policy_action_delta_causal_loss_enters_scorer": False,
        "lewm_goal_free": True,
    }
    if extra:
        forbidden = {"live_ready", "has_physical_labels", "absolute_tip_accuracy_claim"}
        leaked = forbidden.intersection(extra)
        if leaked:
            raise HypothesisScorerError(
                f"cannot override honest checkpoint flags: {sorted(leaked)}"
            )
        metadata.update(dict(extra))
    metadata["live_ready"] = False
    metadata["has_physical_labels"] = False
    metadata["absolute_tip_accuracy_claim"] = False
    metadata["uncertainty_calibrated"] = False
    metadata["training_target"] = TRAINING_TARGET
    return metadata


def validate_checkpoint_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "training_target": TRAINING_TARGET,
        "has_physical_labels": False,
        "absolute_tip_accuracy_claim": False,
        "uncertainty_calibrated": False,
        "live_ready": False,
    }
    for key, expected in required.items():
        if key not in metadata:
            raise HypothesisScorerError(f"checkpoint metadata missing {key}")
        if metadata[key] != expected:
            raise HypothesisScorerError(
                f"checkpoint metadata {key}={metadata[key]!r} must be {expected!r}"
            )
    if metadata.get("lewm_isolation") is not True:
        raise HypothesisScorerError("checkpoint must declare lewm_isolation=true")
    if metadata.get("lewm_dynamics") != "goal_free":
        raise HypothesisScorerError("checkpoint must declare lewm_dynamics=goal_free")
    return dict(metadata)


def _as_unicode_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind == "S":
        decoded = np.array(
            [item.decode("utf-8") for item in array.reshape(-1)],
            dtype=np.str_,
        ).reshape(array.shape)
        return decoded
    if array.dtype.kind in {"U", "O"}:
        return np.array(array.astype(str), dtype=np.str_).reshape(array.shape)
    raise DICGMotionDatasetError(
        f"unicode field has non-text dtype {array.dtype}"
    )


def _require_finite(name: str, array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.dtype.kind not in {"f", "c"}:
        return values
    if not np.all(np.isfinite(values)):
        raise DICGMotionDatasetError(f"{name} contains non-finite values")
    return values


def _as_unit_mask(name: str, array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(array)
    if values.shape != shape:
        raise DICGMotionDatasetError(f"{name} must have shape {shape}, got {values.shape}")
    if values.dtype == np.bool_:
        return values
    flat = values.astype(np.float64)
    if not np.all(np.isfinite(flat)):
        raise DICGMotionDatasetError(f"{name} contains non-finite values")
    unique = set(np.unique(flat).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise DICGMotionDatasetError(f"{name} must be boolean or {{0,1}}")
    return flat > 0.5


def _resolve_existing(path_text: str, root: Path) -> Path:
    raw = Path(str(path_text))
    candidates = [raw] if raw.is_absolute() else [root / raw, Path.cwd() / raw]
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved.is_file():
            return resolved.resolve()
    raise DICGMotionDatasetError(f"image path does not exist: {path_text}")


@dataclass
class DICGMotionArrays:
    image_t_path: np.ndarray
    image_tp1_path: np.ndarray
    q_t: np.ndarray
    q_tp1: np.ndarray
    delta_q: np.ndarray
    hypotheses_tool_offsets_m: np.ndarray
    candidate_t_px: np.ndarray
    candidate_tp1_px: np.ndarray
    observed_flow_px: np.ndarray
    motion_residual_target_px: np.ndarray
    tip_valid: np.ndarray
    hypothesis_valid: np.ndarray
    excitation_px: np.ndarray
    target_error_px: np.ndarray
    split: np.ndarray
    schema: str
    provenance: np.ndarray
    extra_unicode: dict[str, np.ndarray]
    resolved_image_t: tuple[Path, ...]
    resolved_image_tp1: tuple[Path, ...]
    root: Path

    @property
    def n_samples(self) -> int:
        return int(self.split.shape[0])

    @property
    def n_hypotheses(self) -> int:
        return int(self.hypotheses_tool_offsets_m.shape[0])

    def indices(self, split_id: int) -> np.ndarray:
        return np.flatnonzero(self.split == int(split_id)).astype(np.int64)


def validate_dicg_motion_payload(
    payload: Mapping[str, np.ndarray],
    *,
    root: Path,
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
    min_excitation_px: float = DEFAULT_MIN_EXCITATION_PX,
    required_splits: Sequence[int] = (SPLIT_TRAIN, SPLIT_VAL),
) -> DICGMotionArrays:
    keys = set(payload)
    leaked = FORBIDDEN_DATASET_KEYS.intersection(keys)
    if leaked:
        raise DICGMotionDatasetError(
            f"dataset carries forbidden goal/LeWM/absolute-tip keys: {sorted(leaked)}"
        )
    missing = [key for key in REQUIRED_ARRAY_KEYS + REQUIRED_UNICODE_KEYS if key not in keys]
    if missing:
        raise DICGMotionDatasetError(f"missing NPZ keys: {missing}")

    schema_arr = _as_unicode_array(np.asarray(payload["schema"]))
    schema = str(np.asarray(schema_arr).reshape(-1)[0])
    if schema != SCHEMA_NAME:
        raise DICGMotionDatasetError(f"schema must be {SCHEMA_NAME!r}, got {schema!r}")

    image_t_path = _as_unicode_array(np.asarray(payload["image_t_path"]))
    image_tp1_path = _as_unicode_array(np.asarray(payload["image_tp1_path"]))
    if image_t_path.ndim != 1 or image_tp1_path.ndim != 1:
        raise DICGMotionDatasetError("image_*_path must be [M]")
    n_samples = int(image_t_path.shape[0])
    if image_tp1_path.shape != (n_samples,):
        raise DICGMotionDatasetError("image_t_path and image_tp1_path length mismatch")

    hypotheses = _require_finite(
        "hypotheses_tool_offsets_m", np.asarray(payload["hypotheses_tool_offsets_m"])
    ).astype(np.float32)
    if hypotheses.ndim != 3 or hypotheses.shape[1:] != (N_TIPS, HYPOTHESIS_XYZ_DIM):
        raise DICGMotionDatasetError(
            "hypotheses_tool_offsets_m must be [N,2,3], "
            f"got {hypotheses.shape}"
        )
    n_hyp = int(hypotheses.shape[0])
    if n_hyp < MIN_RANKABLE:
        raise DICGMotionDatasetError(
            f"need at least {MIN_RANKABLE} hypotheses, got {n_hyp}"
        )

    shapes = {
        "q_t": (n_samples, JOINT_DIM),
        "q_tp1": (n_samples, JOINT_DIM),
        "delta_q": (n_samples, JOINT_DIM),
        "candidate_t_px": (n_samples, n_hyp, N_TIPS, N_XY),
        "candidate_tp1_px": (n_samples, n_hyp, N_TIPS, N_XY),
        "observed_flow_px": (n_samples, n_hyp, N_TIPS, N_XY),
        "motion_residual_target_px": (n_samples, n_hyp, N_TIPS, N_XY),
        "excitation_px": (n_samples, n_hyp),
        "target_error_px": (n_samples, n_hyp),
        "split": (n_samples,),
    }
    parsed: dict[str, np.ndarray] = {}
    for name, shape in shapes.items():
        array = np.asarray(payload[name])
        if array.shape != shape:
            raise DICGMotionDatasetError(f"{name} must have shape {shape}, got {array.shape}")
        parsed[name] = _require_finite(name, array)

    tip_valid = _as_unit_mask("tip_valid", payload["tip_valid"], (n_samples, n_hyp, N_TIPS))
    hypothesis_valid = _as_unit_mask(
        "hypothesis_valid", payload["hypothesis_valid"], (n_samples, n_hyp)
    )
    split = np.asarray(parsed["split"])
    if not np.issubdtype(split.dtype, np.integer):
        if not np.all(split == np.floor(split)):
            raise DICGMotionDatasetError("split must be integer 0/1/2")
    split_u8 = split.astype(np.uint8)
    if set(np.unique(split_u8).tolist()) - {SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST}:
        raise DICGMotionDatasetError("split values must be 0 (train), 1 (val), 2 (test)")

    provenance = _as_unicode_array(np.asarray(payload["provenance"]))
    if provenance.size not in {1, n_samples}:
        raise DICGMotionDatasetError("provenance must be scalar or [M]")

    extra_unicode: dict[str, np.ndarray] = {}
    for key in OPTIONAL_UNICODE_KEYS:
        if key in payload:
            values = _as_unicode_array(np.asarray(payload[key]))
            if key in ROW_UNICODE_KEYS and values.shape != (n_samples,):
                raise DICGMotionDatasetError(
                    f"{key} must have shape ({n_samples},), got {values.shape}"
                )
            extra_unicode[key] = values

    try:
        for index in range(n_samples):
            tokens = [str(image_t_path[index]), str(image_tp1_path[index])]
            for key in ("session", "session_id"):
                values = extra_unicode.get(key)
                if values is not None:
                    tokens.append(str(values[index]))
            for token in tokens:
                refuse_associated_motion_session(token)
            if int(split_u8[index]) in {SPLIT_TRAIN, SPLIT_VAL}:
                refuse_sealed_test_identity(*tokens)
    except DicgTestSealError as exc:
        raise DICGMotionDatasetError(str(exc)) from exc

    resolved_t = tuple(_resolve_existing(str(path), root) for path in image_t_path.tolist())
    resolved_tp1 = tuple(
        _resolve_existing(str(path), root) for path in image_tp1_path.tolist()
    )

    def _split_paths(split_id: int) -> set[Path]:
        idx = np.flatnonzero(split_u8 == split_id)
        return {resolved_t[int(i)] for i in idx} | {resolved_tp1[int(i)] for i in idx}

    train_paths = _split_paths(SPLIT_TRAIN)
    val_paths = _split_paths(SPLIT_VAL)
    test_paths = _split_paths(SPLIT_TEST)
    if train_paths & val_paths:
        raise DICGMotionDatasetError("train and val image paths intersect")
    if train_paths & test_paths:
        raise DICGMotionDatasetError("train and test image paths intersect")
    if val_paths & test_paths:
        raise DICGMotionDatasetError("val and test image paths intersect")

    required = tuple(int(value) for value in required_splits)
    if not required or len(set(required)) != len(required):
        raise DICGMotionDatasetError("required_splits must contain unique split ids")
    unknown_required = set(required) - set(SPLIT_NAMES)
    if unknown_required:
        raise DICGMotionDatasetError(
            f"required_splits contains unknown ids: {sorted(unknown_required)}"
        )
    for split_id in required:
        if not bool((split_u8 == split_id).any()):
            raise DICGMotionDatasetError(
                f"dataset has no {SPLIT_NAMES[split_id]} split"
            )

    valid_ratio = float(hypothesis_valid.mean())
    if valid_ratio < float(min_valid_ratio):
        raise DICGMotionDatasetError(
            f"valid candidate ratio {valid_ratio:.4f} < {float(min_valid_ratio):.4f}"
        )

    excitation = parsed["excitation_px"].astype(np.float32)
    rankable = hypothesis_valid & (excitation >= float(min_excitation_px))
    n_rankable = rankable.sum(axis=1)
    for split_id in required:
        name = SPLIT_NAMES[split_id]
        idx = np.flatnonzero(split_u8 == split_id)
        if idx.size and not bool((n_rankable[idx] >= MIN_RANKABLE).any()):
            raise DICGMotionDatasetError(
                f"{name} has no sample with at least {MIN_RANKABLE} "
                "valid+excited ranking candidates"
            )

    extra_splits = set(int(value) for value in np.unique(split_u8).tolist()) - set(
        required
    )
    if extra_splits:
        names = [
            SPLIT_NAMES.get(split_id, str(split_id)) for split_id in sorted(extra_splits)
        ]
        raise DICGMotionDatasetError(
            "dataset contains unexpected split(s) "
            f"{names} not in required_splits; extra TEST rows are refused "
            "when only TRAIN/VAL are requested"
        )

    return DICGMotionArrays(
        image_t_path=image_t_path,
        image_tp1_path=image_tp1_path,
        q_t=parsed["q_t"].astype(np.float32),
        q_tp1=parsed["q_tp1"].astype(np.float32),
        delta_q=parsed["delta_q"].astype(np.float32),
        hypotheses_tool_offsets_m=hypotheses,
        candidate_t_px=parsed["candidate_t_px"].astype(np.float32),
        candidate_tp1_px=parsed["candidate_tp1_px"].astype(np.float32),
        observed_flow_px=parsed["observed_flow_px"].astype(np.float32),
        motion_residual_target_px=parsed["motion_residual_target_px"].astype(np.float32),
        tip_valid=tip_valid,
        hypothesis_valid=hypothesis_valid,
        excitation_px=excitation,
        target_error_px=parsed["target_error_px"].astype(np.float32),
        split=split_u8,
        schema=schema,
        provenance=provenance,
        extra_unicode=extra_unicode,
        resolved_image_t=resolved_t,
        resolved_image_tp1=resolved_tp1,
        root=Path(root),
    )


def refuse_sealed_test_npz_path(path: str | Path) -> Path:
    """Refuse any path whose resolved string contains ``test_sealed``."""

    npz_path = Path(path).expanduser()
    try:
        resolved = npz_path.resolve()
    except OSError:
        resolved = npz_path
    if SEALED_TEST_TOKEN in str(resolved):
        raise DICGMotionDatasetError(
            "refusing to load a path containing test_sealed"
        )
    if not resolved.is_file():
        raise DICGMotionDatasetError(f"NPZ does not exist: {resolved}")
    return resolved


def refuse_unexpected_motion_splits(
    split: np.ndarray, required_splits: Sequence[int]
) -> None:
    """Refuse extra split ids, including TEST rows when only TRAIN/VAL are requested."""

    required = tuple(int(value) for value in required_splits)
    if not required or len(set(required)) != len(required):
        raise DICGMotionDatasetError("required_splits must contain unique split ids")
    unknown_required = set(required) - set(SPLIT_NAMES)
    if unknown_required:
        raise DICGMotionDatasetError(
            f"required_splits contains unknown ids: {sorted(unknown_required)}"
        )
    split_u8 = np.asarray(split)
    present = (
        {int(value) for value in np.unique(split_u8).tolist()} if split_u8.size else set()
    )
    for split_id in required:
        if split_id not in present:
            raise DICGMotionDatasetError(
                f"dataset has no {SPLIT_NAMES[split_id]} split"
            )
    extra = present - set(required)
    if extra:
        names = [SPLIT_NAMES.get(split_id, str(split_id)) for split_id in sorted(extra)]
        raise DICGMotionDatasetError(
            "dataset contains unexpected split(s) "
            f"{names} not in required_splits; extra TEST rows are refused "
            "when only TRAIN/VAL are requested"
        )


def load_dicg_motion_npz(
    path: str | Path,
    *,
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
    min_excitation_px: float = DEFAULT_MIN_EXCITATION_PX,
    required_splits: Sequence[int] = (SPLIT_TRAIN, SPLIT_VAL),
) -> DICGMotionArrays:
    npz_path = refuse_sealed_test_npz_path(path)
    required = tuple(int(value) for value in required_splits)
    with np.load(npz_path, allow_pickle=False) as handle:
        if "split" not in handle.files:
            raise DICGMotionDatasetError("missing NPZ keys: ['split']")
        # Lazy per-key access: inspect split before materializing teacher arrays.
        refuse_unexpected_motion_splits(np.asarray(handle["split"]), required)
        payload = {key: handle[key] for key in handle.files}
    return validate_dicg_motion_payload(
        payload,
        root=npz_path.parent,
        min_valid_ratio=min_valid_ratio,
        min_excitation_px=min_excitation_px,
        required_splits=required_splits,
    )


def load_rgb_jpeg(
    path: str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE_PX,
) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).contiguous()
    height, width = int(image_size[1]), int(image_size[0])
    if tuple(tensor.shape[-2:]) != (height, width):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
    return tensor


class DICGMotionDataset(Dataset):
    """JPEG-backed motion pairs for one NPZ split.  Does not open cameras."""

    def __init__(
        self,
        arrays: DICGMotionArrays,
        split_id: int,
        image_size: tuple[int, int] = IMAGE_SIZE_PX,
        row_indices: np.ndarray | None = None,
    ) -> None:
        self.arrays = arrays
        self.split_id = int(split_id)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        split_rows = arrays.indices(self.split_id)
        if row_indices is None:
            self.indices = split_rows
        else:
            chosen = np.unique(np.asarray(row_indices, dtype=np.int64))
            allowed = set(int(i) for i in split_rows.tolist())
            leaked = [int(i) for i in chosen.tolist() if int(i) not in allowed]
            if leaked:
                raise DICGMotionDatasetError(
                    f"row_indices are not a subset of split {self.split_id}: "
                    f"{leaked[:8]}"
                )
            self.indices = chosen
        if self.indices.size == 0:
            raise DICGMotionDatasetError(
                f"{SPLIT_NAMES.get(self.split_id, self.split_id)} split is empty"
            )

    def __len__(self) -> int:
        return int(self.indices.size)

    @property
    def hypotheses(self) -> np.ndarray:
        return self.arrays.hypotheses_tool_offsets_m

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[int(index)])
        image_t = load_rgb_jpeg(self.arrays.resolved_image_t[row], self.image_size)
        image_tp1 = load_rgb_jpeg(self.arrays.resolved_image_tp1[row], self.image_size)
        return {
            "image_t": image_t,
            "image_tp1": image_tp1,
            "q_t": torch.from_numpy(np.ascontiguousarray(self.arrays.q_t[row])),
            "q_tp1": torch.from_numpy(np.ascontiguousarray(self.arrays.q_tp1[row])),
            "delta_q": torch.from_numpy(np.ascontiguousarray(self.arrays.delta_q[row])),
            "candidate_t_px": torch.from_numpy(
                np.ascontiguousarray(self.arrays.candidate_t_px[row])
            ),
            "candidate_tp1_px": torch.from_numpy(
                np.ascontiguousarray(self.arrays.candidate_tp1_px[row])
            ),
            "observed_flow_px": torch.from_numpy(
                np.ascontiguousarray(self.arrays.observed_flow_px[row])
            ),
            "motion_residual_target_px": torch.from_numpy(
                np.ascontiguousarray(self.arrays.motion_residual_target_px[row])
            ),
            "tip_valid": torch.from_numpy(
                self.arrays.tip_valid[row].astype(np.float32)
            ),
            "hypothesis_valid": torch.from_numpy(
                self.arrays.hypothesis_valid[row].astype(np.float32)
            ),
            "excitation_px": torch.from_numpy(
                np.ascontiguousarray(self.arrays.excitation_px[row])
            ),
            "target_error_px": torch.from_numpy(
                np.ascontiguousarray(self.arrays.target_error_px[row])
            ),
            "index": torch.tensor(row, dtype=torch.int64),
        }


__all__ = [
    "SCHEMA_NAME",
    "ALL_MOTION_SPLITS",
    "CHECKPOINT_FORMAT",
    "DEFAULT_PAIRWISE_TIE_TOL_PX",
    "PAIRWISE_RANKING_MODES",
    "RANKING_LISTWISE",
    "RANKING_MODES",
    "RANKING_PAIRWISE",
    "RANKING_PAIRWISE_TOP1_VS_REST",
    "TRAINING_TARGET",
    "ActionConditionedHypothesisScorer",
    "ActionConditionedHypothesisScorerCore",
    "DICGMotionArrays",
    "DICGMotionDataset",
    "DICGMotionDatasetError",
    "FrozenV9ViTPatchEncoder",
    "HypothesisScoreOutput",
    "HypothesisScorerError",
    "ScorerLoss",
    "assert_loss_parts_isolated",
    "assert_module_frozen",
    "assert_output_has_no_absolute_tip",
    "build_checkpoint_metadata",
    "compute_scorer_losses",
    "compute_scorer_metrics",
    "gradients_are_finite",
    "load_dicg_motion_npz",
    "refuse_sealed_test_npz_path",
    "refuse_unexpected_motion_splits",
    "require_pairwise_tie_tol_px",
    "SEALED_TEST_TOKEN",
    "sample_tip_local_features",
    "select_amp_dtype",
    "trainable_parameters",
    "validate_checkpoint_metadata",
    "validate_dicg_motion_payload",
    "vit_tokens_to_feature_map",
]
