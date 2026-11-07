"""Frozen V10 training and model contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass


FORMAT = "so101-fcpg-lewm-v10"
SIGREG_WEIGHT = 0.02
VIT_STATE_PREFIX = "backbone.base.encoder."
HISTORY_OFFSETS = (-60, -30, -15, -10, -5, 0)
FUTURE_OFFSETS = (5, 10, 15, 20)


@dataclass(frozen=True)
class V10Config:
    """The only supported V10 architecture and optimizer budget."""

    latent_dim: int = 192
    state_dim: int = 6
    num_views: int = 2
    num_anchors: int = 3
    image_size: int = 224
    history_offsets: tuple[int, ...] = HISTORY_OFFSETS
    future_offsets: tuple[int, ...] = FUTURE_OFFSETS
    action_horizon: int = 20
    raw_steps_per_chunk: int = 5
    model_dim: int = 192
    encoder_layers: int = 2
    decoder_layers: int = 4
    dynamics_layers: int = 2
    heads: int = 6
    ffn_dim: int = 768
    dropout: float = 0.0
    steps: int = 10_000
    batch_size: int = 24
    fcpg_batch_size: int = 48
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-3
    warmup_steps: int = 250
    sigreg_weight: float = SIGREG_WEIGHT
    view_interruption_probability: float = 0.35
    max_interruption_frames: int = 3
    fcpg_loss_weight: float = 1.0
    reliability_loss_weight: float = 0.2

    @property
    def obs_horizon(self) -> int:
        return len(self.history_offsets)

    @property
    def future_horizon(self) -> int:
        return len(self.future_offsets)

    def validate(self) -> None:
        if self.steps != 10_000:
            raise ValueError("V10 uses one fixed 10,000-step training budget")
        if self.sigreg_weight != SIGREG_WEIGHT:
            raise ValueError("V10 fixes SIGReg to 0.02")
        if self.history_offsets[-1] != 0:
            raise ValueError("history_offsets must terminate at the current frame")
        if tuple(sorted(self.history_offsets)) != self.history_offsets:
            raise ValueError("history_offsets must be strictly ordered")
        if len(set(self.history_offsets)) != len(self.history_offsets):
            raise ValueError("history_offsets must not repeat")
        expected_future = tuple(
            self.raw_steps_per_chunk * index
            for index in range(1, self.future_horizon + 1)
        )
        if self.future_offsets != expected_future:
            raise ValueError("future offsets must match the action chunk horizon")
        if self.action_horizon != self.future_horizon * self.raw_steps_per_chunk:
            raise ValueError("action and future horizons disagree")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if not 0.0 <= self.view_interruption_probability <= 1.0:
            raise ValueError("view interruption probability must lie in [0,1]")
        if self.max_interruption_frames <= 0:
            raise ValueError("max interruption duration must be positive")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


__all__ = [
    "FORMAT",
    "FUTURE_OFFSETS",
    "HISTORY_OFFSETS",
    "SIGREG_WEIGHT",
    "V10Config",
    "VIT_STATE_PREFIX",
]
