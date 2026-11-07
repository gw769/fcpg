"""Observer-only factual motion history; it has no historical image input."""
from dataclasses import dataclass
import torch

from .prod_v20_policy import PolicyContractError, _tensor, _mask


@dataclass(frozen=True)
class ObserverMotionHistory:
    commands_position: torch.Tensor
    readbacks_position: torch.Tensor
    time_deltas: torch.Tensor
    valid: torch.Tensor

    def validate(self, batch):
        _tensor("observer history commands", self.commands_position, (batch, 3, 5, 6))
        _tensor("observer history readbacks", self.readbacks_position, (batch, 3, 6, 6))
        _tensor("observer history elapsed seconds", self.time_deltas, (batch, 3, 5))
        _mask("observer history valid", self.valid, (batch, 3))
        if (self.valid[..., None] & (self.time_deltas <= 0)).any():
            raise PolicyContractError("completed observer history intervals must be positive")
        if (self.valid[:, :-1] & ~self.valid[:, 1:]).any():
            raise PolicyContractError("observer history must be right-aligned")
