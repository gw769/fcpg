"""Physical-unit normalization for the FCPG-H V8 motion tower.

The cached 44-D signature mixes LeRobot's canonical servo coordinates with
CAD positions in metres and Jacobians in metres per canonical unit. Feeding
those raw values to one linear projection made the shared state/action blocks
thousands of times larger than some part-specific CAD blocks. V8 converts
every block to an explicit dimensionless physical convention before the
unchanged V7 motion tower. The conversion is fixed, camera-free and has no
trainable parameters or source-specific statistics.
"""

from __future__ import annotations

from pathlib import Path

import torch

from fcpg.fcpg_h_candidate_first import PART_MOTION_DIM
from fcpg.fcpg_h_centered_counterfactual import (
    CenteredCounterfactualFCPGH,
    CenteredCounterfactualTemporalGrounder,
)


FCPGH_V8_METHOD_REVISION = "fcpg-h-v8-physical-unit-normalized"
FCPGH_V8_CHECKPOINT_FORMAT = "so101-fcpg-h-physical-unit-normalized-checkpoint-v1"
FCPGH_V8_RECEIPT_FORMAT = "so101-fcpg-h-v8-training-receipt-v1"

# SO-101 observations/actions use LeRobot canonical servo coordinates: the
# five arm joints lie in [-100, 100] and the gripper in [0, 100]. Dividing by
# 100 maps both to order-one coordinates without pretending they are degrees.
SO101_CANONICAL_POSITION_UNIT = 100.0

# A single embodiment-level length unit, rounded from the SO-101 working
# reach, converts CAD metres to an order-one fraction of the robot. It is not
# fitted from images, tasks, sources, cameras or the train/dev split.
SO101_REFERENCE_REACH_M = 0.30


def physical_motion_unit_multiplier() -> torch.Tensor:
    """Return the fixed 44-D multiplier for the documented signature layout."""

    multiplier = torch.ones(PART_MOTION_DIM, dtype=torch.float32)
    # state, delta-state, command and servo error: canonical servo -> unit range
    multiplier[:24] = 1.0 / SO101_CANONICAL_POSITION_UNIT
    # XYZ, delta-XYZ and J@delta: metres -> fraction of embodiment reach
    multiplier[24:36] = 1.0 / SO101_REFERENCE_REACH_M
    # Jacobian column norms: metres/canonical -> reach/unit-coordinate
    multiplier[36:42] = (
        SO101_CANONICAL_POSITION_UNIT / SO101_REFERENCE_REACH_M
    )
    # Norms of J@delta are already metres.
    multiplier[42:44] = 1.0 / SO101_REFERENCE_REACH_M
    return multiplier


def normalize_part_motion_units(part_motion: torch.Tensor) -> torch.Tensor:
    """Convert a signature ending in 44 raw mixed-unit values to V8 units."""

    if part_motion.shape[-1] != PART_MOTION_DIM:
        raise ValueError(f"part motion must end in {PART_MOTION_DIM} values")
    multiplier = physical_motion_unit_multiplier().to(
        device=part_motion.device,
        dtype=part_motion.dtype,
    )
    return part_motion * multiplier


class UnitNormalizedTemporalGrounder(CenteredCounterfactualTemporalGrounder):
    """The unchanged V7 core preceded by a learned-free unit conversion."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_buffer(
            "motion_unit_multiplier",
            physical_motion_unit_multiplier(),
            persistent=True,
        )

    def part_motion_queries(self, part_motion: torch.Tensor) -> torch.Tensor:
        normalized = part_motion * self.motion_unit_multiplier.to(
            dtype=part_motion.dtype
        )
        return super().part_motion_queries(normalized)


class UnitNormalizedFCPGH(CenteredCounterfactualFCPGH):
    """Frozen V12 feature extractor plus the V8 unit-normalized core."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        core: UnitNormalizedTemporalGrounder | None = None,
    ) -> None:
        super().__init__(
            checkpoint,
            core=core or UnitNormalizedTemporalGrounder(),
        )
