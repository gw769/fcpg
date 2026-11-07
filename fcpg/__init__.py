"""Feedback-Corrected Proprioceptive Grounding utilities.

Keep package import intentionally lightweight.  Video-decoder workers import
``fcpg.lerobot_archive`` but must not initialize PyTorch/ATen before opening
codec threads.  The historical eager re-export imported ``fcpg.modules`` (and
therefore torch) for every ``fcpg.*`` import, including CPU-only workers.
Public model symbols remain available through lazy attribute loading.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "FCPGOutput",
    "FCPGResidualOnly",
    "GroundedTokenAdapter",
    "ReliabilityConfidenceHead",
    "ResidualCorrectionHead",
    "pixels_to_grid_torch",
    "sample_feature_map_at_pixels",
]


if TYPE_CHECKING:
    from fcpg.modules import (
        FCPGOutput,
        FCPGResidualOnly,
        GroundedTokenAdapter,
        ReliabilityConfidenceHead,
        ResidualCorrectionHead,
        pixels_to_grid_torch,
        sample_feature_map_at_pixels,
    )


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    modules = import_module("fcpg.modules")
    value = getattr(modules, name)
    globals()[name] = value
    return value
