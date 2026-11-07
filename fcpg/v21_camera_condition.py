"""Offline camera-condition preparation; no service wiring or hardware access.

Filtering is a proposal: ``filtered, next_condition = condition.filter(obs)``
does not change ``condition``. A caller commits H0 by retaining next_condition
only after accepting that seed into a new, empty actual-memory state. Discard
next_condition for a preview. Every subsequent delivery must pass through the
committed condition, including refreshes and completion observations.

These controls prepare an experiment. They do not establish policy success or
authorize execution, and must not be applied to an existing live runtime.
"""
from dataclasses import dataclass, field, replace

import torch

from .v21_encoder import DIM, PATCHES
from .v21_memory import VisualObservation, clock, finite


CONDITIONS = ("both_fresh", "wrist_h0", "wrist_cold", "external_h0", "external_cold")
ADDRESS_MODES = ("factual", "zero")


@dataclass(frozen=True)
class V21CameraCondition:
    """Immutable delivery policy, with H0 committed explicitly by its caller.

    ``*_h0`` delivers that view at H0 only. ``*_cold`` never delivers it.
    ``external`` is the existing column view, index 1. The original H0 input
    must contain two real delivered views even for cold conditions; filtering
    then removes the selected view. Later natural camera losses stay missing.
    """

    name: str = "both_fresh"
    address: str = "factual"
    _h0_committed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.name not in CONDITIONS:
            raise ValueError("unknown offline camera condition")
        if self.address not in ADDRESS_MODES:
            raise ValueError("address must be factual or zero")

    @property
    def h0_committed(self):
        return self._h0_committed

    def configure_reader(self, core, *, state=None):
        """Set addressing on a new eval core before any actual state exists.

        The caller must supply any existing actual state; one cannot infer a
        separately held runtime state from a stateless core. A populated reader
        cache is also rejected. Only reader.mode changes: zero removes the 24
        motion-address fields, retaining age, view, patch coordinates and the
        Actor's factual frame_metadata. No hooks or weight changes are used.
        """
        from .v21_core import V21ResearchCore
        from .v21_memory import ReadbackMemoryReader

        if (not isinstance(core, V21ResearchCore)
                or type(core.reader) is not ReadbackMemoryReader):
            raise TypeError("offline conditions require the standard V21 research reader")
        if state is not None or self.h0_committed or core.reader._projection_cache is not None:
            raise ValueError("configure addressing before any actual state or H0 commit")
        if any(module.training for module in core.modules()):
            raise ValueError("offline reader configuration requires eval mode")
        core.reader.mode = "readback" if self.address == "factual" else "temporal"
        return core

    def filter(self, observation):
        """Return an owned, masked observation and a proposed next condition.

        Delivered values retain their original clocks and exposure q. Missing
        values are explicit zeros, never substituted frames or retimestamps.
        An absent object cue remains None; a supplied cue is zero in all eight
        fields whenever wrist delivery is absent. Inputs are never mutated.
        """
        if not isinstance(observation, VisualObservation) or observation.source != "actual_rgb":
            raise ValueError("conditions accept actual RGB VisualObservation inputs only")
        if observation.view_order != ("wrist", "column"):
            raise ValueError("view order must be wrist,column")
        mask = observation.measurement_available
        if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.shape[1] != 2 or mask.dtype != torch.bool:
            raise ValueError("measurement availability must be bool [B,2]")
        batch = mask.shape[0]
        if batch < 1:
            raise ValueError("an observation batch must not be empty")
        for label, value, shape in (
                ("features", observation.features, (batch, 2, PATCHES, DIM)),
                ("exposure q", observation.q_at_image, (batch, 2, 6)),
                ("image times", observation.image_times, (batch, 2))):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{label}: invalid observation shape")
            selected = value[mask.to(value.device)]
            if label == "image times":
                clock(label, selected, (int(mask.sum()),))
            else:
                finite(label, selected)
        cue = observation.object_relative
        if cue is not None:
            finite("current wrist object cue", cue, (batch, 8))
        place = getattr(observation, "place_relative", None)
        if place is not None:
            finite("current wrist place cue", place, (batch, 8))
        if not self.h0_committed and not bool(mask.all()):
            raise ValueError("H0 seed requires both actual camera deliveries")

        delivered = mask.clone()
        if self.name != "both_fresh":
            view = 0 if self.name.startswith("wrist_") else 1
            if self.name.endswith("_cold") or self.h0_committed:
                delivered[:, view] = False

        def masked(value, dimensions):
            available = delivered.to(value.device).reshape(batch, 2, *([1] * dimensions))
            return torch.where(available, value, torch.zeros_like(value))

        filtered = replace(observation,
            features=masked(observation.features, 2),
            q_at_image=masked(observation.q_at_image, 1),
            image_times=masked(observation.image_times, 0),
            measurement_available=delivered,
            object_relative=None if cue is None else torch.where(
                delivered[:, :1].to(cue.device), cue, torch.zeros_like(cue)),
            # the place cue (training overhaul doc section 17) is a wrist cue too: same rule
            **({} if place is None else dict(place_relative=torch.where(
                delivered[:, :1].to(place.device), place, torch.zeros_like(place)))))
        following = self
        if not self.h0_committed:
            following = replace(self)
            object.__setattr__(following, "_h0_committed", True)
        return filtered, following


__all__ = ["V21CameraCondition", "CONDITIONS", "ADDRESS_MODES"]
