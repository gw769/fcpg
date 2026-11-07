"""Persistent object identity. FrozenViT is a measurement, not a re-register.

Identity is a task role that already has a prototype (the crop bound to
``(task_class, role)``). A peak matching that prototype may seed the first
pixel. A later unmatched or far peak cannot replace it. Temporary loss keeps
the last committed point as ``predicted``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LIVE_OBJECT_SOURCES = ("frozen_vit", "predicted", "unavailable")
MEASUREMENT_SOURCE = "frozen_vit_features"


@dataclass
class ObjectBelief:
    """One role/view's committed object, plus the click that created it."""

    role: str
    view: str
    registered_px: np.ndarray
    centroid_px: np.ndarray
    source: str = "predicted"
    lost_rounds: int = 0

    def as_payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "view": self.view,
            "registered_px": [float(self.registered_px[0]), float(self.registered_px[1])],
            "centroid_px": [float(self.centroid_px[0]), float(self.centroid_px[1])],
            "source": self.source,
            "lost_rounds": int(self.lost_rounds),
        }


def _finite_px(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    point = np.asarray(value, dtype=np.float32).reshape(-1)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return None
    return point


def search_anchor_px(belief: ObjectBelief | None) -> np.ndarray | None:
    """Where FrozenViT is allowed to look. Never the whole frame after commit."""

    if belief is None:
        return None
    return np.asarray(belief.centroid_px, dtype=np.float32)


def seed_object_belief(
    role: str, view: str, registered_px: Any
) -> ObjectBelief:
    point = _finite_px(registered_px)
    if point is None:
        raise ValueError("object identity requires a finite seed pixel")
    return ObjectBelief(
        role=str(role),
        view=str(view),
        registered_px=point.copy(),
        centroid_px=point.copy(),
        source="predicted",
        lost_rounds=0,
    )


def _predicted_descriptor(
    belief: ObjectBelief, measurement: dict[str, Any]
) -> dict[str, Any]:
    centroid = np.asarray(belief.centroid_px, dtype=np.float32)
    gripper = measurement.get("gripper_center_px")
    relative = None if gripper is None else centroid - np.asarray(gripper, dtype=np.float32)
    return {
        **measurement,
        "visible": True,
        "configured": True,
        "role": belief.role,
        "view": belief.view,
        "centroid_px": centroid,
        "source": "predicted",
        "identity_held": True,
        "fresh_measurement": False,
        "lost_rounds": int(belief.lost_rounds),
        "relative_centroid_px": relative,
        "reason": measurement.get("reason") or "persistent_identity",
    }


def _unavailable(role: str, view: str, measurement: dict[str, Any]) -> dict[str, Any]:
    return {
        **measurement,
        "visible": False,
        "configured": bool(measurement.get("configured", False)),
        "role": role,
        "view": view,
        "centroid_px": None,
        "source": "unavailable",
        "identity_held": False,
        "fresh_measurement": False,
        "reason": measurement.get("reason") or "target_reference_not_configured",
    }


def update_object_belief(
    belief: ObjectBelief | None,
    measurement: dict[str, Any],
    *,
    hop_px: float,
    prototype_bound: bool = False,
) -> tuple[ObjectBelief | None, dict[str, Any]]:
    """Fold one FrozenViT measurement into the persistent object.

    A peak with no prototype for this role is never an identity. A peak that
    matches the task-bound prototype may seed the first pixel. A later peak
    farther than ``hop_px`` from the committed point is ignored.
    """

    role = str(measurement.get("role") or (belief.role if belief is not None else "pickup"))
    view = str(measurement.get("view") or (belief.view if belief is not None else "column"))
    measured = None
    if bool(measurement.get("visible", False)):
        measured = _finite_px(measurement.get("centroid_px"))

    if belief is None:
        if measured is None:
            return None, _unavailable(role, view, measurement)
        if not prototype_bound:
            return None, _unavailable(
                role,
                view,
                {**measurement, "reason": "identity_not_seeded"},
            )
        live_source = (
            "open_vocab" if measurement.get("source") == "open_vocab" else "frozen_vit"
        )
        seeded = seed_object_belief(role, view, measured)
        seeded.source = live_source
        live = {
            **measurement,
            "visible": True,
            "configured": True,
            "centroid_px": measured,
            "source": live_source,
            "identity_held": False,
            "fresh_measurement": True,
            "reason": None,
        }
        return seeded, live

    committed = np.asarray(belief.centroid_px, dtype=np.float32)
    if measured is not None:
        hop = float(np.linalg.norm(measured - committed))
        if hop <= float(hop_px):
            live_source = (
                "open_vocab"
                if measurement.get("source") == "open_vocab"
                else "frozen_vit"
            )
            belief.centroid_px = measured.copy()
            belief.source = live_source
            belief.lost_rounds = 0
            live = {
                **measurement,
                "visible": True,
                "configured": True,
                "centroid_px": measured,
                "source": live_source,
                "identity_held": False,
                "fresh_measurement": True,
                "identity_hop_px": hop,
                "reason": None,
            }
            return belief, live
        belief.source = "predicted"
        belief.lost_rounds = int(belief.lost_rounds) + 1
        live = _predicted_descriptor(belief, measurement)
        live["identity_hop_px"] = hop
        live["rejected_measurement_px"] = measured
        live["reason"] = "identity_hop_blocked"
        return belief, live

    belief.source = "predicted"
    belief.lost_rounds = int(belief.lost_rounds) + 1
    return belief, _predicted_descriptor(belief, measurement)
