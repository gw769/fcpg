"""User-configured visual targets for SO-101 pick-and-place control.

The runtime plans against operator-selected semantic references, not object
classes or fixed table coordinates. Pickup and drop-zone references may be
persistent; the planner uses the same tracker with no profile root for its
session-only peg context. Low-confidence matches fail closed.

A persisted profile is **geometry-dependent in two ways at once**: the stored
``last_centroid_px`` is a planning-canvas coordinate, and the stored PNG is a
crop of the planning canvas.  Both only mean anything under the preprocessing
geometry that produced the frame the operator clicked on.  Migrating the live
path from ``squash_v0`` (anisotropic resize) to ``letterbox_v1``
(aspect-preserving scale + symmetric pad) moves the same scene content to
different pixels and rescales it in y, so a profile registered under one
protocol is not a valid registration under the other.

Profiles therefore record the geometry they were registered under (see
:meth:`VisualTargetTracker._save`, which stamps
:meth:`fcpg.letterbox_geometry.PlanningGeometry.to_dict`), and a tracker
constructed with ``required_geometry=`` refuses to load a profile that does not
match it.  Rejection is deliberately *not* repaired by resampling the crop: a
template resampled from another geometry is a synthetic registration nobody
approved.  The rejected target is simply unavailable, which surfaces as the
ordinary ``target_reference_not_configured`` / ``pickup_not_visible`` fail-closed
path, and a human re-registers it by clicking.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from fcpg.letterbox_geometry import (
    PLANNING_IMAGE_SIZE,
    PlanningGeometry,
    planning_geometry_from_protocol,
)
from fcpg.so101_calibration_continuity import (
    CALIBRATION_IDENTITY_KEYS,
    check_calibration_continuity,
)
from fcpg.so101_single_view_planning import (
    FROZEN_FEATURE_MIN_PEAK,
    RECENT_APPEARANCE_HOLD_FRAMES,
)


TARGET_ROLES = ("pickup", "drop_zone", "peg", "gripper")
TARGET_VIEWS = ("wrist", "column")
# Must stay below the runtime's execution-time target-drift tolerance (15 px):
# a continuity-supported detection farther than that from the plan-time prior
# would always be rejected at execution and the round would flap. 14 rather
# than 12: live transport rounds showed the static drop-zone match wandering
# 13.2px from its prior on self-similar cardboard, flapping visibility.
CONTINUITY_SUPPORT_RADIUS_PX = 14.0
# Upper bound on consecutive geometry-held track results before failing
# closed: enough for descent + grasp + attach confirmation (~2-3 tracks per
# round), but bounded so a stale fix cannot carry a session indefinitely.
MAX_CONSECUTIVE_HELD_TRACKS = 40

# Manifest key naming the preprocessing geometry a profile was registered
# under.  Taken from PlanningGeometry.to_dict() rather than spelled here, so
# the reader and the writer cannot drift apart.
GEOMETRY_PROTOCOL_KEY = "geometry_protocol"

# Every geometry key PlanningGeometry.to_dict() may emit.  Kept as one tuple so
# _save stamps and _load reads exactly the same field set.
GEOMETRY_KEYS = (
    GEOMETRY_PROTOCOL_KEY,
    "planning_image_size",
    "content_rect",
    "padding",
    "source_width",
    "source_height",
    "scale",
)

# The profile predates geometry versioning: written before any protocol was
# recorded, so what its pixels mean is unknown rather than merely different.
PROFILE_GEOMETRY_UNVERSIONED = "profile_geometry_unversioned"
# The profile names a geometry, and it is not the one the runtime is using.
PROFILE_GEOMETRY_MISMATCH = "profile_geometry_mismatch"
# The profile names a geometry that cannot be rebuilt, or whose derived fields
# contradict each other.
PROFILE_GEOMETRY_MALFORMED = "profile_geometry_malformed"


def _geometry_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The geometry keys actually present in a manifest payload."""

    return {key: payload[key] for key in GEOMETRY_KEYS if key in payload}


def _rejection(
    reason: str, message: str, declared: dict[str, Any], required: PlanningGeometry
) -> dict[str, Any]:
    return {
        "reason": reason,
        "message": message,
        "declared_geometry": declared,
        "required_geometry": required.to_dict(),
    }


def check_profile_geometry(
    payload: Mapping[str, Any], required: PlanningGeometry | None
) -> dict[str, Any] | None:
    """Decide whether a persisted profile may be loaded under ``required``.

    Returns ``None`` when the profile is loadable, otherwise a rejection record
    (``reason``, ``message``, ``declared_geometry``, ``required_geometry``).

    ``required=None`` means the caller has made no claim about its ingestion
    geometry, which is the historical behavior: anything loads.  Once a caller
    does declare one, an undeclared profile is rejected rather than assumed to
    match — a 224x224 crop is consistent with both protocols, so assuming is
    exactly the silent reinterpretation this gate exists to stop.

    The comparison is made on the *planning-canvas* geometry (protocol, canvas
    size, content rectangle), because that is what the stored centroid and the
    stored crop actually live in.  Two source resolutions that letterbox to the
    same content rectangle are genuinely the same planning geometry and are
    accepted; declared source resolutions that disagree are not.
    """

    if required is None:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a mapping, got {type(payload).__name__}")

    declared = _geometry_metadata(payload)
    protocol = declared.get(GEOMETRY_PROTOCOL_KEY)
    if protocol is None:
        return _rejection(
            PROFILE_GEOMETRY_UNVERSIONED,
            "visual target profile predates geometry versioning: it records no "
            f"{GEOMETRY_PROTOCOL_KEY!r}, so the geometry its stored centroids "
            "and appearance templates were cut under is unknown and cannot be "
            f"reinterpreted under {required.protocol!r}. Fresh operator "
            "re-registration on a live frame is required; the stored crop is "
            "deliberately not resampled, because a resampled template is not a "
            "re-registration.",
            declared,
            required,
        )

    try:
        persisted = planning_geometry_from_protocol(
            protocol,
            source_width=declared.get("source_width"),
            source_height=declared.get("source_height"),
            image_size=int(declared.get("planning_image_size", PLANNING_IMAGE_SIZE)),
        )
    except (TypeError, ValueError) as error:
        return _rejection(
            PROFILE_GEOMETRY_MALFORMED,
            f"visual target profile declares a geometry that cannot be rebuilt: "
            f"{error}. Re-registration is required.",
            declared,
            required,
        )

    # Internal consistency: a hand-edited or truncated profile whose derived
    # fields contradict its own source resolution is not trustworthy either.
    expected = persisted.to_dict()
    for key in ("planning_image_size", "content_rect", "padding"):
        if key in declared and _jsonable(declared[key]) != _jsonable(expected[key]):
            return _rejection(
                PROFILE_GEOMETRY_MALFORMED,
                f"visual target profile {key}={declared[key]!r} contradicts the "
                f"{protocol!r} geometry it declares (expected "
                f"{expected[key]!r}). Re-registration is required.",
                declared,
                required,
            )
    if "scale" in declared and "scale" in expected:
        if not np.isclose(float(declared["scale"]), float(expected["scale"]), atol=1e-12):
            return _rejection(
                PROFILE_GEOMETRY_MALFORMED,
                f"visual target profile scale={declared['scale']!r} contradicts "
                f"the {protocol!r} scale {expected['scale']!r}. Re-registration "
                "is required.",
                declared,
                required,
            )

    if persisted.protocol != required.protocol:
        return _rejection(
            PROFILE_GEOMETRY_MISMATCH,
            f"visual target profile was registered under "
            f"{persisted.protocol!r} but the runtime ingests "
            f"{required.protocol!r}: the same scene content sits at different "
            "planning pixels and is scaled differently, so both the stored "
            "centroid and the stored appearance template are invalid here. "
            "Re-registration by the operator is required; the crop is not "
            "resampled.",
            declared,
            required,
        )
    if persisted.content_rect != required.content_rect or (
        persisted.image_size != required.image_size
    ):
        return _rejection(
            PROFILE_GEOMETRY_MISMATCH,
            f"visual target profile declares {persisted.protocol!r} content "
            f"rect {list(persisted.content_rect)} on a "
            f"{persisted.image_size}px canvas, but the runtime uses "
            f"{list(required.content_rect)} on {required.image_size}px. "
            "Re-registration is required.",
            declared,
            required,
        )
    for key in ("source_width", "source_height"):
        mine = declared.get(key)
        theirs = getattr(required, key)
        if mine is not None and theirs is not None and int(mine) != int(theirs):
            return _rejection(
                PROFILE_GEOMETRY_MISMATCH,
                f"visual target profile was registered at {key}={mine!r} but "
                f"the runtime camera reports {theirs!r}. Re-registration is "
                "required.",
                declared,
                required,
            )
    return None


def _validate_frame(frame: np.ndarray, *, minimum_size: int = 32) -> np.ndarray:
    value = np.asarray(frame, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("target frame must be RGB [H,W,3]")
    if min(value.shape[:2]) < minimum_size:
        raise ValueError("target frame is too small")
    return value


def _validate_role(role: str) -> str:
    if role not in TARGET_ROLES:
        raise ValueError(f"unsupported target role {role!r}")
    return role


def _validate_role_view(role: str, view: str) -> tuple[str, str]:
    _validate_role(role)
    if view not in TARGET_VIEWS:
        raise ValueError(f"unsupported target view {view!r}")
    if role in {"drop_zone", "peg", "gripper"} and view != "column":
        raise ValueError(f"{role} must be configured in the column view")
    return role, view


def frozen_grounding_device() -> str:
    """Where the frozen backbone runs.

    Follows the runtime's device rather than grabbing a GPU on its own, so a
    CPU-only replay stays CPU-only and a live run does not quietly open a
    second CUDA context beside the planner's.
    """
    override = os.environ.get("SO101_GROUNDING_DEVICE")
    if override:
        return override
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - absence of torch is not this layer's problem
        return "cpu"


@dataclass
class VisualTargetReference:
    role: str
    view: str
    template_rgb: np.ndarray
    anchor_offset_px: np.ndarray
    selection_radius_px: float
    minimum_score: float = 0.56
    minimum_margin: float = 0.015
    registered_centroid_px: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_role_view(self.role, self.view)
        template = _validate_frame(self.template_rgb, minimum_size=12)
        offset = np.asarray(self.anchor_offset_px, dtype=np.float32)
        if offset.shape != (2,) or not np.all(np.isfinite(offset)):
            raise ValueError("anchor_offset_px must be a finite two-vector")
        height, width = template.shape[:2]
        if not (0 <= offset[0] < width and 0 <= offset[1] < height):
            raise ValueError("target anchor lies outside its template")
        if not np.isfinite(self.selection_radius_px) or self.selection_radius_px < 6:
            raise ValueError("selection radius must be at least 6 px")
        if not 0 < self.minimum_score <= 1:
            raise ValueError("minimum_score must be in (0,1]")
        if not 0 <= self.minimum_margin < 1:
            raise ValueError("minimum_margin must be in [0,1)")
        self.template_rgb = template.copy()
        self.anchor_offset_px = offset.copy()
        if self.registered_centroid_px is not None:
            click = np.asarray(self.registered_centroid_px, dtype=np.float32)
            if click.shape != (2,) or not np.all(np.isfinite(click)):
                raise ValueError("registered_centroid_px must be a finite two-vector")
            self.registered_centroid_px = click.copy()

    def status(self) -> dict[str, Any]:
        registered = self.registered_centroid_px
        return {
            "role": self.role,
            "view": self.view,
            "template_shape": list(self.template_rgb.shape),
            "anchor_offset_px": self.anchor_offset_px.copy(),
            "selection_radius_px": float(self.selection_radius_px),
            "minimum_score": float(self.minimum_score),
            "minimum_margin": float(self.minimum_margin),
            "registered_centroid_px": (
                None if registered is None else registered.copy()
            ),
        }


def resolve_task_profile_root(
    root: Path | None, task_class: int | None
) -> Path | None:
    """Pick the prototype store for one task class.

    ``root/class_<id>`` is the namespaced store. An untagged ``root/targets.json``
    is the legacy single store and is used only when the named store is absent.
    """

    if root is None:
        return None
    base = Path(root)
    if task_class is None:
        return base
    named = base / f"class_{int(task_class)}"
    if (named / "targets.json").is_file():
        return named
    if (base / "targets.json").is_file():
        return base
    return named


class VisualTargetTracker:
    """Track operator-selected targets without class-specific detectors."""

    def __init__(
        self,
        profile_root: Path | None = None,
        *,
        required_geometry: PlanningGeometry | None = None,
        registration_calibration: Mapping[str, Any] | None = None,
        task_class: int | None = None,
    ) -> None:
        """Build a tracker, optionally pinned to an ingestion geometry.

        Args:
            profile_root: where appearance templates persist.  ``None`` keeps
                the tracker session-only (the planner's peg/context tracker),
                in which case nothing is ever read from or written to disk and
                geometry provenance does not apply.
            required_geometry: the planning geometry the *runtime* is feeding
                this tracker.  When given, a persisted profile that was
                registered under a different geometry -- or under no recorded
                geometry at all -- is refused rather than reinterpreted, and
                newly saved profiles are stamped with it.  Defaults to ``None``,
                which preserves the historical unversioned behavior exactly, so
                no existing caller changes meaning by upgrading.
            registration_calibration: identity of the projection calibration
                the runtime has adopted, from
                :func:`fcpg.so101_calibration_continuity.calibration_identity`.
                When given, newly saved profiles record it and a persisted
                profile that recorded a DIFFERENT calibration -- or none at all
                -- is refused rather than reinterpreted: two calibrations can
                share a geometry protocol and still send the same pixel to
                different base coordinates, which the geometry gate cannot see.
                Defaults to ``None``, preserving historical behavior exactly.
        """

        if required_geometry is not None and not isinstance(
            required_geometry, PlanningGeometry
        ):
            raise TypeError(
                "required_geometry must be a PlanningGeometry, got "
                f"{type(required_geometry).__name__}"
            )
        if registration_calibration is not None:
            if not isinstance(registration_calibration, Mapping):
                raise TypeError(
                    "registration_calibration must be a mapping, got "
                    f"{type(registration_calibration).__name__}"
                )
            missing = [
                key
                for key in CALIBRATION_IDENTITY_KEYS
                if key not in registration_calibration
            ]
            if missing:
                raise ValueError(
                    f"registration_calibration is missing {missing}; build it "
                    "with calibration_identity()"
                )
            registration_calibration = {
                key: str(registration_calibration[key])
                for key in CALIBRATION_IDENTITY_KEYS
            }
        self.task_class = None if task_class is None else int(task_class)
        self.class_rejection: dict[str, Any] | None = None
        resolved = resolve_task_profile_root(
            None if profile_root is None else Path(profile_root),
            self.task_class,
        )
        self.profile_root = (
            None if resolved is None else resolved.expanduser().resolve()
        )
        self.required_geometry = required_geometry
        self.registration_calibration = registration_calibration
        #: Populated when a persisted profile was refused on geometry grounds.
        #: The targets stay UNCONFIGURED in that case; this is the record of
        #: why, for the operator-facing status payload.
        self.geometry_rejection: dict[str, Any] | None = None
        #: Populated when a persisted profile was refused because it was
        #: registered against a DIFFERENT calibration than the one in force.
        #: Same fail-closed consequence: the targets stay UNCONFIGURED and an
        #: operator must re-register by clicking.
        self.calibration_rejection: dict[str, Any] | None = None
        self.references: dict[tuple[str, str], VisualTargetReference] = {}
        self.last_centroids: dict[tuple[str, str], np.ndarray] = {}
        self.held_keys: set[tuple[str, str]] = set()
        self.held_counts: dict[tuple[str, str], int] = {}
        #: Keys that recently received a full-margin appearance refresh.
        #: Used so brief occlusion hold still counts as phase-gate reliable.
        self.full_margin_appearance_keys: set[tuple[str, str]] = set()
        #: One frozen-feature prototype per ROLE, not per (role, view): the
        #: whole point is that the description of the object survives the trip
        #: between cameras, so the far view can hand it to the near view when
        #: the arm moves in and takes the far view's line of sight away.
        self.frozen_prototypes: dict[str, Any] = {}
        #: How the live prototype for each role was last obtained, for status.
        self.frozen_prototype_source: dict[str, str] = {}
        self._frozen_grounder: Any | None = None
        self._open_vocab_grounder: Any | None = None
        if self.profile_root is not None:
            self._load()

    def configured(self, role: str, view: str) -> bool:
        _validate_role_view(role, view)
        return (role, view) in self.references

    def registered_click_px(self, role: str, view: str) -> np.ndarray | None:
        """The operator click. Never the last track() centroid."""

        reference = self.references.get(_validate_role_view(role, view))
        if reference is None or reference.registered_centroid_px is None:
            return None
        return np.asarray(reference.registered_centroid_px, dtype=np.float32).copy()

    def register(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        centroid_px: np.ndarray,
        *,
        radius_px: float,
        minimum_score: float = 0.56,
        minimum_margin: float = 0.015,
        persist: bool = True,
        source: str = "operator_reference",
    ) -> dict[str, Any]:
        role, view = _validate_role_view(role, view)
        rgb = _validate_frame(frame)
        point = np.asarray(centroid_px, dtype=np.float32)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("target centroid must be a finite two-vector")
        height, width = rgb.shape[:2]
        if point[0] < 0 or point[1] < 0 or point[0] >= width or point[1] >= height:
            raise ValueError("target centroid lies outside the current frame")
        if not np.isfinite(radius_px) or radius_px < 6 or radius_px > min(height, width) / 2:
            raise ValueError("target radius is outside the supported range")
        radius = int(round(float(radius_px)))
        cx, cy = map(float, point)
        x0 = max(0, int(np.floor(cx - radius)))
        y0 = max(0, int(np.floor(cy - radius)))
        x1 = min(width, int(np.ceil(cx + radius + 1)))
        y1 = min(height, int(np.ceil(cy + radius + 1)))
        template = rgb[y0:y1, x0:x1].copy()
        if min(template.shape[:2]) < 12:
            raise ValueError("selected target template is too small")
        reference = VisualTargetReference(
            role=role,
            view=view,
            template_rgb=template,
            anchor_offset_px=np.asarray([cx - x0, cy - y0], dtype=np.float32),
            selection_radius_px=float(radius_px),
            minimum_score=float(minimum_score),
            minimum_margin=float(minimum_margin),
            registered_centroid_px=point.copy(),
        )
        key = (role, view)
        if key == ("pickup", "column"):
            # A new global pickup selection may identify a different object.
            # Its old eye-in-hand template must not silently satisfy the
            # near-field registration gate for the new target.
            wrist_key = ("pickup", "wrist")
            self.references.pop(wrist_key, None)
            self.last_centroids.pop(wrist_key, None)
            self.held_keys.discard(wrist_key)
            self.held_counts.pop(wrist_key, None)
            self.full_margin_appearance_keys.discard(wrist_key)
            self.frozen_prototypes.pop("pickup", None)
            self.frozen_prototype_source.pop("pickup", None)
        self.references[key] = reference
        self.last_centroids[key] = point.copy()
        if role == "drop_zone":
            self.frozen_prototypes.pop("drop_zone", None)
            self.frozen_prototype_source.pop("drop_zone", None)
        self.held_keys.discard(key)
        self.held_counts.pop(key, None)
        self.full_margin_appearance_keys.add(key)
        if persist:
            self._save()
        return {
            "visible": True,
            "configured": True,
            "role": role,
            "view": view,
            "centroid_px": point.copy(),
            "bbox_xyxy": np.asarray([x0, y0, x1, y1], dtype=np.float32),
            "area_px": int((x1 - x0) * (y1 - y0)),
            "score": 1.0,
            "margin": 1.0,
            "source": str(source),
        }

    def clear(self, role: str, view: str | None = None) -> None:
        if role not in TARGET_ROLES:
            raise ValueError(f"unsupported target role {role!r}")
        views = TARGET_VIEWS if view is None else (view,)
        for candidate_view in views:
            if role in {"drop_zone", "peg", "gripper"} and candidate_view != "column":
                continue
            _validate_role_view(role, candidate_view)
            self.references.pop((role, candidate_view), None)
            self.last_centroids.pop((role, candidate_view), None)
            self.held_keys.discard((role, candidate_view))
            self.held_counts.pop((role, candidate_view), None)
            self.full_margin_appearance_keys.discard((role, candidate_view))
        self._save()


    @property
    def frozen_grounder(self) -> Any:
        """The shared frozen backbone, built on first use."""
        if self._frozen_grounder is None:
            from fcpg.so101_frozen_vit_grounding import FrozenViTGrounder

            self._frozen_grounder = FrozenViTGrounder(
                device=frozen_grounding_device()
            )
        return self._frozen_grounder

    def frozen_prototype(self, role: str) -> Any | None:
        """The live description of ``role``, bootstrapped from registration.

        The operator's crop is the starting point because it is the only thing
        they were ever asked for. It is a small picture from one camera at one
        distance, so it is a weak description of how the object looks from the
        wrist at grasp range -- measured, 7.2 px in the view it came from and
        37.8 px in the other one. :meth:`refresh_frozen_prototype` replaces it
        with the object's own patches as soon as some view holds it clearly.
        """
        role = _validate_role(role)
        cached = self.frozen_prototypes.get(role)
        if cached is not None:
            return cached
        for (candidate_role, _view), reference in sorted(self.references.items()):
            if candidate_role != role:
                continue
            prototype = self.frozen_grounder.prototype_from_crop(
                reference.template_rgb
            )
            self.frozen_prototypes[role] = prototype
            self.frozen_prototype_source[role] = "registration_crop"
            return prototype
        return None

    def refresh_frozen_prototype(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        centroid_px: np.ndarray,
        radius_px: float,
    ) -> None:
        """Re-describe ``role`` from a view that is holding it clearly right now.

        Called by whichever view is currently confident, so the description
        follows the object through the approach: far and small at the start,
        near and large by the time the jaws are around it. Nothing is fitted --
        this is a pooled read of frozen features, not a training step -- so it
        costs one forward pass and cannot drift into a different object unless
        the caller was already confidently wrong about where this one is.
        The vector stays in process memory for this episode. It is not a
        disk crop library.
        """
        role, view = _validate_role_view(role, view)
        rgb = _validate_frame(frame)
        self.frozen_prototypes[role] = self.frozen_grounder.prototype_from_frame(
            rgb, np.asarray(centroid_px, dtype=np.float32), float(radius_px)
        )
        self.frozen_prototype_source[role] = f"{view}_live"

    def session_appearance_ready(self, role: str) -> bool:
        """True when this episode already bound a live prototype for ``role``."""

        source = self.frozen_prototype_source.get(_validate_role(role))
        return bool(source) and str(source).endswith("_live")

    def locate_by_frozen_features(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        *,
        gripper_center_px: np.ndarray | None = None,
        jaw_pair_px: np.ndarray | None = None,
        search_centre_px: np.ndarray | None = None,
        search_radius_px: float | None = None,
        view_specific_prototype: bool = False,
        prototype: Any | None = None,
    ) -> dict[str, Any]:
        """Find ``role`` in ``view`` by matching frozen general ViT features.

        Carries no assumption about what the object looks like: the backbone is
        pretrained, frozen, and shared by every view and every task, and the
        only per-task state is one prototype vector. That is what lets the same
        call serve the wrist, whose crop the correlation matcher cannot use
        because it is fixed-scale and the wrist sees the object many times
        larger, and lets the whole thing keep working on a sensor with no
        colour at all.

        When jaws occlude the object, pass ``jaw_pair_px`` so claw pixels are
        kept out of the support / 重心 (geometry, not colour).

        The column-only roles (drop zone, peg, gripper) are registered in the
        column view only; their role prototype may still be matched in the
        wrist, which is what the shared prototype is for (a view-specific
        prototype there has no registration and stays unconfigured).

        ``prototype`` matches a caller's own prototype vector instead of the
        role's or the view's registration (fcpg/v21_target_tracker.py: the drop
        zone's wrist-view registration, kept out of the persisted profile).
        """
        role = _validate_role(role)
        if view not in TARGET_VIEWS:
            raise ValueError(f"unsupported target view {view!r}")
        if prototype is not None and view_specific_prototype:
            raise ValueError("an explicit prototype excludes the view-specific one")
        if view_specific_prototype:
            role, view = _validate_role_view(role, view)
        rgb = _validate_frame(frame)
        if prototype is not None:
            pass
        elif view_specific_prototype:
            reference = self.references.get((role, view))
            # Reference identity invalidates the cache when a registration is replaced.
            cache = getattr(self, '_view_prototype_cache', {})
            key = (role, view)
            cached = cache.get(key)
            if reference is None:
                prototype = None
            elif cached is not None and cached[0] is reference:
                prototype = cached[1]
            else:
                prototype = self.frozen_grounder.prototype_from_crop(reference.template_rgb)
                cache[key] = (reference, prototype)
            self._view_prototype_cache = cache
        else:
            prototype = self.frozen_prototype(role)
        if prototype is None:
            return self._not_visible_frozen(
                role, view, "target_reference_not_configured"
            )
        hit = self.frozen_grounder.locate(
            rgb,
            prototype,
            search_centre_px=(
                None if search_centre_px is None else tuple(np.asarray(search_centre_px))
            ),
            search_radius_px=search_radius_px,
            jaw_pair_px=jaw_pair_px,
        )
        if hit.peak < FROZEN_FEATURE_MIN_PEAK:
            rejected = self._not_visible_frozen(role, view, "peak_below_threshold")
            # Retain measured evidence without exposing a rejected location.
            rejected.update(score=float(hit.peak), separation=float(hit.separation))
            return rejected
        from .so101_frozen_vit_grounding import support_is_clipped

        centroid = np.asarray(hit.centroid_px, dtype=np.float32)
        mask = None if hit.support_mask is None else np.asarray(hit.support_mask, dtype=bool)
        clipped = bool(mask is None or support_is_clipped(mask))
        aimable = bool(mask is not None and int(mask.sum()) >= 32 and not clipped)
        relative = (
            None
            if gripper_center_px is None
            else centroid - np.asarray(gripper_center_px, dtype=np.float32)
        )
        return {
            "visible": True,
            "configured": True,
            "role": role,
            "view": view,
            "centroid_px": centroid,
            "bbox_xyxy": None,
            "score": float(hit.peak),
            "second_score": -1.0,
            # A dense similarity field has no runner-up box to difference
            # against. What it has instead is how far the peak rose above the
            # rest of the frame, which is reported as `separation` and left for
            # the grounding layer to weigh against everything else it knows.
            "margin": None,
            "separation": float(hit.separation),
            # Which way the object lies and how wide it is square to that. A
            # two-fingered jaw has to close across an object, not along it, so
            # this is what decides both where to grasp and how to turn the
            # wrist before doing it.
            "axis_px": np.asarray(hit.shape.axis_px, dtype=np.float32),
            "grasp_axis_px": np.asarray(hit.shape.grasp_axis_px, dtype=np.float32),
            "width_px": float(hit.shape.width_px),
            "length_px": float(hit.shape.length_px),
            "elongation": float(hit.shape.elongation),
            "source": "frozen_vit_features",
            "centroid_source": "shape_prior_uniform_density_cog",
            "grasp_kind": "shape_prior_uniform_density_cog",
            "cog_px": centroid,
            "symmetry_iou": float(hit.symmetry_iou),
            "support_mask": None if hit.support_mask is None else np.asarray(hit.support_mask, dtype=bool),
            "area_px": 0 if hit.support_mask is None else int(np.asarray(hit.support_mask).sum()),
            "support_clipped": clipped,
            "aimable": aimable,
            "prototype_source": self.frozen_prototype_source.get(role, "unknown"),
            "gripper_center_px": (
                None
                if gripper_center_px is None
                else np.asarray(gripper_center_px, dtype=np.float32)
            ),
            "relative_centroid_px": relative,
            "reason": None,
        }

    @staticmethod
    def _not_visible_frozen(role: str, view: str, reason: str) -> dict[str, Any]:
        return {
            "visible": False,
            "configured": reason != "target_reference_not_configured",
            "role": role,
            "view": view,
            "centroid_px": None,
            "bbox_xyxy": None,
            "area_px": 0,
            "score": -1.0,
            "second_score": -1.0,
            "margin": None,
            "separation": 0.0,
            "source": "frozen_vit_features",
            "prototype_source": "unavailable",
            "gripper_center_px": None,
            "relative_centroid_px": None,
            "reason": reason,
        }

    @property
    def open_vocab_grounder(self) -> Any:
        if self._open_vocab_grounder is None:
            from fcpg.so101_open_vocab_grounding import FrozenOpenVocabGrounder

            self._open_vocab_grounder = FrozenOpenVocabGrounder(
                device=frozen_grounding_device()
            )
        return self._open_vocab_grounder

    def locate_by_open_vocab(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        *,
        query: str,
        gripper_center_px: np.ndarray | None = None,
        search_centre_px: np.ndarray | None = None,
        search_radius_px: float | None = None,
    ) -> dict[str, Any]:
        """Find ``role`` by matching the task's text phrase in the frame."""

        role, view = _validate_role_view(role, view)
        rgb = _validate_frame(frame)
        phrase = str(query).strip()
        if not phrase:
            return self._not_visible_open_vocab(
                role, view, phrase, "target_query_not_configured"
            )
        hit = self.open_vocab_grounder.locate(
            rgb,
            phrase,
            search_centre_px=(
                None if search_centre_px is None else tuple(np.asarray(search_centre_px))
            ),
            search_radius_px=search_radius_px,
        )
        if hit is None:
            return self._not_visible_open_vocab(
                role, view, phrase, "open_vocab_not_found"
            )
        return self._open_vocab_payload(
            role,
            view,
            phrase,
            hit,
            gripper_center_px=gripper_center_px,
        )

    def locate_roles_by_open_vocab(
        self,
        view: str,
        frame: np.ndarray,
        queries: Mapping[str, str | tuple[str, ...] | list[str]],
        *,
        content_rect: tuple[int, int, int, int] | None = None,
        gripper_center_px: np.ndarray | None = None,
    ) -> dict[str, dict[str, Any]]:
        """One forward, every task phrase, exclusive boxes."""

        rgb = _validate_frame(frame)
        assigned = self.open_vocab_grounder.locate_many(
            rgb, queries, content_rect=content_rect
        )
        located: dict[str, dict[str, Any]] = {}
        for role, phrase in queries.items():
            role_name, view_name = _validate_role_view(str(role), view)
            if isinstance(phrase, str):
                label = phrase.strip()
            else:
                label = ", ".join(str(item).strip() for item in phrase if str(item).strip())
            hit = assigned.hits.get(role_name)
            if hit is None:
                located[role_name] = self._not_visible_open_vocab(
                    role_name,
                    view_name,
                    label,
                    assigned.reasons.get(role_name, "open_vocab_not_found"),
                )
                continue
            located[role_name] = self._open_vocab_payload(
                role_name,
                view_name,
                hit.query or label,
                hit,
                gripper_center_px=gripper_center_px,
            )
        return located

    def _open_vocab_payload(
        self,
        role: str,
        view: str,
        phrase: str,
        hit: Any,
        *,
        gripper_center_px: np.ndarray | None,
    ) -> dict[str, Any]:
        centroid = np.asarray(hit.centroid_px, dtype=np.float32)
        relative = (
            None
            if gripper_center_px is None
            else centroid - np.asarray(gripper_center_px, dtype=np.float32)
        )
        return {
            "visible": True,
            "configured": True,
            "role": role,
            "view": view,
            "centroid_px": centroid,
            "bbox_xyxy": np.asarray(hit.bbox_xyxy, dtype=np.float32),
            "area_px": int(
                max(0.0, hit.bbox_xyxy[2] - hit.bbox_xyxy[0])
                * max(0.0, hit.bbox_xyxy[3] - hit.bbox_xyxy[1])
            ),
            "score": float(hit.score),
            "second_score": float(getattr(hit, "second_score", -1.0)),
            "margin": getattr(hit, "margin", None),
            "source": "open_vocab",
            "query": phrase,
            "gripper_center_px": (
                None
                if gripper_center_px is None
                else np.asarray(gripper_center_px, dtype=np.float32)
            ),
            "relative_centroid_px": relative,
            "reason": None,
        }

    @staticmethod
    def _not_visible_open_vocab(
        role: str, view: str, query: str, reason: str
    ) -> dict[str, Any]:
        return {
            "visible": False,
            "configured": reason != "target_query_not_configured",
            "role": role,
            "view": view,
            "centroid_px": None,
            "bbox_xyxy": None,
            "area_px": 0,
            "score": -1.0,
            "source": "open_vocab",
            "query": query,
            "gripper_center_px": None,
            "relative_centroid_px": None,
            "reason": reason,
        }


    def status(self) -> dict[str, Any]:
        configured = []
        for key in sorted(self.references):
            row = self.references[key].status()
            centroid = self.last_centroids.get(key)
            row["last_centroid_px"] = None if centroid is None else centroid.copy()
            configured.append(row)
        return {
            "profile_root": None if self.profile_root is None else self.profile_root,
            "task_class": self.task_class,
            "class_rejection": (
                None if self.class_rejection is None else dict(self.class_rejection)
            ),
            "configured": configured,
            "pickup_ready": ("pickup", "column") in self.references,
            "drop_zone_ready": ("drop_zone", "column") in self.references,
            "wrist_pickup_ready": ("pickup", "wrist") in self.references,
            "peg_ready": ("peg", "column") in self.references,
            "required_geometry": (
                None
                if self.required_geometry is None
                else self.required_geometry.to_dict()
            ),
            "registration_calibration": (
                None
                if self.registration_calibration is None
                else dict(self.registration_calibration)
            ),
            # Non-None means a persisted profile was refused: the *_ready flags
            # above are False because the targets were never loaded, and an
            # operator must re-register them.
            "geometry_rejection": (
                None if self.geometry_rejection is None else dict(self.geometry_rejection)
            ),
            # Same fail-closed meaning, for a calibration change rather than a
            # geometry change: the pilot continuation is invalid until an
            # operator re-registers.
            "calibration_rejection": (
                None
                if self.calibration_rejection is None
                else dict(self.calibration_rejection)
            ),
        }


    def _save(self) -> None:
        if self.profile_root is None:
            return
        self.profile_root.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for (role, view), reference in sorted(self.references.items()):
            filename = f"{role}_{view}.png"
            Image.fromarray(reference.template_rgb, mode="RGB").save(
                self.profile_root / filename
            )
            row = reference.status()
            row["template_file"] = filename
            centroid = self.last_centroids.get((role, view))
            row["last_centroid_px"] = (
                None if centroid is None else centroid.tolist()
            )
            registered = reference.registered_centroid_px
            row["registered_centroid_px"] = (
                None if registered is None else registered.tolist()
            )
            rows.append(row)
        payload: dict[str, Any] = {"version": 1, "targets": rows}
        if self.task_class is not None:
            payload["task_class"] = int(self.task_class)
        if self.required_geometry is not None:
            # Stamp the geometry these crops and centroids were cut under.  The
            # field names come from PlanningGeometry.to_dict(), not from string
            # literals here, so writer and reader cannot drift.
            payload.update(self.required_geometry.to_dict())
        if self.registration_calibration is not None:
            # Stamp WHICH calibration these clicks backproject through, as plain
            # path/name/protocol strings.  Geometry alone does not identify a
            # calibration: two letterbox_v1 fits send one pixel to two
            # different base coordinates.
            payload.update(self.registration_calibration)
        (self.profile_root / "targets.json").write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # What is on disk now matches the geometry AND the calibration in force,
        # so any earlier rejection is resolved: this is the operator
        # re-registration path, the only thing that may clear either rejection.
        self.geometry_rejection = None
        self.calibration_rejection = None

    def _load(self) -> None:
        assert self.profile_root is not None
        manifest = self.profile_root / "targets.json"
        if not manifest.is_file():
            return
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("version") != 1 or not isinstance(payload.get("targets"), list):
            raise ValueError("unsupported visual target profile")
        stored_class = payload.get("task_class")
        if (
            stored_class is not None
            and self.task_class is not None
            and int(stored_class) != int(self.task_class)
        ):
            self.class_rejection = {
                "reason": "task_class_mismatch",
                "stored_task_class": int(stored_class),
                "requested_task_class": int(self.task_class),
            }
            return
        # Geometry is checked BEFORE any template is opened.  A refused profile
        # must leave no reference, no prior and no decoded crop behind: the
        # target is simply unconfigured, which fails closed through the normal
        # target_reference_not_configured / pickup_not_visible path and waits
        # for a human click.  Nothing here rewrites the profile on disk.
        rejection = check_profile_geometry(payload, self.required_geometry)
        if rejection is not None:
            self.geometry_rejection = rejection
            return
        # Calibration continuity is checked next, and before any template is
        # opened, for the same reason: a profile registered against a different
        # projection calibration holds clicks that backproject elsewhere, so it
        # must leave no reference and no prior behind.  Fails closed through the
        # ordinary target_reference_not_configured / pickup_not_visible path and
        # waits for a human click.  Nothing here rewrites the profile on disk.
        continuity = check_calibration_continuity(payload, self.registration_calibration)
        if continuity is not None:
            self.calibration_rejection = continuity
            return
        for row in payload["targets"]:
            template_path = self.profile_root / str(row["template_file"])
            template = np.asarray(Image.open(template_path).convert("RGB"), dtype=np.uint8)
            registered = row.get("registered_centroid_px")
            registered_px = None
            if registered is not None:
                click = np.asarray(registered, dtype=np.float32)
                if click.shape == (2,) and np.all(np.isfinite(click)):
                    registered_px = click
            reference = VisualTargetReference(
                role=str(row["role"]),
                view=str(row["view"]),
                template_rgb=template,
                anchor_offset_px=np.asarray(row["anchor_offset_px"], dtype=np.float32),
                selection_radius_px=float(row["selection_radius_px"]),
                minimum_score=float(row["minimum_score"]),
                minimum_margin=float(row["minimum_margin"]),
                registered_centroid_px=registered_px,
            )
            self.references[(reference.role, reference.view)] = reference
            # Restore the tracking prior: without it a static target whose
            # template matches near-perfectly (score ~0.999) but sits on
            # self-similar texture (margin below the floor) can never pass
            # the continuity path after a restart.
            centroid = row.get("last_centroid_px")
            if centroid is not None:
                point = np.asarray(centroid, dtype=np.float32)
                if point.shape == (2,) and np.all(np.isfinite(point)):
                    self.last_centroids[(reference.role, reference.view)] = (
                        point
                    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "TARGET_ROLES",
    "TARGET_VIEWS",
    "GEOMETRY_PROTOCOL_KEY",
    "GEOMETRY_KEYS",
    "PROFILE_GEOMETRY_UNVERSIONED",
    "PROFILE_GEOMETRY_MISMATCH",
    "PROFILE_GEOMETRY_MALFORMED",
    "check_profile_geometry",
    "resolve_task_profile_root",
    "VisualTargetReference",
    "VisualTargetTracker",
]
