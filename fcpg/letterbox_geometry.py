"""Canonical raw-pixel <-> ``letterbox_v1`` planning-pixel geometry.

This module is the **single source of truth** for the aspect-preserving square
preprocessing that every trained FCPG/SO-101 component saw.  All caches were
built with ffmpeg ``force_original_aspect_ratio=decrease`` + pad (measured
28/28 pad rows on the 4:3 caches, 49/49 on ``drawing_front_224``), and the
validated runtime implementation is ``v10/live_pilot.py:65-88
_letterbox_rgb``.  The convention reproduced here is exactly that one::

    scale           = min(S / width, S / height)          # ONE uniform scale
    content_width   = max(1, min(S, int(round(width  * scale))))
    content_height  = max(1, min(S, int(round(height * scale))))
    pad_x           = (S - content_width)  // 2           # integer floor div
    pad_y           = (S - content_height) // 2

with ``S = 224`` and the scaled content pasted onto a black canvas at
``(pad_x, pad_y)``.  The rounding conventions matter: content size is
round-then-clamp, the pad offset is floor division.  ``pad_x``/``pad_y`` must
equal where PIL actually pastes, otherwise every pixel derived from a live
frame is biased.  ``tests/test_letterbox_geometry.py`` pins this against real
PIL output.

Forward map (raw camera pixel -> ``letterbox_v1`` planning pixel)::

    u = pad_x + x_raw * scale
    v = pad_y + y_raw * scale

Legacy ``squash_v0`` (the pre-migration live path, ``PIL.resize((224, 224))``)
used an anisotropic map ``u_s = x_raw * S / W``, ``v_s = y_raw * S / H``.  The
``squash_v0 -> letterbox_v1`` conversion is therefore a per-axis affine::

    u_letterbox = (scale * W / S) * u_squash + pad_x
    v_letterbox = (scale * H / S) * v_squash + pad_y

Verified coefficients (see the test module, which recomputes them):

===========  =======  ==============  =====  =====  ==================  ===================
source       scale    content         pad_x  pad_y  u_lb from u_squash  v_lb from v_squash
===========  =======  ==============  =====  =====  ==================  ===================
640 x 480    0.35     224 x 168       0      28     1.0000 u + 0        0.7500 v + 28
1280 x 720   0.175    224 x 126       0      49     1.0000 u + 0        0.5625 v + 49
640 x 360    0.35     224 x 126       0      49     1.0000 u + 0        0.5625 v + 49
===========  =======  ==============  =====  =====  ==================  ===================

Under ``letterbox_v1`` there is a **single uniform scale**.  Independent X/Y
scales are a ``squash_v0``-era artifact and are deliberately not exposed; this
module neither imports nor depends on ``fcpg.error_field``.

Out-of-region behavior is **fail-closed, in both directions**.  A planning
pixel in the pad band corresponds to no raw pixel at all, and a raw pixel
outside ``[0, W] x [0, H]`` corresponds to no planning pixel inside the
content region, so both raise :class:`ValueError` rather than returning a
coordinate that merely looks plausible.  Callers that need to filter instead
of catching should test :meth:`LetterboxGeometry.contains_planning` /
:meth:`LetterboxGeometry.contains_raw` first.  Nothing here silently clamps a
pad-band pixel onto the image edge: a safety path must never fail open.

Coordinates use the corner convention: ``x`` in ``[0, W]``, ``y`` in
``[0, H]``, where integer ``x`` is the left edge of column ``x``.  Bounds
checks allow a tolerance of ``_BOUNDS_TOL_PX`` (1e-6 px) so that a
round-tripped coordinate landing a few float ULPs outside the box is not
rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

#: Geometry protocol implemented by this module.
LETTERBOX_V1 = "letterbox_v1"

#: Legacy pre-migration protocol: anisotropic resize to (S, S), no padding.
SQUASH_V0 = "squash_v0"

#: Square planning-image side length every trained component consumed.
PLANNING_IMAGE_SIZE = 224

#: Slack (in pixels) allowed by bounds checks, to absorb float round-off.
_BOUNDS_TOL_PX = 1e-6

__all__ = [
    "LETTERBOX_V1",
    "SQUASH_V0",
    "PLANNING_IMAGE_SIZE",
    "LetterboxGeometry",
    "letterbox_geometry",
    "letterbox_rgb_image",
    "PlanningGeometry",
    "planning_geometry_from_protocol",
    "raw_to_planning",
    "planning_to_raw",
    "squash_to_planning",
    "planning_to_squash",
    "require_letterbox_v1",
]


def _as_points(values: Any, *, label: str) -> tuple[np.ndarray, bool]:
    """Normalize ``(2,)`` or ``(N, 2)`` input to a float64 ``(N, 2)`` array."""

    points = np.asarray(values, dtype=np.float64)
    single = points.ndim == 1
    if single:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"{label} must have shape (2,) or (N, 2), got {np.shape(values)}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{label} must be finite")
    return points, single


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be an int (pixel count), got {type(value).__name__}"
        )
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


@dataclass(frozen=True)
class LetterboxGeometry:
    """The ``letterbox_v1`` descriptor for one source resolution.

    Only the source resolution and the planning-image size are stored; every
    derived quantity (``scale``, content size, padding) is computed from them
    on access, so a :class:`LetterboxGeometry` cannot hold an internally
    inconsistent geometry.
    """

    source_width: int
    source_height: int
    image_size: int = PLANNING_IMAGE_SIZE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_width", _require_positive_int(self.source_width, name="source_width")
        )
        object.__setattr__(
            self,
            "source_height",
            _require_positive_int(self.source_height, name="source_height"),
        )
        object.__setattr__(
            self, "image_size", _require_positive_int(self.image_size, name="image_size")
        )

    # ---------------------------------------------------------------- descriptor

    @property
    def protocol(self) -> str:
        return LETTERBOX_V1

    @property
    def scale(self) -> float:
        """The single uniform raw-pixel -> planning-pixel scale."""

        return min(
            self.image_size / self.source_width, self.image_size / self.source_height
        )

    @property
    def content_width(self) -> int:
        """Painted content width, matching the reference resize rounding."""

        return max(1, min(self.image_size, int(round(self.source_width * self.scale))))

    @property
    def content_height(self) -> int:
        """Painted content height, matching the reference resize rounding."""

        return max(1, min(self.image_size, int(round(self.source_height * self.scale))))

    @property
    def content_size(self) -> tuple[int, int]:
        return (self.content_width, self.content_height)

    @property
    def pad_x(self) -> int:
        """Left pad column count; where PIL pastes the content."""

        return (self.image_size - self.content_width) // 2

    @property
    def pad_y(self) -> int:
        """Top pad row count; where PIL pastes the content."""

        return (self.image_size - self.content_height) // 2

    @property
    def padding(self) -> tuple[int, int]:
        return (self.pad_x, self.pad_y)

    @property
    def content_extent_x(self) -> float:
        """Exact affine width of the content, ``W * scale`` (unrounded)."""

        return self.source_width * self.scale

    @property
    def content_extent_y(self) -> float:
        """Exact affine height of the content, ``H * scale`` (unrounded)."""

        return self.source_height * self.scale

    @property
    def planning_bounds(self) -> tuple[float, float, float, float]:
        """``(left, top, right, bottom)`` of the invertible content region.

        These are the *affine* bounds.  They can be up to 0.5 px tighter than
        :meth:`content_box`, because the painted content size is rounded to
        whole pixels while the affine map is not.  The affine bounds are the
        authoritative ones for coordinate work: outside them the inverse map
        is not defined by the forward map.
        """

        return (
            float(self.pad_x),
            float(self.pad_y),
            self.pad_x + self.content_extent_x,
            self.pad_y + self.content_extent_y,
        )

    def content_box(self) -> tuple[int, int, int, int]:
        """Integer ``(left, top, right, bottom)`` box PIL pastes content into."""

        return (
            self.pad_x,
            self.pad_y,
            self.pad_x + self.content_width,
            self.pad_y + self.content_height,
        )

    # ----------------------------------------------------------------- predicates

    def contains_raw(self, points_xy: Any) -> np.ndarray | np.bool_:
        """Whether raw ``(x, y)`` lies inside ``[0, W] x [0, H]``."""

        points, single = _as_points(points_xy, label="points_xy")
        inside = self._inside(
            points, 0.0, 0.0, float(self.source_width), float(self.source_height)
        )
        return inside[0] if single else inside

    def contains_planning(self, points_uv: Any) -> np.ndarray | np.bool_:
        """Whether planning ``(u, v)`` lies inside the content region."""

        points, single = _as_points(points_uv, label="points_uv")
        inside = self._inside(points, *self.planning_bounds)
        return inside[0] if single else inside

    @staticmethod
    def _inside(
        points: np.ndarray, left: float, top: float, right: float, bottom: float
    ) -> np.ndarray:
        low = np.array([left, top], dtype=np.float64) - _BOUNDS_TOL_PX
        high = np.array([right, bottom], dtype=np.float64) + _BOUNDS_TOL_PX
        return np.all((points >= low) & (points <= high), axis=1)

    @classmethod
    def _require_inside(
        cls,
        points: np.ndarray,
        left: float,
        top: float,
        right: float,
        bottom: float,
        *,
        label: str,
    ) -> None:
        outside = ~cls._inside(points, left, top, right, bottom)
        if bool(np.any(outside)):
            first = points[int(np.argmax(outside))]
            raise ValueError(
                f"{label} ({first[0]!r}, {first[1]!r}) lies outside the valid "
                f"region [{left}, {right}] x [{top}, {bottom}]; "
                f"{int(outside.sum())} of {points.shape[0]} point(s) rejected"
            )

    # ----------------------------------------------------------------- transforms

    def raw_to_planning(self, points_xy: Any) -> np.ndarray:
        """Map raw camera ``(x, y)`` to ``letterbox_v1`` planning ``(u, v)``.

        Raises:
            ValueError: if any point lies outside ``[0, W] x [0, H]``.
        """

        points, single = _as_points(points_xy, label="points_xy")
        self._require_inside(
            points,
            0.0,
            0.0,
            float(self.source_width),
            float(self.source_height),
            label="raw pixel",
        )
        offset = np.array([self.pad_x, self.pad_y], dtype=np.float64)
        result = points * self.scale + offset
        return result[0] if single else result

    def planning_to_raw(self, points_uv: Any) -> np.ndarray:
        """Map ``letterbox_v1`` planning ``(u, v)`` back to raw ``(x, y)``.

        Raises:
            ValueError: if any point lies in the pad band, i.e. outside the
                content region.  Such a point has no raw pre-image, so this
                fails closed instead of returning an edge coordinate.
        """

        points, single = _as_points(points_uv, label="points_uv")
        left, top, right, bottom = self.planning_bounds
        self._require_inside(
            points, left, top, right, bottom, label=f"{LETTERBOX_V1} planning pixel"
        )
        offset = np.array([self.pad_x, self.pad_y], dtype=np.float64)
        result = (points - offset) / self.scale
        # Points that passed validation can still sit a few float ULPs (at most
        # _BOUNDS_TOL_PX / scale) beyond the frame; snap those onto it.
        np.clip(
            result,
            (0.0, 0.0),
            (float(self.source_width), float(self.source_height)),
            out=result,
        )
        return result[0] if single else result

    def squash_affine(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((ax, bx), (ay, by))`` for the ``squash_v0`` -> ``letterbox_v1`` map.

        ``u_letterbox = ax * u_squash + bx`` and
        ``v_letterbox = ay * v_squash + by``.
        """

        return (
            (self.scale * self.source_width / self.image_size, float(self.pad_x)),
            (self.scale * self.source_height / self.image_size, float(self.pad_y)),
        )

    def squash_to_planning(self, points_uv: Any) -> np.ndarray:
        """Map legacy ``squash_v0`` planning pixels to ``letterbox_v1`` pixels."""

        points, single = _as_points(points_uv, label="points_uv")
        self._require_inside(
            points,
            0.0,
            0.0,
            float(self.image_size),
            float(self.image_size),
            label=f"{SQUASH_V0} planning pixel",
        )
        (ax, bx), (ay, by) = self.squash_affine()
        result = points * np.array([ax, ay]) + np.array([bx, by])
        return result[0] if single else result

    def planning_to_squash(self, points_uv: Any) -> np.ndarray:
        """Map ``letterbox_v1`` planning pixels back to legacy ``squash_v0``.

        Raises:
            ValueError: if any point lies in the pad band.
        """

        points, single = _as_points(points_uv, label="points_uv")
        left, top, right, bottom = self.planning_bounds
        self._require_inside(
            points, left, top, right, bottom, label=f"{LETTERBOX_V1} planning pixel"
        )
        (ax, bx), (ay, by) = self.squash_affine()
        result = (points - np.array([bx, by])) / np.array([ax, ay])
        np.clip(
            result, (0.0, 0.0), (float(self.image_size), float(self.image_size)), out=result
        )
        return result[0] if single else result

    # --------------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        """Serialize for calibration profiles, tagged with the protocol."""

        return {
            "protocol": self.protocol,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "image_size": self.image_size,
            "scale": self.scale,
            "content_width": self.content_width,
            "content_height": self.content_height,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LetterboxGeometry":
        """Rebuild from :meth:`to_dict`, hard-rejecting non-``letterbox_v1`` data.

        Derived keys, if present, are re-derived and cross-checked; a mismatch
        raises rather than being silently reinterpreted.
        """

        if not isinstance(payload, Mapping):
            raise TypeError(f"payload must be a mapping, got {type(payload).__name__}")
        require_letterbox_v1(payload.get("protocol"))
        for key in ("source_width", "source_height"):
            if key not in payload:
                raise ValueError(f"payload is missing required key {key!r}")
        geometry = cls(
            source_width=payload["source_width"],
            source_height=payload["source_height"],
            image_size=payload.get("image_size", PLANNING_IMAGE_SIZE),
        )
        for key in ("content_width", "content_height", "pad_x", "pad_y"):
            if key in payload and int(payload[key]) != getattr(geometry, key):
                raise ValueError(
                    f"payload {key}={payload[key]!r} disagrees with the "
                    f"{LETTERBOX_V1} geometry for "
                    f"{geometry.source_width}x{geometry.source_height} "
                    f"({getattr(geometry, key)})"
                )
        if "scale" in payload and not np.isclose(
            float(payload["scale"]), geometry.scale, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"payload scale={payload['scale']!r} disagrees with the "
                f"{LETTERBOX_V1} scale {geometry.scale!r}"
            )
        return geometry


def letterbox_geometry(
    source_width: int, source_height: int, image_size: int = PLANNING_IMAGE_SIZE
) -> LetterboxGeometry:
    """Build the :class:`LetterboxGeometry` for a source resolution."""

    return LetterboxGeometry(
        source_width=source_width, source_height=source_height, image_size=image_size
    )


def raw_to_planning(
    points_xy: Any,
    source_width: int,
    source_height: int,
    image_size: int = PLANNING_IMAGE_SIZE,
) -> np.ndarray:
    """Raw ``(x, y)`` at ``(W, H)`` -> ``letterbox_v1`` planning ``(u, v)``."""

    return letterbox_geometry(source_width, source_height, image_size).raw_to_planning(
        points_xy
    )


def planning_to_raw(
    points_uv: Any,
    source_width: int,
    source_height: int,
    image_size: int = PLANNING_IMAGE_SIZE,
) -> np.ndarray:
    """``letterbox_v1`` planning ``(u, v)`` -> raw ``(x, y)`` at ``(W, H)``."""

    return letterbox_geometry(source_width, source_height, image_size).planning_to_raw(
        points_uv
    )


def squash_to_planning(
    points_uv: Any,
    source_width: int,
    source_height: int,
    image_size: int = PLANNING_IMAGE_SIZE,
) -> np.ndarray:
    """Legacy ``squash_v0`` planning pixels -> ``letterbox_v1`` planning pixels."""

    return letterbox_geometry(
        source_width, source_height, image_size
    ).squash_to_planning(points_uv)


def planning_to_squash(
    points_uv: Any,
    source_width: int,
    source_height: int,
    image_size: int = PLANNING_IMAGE_SIZE,
) -> np.ndarray:
    """``letterbox_v1`` planning pixels -> legacy ``squash_v0`` planning pixels."""

    return letterbox_geometry(
        source_width, source_height, image_size
    ).planning_to_squash(points_uv)


def require_letterbox_v1(protocol: Any) -> str:
    """Hard-reject any geometry protocol that is not ``letterbox_v1``.

    Old ``squash_v0`` profiles must be re-fitted, never reinterpreted under
    the new geometry.
    """

    if protocol != LETTERBOX_V1:
        raise ValueError(
            f"geometry protocol must be {LETTERBOX_V1!r}, got {protocol!r}; "
            f"{SQUASH_V0!r} profiles are rejected and must be re-fitted, not "
            "reinterpreted"
        )
    return LETTERBOX_V1


def letterbox_rgb_image(frame: Any, image_size: int = PLANNING_IMAGE_SIZE) -> np.ndarray:
    """Produce the ``letterbox_v1`` planning image for one raw RGB frame.

    The pixel counterpart of :class:`LetterboxGeometry`: the content size and
    the paste offset both come from the same descriptor the coordinate
    transforms use, so an image and a pixel mapped through this module cannot
    disagree.  Bit-identical to the validated ``v10/live_pilot.py:65-88``
    reference.

    Validation order (``image_size`` before dtype) is deliberate: it matches
    the reference so its pinned error messages stay reachable.
    """

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    values = np.asarray(frame)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
        raise ValueError("letterbox expects uint8 RGB [H, W, 3]")
    height, width = values.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("letterbox expects a non-empty image")

    from PIL import Image

    geometry = letterbox_geometry(int(width), int(height), int(image_size))
    resized = Image.fromarray(values, mode="RGB").resize(
        geometry.content_size, Image.Resampling.BICUBIC
    )
    canvas = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
    canvas.paste(resized, geometry.padding)
    return np.array(canvas, dtype=np.uint8, copy=True)


def _as_pixel_array(values: Any, *, label: str) -> np.ndarray:
    """Normalize an arbitrary-leading-shape ``[..., 2]`` pixel array."""

    points = np.asarray(values, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError(f"{label} must have shape [..., 2], got {points.shape}")
    return points


@dataclass(frozen=True)
class PlanningGeometry:
    """Which planning-canvas pixels actually carry camera data.

    The planning canvas is always ``image_size x image_size``.  Under
    ``squash_v0`` the whole canvas is camera content.  Under ``letterbox_v1``
    only the pasted content rectangle is; the padding bands carry no camera
    data at all, and a projected anchor landing there is **not** observable
    however finite its coordinates are.

    This distinction is the reason the class exists.  The pre-migration live
    path squashed, so ``0 <= v < image_size`` was a correct content test and
    the runtime used it everywhere.  The moment the live path letterboxes,
    that same test silently accepts the padding bands -- 56 of 224 rows at
    4:3, a quarter of the canvas -- and reports pixels as observable that
    contain nothing.  Both protocols therefore have to be expressible by one
    predicate, so that switching the ingestion geometry cannot leave a
    validity check behind.

    Bounds are HALF-OPEN (``low <= p < high``), matching the runtime checks
    this replaces, so ``squash_v0`` reproduces the historical behavior
    exactly rather than approximately.
    """

    protocol: str
    image_size: int = PLANNING_IMAGE_SIZE
    source_width: int | None = None
    source_height: int | None = None

    def __post_init__(self) -> None:
        if self.protocol not in (SQUASH_V0, LETTERBOX_V1):
            raise ValueError(
                f"geometry protocol must be {SQUASH_V0!r} or {LETTERBOX_V1!r}, "
                f"got {self.protocol!r}"
            )
        object.__setattr__(
            self, "image_size", _require_positive_int(self.image_size, name="image_size")
        )
        if self.protocol == LETTERBOX_V1:
            if self.source_width is None or self.source_height is None:
                raise ValueError(
                    f"{LETTERBOX_V1!r} requires source_width and source_height: "
                    "the content rectangle cannot be derived from the planning "
                    "canvas alone, and a 224x224 shape does not identify a "
                    "geometry"
                )
        if self.source_width is not None:
            object.__setattr__(
                self,
                "source_width",
                _require_positive_int(self.source_width, name="source_width"),
            )
        if self.source_height is not None:
            object.__setattr__(
                self,
                "source_height",
                _require_positive_int(self.source_height, name="source_height"),
            )

    # ------------------------------------------------------------- constructors

    @classmethod
    def squash_v0(
        cls,
        image_size: int = PLANNING_IMAGE_SIZE,
        *,
        source_width: int | None = None,
        source_height: int | None = None,
    ) -> "PlanningGeometry":
        """Legacy anisotropic resize: the entire canvas is camera content."""

        return cls(
            protocol=SQUASH_V0,
            image_size=image_size,
            source_width=source_width,
            source_height=source_height,
        )

    @classmethod
    def letterbox_v1(
        cls,
        source_width: int,
        source_height: int,
        image_size: int = PLANNING_IMAGE_SIZE,
    ) -> "PlanningGeometry":
        """Aspect-preserving letterbox: only the pasted rectangle is content."""

        return cls(
            protocol=LETTERBOX_V1,
            image_size=image_size,
            source_width=source_width,
            source_height=source_height,
        )

    # -------------------------------------------------------------- descriptor

    @property
    def letterbox(self) -> LetterboxGeometry | None:
        """The underlying letterbox descriptor, or ``None`` under squash."""

        if self.protocol != LETTERBOX_V1:
            return None
        assert self.source_width is not None and self.source_height is not None
        return letterbox_geometry(
            self.source_width, self.source_height, self.image_size
        )

    @property
    def canvas_size(self) -> tuple[int, int]:
        return (self.image_size, self.image_size)

    @property
    def content_rect(self) -> tuple[int, int, int, int]:
        """``(left, top, right, bottom)`` content box, right/bottom exclusive."""

        box = self.letterbox
        if box is None:
            return (0, 0, self.image_size, self.image_size)
        return (
            box.pad_x,
            box.pad_y,
            box.pad_x + box.content_width,
            box.pad_y + box.content_height,
        )

    @property
    def content_width(self) -> int:
        left, _, right, _ = self.content_rect
        return right - left

    @property
    def content_height(self) -> int:
        _, top, _, bottom = self.content_rect
        return bottom - top

    @property
    def pad_x(self) -> int:
        box = self.letterbox
        return 0 if box is None else box.pad_x

    @property
    def pad_y(self) -> int:
        box = self.letterbox
        return 0 if box is None else box.pad_y

    @property
    def pad_rows(self) -> int:
        """Canvas rows carrying no camera data (0 under ``squash_v0``)."""

        return self.image_size - self.content_height

    @property
    def pad_columns(self) -> int:
        return self.image_size - self.content_width

    # --------------------------------------------------------------- predicates

    def contains_canvas_pixel(
        self, points_uv: Any, *, margin: float = 0.0
    ) -> np.ndarray | np.bool_:
        """Whether pixels lie on the planning CANVAS (geometry-independent)."""

        return self._inside_box(
            points_uv, (0, 0, self.image_size, self.image_size), margin=margin
        )

    def contains_content_pixel(
        self, points_uv: Any, *, margin: float = 0.0
    ) -> np.ndarray | np.bool_:
        """Whether pixels lie on actual CAMERA CONTENT.

        Under ``squash_v0`` this is identical to
        :meth:`contains_canvas_pixel`.  Under ``letterbox_v1`` the padding
        bands are excluded.  Non-finite coordinates are always rejected.
        """

        return self._inside_box(points_uv, self.content_rect, margin=margin)

    def content_mask(self, points_uv: Any, *, margin: float = 0.0) -> np.ndarray:
        """Content predicate over ``[..., 2]``, returning the leading shape."""

        result = self._inside_box(points_uv, self.content_rect, margin=margin)
        return np.asarray(result, dtype=bool)

    def _inside_box(
        self,
        points_uv: Any,
        box: tuple[int, int, int, int],
        *,
        margin: float,
    ) -> np.ndarray | np.bool_:
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError(f"margin must be finite and non-negative, got {margin!r}")
        points = _as_pixel_array(points_uv, label="points_uv")
        left, top, right, bottom = (float(v) for v in box)
        u = points[..., 0]
        v = points[..., 1]
        inside = (
            np.isfinite(u)
            & np.isfinite(v)
            & (u >= left + margin)
            & (u < right - margin)
            & (v >= top + margin)
            & (v < bottom - margin)
        )
        if points.ndim == 1:
            return np.bool_(inside)
        return inside

    # -------------------------------------------------------------- transforms

    def raw_to_planning(self, points_xy: Any) -> np.ndarray:
        """Raw camera ``(x, y)`` -> planning ``(u, v)`` under this protocol."""

        box = self.letterbox
        if box is not None:
            return box.raw_to_planning(points_xy)
        width, height = self._require_source("raw_to_planning")
        points = _as_pixel_array(points_xy, label="points_xy")
        scale = np.array(
            [self.image_size / width, self.image_size / height], dtype=np.float64
        )
        return points * scale

    def planning_to_raw(self, points_uv: Any) -> np.ndarray:
        """Planning ``(u, v)`` -> raw camera ``(x, y)`` under this protocol."""

        box = self.letterbox
        if box is not None:
            return box.planning_to_raw(points_uv)
        width, height = self._require_source("planning_to_raw")
        points = _as_pixel_array(points_uv, label="points_uv")
        scale = np.array(
            [width / self.image_size, height / self.image_size], dtype=np.float64
        )
        return points * scale

    def _require_source(self, operation: str) -> tuple[int, int]:
        if self.source_width is None or self.source_height is None:
            raise ValueError(
                f"{operation} under {SQUASH_V0!r} needs source_width and "
                "source_height; the anisotropic scale cannot be recovered "
                "from the planning canvas alone"
            )
        return self.source_width, self.source_height

    # ------------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        """Explicit geometry metadata for a persisted artifact or contract."""

        left, top, right, bottom = self.content_rect
        payload: dict[str, Any] = {
            "geometry_protocol": self.protocol,
            "planning_image_size": self.image_size,
            "content_rect": [left, top, right, bottom],
            "padding": [self.pad_x, self.pad_y],
        }
        if self.source_width is not None:
            payload["source_width"] = self.source_width
        if self.source_height is not None:
            payload["source_height"] = self.source_height
        box = self.letterbox
        if box is not None:
            payload["scale"] = box.scale
        return payload


def planning_geometry_from_protocol(
    protocol: Any,
    *,
    source_width: int | None = None,
    source_height: int | None = None,
    image_size: int = PLANNING_IMAGE_SIZE,
) -> PlanningGeometry:
    """Build a :class:`PlanningGeometry`, failing closed on an unknown protocol.

    A missing or unrecognized protocol raises rather than defaulting.  Geometry
    is never inferred: a ``224x224`` array is consistent with both protocols,
    so silently assuming one would be the exact reinterpretation this module
    exists to prevent.
    """

    if protocol == SQUASH_V0:
        return PlanningGeometry.squash_v0(
            image_size=image_size,
            source_width=source_width,
            source_height=source_height,
        )
    if protocol == LETTERBOX_V1:
        if source_width is None or source_height is None:
            raise ValueError(
                f"{LETTERBOX_V1!r} requires source_width and source_height"
            )
        return PlanningGeometry.letterbox_v1(
            source_width, source_height, image_size=image_size
        )
    raise ValueError(
        f"unknown geometry protocol {protocol!r}; expected {SQUASH_V0!r} or "
        f"{LETTERBOX_V1!r}. Geometry is never inferred from array shape: a "
        "224x224 frame is consistent with both, so the caller must declare it."
    )
