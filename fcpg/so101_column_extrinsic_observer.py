"""FCPG column-camera extrinsic observer.

Stage-I ``O_FCPG`` predicts a local 2-D residual around the nominal FK
projection and is gated at ``fcpg_max_correction_px`` (18).  A Jieli unplug
or re-aim is a ~46 px image translation: outside that envelope, and also
outside the pose-local support ball of a one-shot jaw-pulse live-anchor.

This module is the FCPG-side correction for that camera residual:

1. **Measurement, now (crude):** metal jaw-tip appearance versus FK.  A
   14 px-pitch locator on wood is never a measurement.
2. **Estimator, now:** a 2-D complementary / information filter on image
   translation.  The camera is mostly static, so accepted updates shrink
   the variance — the control-theoretic “越估计越准”.
3. **Later:** replace the metal measurement with a learned residual head
   (or a pulse) that returns ``(delta_px, R)``.  Do not use LeWM for this
   re-centering; it already lost to hold / CV / Kalman on the recursive
   re-centering Gate.

When metal L/R pass the gates, that frame's translation is applied immediately
(hard match) so ranking / servo / overlays sit on the metal pixels.  The
filter is the hold / seed path for frames where metal is not visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any

import numpy as np

from fcpg.so101_active_gripper import apply_live_similarity_transform
from fcpg.so101_column_tip_measurement import (
    column_metal_operator_tips,
    is_patch_pitch_triple,
)


DEFAULT_MAX_OFFSET_PX = 60.0
DEFAULT_MAX_RESIDUAL_PX = 8.0
DEFAULT_PROCESS_VARIANCE_PX2 = 0.25
DEFAULT_MEASUREMENT_VARIANCE_PX2 = 16.0
DEFAULT_JUMP_INNOVATION_PX = 25.0
_MIN_TIP_SEPARATION_PX = 8.0


def _ray_exit_box(
    origin: np.ndarray,
    direction: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> np.ndarray:
    axis = np.asarray(direction, dtype=np.float32).reshape(2)
    length = float(np.linalg.norm(axis))
    if length < 1.0e-6:
        return np.asarray(origin, dtype=np.float32).reshape(2)
    axis = axis / length
    hits: list[float] = []
    if axis[0] > 1.0e-6:
        hits.append((x1 - float(origin[0])) / float(axis[0]))
    elif axis[0] < -1.0e-6:
        hits.append((x0 - float(origin[0])) / float(axis[0]))
    if axis[1] > 1.0e-6:
        hits.append((y1 - float(origin[1])) / float(axis[1]))
    elif axis[1] < -1.0e-6:
        hits.append((y0 - float(origin[1])) / float(axis[1]))
    positive = [step for step in hits if step > 0.0]
    if not positive:
        return np.asarray(origin, dtype=np.float32).reshape(2)
    return (np.asarray(origin, dtype=np.float32).reshape(2) + min(positive) * axis)


def fit_fk_triple_in_box(
    fk_clr_px: np.ndarray,
    bbox_xyxy: Sequence[float],
    *,
    content_rect: Sequence[int] | None = None,
    distal_toward_px: np.ndarray | None = None,
) -> np.ndarray | None:
    """Sit C/L/R on the distal end of the open-vocab robot box.

    The box is the visible arm, not the jaw tips. Putting FK at the box
    centre parks the marks on the forearm. Camera-to-robot here uses the
    box as a support and the jaw opening from FK: C goes to the box edge
    toward the current target (or away from the frame entry), L/R keep
    the FK opening direction at a jaw-scale span.
    """

    fk = np.asarray(fk_clr_px, dtype=np.float32).reshape(3, 2)
    if not np.all(np.isfinite(fk)):
        return None
    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    if x1 <= x0 or y1 <= y0:
        return None
    box_center = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
    direction = None
    if distal_toward_px is not None:
        toward = np.asarray(distal_toward_px, dtype=np.float32).reshape(2)
        if np.all(np.isfinite(toward)):
            direction = toward - box_center
    if direction is None or float(np.linalg.norm(direction)) < 1.0e-3:
        if content_rect is not None:
            left, top, right, bottom = (float(value) for value in content_rect)
            sides = (
                (x0 - left, np.asarray([1.0, 0.0], dtype=np.float32)),
                (right - x1, np.asarray([-1.0, 0.0], dtype=np.float32)),
                (y0 - top, np.asarray([0.0, 1.0], dtype=np.float32)),
                (bottom - y1, np.asarray([0.0, -1.0], dtype=np.float32)),
            )
            direction = min(sides, key=lambda item: item[0])[1]
        else:
            direction = np.asarray([0.0, 1.0], dtype=np.float32)
    exit_pt = _ray_exit_box(box_center, direction, x0, y0, x1, y1)
    new_c = exit_pt - 0.20 * (exit_pt - box_center)
    opening = fk[1] - fk[2]
    fk_span = float(np.linalg.norm(opening))
    if fk_span < 1.0e-3:
        opening = np.asarray([1.0, 0.0], dtype=np.float32)
        fk_span = 1.0
    hat = opening / fk_span
    target_span = float(np.clip(fk_span, 12.0, min(36.0, 0.40 * (x1 - x0))))
    half = (0.5 * target_span) * hat
    placed = np.stack([new_c, new_c + half, new_c - half]).astype(np.float32)
    if content_rect is not None:
        left, top, right, bottom = (float(value) for value in content_rect)
        for _ in range(8):
            if all(
                left <= float(point[0]) < right and top <= float(point[1]) < bottom
                for point in placed
            ):
                break
            placed = new_c + (placed - new_c) * 0.75
            placed[0] = new_c
    return placed


def translation_transform_px(translation_px: np.ndarray) -> np.ndarray:
    offset = np.asarray(translation_px, dtype=np.float32).reshape(2)
    return np.asarray(
        [[1.0, 0.0, float(offset[0])], [0.0, 1.0, float(offset[1])]],
        dtype=np.float32,
    )


def transform_translation_px(transform_px: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform_px, dtype=np.float32)
    if transform.shape != (2, 3):
        raise ValueError("transform_px must be a 2x3 matrix")
    return transform[:, 2].astype(np.float32, copy=True)


def apply_column_frame_offset_px(
    paths: np.ndarray,
    offset_px: np.ndarray | None,
) -> np.ndarray:
    """Translate projected column paths into the applied metal / hard-match frame.

    A camera-global translation is shape ``(2,)``.  Visible metal L/R is often
    tilted relative to pulse/FK (live Jieli: L ≈14 px below R), so scoring and
    overlays use a per-anchor ``(3, 2)`` residual.  C-only arrays take the C
    row of that residual.
    """

    values = np.asarray(paths, dtype=np.float32)
    if offset_px is None:
        return values
    offset = np.asarray(offset_px, dtype=np.float32)
    if offset.shape == (3, 2):
        if not np.all(np.isfinite(offset)):
            return values
        if values.ndim >= 2 and values.shape[-2:] == (3, 2):
            return values + offset
        if values.shape[-1] == 2:
            return values + offset[0]
        return values
    offset = offset.reshape(2)
    if not np.all(np.isfinite(offset)):
        return values
    return values + offset.reshape((1,) * (values.ndim - 1) + (2,))


def column_hard_match_residual_px(
    projected_tips_px: np.ndarray,
    metal_points_px: np.ndarray | None,
) -> np.ndarray | None:
    """Per-anchor offset from a uniform-transform projection onto visible metal.

    ``refine_column_live_nominal`` may overwrite the current tip triple with
    metal while ``project_future_anchor_pixels`` still uses a 2-D translation.
    The first IK waypoint then sits on the centroid-aligned (flat) triple and
    the snapshot's first segment jumps ~10 px off the metal R/L marks.
    """

    if metal_points_px is None:
        return None
    projected = np.asarray(projected_tips_px, dtype=np.float32).reshape(3, 2)
    metal = np.asarray(metal_points_px, dtype=np.float32).reshape(3, 2)
    if not np.all(np.isfinite(projected)) or not np.all(np.isfinite(metal)):
        return None
    if is_patch_pitch_triple(metal):
        return None
    residual = (metal - projected).astype(np.float32)
    if not np.all(np.isfinite(residual)):
        return None
    return residual


@dataclass(frozen=True)
class ColumnTipTranslationMeasurement:
    """One accepted metal-vs-FK translation on the 224 planning canvas."""

    translation_px: np.ndarray
    residual_rms_px: float
    metal_points_px: np.ndarray
    fk_points_px: np.ndarray
    metal_separation_px: float
    fk_separation_px: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": "metal",
            "translation_px": self.translation_px.astype(np.float32),
            "residual_rms_px": float(self.residual_rms_px),
            "metal_points_px": self.metal_points_px.astype(np.float32),
            "fk_points_px": self.fk_points_px.astype(np.float32),
            "metal_separation_px": float(self.metal_separation_px),
            "fk_separation_px": float(self.fk_separation_px),
        }


def measure_column_fk_translation(
    frame: np.ndarray,
    nominal_clr_px: np.ndarray,
    *,
    max_offset_px: float = DEFAULT_MAX_OFFSET_PX,
    max_residual_px: float = DEFAULT_MAX_RESIDUAL_PX,
    min_separation_ratio: float = 0.5,
    max_separation_ratio: float = 1.5,
) -> ColumnTipTranslationMeasurement | None:
    """Return the image translation that takes FK L/R onto visible metal L/R.

    L/R correspondence is resolved by image-x order so a swapped planner
    convention cannot inflate the residual and reject a real jaw pair.
    """

    nominal = np.asarray(nominal_clr_px, dtype=np.float32).reshape(3, 2)
    if not np.all(np.isfinite(nominal[1:3])):
        return None
    metal = column_metal_operator_tips(frame)
    if metal is None or is_patch_pitch_triple(metal):
        return None
    fk_lr = nominal[1:3]
    metal_lr = metal[1:3]
    fk_sorted = fk_lr[np.argsort(fk_lr[:, 0])]
    metal_sorted = metal_lr[np.argsort(metal_lr[:, 0])]
    fk_separation = float(np.linalg.norm(fk_sorted[1] - fk_sorted[0]))
    metal_separation = float(np.linalg.norm(metal_sorted[1] - metal_sorted[0]))
    if fk_separation < _MIN_TIP_SEPARATION_PX:
        return None
    ratio = metal_separation / fk_separation
    if not min_separation_ratio <= ratio <= max_separation_ratio:
        return None
    translation = metal_sorted.mean(axis=0) - fk_sorted.mean(axis=0)
    if float(np.linalg.norm(translation)) > float(max_offset_px):
        return None
    residual = float(
        np.sqrt(np.mean(np.sum((fk_sorted + translation - metal_sorted) ** 2, axis=-1)))
    )
    if residual > float(max_residual_px):
        return None
    return ColumnTipTranslationMeasurement(
        translation_px=translation.astype(np.float32),
        residual_rms_px=residual,
        metal_points_px=metal.astype(np.float32),
        fk_points_px=nominal.copy(),
        metal_separation_px=metal_separation,
        fk_separation_px=fk_separation,
    )


@dataclass
class ColumnExtrinsicObserver:
    """Recursive 2-D translation estimate for a slowly moving column camera."""

    process_variance_px2: float = DEFAULT_PROCESS_VARIANCE_PX2
    measurement_variance_px2: float = DEFAULT_MEASUREMENT_VARIANCE_PX2
    jump_innovation_px: float = DEFAULT_JUMP_INNOVATION_PX
    max_offset_px: float = DEFAULT_MAX_OFFSET_PX
    translation_px: np.ndarray | None = None
    variance_px2: float | None = None
    n_updates: int = 0
    last_source: str | None = None
    last_residual_rms_px: float | None = None
    last_update: dict[str, Any] = field(default_factory=dict)

    def predict(self) -> None:
        if self.variance_px2 is None:
            return
        self.variance_px2 = float(self.variance_px2) + float(self.process_variance_px2)

    def update(
        self,
        measurement_px: np.ndarray,
        *,
        residual_rms_px: float,
        source: str,
    ) -> dict[str, Any]:
        measured = np.asarray(measurement_px, dtype=np.float32).reshape(2)
        if not np.all(np.isfinite(measured)):
            raise ValueError("column extrinsic measurement must be finite")
        if float(np.linalg.norm(measured)) > float(self.max_offset_px):
            record = {
                "accepted": False,
                "reason": "offset_envelope",
                "source": source,
                "innovation_px": None,
                "gain": 0.0,
            }
            self.last_update = record
            return record
        measurement_r = float(self.measurement_variance_px2) * max(
            1.0, (float(residual_rms_px) / 4.0) ** 2
        )
        if self.translation_px is None or self.variance_px2 is None:
            self.translation_px = measured.copy()
            self.variance_px2 = measurement_r
            self.n_updates = 1
            self.last_source = source
            self.last_residual_rms_px = float(residual_rms_px)
            record = {
                "accepted": True,
                "reason": "init",
                "source": source,
                "innovation_px": np.zeros(2, dtype=np.float32),
                "gain": 1.0,
            }
            self.last_update = record
            return record

        innovation = measured - self.translation_px
        innovation_norm = float(np.linalg.norm(innovation))
        jumped = (
            innovation_norm > float(self.jump_innovation_px)
            and float(residual_rms_px) <= 6.0
        )
        if jumped:
            self.translation_px = measured.copy()
            self.variance_px2 = measurement_r
            self.n_updates = 1
            self.last_source = source
            self.last_residual_rms_px = float(residual_rms_px)
            record = {
                "accepted": True,
                "reason": "camera_jump_reset",
                "source": source,
                "innovation_px": innovation.astype(np.float32),
                "gain": 1.0,
            }
            self.last_update = record
            return record

        prior_variance = float(self.variance_px2)
        gain = prior_variance / (prior_variance + measurement_r)
        self.translation_px = self.translation_px + gain * innovation
        self.variance_px2 = (1.0 - gain) * prior_variance
        self.n_updates += 1
        self.last_source = source
        self.last_residual_rms_px = float(residual_rms_px)
        record = {
            "accepted": True,
            "reason": "kalman",
            "source": source,
            "innovation_px": innovation.astype(np.float32),
            "gain": float(gain),
        }
        self.last_update = record
        return record

    def seed_from_pulse(
        self, translation_px: np.ndarray, residual_rms_px: float
    ) -> dict[str, Any]:
        return self.update(
            translation_px,
            residual_rms_px=residual_rms_px,
            source="pulse",
        )

    def transform_px(self) -> np.ndarray | None:
        if self.translation_px is None:
            return None
        return translation_transform_px(self.translation_px)

    def snapshot(self) -> dict[str, Any]:
        return {
            "present": self.translation_px is not None,
            "translation_px": (
                None
                if self.translation_px is None
                else self.translation_px.astype(np.float32)
            ),
            "variance_px2": self.variance_px2,
            "n_updates": int(self.n_updates),
            "last_source": self.last_source,
            "last_residual_rms_px": self.last_residual_rms_px,
            "last_update": dict(self.last_update),
        }


@dataclass(frozen=True)
class ColumnLiveRefinement:
    nominal_px: np.ndarray
    transform_px: np.ndarray | None
    applied_source: str | None
    block_out_of_support: bool
    diagnostics: dict[str, Any]
    hard_match_residual_px: np.ndarray | None = None


def _metal_aligned_nominal(
    raw_nominal_px: np.ndarray,
    transform_px: np.ndarray,
    metal_points_px: np.ndarray,
) -> np.ndarray:
    aligned = np.asarray(
        apply_live_similarity_transform(raw_nominal_px, transform_px),
        dtype=np.float32,
    ).copy()
    aligned[-1] = np.asarray(metal_points_px, dtype=np.float32).reshape(3, 2)
    return aligned


def refine_column_live_nominal(
    raw_nominal_px: np.ndarray,
    frame: np.ndarray,
    observer: ColumnExtrinsicObserver,
    *,
    pulse_applied: bool,
    pulse_transform_px: np.ndarray | None,
    pulse_residual_rms_px: float | None = None,
    max_offset_px: float = DEFAULT_MAX_OFFSET_PX,
    max_residual_px: float = DEFAULT_MAX_RESIDUAL_PX,
) -> ColumnLiveRefinement:
    """Hard-match visible metal tips; otherwise hold pulse / observer.

    When metal L/R pass the existing gates, this frame's metal C/L/R is the
    column tip triple — not a Kalman average and not gated on the pulse
    support ball.  The observer still holds the last accepted translation
    when metal disappears.  A pulse still seeds an empty observer so leaving
    the support ball does not drop to raw FK.  OOS is blocked only when a
    pulse exists, the ball does not apply, and there is still no observer
    estimate.
    """

    raw = np.asarray(raw_nominal_px, dtype=np.float32)
    observer.predict()
    pulse_residual = (
        2.0 if pulse_residual_rms_px is None else float(pulse_residual_rms_px)
    )
    # The pulse translation is camera-global.  Seed the observer from it
    # whenever the ball applies, or whenever the observer is still empty —
    # leaving the support ball must not drop the ~46 px Jieli shift.
    if pulse_transform_px is not None and (
        pulse_applied or observer.translation_px is None
    ):
        observer.seed_from_pulse(
            transform_translation_px(pulse_transform_px), pulse_residual
        )

    if pulse_applied and pulse_transform_px is not None:
        working_transform = np.asarray(pulse_transform_px, dtype=np.float32)
        working = apply_live_similarity_transform(raw, working_transform)
    else:
        working_transform = observer.transform_px()
        working = (
            raw
            if working_transform is None
            else apply_live_similarity_transform(raw, working_transform)
        )

    measure_kwargs = {
        "max_offset_px": max_offset_px,
        "max_residual_px": max_residual_px,
    }
    measurement = measure_column_fk_translation(frame, raw[-1], **measure_kwargs)
    used_working_prior = False
    if measurement is None and working_transform is not None:
        # Raw FK can sit ~90 px from metal after a Jieli shift plus pulse
        # residual.  The 50 px envelope is on the remaining offset from the
        # current belief, not on the stacked total from factory calibration.
        measurement = measure_column_fk_translation(
            frame, working[-1], **measure_kwargs
        )
        used_working_prior = measurement is not None
    measurement_record = None if measurement is None else measurement.as_dict()

    if measurement is not None:
        if used_working_prior:
            total = (
                transform_translation_px(working_transform)
                + measurement.translation_px
            )
            if float(np.linalg.norm(total)) <= float(max_offset_px):
                observer.update(
                    total,
                    residual_rms_px=measurement.residual_rms_px,
                    source="metal",
                )
            transform = translation_transform_px(total)
        else:
            observer.update(
                measurement.translation_px,
                residual_rms_px=measurement.residual_rms_px,
                source="metal",
            )
            transform = translation_transform_px(measurement.translation_px)
        diagnostics = {
            "applied_source": "metal",
            "hard_match_this_frame": True,
            "measurement": measurement_record,
            "observer": observer.snapshot(),
            "residual_rms_px": measurement.residual_rms_px,
        }
        projected_tips = apply_live_similarity_transform(raw[-1], transform)
        return ColumnLiveRefinement(
            nominal_px=_metal_aligned_nominal(
                raw, transform, measurement.metal_points_px
            ),
            transform_px=transform,
            applied_source="metal",
            block_out_of_support=False,
            diagnostics=diagnostics,
            hard_match_residual_px=column_hard_match_residual_px(
                projected_tips, measurement.metal_points_px
            ),
        )

    if pulse_applied and pulse_transform_px is not None:
        diagnostics = {
            "applied_source": "pulse",
            "hard_match_this_frame": False,
            "measurement": measurement_record,
            "observer": observer.snapshot(),
            "residual_rms_px": pulse_residual,
        }
        return ColumnLiveRefinement(
            nominal_px=apply_live_similarity_transform(raw, pulse_transform_px),
            transform_px=np.asarray(pulse_transform_px, dtype=np.float32),
            applied_source="pulse",
            block_out_of_support=False,
            diagnostics=diagnostics,
        )

    transform = observer.transform_px()
    if transform is not None:
        diagnostics = {
            "applied_source": "column_extrinsic_observer",
            "hard_match_this_frame": False,
            "measurement": measurement_record,
            "observer": observer.snapshot(),
            "residual_rms_px": observer.last_residual_rms_px,
        }
        return ColumnLiveRefinement(
            nominal_px=apply_live_similarity_transform(raw, transform),
            transform_px=transform,
            applied_source="column_extrinsic_observer",
            block_out_of_support=False,
            diagnostics=diagnostics,
        )

    diagnostics = {
        "applied_source": None,
        "hard_match_this_frame": False,
        "measurement": measurement_record,
        "observer": observer.snapshot(),
        "residual_rms_px": None,
    }
    return ColumnLiveRefinement(
        nominal_px=raw,
        transform_px=None,
        applied_source=None,
        block_out_of_support=pulse_transform_px is not None,
        diagnostics=diagnostics,
    )


__all__ = [
    "ColumnExtrinsicObserver",
    "ColumnLiveRefinement",
    "ColumnTipTranslationMeasurement",
    "apply_column_frame_offset_px",
    "column_hard_match_residual_px",
    "fit_fk_triple_in_box",
    "measure_column_fk_translation",
    "refine_column_live_nominal",
    "transform_translation_px",
    "translation_transform_px",
]
