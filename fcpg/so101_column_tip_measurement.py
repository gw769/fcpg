"""Column-view metal jaw tips on the 224 planning canvas.

This is the crude, appearance-based measurement FCPG uses until a learned
residual head can replace it.  A 14 px-pitch locator on wood is never a
tip measurement.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from fcpg.letterbox_geometry import PLANNING_IMAGE_SIZE


LOCATOR_PATCH_PITCH_PX = 14.0
_LETTERBOX_PAD_LUMA = 8.0
_CONTENT_ROW_LUMA = 20.0


def luma_image(frame: np.ndarray) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float32)
    return 0.299 * values[..., 0] + 0.587 * values[..., 1] + 0.114 * values[..., 2]


def column_content_rows(frame: np.ndarray) -> tuple[int, int]:
    luma = luma_image(frame)
    rows = np.where(luma.mean(axis=1) > _CONTENT_ROW_LUMA)[0]
    if rows.size == 0:
        return 0, int(frame.shape[0])
    return int(rows[0]), int(rows[-1]) + 1


def has_letterbox_pad(frame: np.ndarray) -> bool:
    luma = luma_image(frame)
    if luma.shape[0] < 40:
        return False
    return float(luma[:16].mean()) < _LETTERBOX_PAD_LUMA and float(
        luma[28:40].mean()
    ) > _CONTENT_ROW_LUMA


def is_patch_pitch_triple(points: np.ndarray) -> bool:
    """True when C/L/R sit on the frozen-ViT 14 px locator lattice."""

    pts = np.asarray(points, dtype=np.float32).reshape(3, 2)
    if float(pts[:, 1].max() - pts[:, 1].min()) > 3.0:
        return False
    separation = float(np.linalg.norm(pts[1] - pts[2]))
    if not any(
        abs(separation - step * LOCATOR_PATCH_PITCH_PX) <= 3.0 for step in (1, 2, 3)
    ):
        return False
    half = 0.5 * LOCATOR_PATCH_PITCH_PX
    residual = np.abs(pts / half - np.round(pts / half)) * half
    return float(np.max(residual)) <= 1.5


def column_pixel_on_metal(
    frame: np.ndarray, x: int, y: int, y0: int, y1: int
) -> bool:
    if not (y0 <= y < y1 and 0 <= x < frame.shape[1]):
        return False
    pixel = frame[y, x].astype(np.float32)
    red, green, blue = (float(channel) for channel in pixel)
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    sat = max(red, green, blue) - min(red, green, blue)
    # Saturated warm objects (pickup) are never jaw metal.
    if red > 140.0 and red > green + 25.0 and red > blue + 15.0:
        return False
    # Warm wood: red-leaning, low sat, bright.
    if red >= blue + 3.0 and luma > 135.0 and sat < 32.0:
        return False
    # Cool wood / box highlights sit at luma>175 with weak blue excess.
    if luma < 35.0 or luma > 175.0:
        return False
    return (blue - red) >= 12.0 and 18.0 <= sat <= 90.0


def column_l_r_on_metal(frame: np.ndarray, points: np.ndarray) -> bool:
    pts = np.asarray(points, dtype=np.float32).reshape(3, 2)
    y0, y1 = column_content_rows(frame)
    for sample in pts[1], pts[2]:
        x, y = int(round(float(sample[0]))), int(round(float(sample[1])))
        if not column_pixel_on_metal(frame, x, y, y0, y1):
            return False
    return True


_WRIST_BAND_PX = 50
_HANG_GAP_PX = 2
_MIN_WRIST_PIXELS = 3
_MIN_HANG_LEN_PX = 8
_MAX_FINGER_WIDTH_PX = 32
_WRIST_WINDOW_PX = 72
_CLOSE_RADIUS_PX = 8
_LOCAL_MEDIAN_RADIUS_PX = 24


def _close_vertical(mask: np.ndarray, radius: int) -> np.ndarray:
    """Fill short vertical holes so overexposed wrist plastic does not break a finger."""

    dilated = mask.copy()
    for shift in range(1, radius + 1):
        dilated[shift:] |= mask[:-shift]
        dilated[:-shift] |= mask[shift:]
    closed = dilated.copy()
    for shift in range(1, radius + 1):
        closed[shift:] &= dilated[:-shift]
        closed[:-shift] &= dilated[shift:]
    return closed


def _local_row_median(pixels: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(pixels, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    windows = sliding_window_view(padded, 2 * radius + 1, axis=1)
    return np.median(windows, axis=-1)


def gripper_like_mask(image: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """Pixels that can belong to a hanging SO-101 gripper.

    Includes gray blades, white/cool metal, and the dark wrist camera.
    Excludes letterbox, saturated red pickup, warm beige wood, and brown
    cardboard.  A local-median outlier term catches mid-gray blades that
    fail the cool-metal test without promoting the table.
    """

    pixels = image.astype(np.float32)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    sat = pixels.max(axis=-1) - pixels.min(axis=-1)
    blue_minus_red = blue - red
    red_minus_green = red - green
    red_minus_blue = red - blue

    letterbox = luma < 16.0
    fish = (
        (red > 135.0)
        & (red_minus_green > 22.0)
        & (red_minus_blue > 12.0)
        & (sat > 28.0)
    ) | ((red_minus_green > 35.0) & (sat > 45.0) & (red_minus_blue > 8.0))
    warm_wood = (
        (red_minus_blue >= 4.0)
        & (red_minus_green >= -1.0)
        & (luma > 148.0)
        & (sat < 30.0)
        & (red_minus_blue < 28.0)
    )
    brown = (
        (luma > 30.0)
        & (luma < 125.0)
        & (red_minus_green > 6.0)
        & (red_minus_blue > -4.0)
        & (sat > 5.0)
        & (sat < 38.0)
    )
    cool_metal = (
        (blue_minus_red >= 10.0)
        & (sat >= 14.0)
        & (sat <= 95.0)
        & (luma >= 30.0)
        & (luma <= 195.0)
    )
    dark = (
        (luma >= 16.0)
        & (luma <= 82.0)
        & (sat <= 48.0)
        & (red_minus_green < 18.0)
        & (red_minus_blue < 20.0)
        & ~fish
    )
    gray = (
        (sat <= 20.0)
        & (luma >= 50.0)
        & (luma <= 168.0)
        & (np.abs(red_minus_green) <= 12.0)
        & (np.abs(blue_minus_red) <= 18.0)
    )
    median = _local_row_median(pixels, _LOCAL_MEDIAN_RADIUS_PX)
    deviation = np.linalg.norm(pixels - median, axis=-1)
    median_luma = (
        0.299 * median[..., 0] + 0.587 * median[..., 1] + 0.114 * median[..., 2]
    )
    outlier = (deviation >= 16.0) & (luma < median_luma + 8.0)
    grip = cool_metal | dark | gray | (
        outlier & ~warm_wood & (sat < 50.0) & (luma < 200.0)
    )
    grip &= ~letterbox & ~fish & ~warm_wood & ~brown
    grip[:y0] = False
    grip[y1:] = False
    closed = _close_vertical(grip, _CLOSE_RADIUS_PX)
    closed[:y0] = False
    closed[y1:] = False
    # Closing may paint across a thin fish/wood gap; put those materials back.
    return closed & ~letterbox & ~fish & ~warm_wood


def _hanging_distal(
    grip: np.ndarray, y0: int, y1: int
) -> tuple[dict[int, int], dict[int, int]]:
    """Flood down from the wrist component, including diagonal fingers.

    A vertical per-column walk stops at the camera when the blades leave
    that column.  The hanging structure is the connected component that
    meets the top band, plus gripper pixels reachable downward / diagonally
    from it, so a tilted jaw still contributes its distal y.
    """

    height, width = grip.shape
    wrist_band = grip[y0 : min(y1, y0 + _WRIST_BAND_PX)]
    if not np.any(wrist_band):
        return {}, {}
    # Seed: gripper pixels in the wrist band, ignoring the right-edge box.
    seed = np.zeros_like(grip, dtype=bool)
    seed[y0 : min(y1, y0 + _WRIST_BAND_PX), :173] = wrist_band[:, :173]
    if int(seed.sum()) < _MIN_WRIST_PIXELS:
        return {}, {}
    # Downward / diagonal flood through the gripper mask.
    hanging = seed.copy()
    changed = True
    while changed:
        changed = False
        # Shift hanging down, down-left, down-right and keep gripper pixels.
        down = np.zeros_like(hanging)
        down[1:] = hanging[:-1]
        down_left = np.zeros_like(hanging)
        down_left[1:, :-1] = hanging[:-1, 1:]
        down_right = np.zeros_like(hanging)
        down_right[1:, 1:] = hanging[:-1, :-1]
        grown = hanging | ((down | down_left | down_right) & grip)
        grown[:, 173:] = hanging[:, 173:]
        if np.any(grown != hanging):
            hanging = grown
            changed = True
    hanging[:y0] = False
    hanging[y1:] = False
    distal: dict[int, int] = {}
    attach: dict[int, int] = {}
    for x in range(min(width, 173)):
        rows = np.flatnonzero(hanging[y0:y1, x])
        if rows.size == 0:
            continue
        first = int(rows[0])
        last = int(rows[-1])
        if last - first + 1 < _MIN_HANG_LEN_PX:
            continue
        distal[x] = y0 + last
        attach[x] = y0 + first
    return distal, attach


def _peak_support_width(distal: dict[int, int], peak_x: int) -> int:
    """Width of a near-level run.  A solid box is one wide plateau."""

    height = distal[peak_x]
    left = right = peak_x
    while left - 1 in distal and abs(distal[left - 1] - height) <= 6:
        left -= 1
        if peak_x - left > 48:
            break
    while right + 1 in distal and abs(distal[right + 1] - height) <= 6:
        right += 1
        if right - peak_x > 48:
            break
    return right - left + 1


def _wrist_center_x(
    grip: np.ndarray,
    distal: dict[int, int],
    attach: dict[int, int],
    y0: int,
) -> float:
    """Compact high-attaching gripper mass; the box starts lower and loses."""

    width = grip.shape[1]
    mass = grip[y0 : y0 + _WRIST_BAND_PX].sum(axis=0).astype(np.float32)
    weight = np.zeros(width, dtype=np.float32)
    for x, attach_y in attach.items():
        if attach_y > y0 + 8:
            continue
        density = float(mass[x]) / float(_WRIST_BAND_PX)
        if density < 0.16:
            continue
        weight[x] = mass[x] * (_WRIST_BAND_PX - (attach_y - y0) + 4.0)
    if float(weight.sum()) < 20.0:
        for x in distal:
            weight[x] = mass[x] + 1.0
    best_score = -1.0
    best_center = 112.0
    window = _WRIST_WINDOW_PX
    offsets = np.arange(window, dtype=np.float32)
    for start in range(0, width - window + 1):
        segment = weight[start : start + window]
        total = float(segment.sum())
        if total > best_score:
            best_score = total
            best_center = start + float((offsets * segment).sum() / max(total, 1.0e-6))
    return best_center


def column_metal_operator_tips(frame: np.ndarray | None) -> np.ndarray | None:
    """Distal inner tips of the two column-view jaws, or None.

    Hang two fingers from the wrist: build a gripper-like mask, walk each
    column down from the top content band through gripper pixels, then pick
    two narrow distal peaks with a gap.  A solid right-edge box is one wide
    plateau and is rejected.  The pair that attaches higher (the true wrist)
    beats a lower isolated blob.
    """

    if frame is None:
        return None
    image = np.asarray(frame)
    if image.shape != (PLANNING_IMAGE_SIZE, PLANNING_IMAGE_SIZE, 3):
        return None
    y0, y1 = column_content_rows(image)
    grip = gripper_like_mask(image, y0, y1)
    distal, attach = _hanging_distal(grip, y0, y1)
    if len(distal) < 8:
        return None

    peak_xs: list[int] = []
    for x, height in distal.items():
        neighbors = [
            distal[other]
            for other in range(x - 4, x + 5)
            if other != x and other in distal
        ]
        if neighbors and height >= max(neighbors) and height >= y0 + 8:
            peak_xs.append(x)
    peak_xs.sort()
    merged: list[int] = []
    for x in peak_xs:
        if merged and x - merged[-1] <= 8:
            if distal[x] > distal[merged[-1]]:
                merged[-1] = x
        else:
            merged.append(x)

    fingers: list[int] = []
    for x in merged:
        if x < 22 or x > 172:
            continue
        if _peak_support_width(distal, x) > _MAX_FINGER_WIDTH_PX:
            continue
        if distal[x] - attach[x] < 10:
            continue
        fingers.append(x)
    if len(fingers) < 2:
        return None

    wrist_cx = _wrist_center_x(grip, distal, attach, y0)

    def _inner_x(peak_x: int, *, inner: str) -> int:
        peak_y = distal[peak_x]
        column = peak_x
        step = 1 if inner == "right" else -1
        while (
            column + step in distal
            and peak_y - 2 <= distal[column + step] <= peak_y + 4
            and abs((column + step) - peak_x) <= 22
        ):
            column += step
        return column

    best: tuple[int, int, int, int] | None = None
    best_score = -1.0e9
    for index, left_peak in enumerate(fingers):
        for right_peak in fingers[index + 1 :]:
            left_x = _inner_x(left_peak, inner="right")
            right_x = _inner_x(right_peak, inner="left")
            left_y = distal[left_x]
            right_y = distal[right_x]
            separation = float(right_x - left_x)
            if not 10.0 <= separation <= 80.0:
                continue
            if abs(left_y - right_y) > 28.0:
                continue
            between = range(left_x + 3, right_x - 2)
            present = [x for x in between if x in distal]
            valley = 0.0
            if between:
                if len(present) == len(between):
                    mid_min = min(distal[x] for x in present)
                    if mid_min >= min(left_y, right_y) - 3:
                        continue
                    valley = float(min(left_y, right_y) - mid_min)
                else:
                    valley = float(
                        min(left_y, right_y)
                        - min((distal[x] for x in present), default=y0)
                    )
            candidate = np.asarray(
                [
                    [0.5 * (left_x + right_x), 0.5 * (left_y + right_y)],
                    [right_x, right_y],
                    [left_x, left_y],
                ],
                dtype=np.float32,
            )
            if is_patch_pitch_triple(candidate):
                continue
            attach_y = float(
                min(attach.get(left_peak, left_y), attach.get(right_peak, right_y))
            )
            pair_cx = 0.5 * (left_x + right_x)
            pair_cy = 0.5 * (left_y + right_y)
            if abs(pair_cx - wrist_cx) > 55.0:
                continue
            # Higher attach (true wrist) first; then more distal tips.
            score = (
                -attach_y * 200.0
                + pair_cy * 50.0
                - abs(pair_cx - wrist_cx) * 12.0
                - abs(left_y - right_y) * 4.0
                + valley * 2.0
            )
            if score > best_score:
                best_score = score
                best = (left_x, left_y, right_x, right_y)
    if best is None:
        return None
    left_x, left_y, right_x, right_y = best
    left_tip = (float(left_x), float(left_y))
    right_tip = (float(right_x), float(right_y))
    if min(left_y, right_y) < y0 + 5:
        return None
    for tip in (left_tip, right_tip):
        tx, ty = int(round(tip[0])), int(round(tip[1]))
        if not (0 <= ty < image.shape[0] and 0 <= tx < image.shape[1]):
            return None
        red, green, blue = (float(channel) for channel in image[ty, tx])
        if red > 140.0 and red > green + 25.0 and red > blue + 15.0:
            return None
    center = (
        0.5 * (left_tip[0] + right_tip[0]),
        0.5 * (left_tip[1] + right_tip[1]),
    )
    # Same C / L / R order as the planner: L is image-right, R is image-left.
    return np.asarray([center, right_tip, left_tip], dtype=np.float32)


def column_metal_may_override_fk(
    metal_points_px: np.ndarray | None,
    fk_points_px: np.ndarray,
    *,
    max_offset_px: float = 50.0,
    exclude_centroid_px: np.ndarray | None = None,
    exclude_radius_px: float = 60.0,
) -> bool:
    """True when a metal triple is allowed to replace the FK gripper.

    A blob on the pickup object, or a table pair tens of pixels from FK, is
    not the jaw. Control must keep FK (or an accepted observer) in that case.
    """

    if metal_points_px is None:
        return False
    metal = np.asarray(metal_points_px, dtype=np.float32).reshape(3, 2)
    fk = np.asarray(fk_points_px, dtype=np.float32).reshape(3, 2)
    if not np.all(np.isfinite(metal)) or not np.all(np.isfinite(fk)):
        return False
    if is_patch_pitch_triple(metal):
        return False
    if float(np.linalg.norm(metal[0] - fk[0])) > float(max_offset_px):
        return False
    if exclude_centroid_px is not None:
        exclude = np.asarray(exclude_centroid_px, dtype=np.float32).reshape(2)
        if np.all(np.isfinite(exclude)) and float(
            np.linalg.norm(metal[0] - exclude)
        ) <= float(exclude_radius_px):
            return False
    return True


__all__ = [
    "LOCATOR_PATCH_PITCH_PX",
    "column_content_rows",
    "column_l_r_on_metal",
    "column_metal_may_override_fk",
    "column_metal_operator_tips",
    "column_pixel_on_metal",
    "has_letterbox_pad",
    "is_patch_pitch_triple",
    "luma_image",
]
