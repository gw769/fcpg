"""View-local object overlay geometry — no Actor loss, display only."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def same_view_object_contour(
    rec: Mapping[str, Any] | None,
    view: str,
    frame_hw: tuple[int, int],
) -> list[list[float]]:
    """Contour for ``view`` only — refuse cross-camera paste.

    Column JPEG must use column support pixels; wrist JPEG must use wrist.
    Returns [] when the record is missing, tagged for another camera, or the
    polygon leaves this frame. Dense masks are optional and must match ``frame_hw``.
    """
    if not isinstance(rec, Mapping) or not rec.get("visible"):
        return []
    if rec.get("view") not in (None, view):
        return []
    height, width = int(frame_hw[0]), int(frame_hw[1])
    contour = list(rec.get("contour_px") or [])
    if len(contour) < 3:
        mask = rec.get("support_mask")
        if mask is not None:
            array = np.asarray(mask)
            if array.ndim == 2 and array.shape[:2] == (height, width):
                from .so101_frozen_vit_grounding import support_contour_px

                contour = support_contour_px(array)
            else:
                contour = []
    kept: list[list[float]] = []
    for point in contour:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x, y = float(point[0]), float(point[1])
        if 0.0 <= x < width and 0.0 <= y < height:
            kept.append([x, y])
    return kept if len(kept) >= 3 else []


__all__ = ["same_view_object_contour"]
