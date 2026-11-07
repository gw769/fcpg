"""Locate task roles with a frozen open-vocabulary detector.

The detector is a pretrained open-vocab model. It is never trained on this
robot, this table, or this object. A role is whatever phrase the task table
gives it. Finding the roles is one text-conditioned forward plus an exclusive
assignment: two roles cannot own the same box. There is no crop library and
no colour model.

``DEFAULT_MIN_SCORE`` is a noise floor for collecting proposals, not the
accept gate. A box is accepted when it wins its role, does not overlap a
higher-scoring assigned box, sits in the camera content rectangle, and
stands out from the other proposals in that frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

DEFAULT_MODEL = "google/owlvit-base-patch32"
# OWL-ViT scores after the library's post-process. Below this a phrase did
# not fire on its own; relative ranking may still accept a standout below it.
DEFAULT_MIN_SCORE = 0.08
DEFAULT_COLLECTION_THRESHOLD = 0.002
DEFAULT_RELATIVE_MARGIN = 0.02
DEFAULT_RELATIVE_RATIO = 1.25
DEFAULT_EXCLUSIVE_IOU = 0.5
# A role is an object in the scene. A box that is most of the content
# rectangle is the frame; a few-pixel sliver is noise.
DEFAULT_MAX_CONTENT_FRACTION = 0.45
DEFAULT_MIN_CONTENT_FRACTION = 0.02
# Detect on the camera content, not the letterbox pad, at a larger canvas
# so a 224 planning frame still has enough patches for a table-scale object.
DEFAULT_CONTENT_SCALE = 2.0


@dataclass(frozen=True)
class OpenVocabHit:
    centroid_px: tuple[float, float]
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    query: str
    second_score: float = -1.0
    margin: float | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "centroid_px": [float(self.centroid_px[0]), float(self.centroid_px[1])],
            "score": float(self.score),
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
            "query": self.query,
            "second_score": float(self.second_score),
            "margin": None if self.margin is None else float(self.margin),
        }


@dataclass(frozen=True)
class OpenVocabProposal:
    role: str
    query: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def centroid_px(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox_xyxy
        return (0.5 * (float(x0) + float(x1)), 0.5 * (float(y0) + float(y1)))


@dataclass(frozen=True)
class ExclusiveAssignment:
    hits: dict[str, OpenVocabHit | None]
    reasons: dict[str, str]


def box_iou(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    ix0 = max(float(left[0]), float(right[0]))
    iy0 = max(float(left[1]), float(right[1]))
    ix1 = min(float(left[2]), float(right[2]))
    iy1 = min(float(left[3]), float(right[3]))
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_left = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    area_right = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = area_left + area_right - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def content_detection_view(
    rgb: np.ndarray,
    content_rect: Sequence[int] | None,
    *,
    scale: float = DEFAULT_CONTENT_SCALE,
) -> tuple[np.ndarray, Any]:
    """Crop pad, upscale content. Returns (image, box_to_canvas)."""

    array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))

    def _identity(bbox: Sequence[float]) -> tuple[float, float, float, float]:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    if content_rect is None or float(scale) <= 1.0:
        return array, _identity
    left, top, right, bottom = (int(v) for v in content_rect)
    if right <= left or bottom <= top:
        return array, _identity
    crop = array[top:bottom, left:right]
    if crop.size == 0:
        return array, _identity
    import cv2

    width = max(1, int(round(crop.shape[1] * float(scale))))
    height = max(1, int(round(crop.shape[0] * float(scale))))
    view = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)

    def _to_canvas(bbox: Sequence[float]) -> tuple[float, float, float, float]:
        inv = 1.0 / float(scale)
        return (
            float(bbox[0]) * inv + left,
            float(bbox[1]) * inv + top,
            float(bbox[2]) * inv + left,
            float(bbox[3]) * inv + top,
        )

    return view, _to_canvas


def centroid_in_content(
    centroid_px: Sequence[float],
    content_rect: Sequence[int] | None,
) -> bool:
    if content_rect is None:
        return True
    left, top, right, bottom = (int(v) for v in content_rect)
    x, y = float(centroid_px[0]), float(centroid_px[1])
    return left <= x < right and top <= y < bottom


def stands_out(
    best: float,
    second: float | None,
    *,
    noise_floor: float = DEFAULT_MIN_SCORE,
    relative_margin: float = DEFAULT_RELATIVE_MARGIN,
    relative_ratio: float = DEFAULT_RELATIVE_RATIO,
) -> bool:
    """True when ``best`` is the role's box, not just a score above a cutoff."""

    if second is None or second < 0.0:
        return float(best) >= float(noise_floor)
    gap = float(best) - float(second)
    if gap >= float(relative_margin):
        return True
    if second > 0.0 and (float(best) / float(second)) >= float(relative_ratio):
        return True
    return float(best) >= float(noise_floor) and gap > 0.0


def exclusive_assign(
    proposals: Sequence[OpenVocabProposal],
    roles: Sequence[str],
    *,
    content_rect: Sequence[int] | None = None,
    noise_floor: float = DEFAULT_MIN_SCORE,
    relative_margin: float = DEFAULT_RELATIVE_MARGIN,
    relative_ratio: float = DEFAULT_RELATIVE_RATIO,
    exclusive_iou: float = DEFAULT_EXCLUSIVE_IOU,
    max_content_fraction: float = DEFAULT_MAX_CONTENT_FRACTION,
    min_content_fraction: float = DEFAULT_MIN_CONTENT_FRACTION,
) -> ExclusiveAssignment:
    """Greedy highest-score assignment. One box cannot serve two roles."""

    hits: dict[str, OpenVocabHit | None] = {str(role): None for role in roles}
    reasons: dict[str, str] = {
        str(role): "open_vocab_not_found" for role in roles
    }
    content_area = None
    if content_rect is not None:
        left, top, right, bottom = (int(v) for v in content_rect)
        content_area = max(0, right - left) * max(0, bottom - top)
    max_area = (
        None
        if content_area is None
        else float(max_content_fraction) * float(content_area)
    )
    min_area = (
        None
        if content_area is None
        else float(min_content_fraction) * float(content_area)
    )

    def _area(proposal: OpenVocabProposal) -> float:
        x0, y0, x1, y1 = proposal.bbox_xyxy
        return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))

    def _rank(proposal: OpenVocabProposal) -> float:
        area = _area(proposal)
        if content_area and content_area > 0:
            frac = min(max(area / float(content_area), 1.0e-6), 1.0)
            return float(proposal.score) * (frac ** 0.25)
        return float(proposal.score)

    in_content = [
        proposal
        for proposal in proposals
        if str(proposal.role) in hits
        and centroid_in_content(proposal.centroid_px, content_rect)
        and (max_area is None or _area(proposal) <= max_area)
        and (min_area is None or _area(proposal) >= min_area)
    ]
    outside = [
        proposal
        for proposal in proposals
        if str(proposal.role) in hits
        and not centroid_in_content(proposal.centroid_px, content_rect)
    ]
    too_large = [
        proposal
        for proposal in proposals
        if str(proposal.role) in hits
        and centroid_in_content(proposal.centroid_px, content_rect)
        and max_area is not None
        and _area(proposal) > max_area
    ]
    too_small = [
        proposal
        for proposal in proposals
        if str(proposal.role) in hits
        and centroid_in_content(proposal.centroid_px, content_rect)
        and min_area is not None
        and _area(proposal) < min_area
    ]
    for proposal in outside:
        if reasons[str(proposal.role)] == "open_vocab_not_found":
            reasons[str(proposal.role)] = "open_vocab_outside_content"
    for proposal in too_large:
        if reasons[str(proposal.role)] in {
            "open_vocab_not_found",
            "open_vocab_outside_content",
        }:
            reasons[str(proposal.role)] = "open_vocab_too_large"
    for proposal in too_small:
        if reasons[str(proposal.role)] in {
            "open_vocab_not_found",
            "open_vocab_outside_content",
        }:
            reasons[str(proposal.role)] = "open_vocab_too_small"
    if not in_content:
        return ExclusiveAssignment(hits=hits, reasons=reasons)

    ranked = sorted(in_content, key=_rank, reverse=True)
    taken: list[OpenVocabProposal] = []
    for proposal in ranked:
        role = str(proposal.role)
        if hits[role] is not None:
            continue
        if any(
            box_iou(proposal.bbox_xyxy, owned.bbox_xyxy) >= float(exclusive_iou)
            for owned in taken
        ):
            if reasons[role] == "open_vocab_not_found":
                reasons[role] = "open_vocab_not_exclusive"
            continue
        same_role = [
            other
            for other in in_content
            if other is not proposal
            and str(other.role) == role
            and not any(
                box_iou(other.bbox_xyxy, owned.bbox_xyxy) >= float(exclusive_iou)
                for owned in taken
            )
        ]
        second = max((float(other.score) for other in same_role), default=None)
        refined_scene = second is None and any(
            str(other.role) == role for other in too_large
        )
        if refined_scene:
            distinct = float(proposal.score) > 0.0
        else:
            distinct = stands_out(
                float(proposal.score),
                second,
                noise_floor=noise_floor,
                relative_margin=relative_margin,
                relative_ratio=relative_ratio,
            )
        if not distinct:
            reasons[role] = "open_vocab_not_distinct"
            continue
        margin = None if second is None else float(proposal.score) - float(second)
        hits[role] = OpenVocabHit(
            centroid_px=proposal.centroid_px,
            score=float(proposal.score),
            bbox_xyxy=tuple(float(v) for v in proposal.bbox_xyxy),
            query=str(proposal.query),
            second_score=-1.0 if second is None else float(second),
            margin=margin,
        )
        reasons[role] = "accepted"
        taken.append(proposal)
    return ExclusiveAssignment(hits=hits, reasons=reasons)


class FrozenOpenVocabGrounder:
    """Frozen text-conditioned boxes. Shared by every view and every task."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        device: torch.device | str = "cpu",
        min_score: float = DEFAULT_MIN_SCORE,
        collection_threshold: float = DEFAULT_COLLECTION_THRESHOLD,
        relative_margin: float = DEFAULT_RELATIVE_MARGIN,
        relative_ratio: float = DEFAULT_RELATIVE_RATIO,
        exclusive_iou: float = DEFAULT_EXCLUSIVE_IOU,
        max_content_fraction: float = DEFAULT_MAX_CONTENT_FRACTION,
        min_content_fraction: float = DEFAULT_MIN_CONTENT_FRACTION,
    ) -> None:
        self.model_name = str(model_name)
        self.device = torch.device(device)
        self.min_score = float(min_score)
        self.collection_threshold = float(collection_threshold)
        self.relative_margin = float(relative_margin)
        self.relative_ratio = float(relative_ratio)
        self.exclusive_iou = float(exclusive_iou)
        self.max_content_fraction = float(max_content_fraction)
        self.min_content_fraction = float(min_content_fraction)
        self._model: Any | None = None
        self._processor: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        import os

        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        offline = os.environ.get("HF_HUB_OFFLINE", "").strip() in {"1", "true", "TRUE"}
        kwargs: dict[str, Any] = {"local_files_only": offline}
        processor = OwlViTProcessor.from_pretrained(self.model_name, **kwargs)
        model = OwlViTForObjectDetection.from_pretrained(self.model_name, **kwargs)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._processor = processor
        self._model = model.to(self.device).eval()
        return self._model, self._processor

    @torch.no_grad()
    def _proposals(
        self,
        rgb: np.ndarray,
        phrases: Sequence[str],
    ) -> list[tuple[int, float, tuple[float, float, float, float]]]:
        array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB, received {array.shape}")
        if not phrases:
            return []
        height, width = int(array.shape[0]), int(array.shape[1])
        model, processor = self._load()
        inputs = processor(text=[list(phrases)], images=array, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = model(**inputs)
        target_sizes = torch.tensor([[height, width]], device=self.device)
        processed = processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.collection_threshold,
            target_sizes=target_sizes,
        )[0]
        scores = processed["scores"].detach().cpu().numpy()
        boxes = processed["boxes"].detach().cpu().numpy()
        labels = processed["labels"].detach().cpu().numpy()
        proposals: list[tuple[int, float, tuple[float, float, float, float]]] = []
        phrase_count = len(phrases)
        for score, box, label in zip(scores, boxes, labels):
            index = int(label)
            if index < 0 or index >= phrase_count:
                continue
            proposals.append(
                (
                    index,
                    float(score),
                    (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                )
            )
        return proposals

    def locate_many(
        self,
        rgb: np.ndarray,
        queries: Mapping[str, str | Sequence[str]],
        *,
        content_rect: Sequence[int] | None = None,
        search_centre_px: tuple[float, float] | None = None,
        search_radius_px: float | None = None,
    ) -> ExclusiveAssignment:
        role_phrases: list[tuple[str, str]] = []
        for role, query in queries.items():
            if isinstance(query, str):
                items: Sequence[str] = (query,)
            else:
                items = query
            for item in items:
                phrase = str(item).strip()
                if phrase:
                    role_phrases.append((str(role), phrase))
        empty = ExclusiveAssignment(
            hits={str(role): None for role in queries},
            reasons={
                str(role): (
                    "target_query_not_configured"
                    if not role_phrases
                    or not any(name == str(role) for name, _ in role_phrases)
                    else "open_vocab_not_found"
                )
                for role in queries
            },
        )
        if not role_phrases:
            return empty
        phrases = [phrase for _, phrase in role_phrases]
        view, to_canvas = content_detection_view(rgb, content_rect)
        raw = self._proposals(view, phrases)
        proposals: list[OpenVocabProposal] = []
        for index, score, bbox in raw:
            bbox = to_canvas(bbox)
            cx = 0.5 * (bbox[0] + bbox[2])
            cy = 0.5 * (bbox[1] + bbox[3])
            if search_centre_px is not None and search_radius_px is not None:
                dist = float(
                    np.hypot(
                        cx - float(search_centre_px[0]),
                        cy - float(search_centre_px[1]),
                    )
                )
                if dist > float(search_radius_px):
                    continue
            role, phrase = role_phrases[index]
            proposals.append(
                OpenVocabProposal(
                    role=role,
                    query=phrase,
                    score=score,
                    bbox_xyxy=bbox,
                )
            )
        assigned = exclusive_assign(
            proposals,
            list(queries),
            content_rect=content_rect,
            noise_floor=self.min_score,
            relative_margin=self.relative_margin,
            relative_ratio=self.relative_ratio,
            exclusive_iou=self.exclusive_iou,
            max_content_fraction=self.max_content_fraction,
            min_content_fraction=self.min_content_fraction,
        )
        hits = {
            str(role): assigned.hits.get(str(role)) for role in queries
        }
        reasons = {
            str(role): assigned.reasons.get(str(role), "open_vocab_not_found")
            for role in queries
        }
        for role, query in queries.items():
            if not str(query).strip():
                reasons[str(role)] = "target_query_not_configured"
        return ExclusiveAssignment(hits=hits, reasons=reasons)

    @torch.no_grad()
    def locate(
        self,
        rgb: np.ndarray,
        query: str,
        *,
        search_centre_px: tuple[float, float] | None = None,
        search_radius_px: float | None = None,
        content_rect: Sequence[int] | None = None,
    ) -> OpenVocabHit | None:
        phrase = str(query).strip()
        if not phrase:
            return None
        assigned = self.locate_many(
            rgb,
            {"_": phrase},
            content_rect=content_rect,
            search_centre_px=search_centre_px,
            search_radius_px=search_radius_px,
        )
        return assigned.hits.get("_")
