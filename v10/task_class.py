"""Human-assigned actor task class IDs.

The operator sets the class at train, eval, and live serve. The model does
not infer which skill to run. LeWM stays class-free.

This is not LeWM A4 ``task_class_index`` (0-based public-mix identity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


FISH_TO_BOX = 1


@dataclass(frozen=True)
class TaskClass:
    task_class: int
    name: str
    title: str
    dataset: str | None
    phases: tuple[str, ...]
    status: str
    visual_roles: tuple[str, ...]
    visual_queries: Mapping[str, str | tuple[str, ...]]


TASK_CLASSES: dict[int, TaskClass] = {
    FISH_TO_BOX: TaskClass(
        task_class=FISH_TO_BOX,
        name="fish_to_box",
        title="Pick up the red fish toy and place it in the cardboard box",
        dataset="fish_to_box_task_v1",
        phases=("approach", "grasp"),
        status="active",
        visual_roles=("pickup", "drop_zone", "gripper"),
        visual_queries={
            "pickup": ("red fish toy",),
            "drop_zone": (
                "cardboard box",
                "shipping box",
                "open cardboard box",
                "box",
            ),
            "gripper": (
                "robot gripper",
                "robotic parallel jaw gripper",
                "white robot arm",
            ),
        },
    ),
}


def parse_task_class(value: Any) -> int:
    """Accept a human-set class ID. Reject bools, floats, and unknown IDs."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("task_class must be a human-set integer")
    spec = TASK_CLASSES.get(value)
    if spec is None:
        known = ", ".join(str(key) for key in sorted(TASK_CLASSES))
        raise ValueError(f"unknown task_class {value}; known IDs: {known}")
    return spec.task_class


def require_matching_task_class(requested: Any, declared: Any) -> int:
    left = parse_task_class(requested)
    right = parse_task_class(declared)
    if left != right:
        raise ValueError(
            f"task_class {left} does not match declared class {right}"
        )
    return left


def require_source_task_class(payload: Mapping[str, Any], task_class: Any) -> int:
    if "task_class" not in payload:
        raise ValueError("dataset source is missing task_class")
    return require_matching_task_class(task_class, payload["task_class"])


def require_catalog_task_class(payload: Mapping[str, Any], task_class: Any) -> int:
    """Every catalog slice must carry the same human-set class as the source."""

    tagged = require_source_task_class(payload, task_class)
    slices = payload.get("slices")
    if not isinstance(slices, list) or not slices:
        raise ValueError("catalog has no slices to tag")
    for index, item in enumerate(slices):
        if not isinstance(item, Mapping):
            raise ValueError(f"catalog slice {index} is not a mapping")
        if "task_class" not in item:
            raise ValueError(f"catalog slice {index} is missing task_class")
        require_matching_task_class(tagged, item["task_class"])
    return tagged


def task_class_record(task_class: Any) -> dict[str, Any]:
    spec = TASK_CLASSES[parse_task_class(task_class)]
    return {
        "task_class": spec.task_class,
        "task_class_name": spec.name,
        "task_class_title": spec.title,
        "task_class_dataset": spec.dataset,
        "task_class_phases": list(spec.phases),
        "task_class_visual_roles": list(spec.visual_roles),
        "task_class_visual_queries": {
            role: list(_as_phrases(value))
            for role, value in spec.visual_queries.items()
        },
    }


def task_visual_roles(task_class: Any) -> tuple[str, ...]:
    """Roles this skill must ground. Data, not a detector."""

    spec = TASK_CLASSES[parse_task_class(task_class)]
    if spec.visual_queries:
        return tuple(spec.visual_queries)
    return spec.visual_roles


def _as_phrases(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        items = (value,)
    elif isinstance(value, (list, tuple)):
        items = tuple(value)
    else:
        return ()
    return tuple(str(item).strip() for item in items if str(item).strip())


def task_visual_queries(task_class: Any) -> dict[str, tuple[str, ...]]:
    """Open-vocab phrases for each role. Data, not a crop library."""

    raw = TASK_CLASSES[parse_task_class(task_class)].visual_queries
    return {
        str(role): phrases
        for role, value in raw.items()
        if (phrases := _as_phrases(value))
    }


__all__ = [
    "FISH_TO_BOX",
    "TASK_CLASSES",
    "TaskClass",
    "parse_task_class",
    "require_matching_task_class",
    "require_source_task_class",
    "require_catalog_task_class",
    "task_class_record",
    "task_visual_roles",
    "task_visual_queries",
]
