"""Resolve the V11.5 live structured goal from existing 6-D joint targets.

Training uses a workspace slice terminal (deg, then /100). Live must pass the
same kind of task-stage endpoint, not the next 5-step chunk.

Live actor goal is two-phase. Workspace never auto-selects the stale Seeded IK:

    if current_lift <= WORKSPACE_LIFT_DEG (-50):
        PARK → C1 workspace-entry (leave-park ep0 cache index 94)
    else:
        if caller provided fresh_seeded_ik (6D, lift>-50, finite): use it
        else: C1 workspace-entry and needs_fresh_seeded_ik=True
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from v10.fish_to_box_task import TASK_CLASS, WORKSPACE_LIFT_DEG
from v10.task_class import FISH_TO_BOX, parse_task_class

_CATALOG = (
    Path(__file__).resolve().parents[1]
    / "v10"
    / "local_sources"
    / "fish_to_box_task_v1_catalog.json"
)

# Leave-park ep0 slice [65, 95) last frame (cache index 94). Degrees.
C1_WORKSPACE_ENTRY = np.array(
    [
        13.078034400939941,
        -49.279537200927734,
        66.45962524414062,
        63.087249755859375,
        -63.584415435791016,
        16.097240447998047,
    ],
    dtype=np.float32,
)
C1_WORKSPACE_ENTRY_SOURCE = "c1_workspace_entry:leave_park_ep0"

# Historical slice-6 Seeded IK. Degrees. Test/contrast only — resolve must
# never select this constant. A live workspace goal needs a fresh 6-D IK.
SEEDED_APPROACH_IK = np.array(
    [-0.090, -28.482, 53.787, 51.849, -50.901, 15.637],
    dtype=np.float32,
)
SEEDED_APPROACH_IK_SOURCE = "seeded_approach_ik:slice6_direct_contact"
FRESH_SEEDED_IK_SOURCE = "fresh_seeded_ik"
FRESH_IK_MAX_ERROR_M = 3e-3
FRESH_IK_MAX_FK_TO_FISH_M = 0.015
_VOID_IK_MIN_L2 = 1.0


def designed_approach_terminal(task_class: int) -> np.ndarray:
    """Locked class-1 train approach slice terminal. Catalog read only."""

    if parse_task_class(task_class) != FISH_TO_BOX:
        raise ValueError("designed approach terminal is only defined for task_class 1")
    catalog = json.loads(_CATALOG.read_text(encoding="utf-8"))
    if int(catalog.get("task_class", -1)) != TASK_CLASS:
        raise ValueError("fish_to_box catalog task_class is not 1")
    for item in catalog["slices"]:
        if item["phase"] == "approach" and item["split"] == "train":
            goal = np.asarray(item["terminal_state"], dtype=np.float32)
            if goal.shape != (6,) or not np.all(np.isfinite(goal)):
                raise ValueError("designed slice terminal must be a finite 6-vector")
            return goal
    raise RuntimeError("fish_to_box catalog has no train approach terminal")


def c1_workspace_entry() -> np.ndarray:
    """Locked PARK goal: C1 leave-park ep0 workspace-entry. Do not recompute IK."""

    return C1_WORKSPACE_ENTRY.copy()


def seeded_approach_ik() -> np.ndarray:
    """Historical slice-6 Seeded IK copy. Not a live resolve fallback."""

    return SEEDED_APPROACH_IK.copy()


def plan_relative_goal_xyz(plan: Any) -> np.ndarray | None:
    """Fish XYZ from plan.target.relative_goal_xyz. Never base_xyz."""

    target = getattr(plan, "target", None)
    raw = None
    if isinstance(target, dict):
        raw = target.get("relative_goal_xyz")
    elif target is not None:
        raw = getattr(target, "relative_goal_xyz", None)
    if raw is None:
        return None
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        return None
    return values


def solve_fresh_seeded_ik(
    *,
    fish_xyz: np.ndarray,
    kinematics: Any,
    live_gripper: float,
    generate_prefixes: Any | None = None,
) -> dict[str, Any] | None:
    """Re-solve contact IK for the current fish, seeded from C1, not live lift.

    Actor goal stays a 6-D joint vector. Pixel/XYZ are never returned as the
    goal. Failure returns None so the caller keeps C1 and the fresh-IK flag.
    """

    fish = np.asarray(fish_xyz, dtype=np.float64)
    if fish.shape != (3,) or not np.all(np.isfinite(fish)):
        return None
    if not np.isfinite(live_gripper):
        return None
    seed = c1_workspace_entry().astype(np.float64)
    if generate_prefixes is None:
        from fcpg.so101_ik_candidates import (
            REALTIME_CONTACT_PATH_SPECS,
            generate_ik_candidate_prefixes,
        )

        generate_prefixes = generate_ik_candidate_prefixes
        path_specs = REALTIME_CONTACT_PATH_SPECS
    else:
        path_specs = None
    kwargs: dict[str, Any] = {
        "frameskip": 5,
        "horizon": 8,
        "tolerance_m": FRESH_IK_MAX_ERROR_M,
    }
    if path_specs is not None:
        kwargs["path_specs"] = path_specs
    batch = generate_prefixes(kinematics, seed, fish, **kwargs)
    accepted: list[tuple[float, str, np.ndarray, float]] = []
    for index, name in enumerate(batch.names):
        if not bool(batch.converged[index]):
            continue
        error = float(batch.terminal_ik_error_m[index])
        if not np.isfinite(error) or error >= FRESH_IK_MAX_ERROR_M:
            continue
        terminal = np.asarray(batch.horizon_states[index, -1], dtype=np.float64)
        if terminal.shape != (6,) or not np.all(np.isfinite(terminal)):
            continue
        terminal = terminal.copy()
        terminal[5] = float(live_gripper)
        if float(terminal[1]) <= WORKSPACE_LIFT_DEG:
            continue
        if float(np.linalg.norm(terminal[:5] - SEEDED_APPROACH_IK[:5])) < _VOID_IK_MIN_L2:
            continue
        achieved = np.asarray(kinematics.forward(terminal), dtype=np.float64)
        if achieved.shape != (3,) or not np.all(np.isfinite(achieved)):
            continue
        fk_to_fish = float(np.linalg.norm(achieved - fish))
        if fk_to_fish >= FRESH_IK_MAX_FK_TO_FISH_M:
            continue
        accepted.append((error, str(name), terminal.astype(np.float32), fk_to_fish))
    if not accepted:
        return None
    direct = [row for row in accepted if row[1] == "direct_contact"]
    chosen = min(direct or accepted, key=lambda row: row[0])
    return {
        "goal_state": chosen[2],
        "source": f"{FRESH_SEEDED_IK_SOURCE}:c1_{chosen[1]}",
        "variant": chosen[1],
        "ik_error_m": chosen[0],
        "fk_to_fish_mm": 1000.0 * chosen[3],
        "fish_xyz": fish.astype(np.float64).tolist(),
    }


def _fresh_seeded_ik_or_none(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    values = np.asarray(raw, dtype=np.float32)
    if not is_workspace_joint_goal(values):
        return None
    return values


def is_workspace_joint_goal(state: np.ndarray) -> bool:
    values = np.asarray(state, dtype=np.float32)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        return False
    return float(values[1]) > WORKSPACE_LIFT_DEG


def _candidate_terminal(row: dict[str, Any]) -> np.ndarray | None:
    raw = row.get("terminal_joint_state")
    if raw is None:
        return None
    values = np.asarray(raw, dtype=np.float32)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        return None
    return values


def pickup_ik_workspace_terminal(plan: Any) -> tuple[np.ndarray, str] | None:
    """Best converged pickup/contact IK terminal that is already in-workspace.

    Not used as a live actor goal. Folded-seed live IK is rejected; catalog
    approach terminals are also not live goals.
    """

    rows = list(getattr(plan, "candidate_rows", None) or [])
    best: tuple[float, np.ndarray, str] | None = None
    for row in rows:
        if row.get("source") != "calibrated_cartesian_ik":
            continue
        if row.get("goal_role") != "pickup":
            continue
        if row.get("ik_converged") is not True:
            continue
        terminal = _candidate_terminal(row)
        if terminal is None or not is_workspace_joint_goal(terminal):
            continue
        error = row.get("terminal_ik_error_m")
        rank = float(error) if error is not None and np.isfinite(error) else 1.0
        variant = str(row.get("variant") or "pickup_ik")
        if best is None or rank < best[0]:
            best = (rank, terminal, variant)
    if best is None:
        return None
    return best[1], f"pickup_ik_terminal:{best[2]}"


def resolve_v11_live_goal(
    plan: Any,
    *,
    task_class: int,
    current_state: np.ndarray | None = None,
    fresh_seeded_ik: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return a 6-D joint task goal. Never current/chunk/commanded endpoints.

    PARK stays C1. Workspace uses caller fresh_seeded_ik only when it is a
    finite 6-vector with lift > -50; otherwise keeps C1 and flags
    needs_fresh_seeded_ik. Never selects SEEDED_APPROACH_IK.
    """

    parse_task_class(task_class)
    state = np.asarray(
        current_state
        if current_state is not None
        else getattr(plan, "current_state"),
        dtype=np.float32,
    )
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("current_state must be a finite 6-vector")

    needs_fresh_seeded_ik = False
    if float(state[1]) <= WORKSPACE_LIFT_DEG:
        goal = c1_workspace_entry()
        source = C1_WORKSPACE_ENTRY_SOURCE
        phase = "park"
    else:
        fresh = _fresh_seeded_ik_or_none(fresh_seeded_ik)
        if fresh is not None:
            goal = fresh.astype(np.float32, copy=True)
            source = FRESH_SEEDED_IK_SOURCE
            phase = "workspace"
        else:
            goal = c1_workspace_entry()
            source = C1_WORKSPACE_ENTRY_SOURCE
            phase = "workspace"
            needs_fresh_seeded_ik = True

    delta = goal[:5] - state[:5]
    return {
        "goal_state": goal.astype(np.float32),
        "source": source,
        "phase": phase,
        "l2_arm_deg": float(np.linalg.norm(delta)),
        "goal_lift_deg": float(goal[1]),
        "state_lift_deg": float(state[1]),
        "workspace_goal": is_workspace_joint_goal(goal),
        "needs_fresh_seeded_ik": needs_fresh_seeded_ik,
    }


__all__ = [
    "C1_WORKSPACE_ENTRY",
    "C1_WORKSPACE_ENTRY_SOURCE",
    "FRESH_IK_MAX_ERROR_M",
    "FRESH_IK_MAX_FK_TO_FISH_M",
    "FRESH_SEEDED_IK_SOURCE",
    "SEEDED_APPROACH_IK",
    "SEEDED_APPROACH_IK_SOURCE",
    "c1_workspace_entry",
    "designed_approach_terminal",
    "is_workspace_joint_goal",
    "pickup_ik_workspace_terminal",
    "plan_relative_goal_xyz",
    "resolve_v11_live_goal",
    "seeded_approach_ik",
    "solve_fresh_seeded_ik",
]
