"""SO-101 safety checker adapter.

Wraps the existing preflight_five anchor-clearance gate behind
SafetyCheckerInterface.  All thresholds come from WorkspaceConfig
(loaded from YAML), not hardcoded here.
"""

from __future__ import annotations

import numpy as np

from interfaces.planning import SafetyCheckerInterface, WorkspaceConfig
from interfaces.robot import PreflightResult


class SO101SafetyChecker(SafetyCheckerInterface):
    """SafetyCheckerInterface for SO-101 using FK anchor clearance.

    Config keys
    -----------
    (none beyond what WorkspaceConfig provides)
    """

    def __init__(self) -> None:
        from fcpg.so101_kinematics import SO101Kinematics
        self._kinematics = SO101Kinematics()

    def check(
        self,
        commands: np.ndarray,
        current_state: np.ndarray,
        workspace: WorkspaceConfig,
    ) -> PreflightResult:
        """Check every anchor across the command chunk against table clearance."""
        path = np.vstack([current_state[None], commands])   # (K+1, 6)
        try:
            anchors = np.asarray(
                self._kinematics.forward_anchors(path), dtype=np.float64
            )
        except Exception as exc:
            return PreflightResult(safe=False, reason=f"FK failed: {exc}")

        if not np.all(np.isfinite(anchors)):
            return PreflightResult(safe=False, reason="FK returned non-finite anchors")

        min_z = float(np.min(anchors[..., 2]))
        clearance = min_z - workspace.table_z_m

        if clearance < workspace.minimum_anchor_clearance_m:
            return PreflightResult(
                safe=False,
                reason=(
                    f"anchor clearance {clearance:.4f} m < "
                    f"required {workspace.minimum_anchor_clearance_m:.4f} m"
                ),
                minimum_clearance_m=clearance,
            )

        # workspace XY bounds
        xyz = anchors.reshape(-1, 3)
        lo = np.array(workspace.lower_bound_m)
        hi = np.array(workspace.upper_bound_m)
        if not np.all((xyz >= lo) & (xyz <= hi)):
            return PreflightResult(safe=False, reason="command leaves workspace bounds")

        return PreflightResult(safe=True, minimum_clearance_m=clearance)
