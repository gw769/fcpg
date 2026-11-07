"""Explicit adapters for the existing real-camera projection fits.

These are approximate pixel projections, NOT fabricated pinhole extrinsics or
metric 3-D ground truth. CAD uses the separate actual K/T projection adapter.
"""
import json
from pathlib import Path

import torch
from torch import nn

from .prod_v20_observer import V20MechanicalProjection
from .robot_kinematic_graph import DifferentiableRobotKinematics, so101_robot_spec


class V20FittedMechanicalProjection(V20MechanicalProjection):
    geometry_kind = "existing-affine-pixel-fit-not-metric-extrinsics"

    def __init__(self, calibration_path, *, device="cpu", camera_valid=None, kinematics=None):
        nn.Module.__init__(self)
        payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        expected = dict(wrist=("x_gripper", "y_gripper", "z_gripper", "opening", "bias"),
                        column=("x_base", "y_base", "z_base", "bias"))
        for name, features in expected.items():
            row = payload.get("fits", {}).get(name, {})
            if tuple(row.get("feature_names", ())) != features or row.get("view") != name:
                raise ValueError(f"{name} projection feature convention differs")
            coefficients = torch.as_tensor(row["coefficients"], dtype=torch.float32, device=device)
            if coefficients.shape != (len(features), 2) or not torch.isfinite(coefficients).all():
                raise ValueError("invalid fitted camera coefficients")
            self.register_buffer(name + "_coefficients", coefficients)
        self.kinematics = (kinematics or DifferentiableRobotKinematics(so101_robot_spec())).to(device)
        self.register_buffer("camera_valid", torch.ones(2, dtype=torch.bool, device=device) if camera_valid is None
                             else torch.as_tensor(camera_valid, dtype=torch.bool, device=device))

    def project(self, q_position):
        if q_position.ndim != 2 or q_position.shape[1] != 6 or not torch.isfinite(q_position).all():
            raise ValueError("projection expects finite calibrated positions [B,6]")
        q = q_position.float()
        # Keep geometry algebra in fp32 even inside bf16 training.
        with torch.autocast(device_type=q.device.type, enabled=False):
            base, _, _ = self.kinematics(q)
            references = self.kinematics.anchors_in_view_references(q)
            indices = [self.kinematics.spec.fixed_tip_index, self.kinematics.spec.moving_tip_index]
            wrist = references[:, 0, indices]
            column = base[:, indices]
            ones = torch.ones_like(column[..., :1])
            opening = (q[:, -1].clamp(0, 100) / 100.)[:, None, None].expand(-1, 2, 1)
            wrist_features = torch.cat((wrist, opening, ones), -1)
            column_features = torch.cat((column, ones), -1)
            pixels = torch.stack((wrist_features @ self.wrist_coefficients,
                                  column_features @ self.column_coefficients), 1)
        valid = self.camera_valid
        if valid.ndim == 1:
            valid = valid[None].expand(len(q), -1)
        valid = valid[..., None].expand(-1, -1, 2) & torch.isfinite(pixels).all(-1)
        return torch.where(valid[..., None], pixels, torch.zeros_like(pixels)), valid

    forward = project
