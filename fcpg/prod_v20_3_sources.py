"""Unchanged V20.1 data correspondence; explicit V20.3 geometry adapters.

No invented archive calibration and no dataset-ID-as-extrinsics replacement.
"""
from .prod_v20_1_sources import V201Sources, V201SourcePaths
from .prod_v20_1_geometry import V201FittedMechanicalProjection
from .prod_v20_3_geometry import V203MechanicalProjection, V203FittedMechanicalProjection
import numpy as np

V203SourcePaths = V201SourcePaths


class V203Sources(V201Sources):
    def motion_chain_batch(self, rows):
        """Factual archive physics only; ARM consumes no image/task fields."""
        selected = np.asarray(rows, dtype=np.int64)
        if selected.ndim != 1 or not len(selected) or (selected < 0).any() or (selected >= self.counts["archive"]).any():
            raise ValueError("invalid motion archive rows")
        chain = self.archive.chains(selected)
        return dict(q_position=chain.q_position, commands_position=chain.commands_position,
            measured_q_paths_position=chain.measured_q_paths_position,
            time_deltas=np.diff(chain.readback_time_offsets_s, axis=-1),
            source_index=self._archive_source_index[selected].copy(),
            initial_time_s=chain.endpoint_time_s[:, 0])

    def fish_geometry(self):
        if self._fish_geometry is None:
            self._fish_geometry = V203FittedMechanicalProjection(
                self.paths.real_calibration, device=self.device, kinematics=self.kinematics())
        return self._fish_geometry

    def _geometry_adapter(self, batch):
        geometry = batch.get("geometry")
        if geometry is None or isinstance(geometry, (V203MechanicalProjection, V203FittedMechanicalProjection)):
            return batch
        if isinstance(geometry, V201FittedMechanicalProjection):
            updated = V203FittedMechanicalProjection(self.paths.real_calibration,
                device=self.device, camera_valid=geometry.camera_valid, kinematics=geometry.kinematics)
        else:
            updated = V203MechanicalProjection(geometry.kinematics, geometry.camera_from_reference,
                geometry.intrinsics, geometry.image_transform, geometry.camera_valid)
        return dict(batch, geometry=updated)

    def cad_batch(self, indices):
        return self._geometry_adapter(super().cad_batch(indices))

    def real_batch(self, indices=None, *, batch_size=None):
        return self._geometry_adapter(super().real_batch(indices, batch_size=batch_size))

    def real_evaluation_batch(self, indices):
        return self._geometry_adapter(super().real_evaluation_batch(indices))

    def preflight(self, *, media=False):
        report = super().preflight(media=media)
        report["actual_localization"] = {
            "actual_readback_updates": True, "vision_corrects_once": True,
            "archive_camera_geometry_available": False,
            "fish_geometry_is_metric_stereo": False,
            "missing_geometry_is_not_occlusion_tracking_evidence": True,
        }
        return report
