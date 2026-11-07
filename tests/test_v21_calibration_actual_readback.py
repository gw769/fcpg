from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_v21_live_service import service  # noqa: F401


def prepare(service,actual_delta):
    service.owner.last_object_diagnostic=[1.,.8,0.,0.,.2,.1,1.,0.]
    original=service.owner.last_endpoint
    def run(builder,source):
        builder(SimpleNamespace(q=original.q))
        observation=original.observation
        q=observation.q_at_image.clone();q[0,0,0]+=actual_delta
        updated=replace(observation,q_at_image=q,image_times=observation.image_times+.5)
        service.owner.last_endpoint=replace(original,observation=updated)
        service.owner.last_object_diagnostic=[1.,.8,4./224.,0.,.2,.1,1.,0.]
        return dict(completed=True)
    service._run_alignment_motion=run


def test_scalar_gain_uses_achieved_motion_not_requested_two_units(service):
    prepare(service,1.)
    result=service._grounding_calibrate(2.)['calibration']
    assert result['j1_moved']==pytest.approx(1.)
    assert result['px_per_unit']==pytest.approx(4.)
    assert not result['requested_motion_used_as_readback']


def test_negligible_actual_movement_does_not_update_gain(service):
    prepare(service,.05);service.grounding_px_per_unit=7.
    with pytest.raises(RuntimeError,match='insufficient'):
        service._grounding_calibrate(2.)
    assert service.grounding_px_per_unit==7.


def test_oversized_calibration_probe_is_rejected_before_runner(service):
    service._run_alignment_motion=lambda *a:pytest.fail('must not dispatch')
    with pytest.raises(ValueError,match='at most two'):
        service._grounding_calibrate(3.)
