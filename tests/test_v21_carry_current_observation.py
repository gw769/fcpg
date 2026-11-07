"""Release needs current exposure evidence, even when old targets look aligned."""
from dataclasses import replace

import pytest
import torch

from test_v21_live_service import service  # noqa: F401


@pytest.mark.parametrize('evidence', [
    'current', 'historical', 'missing_camera', 'wrong_exposure',
    'missing_provenance', 'expired', 'future',
])
def test_release_cannot_use_historical_or_unpaired_column_targets(service, evidence):
    owner = service.owner
    owner.runtime.state = replace(owner.runtime.state,
        q_error=torch.zeros_like(owner.runtime.state.q_error),
        feedback_available=torch.ones_like(owner.runtime.state.feedback_available))
    stamp = float(owner.last_endpoint.observation.image_times[0, 1])
    box = dict(visible=True, centroid_px=[100., 100.], image_time_s=stamp,
               view_source='current_observation')
    fish = dict(box, centroid_px=[100., 100.-service.CARRY_TARGET_ABOVE_PX])
    if evidence == 'historical':
        fish['view_source'] = box['view_source'] = 'historical_observation_not_live'
    elif evidence == 'missing_camera':
        owner.last_endpoint.observation.measurement_available[0, 1] = False
    elif evidence == 'wrong_exposure':
        fish['image_time_s'] -= 1.
    elif evidence == 'missing_provenance':
        del box['view_source']
    elif evidence == 'expired':
        owner.advance_clock(owner.runtime.max_plan_age_s + 1.)
    elif evidence == 'future':
        owner.advance_clock(-1.)
    service.targets = dict(targets=dict(drop_zone=dict(column=box), pickup=dict(column=fish)))
    service._carry = dict(phase='release', blocks=0)
    q = torch.tensor([0., 0., 0., 0., 0., 5.])
    commands = q[None, None].repeat(1, 5, 1)
    record = {}
    result = service._carry_builder(commands, q, record)
    accepted = evidence == 'current'
    gate = record['carry']['lewm_release_gate']
    assert gate['accepted'] is accepted
    assert gate['current_column_observation'] is accepted
    if accepted:
        assert result[0, -1, 5] > q[5]
    else:
        assert torch.equal(result[0, :, 5], q[5].expand(5))
        assert record['carry']['release_blocked'] == 'column_observation_untrusted'
    assert owner.writes == 0
