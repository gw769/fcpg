"""The unmodified Actor route retains LeWM without requiring pixel transport."""
from unittest.mock import Mock

from test_v21_live_service import service  # noqa: F401


def test_raw_actor_plans_without_jacobian_and_keeps_physical_and_lewm_checks(service, monkeypatch):
    service.raw_actor = True
    service.lewm_cem = False
    service.column_dino_fg_calibration_enabled = False
    service.column_dino_fg_response = None
    service.fingertips = {}
    runtime = service.owner.runtime
    actual = runtime.state

    def forbidden(*_args, **_kwargs):
        raise AssertionError('Actor route must not call a probe or command rewrite')

    for name in ('_column_dino_fg_probe', '_lewm_cem_builder', '_vertical_pinch_builder'):
        monkeypatch.setattr(service, name, forbidden)
    physical = Mock(wraps=runtime.physical_check)
    prediction = Mock(wraps=runtime.core.predict)
    monkeypatch.setattr(runtime, 'physical_check', physical)
    monkeypatch.setattr(runtime.core, 'predict', prediction)

    result = service.handle('plan')
    assert result['plan']['action_source'] == 'v21_actor'
    assert not service._uses_vertical_pinch_geometry()
    assert runtime._pending_builder is None
    assert physical.call_count >= 1
    prediction.assert_called_once()
    assert runtime.pending.response.predicted_q_path.shape == (1, 6, 6)
    assert runtime.state is actual
    assert service.owner.writes == 0
