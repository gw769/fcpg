"""Recovery markers and rejected readback telemetry are hardware-free."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.canary_so101_v21_research import TrialOwner
from test_v21_live_service import service  # noqa: F401


def test_restart_restores_partial_execution_latch(service):
    path = service.output/'incomplete-execution.json'
    path.write_text(json.dumps(dict(requires_home=True)))
    service._restore_incomplete_execution()
    assert service.owner.runtime.requires_home_after_incomplete_execution
    with pytest.raises(RuntimeError, match='HOME required'):
        service.handle('plan')
    assert service.owner.writes == 0


def test_malformed_recovery_marker_cannot_authorize_restart(service):
    (service.output/'incomplete-execution.json').write_text('{"requires_home": "false"}')
    with pytest.raises(ValueError, match='recovery record'):
        service._restore_incomplete_execution()


def test_rejected_actual_q_is_retained_as_failed_diagnostic_only():
    owner = TrialOwner.__new__(TrialOwner)
    q = np.arange(6, dtype=np.float32)
    owner._read_q = lambda: q.copy()
    def reject(*args, **kwargs):
        raise RuntimeError('actual table clearance')
    owner.clearance = SimpleNamespace(require_readback=reject)
    with pytest.raises(RuntimeError, match='actual table clearance'):
        owner.read_checked()
    record = owner.last_readback_diagnostic
    assert record['q_position'] == q.tolist()
    assert record['time_s'] > 0 and record['clearance_validated'] is False
    assert 'actual table clearance' in record['error']
