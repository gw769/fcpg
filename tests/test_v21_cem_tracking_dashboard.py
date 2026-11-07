"""CEM reference/forecast comparison uses one record and actual q0."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def compare(entry, *, execution=False, joint=2):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js required for dashboard tests')
    html = (Path(__file__).resolve().parents[1] / 'fcpg/v21_dashboard.html').read_text()
    code = html[html.index('function cemTrackingComparison('):html.index('function drawHistory(')]
    script = "const first=x=>x?.[0],number=(x,n)=>Number.isFinite(x)?x.toFixed(n):'—';\n"
    script += code + '\nconsole.log(JSON.stringify(cemTrackingComparison('
    script += json.dumps(entry) + ',' + json.dumps(execution) + ',' + str(joint) + ')));'
    result = subprocess.run([node], input=script, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def record():
    q = [0., -97., 84., 63., -54., 30.]
    def row(delta):
        return [*q[:2], q[2]+delta, *q[3:]]
    return dict(action_source='v21_lewm_cem', source_q_position=[q],
        commands_position=[[row(-3.) for _ in range(5)]],
        predicted_q_path=[[q]+[row(-.38) for _ in range(5)]],
        lewm_cem=dict(objective='actor_response_tracking',
            reference_commands_position=[row(-2.49) for _ in range(5)]))


def test_weak_response_is_not_confused_with_command_or_reference():
    value = compare(record())
    assert value['referenceDelta'] == pytest.approx(-2.49)
    assert value['commandDelta'] == pytest.approx(-3.)
    assert value['predictedDelta'] == pytest.approx(-.38)
    assert value['residual'] == pytest.approx(2.11)
    assert '预测末点 − 参考 2.110' in value['text']
    assert '到位需实际读回验证' in value['text']


@pytest.mark.parametrize('damage', ['source', 'short', 'unavailable', 'foreign', 'objective'])
def test_missing_or_unpaired_data_never_gets_a_numeric_comparison(damage):
    entry = record()
    if damage == 'source':
        entry['predicted_q_path'][0][0] = [1., -97., 84., 63., -54., 30.]
    elif damage == 'short':
        entry['predicted_q_path'][0].pop()
    elif damage == 'unavailable':
        entry['lewm_cem'].pop('reference_commands_position')
    elif damage == 'foreign':
        entry['action_source'] = 'v21_actor'
    else:
        entry['lewm_cem']['objective'] = 'column_visual'
    assert compare(entry) is None


def test_execution_uses_its_retimed_forecast_not_the_planning_prediction():
    entry = record()
    for name in ('source_q_position', 'commands_position'):
        entry[name] = entry[name][0]
    entry['retimed_predicted_q_path'] = entry['predicted_q_path'][0]
    entry['retimed_predicted_q_path'][-1][2] = 83.3
    entry['predicted_q_path'] = None
    value = compare(entry, execution=True)
    assert value['predictedDelta'] == pytest.approx(-.7)
    assert value['residual'] == pytest.approx(1.79)


def test_stationary_axis_shows_zero_without_dividing_by_reference():
    value = compare(record(), joint=0)
    assert value['referenceDelta'] == value['commandDelta'] == value['predictedDelta'] == 0.
    assert value['residual'] == 0.
