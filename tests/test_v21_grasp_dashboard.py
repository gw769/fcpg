"""Execute the Dashboard score formatter, not a duplicate Python formatter."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def run_js(function, value):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for dashboard formatter tests')
    html = (Path(__file__).resolve().parents[1]/'fcpg/v21_dashboard.html').read_text()
    if function == 'graspCemSummary':
        code = html[html.index('function graspCemSummary('):html.index('function renderGraspCem(')]
    else:
        code = html[html.index('function cemSourceName('):html.index('function columnTemporalSummary(')]
    script = "const number=(v,n)=>Number.isFinite(v)?v.toFixed(n):'—';\n" + code
    script += '\nconsole.log(JSON.stringify('+function+'('+json.dumps(value)+')));'
    result = subprocess.run([node], input=script, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def status():
    receipt = dict(objective='wrist_distance_alignment', available=True, cem_bypassed=False,
        selected_scores=dict(distance_px=[30.]+[25.]*40, angle_error_deg=[40.]+[35.]*40,
                             distance_cost=.112, alignment_cost=.388),
        visual_task_cost=dict(source='measured_wrist_distance_alignment',
                              source_distance_px=30., source_angle_error_deg=40.))
    return dict(model=dict(lewm_cem_objective='wrist_distance_alignment', lewm_cem_last=receipt),
                plan=dict(source_time_s=10., lewm_cem=receipt), snapshot=dict(time=10.),
                runtime=dict(pending=True))


def test_display_separates_the_two_scores_and_labels_prediction():
    result = run_js('graspCemSummary', status())
    assert result['rows'] == [['距离 / px', '30.00', '25.00', '0.1120'],
                              ['蓝白夹角 / °', '40.00', '35.00', '0.3880']]
    assert '0°与180°都算对齐' in result['note']
    assert '数值为预测' in result['note']


@pytest.mark.parametrize('case', ['no_plan', 'stale', 'stopped', 'no_clock', 'missing_angle', 'bad_angle', 'replan', 'no_fit'])
def test_stale_or_unavailable_plan_never_displays_numeric_scores(case):
    value = status()
    if case == 'no_plan': value['plan'] = None
    if case == 'stale': value['snapshot']['time'] += 1.
    if case == 'stopped': value['runtime']['pending'] = False
    if case == 'no_clock':
        value['plan'].pop('source_time_s'); value['snapshot'].pop('time')
    if case == 'missing_angle': value['plan']['lewm_cem']['selected_scores'].pop('angle_error_deg')
    if case == 'bad_angle': value['plan']['lewm_cem']['selected_scores']['angle_error_deg'][5] = 180.
    if case == 'replan': value['plan_requires_replan'] = True
    if case == 'no_fit': value['model']['lewm_cem_last'] = dict(available=False, reason='calibration required')
    assert run_js('graspCemSummary', value)['rows'] == []


def test_legacy_actor_bypass_is_not_displayed_as_cem_optimization():
    text = run_js('pipelineDescription', dict(lewm_cem=True,
        lewm_cem_objective='actor_response_tracking', lewm_cem_last=dict(cem_bypassed=True)))
    assert '绕过 CEM' in text and '未进行距离与对齐选优' in text
