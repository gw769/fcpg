"""Offline execution of the actual J3 banner renderer; no service or hardware."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def render_banner(status):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for dashboard state test')
    html = (Path(__file__).resolve().parents[1] / 'fcpg/v21_dashboard.html').read_text()
    source = html[html.index('function renderJ3RangeArrival('):html.index('\nfunction render(d)')]
    script = """const banner={textContent:'old',hidden:false,className:'good'};
const $=id=>{if(id!=='j3RangeArrival')throw Error(id);return banner};
const number=(value,precision)=>Number.isFinite(value)?value.toFixed(precision):'—';
""" + source + '\nrenderJ3RangeArrival(' + json.dumps(status) + ');console.log(JSON.stringify(banner));'
    result = subprocess.run([node], input=script, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def arrival(stamp, *, reached=False, residual=2.290, budget=5, **extras):
    return dict(completed=True, finished_s=stamp, command_budget=budget,
        original_target_reached=reached, original_target_arrival=dict(
            max_abs_residual=residual, position_unit='lerobot_calibrated_position',
            reason='within_position_and_step_tolerances' if reached else 'outside_position_tolerance'),
        **extras)


def test_newer_native_receipt_supersedes_old_successful_compensation():
    banner = render_banner(dict(
        j3_compensation_diagnostic=dict(previous_result=arrival(10, reached=True, residual=.082)),
        j3_range_diagnostic=dict(last_result=arrival(15, budget=10)),
        j3_native_response_diagnostic=dict(enabled=False, consumed=True,
            previous_result=arrival(20))))
    assert '上次原生响应诊断（5步）' in banner['textContent']
    assert '诊断完成' in banner['textContent'] and '原目标未到位' in banner['textContent']
    assert '2.290 校准单位' in banner['textContent']
    assert '0.082' not in banner['textContent'] and '10步' not in banner['textContent']
    assert banner['className'] == 'warn' and not banner['hidden']


@pytest.mark.parametrize('newest', ['native_response', 'range', 'compensation'])
def test_timestamp_order_applies_to_all_diagnostic_kinds(newest):
    labels = dict(native_response='原生响应诊断', range='诊断（10步）', compensation='补偿诊断')
    status = {f'j3_{kind}_diagnostic': dict(last_result=arrival(
        30 if kind == newest else 20, residual=.123 if kind == newest else 9,
        budget=10 if kind == 'range' else 5)) for kind in labels}
    banner = render_banner(status)
    assert labels[newest] in banner['textContent']
    assert '0.123 校准单位' in banner['textContent'] and '9.000' not in banner['textContent']


def test_newer_in_progress_record_does_not_borrow_previous_arrival():
    banner = render_banner(dict(j3_compensation_diagnostic=dict(
        previous_result=arrival(11, reached=True, residual=.082)),
        j3_native_response_diagnostic=dict(command_budget=5,
            last_result=dict(completed=False, started_s=12))))
    assert '原生响应诊断（5步）' in banner['textContent']
    assert '诊断进行中' in banner['textContent'] and '原目标未评估' in banner['textContent']
    assert '0.082' not in banner['textContent'] and banner['className'] == 'warn'


@pytest.mark.parametrize('stamp', [None, '999', True, -1])
def test_unknown_time_does_not_override_dated_receipt_or_claim_recency(stamp):
    banner = render_banner(dict(j3_compensation_diagnostic=dict(
        previous_result=arrival(stamp, reached=True, residual=.082)),
        j3_native_response_diagnostic=dict(last_result=arrival(20))))
    assert '原生响应诊断' in banner['textContent']
    assert '另有诊断时间未报告或无效，无法比较先后' in banner['textContent']
    assert '最新' not in banner['textContent'] and '0.082' not in banner['textContent']


def test_undated_only_record_is_explicitly_not_established_as_latest():
    banner = render_banner(dict(j3_compensation_diagnostic=dict(
        previous_result=arrival(None, reached=True, residual=.082))))
    assert '原目标到位' in banner['textContent']
    assert '记录时间未报告或无效，无法确认先后' in banner['textContent']
    assert '最新' not in banner['textContent']


def test_reversed_clock_record_cannot_override_known_recent_result():
    banner = render_banner(dict(j3_compensation_diagnostic=dict(
        last_result=arrival(999, started_s=1000, reached=True, residual=.082)),
        j3_native_response_diagnostic=dict(last_result=arrival(20))))
    assert '原生响应诊断' in banner['textContent']
    assert '无法比较先后' in banner['textContent']


@pytest.mark.parametrize('error_key', ['error', 'cleanup_error', 'stop_latch_error',
                                      'recording_error', 'final_recording_error'])
def test_failure_does_not_erase_actual_arrival_but_never_renders_success(error_key):
    banner = render_banner(dict(j3_native_response_diagnostic=dict(
        last_result=arrival(20, reached=True, residual=.05, **{error_key: 'fixture failure'}))))
    assert '诊断失败' in banner['textContent'] and '原目标到位' in banner['textContent']
    assert banner['className'] == 'bad'


def test_incomplete_saved_attempt_is_aborted_not_running_after_restart():
    banner = render_banner(dict(j3_native_response_diagnostic=dict(command_budget=5,
        previous_result=dict(completed=False, started_s=20))))
    assert '上次原生响应诊断（5步）' in banner['textContent']
    assert '诊断中止' in banner['textContent'] and banner['className'] == 'bad'


def test_no_diagnostic_hides_and_clears_previous_banner():
    banner = render_banner({})
    assert banner['hidden'] and banner['textContent'] == ''
