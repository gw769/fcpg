"""Pure UI state tests; no browser, camera or serial connection."""
import json
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess

import pytest


def test_auxiliary_banner_explains_restored_and_incomplete_accounting():
    node=shutil.which('node')
    if node is None:pytest.skip('Node.js required for dashboard state test')
    html=_dashboard_html()
    function='function auxiliaryControlState(d){'+html.split('function auxiliaryControlState(d){',1)[1].split('\nfunction render(d)',1)[0]
    cases=[{},dict(auxiliary_trial=dict(active=False)),dict(auxiliary_trial=dict(active=True,
        block_attempts=3,maximum_blocks=10,policy_commands_returned=15,requires_home_after_restart=True)),
        dict(auxiliary_trial=dict(active=True,block_attempts=1,maximum_blocks=10,
            policy_commands_returned=0,pending_reservation=True))]
    script=function+'\nprocess.stdout.write(JSON.stringify('+json.dumps(cases)+'.map(auxiliaryControlState)));'
    result=subprocess.run([node],input=script,text=True,capture_output=True,check=True)
    values=json.loads(result.stdout)
    assert values[:2]==[dict(active=False,message='')]*2
    assert values[2]['active'] and '3/10' in values[2]['message'] and '15 条' in values[2]['message']
    assert '未恢复' in values[2]['message'] and 'HOME' in values[2]['message']
    assert '未决命令数量' in values[3]['message'] and '不可按零处理' in values[3]['message']


def _dashboard_html():
    return (Path(__file__).parents[1] / 'fcpg/v21_dashboard.html').read_text(encoding='utf-8')


def _dashboard_helpers(html):
    return ('function auxiliaryControlState(d){'
            + html.split('function auxiliaryControlState(d){', 1)[1].split('\nfunction render(d)', 1)[0])


def _render_controls(changes=None, *, busy=False):
    """Execute the actual render function against a minimal DOM, without network."""
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for dashboard state test')
    html = _dashboard_html()
    source = html[html.index('function auxiliaryControlState('):html.index('\nfunction commandStepPoints(')]
    helpers = html[html.index('const number='):html.index("const pathNS=")]
    status = dict(connected=True, phase='idle', ready=True,
                  runtime=dict(stopped=True, history_chunks=[0]),
                  capped_trial=dict(outcome='not_started', max_blocks=10))
    status.update(changes or {})
    script = """
const elements={};
const makeElement=()=>({disabled:false,hidden:false,value:'',textContent:'',className:'',
                       replaceChildren(){},append(){}});
const $=id=>elements[id]||(elements[id]=makeElement());
const document={querySelector:()=>$('no-point'),createElement:makeElement};
const views=['wrist','column'];
const CAMERA_CONDITIONS=['both_fresh','wrist_h0','wrist_cold','external_h0','external_cold'];
let latest=null,lastError='',busy=BUSY;const draw=()=>{};
""".replace('BUSY', json.dumps(busy)) + helpers + source
    script += '\nrender(' + json.dumps(status) + ');console.log(JSON.stringify(elements));'
    result = subprocess.run([node], input=script, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize('status,expected', [
    (dict(available=False, reason='completed_response_not_verified'), '上次执行的实际响应未通过检查'),
    (dict(available=False, reason='source_visual_pair_unavailable'), '缺少执行前的有效指尖观测'),
    (dict(available=True, reason=None), '历史图像＋LeWM/FK 预测'),
])
def test_column_temporal_participation_is_visible_without_claiming_fcpg_pixels(status, expected):
    controls = _render_controls(dict(fingertips=dict(column=dict(
        available=False, temporal_prior_status=status))))
    assert expected in controls['no-point']['textContent']


@pytest.mark.parametrize('matches_current,disabled', [(True, True), (False, False), (None, False)])
def test_new_failed_execution_can_compensate_despite_previous_success(matches_current, disabled):
    controls = _render_controls(dict(
        runtime=dict(stopped=True, requires_home_after_incomplete_execution=True, history_chunks=[0]),
        fixed_goal_compensation=dict(enabled=True, completed=True, configuration_restored=True,
                                     source_matches_current_execution=matches_current)))
    assert controls['fixed-goal-compensate']['disabled'] is disabled


def test_learned_policy_grid_markup_uses_planned_condition_api():
    html = _dashboard_html()
    for name in ('both_fresh', 'wrist_h0', 'wrist_cold', 'external_h0', 'external_cold'):
        assert f'value="{name}"' in html
    assert 'id="cameraCondition"' in html
    assert 'id="cameraConditionSelect"' in html
    assert "request('/api/camera-condition'" in html
    assert 'JSON.stringify({camera_condition:name})' in html
    assert 'id="alignmentProbe"' not in html
    class Buttons(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = []
        def handle_starttag(self, tag, attrs):
            if tag == 'button':
                self.ids.append(dict(attrs).get('id'))
    buttons = Buttons()
    buttons.feed(html)
    assert {'preview', 'candidate-preview', 'observe', 'plan', 'execute',
            'target-track', 'trial', 'home', 'stop', 'copy'} <= set(buttons.ids)
    assert len(buttons.ids) == len(set(buttons.ids)) and None not in buttons.ids


def test_learned_policy_grid_reads_status_and_blocks_assisted_trial():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for dashboard state test')
    html = _dashboard_html()
    names = [
        {},
        dict(camera_condition='wrist_h0'),
        dict(camera_condition=dict(name='external_cold')),
        dict(model=dict(camera_condition='wrist_cold')),
        dict(model=dict(camera_condition=dict(name='both_fresh'))),
    ]
    blocks = [
        {},
        dict(model=dict(grounding_correction=dict(enabled=True))),
        dict(model=dict(head_grasp_enabled=True)),
        dict(model=dict(alignment_motion_enabled=True)),
        dict(auxiliary_trial=dict(active=True, block_attempts=1, maximum_blocks=10,
                                 policy_commands_returned=0)),
        dict(model=dict(grounding_correction=dict(enabled=False), head_grasp_enabled=False,
                       alignment_motion_enabled=False),
             auxiliary_trial=dict(active=False)),
    ]
    script = (_dashboard_helpers(html)
              + '\nconst names=' + json.dumps(names)
              + '.map(cameraConditionName);'
              + '\nconst blocked=' + json.dumps(blocks)
              + '.map(learnedPolicyBlocked);'
              + '\nprocess.stdout.write(JSON.stringify({names,blocked}));')
    result = subprocess.run([node], input=script, text=True, capture_output=True, check=True)
    values = json.loads(result.stdout)
    assert values['names'] == ['', 'wrist_h0', 'external_cold', 'wrist_cold', 'both_fresh']
    assert values['blocked'] == [False, True, True, True, True, False]


def test_auxiliary_budget_blocks_policy_but_preserves_read_only_controls_and_stop():
    elements = _render_controls(dict(auxiliary_trial=dict(active=True,
        block_attempts=3, maximum_blocks=10, policy_commands_returned=15,
        requires_home_after_restart=True)))
    for name in ('plan', 'execute', 'trial', 'target-track', 'cameraConditionSelect'):
        assert elements[name]['disabled'], name
    for name in ('preview', 'candidate-preview', 'observe', 'home', 'stop'):
        assert not elements[name]['disabled'], name
    assert '3/10' in elements['auxiliaryProgress']['textContent']
    assert not elements['auxiliaryProgress']['hidden']


def test_direct_policy_controls_and_ros_supported_routes_are_distinct():
    direct = _render_controls()
    ros = _render_controls(dict(ros_transport={}))
    assert not direct['trial']['disabled'] and not ros['trial']['disabled']
    assert not direct['target-track']['disabled'] and ros['target-track']['disabled']
    for name in ('preview', 'candidate-preview', 'observe', 'plan', 'execute', 'home', 'stop'):
        assert not direct[name]['disabled'] and not ros[name]['disabled'], name
    assert '连续试验保持关闭' not in ros['trialProgress']['textContent']


@pytest.mark.parametrize('changes,busy', [
    ({'connected': False}, False), ({'phase': 'executing'}, False), ({}, True),
    ({'runtime': {'pending': True}}, False), ({'capped_trial': None}, False),
    ({'model': {'grounding_correction': {'enabled': True}}}, False),
    ({'model': {'head_grasp_enabled': True}}, False),
    ({'model': {'alignment_motion_enabled': True}}, False),
])
def test_trial_remains_disabled_when_runtime_or_policy_disallows_it(changes, busy):
    elements = _render_controls(changes, busy=busy)
    assert elements['trial']['disabled']
    assert elements['stop']['disabled'] is (changes.get('connected') is False)


@pytest.mark.parametrize('changes', [
    {'runtime': {'history_chunks': [1]}}, {'runtime': {'pending': True}},
    {'auxiliary_trial': {'active': True}}, {'model': {'head_grasp_enabled': True}},
])
def test_camera_conditions_cannot_change_during_history_pending_or_assisted_policy(changes):
    assert _render_controls(changes)['cameraConditionSelect']['disabled']


def test_grounding_description_reflects_inputs_used_by_each_view():
    elements = _render_controls(dict(model=dict(object_relative=[1, 0, .1, .2]),
        fingertips=dict(wrist=dict(available=True, fm_px=[[1, 2], [3, 4]]),
            column=dict(available=True, source='column-rgb-jaw-contour', fm_px=[[5, 6], [7, 8]]))))
    note = elements['no-point']['textContent']
    assert 'Wrist 机械参考（FK' in note and 'Column 图像轮廓端点' in note
    assert 'Wrist 参考参与 Actor 的目标相对量' in note
    assert 'Column 使用独立图像识别' in note


def test_grounding_description_keeps_prediction_suggestion_distinct_from_observation():
    elements = _render_controls(dict(fingertips=dict(
        column=dict(available=False, observed=False, source='column-rgb-jaw-contour',
                    reason='no supported current pair', fm_px=None,
                    prediction_suggestion=dict(available=False, observed=False,
                        suggested_fm_px=[[90, 143], [124, 143]])))))
    note = elements['no-point']['textContent']
    assert '预测建议 PF (90.00  143.00) / PG (124.00  143.00)' in note
    assert '未观测，不写入视觉记忆' in note


def test_candidate_preview_is_explicitly_separate_from_live_cem():
    html = _dashboard_html()
    assert '独立 Actor 候选诊断 · 不执行' in html
    assert '独立只读候选诊断 · 不参与 CEM 计划' in html
    assert '不是实际 v21_lewm_cem 搜索池、赢家或动作计划' in html
    assert '诊断内建议：' in html
    assert '内部选择建议：' not in html


@pytest.mark.parametrize(('condition', 'scale'), [
    ('both_fresh', 3.0),
    ('wrist_h0', 3.0),
    ('both_fresh', 1.5),
])
def test_dashboard_reads_cem_scale_independently_from_camera_condition(condition, scale):
    elements = _render_controls(dict(camera_condition=condition, model=dict(
        lewm_cem=True, lewm_cem_action_scale=scale,
        lewm_cem_objective='actor_response_tracking',
        lewm_cem_candidate_parameterization='amplified_actor_sequence')))
    note = elements['cemPlanStatus']['textContent']
    assert f'动作倍率 {scale:.1f}x' in note
    assert f'相机条件 {condition} 只决定视图交付' in note
    assert 'actor_response_tracking' in note
    assert 'amplified_actor_sequence' in note
