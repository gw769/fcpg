"""Actual dashboard functions with fake responses; no HTTP or hardware access."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from test_v21_live_service import service  # noqa: F401


HTML = Path(__file__).resolve().parents[1]/'fcpg/v21_dashboard.html'


def run_js(setup, code, body):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for dashboard tests')
    script = setup + '\n' + code + '\n(async()=>{' + body + (
        '\n})().catch(error=>{console.error(error);process.exitCode=1});')
    result = subprocess.run([node], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def summary_source(html):
    return html[html.index('function executionArrival('):html.index('function drawHistory(')]


def entry(reached):
    value = dict(completed=True, source_time_s=10., commands_position=[[0.]*6]*5,
                 command_times_s=[10.+i*.1 for i in range(5)],
                 measured_q_path=[[0.]*6]*6, readback_times_s=[10.+i*.1 for i in range(6)])
    if reached is not None:
        value.update(original_target_reached=reached,
            original_target_arrival=dict(reached=reached,
                max_abs_residual=.12 if reached else 2.62133,
                reason='within_position_and_step_tolerances' if reached else 'outside_position_tolerance',
                position_unit='lerobot_calibrated_position'))
    return value


@pytest.mark.parametrize('reached,label,css',
    [(True,'原目标到位','good'),(False,'原目标未到位','bad'),(None,'原目标未验证','warn')])
def test_execute_main_banner_distinguishes_arrival_without_changing_transaction(reached,label,css):
    html = HTML.read_text()
    operations = html[html.index('async function operate('):
                      html.index("['preview','candidate-preview','observe','plan','execute'")]
    setup = """
let busy=false,controlUiGeneration=0,latest=null,frameGeneration=0;
const elements={recordSource:{value:'history'},source:{value:'endpoint'},operation:{className:'good'}};
const $=id=>elements[id]??(elements[id]={}),text=(id,value)=>{$(id).textContent=value};
const number=(value,n)=>Number.isFinite(value)?value.toFixed(n):'—';
const render=value=>{latest=value},showError=()=>{},draw=()=>{},confirm=()=>true;
async function request(){return response;}async function waitForControl(value){return value;}
""" + '\nconst response=' + json.dumps(dict(last_execution=entry(reached), runtime=dict(stopped=True))) + ';'
    result = run_js(setup, summary_source(html)+operations,
        "const before=JSON.stringify(response);await operate('execute');"
        "console.log(JSON.stringify({banner:elements.operation,source:elements.recordSource.value,"
        "unchanged:before===JSON.stringify(response),completed:latest.last_execution.completed}));")
    assert label in result['banner']['textContent']
    assert result['banner']['className'] == css
    assert '五步事务记录完整' in result['banner']['textContent']
    assert '已 STOP' in result['banner']['textContent']
    if reached is False:
        assert '2.621 校准单位' in result['banner']['textContent']
    assert result['completed'] and result['unchanged'] and result['source']=='execution'


@pytest.mark.parametrize('damage',['missing_residual','negative_residual','contradiction',
                                  'incomplete','legacy_hold','insufficient_samples','invalid_flag'])
def test_incomplete_or_inconsistent_evidence_cannot_be_green(damage):
    value = entry(True)
    if damage=='missing_residual':value['original_target_arrival'].pop('max_abs_residual')
    elif damage=='negative_residual':value['original_target_arrival']['max_abs_residual']=-1
    elif damage=='contradiction':value['original_target_reached']=False
    elif damage=='incomplete':value['completed']=False
    elif damage=='legacy_hold':
        value=entry(None);value['normal_follow']=dict(goal_reached=True,hold_eligible=True)
    elif damage=='insufficient_samples':value['original_target_arrival']['reason']='insufficient_samples'
    else:value['original_target_reached']='true'
    result = run_js('const number=(value,n)=>Number.isFinite(value)?value.toFixed(n):"—";',
                    summary_source(HTML.read_text()),
                    'console.log(JSON.stringify(executionArrival('+json.dumps(value)+')));')
    assert result['reached'] is None and result['css']=='warn'
    assert result['label']=='原目标未验证'


def test_each_history_row_displays_its_own_arrival_and_residual():
    html = HTML.read_text()
    code = html[html.index('function executionArrival('):html.index('function draw(){')]
    setup = """
class Element {
 constructor(){this.children=[];this.textContent='';this.className='';}
 append(...values){this.children.push(...values);}replaceChildren(...values){this.children=values;}
 setAttribute(){}closest(){return {querySelectorAll:()=>headers};}
}
const headers=Array.from({length:6},()=>new Element());
const elements={traceBody:new Element(),chart:new Element(),traceLabel:new Element(),joint:{value:'2'}};
const $=id=>elements[id],text=(id,value)=>{$(id).textContent=value};
const document={createElement:()=>new Element(),createElementNS:()=>new Element()};
const number=(value,n)=>Number.isFinite(value)?value.toFixed(n):'—';
const commandStepPoints=(commands,times,j)=>commands.map((q,i)=>[times[i],q[j]]);
""" + '\nconst latest=' + json.dumps(dict(execution_history=[entry(True),entry(False),entry(None)])) + ';'
    result = run_js(setup, code,
        "const before=JSON.stringify(latest);drawHistory();console.log(JSON.stringify({"
        "rows:elements.traceBody.children.map(row=>row.children[0]),"
        "label:elements.traceLabel.textContent,unchanged:before===JSON.stringify(latest)}));")
    for row,label,css in zip(result['rows'],['原目标到位','原目标未到位','原目标未验证'],
                             ['good','bad','warn']):
        assert label in row['textContent'] and row['className']==css
    assert '2.621 校准单位' in result['rows'][1]['textContent']
    assert '五步事务记录；目标是否到位见各块标记' in result['label']
    assert result['unchanged']


def test_saved_history_keeps_arrival_evidence_and_original_five_readbacks(service):
    result = entry(False)
    result['normal_follow'] = dict(goal_reached=False, hold_eligible=True,
        goal_reached_scope='original_command_position_arrival',
        original_target_arrival=deepcopy(result['original_target_arrival']),
        positions=[[0.,0.,2.6,0.,0.,0.]]*3,times_s=[10.55,10.6,10.65])
    original_path=deepcopy(result['measured_q_path'])
    original_times=deepcopy(result['readback_times_s'])
    service._save_execution(result)
    historical=service.status()['execution_history'][-1]
    assert historical['completed'] is True and historical['original_target_reached'] is False
    for field in ('original_target_arrival','normal_follow'):
        assert historical[field]==result[field]
    assert historical['measured_q_path']==original_path
    assert historical['readback_times_s']==original_times
    assert len(historical['commands_position'])==5 and len(historical['measured_q_path'])==6
    result['normal_follow']['positions'][0][2]=99
    assert historical['normal_follow']['positions'][0][2]==2.6
    assert service.owner.writes==0
