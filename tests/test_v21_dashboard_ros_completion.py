"""Exercise the actual dashboard client against out-of-order ROS snapshots."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


HTML = Path(__file__).resolve().parents[1] / 'fcpg/v21_dashboard.html'


def run_client(body):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for dashboard client test')
    html = HTML.read_text()
    client = html[html.index('function serviceStatus('):html.index('function auxiliaryControlState(')]
    script = """const assert = require('node:assert/strict');
const setTimeout = callback => callback();
const accepted = {request_id:'preview-1', poll:'/api/request/preview-1'};
const done = {request_id:'preview-1', state:'succeeded', completed_monotonic_s:12,
              result:{preview:{id:'new'}, runtime:{stopped:true}}};
function envelope(time, records, preview='new', active=null) {
 return {schema:'so101-ros-control-status-v1', published_monotonic_s:time,
         active_request_id:active, requests:records,
         service:{preview:{id:preview}, runtime:{stopped:true}}, transport:{}};
}
let responses=[], paths=[];
async function request(path) {paths.push(path); assert.ok(responses.length, 'unexpected request'); return responses.shift();}
"""
    result = subprocess.run([node], input=script + client + '\n(async()=>{' + body +
                            "\nconsole.log(JSON.stringify({ok:true}));})().catch(e=>{console.error(e);process.exitCode=1});",
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['ok']


def test_completion_waits_for_corresponding_status_instead_of_old_preview():
    run_client("""
responses=[done, envelope(11, {}, 'old', 'preview-1'),
           envelope(12.1, {'preview-1':done})];
const value=await waitForControl(accepted, 1000);
assert.equal(value.preview.id, 'new');
assert.deepEqual(paths, ['/api/request/preview-1','/api/status','/api/status']);
assert.equal(responses.length, 0);
""")


def test_idle_status_from_before_request_is_not_completion():
    run_client("""
responses=[done, envelope(11, {}, 'old'), envelope(12.1, {'preview-1':done})];
await waitForControl(accepted, 1000);
assert.equal(paths.filter(x=>x==='/api/status').length, 2);
""")


def test_late_poll_response_cannot_replace_newer_ros_status():
    run_client("""
responses=[envelope(13, {'preview-1':done}), envelope(11, {}, 'old')];
assert.equal((await currentStatus()).preview.id, 'new');
assert.equal((await currentStatus()).preview.id, 'new');
""")


def test_failed_request_does_not_fetch_or_claim_completed_status():
    run_client("""
responses=[{...done,state:'failed',error:'planner failed'}];
await assert.rejects(waitForControl(accepted, 1000), /planner failed/);
assert.deepEqual(paths, ['/api/request/preview-1']);
""")


def test_direct_service_response_is_preserved():
    run_client("""
assert.deepEqual(await waitForControl(done.result, 1000), done.result);
assert.deepEqual(paths, []);
""")


@pytest.mark.parametrize('action', ['preview', 'camera-condition'])
@pytest.mark.parametrize('stop_first', [True, False])
def test_stop_owns_ui_until_its_completion_despite_earlier_request(action, stop_first):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for dashboard client test')
    html = HTML.read_text()
    operations = html[html.index('async function operate('):
                      html.index("['preview','candidate-preview','observe','plan','execute'")]
    setup = """
const assert=require('node:assert/strict');
let busy=false,controlUiGeneration=0,latest={runtime:{stopped:true}},frameGeneration=0;
const elements={cameraConditionSelect:{value:'both_fresh'},recordSource:{value:'execution'},source:{value:'endpoint'}};
const $=id=>elements[id]??(elements[id]={}), text=(id,value)=>{$(id).textContent=value};
const CAMERA_CONDITIONS=['both_fresh'];
const cameraConditionName=()=> 'both_fresh';
const render=d=>{latest=d},showError=()=>{},draw=()=>{},confirm=()=>true;
const waiters={};
async function request(path){const action=path.split('/').pop();return {request_id:action,poll:action};}
async function waitForControl(accepted){return new Promise(resolve=>{waiters[accepted.request_id]=resolve});}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
"""
    body = """
const first=ACTION==='preview'?operate('preview'):setCameraCondition();await tick();
const stop=operate('stop');await tick();
const completed={runtime:{stopped:true}};
if(STOP_FIRST){
 waiters.stop(completed);await stop;
 assert.equal($('operation').textContent,'stop 完成');
 waiters[ACTION](completed);await first;
}else{
 waiters[ACTION](completed);await first;
 assert.equal(busy,true,'earlier request must not unlock pending STOP');
 assert.ok($('operation').textContent.includes('STOP'));
 waiters.stop(completed);await stop;
}
assert.equal(busy,false);
assert.equal($('operation').textContent,'stop 完成');
assert.equal($('recordSource').value,'execution');
""".replace('ACTION', json.dumps(action)).replace('STOP_FIRST', json.dumps(stop_first))
    result = subprocess.run([node], input=setup + operations + '\n(async()=>{' + body +
                            '})().catch(e=>{console.error(e);process.exitCode=1});',
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('automatic_stop_failed', [False, True])
@pytest.mark.parametrize('manual_stop_first', [False, True])
def test_timed_out_operation_cannot_reclaim_ui_after_manual_stop(automatic_stop_failed, manual_stop_first):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for dashboard client test')
    html = HTML.read_text()
    operations = html[html.index('async function operate('):
                      html.index("['preview','candidate-preview','observe','plan','execute'")]
    setup = """
const assert=require('node:assert/strict');
let busy=false,controlUiGeneration=0,latest=null,frameGeneration=0,lastError='';
const elements={recordSource:{value:'execution'}};
const $=id=>elements[id]??(elements[id]={}), text=(id,value)=>{$(id).textContent=value};
const CAMERA_CONDITIONS=['both_fresh'],cameraConditionName=()=> 'both_fresh';
const render=d=>{latest=d},showError=error=>{lastError=error},draw=()=>{},confirm=()=>true;
let stopCount=0,acceptAutomaticStop;const waiters={};
async function request(path){
 if(path==='/api/preview'){
  const error=new Error('preview timed out');error.name='TimeoutError';throw error;
 }
 assert.equal(path,'/api/stop');
 if(++stopCount===1)return new Promise(resolve=>{acceptAutomaticStop=resolve});
 return {request_id:'manual-stop'};
}
async function waitForControl(accepted){
 return new Promise((resolve,reject)=>{waiters[accepted.request_id]={resolve,reject}});
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
"""
    body = """
const earlier=operate('preview');await tick();
assert.ok(acceptAutomaticStop,'timeout recovery must have started');
const manual=operate('stop');await tick();
acceptAutomaticStop({request_id:'automatic-stop'});await tick();
const completed={runtime:{stopped:true}};
if(MANUAL_FIRST){waiters['manual-stop'].resolve(completed);await manual;}
if(AUTOMATIC_FAILED)waiters['automatic-stop'].reject(new Error('automatic STOP failed'));
else waiters['automatic-stop'].resolve(completed);
await earlier;
assert.equal(lastError,'','superseded timeout/automatic STOP error must not overwrite manual STOP');
if(!MANUAL_FIRST){
 assert.equal(busy,true,'manual STOP is still waiting for its own receipt');
 assert.ok($('operation').textContent.includes('STOP'));
 waiters['manual-stop'].resolve(completed);await manual;
}
assert.equal(busy,false);
assert.equal($('operation').textContent,'stop 完成');
assert.equal($('recordSource').value,'execution');
assert.equal(stopCount,2);
""".replace('MANUAL_FIRST', json.dumps(manual_stop_first)).replace(
        'AUTOMATIC_FAILED', json.dumps(automatic_stop_failed))
    result = subprocess.run([node], input=setup + operations + '\n(async()=>{' + body +
                            '})().catch(e=>{console.error(e);process.exitCode=1});',
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
