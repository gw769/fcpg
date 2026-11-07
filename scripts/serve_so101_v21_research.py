#!/usr/bin/env python3
"""Resident V21.0.2 dashboard. Bind loopback BEFORE acquiring any hardware.

Enabled engineering assistance is a biased task-specific path, not a pure
learned-policy result. See docs/ASSISTED_PATH_BIAS.md before interpreting trials.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import sys
import threading
from urllib.parse import urlsplit, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcpg.v21_camera_condition import ADDRESS_MODES, CONDITIONS
from fcpg.v21_core import VERSION, CONTRACT
from fcpg.v21_live_service import V21LiveService, TASK, WRIST_REFERENCE

HTML = Path(__file__).resolve().parents[1]/'fcpg'/'v21_dashboard.html'
DEFAULT_ROOT = Path('/home/james/so101-vda-disk/feedback-corrected-grounding/outputs/lerobot/so101/prod_v21_readback_visual_memory_v2')
C4_DEFAULTS = dict(   # the wired-back C4-all and its goal library (docs/V21_LEWM_C4_CONTROL_PLAN_20261011.md)
    checkpoint=Path('/home/james/so101-c4-step2/models/C4-all/final.pt'),
    checkpoint_sha256='39ac3d20ff885fcb442019515f8b2c9b928d289c1aee68f152f829aa0deb2743',
    library=Path('/home/james/so101-c4-step2/models/C4-all/goal-library/goal-library.npz'),
    library_sha256='b84d1e8196444a3c87daa56a11f79136c9a206a47aa1bfd39f47fe2e1e5c3ac5',
    dino_repo=Path('/home/james/.cache/torch/hub/facebookresearch_dinov2_main'),
    dino_weights=Path('/home/james/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'))


def startup_failure_payload(error):
    """Return a JSON-safe, non-controlling record for failed hardware startup."""
    value = dict(
        schema='v21-startup-failure-v1',
        error_type=type(error).__name__,
        error=f'hardware startup failed: {type(error).__name__}: {error}',
        motion_authorized=False,
    )
    diagnostic = getattr(error, 'camera_startup_diagnostic', None)
    value['camera'] = diagnostic if isinstance(diagnostic, dict) else None
    return value


def make_server(address, app):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, value, kind='application/json'):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if kind == 'application/json' else value
            self.send_response(code); self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data))); self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff'); self.end_headers()
            try: self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError): pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == '/': return self.reply(200, HTML.read_bytes(), 'text/html; charset=utf-8')
            service = app.get('service')
            if service is None:
                payload = dict(error=app.get('startup_error','hardware controller not enabled or still loading'),
                    model=dict(version=VERSION,contract=CONTRACT),connected=False,ready=False,phase='loading')
                if app.get('startup_diagnostic') is not None:
                    payload['startup_diagnostic'] = app['startup_diagnostic']
                return self.reply(503, payload)
            if parsed.path == '/api/status': return self.reply(200, service.status())
            if parsed.path in ('/api/frame/wrist.jpg','/api/frame/column.jpg'):
                query = parse_qs(parsed.query); view = parsed.path.split('/')[-1][:-4]
                try:
                    jpeg = service.frame(view, query.get('source',['live'])[0], query.get('sequence',[None])[0])
                    return self.reply(200, jpeg, 'image/jpeg')
                except Exception as exc: return self.reply(409, dict(error=str(exc)))
            return self.reply(404, dict(error='not found'))

        def do_POST(self):
            # JSON endpoints are local operator actions, not cross-site form targets.
            origin = self.headers.get('Origin')
            allowed = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
            if origin is not None and origin not in allowed:
                return self.reply(403, dict(error='cross-origin control rejected'))
            path = urlsplit(self.path).path
            if path not in ('/api/observe','/api/preview','/api/candidate-preview','/api/plan','/api/execute','/api/target-track','/api/carry-visual-probe','/api/home','/api/stop','/api/trial',
                            '/api/camera-condition','/api/motion-validation-probe','/api/motion-validation-return',
                            '/api/alignment-observe','/api/alignment-probe','/api/alignment-return',
                            '/api/alignment-fit','/api/alignment-step','/api/alignment-reset','/api/alignment-departure',
                            '/api/gripper-follow-check','/api/alignment-view-step','/api/alignment-reach','/api/auxiliary-budget','/api/actuator-observe',
                            '/api/grounding-calibrate','/api/column-dino-fg-probe','/api/fixed-goal-compensate'):
                return self.reply(404, dict(error='not found'))
            if app.get('service') is None: return self.reply(423, dict(error='hardware controller unavailable'))
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 <= length <= 4096: raise ValueError('invalid request size')
                payload = json.loads(self.rfile.read(length) or b'{}')
                if not isinstance(payload,dict):raise ValueError('operation requires a JSON object')
                if path not in ('/api/alignment-probe','/api/motion-validation-probe','/api/carry-visual-probe','/api/auxiliary-budget','/api/home','/api/camera-condition','/api/column-dino-fg-probe') and payload:
                    raise ValueError('operation accepts an empty object only')
                return self.reply(200, app['service'].handle(path.rsplit('/',1)[-1],payload))
            except Exception as exc: return self.reply(409, dict(error=f'{type(exc).__name__}: {exc}'))

        def log_message(self, *_args): pass
    server = ThreadingHTTPServer(address, Handler); server.daemon_threads = True
    return server


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', choices=('127.0.0.1','localhost'), default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--model-output', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--output', type=Path, default=DEFAULT_ROOT/'live')
    p.add_argument('--task', default=TASK)
    p.add_argument('--enable-hardware', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--trial-max-blocks', type=int, choices=range(1,33), default=16,
                   help='maximum policy blocks in one capped trial')
    p.add_argument('--column-memory',type=Path)
    p.add_argument('--column-scene-unchanged',action='store_true')
    p.add_argument('--wrist-only',action='store_true',help='start without current Column; do not fabricate history')
    p.add_argument('--enable-gripper-follow-check',action='store_true',help='one manual empty-jaw diagnostic at HOME; never automatic')
    p.add_argument('--normal-follow-budget-s',type=float,choices=(0.,.4,2.),default=0.,
                   help='optional bounded following after original q5; STOP remains immediate')
    p.add_argument('--record-policy-inputs', action='store_true',
                   help='also record successful blocks; complete rejected CEM responses always retain actual input tensors')
    p.add_argument('--lewm-shadow-checkpoint',type=Path,action='append',default=[],
                   help='strict LeWM candidate for read-only paired prediction evaluation')
    p.add_argument('--lewm-shadow-parent-sha256',
                   help='required original checkpoint SHA256 for all shadow candidates')
    p.add_argument('--lewm-control-checkpoint', type=Path,
                   help='explicit experimental LeWM controller candidate; requires both pinned SHA256 values')
    p.add_argument('--lewm-control-parent-sha256')
    p.add_argument('--lewm-control-candidate-sha256')
    p.add_argument('--enable-visual-alignment',action='store_true',help='enable explicit small alignment probes; no automatic motion')
    p.add_argument('--enable-column-dino-fg-calibration',action='store_true',
                   help='enable one confirmed, open-jaw Column DINO F/G response probe at a time; no automatic motion')
    p.add_argument('--alignment-scope',choices=('full','lateral'),default='full',
                   help='lateral only identifies J1 image response; no automatic grasp or carry')
    p.add_argument('--resume-auxiliary-state',type=Path,
                   help='explicitly restore a verified completed actual state and its unchanged budget; no motion resumes automatically')
    p.add_argument('--wrist-reference',type=Path,default=WRIST_REFERENCE)
    p.add_argument('--grounding-correction',action='store_true',help='Actor plans corrected by wrist object-vs-jaw grounding (J1 recentre, close veto); own action source')
    p.add_argument('--head-grasp',action='store_true',help='explicit engineering head-target/contact-reference and vertical-approach mode; Actor weights/input unchanged')
    p.add_argument('--grasp-target',choices=('body','head'),default='body',
                   help='assisted target region; head requires a fresh registered head candidate, without enabling the experimental vertical solver')
    p.add_argument('--grounding-px-per-unit',type=float,default=None,help='signed wrist dx px per J1 unit; omit and use /api/grounding-calibrate')
    p.add_argument('--gripper-force',choices=('lerobot','firm','strong','grip'),default='lerobot',
                   help='gripper motor only: raise P/torque/current/overload registers after connect; arm joints unchanged')
    p.add_argument('--camera-condition',choices=CONDITIONS,default='both_fresh',
                   help='model-delivered views for a trial: both_fresh, wrist/external H0-only, or cold')
    p.add_argument('--camera-address',choices=ADDRESS_MODES,default='factual',
                   help='factual keeps motion addresses (P2-1 scored grid). zero is a later capability, not that grid')
    p.add_argument('--raw-actor', action='store_true',
                   help='explicit comparison: no execution-geometry command rewrite; mechanical admission unchanged')
    p.add_argument('--lewm-cem', action=argparse.BooleanOptionalAction, default=True,
                   help='default on: Actor inits CEM 32/8/3 inside the physical gate; --no-lewm-cem restores geometry')
    p.add_argument('--lewm-cem-objective', choices=('column_visual','actor_response_tracking','actor_command_tracking',
                                                     'wrist_distance_alignment'),
                   default='column_visual', help='explicit CEM cost: visual transport or predicted response tracking of the fixed Actor reference')
    p.add_argument('--wrist-grasp-calibration',type=Path,
                   help='actual completed Wrist distance/blue-white response pairs for wrist_distance_alignment CEM')
    p.add_argument('--lewm-cem-candidate-parameterization',
                   choices=('bounded_endpoint_then_hold','five_command_sequence','scaled_actor_sequence',
                            'scaled_actor_endpoint_hold','amplified_actor_sequence','smooth_actor_endpoint'), default=None,
                   help='explicit CEM candidate shape; omit to retain the objective default')
    p.add_argument('--lewm-cem-action-scale', type=float, choices=(1.0, 1.5, 3.0), default=1.0,
                   help='amplify Actor deltas on all six axes; nondefault requires response tracking with amplified_actor_sequence')
    p.add_argument('--temporal-profile', choices=('actual-readback-progress-compatibility-v1',),
                   help='explicit audited Actor/FCPG readback-progress compatibility; actual clocks stay unchanged')
    # docs/V21_LEWM_C4_CONTROL_PLAN_20261011.md: shadow = phase A (Actor executes, C4 recorded),
    # control = phase B (C4's selection executes). Needs the native40 Actor; turns the old LeWM CEM off.
    p.add_argument('--c4-planner', choices=('shadow', 'control'))
    # fcpg/v21_alignment_cem.py: the paper's CEM on the system LeWM (Actor proposes, LeWM forecasts every
    # candidate, ranking by jaw-to-fish alignment at the forecast prefix state). shadow = recorded only.
    p.add_argument('--alignment-cem', choices=('shadow', 'control'))
    p.add_argument('--alignment-response', type=Path,
                   help='measured wrist response from scripts/fit_v21_wrist_response.py (needed by --alignment-cem)')
    p.add_argument('--alignment-close-tolerance-px', type=float, default=15.,
                   help='a close is admitted only when the forecast jaw-to-fish offset is within this')
    p.add_argument('--alignment-fp32', action='store_true', help='forecast in FP32 instead of BF16 autocast')
    p.add_argument('--alignment-dynamics', choices=('lewm', 'commands'), default='lewm',
                   help='what forecasts the arm for every candidate: the system LeWM, or (an ablation of the LeWM) the '
                        'arm reaching every command exactly')
    p.add_argument('--alignment-steer-approach', action='store_true',
                   help='training overhaul doc section 17.101: with the jaws open the CEM also searches the blocks '
                        'where the Actor itself travels (default: those pass unchanged, fcpg/v21_alignment_cem.TRAVEL)')
    p.add_argument('--alignment-roll', action='store_true',
                   help='11/03: with the jaws open and aimed at the fish, the CEM also turns the wrist roll so the '
                        'jaws close across it: the fish\'s axis from the column picture and the jaw line by forward '
                        'kinematics, both on the table (fcpg/v21_alignment_cem.py align_roll; needs the geometric '
                        'response and --alignment-box-solid)')
    p.add_argument('--alignment-no-box', action='store_true',
                   help='training overhaul doc section 17.116: with the jaws shut the CEM does not align the box (the '
                        'carry is the Actor\'s; the box is still found, for the grip latch and released_over_box)')
    p.add_argument('--alignment-box-height-mm', type=float, default=None,
                   help='training overhaul doc section 17.116: the box is aligned on the plane of its rim, this high '
                        'above the table (default: on the table plane, as the fish)')
    p.add_argument('--alignment-box-solid', type=Path, default=None, metavar='COLUMN_CAMERA',
                   help='training overhaul doc section 17.118: measure the box as a solid from the column picture with '
                        'this column self-calibration (v21-column-self-calibration-v1) and aim the shut jaws over its '
                        'centre (needs --box-finder-model; replaces --alignment-box-height-mm)')
    p.add_argument('--grip-latch', action='store_true',
                   help='the gripper rule of training overhaul doc section 17.45 (fcpg/v21_alignment_cem.GripLatch; '
                        'James 10/20 "j6可以hack"): held SQUEEZE below the read-back from the grasp until the Actor '
                        'opens with the box in view; executed by --alignment-cem control, recorded by shadow')
    p.add_argument('--wrist-drop-zone', type=Path,
                   help="the drop zone's wrist-view registration (scripts/register_v21_wrist_drop_zone.py, fcpg/"
                        "v21_target_tracker.py); default: its column registration's role prototype in the wrist")
    p.add_argument('--place-goal-column-camera', type=Path,
                   help='a model that reads the place goal (training overhaul doc section 17.7): the robot-calibrated '
                        'column camera that locates the box (fcpg/v21_place_goal.LivePlaceGoal)')
    # training overhaul doc section 17.32: the live object finder. The operator colour model is gone from the code
    # (10/31, James 「不要的废弃代码 这种颜色函数删掉」): learned or SAM 3 (served by scripts/serve_v21_sam3_finder.py on
    # the A100), named on the command line; a model that reads the wrist object cue does not start without one
    # (fcpg/v21_live_service.V21LiveService)
    p.add_argument('--object-finder', choices=('sam3', 'learned'),
                   help="learned: the learned finder on the exposure's own DINO tokens (section 17.34, no SAM; needs "
                        "--object-finder-model); sam3: SAM 3 named by the task's object word, anchored on sure "
                        'detections and followed by its video tracker (wrist token, CEM, column fish)')
    p.add_argument('--object-finder-model', type=Path,
                   help='the learned finder (scripts/train_v21_object_finder.py final.pt; its fish reference inside)')
    p.add_argument('--column-finder-model', type=Path,
                   help='section 17.51, with --object-finder learned: the learned column finder (final.pt trained '
                        'with --column-pairs) that gives the column fish of the grasp target (models that read the '
                        'place goal)')
    p.add_argument('--box-finder-model', type=Path,
                   help='section 17.57, with --object-finder learned: the learned box finder (final.pt trained with '
                        '--box-pairs) gives the box in the wrist and the column in place of the frozen-DINOv2 tracker '
                        '(not with --wrist-drop-zone); a V21_DINOV2=off service needs it')
    # James 10/28 「改代码 不用sam3. 任务指令要改阿」: the task text names the thing (fcpg/v21_text_finder.py)
    p.add_argument('--text-finder', action='store_true',
                   help='with --object-finder learned: the wrist and column finders take their reference from the task '
                        "text's object phrase through DINOv3's text alignment (dino.txt), not the fish reference")
    p.add_argument('--text-finder-phrase', default=None,
                   help="default: the task text's object phrase, colour words kept (fcpg/v21_text_finder.object_phrase)")
    p.add_argument('--text-finder-weights', type=Path, default=None,
                   help="Meta's dinov3_vitl16_dinotxt vision head and text encoder file (default in fcpg/v21_text_finder)")
    p.add_argument('--sam3-finder-socket', type=Path,
                   default=Path('~/so101-c4-step2/sam3-finder-20261019/finder.sock').expanduser())
    p.add_argument('--sam3-object-word', default='fish', help="one of the task text's object words")
    p.add_argument('--sam3-anchor-score', type=float, default=.5)
    p.add_argument('--sam3-hold-reference', type=Path,
                   default=Path('~/so101-c4-step2/outline-actor-20261019/fish-cues.npz').expanduser(),
                   help="the demonstrations' fish cue report (analysis/v21/demo_fish_cues.py), for models that learned "
                        'the outline cue (section 17.30); only demonstrations 0 and 1 are used')
    p.add_argument('--home-state', type=Path,
                   help='HOME pose file (current_state, six joints); default: the legacy V17.4 fish-task HOME')
    p.add_argument('--table-clearance-floor-mm', type=float, default=None,
                   help='table gate floor for planned paths and read-backs, at least -10 mm (default: the established '
                        '-6.0). James 10/16: -8, the jaws resting on the table read -6.4 to -6.7 mm (audit doc 14.12)')
    p.add_argument('--table-clearance-plan-mm', type=float, default=None,
                   help='with --table-clearance-floor-mm: planned paths at least this high (mm) while read-backs keep '
                        'the floor (training overhaul doc 17.91, James 10/24: near the table the arm ends 5-8 mm under '
                        'its commands)')
    p.add_argument('--home-lift-below-floor', action='store_true',
                   help='HOME first lifts one joint when the arm stopped just under the read-back floor '
                        '(fcpg/v21_home.lift_below_floor, training overhaul doc 17.91)')
    p.add_argument('--arm-p-coefficient', type=int, choices=(32,), default=None,
                   help='shoulder lift and elbow flex at this servo P after the verified connect instead of '
                        "LeRobot's 16, with torque on and not saved after power-off (fcpg/v21_arm_gain.py, 17.92)")
    # fcpg/v21_explore.py: each block runs the Actor plus a random arm offset inside the physical gate (recording only)
    p.add_argument('--explore-actor', action='store_true')
    p.add_argument('--explore-seed', type=int, default=0)
    # The same two settings the ROS owner (scripts/serve_so101_ros_control.py) ran with on 9/22-23.
    p.add_argument('--hardware-maximum-lateness-s', type=float, choices=(0.03, 0.08, 0.10, 0.20),
                   help='allowed transport lateness after each fixed 0.1 s action boundary (owner default 0.03 s)')
    p.add_argument('--replan-on-execute', action=argparse.BooleanOptionalAction, default=True,
                   help='--no-replan-on-execute executes the explicit plan as planned (one planning pass per block)')
    p.add_argument('--allow-gpu0', action='store_true',
                   help='run the live service on CUDA_VISIBLE_DEVICES=0 (James 2026-10-11: .102 GPU1 is offline)')
    p.add_argument('--c4-checkpoint', type=Path, default=C4_DEFAULTS['checkpoint'])
    p.add_argument('--c4-checkpoint-sha256', default=C4_DEFAULTS['checkpoint_sha256'])
    p.add_argument('--c4-goal-library', type=Path, default=C4_DEFAULTS['library'])
    p.add_argument('--c4-goal-library-sha256', default=C4_DEFAULTS['library_sha256'])
    p.add_argument('--c4-dino-repo', type=Path, default=C4_DEFAULTS['dino_repo'])
    p.add_argument('--c4-dino-weights', type=Path, default=C4_DEFAULTS['dino_weights'])
    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    if args.lewm_cem_action_scale != 1.0 and (not args.lewm_cem or args.raw_actor
            or args.lewm_cem_objective != 'actor_response_tracking'
            or args.lewm_cem_candidate_parameterization not in ('amplified_actor_sequence', 'smooth_actor_endpoint')):
        p.error('nondefault --lewm-cem-action-scale requires LeWM CEM '
                'actor_response_tracking with amplified_actor_sequence')
    if (args.lewm_cem_candidate_parameterization == 'amplified_actor_sequence'
            and args.lewm_cem_action_scale == 1.0):
        p.error('amplified_actor_sequence requires nondefault --lewm-cem-action-scale')
    control_options = (args.lewm_control_checkpoint, args.lewm_control_parent_sha256,
                       args.lewm_control_candidate_sha256)
    control_enabled = any(value is not None for value in control_options)
    if control_enabled and not all(value is not None for value in control_options):
        p.error('LeWM control checkpoint and both SHA256 values are required together')
    if control_enabled:
        if (not args.lewm_cem or args.lewm_cem_objective != 'actor_response_tracking' or args.raw_actor
                or args.grounding_correction or args.head_grasp or args.enable_visual_alignment
                or args.enable_column_dino_fg_calibration or args.enable_gripper_follow_check
                or args.grasp_target != 'body' or args.resume_auxiliary_state is not None
                or args.column_memory is not None or args.trial_max_blocks > 2):
            p.error('LeWM control requires unassisted response CEM, no resume, and trial_max_blocks <= 2')
        if not args.lewm_shadow_checkpoint or args.lewm_shadow_parent_sha256 != args.lewm_control_parent_sha256:
            p.error('LeWM control requires shadow candidates with the same original parent SHA256')
    if bool(args.lewm_shadow_checkpoint) != bool(args.lewm_shadow_parent_sha256):
        p.error('shadow candidates require --lewm-shadow-parent-sha256')
    if args.lewm_cem_objective != 'column_visual' and (args.raw_actor or not args.lewm_cem):
        p.error('actor_response_tracking requires LeWM CEM and cannot use raw Actor mode')
    if args.lewm_cem_objective == 'wrist_distance_alignment' and args.wrist_grasp_calibration is None:
        p.error('wrist_distance_alignment requires --wrist-grasp-calibration')
    if args.lewm_cem_candidate_parameterization is not None and (args.raw_actor or not args.lewm_cem):
        p.error('candidate parameterization requires LeWM CEM and cannot use raw Actor mode')
    if (args.head_grasp or args.grasp_target=='head') and not args.grounding_correction:
        p.error('head targeting requires --grounding-correction')
    if args.raw_actor:
        args.lewm_cem = False
    if args.trial_max_blocks > 16 and args.c4_planner is None and args.alignment_cem is None and not args.explore_actor:
        p.error('--trial-max-blocks above 16 is only for --c4-planner, --alignment-cem or --explore-actor trials')
    if args.explore_actor:
        if (args.c4_planner is not None or args.alignment_cem is not None or args.raw_actor or args.grounding_correction
                or args.head_grasp or args.enable_visual_alignment or args.enable_column_dino_fg_calibration
                or args.enable_gripper_follow_check or args.lewm_shadow_checkpoint or control_enabled
                or args.resume_auxiliary_state is not None or args.column_memory is not None or args.wrist_only
                or args.lewm_cem_candidate_parameterization is not None or args.lewm_cem_action_scale != 1.0
                or args.lewm_cem_objective != 'column_visual'):
            p.error('--explore-actor runs on the unassisted Actor only')
        args.lewm_cem = False
        print(json.dumps(dict(event='v21_explore_actor', seed=args.explore_seed, old_lewm_cem='off')), flush=True)
    if (args.alignment_cem is None) != (args.alignment_response is None):
        p.error('--alignment-cem and --alignment-response are given together')
    if args.grip_latch and args.alignment_cem is None:
        p.error('--grip-latch is part of --alignment-cem')
    if args.alignment_steer_approach and args.alignment_cem is None:
        p.error('--alignment-steer-approach is part of --alignment-cem')
    if args.alignment_roll and (args.alignment_cem is None or args.alignment_box_solid is None):
        p.error('--alignment-roll is part of --alignment-cem and reads the column camera (--alignment-box-solid)')
    if args.alignment_no_box and args.alignment_cem is None:
        p.error('--alignment-no-box is part of --alignment-cem')
    if args.alignment_box_height_mm is not None and (args.alignment_cem is None or args.alignment_no_box):
        p.error('--alignment-box-height-mm is part of --alignment-cem with the box aligned')
    if args.alignment_box_solid is not None and (args.alignment_cem is None or args.alignment_no_box
                                                 or args.alignment_box_height_mm is not None
                                                 or args.box_finder_model is None):
        p.error('--alignment-box-solid is part of --alignment-cem with the box aligned and the learned box finder, '
                'without --alignment-box-height-mm')
    if args.alignment_cem is not None:
        if (args.c4_planner is not None or args.raw_actor or args.grounding_correction or args.head_grasp
                or args.enable_visual_alignment or args.enable_column_dino_fg_calibration
                or args.enable_gripper_follow_check or args.lewm_shadow_checkpoint or control_enabled
                or args.resume_auxiliary_state is not None or args.column_memory is not None or args.wrist_only
                or args.lewm_cem_candidate_parameterization is not None or args.lewm_cem_action_scale != 1.0
                or args.lewm_cem_objective != 'column_visual' or not args.alignment_close_tolerance_px > 0):
            p.error('--alignment-cem runs on the unassisted Actor only')
        args.lewm_cem = False      # the September CEM path stays off; the system LeWM itself is what forecasts
        print(json.dumps(dict(event='v21_alignment_cem', mode=args.alignment_cem, old_lewm_cem='off',
                              lewm='system')), flush=True)
    if args.c4_planner is not None:
        if (args.raw_actor or args.grounding_correction or args.head_grasp or args.enable_visual_alignment
                or args.enable_column_dino_fg_calibration or args.enable_gripper_follow_check
                or args.lewm_shadow_checkpoint or control_enabled or args.resume_auxiliary_state is not None
                or args.column_memory is not None or args.wrist_only
                or args.lewm_cem_candidate_parameterization is not None or args.lewm_cem_action_scale != 1.0
                or args.lewm_cem_objective != 'column_visual'):
            p.error('--c4-planner runs on the unassisted native40 Actor only')
        args.lewm_cem = False
        print(json.dumps(dict(event='v21_c4_planner', mode=args.c4_planner, old_lewm_cem='off')), flush=True)
    visible = ('1', '0') if args.allow_gpu0 else ('1',)
    if args.enable_hardware and (not args.device.startswith('cuda') or os.environ.get('CUDA_VISIBLE_DEVICES') not in visible):
        p.error('live deployment uses physical GPU1: CUDA_VISIBLE_DEVICES=1 (GPU0 only with --allow-gpu0)')
    app = {}; server = make_server((args.host,args.port),app)  # fail before any IO if occupied
    serving = threading.Event()
    stopping = threading.Event()
    def stop(_number, _frame):
        stopping.set()
        if 'service' in app: app['service'].owner.stop_event.set()
        if serving.is_set(): threading.Thread(target=server.shutdown,daemon=True).start()
    for number in (signal.SIGINT,signal.SIGTERM): signal.signal(number,stop)
    worker = threading.Thread(target=lambda:server.serve_forever(poll_interval=.1), daemon=True)
    serving.set(); worker.start()
    try:
        if args.enable_hardware:
            try:
                service_kwargs = ({} if args.lewm_cem_candidate_parameterization is None else
                    dict(lewm_cem_candidate_parameterization=args.lewm_cem_candidate_parameterization))
                if args.temporal_profile is not None:
                    service_kwargs['temporal_profile'] = args.temporal_profile
                if args.hardware_maximum_lateness_s is not None:
                    service_kwargs['hardware_maximum_lateness_s'] = args.hardware_maximum_lateness_s
                if not args.replan_on_execute:
                    service_kwargs['replan_on_execute'] = False
                if args.c4_planner is not None:
                    from fcpg.v21_c4_service import real_planner_factory
                    service_kwargs.update(c4_planner=args.c4_planner, c4_planner_factory=real_planner_factory(
                        args.c4_checkpoint, args.c4_goal_library, dino_repo=args.c4_dino_repo,
                        dino_weights=args.c4_dino_weights, device=args.device,
                        checkpoint_sha256=args.c4_checkpoint_sha256, library_sha256=args.c4_goal_library_sha256))
                if args.alignment_cem is not None:
                    from fcpg.v21_alignment_cem import load_response
                    service_kwargs.update(alignment_cem=args.alignment_cem,
                        alignment_response=load_response(args.alignment_response),
                        alignment_cem_options=dict(close_tolerance_px=args.alignment_close_tolerance_px,
                                                   bf16=not args.alignment_fp32,
                                                   **(dict(dynamics=args.alignment_dynamics)
                                                      if args.alignment_dynamics != 'lewm' else {}),
                                                   **(dict(grip_latch=True) if args.grip_latch else {}),
                                                   **(dict(steer_approach=True) if args.alignment_steer_approach
                                                      else {}),
                                                   **(dict(align_roll=True) if args.alignment_roll else {}),
                                                   **(dict(align_box=False) if args.alignment_no_box else {}),
                                                   **(dict(box_height_m=args.alignment_box_height_mm / 1000.)
                                                      if args.alignment_box_height_mm is not None else {}),
                                                   **(dict(box_solid_camera=args.alignment_box_solid)
                                                      if args.alignment_box_solid is not None else {})))
                if args.explore_actor:
                    service_kwargs.update(explore_actor=True, explore_seed=args.explore_seed)
                if args.home_state is not None:
                    service_kwargs.update(home_state=args.home_state)
                if args.wrist_drop_zone is not None:
                    service_kwargs.update(wrist_drop_zone=args.wrist_drop_zone)
                if args.place_goal_column_camera is not None:
                    service_kwargs.update(place_goal_camera=args.place_goal_column_camera)
                if args.table_clearance_floor_mm is not None:
                    service_kwargs.update(clearance_floor_mm=args.table_clearance_floor_mm)
                if args.table_clearance_plan_mm is not None:
                    if args.table_clearance_floor_mm is None:
                        raise SystemExit('--table-clearance-plan-mm comes with --table-clearance-floor-mm (17.91)')
                    service_kwargs.update(clearance_plan_mm=args.table_clearance_plan_mm)
                if args.home_lift_below_floor:
                    service_kwargs.update(home_lift_below_floor=True)
                if args.arm_p_coefficient is not None:
                    service_kwargs.update(arm_p_coefficient=args.arm_p_coefficient)
                if args.column_finder_model is not None and args.object_finder != 'learned':
                    raise SystemExit('--column-finder-model comes with --object-finder learned (section 17.51)')
                if args.box_finder_model is not None and args.object_finder != 'learned':
                    raise SystemExit('--box-finder-model comes with --object-finder learned (section 17.57)')
                if (args.text_finder or args.text_finder_phrase or args.text_finder_weights) and (
                        args.object_finder != 'learned' or not args.text_finder):
                    raise SystemExit('--text-finder (and its phrase / weights) comes with --object-finder learned')
                if args.object_finder == 'sam3':
                    service_kwargs.update(object_finder=dict(
                        socket=args.sam3_finder_socket, word=args.sam3_object_word,
                        anchor_score=args.sam3_anchor_score, hold_reference=args.sam3_hold_reference))
                elif args.object_finder == 'learned':
                    if args.object_finder_model is None:
                        raise SystemExit('--object-finder learned needs --object-finder-model')
                    text = None
                    if args.text_finder:
                        from fcpg.v21_text_finder import DEFAULT_BPE, DEFAULT_REPO, DEFAULT_WEIGHTS
                        text = dict(phrase=args.text_finder_phrase, weights=args.text_finder_weights or DEFAULT_WEIGHTS,
                                    bpe=DEFAULT_BPE, repo=DEFAULT_REPO)
                    service_kwargs.update(object_finder=dict(
                        kind='learned', model=args.object_finder_model, hold_reference=args.sam3_hold_reference,
                        column_model=args.column_finder_model, box_model=args.box_finder_model, text=text))
                if control_enabled:
                    service_kwargs.update(lewm_control_checkpoint=args.lewm_control_checkpoint,
                        lewm_control_parent_sha256=args.lewm_control_parent_sha256,
                        lewm_control_candidate_sha256=args.lewm_control_candidate_sha256)
                app['service'] = V21LiveService(args.model_output,args.output,device=args.device,task=args.task,
                    column_memory=args.column_memory,column_scene_unchanged=args.column_scene_unchanged,
                    alignment_motion_enabled=args.enable_visual_alignment,wrist_reference=args.wrist_reference,
                    column_dino_fg_calibration=args.enable_column_dino_fg_calibration,
                    wrist_only=args.wrist_only,gripper_check_enabled=args.enable_gripper_follow_check,
                    normal_follow_budget_s=args.normal_follow_budget_s,gripper_force=args.gripper_force,
                    grounding_correction=args.grounding_correction,grounding_px_per_unit=args.grounding_px_per_unit,
                    head_grasp=args.head_grasp,grasp_target=args.grasp_target,
                    record_policy_inputs=args.record_policy_inputs,alignment_scope=args.alignment_scope,
                    resume_auxiliary_state=args.resume_auxiliary_state,
                    camera_condition=args.camera_condition,camera_address=args.camera_address,raw_actor=args.raw_actor,
                    lewm_cem=args.lewm_cem,lewm_cem_objective=args.lewm_cem_objective,
                    wrist_grasp_calibration=args.wrist_grasp_calibration,
                    lewm_cem_action_scale=args.lewm_cem_action_scale,
                    lewm_shadow_checkpoints=args.lewm_shadow_checkpoint,
                    lewm_shadow_parent_sha256=args.lewm_shadow_parent_sha256,
                    trial_max_blocks=args.trial_max_blocks, **service_kwargs)
            except Exception as exc:
                # Failed IO construction already releases its lease. Keep a
                # truthful, non-controlling dashboard available for diagnosis.
                failure = startup_failure_payload(exc)
                app['startup_error'] = failure['error']
                app['startup_diagnostic'] = failure
                print(json.dumps(dict(event='v21_hardware_startup_failed', **failure),
                                  ensure_ascii=False, allow_nan=False),flush=True)
        print(json.dumps(dict(event='v21_dashboard_ready',version=VERSION,contract=CONTRACT,
                              port=args.port,hardware='service' in app,
                              camera_condition=args.camera_condition,camera_address=args.camera_address)),flush=True)
        while worker.is_alive() and not stopping.is_set(): worker.join(timeout=.2)
    finally:
        server.shutdown()
        try:
            if 'service' in app: app['service'].close()
        finally:
            server.server_close()
            worker.join(timeout=2.)


if __name__ == '__main__': main()
