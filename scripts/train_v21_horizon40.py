#!/usr/bin/env python3
"""V21.0.4: native 40-step training, with a five-step execution window.

Explicitly choose all-component random initialization or a named warm start.
Reuse factual recordings and raw DINO features. No robot access.
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcpg.v21_archive_camera_layout import install as install_camera_layout
from fcpg.v21_checks import Pending, deferred, unchecked
from fcpg.v21_data import V21Sources, TaskVocabulary
from fcpg.v21_horizon_training import (
    horizon_targets, horizon_metadata, factual_state, actor_loss, lewm_loss, horizon_errors)
from fcpg.v21_lewm_inputs import lewm_inputs
from fcpg.v21_encoder import DIM, ENCODER, PATCHES
from fcpg.v21_horizon_precision import compute_context
from fcpg.v21_native_horizon import core_from_parent, initialize_native40
from fcpg.v21_scratch_features import ScratchDinoFeatureBank
from fcpg.v21_training import write_json, rng_state, restore_rng
from fcpg.v21_temporal_contract import (constrain_training_age_columns, begin_training_temporal_contract,
    PROGRESS_TRAINING_POLICY)
from fcpg.v21_core import V21Config, V21ResearchCore, CONTRACT, POSITION_UNIT
from scripts.train_v21_from_scratch import (configure_device, derive_splits, DeviceStagedSources,
    stage_chain_to_device, training_loss_scale)
from scripts.train_so101_v21_research import measurement_dropout
from fcpg.v21_scene_lewm import (V21SceneLeWM, SCENE_ARCHITECTURES,
    SCENE_SIGREG_SOURCE, scene_normalization_from_checkpoint, scene_prediction_loss,
    ABSOLUTE_SCENE, RESIDUAL_SCENE, SCENE_PREDICTION_MODES, scene_architecture,
    scene_prediction_from_checkpoint, reset_scene_output, reset_scene_branch,
    SCENE_BRANCH_PREFIXES, SCENE_BRANCH_RESET)
from fcpg.v21_scene_targets import FutureSceneTargets, SCENE_SUPERVISION_MODES
from fcpg.v21_scene_fg import (FG_ARCHITECTURE, enable_fg_readout, fg_checkpoint_mode,
    fg_batch, memory_fg_batch, spatial_supervision, audit_spatial_gradients)
from fcpg.v21_scene_q_grounding import (Q_ARCHITECTURE, Q_TARGET_KIND, enable_q_readout,
    q_checkpoint_mode, q_grounding, audit_q_gradients)
from fcpg.v21_command_memory import COMMAND_MODES, LEGACY_COMMANDS, FULL_COMMANDS
from fcpg.v21_camera_types import LEGACY_SLOTS, REVIEWED_TYPES, camera_mode_from_checkpoint
from fcpg.v21_training_population import (CONFIG_KEY as POPULATION_CONTRACT_KEY,
    isolate_reviewed_recordings, validate_population_resume, effective_data_contract)
from fcpg.v21_training_determinism import (
    configure_training_determinism, reproducibility_record, validate_determinism_resume)
from fcpg.v21_scene_gradient_routing import (COUPLED, ISOLATED, SCENE_GRADIENT_ROUTING,
    saved_scene_gradient_routing, backward_scene_loss)

SCHEMA = 'v21.0.4-native40-training-v1'
TRAINING_MODE_POLICY = 'stage-train-frozen-eval-v1'
SIGREG_FEATURE_SOURCE = 'full-patch-pre-final-layernorm-query-v1'
MEASUREMENT_DROPOUT_MODES = ('none', 'burst-cold-v1')


def resolve_measurement_dropout(requested, *, from_scratch=False):
    """Preserve the historical scratch/warm-start defaults, unless opted in."""
    value = ('burst-cold-v1' if from_scratch else 'none') if requested is None else requested
    if value not in MEASUREMENT_DROPOUT_MODES:
        raise ValueError('unsupported native40 measurement dropout')
    return value


def saved_measurement_dropout(protocol):
    schedule = protocol.get('schedule', {})
    expected = resolve_measurement_dropout(schedule.get('measurement_dropout'),
                                          from_scratch=schedule.get('from_scratch', False))
    actual = protocol.get('measurement_dropout', expected)
    if actual != expected:
        raise ValueError('measurement dropout disagrees with saved schedule')
    return actual


def augment_measurements(chain, rows, *, protocol, variant, images=True, evaluation=False):
    """Only input delivery masks change; actual exposures and targets do not."""
    protocol = resolve_measurement_dropout(protocol)
    if protocol == 'none' or not images or evaluation or not torch.is_grad_enabled():
        return chain
    return measurement_dropout(chain, rows, protocol=protocol, variant=variant)


def configure_training_stage(core, stage, *, from_scratch=False):
    """Only modules selected for optimization enter training mode."""
    if stage not in ('lewm_archive', 'lewm_fish', 'actor_archive', 'actor_dense', 'actor_fish'):
        raise ValueError('unknown native40 training stage')
    core.eval().requires_grad_(False)
    module = core.lewm if stage.startswith('lewm') else core.actor
    module.requires_grad_(True).train()
    if from_scratch and stage.startswith('actor'):
        core.reader.requires_grad_(True).train()
    return module


@contextmanager
def temporary_eval(module):
    """Restore mixed parent/child modes even after an evaluation failure."""
    modes = [(child, child.training) for child in module.modules()]
    try:
        module.eval()
        yield
    finally:
        for child, training in modes:
            child.training = training


@contextmanager
def development_mode(core):
    collect = getattr(core, '_temporal_collect_training_coverage', False)
    with temporary_eval(core):
        try:
            core._temporal_collect_training_coverage = False
            yield
        finally:
            core._temporal_collect_training_coverage = collect


def lewm_sigreg_loss(response, sigreg):
    """Regularize the shared pre-LayerNorm query, never q or pooled patches."""
    features = response.regularization_hidden
    if features is None:
        raise ValueError('full-patch SIGReg requires the pre-final-LayerNorm query')
    return sigreg(features.transpose(0, 1))


def scene_sigreg_coefficient(protocol):
    """Legacy scene runs used .02; redundant saved fields must agree."""
    schedule = protocol.get('schedule', {})
    value = protocol.get('sigreg_weight', schedule.get('scene_sigreg_weight', .02))
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError('scene SIGReg coefficient must be finite and positive')
    if schedule.get('scene_sigreg_weight', value) != value:
        raise ValueError('scene SIGReg coefficient disagrees with saved schedule')
    return float(value)


def scene_training_loss(scene_loss, regularization, joint_loss, *, sigreg_weight=.02, joint_weight=1.):
    """Preserve the original operation order and default gradients/RNG."""
    return scene_loss + sigreg_weight * regularization + joint_weight * joint_loss


def scene_sampling_settings(protocol):
    """Legacy runs mean sparse8 / 1024; never reinterpret a saved optimizer."""
    schedule = protocol.get('schedule', {})
    supervision = protocol.get('scene_supervision', schedule.get('scene_supervision', 'sparse8'))
    projections = protocol.get('sigreg_projections', schedule.get('scene_sigreg_projections', 1024))
    if supervision not in SCENE_SUPERVISION_MODES or projections not in (1024, 8192):
        raise ValueError('unsupported scene sampling settings')
    if (schedule.get('scene_supervision', supervision) != supervision or
            schedule.get('scene_sigreg_projections', projections) != projections):
        raise ValueError('scene sampling settings disagree with saved schedule')
    return supervision, projections


def validate_training_semantics(previous, *, full_patch_lewm, scene_latent_lewm=False,
                                scene_normalization='batchnorm', command_conditioning=LEGACY_COMMANDS,
                                scene_prediction_mode=ABSOLUTE_SCENE, scene_sigreg_weight=.02,
                                deterministic_training=False, scene_supervision='sparse8',
                                scene_sigreg_projections=1024, measurement_dropout=None,
                                scene_joint_gradient_routing=COUPLED, archive_batch_sampling='iid-v1',
                                archive_q_weight=0.):
    validate_determinism_resume(previous, deterministic_training)
    if previous.get('schedule', {}).get('archive_batch_sampling', 'iid-v1') != archive_batch_sampling:
        raise ValueError('Archive sampling changed; use a new output directory')
    if float(previous.get('schedule', {}).get('archive_q_weight', 0.) or 0.) != float(archive_q_weight):
        raise ValueError('Archive q grounding changed; use a new output directory')
    if saved_scene_gradient_routing(previous) != scene_joint_gradient_routing:
        raise ValueError('scene joint gradient routing changed; use a new output directory')
    # Missing historical fields retain the historical scratch/warm-start rule.
    requested = resolve_measurement_dropout(measurement_dropout,
        from_scratch=previous.get('schedule', {}).get('from_scratch', False))
    if saved_measurement_dropout(previous) != requested:
        raise ValueError('measurement dropout changed; use a new output directory')
    expected_features = (SCENE_SIGREG_SOURCE if scene_latent_lewm else
                         SIGREG_FEATURE_SOURCE if full_patch_lewm else None)
    if (previous.get('training_mode_policy') != TRAINING_MODE_POLICY
            or previous.get('sigreg_feature_source') != expected_features):
        raise ValueError('training modes or SIGReg representation changed; use a new output directory')
    if scene_latent_lewm and (
            previous.get('scene_normalization', 'batchnorm') != scene_normalization
            or previous.get('scene_lewm_architecture') != scene_architecture(scene_normalization, scene_prediction_mode)):
        raise ValueError('scene normalization changed; use a new output directory')
    if previous.get('scene_prediction_mode', ABSOLUTE_SCENE) != scene_prediction_mode:
        raise ValueError('scene prediction mode changed; use a new output directory')
    if previous.get('lewm_command_conditioning', LEGACY_COMMANDS) != command_conditioning:
        raise ValueError('command conditioning changed; use a new output directory')
    if scene_latent_lewm and scene_sigreg_coefficient(previous) != scene_sigreg_weight:
        raise ValueError('scene SIGReg weight changed; use a new output directory')
    if scene_latent_lewm and scene_sampling_settings(previous) != (scene_supervision, scene_sigreg_projections):
        raise ValueError('scene sampling changed; use a new output directory')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(state, prefix):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if key.startswith(prefix):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def paired_rng_evidence(device):
    """Capture actual dropout/SIGReg RNG state, without consuming any draws."""
    encode = lambda state: base64.b64encode(state.cpu().numpy().tobytes()).decode('ascii')
    return dict(cpu=encode(torch.get_rng_state()),
                cuda=encode(torch.cuda.get_rng_state(device)) if device.type == 'cuda' else None)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    initialization = p.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--parent', type=Path)
    initialization.add_argument('--from-scratch', action='store_true',
                                help='Randomly initialize and train FCPG, Actor and LeWM with 40 independent queries')
    p.add_argument('--source-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda:0'], default='cuda:0')
    p.add_argument('--gpu-memory-limit-gib', type=float, default=24.)
    p.add_argument('--feature-bank-gib', type=float, default=8.)
    p.add_argument('--feature-bank-device', choices=('training', 'cpu'), default='training',
                   help='CPU cache is available when a shared GPU cannot hold a large feature bank')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lewm-batch-size', type=int, default=None,
                   help='Override archive batch size for LeWM; Actor retains --batch-size')
    p.add_argument('--fish-batch-size', type=int, default=32)
    p.add_argument('--archive-passes', type=int, default=1)
    p.add_argument('--lewm-fish-updates', type=int, default=1000)
    p.add_argument('--actor-fish-updates', type=int, default=4000)
    p.add_argument('--learning-rate', type=float, default=3e-5)
    p.add_argument('--fish-learning-rate', type=float, default=None)
    p.add_argument('--tf32', action='store_true', help='Use GPU TF32 matmuls; keep tensors and losses FP32')
    p.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32',
                   help='BF16 autocast with FP32 master weights, optimizer and physical losses')
    p.add_argument('--fused-optimizer', action='store_true', help='Use fused CUDA AdamW')
    p.add_argument('--deterministic-training', action='store_true',
                   help='Strict numerical policy for a NEW independent run; cannot change on resume')
    p.add_argument('--cache-frozen-actor-inputs', action='store_true',
                   help='Precompute frozen factual inputs for the unchanged actor_fish sampling schedule')
    p.add_argument('--fish-train-episodes', nargs='+', type=int, default=[0, 1],
                   help='Explicit fish fitting episodes: 0 1, or user-authorized 0 1 3')
    p.add_argument('--checkpoint-every', type=int, default=250)
    p.add_argument('--development-rows', type=int, default=512)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=21004)
    p.add_argument('--response-pack', type=Path)
    p.add_argument('--robot-block-pack', type=Path,
                   help="10/31: the robot's own completed blocks with their pictures (fcpg/v21_robot_block_pack.py, a "
                        'train split): --robot-block-rows of them in every LeWM update, beside the response pack')
    p.add_argument('--robot-block-rows', type=int, default=8)
    p.add_argument('--lewm-controller', action='store_true',
                   help="17.153: the new full-patch LeWM reads the arm's control setting [shoulder_lift/elbow_flex "
                        "at P 32, commands from the service's planner]: LeRobot recordings [0, 0], the response pack "
                        "[0, 1], the robot's blocks [P 32 in their session, 1]")
    p.add_argument('--lewm-archive-demo-rows', type=int, default=0,
                   help="17.154: rows of demonstrations 0 and 1 beside every Archive-stage LeWM update (the fish table "
                        "then has LeRobot recordings beside the robot's blocks from the start), with replacement")
    p.add_argument('--lewm-archive-demo-weight', type=float, default=.5,
                   help="17.154: their loss weight (the robot blocks' is .5, the Archive rows' 1)")
    p.add_argument('--robot-arm-p', type=Path,
                   help="17.153: the robot sessions' arm P table (scripts/build_v21_robot_arm_p.py), with "
                        '--lewm-controller and --robot-block-pack')
    p.add_argument('--robot-block-weight', type=float, default=.5,
                   help="10/31: the robot blocks' loss weight (the response pack's is .5)")
    p.add_argument('--robot-blocks-fish-only', action='store_true',
                   help="17.155: the robot blocks in the LeWM's fish stage only (without it they are in the Archive "
                        'stage from the first update)')
    p.add_argument('--lewm-balanced-inputs', action='store_true',
                   help='17.160: the new full-patch LeWM reads every input on the scale of the others: a LayerNorm on '
                        'the picture bank in every layer, (active command - joints) / increment, times / 4 s')
    p.add_argument('--lewm-command-symlog', action='store_true',
                   help="17.161: with --lewm-balanced-inputs, the command distance through DreamerV3's symlog "
                        '(sign(x) log(1 + |x|)): the 4-second queries\' distances shrink to the joints\' scale')
    p.add_argument('--lewm-sam-labels', type=Path,
                   help='17.163: the SAM 3.1 label table of the Archive pictures (scripts/build_v21_sam_label_table.py) '
                        'for --lewm-attention-weight and --lewm-label-head-weight')
    p.add_argument('--lewm-attention-weight', type=float, default=0.,
                   help="17.163: weight of -log(the supervised heads' picture-attention mass on the SAM 3.1 object and "
                        'robot), Archive rows only (0 = none)')
    p.add_argument('--lewm-attention-heads', type=int, default=4,
                   help='17.163: heads per layer under the attention penalty (the first ones; 4 of 8)')
    p.add_argument('--lewm-label-head-weight', type=float, default=0.,
                   help="17.163: weight of a label head's prediction of the SAM 3.1 coverage of the chain's later "
                        'pictures, Archive rows only (0 = no label head)')
    p.add_argument('--full-patch-lewm', action='store_true',
                   help='Train a new 384-d 4-layer LeWM that keeps all 1536 DINO patches')
    p.add_argument('--lewm-width', type=int, default=384,
                   help='17.106: width of the new full-patch LeWM (a multiple of 8; 384 = the deployed LeWM)')
    p.add_argument('--lewm-layers', type=int, default=4,
                   help='17.106: layers of the new full-patch LeWM (4 = the deployed LeWM)')
    p.add_argument('--lewm-patch-encoder-layers', type=int, default=0,
                   help="17.164: self-attention layers of the new full-patch LeWM's own picture encoder over the frozen "
                        'DINO patches of both cameras and all three pictures, before the queries read them (0 = none)')
    p.add_argument('--lewm-memory-pictures', type=int, default=0,
                   help='17.166: the new full-patch LeWM reads the last N pictures of each camera (fcpg/v21_picture_'
                        'memory.py), each pooled into --lewm-memory-tokens tokens, the newest in all its patches (0 = '
                        'the chain\'s three pictures as before)')
    p.add_argument('--lewm-memory-tokens', type=int, default=8,
                   help='17.166: tokens per picture of the picture memory (RT-1: 8)')
    p.add_argument('--lewm-pose-position', action='store_true',
                   help='11/03 (James 「AOK阿」): a concatenating LeWM (--lewm-concat-inputs) whose bank also joins each '
                        'patch\'s position basis x its picture\'s joints as taken and their change since, so the joints '
                        'as taken can weigh patches, not only whole pictures (fcpg/v21_full_patch_lewm.POSE_POSITION)')
    p.add_argument('--lewm-concat-inputs', action='store_true',
                   help='James 11/02 「能concat的就concat」「能合并的就合并网络」: the picture-memory LeWM concatenates '
                        'its inputs and the network that reads them reads them as they are: the pool [patch features, '
                        'position code, camera code], the query layers the bank\'s [token, joints at capture, joints '
                        'now, their change, age] and each step as [learned step query, step inputs, recent motion, '
                        'control setting] (Fig. 1 c), in place of added vectors and the FiLM')
    p.add_argument('--lewm-memory-strides', type=int, nargs='+', default=[1, 2, 3],
                   help="17.166: training draws each row's picture cadence from these half-seconds (the robot's is ~1.2 s)")
    p.add_argument('--lewm-memory-drop', type=float, default=.5,
                   help='17.166: the share of training rows that lose one camera for its newest pictures (view loss)')
    p.add_argument('--lewm-memory-drop-seconds', type=float, default=5.,
                   help='17.166: the longest view loss drawn, seconds (at least one picture)')
    p.add_argument('--lewm-memory-eval-stride', type=int, default=1,
                   help="17.166: the cadence of the run's own development evaluations (no view loss)")
    p.add_argument('--robot-memory-pack', type=Path,
                   help='17.166: the --robot-block-pack blocks\' picture memories from their sessions\' streams '
                        '(robot-pack/train-memory50.pt, build_robot_memory_packs_20261101.py); without it a picture-'
                        'memory LeWM gets their snapshots\' three pictures')
    p.add_argument('--lewm-sigreg-weight', type=float, default=.02,
                   help='17.108: the SIGReg weight of a new full-patch LeWM (on its pre-final-LayerNorm queries, 1024 '
                        'projections); .02 = the deployed LeWM\'s, the original LeWM uses .09')
    p.add_argument('--window-readers', type=int, default=0,
                   help='17.106, LeWM-only runs: the Archive LeWM stage reads each locality window of 2048 rows at '
                        'once in this many separate processes and holds it on the GPU (the Actor trainer\'s '
                        'WindowFeatures, sections 17.79-17.82), the next windows are read while one trains, and the '
                        'next batch\'s labels and chain are built in a separate process; the same draws and values. '
                        '0 = the per-batch row cache, as before')
    p.add_argument('--scene-latent-lewm', action='store_true',
                   help='Warm-start full-patch LeWM; jointly predict learned scenes and actual joints')
    p.add_argument('--scene-normalization', choices=tuple(SCENE_ARCHITECTURES), default='batchnorm',
                   help='Scene projector only: preserve BN baseline or start a separate per-observation LN run')
    p.add_argument('--scene-joint-weight', type=float, default=1.,
                   help='Joint auxiliary loss coefficient in the learned-scene experiment')
    p.add_argument('--scene-joint-gradient-routing', choices=SCENE_GRADIENT_ROUTING, default=COUPLED,
                   help='Training only: isolate joint gradients from scene encoder; preserve all visual gradients')
    p.add_argument('--fish-fg-table', type=Path,
                   help='Opt-in complete production F/G pseudo-labels; spatial auxiliary, no policy input')
    p.add_argument('--scene-fg-weight', type=float, default=.1,
                   help='Separate spatial auxiliary coefficient; does not replace scene SIGReg0.02')
    p.add_argument('--archive-q-weight', type=float, default=0.,
                   help='Opt-in Archive proprioceptive grounding: read capture-time joint angles back '
                        'from scene384 (current, real future, predicted), like the fish F/G readout; 0 = off')
    p.add_argument('--scene-sigreg-weight', type=float, default=.02,
                   help='Scene observation SIGReg coefficient; fixed for the entire independent run')
    p.add_argument('--scene-sigreg-projections', type=int, choices=(1024, 8192), default=1024,
                   help='Random SIGReg directions, NOT visual embedding dimensions')
    p.add_argument('--scene-supervision', choices=SCENE_SUPERVISION_MODES, default='sparse8',
                   help='Visual MSE label times only; SIGReg and primary evaluation keep the original eight times')
    p.add_argument('--scene-prediction-mode', choices=SCENE_PREDICTION_MODES, default=ABSOLUTE_SCENE,
                   help='Explicit separate residual-scene ablation; old runs stay absolute')
    p.add_argument('--reset-scene-head', action='store_true',
                   help='Zero only the scene output layer after loading the parent')
    p.add_argument('--reset-scene-branch', action='store_true',
                   help='Fresh seeded encoder/context/output only; preserve all other parent tensors and training RNG')
    p.add_argument('--paired-schedule-audit', action='store_true',
                   help='Record every selected row, mask and RNG state for an exact paired experiment')
    p.add_argument('--measurement-dropout', choices=MEASUREMENT_DROPOUT_MODES, default=None,
                   help='Input delivery masks only; default burst-cold-v1 for scratch, none for warm start')
    p.add_argument('--lewm-command-conditioning', choices=COMMAND_MODES, default=LEGACY_COMMANDS,
                   help='Opt-in complete causal command tokens; old checkpoints retain sampled-active semantics')
    p.add_argument('--lewm-only', action='store_true',
                   help='Update LeWM only; leave the loaded Actor and FCPG frozen')
    p.add_argument('--lewm-mixed-only', action='store_true', help='Independent bounded mixed-only clock repair')
    p.add_argument('--lewm-archive-only-diagnostic', action='store_true',
                   help='Replace only the fish half of bounded LeWM updates with independent Archive samples')
    p.add_argument('--archive-batch-sampling', choices=('iid-v1', 'episode-v1', 'episode-pair-v1'), default='iid-v1',
                   help='Independent bounded experiment: Archive half from one episode, or from two '
                        'episodes x 8 like the fish half (episode-pair-v1)')
    p.add_argument('--locked-development-report', type=Path)
    p.add_argument('--clock-baseline-report', type=Path)
    p.add_argument('--canary', action='store_true')
    # docs/V21_FISH_STAGE_REDO_20261012.md: continue a native40 from-scratch run from one of its completed Actor
    # stages and train only the fish stage; keep and evaluate chosen intermediate optimizer steps.
    p.add_argument('--from-stage-checkpoint', type=Path,
                   help='completed actor_archive/actor_dense checkpoint of a native40 from-scratch run; only actor_fish trains')
    p.add_argument('--fish-stage-lewm', type=Path,
                   help='with --from-stage-checkpoint: checkpoint whose LeWM (frozen) supplies the feedback inputs')
    p.add_argument('--keep-steps', type=int, nargs='*', default=[],
                   help='also keep step-XXXXX.pt and write a development evaluation at these optimizer steps')
    p.add_argument('--camera-layout', type=Path,
                   help='LeWM-only: layout.npz from the Archive camera audit; Archive cameras in the model order '
                        '(slot 0 wrist) and only rows of episodes whose layout is decided (section 17.59)')
    p.add_argument('--object-cues', type=Path,
                   help='section 17.120, LeWM-only with --camera-layout: the 17.117 cue tables of that layout '
                        '(scripts/build_v21_halfsecond_cue_tables.py), which tell the Archive rows that hold something; '
                        'the LeWM reads no cue, the frozen Actor\'s evaluations do')
    p.add_argument('--carry-forecast-weight', type=float, default=0.,
                   help='section 17.120 (James 10/27 「重重罚阿 还有lewm」), LeWM stages: on every training row that '
                        'holds something (the grasp cue on; fish rows also with the jaws shut), this weight times the '
                        'mean squared amount (per 10 mm beyond 2 mm) by which the forecast jaws of the first five '
                        'read-backs end ABOVE the recorded ones (the planner would believe the arm higher than it '
                        'will be); 0 changes nothing')
    p.add_argument('--carry-place-factor', type=float, default=1.,
                   help='section 17.120: with --carry-forecast-weight, held rows that also see where the thing goes '
                        '(the place cue on) count this many times')
    p.add_argument('--lewm-archive-from', type=Path,
                   help='section 17.123, fresh --lewm-only --full-patch-lewm runs: the completed lewm_archive/final.pt '
                        'of a run with this recipe, restored in place of training the Archive stage (its weights, RNG '
                        'and response draws, as a resume restores them): only lewm_fish trains, from that same start')
    p.add_argument('--lewm-fish-from-archive', type=Path,
                   help='section 17.124, fresh --lewm-only --full-patch-lewm runs: a task pool '
                        '(make_task_pools_20261029.py) of a collection the Archive stage never trained on; the fish '
                        'stage learns its two episodes in place of the fish demonstrations (the same half of every '
                        'batch) and is examined on its other episodes (report "task"); the Archive development report '
                        'leaves the collection out')
    p.add_argument('--lewm-fish-start', type=Path,
                   help='section 17.125, with --lewm-archive-from: the fish/final.pt of a completed formal LeWM-only '
                        'run that restored the same Archive stage; the fish stage starts from its LeWM instead')
    p.add_argument('--lewm-fish-demos-only', action='store_true',
                   help='section 17.125: the fish stage learns the demonstrations alone: no Archive half (its rows '
                        'still drawn, so the demonstration rows are those of the mixed run) and no retention pack')
    p.add_argument('--fish-stage-seed', type=int,
                   help='section 17.127, with --lewm-archive-from: another draw of the LeWM fish stage alone (its rows, '
                        'the Archive rows that replace the fish, dropout, SIGReg directions and retention draws); the '
                        'restored Archive stage and the examination rows stay those of --seed')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--resume-with-resources', action='store_true',
                   help='Resume saved optimizer/sampler state with only GPU cap or feature cache changes')
    p.add_argument('--resume-with-precision', action='store_true',
                   help='Explicitly record a precision-only continuation of the same initialized run')
    return p


def carry_forecast_penalty(predicted, recorded, valid, gradient, held, factor=None, *, margin_mm=2., scale_mm=10.):
    """section 17.120: over held rows' valid read-backs, the mean squared amount (per scale_mm, beyond margin_mm) by
    which the forecast jaws end above the recorded ones, linearised by the row's height gradient; each row's cost times
    factor [B] (None: 1). predicted, recorded [B, n, 6]; valid [B, n] bool; gradient [B, 5] mm per unit; held [B]."""
    above = ((predicted[..., :5] - recorded[..., :5]) * gradient[:, None, :]).sum(-1)
    cost = (torch.relu(above - margin_mm) / scale_mm).square()
    if factor is not None:
        cost = cost * factor[:, None].to(cost.dtype)
    weight = (valid & held[:, None]).to(cost.dtype)
    return (cost * weight).sum() / weight.sum().clamp_min(1.)


def ordered_groups(stage, splits, args, *, archive_episode_index=None, task_rows=None):
    """One original row per pass, balanced over factual context lengths 0..3. task_rows (17.124): in lewm_fish, the
    fish half of every batch is drawn from these Archive rows instead (the same draw, labelled archive). With
    --lewm-fish-demos-only (17.125) the lewm_fish batches keep that half alone."""
    fish_stage_seed = getattr(args, 'fish_stage_seed', None)       # 17.127: another draw of the LeWM fish stage
    seed = fish_stage_seed if stage == 'lewm_fish' and fish_stage_seed is not None else args.seed
    rng = np.random.default_rng(seed + ['lewm_archive', 'lewm_fish', 'actor_archive', 'actor_dense', 'actor_fish'].index(stage))
    if stage.endswith('_fish'):
        count = args.lewm_fish_updates if stage.startswith('lewm') else args.actor_fish_updates
        half = args.fish_batch_size // 2
        archive_diagnostic = (stage == 'lewm_fish'
                              and getattr(args, 'lewm_archive_only_diagnostic', False))
        replacement_rng = np.random.default_rng(seed + 10001) if archive_diagnostic else None
        grouped = None
        mode = getattr(args, 'archive_batch_sampling', 'iid-v1')
        if mode != 'iid-v1':
            from fcpg.v21_episode_sampling import EpisodeBatchSampler, EPISODES_PER_BATCH
            if mode not in EPISODES_PER_BATCH:
                raise ValueError('unknown Archive batch sampling mode')
            if stage != 'lewm_fish' or archive_diagnostic or archive_episode_index is None:
                raise ValueError('episode grouping requires mixed LeWM and explicit TRAIN episode identities')
            grouped = EpisodeBatchSampler(splits['train']['archive'], archive_episode_index)
            episodes_per_batch = EPISODES_PER_BATCH[mode]
            grouped_rng = np.random.default_rng(args.seed + 30001)
        task = task_rows is not None and stage == 'lewm_fish'
        if task and (archive_diagnostic or grouped is not None):
            raise ValueError('a task pool replaces the fish half of plain iid mixed batches only')
        for index in range(2 if args.canary else count):
            groups = [(kind, rng.choice(splits['train'][kind], half, replace=True))
                      if not (task and kind == 'fish') else ('archive', rng.choice(task_rows, half, replace=True))
                      for kind in ('fish', 'archive')]
            if archive_diagnostic:
                # Consume the legacy index draws (not fish examples) so the
                # original Archive half remains exactly paired with the control.
                # Keep two independent loss/SIGReg groups instead of pooling B.
                groups[0] = ('archive', replacement_rng.choice(splits['train']['archive'], half, replace=True))
            if grouped is not None:
                # Consume the original draws above: fish rows/RNG remain exactly
                # paired. Only Archive's within-batch correlation is changed.
                groups[1] = ('archive', grouped.draw_episodes(grouped_rng, half, episodes_per_batch))
            if stage == 'lewm_fish' and getattr(args, 'lewm_fish_demos_only', False):
                groups = groups[:1]          # 17.125: the Archive half was drawn above, so these rows stay paired
            yield index % 4, groups
    else:
        batch_size = (getattr(args, 'lewm_batch_size', None) or args.batch_size) if stage.startswith('lewm') else args.batch_size
        population = splits['train']['dense' if stage.endswith('_dense') else 'archive']
        index = 0
        for _ in range(1 if stage.endswith('_dense') else args.archive_passes):
            # Locality windows retain randomization while avoiding repeated random
            # multi-gigabyte reads of immutable visual features.
            windows = [population[i:i+2048] for i in range(0, len(population), 2048)]
            for window_id in rng.permutation(len(windows)):
                window = rng.permutation(windows[window_id])
                for start in range(0, len(window), batch_size):
                    contexts = range(4) if getattr(args, 'from_scratch', False) else [index % 4]
                    for h in contexts:
                        yield h, [('archive', window[start:start+batch_size])]
                        index += 1
                        if args.canary and index == (4 if getattr(args, 'from_scratch', False) else 2):
                            return


def archive_windows(batches, population, readers, size=2048):
    """17.106: ordered_groups()' Archive draws unchanged, each with its locality window: (h, groups, window), window =
    (key, rows, last, following). key numbers the windows in the order they are drawn, rows are the window's
    population rows (population[i:i+size], ascending: one near-sequential read), last marks the batch that draws its
    final rows, following lists the next `readers` windows [(key, rows)] to read ahead. ordered_groups() draws every
    window's rows in consecutive batches (one context per batch), so a window is done once all its rows were drawn."""
    order = np.argsort(population, kind='stable')
    ordered = population[order]

    def runs():
        key, window, left = -1, None, 0
        for h, groups in batches:
            (kind, rows), = groups
            at = order[np.searchsorted(ordered, rows).clip(max=len(ordered) - 1)]
            if kind != 'archive' or not np.array_equal(population[at], rows):
                raise ValueError('window reading takes Archive population rows only')
            w = at // size
            if (w != w[0]).any():
                raise ValueError('an Archive batch spans two locality windows')
            if not left:
                key, window = key + 1, int(w[0])
                left = len(population[window * size:(window + 1) * size])
            elif int(w[0]) != window:
                raise ValueError('a locality window was left before all of its rows were drawn')
            left -= len(rows)
            if left < 0:
                raise ValueError('a locality window gave more rows than it has')
            yield h, groups, key, window, not left

    pending, drawn = deque(), runs()
    while True:
        while not pending or pending[-1][2] < pending[0][2] + readers:     # up to the first batch of the last to read
            item = next(drawn, None)
            if item is None:
                break
            pending.append(item)
        if not pending:
            return
        h, groups, key, window, last = pending.popleft()
        following, seen = [], set()
        for _, _, k, w, _ in pending:
            if k != key and k not in seen and len(following) < readers:
                seen.add(k)
                following.append((k, population[w * size:(w + 1) * size]))
        yield h, groups, (key, population[window * size:(window + 1) * size], last, following)


def one_ahead(items):
    """Each item with the one after it (None after the last): the next batch is built while this one trains."""
    items = iter(items)
    current = next(items, None)
    while current is not None:
        upcoming = next(items, None)
        yield current, upcoming
        current = upcoming


_BUILDING = {}          # 17.106: what the batch-building process runs, set before it is forked (the Actor's 17.81)


def _build_in_process(*args):
    """prepare_host() in the batch-building process (forked from the trainer; it never touches the GPU)."""
    return _BUILDING['prepare'](*args)


def select_fish_training_episodes(splits, episodes):
    episodes = list(episodes)
    if episodes not in ([0, 1], [0, 1, 3]):
        raise ValueError('fish fitting episodes must be 0 1 or 0 1 3')
    result = {role: {kind: rows.copy() for kind, rows in groups.items()}
              for role, groups in splits.items() if role in ('train', 'development')}
    if episodes == [0, 1, 3]:
        result['train']['fish'] = np.sort(np.concatenate((result['train']['fish'], result['development']['fish'])))
        result['development']['fish'] = np.empty(0, dtype=np.int64)
    return result


# Recording isolation is a shared data boundary for BOTH arms of a sampling
# A/B, so the fish-like two-episode sampler may use the isolated pool. The
# single-episode mode stays separate (it also changes prediction diversity).
POPULATION_SAMPLING = ('iid-v1', 'episode-pair-v1')


def validate_population_options(args, population_contract):
    if population_contract is not None and (
            not args.lewm_only or not args.scene_latent_lewm or args.from_scratch
            or args.archive_batch_sampling not in POPULATION_SAMPLING
            or args.reset_scene_head or args.reset_scene_branch):
        raise ValueError('recording isolation keeps Actor/FCPG frozen, IID or paired-episode '
                         'sampling and parent scene heads')


Q_READOUT_SEED_OFFSET = 50001


def validate_q_grounding_options(args):
    """Archive q grounding is a bounded scene-LeWM-only warm-start factor."""
    weight = getattr(args, 'archive_q_weight', 0.)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError('Archive q grounding weight must be finite and nonnegative')
    if weight and (not args.scene_latent_lewm or not args.lewm_only or args.from_scratch
                   or not args.lewm_mixed_only or args.lewm_archive_only_diagnostic
                   or args.reset_scene_head or args.reset_scene_branch):
        raise ValueError('Archive q grounding requires a mixed scene-LeWM-only warm start '
                         'with the parent scene heads')


def prepare_training_splits(source, config, fish_train_episodes):
    """Single population boundary for Archive, dense and mixed training stages."""
    contract = config.get(POPULATION_CONTRACT_KEY)
    if contract is not None and list(fish_train_episodes) != [0, 1]:
        raise ValueError('recording isolation preserves fish0/1 TRAIN and fish3 DEV')
    splits = select_fish_training_episodes(derive_splits(source), fish_train_episodes)
    return isolate_reviewed_recordings(splits, source, contract)


def apply_camera_layout(source, splits, path):
    """Section 17.59 (4th fix): Archive cameras in the model's order, slot 0 the wrist camera as on the fish robot
    (fcpg/v21_archive_camera_layout.py), and in both Archive splits only rows of episodes whose layout is decided.
    Installs on the source, so it runs before any feature bank or staged copy is made; fish rows are untouched.
    Returns the protocol record (table identity and the coverage before the rows were dropped)."""
    layout = install_camera_layout(source, path)
    record = dict(layout.identity, archive_rows={role: layout.coverage(source.archive, splits[role]['archive'])
                                                 for role in ('train', 'development')})
    for role in ('train', 'development'):
        splits[role]['archive'] = layout.admitted_rows(source.archive, splits[role]['archive'])
    return record


def canonical_protocol(protocol):
    value = deepcopy(protocol)
    routing = saved_scene_gradient_routing(value)
    value['scene_joint_gradient_routing'] = routing
    value['schedule']['scene_joint_gradient_routing'] = routing
    measurement = saved_measurement_dropout(value)
    value['measurement_dropout'] = measurement
    value['schedule']['measurement_dropout'] = measurement
    value['schedule'].setdefault('deterministic_training', False)
    value.setdefault('reproducibility', None)
    # Native40 runs predating the precision option used FP32, optionally TF32.
    value['schedule'].setdefault('precision', 'fp32')
    value['schedule'].setdefault('feature_bank_device', 'training')
    value['schedule'].setdefault('scene_latent_lewm', False)
    value['schedule'].setdefault('scene_joint_weight', 1.)
    value['schedule'].setdefault('fish_fg_table', None)
    value['schedule'].setdefault('scene_fg_weight', .1)
    for key in ('scene_fg_architecture', 'scene_fg_table', 'scene_fg_weight', 'scene_fg_label_contract'):
        value.setdefault(key, None)
    value['schedule'].setdefault('archive_q_weight', 0.)
    for key in ('scene_q_architecture', 'archive_q_weight', 'scene_q_domains', 'scene_q_target',
                'scene_q_initialization'):
        value.setdefault(key, None)
    scene_weight = scene_sigreg_coefficient(value) if value['schedule']['scene_latent_lewm'] else .02
    value['schedule'].setdefault('scene_sigreg_weight', scene_weight)
    if value['schedule']['scene_latent_lewm']:
        value.setdefault('sigreg_weight', scene_weight)
        supervision, projections = scene_sampling_settings(value)
        value.setdefault('scene_supervision', supervision)
        value.setdefault('sigreg_projections', projections)
    else:
        supervision, projections = 'sparse8', 1024
        value.setdefault('scene_supervision', None)
    value['schedule'].setdefault('scene_supervision', supervision)
    value['schedule'].setdefault('scene_sigreg_projections', projections)
    value['schedule'].setdefault('lewm_mixed_only', False)
    value['schedule'].setdefault('lewm_archive_only_diagnostic', False)
    value['schedule'].setdefault('archive_batch_sampling', 'iid-v1')
    value.setdefault('archive_batch_sampling', 'iid-v1')
    value.setdefault('archive_camera_role_adapter', None)
    value['schedule'].setdefault('locked_development_report', None)
    value['schedule'].setdefault('clock_baseline_report', None)
    value.setdefault('scene_prediction_mode', ABSOLUTE_SCENE)
    value['schedule'].setdefault('scene_prediction_mode', value['scene_prediction_mode'])
    value['schedule'].setdefault('reset_scene_head', False)
    value['schedule'].setdefault('reset_scene_branch', False)
    value.setdefault('scene_branch_initialization', None)
    value['schedule'].setdefault('paired_schedule_audit', False)
    value['schedule'].setdefault('camera_layout', None)
    value.setdefault('scene_head_initialization', 'parent-or-default-v1')
    value.setdefault('corrected_cache_generation', None)
    value.setdefault('lewm_command_conditioning', LEGACY_COMMANDS)
    value['schedule'].setdefault('lewm_command_conditioning', value['lewm_command_conditioning'])
    for key in ('scene_lewm_architecture', 'scene_targets',
                'scene_target_encoder_stop_gradient', 'joint_auxiliary_weight'):
        value.setdefault(key, None)
    value.setdefault('scene_normalization', next((name for name, architecture in SCENE_ARCHITECTURES.items()
        if architecture == value['scene_lewm_architecture']), None))
    value['schedule'].setdefault('scene_normalization', value['scene_normalization'] or 'batchnorm')
    value.setdefault('scene_dino_patch_reconstruction', False)
    # 17.106 / 17.108 options: runs from before them used these values
    value['schedule'].setdefault('lewm_width', 384)
    value['schedule'].setdefault('lewm_layers', 4)
    value['schedule'].setdefault('window_readers', 0)
    value['schedule'].setdefault('lewm_sigreg_weight', .02)
    # 17.123 options: runs from before them trained their own Archive stage
    value['schedule'].setdefault('lewm_archive_from', None)
    value.setdefault('lewm_archive_stage_from', None)
    # runs saved before the LeWM-only vocabulary note read the parent's words with nothing to note (None)
    value.setdefault('vocabulary_note', None)
    # 17.124: runs from before it learned the fish demonstrations in the fish stage
    value['schedule'].setdefault('lewm_fish_from_archive', None)
    value.setdefault('lewm_fish_from_archive', None)
    # 17.125: runs from before it started the fish stage where the Archive stage ended and mixed Archive into it
    value['schedule'].setdefault('lewm_fish_start', None)
    value['schedule'].setdefault('lewm_fish_demos_only', False)
    value.setdefault('lewm_fish_start', None)
    # 17.127: runs from before it drew their fish stage from --seed
    value['schedule'].setdefault('fish_stage_seed', None)
    # 17.160: runs from before it read their LeWM inputs unbalanced
    value['schedule'].setdefault('lewm_balanced_inputs', False)
    value['schedule'].setdefault('lewm_command_symlog', False)     # 17.161
    # 17.163: runs from before the SAM 3.1 attention penalty and label head
    value['schedule'].setdefault('lewm_sam_labels', None)
    value['schedule'].setdefault('lewm_attention_weight', 0.)
    value['schedule'].setdefault('lewm_attention_heads', 4)
    value['schedule'].setdefault('lewm_label_head_weight', 0.)
    value['schedule'].setdefault('lewm_patch_encoder_layers', 0)    # 17.164: runs from before the picture encoder
    # 17.166: runs from before the picture memory
    for key, default in (('lewm_memory_pictures', 0), ('lewm_memory_tokens', 8), ('lewm_memory_strides', [1, 2, 3]),
                         ('lewm_memory_drop', .5), ('lewm_memory_drop_seconds', 5.), ('lewm_memory_eval_stride', 1),
                         ('robot_memory_pack', None), ('lewm_concat_inputs', False), ('lewm_pose_position', False)):
        value['schedule'].setdefault(key, default)
    return value


# 17.123: what may differ between a run and the run whose Archive stage it restores (--lewm-archive-from): where the
# data, the outputs and the source config are, the GPU memory caps, the smoke flag and the fish-stage options of
# 17.123-17.125. The LeWM's random initialization is not compared (its weights are replaced by the restored ones,
# loaded strictly).
ARCHIVE_FROM_FREE = ('original_source_config', 'source_config_sha256', 'smoke', 'lewm_archive_stage_from',
                     'lewm_fish_from_archive', 'lewm_fish_start')
ARCHIVE_FROM_FREE_SCHEDULE = ('output', 'source_config', 'canary', 'lewm_archive_from', 'lewm_fish_from_archive',
                              'lewm_fish_start', 'lewm_fish_demos_only', 'lewm_archive_only_diagnostic',
                              'fish_stage_seed', 'gpu_memory_limit_gib', 'feature_bank_gib', 'feature_bank_device',
                              'cache_frozen_actor_inputs')


def load_fish_start(path, archive_from):
    """17.125: the LeWM a completed formal LeWM-only run left after its fish stage, from the Archive stage this run
    restores (--lewm-archive-from): where this run's fish stage starts."""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    protocol = saved.get('native40_protocol') or {}
    if (saved.get('complete') is not True or saved.get('smoke') is not False or saved.get('stage') != 'fish'
            or protocol.get('trained_modules') != ['lewm']):
        raise ValueError('--lewm-fish-start needs the fish/final.pt of a completed formal LeWM-only run')
    if (protocol.get('lewm_archive_stage_from') or {}).get('sha256') != sha(archive_from):
        raise ValueError('--lewm-fish-start: that run restored another Archive stage')
    schedule = protocol.get('schedule') or {}
    return dict(lewm={k[len('lewm.'):]: v for k, v in saved['model'].items() if k.startswith('lewm.')},
                record=dict(checkpoint=str(path), sha256=sha(path), run=schedule.get('output'),
                            archive_only=bool(schedule.get('lewm_archive_only_diagnostic')),
                            fish_from_archive=protocol.get('lewm_fish_from_archive'),
                            demos_only=bool(schedule.get('lewm_fish_demos_only'))))


def load_task_pool(path, splits, archive):
    """17.124: a task pool's rows, checked against this run's splits: learning and test rows in the development split
    only (the Archive stage never trained on them), from different episodes, none sharing an episode with a training
    row; excluded_rows (the collection and any copy of it) leave the Archive development report."""
    pool = json.loads(Path(path).read_text())
    rows = {key: np.asarray(pool[key], dtype=np.int64) for key in ('learn_rows', 'test_rows', 'excluded_rows')}
    development, train = splits['development']['archive'], splits['train']['archive']
    episode = lambda r: np.asarray(archive.endpoint_episode_index[archive.row_endpoint_index[r, 0]])
    if pool.get('section') != '17.124' or not len(rows['learn_rows']) or not len(rows['test_rows']):
        raise ValueError('not a 17.124 task pool')
    if not all(np.isin(rows[key], development).all() for key in rows):
        raise ValueError('a task pool row is not in the Archive development split')
    if np.isin(episode(rows['learn_rows']), episode(rows['test_rows'])).any():
        raise ValueError('the task pool learns and tests on the same episode')
    if np.isin(np.concatenate([episode(rows['learn_rows']), episode(rows['test_rows'])]), episode(train)).any():
        raise ValueError('a task pool episode has Archive training rows')
    if not (np.isin(rows['learn_rows'], rows['excluded_rows']).all() and np.isin(rows['test_rows'], rows['excluded_rows']).all()):
        raise ValueError('the task pool\'s own rows must leave the Archive development report')
    record = dict(path=str(path), sha256=sha(path), task=pool['task'], collections=pool['collections'],
                  learn_episodes=pool['learn_episodes'], test_episodes=len(pool['test_episodes']),
                  **{key: int(len(value)) for key, value in rows.items()})
    return dict(rows, record=record)


def archive_stage_recipe(protocol, defaults):
    """The recipe a restored Archive stage must share (see ARCHIVE_FROM_FREE); an option the saved run predates counts
    as its default value (the option did not exist, so the run had what the default does)."""
    value = canonical_protocol(protocol)
    value['schedule'] = dict({k: v for k, v in defaults.items() if k != 'help'}, **value['schedule'])
    for key in ARCHIVE_FROM_FREE:
        value.pop(key, None)
    for key in ARCHIVE_FROM_FREE_SCHEDULE + ('resume', 'resume_with_resources', 'resume_with_precision'):
        value['schedule'].pop(key, None)
    for key in ('gpu_memory_limit_gib', 'gpu_memory_fraction'):
        (value.get('device') or {}).pop(key, None)
    value['initial_module_sha256'] = {k: v for k, v in (value.get('initial_module_sha256') or {}).items() if k != 'lewm'}
    return value


def _flat(value, prefix=''):
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in _flat(item, f'{prefix}{key}.').items()}
    return {prefix[:-1]: value}


def validate_archive_stage_source(saved, protocol, source_config, defaults):
    """17.123: --lewm-archive-from restores a completed formal lewm_archive stage of this run's recipe. A different
    source config may only point at another copy of the same DINO cache (the same manifest, e.g. the NVMe one of
    17.65), with nothing else changed but notes."""
    if saved.get('complete') is not True or saved.get('smoke') is not False or saved.get('stage') != 'lewm_archive':
        raise ValueError('--lewm-archive-from needs the final checkpoint of a completed formal lewm_archive stage')
    previous = saved['native40_protocol']
    old, new = _flat(archive_stage_recipe(previous, defaults)), _flat(archive_stage_recipe(protocol, defaults))
    differing = sorted(k for k in set(old) | set(new) if old.get(k, '<absent>') != new.get(k, '<absent>'))
    if differing:
        raise ValueError('--lewm-archive-from: the restored run had another recipe: '
                         + '; '.join(f'{k}: {old.get(k, "<absent>")!r} -> {new.get(k, "<absent>")!r}' for k in differing))
    if sha(previous['original_source_config']) != previous['source_config_sha256']:
        raise ValueError('--lewm-archive-from: the restored run\'s source config changed since')
    if sha(source_config) != previous['source_config_sha256']:
        before = json.loads(Path(previous['original_source_config']).read_text())
        after = json.loads(Path(source_config).read_text())
        roots = before['source_paths']['dino_cache_root'], after['source_paths']['dino_cache_root']
        strip = lambda c: {k: v for k, v in _flat(c).items()
                           if k != 'source_paths.dino_cache_root' and not k.split('.')[-1].endswith('_note')}
        manifests = [json.loads((Path(root)/'manifest.json').read_text()) for root in roots]
        if strip(before) != strip(after) or manifests[0] != manifests[1]:
            raise ValueError('--lewm-archive-from: another source config may only hold another copy of the same cache')
    return dict(checkpoint_stage=saved['stage'], stage_optimizer_step=saved.get('stage_optimizer_step'),
                same_source_config=sha(source_config) == previous['source_config_sha256'])


def validate_resource_continuation(previous, current):
    """Allow storage capacity changes without changing the training recipe."""
    def recipe(protocol):
        value = canonical_protocol(protocol)
        for key in ('gpu_memory_limit_gib', 'feature_bank_gib', 'feature_bank_device', 'cache_frozen_actor_inputs'):
            value['schedule'].pop(key, None)
        for key in ('gpu_memory_limit_gib', 'gpu_memory_fraction'):
            value['device'].pop(key, None)
        return value
    if recipe(previous) != recipe(current):
        raise ValueError('resource continuation cannot change data, GPU identity, model, precision or optimizer schedule')


def validate_precision_continuation(previous, current):
    old, new = canonical_protocol(previous), canonical_protocol(current)
    before = old['schedule'].pop('precision')
    after = new['schedule'].pop('precision')
    if before not in ('fp32', 'bf16') or after not in ('fp32', 'bf16') or before == after or old != new:
        raise ValueError('precision continuation may only change fp32/bf16, preserving data, GPU, initialization and optimizer schedule')


def restore_training_checkpoint(core, saved, optimizer=None, *, scene_sigreg_weight=None,
                                scene_sampling=None,
                                deterministic_training=None, current_runtime=None,
                                scene_joint_gradient_routing=COUPLED,
                                population_contract=None, population_report=None):
    validate_population_resume(saved.get('native40_protocol', {}), population_contract,
                               report=population_report)
    if saved_scene_gradient_routing(saved.get('native40_protocol', {})) != scene_joint_gradient_routing:
        raise ValueError('checkpoint scene joint gradient routing differs')
    if camera_mode_from_checkpoint(saved) != getattr(core.lewm, 'camera_conditioning', LEGACY_SLOTS):
        raise ValueError('checkpoint camera conditioning differs')
    if q_checkpoint_mode(saved) != getattr(core.lewm, 'scene_q_architecture', None):
        raise ValueError('checkpoint Archive q readout differs')
    if deterministic_training is not None:
        validate_determinism_resume(saved.get('native40_protocol', {}), deterministic_training,
                                    current_runtime=current_runtime)
    contract = saved.get('temporal_training_contract')
    if contract != getattr(core, 'temporal_training_contract', None):
        raise ValueError('checkpoint temporal training policy differs')
    if contract is not None and 'temporal_feature_coverage' not in saved:
        raise ValueError('checkpoint is missing temporal training coverage')
    if isinstance(core.lewm, V21SceneLeWM):
        if scene_sampling is not None and scene_sampling_settings(saved.get('native40_protocol', {})) != scene_sampling:
            raise ValueError('checkpoint scene sampling differs')
        if (scene_sigreg_weight is not None and
                scene_sigreg_coefficient(saved.get('native40_protocol', {})) != scene_sigreg_weight):
            raise ValueError('checkpoint scene SIGReg weight differs')
        if scene_normalization_from_checkpoint(saved) != core.lewm.scene_normalization:
            raise ValueError('checkpoint scene normalization differs')
        if scene_prediction_from_checkpoint(saved) != core.lewm.scene_prediction_mode:
            raise ValueError('checkpoint scene prediction mode differs')
    core.load_state_dict(saved['model'])
    if optimizer is not None:
        optimizer.load_state_dict(saved['optimizer'])
    if contract is not None:
        core.temporal_feature_coverage = deepcopy(saved['temporal_feature_coverage'])
    restore_rng(saved['rng'])


def load_response_pack(path, device):
    if path is None:
        return None
    from fcpg.prod_v20_policy import ActualHistory
    from fcpg.v21_models import CommandTrace
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value.get('schema') != 'v21-h40-local-response-retention-v1' or value.get('split') != 'train':
        raise ValueError('local retention pack must contain previously eligible real TRAIN readbacks')
    # 17.171: the pack (6 MB) is held on the device and each batch gathered there: copying every batch to the GPU made
    # the CPU wait for it eleven times an update; the same rows and values
    on = lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
    data = {k: ({n: on(t) for n, t in v.items()} if isinstance(v, dict) else on(v)) for k, v in value['data'].items()}
    def batch(ids):
        ids = torch.from_numpy(np.asarray(ids, np.int64))
        ids = ids.pin_memory().to(device, non_blocking=True) if torch.device(device).type == 'cuda' else ids.to(device)
        take = lambda x: x[ids]
        return (take(data['q']),
            CommandTrace(**{k: take(v) if isinstance(v, torch.Tensor) else v for k, v in data['trace'].items()}),
            ActualHistory(**{k: take(v) for k, v in data['history'].items()}),
            take(data['query_offsets']), take(data['actual']))
    return len(data['q']), batch


def initialize_scratch40(vocabulary, scale, seed, device):
    """Native random query initialization; no checkpoint or five-query tiling."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    core = V21ResearchCore(V21Config(len(vocabulary.words), seed=seed, object_relative=True,
                                   prediction_horizon=40, execution_horizon=5)).float()
    core.set_loss_scale(scale)
    core = core.to(device)
    begin_training_temporal_contract(core, PROGRESS_TRAINING_POLICY)
    return core


def main(argv=None):
    args = parser().parse_args(argv)
    if args.archive_batch_sampling != 'iid-v1' and (
            not args.lewm_mixed_only or not args.lewm_only or not args.scene_latent_lewm
            or args.from_scratch or args.lewm_archive_only_diagnostic):
        raise ValueError('episode sampling requires a bounded mixed scene-LeWM-only experiment')
    if args.scene_joint_gradient_routing != COUPLED and (
            not args.scene_latent_lewm or not args.lewm_only or args.from_scratch):
        raise ValueError('scene gradient isolation requires a scene LeWM-only parent experiment')
    args.measurement_dropout = resolve_measurement_dropout(args.measurement_dropout,
                                                         from_scratch=args.from_scratch)
    if not args.from_scratch and args.measurement_dropout != 'none' and not args.lewm_only:
        raise ValueError('warm-start measurement dropout requires frozen Actor/FCPG with --lewm-only')
    if args.fish_fg_table is not None and (
            not args.scene_latent_lewm or not args.lewm_only or args.from_scratch or args.resume
            or args.reset_scene_branch or args.scene_supervision != 'dense40'
            or args.scene_normalization != 'layernorm' or args.precision != 'bf16'
            or args.scene_sigreg_weight != .02 or args.scene_sigreg_projections != 8192
            or args.fish_train_episodes != [0, 1]):
        raise ValueError('F/G spatial experiment requires fresh BF16 LN dense40 LeWM-only, SIGReg0.02/8192, fish0/1 with3 held out')
    if not np.isfinite(args.scene_fg_weight) or args.scene_fg_weight <= 0:
        raise ValueError('F/G spatial weight must be finite and positive')
    validate_q_grounding_options(args)
    if args.reset_scene_branch and (
            not args.lewm_mixed_only or not args.lewm_only or not args.scene_latent_lewm
            or args.from_scratch or args.reset_scene_head or args.resume
            or args.lewm_archive_only_diagnostic or args.scene_normalization != 'layernorm'
            or args.scene_prediction_mode != ABSOLUTE_SCENE):
        raise ValueError('scene-branch reset requires a fresh bounded absolute LN mixed-only parent experiment')
    # 17.125: the same diagnostic is also the full-patch fish stage with Archive rows in place of the fish (checked below)
    full_patch_diagnostic = args.lewm_archive_only_diagnostic and args.full_patch_lewm and not args.scene_latent_lewm
    if args.lewm_archive_only_diagnostic and not full_patch_diagnostic and (
            not args.lewm_mixed_only or not args.lewm_only or not args.scene_latent_lewm
            or args.from_scratch or args.reset_scene_head or args.resume
            or args.scene_prediction_mode != ABSOLUTE_SCENE):
        raise ValueError('archive-only diagnostic requires a fresh bounded scene-parent run without head reset')
    if args.lewm_mixed_only and (not args.lewm_only or not args.scene_latent_lewm):
        raise ValueError('mixed-only clock repair requires scene LeWM and frozen Actor/FCPG')
    if bool(args.locked_development_report) != bool(args.clock_baseline_report):
        raise ValueError('locked development windows and corrected baseline are required together')
    if args.locked_development_report is not None and args.feature_bank_gib <= 0:
        raise ValueError('locked clock evaluation requires the bounded feature bank')
    if args.resume_with_resources and not args.resume:
        raise ValueError('resource continuation requires --resume')
    if args.resume_with_precision and not args.resume:
        raise ValueError('precision continuation requires --resume')
    if args.resume_with_precision and args.resume_with_resources:
        raise ValueError('change precision or resources separately to preserve continuation provenance')
    if args.from_scratch and args.cache_frozen_actor_inputs:
        raise ValueError('scratch FCPG changes every Actor update; only raw DINO features may be cached')
    if (args.from_stage_checkpoint is None) != (args.fish_stage_lewm is None):
        raise ValueError('--from-stage-checkpoint and --fish-stage-lewm are given together')
    if args.camera_layout is not None and not args.lewm_only:
        raise ValueError('--camera-layout is for LeWM-only runs; the Actor trains with scripts/train_v21_fcpg_actor.py')
    if args.from_stage_checkpoint is not None and (not args.from_scratch or args.resume or args.lewm_only
                                                   or args.full_patch_lewm or args.scene_latent_lewm):
        raise ValueError('a fish stage from a stage checkpoint is a fresh from-scratch native40 run, nothing else')
    if args.scene_latent_lewm and (args.from_scratch or not args.lewm_only or args.full_patch_lewm):
        raise ValueError('learned-scene run requires a full-patch --parent and --lewm-only, not random initialization')
    if not args.scene_latent_lewm and args.scene_normalization != 'batchnorm':
        raise ValueError('--scene-normalization requires --scene-latent-lewm')
    if not args.scene_latent_lewm and (args.reset_scene_head or args.scene_prediction_mode != ABSOLUTE_SCENE):
        raise ValueError('scene output changes require --scene-latent-lewm')
    if args.paired_schedule_audit and (not args.lewm_mixed_only or args.resume):
        raise ValueError('paired schedule audit requires a fresh mixed-only experiment')
    if args.lewm_command_conditioning != LEGACY_COMMANDS and not (args.scene_latent_lewm or args.full_patch_lewm):
        raise ValueError('--lewm-command-conditioning requires a full-patch or scene experiment')
    if not np.isfinite(args.scene_joint_weight) or args.scene_joint_weight <= 0:
        raise ValueError('joint auxiliary coefficient must be finite and positive')
    if not np.isfinite(args.scene_sigreg_weight) or args.scene_sigreg_weight <= 0:
        raise ValueError('scene SIGReg coefficient must be finite and positive')
    if not args.scene_latent_lewm and args.scene_sigreg_weight != .02:
        raise ValueError('--scene-sigreg-weight requires --scene-latent-lewm')
    if not args.scene_latent_lewm and (args.scene_supervision != 'sparse8' or args.scene_sigreg_projections != 1024):
        raise ValueError('scene sampling requires --scene-latent-lewm')
    if args.scene_supervision == 'dense40' and args.scene_normalization != 'layernorm':
        raise ValueError('dense40 requires per-observation layernorm; BN would change shared target encodings')
    if (not 1 <= args.batch_size <= 512
            or (args.lewm_batch_size is not None and not 1 <= args.lewm_batch_size <= 512)
            or not 2 <= args.fish_batch_size <= 256
            or args.fish_batch_size % 2 or not 1 <= args.archive_passes <= 4
            or not 1 <= args.lewm_fish_updates <= 4000 or not 1 <= args.actor_fish_updates <= 8000
            or not 1 <= args.threads <= 8 or not 0 < args.learning_rate <= 1e-3      # 17.172: 1e-4 before bigger batches
            or (args.fish_learning_rate is not None and not 0 < args.fish_learning_rate <= 1e-3)
            or args.checkpoint_every < 1 or args.development_rows < 1):
        raise ValueError('invalid horizon training schedule')
    if (args.lewm_width, args.lewm_layers) != (384, 4) and (
            not args.full_patch_lewm or args.scene_latent_lewm or args.from_stage_checkpoint is not None):
        raise ValueError('--lewm-width / --lewm-layers size a new --full-patch-lewm only')
    if not 0 <= args.lewm_patch_encoder_layers <= 8 or (args.lewm_patch_encoder_layers and (
            not args.full_patch_lewm or args.scene_latent_lewm or args.from_stage_checkpoint is not None)):
        raise ValueError('--lewm-patch-encoder-layers (0-8) builds the picture encoder of a new --full-patch-lewm only')
    if args.lewm_concat_inputs and not (args.lewm_memory_pictures and args.lewm_controller and args.lewm_balanced_inputs):
        raise ValueError('--lewm-concat-inputs builds a picture-memory LeWM with --lewm-controller and '
                         '--lewm-balanced-inputs')
    if args.lewm_pose_position and not args.lewm_concat_inputs:
        raise ValueError('--lewm-pose-position joins the bank of a --lewm-concat-inputs LeWM')
    if args.robot_memory_pack is not None and (not args.lewm_memory_pictures or args.robot_block_pack is None):
        raise ValueError('--robot-memory-pack gives the robot blocks of --robot-block-pack to a picture-memory LeWM')
    if args.lewm_memory_pictures and (
            not 3 <= args.lewm_memory_pictures <= 64 or not 1 <= args.lewm_memory_tokens <= 32
            or not args.full_patch_lewm or args.scene_latent_lewm or args.from_stage_checkpoint is not None
            or not args.lewm_only or args.lewm_patch_encoder_layers
            or args.lewm_attention_weight or args.lewm_label_head_weight
            or not all(1 <= s <= 4 for s in args.lewm_memory_strides) or not 1 <= args.lewm_memory_eval_stride <= 4
            or not 0 <= args.lewm_memory_drop <= 1 or not 0 < args.lewm_memory_drop_seconds <= 30):
        raise ValueError('--lewm-memory-pictures (3-64) builds the picture memory of a new --full-patch-lewm --lewm-only '
                         'without a patch encoder or SAM losses; strides 1-4, drop 0-1, 0-30 s')
    if not (np.isfinite(args.lewm_sigreg_weight) and 0 < args.lewm_sigreg_weight <= 1) or (
            args.lewm_sigreg_weight != .02 and (not args.full_patch_lewm or args.scene_latent_lewm)):
        raise ValueError('--lewm-sigreg-weight (0, 1] weights the SIGReg of a new --full-patch-lewm only')
    if not 0 <= args.window_readers <= 8 or (args.window_readers and (
            not args.lewm_only or args.from_scratch or args.scene_latent_lewm or args.device != 'cuda:0'
            or not args.feature_bank_gib or args.feature_bank_device != 'training' or args.paired_schedule_audit)):
        raise ValueError('--window-readers (1-8) reads the Archive windows of a LeWM-only run on the GPU')
    if not (np.isfinite(args.carry_forecast_weight) and args.carry_forecast_weight >= 0):
        raise ValueError('--carry-forecast-weight must be finite and at least 0')
    if not (np.isfinite(args.carry_place_factor) and args.carry_place_factor >= 1) or (
            args.carry_place_factor != 1 and not args.carry_forecast_weight):
        raise ValueError('--carry-place-factor (at least 1) weights --carry-forecast-weight')
    if args.object_cues is not None and (args.camera_layout is None or not args.lewm_only):
        raise ValueError('--object-cues needs --camera-layout (a LeWM-only run)')
    if args.carry_forecast_weight and (args.object_cues is None or not args.full_patch_lewm or args.scene_latent_lewm):
        raise ValueError('--carry-forecast-weight trains a new --full-patch-lewm and finds held rows by --object-cues')
    if args.lewm_archive_from is not None and (
            not args.lewm_only or not args.full_patch_lewm or args.scene_latent_lewm or args.lewm_mixed_only
            or args.from_scratch or args.resume or args.from_stage_checkpoint is not None):
        raise ValueError('--lewm-archive-from is for a fresh --lewm-only --full-patch-lewm run')
    if args.lewm_fish_from_archive is not None and (
            not args.lewm_only or not args.full_patch_lewm or args.scene_latent_lewm or args.lewm_mixed_only
            or args.from_scratch or args.resume or args.from_stage_checkpoint is not None
            or args.lewm_archive_only_diagnostic or args.archive_batch_sampling != 'iid-v1'):
        raise ValueError('--lewm-fish-from-archive is for a fresh --lewm-only --full-patch-lewm run with iid Archive '
                         'batches')
    if full_patch_diagnostic and (args.lewm_archive_from is None or args.lewm_fish_demos_only
                                  or args.lewm_fish_start is not None):
        raise ValueError('the full-patch archive-only diagnostic fine-tunes the whole LeWM of a restored Archive '
                         'stage (--lewm-archive-from), without a start or --lewm-fish-demos-only')
    if args.lewm_fish_start is not None and args.lewm_archive_from is None:
        raise ValueError('--lewm-fish-start continues a restored Archive stage (--lewm-archive-from)')
    if args.lewm_fish_demos_only and (
            not args.lewm_only or not args.full_patch_lewm or args.scene_latent_lewm or args.lewm_mixed_only
            or args.from_scratch or args.resume or args.from_stage_checkpoint is not None
            or args.archive_batch_sampling != 'iid-v1'):
        raise ValueError('--lewm-fish-demos-only is for a fresh --lewm-only --full-patch-lewm run with iid Archive '
                         'batches')
    if args.fish_stage_seed is not None and (args.lewm_archive_from is None or not 0 <= args.fish_stage_seed < 2**31):
        raise ValueError('--fish-stage-seed (0 to 2**31 - 1) draws again the fish stage of a run that restores an '
                         'Archive stage (--lewm-archive-from)')
    if args.robot_block_pack is not None and (
            not args.lewm_only or not args.full_patch_lewm or args.scene_latent_lewm
            or not 1 <= args.robot_block_rows <= 64 or not 0 < args.robot_block_weight <= 1):
        raise ValueError('--robot-block-pack is for a --lewm-only --full-patch-lewm run (1-64 rows, weight in (0, 1])')
    if args.robot_blocks_fish_only and (args.robot_block_pack is None or args.lewm_fish_demos_only):
        raise ValueError('--robot-blocks-fish-only needs --robot-block-pack and a fish stage that reads them')
    if args.lewm_controller and (
            not args.lewm_only or not args.full_patch_lewm or args.scene_latent_lewm
            or args.from_scratch or args.resume or args.from_stage_checkpoint is not None
            or args.lewm_archive_from is not None or args.lewm_fish_start is not None
            or args.lewm_fish_demos_only or args.lewm_mixed_only
            or (args.robot_block_pack is not None) != (args.robot_arm_p is not None)):
        raise ValueError('--lewm-controller trains a new --lewm-only --full-patch-lewm from its parent (no restored '
                         'stage, start or resume); robot blocks need their --robot-arm-p table')
    if args.robot_arm_p is not None and not args.lewm_controller:
        raise ValueError('--robot-arm-p is read by --lewm-controller only')
    if args.lewm_balanced_inputs and (
            not args.full_patch_lewm or args.scene_latent_lewm or args.from_scratch or args.resume
            or args.from_stage_checkpoint is not None or args.lewm_archive_from is not None
            or args.lewm_fish_start is not None):
        raise ValueError('--lewm-balanced-inputs builds a new --full-patch-lewm (no restored stage, start or resume)')
    if args.lewm_command_symlog and not args.lewm_balanced_inputs:
        raise ValueError('--lewm-command-symlog shapes the command distance of --lewm-balanced-inputs')
    if (args.lewm_attention_weight or args.lewm_label_head_weight) and (
            args.lewm_sam_labels is None or not args.full_patch_lewm or not args.lewm_only or args.scene_latent_lewm
            or args.from_scratch or args.resume or args.from_stage_checkpoint is not None
            or args.lewm_archive_from is not None or args.lewm_fish_start is not None or args.camera_layout is None
            or (not (args.lewm_controller or args.lewm_balanced_inputs)
                and (args.lewm_width, args.lewm_layers) == (384, 4))
            or args.lewm_attention_weight < 0 or args.lewm_label_head_weight < 0
            or not 1 <= args.lewm_attention_heads <= 8):
        raise ValueError('--lewm-attention-weight / --lewm-label-head-weight (>= 0; 1-8 heads) train a new --lewm-only '
                         '--full-patch-lewm that attends directly, from its parent, with --lewm-sam-labels and '
                         '--camera-layout')
    if args.lewm_sam_labels is not None and not (args.lewm_attention_weight or args.lewm_label_head_weight):
        raise ValueError('--lewm-sam-labels is read by --lewm-attention-weight or --lewm-label-head-weight')
    if args.lewm_archive_demo_rows and (
            not args.lewm_only or not args.full_patch_lewm or args.scene_latent_lewm or args.lewm_mixed_only
            or args.lewm_archive_from is not None or args.lewm_archive_only_diagnostic
            or not 1 <= args.lewm_archive_demo_rows <= 64 or not 0 < args.lewm_archive_demo_weight <= 1):
        raise ValueError('--lewm-archive-demo-rows (1-64, weight in (0, 1]) is for a --lewm-only --full-patch-lewm run '
                         'that trains its own Archive stage')
    if args.output.exists() and not args.resume:
        raise FileExistsError('use a new version directory or explicitly resume the same protocol')
    config = json.loads(args.source_config.read_text())
    population_contract = config.get(POPULATION_CONTRACT_KEY)
    validate_population_options(args, population_contract)
    if args.resume:
        # Reject a changed training objective before touching old run artifacts.
        previous = json.loads((args.output/'protocol.json').read_text())
        validate_population_resume(previous, population_contract)
        validate_training_semantics(previous,
                                    full_patch_lewm=args.full_patch_lewm,
                                    scene_latent_lewm=args.scene_latent_lewm,
                                    scene_normalization=args.scene_normalization,
                                    scene_prediction_mode=args.scene_prediction_mode,
                                    scene_sigreg_weight=args.scene_sigreg_weight,
                                    scene_supervision=args.scene_supervision,
                                    scene_sigreg_projections=args.scene_sigreg_projections,
                                    deterministic_training=args.deterministic_training,
                                    measurement_dropout=args.measurement_dropout,
                                    scene_joint_gradient_routing=args.scene_joint_gradient_routing,
                                    archive_batch_sampling=args.archive_batch_sampling,
                                    archive_q_weight=args.archive_q_weight,
                                    command_conditioning=args.lewm_command_conditioning)
    configure_training_determinism(args.deterministic_training)
    torch.set_num_threads(args.threads)
    torch.backends.mha.set_fastpath_enabled(False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device, device_info = configure_device(args.device, args.gpu_memory_limit_gib)
    if args.precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 training requires a CUDA GPU with native BF16 support')
    if device.type == 'cuda' and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_info['tf32_enabled'] = True
    if args.fused_optimizer and device.type != 'cuda':
        raise ValueError('fused AdamW requires CUDA')
    source = V21Sources(config['source_paths'], device='cpu', object_relative_table=config['body_table'])
    camera_mode = REVIEWED_TYPES if source.camera_types is not None else LEGACY_SLOTS
    if camera_mode == REVIEWED_TYPES and (
            not args.lewm_only or not args.scene_latent_lewm or args.from_scratch
            or args.archive_batch_sampling != 'iid-v1' or args.reset_scene_head or args.reset_scene_branch):
        raise ValueError('camera type experiment keeps Actor/FCPG frozen, IID sampling and parent scene heads')
    splits, population_report = prepare_training_splits(source, config, args.fish_train_episodes)
    camera_layout = apply_camera_layout(source, splits, args.camera_layout) if args.camera_layout is not None else None
    memory_lookback = 0         # 17.166: endpoints before a row's first one its picture memories can reach
    if args.lewm_memory_pictures:
        from fcpg.v21_picture_memory import lookback_endpoints
        memory_lookback = lookback_endpoints(args.lewm_memory_pictures, args.lewm_memory_strides,
                                             args.lewm_memory_drop_seconds, args.lewm_memory_eval_stride)
    sam_labels = None           # 17.163: SAM 3.1 labels of the Archive pictures, on the DINO cache's own rows
    if args.lewm_sam_labels is not None:
        from fcpg.v21_sam_labels import SamLabelTable
        sam_labels = SamLabelTable(args.lewm_sam_labels, source.archive_dino)
    label_box, sam_step = {}, {}
    label_counts = dict(rows=0, rows_with_target=0, bank_pictures=0, labelled_bank_pictures=0)
    task_pool = (load_task_pool(args.lewm_fish_from_archive, splits, source.archive)
                 if args.lewm_fish_from_archive is not None else None)
    fish_start = (load_fish_start(args.lewm_fish_start, args.lewm_archive_from)
                  if args.lewm_fish_start is not None else None)
    # section 17.120: the Archive rows that hold something (the 17.117 grasp cue) and the jaws' height kinematics
    cue_manifest, carry_gate = None, None
    if args.object_cues is not None:
        from fcpg.v21_outline_cue import attach_archive_cues
        cue_manifest = attach_archive_cues(source, source.archive_camera_roles, args.object_cues)
    if args.carry_forecast_weight:
        from fcpg.so101_kinematics import SO101MuJoCoKinematics
        from scripts.serve_so101_prod_v17_7 import DEFAULT_KINEMATICS_MODEL, TABLE_Z_MM, V177ClearanceGate
        carry_gate = V177ClearanceGate(kinematics=SO101MuJoCoKinematics(DEFAULT_KINEMATICS_MODEL), table_z_mm=TABLE_Z_MM,
                                       planned_clearance_mm=-10., readback_clearance_mm=-10., physical_floor_mm=-10.)
    carry_box, carry_counts = {}, dict(rows=0, held=0, held_seeing_place=0)
    if args.resume:
        validate_population_resume(previous, population_contract, report=population_report)
    parent_q = None
    vocabulary_note = None
    if args.from_scratch:
        vocabulary = source.vocabulary
        scale = training_loss_scale(source, splits['train']['archive'])
        core = initialize_scratch40(vocabulary, scale, args.seed, device).eval()
        parent = dict(contract=CONTRACT, position_unit=POSITION_UNIT,
                      vocabulary=list(vocabulary.words), text_max_length=vocabulary.max_length,
                      initialization='random-all-three-components-native40', external_weights_loaded=False)
        if args.from_stage_checkpoint is not None:
            saved = torch.load(args.from_stage_checkpoint, map_location='cpu', weights_only=False)
            if (saved.get('complete') is not True or saved.get('smoke') is not False
                    or saved.get('stage') not in ('actor_archive', 'actor_dense')
                    or list(saved.get('vocabulary') or []) != list(vocabulary.words)):
                raise ValueError('stage checkpoint must be a completed formal actor_archive/actor_dense stage '
                                 'with the same vocabulary')
            if saved.get('temporal_training_contract') != getattr(core, 'temporal_training_contract', None):
                raise ValueError('stage checkpoint temporal training policy differs')
            # reader, Actor and loss scale of that run; its joint-only LeWM is not rebuilt (see --fish-stage-lewm)
            missing, unexpected = core.load_state_dict(
                {k: v for k, v in saved['model'].items() if not k.startswith('lewm.')}, strict=False)
            if unexpected or any(not key.startswith('lewm.') for key in missing):
                raise ValueError('stage checkpoint reader/Actor do not match this core')
            donor = torch.load(args.fish_stage_lewm, map_location='cpu', weights_only=False)
            lewm_state = {k[len('lewm.'):]: v for k, v in donor['model'].items() if k.startswith('lewm.')}
            if any(key.startswith(('film.', 'concat_inputs')) for key in lewm_state):
                from fcpg.v21_full_patch_lewm import V21FullPatchLeWM, architecture_from_state
                core.lewm = V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(),
                                             horizon=40, **architecture_from_state(lewm_state)).to(device)
            core.lewm.load_state_dict(lewm_state)
            del donor, lewm_state
            if saved.get('temporal_training_contract') is not None:
                core.temporal_feature_coverage = deepcopy(saved['temporal_feature_coverage'])
            parent.update(initialization='native40-stage-checkpoint-then-fish', external_weights_loaded=True)
            del saved
    else:
        parent = torch.load(args.parent, map_location='cpu', weights_only=False)
        if parent.get('complete') is not True or parent.get('smoke') is not False or parent.get('stage') != 'fish':
            raise ValueError('native40 requires a completed formal parent')
        if args.scene_latent_lewm:
            core = core_from_parent(parent).to(device).eval()
            parent_fg = fg_checkpoint_mode(parent)
            if parent_fg is not None and args.fish_fg_table is None:
                raise ValueError('spatial parent requires explicit F/G table and objective')
            parent_q = q_checkpoint_mode(parent)
            if parent_q is not None and not args.archive_q_weight:
                raise ValueError('q-readout parent requires the explicit Archive q objective')
            from fcpg.v21_full_patch_lewm import V21FullPatchLeWM
            if not isinstance(core.lewm, V21FullPatchLeWM) or core.actor.horizon != 40:
                raise ValueError('scene warm start requires trained native40 full-patch LeWM')
            if (args.reset_scene_head or args.reset_scene_branch) and not isinstance(core.lewm, V21SceneLeWM):
                raise ValueError('scene head/branch reset requires a trained scene parent')
            if (isinstance(core.lewm, V21SceneLeWM)
                    and core.lewm.scene_normalization != args.scene_normalization):
                raise ValueError('scene parent normalization differs; start the ablation from the original full-patch q parent')
            if (isinstance(core.lewm, V21SceneLeWM)
                    and core.lewm.scene_prediction_mode != args.scene_prediction_mode
                    and not args.reset_scene_head):
                raise ValueError('changing scene prediction mode requires explicit --reset-scene-head')
            replacement = V21SceneLeWM(increment_scale=core.lewm.increment_scale.detach().cpu(),
                                       normalization=args.scene_normalization,
                                       scene_prediction_mode=args.scene_prediction_mode,
                                       command_conditioning=args.lewm_command_conditioning).to(device)
            # Restore a typed parent before strict matching; a legacy parent is
            # expanded only after every old tensor has been restored below.
            replacement.enable_camera_conditioning(camera_mode_from_checkpoint(parent))
            if args.fish_fg_table is not None:
                enable_fg_readout(replacement)
            if args.archive_q_weight:
                # Fresh head from its own seed; the training RNG streams stay
                # bit-identical to a control run without the q objective.
                initialization_rng = paired_rng_evidence(device)
                enable_q_readout(replacement, seed=args.seed + Q_READOUT_SEED_OFFSET)
                if paired_rng_evidence(device) != initialization_rng:
                    raise ValueError('q readout initialization consumed the CPU/CUDA training RNG')
            missing, unexpected = replacement.load_state_dict(core.lewm.state_dict(), strict=False)
            allowed = ('scene_encoder.', 'scene_context.', 'scene_prediction.')
            if args.fish_fg_table is not None:
                allowed += ('fg_readout.',)
            if args.archive_q_weight:
                allowed += ('q_readout.',)
            if args.lewm_command_conditioning == FULL_COMMANDS:
                allowed += ('command_projection.', 'command_readers.')
            if unexpected or any(not key.startswith(allowed) for key in missing):
                raise ValueError('scene warm start failed to preserve the complete q predictor')
            new_head = lambda key: ((args.fish_fg_table is not None and parent_fg is None and key.startswith('fg_readout.'))
                                    or (args.archive_q_weight and parent_q is None and key.startswith('q_readout.')))
            if (isinstance(core.lewm, V21SceneLeWM)
                    and core.lewm.command_conditioning == args.lewm_command_conditioning
                    and any(not new_head(key) for key in missing)):
                raise ValueError('same-architecture scene parent must restore every learned parameter')
            if args.reset_scene_head:
                reset_scene_output(replacement)
            if args.reset_scene_branch:
                initialization_rng = paired_rng_evidence(device)
                reset_scene_branch(replacement, seed=args.seed + 20001)
                if paired_rng_evidence(device) != initialization_rng:
                    raise ValueError('scene-branch reset consumed the CPU/CUDA training RNG')
            core.lewm = replacement
        elif args.full_patch_lewm:
            # a DINOv3-L process (V21_ENCODER) builds the frozen reader / Actor for its features; only parameters
            # of the same shape come from the DINOv2 parent below, and the full-patch LeWM is new either way
            core = V21ResearchCore(V21Config(**{**parent['config'], 'encoder': ENCODER})).to(device).eval()
            if core.actor.query.shape[1] == 5:
                core = initialize_native40(core)
            current = core.state_dict()
            matched = {k: v for k, v in parent['model'].items()
                       if k in current and tuple(current[k].shape) == tuple(v.shape) and not k.startswith('lewm.')}
            core.load_state_dict(matched, strict=False)
            from fcpg.v21_full_patch_lewm import V21FullPatchLeWM
            core.lewm = V21FullPatchLeWM(
                increment_scale=core.actor.increment_scale.detach().cpu(), horizon=40,
                command_conditioning=args.lewm_command_conditioning,
                width=args.lewm_width, layers=args.lewm_layers,
                controller_input=args.lewm_controller, balanced_inputs=args.lewm_balanced_inputs,
                command_symlog=args.lewm_command_symlog, label_head=args.lewm_label_head_weight > 0,
                patch_encoder_layers=args.lewm_patch_encoder_layers, memory_pictures=args.lewm_memory_pictures,
                memory_tokens=args.lewm_memory_tokens, concat_inputs=args.lewm_concat_inputs,
                pose_position=args.lewm_pose_position).to(device)
        else:
            core = core_from_parent(parent).to(device).eval()
            if core.actor.query.shape[1] == 5:
                core = initialize_native40(core)
            elif core.actor.query.shape[1] != 40 or core.lewm.query.shape[1] not in (5, 40):
                raise ValueError('parent Actor/LeWM query horizon is not 5 or 40')
        vocabulary = TaskVocabulary(tuple(parent['vocabulary']), parent['text_max_length'])
        if source.vocabulary != vocabulary:
            # 17.90: an Archive whose added recordings bring task words the parent never saw. The full-patch LeWM
            # reads no text, and with --lewm-only the parent's reader and Actor stay frozen, so the sources read task
            # text with the parent's words (each new word becomes <unk>); anything else still refuses.
            parent_words, source_words = set(vocabulary.words), set(source.vocabulary.words)
            if not (args.lewm_only and args.full_patch_lewm) or not parent_words <= source_words:
                raise ValueError('formal vocabulary and source task identities differ')
            vocabulary.encode([*source.archive.task_names, source.fish_task])     # every task text still fits
            vocabulary_note = dict(rule='17.90: parent vocabulary; task words the parent lacks read as <unk> '
                                        '(LeWM-only, full-patch LeWM reads no text, reader and Actor frozen)',
                                   parent_words=len(parent_words), source_words=len(source_words),
                                   words_read_as_unk=len(source_words - parent_words))
            source.vocabulary = vocabulary
    if hasattr(core.lewm, 'enable_camera_conditioning'):
        core.lewm.enable_camera_conditioning(camera_mode)
    elif camera_mode != LEGACY_SLOTS:
        raise ValueError('camera type metadata requires a full-patch scene model')
    frozen_reader_sha = state_hash(core.state_dict(), 'reader.')
    initial_hashes = {name: state_hash(core.state_dict(), name+'.') for name in ('reader', 'actor', 'lewm')}
    robot_blocks = None
    if args.robot_block_pack is not None:
        # 10/31: the robot's own completed blocks with their pictures (fcpg/v21_robot_block_pack.py)
        from fcpg.v21_robot_block_pack import RobotBlocks
        robot_blocks = RobotBlocks(torch.load(args.robot_block_pack, map_location='cpu', weights_only=False), device,
                                   None, arm_p=None if args.robot_arm_p is None else json.loads(args.robot_arm_p.read_text()))
    robot_memory = None
    if args.robot_memory_pack is not None:
        # 17.166: those blocks' picture memories, from their sessions' streams (build_robot_memory_packs_20261101.py)
        from fcpg.v21_picture_memory import RobotPictureMemory
        index = torch.load(args.robot_memory_pack, map_location='cpu', weights_only=False)
        if (robot_blocks is None or index.get('split') != 'train' or index.get('blocks') != robot_blocks.size
                or index.get('slots', 0) < args.lewm_memory_pictures):
            raise ValueError('--robot-memory-pack must be the train split of --robot-block-pack, holding '
                             '--lewm-memory-pictures pictures per camera')
        robot_memory = RobotPictureMemory(index, device)
        del index
    if args.lewm_archive_only_diagnostic and not full_patch_diagnostic:
        # the scene run starts from the parent's own tensors; the full-patch run (17.125) has a new LeWM that the
        # restored Archive stage replaces whole (loaded strictly after the recipe check)
        current = core.state_dict()
        if current.keys() != parent['model'].keys() or any(
                not torch.equal(value.detach().cpu(), parent['model'][key].cpu())
                for key, value in current.items()):
            raise ValueError('archive-only diagnostic must preserve every original parent tensor')
        del current
    bank_device = torch.device('cpu') if args.feature_bank_device == 'cpu' else device
    feature_bank = ScratchDinoFeatureBank(source, bank_device, int(args.feature_bank_gib * 1024**3)) if args.feature_bank_gib else None
    staged = DeviceStagedSources(source, device, feature_bank)
    future_scenes = (FutureSceneTargets(source, feature_bank, supervision=args.scene_supervision)
                     if args.scene_latent_lewm else None)
    evaluation_scenes = FutureSceneTargets(source, feature_bank) if args.scene_latent_lewm else None
    fg_table = None
    if args.fish_fg_table is not None:
        from fcpg.v21_production_fg_labels import ProductionFGTable
        fg_table = ProductionFGTable(args.fish_fg_table,
            clock_version=source.paths.get('dino_clock_version'), extraction_id=source.paths.get('dino_extraction_id'))
        if fg_table.manifest.get('smoke'):
            raise ValueError('partial smoke F/G table cannot supervise training')
        expected_ids = np.flatnonzero(np.isin(source.fish.episodes, [0,1,3]))
        if not np.array_equal(fg_table.ids, expected_ids):
            raise ValueError('F/G table does not cover every admitted fish frame')
        if not args.canary:
            fg_table.require_training_admission()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reset_scene_branch:
        current = core.state_dict()
        allowed = tuple('lewm.' + prefix for prefix in SCENE_BRANCH_PREFIXES)
        if current.keys() != parent['model'].keys():
            raise ValueError('scene-branch reset changed the parameter set')
        changed = [key for key, value in current.items()
                   if not torch.equal(value.detach().cpu(), parent['model'][key].cpu())]
        if (not changed or any(not key.startswith(allowed) for key in changed)
                or any(not any(key.startswith(prefix) for key in changed) for prefix in allowed)):
            raise ValueError('scene-branch reset did not change exactly the three allowed modules')
        if (torch.count_nonzero(core.lewm.scene_context.weight)
                or not torch.count_nonzero(core.lewm.scene_prediction.weight)):
            raise ValueError('scene-branch context/output initialization differs')
        write_json(args.output/'scene-branch-initialization.json', dict(
            schema=SCENE_BRANCH_RESET, seed=args.seed + 20001, parent=str(args.parent),
            changed_parameters=changed, allowed_prefixes=list(allowed),
            all_other_parent_tensors_equal=True, context_weight_zero=True,
            output_fresh_default=True, global_rng_preserved_by_initializer=True))
        del current
    if args.reset_scene_head:
        current = core.state_dict()
        if current.keys() != parent['model'].keys():
            raise ValueError('scene-head ablation changed the parent parameter set')
        changed = [key for key, value in current.items()
                   if not torch.equal(value.detach().cpu(), parent['model'][key].cpu())]
        if set(changed) - {'lewm.scene_prediction.weight', 'lewm.scene_prediction.bias'}:
            raise ValueError('scene-head ablation changed parameters outside the output layer')
        if any(torch.count_nonzero(value) for value in core.lewm.scene_prediction.parameters()):
            raise ValueError('scene-head ablation did not start with an exactly zero output layer')
        write_json(args.output/'scene-head-initialization.json', dict(
            parent=str(args.parent), prediction_mode=args.scene_prediction_mode,
            changed_parameters=changed, all_other_parent_parameters_equal=True,
            output_weight_and_bias_zero=True))
        del current
    readiness = {}
    for role in ('train', 'development'):
        readiness[role] = {}
        for kind in ('archive', 'fish'):
            rows = splits[role][kind]
            labelled = [horizon_metadata(source, rows, kind, h)[-1] if len(rows) else None for h in range(4)]
            readiness[role][kind] = dict(original_chains=len(rows),
                full40_by_context=[int(v.all(1).sum()) if v is not None else 0 for v in labelled],
                supervised_by_context=[int(v[:, 0].sum()) if v is not None else 0 for v in labelled])
            if population_report is not None:
                readiness[role][kind].update(
                    original_chains=population_report['original_rows'][role][kind],
                    eligible_chains=len(rows))
    write_json(args.output/'data-readiness.json', dict(readiness,
        original_features_reused=source.recorded_clock is None,
        corrected_cache_generation=None if source.recorded_clock is None else source.recorded_clock.metadata['extraction_id'],
        new_recordings_required=False, test_episode_2_values_read=False))
    if future_scenes is not None:
        scene_readiness = {}
        for role in ('train', 'development'):
            scene_readiness[role] = {}
            for kind in ('archive', 'fish'):
                rows = splits[role][kind]
                rows = rows[::max(1, len(rows)//2048)][:2048]
                scene_readiness[role][kind] = dict(sampled_chains=len(rows),
                    exact_cached_frames_by_context=[(future_scenes.identities(rows, kind, h) >= 0).sum(0).tolist()
                                                    if len(rows) else [0]*len(future_scenes.query_indices) for h in range(4)])
        write_json(args.output/'scene-data-readiness.json', dict(
            query_indices_one_based=(future_scenes.query_indices+1).tolist(), counts_before_per_view_validity=scene_readiness,
            missing_frames_masked=True, future_images_in_input=False,
            test_episode_2_values_read=False))
    schedule = {k: v for k, v in vars(args).items()
                if k not in ('resume', 'resume_with_resources', 'resume_with_precision')}
    schedule = {k: str(v) if isinstance(v, Path) else v for k, v in schedule.items()}
    protocol = dict(schema=SCHEMA, version='21.0.4', prediction_horizon=40, execution_horizon=5,
        parent_checkpoint=str(args.parent) if args.parent else None, parent_sha256=sha(args.parent) if args.parent else None,
        parent_actor_sha256=state_hash(parent['model'], 'actor.') if args.parent else None,
        frozen_reader_sha256=None if args.from_scratch else frozen_reader_sha,
        initialization=('trained-full-patch-q-plus-new-scene-heads' if args.scene_latent_lewm else
                        'random-all-three-components-native40' if args.from_scratch else 'formal-parent-native40-expansion'),
        external_weights_loaded=not args.from_scratch, initial_module_sha256=initial_hashes,
        vocabulary_note=vocabulary_note,
        original_source_config=str(args.source_config), source_config_sha256=sha(args.source_config),
        body_table_sha256=sha(config['body_table']), response_pack_sha256=sha(args.response_pack) if args.response_pack else None,
        lewm_controller=None if not args.lewm_controller else dict(
            section='17.153', inputs=['arm_p32', 'planner'], archive=[0, 0], demonstrations=[0, 0],
            response_pack=[0, 1], robot_blocks='[P 32 in their session, 1]',
            robot_arm_p=None if args.robot_arm_p is None else dict(path=str(args.robot_arm_p), sha256=sha(args.robot_arm_p)),
            robot_blocks_p32=None if robot_blocks is None else int(robot_blocks.data['arm_p32'].sum())),
        lewm_archive_demos=None if not args.lewm_archive_demo_rows else dict(
            section='17.154', rows_per_update=args.lewm_archive_demo_rows, weight=args.lewm_archive_demo_weight,
            rows="demonstrations 0 and 1 (the fish stage's TRAIN rows) at the update's context, with replacement",
            generator='numpy seed + 960', controller=[0, 0],
            reason='the fish table has LeRobot recordings beside the robot blocks from the first update'),
        robot_block_pack=None if robot_blocks is None else dict(
            robot_blocks.record, path=str(args.robot_block_pack), sha256=sha(args.robot_block_pack),
            rows_per_update=args.robot_block_rows, weight=args.robot_block_weight,
            stages=['lewm_fish'] if args.robot_blocks_fish_only else ['lewm_archive', 'lewm_fish'],
            sigreg='as the response pack: the LeWM SIGReg of their hidden states'),
        train_rows={k: len(v) for k, v in splits['train'].items()},
        development_rows={k: len(v) for k, v in splits['development'].items()},
        fish_train_episodes=args.fish_train_episodes,
        fish_development_episodes=[] if 3 in args.fish_train_episodes else [3],
        fish_split_change='user requested all three available demonstrations for fitting' if 3 in args.fish_train_episodes else None,
        test_episode_2_values_read=False,
        targets='same-episode recorded commands and actual readbacks; unobserved tails masked',
        sample_clock=('nearest original frame to 10 Hz; offsets use declared source fps' if source.recorded_clock is None
                      else 'nearest real recorded timestamp to 10 Hz; camera exposure uses decoded PTS'),
        corrected_cache_generation=None if source.recorded_clock is None else source.recorded_clock.metadata,
        archive_camera_role_adapter=source.paths.get('archive_camera_role_adapter'),
        archive_batch_sampling=args.archive_batch_sampling,
        command_traces='all recorded writes before final supervised readback, up to 240 at 60 Hz',
        lewm_command_conditioning=getattr(core.lewm, 'command_conditioning', LEGACY_COMMANDS),
        visual_features=('original frozen DINO cache; only factual prefix exposures' if source.recorded_clock is None
                         else 'rebuilt frozen DINO with actual capture times/q and original camera masks'),
        training_order=(['lewm_fish'] if args.lewm_mixed_only else ['lewm_archive', 'lewm_fish'] if args.lewm_only else
                        ['lewm_archive', 'lewm_fish', 'actor_archive', 'actor_dense', 'actor_fish']),
        trained_modules=['lewm'] if args.lewm_only else (
            ['lewm', 'reader', 'actor'] if args.from_scratch else ['lewm', 'actor']),
        frozen_modules=(['reader', 'actor'] if args.lewm_only else (
            [] if args.from_scratch else ['reader'])),
        full_patch_lewm=bool(args.full_patch_lewm or args.scene_latent_lewm),
        lewm_archive_stage_from=(dict(checkpoint=str(args.lewm_archive_from), sha256=sha(args.lewm_archive_from))
                                 if args.lewm_archive_from is not None else None),
        lewm_fish_from_archive=task_pool['record'] if task_pool is not None else None,
        lewm_fish_start=fish_start['record'] if fish_start is not None else None,
        scene_lewm_architecture=core.lewm.architecture if args.scene_latent_lewm else None,
        scene_normalization=core.lewm.scene_normalization if args.scene_latent_lewm else None,
        scene_fg_architecture=FG_ARCHITECTURE if fg_table is not None else None,
        scene_fg_table=str(args.fish_fg_table) if fg_table is not None else None,
        scene_fg_weight=args.scene_fg_weight if fg_table is not None else None,
        scene_fg_label_contract=fg_table.manifest if fg_table is not None else None,
        scene_fg_current_label_binding='captured-memory-exposure-v1' if fg_table is not None else None,
        scene_q_architecture=Q_ARCHITECTURE if args.archive_q_weight else None,
        archive_q_weight=args.archive_q_weight or None,
        scene_q_domains=['archive'] if args.archive_q_weight else None,
        scene_q_target=Q_TARGET_KIND if args.archive_q_weight else None,
        scene_q_initialization=(dict(seed=args.seed + Q_READOUT_SEED_OFFSET,
                                     parent_head=parent_q is not None)
                                if args.archive_q_weight else None),
        scene_prediction_mode=args.scene_prediction_mode,
        scene_head_initialization='zero-v1' if args.reset_scene_head else 'parent-or-default-v1',
        scene_branch_initialization=(dict(policy=SCENE_BRANCH_RESET, seed=args.seed + 20001)
                                     if args.reset_scene_branch else None),
        scene_supervision=args.scene_supervision if args.scene_latent_lewm else None,
        scene_targets=(('learned encoder of actual same-episode frames at queries 1/2/.../40; exact timestamps only'
                        if args.scene_supervision == 'dense40' else
                        'learned encoder of actual same-episode frames at queries 5/10/.../40; exact timestamps only')
                       if args.scene_latent_lewm else None),
        scene_target_encoder_stop_gradient=False if args.scene_latent_lewm else None,
        scene_dino_patch_reconstruction=False,
        joint_auxiliary_weight=args.scene_joint_weight if args.scene_latent_lewm else None,
        scene_joint_gradient_routing=args.scene_joint_gradient_routing,
        visual_tokens=6 * PATCHES if args.full_patch_lewm or args.scene_latent_lewm else None,
        visual_token_dim=DIM if args.full_patch_lewm or args.scene_latent_lewm else None, visual_encoder=ENCODER,
        sigreg_weight=(args.scene_sigreg_weight if args.scene_latent_lewm else
                       args.lewm_sigreg_weight if args.full_patch_lewm else None),
        sigreg_projections=(args.scene_sigreg_projections if args.scene_latent_lewm else
                            1024 if args.full_patch_lewm else None),
        sigreg_feature_source=(SCENE_SIGREG_SOURCE if args.scene_latent_lewm else
                              SIGREG_FEATURE_SOURCE if args.full_patch_lewm else None),
        training_mode_policy=TRAINING_MODE_POLICY,
        predictor_dropout=0.1 if args.full_patch_lewm or args.scene_latent_lewm else None,
        random_view_or_action_occlusion=False,
        measurement_dropout=args.measurement_dropout,
        archive_contexts_per_row_per_pass=4 if args.from_scratch else 1,
        native_one_forward_per_model=True, recursive_predictions_as_labels=False,
        hardware_used=False, deployed=False, physical_success_verified=False,
        smoke=args.canary, schedule=schedule, device=device_info,
        reproducibility=reproducibility_record(args.deterministic_training))
    if args.from_stage_checkpoint is not None:
        protocol.update(initialization='native40-stage-checkpoint-then-fish', external_weights_loaded=True,
                        parent_checkpoint=str(args.from_stage_checkpoint), parent_sha256=sha(args.from_stage_checkpoint),
                        fish_stage_lewm=str(args.fish_stage_lewm), fish_stage_lewm_sha256=sha(args.fish_stage_lewm),
                        full_patch_lewm=type(core.lewm).__name__ == 'V21FullPatchLeWM',
                        training_order=['actor_fish'], trained_modules=['reader', 'actor'], frozen_modules=['lewm'])
    # Do not add keys to legacy protocols: old runs remain exactly resumable.
    if args.lewm_balanced_inputs:
        from fcpg.v21_full_patch_lewm import STEP_SCALE, TIME_SCALE
        protocol.update(lewm_balanced_inputs=dict(
            section='17.160', architecture=core.lewm.architecture(),
            picture_bank='a LayerNorm in every layer before the picture keys and values (norm_m)',
            command='(active command - joints) / increment beside the -1..1 positions, in FP32 before autocast',
            times=f'read-back time / {TIME_SCALE} s, its step / {STEP_SCALE} s',
            **(dict(command_symlog="17.161: the distance through DreamerV3's symlog, sign(x) log(1 + |x|)")
               if args.lewm_command_symlog else {})))
    if sam_labels is not None:
        protocol.update(lewm_sam_attention=dict(
            section='17.163', table=str(args.lewm_sam_labels), table_manifest_sha256=sha(args.lewm_sam_labels / 'manifest.json'),
            table_counts=sam_labels.manifest['counts'], table_missing_recordings=sam_labels.manifest['missing_recordings'],
            attention_weight=args.lewm_attention_weight, attention_heads=args.lewm_attention_heads,
            label_head_weight=args.lewm_label_head_weight, architecture=core.lewm.architecture(),
            rows='Archive rows of the LeWM stages only (fish rows, robot blocks and the response pack have no labels)',
            alignment="a label is the one of the cache row the picture's tokens and capture q come from (the chain's "
                      "endpoints, checked by their capture times), moved by the camera layout like them; a bank "
                      "picture finds its chain picture by its capture time",
            penalty='mean of max(0, log .5 - log(1e-4 + mass)), mass = sum over keys of p x coverage: of the first '
                    'heads of every layer, half on the object, half on the robot; all 40 queries; the (row, head) '
                    'pairs whose bank has that kind labelled somewhere',
            label_head="binary cross-entropy of 2 cameras x (object, robot) x 196 logits at the read-back nearest each "
                       "later chain picture (within .05 s) against its coverage / 255"))
    if args.lewm_memory_pictures:
        protocol.update(lewm_picture_memory=dict(
            section='17.166', architecture=core.lewm.architecture(), pictures=args.lewm_memory_pictures,
            tokens=args.lewm_memory_tokens, strides=list(args.lewm_memory_strides), drop=args.lewm_memory_drop,
            drop_seconds=args.lewm_memory_drop_seconds, evaluation_stride=args.lewm_memory_eval_stride,
            lookback=memory_lookback, lewm_parameters=sum(t.numel() for t in core.lewm.parameters()),
            robot_memory_pack=None if args.robot_memory_pack is None else dict(
                path=str(args.robot_memory_pack), sha256=sha(args.robot_memory_pack)),
            reads='per camera the last N pictures taken (fcpg/v21_picture_memory.py: the cache rows and recorded clock '
                  'of the chain, earlier pictures of the same recording at the row\'s stride, valid new exposures '
                  'only), each pooled into K tokens by learned queries, the newest also in all its patches',
            training='each row draws its stride; a share of rows loses one camera for its newest pictures; robot '
                     'blocks keep their three pictures (from_visual_memory); evaluations use the evaluation stride '
                     'and no loss of view'))
    if args.lewm_patch_encoder_layers:
        encoder = core.lewm.patch_encoder
        protocol.update(lewm_patch_encoder=dict(
            section='17.164', layers=len(encoder), architecture=core.lewm.architecture(),
            parameters=sum(t.numel() for t in encoder.parameters()),
            lewm_parameters=sum(t.numel() for t in core.lewm.parameters()),
            reads='every readable token of the bank (2 cameras x 3 pictures x 196 patches, after FiLM) reads every '
                  'other one, pre-LN, 8 heads, before the query layers; the same loss',
            start='attention and feed-forward outputs from 0 (the bank unchanged); built on a copy of the RNG state, '
                  'so every other weight is the one of the same LeWM without it'))
    if (args.lewm_width, args.lewm_layers) != (384, 4):
        protocol.update(lewm_architecture=dict(core.lewm.architecture(), section='17.106',
                                               parameters=sum(t.numel() for t in core.lewm.parameters()),
                                               attention='direct (fcpg/v21_full_patch_lewm.py _attend)'))
    if args.window_readers:
        protocol.update(archive_window_reading=dict(section='17.106', readers=args.window_readers, rows_per_window=2048,
            reading='each locality window of ordered_groups() read at once by separate processes and held on the GPU '
                    '(scripts/train_v21_fcpg_actor.py WindowFeatures); the same draws; checked once against the row cache',
            building='the next batch\'s labels and chain in a separate process'))
    if population_report is not None:
        protocol.update(training_population_contract=deepcopy(population_contract),
                        training_population=population_report)
    if camera_layout is not None:
        protocol.update(archive_camera_layout=camera_layout)
    if cue_manifest is not None:
        protocol.update(object_cues=dict(
            section='17.120', tables=str(args.object_cues),
            sha256={name: hashlib.sha256((args.object_cues / name).read_bytes()).hexdigest()
                    for name in ('archive_object_cue.npy', 'object_cue.manifest.json')},
            rule='the 17.117 Archive grasp (and place) cues attached to tell the rows that hold something; the LeWM '
                 'reads no cue (the frozen Actor\'s evaluations do)'))
    if args.carry_forecast_weight:
        protocol.update(carry_forecast=dict(
            section='17.120', weight=args.carry_forecast_weight, place_factor=args.carry_place_factor,
            margin_mm=2., scale_mm=10., read_backs='the first five',
            held='rows with the grasp cue on; fish rows also with the jaws reading below 12',
            height='scripts/serve_so101_prod_v17_7.V177ClearanceGate.clearance_mm, linearised at the row\'s pose',
            rule='LeWM stages, training rows: each mm the forecast jaws end above the recorded ones beyond the margin, '
                 'squared per scale, mean over held rows (rows seeing the place times place_factor)'))
    if camera_mode == REVIEWED_TYPES:
        protocol.update(lewm_camera_conditioning=camera_mode,
                        camera_type_contract=source.camera_types.contract,
                        camera_type_coverage={split: source.camera_types.coverage(
                            source.archive.row_endpoint_index[splits[split]['archive'], 2])
                            for split in ('train', 'development')})
    if args.resume:
        previous = json.loads((args.output/'protocol.json').read_text())
        if canonical_protocol(previous) != canonical_protocol(protocol):
            if not (args.resume_with_resources or args.resume_with_precision):
                raise ValueError('resume protocol differs')
            if args.resume_with_precision:
                validate_precision_continuation(previous, protocol)
                history_directory = 'precision-history'
            else:
                validate_resource_continuation(previous, protocol)
                history_directory = 'resources-history'
            if not any(args.output.glob('*/latest.pt')) and not any(args.output.glob('*/final.pt')):
                raise ValueError('continuation requires a saved checkpoint')
            write_json(args.output/history_directory/f'{time.time_ns()}.json', dict(
                previous=previous, current=protocol, continuation_requested=True,
                resume_from_saved_optimizer_and_sampler=True))
            write_json(args.output/'protocol.json', protocol)
    else:
        write_json(args.output/'protocol.json', protocol)
        if args.from_scratch:
            write_json(args.output/'initialization.json', dict(
                initialization=protocol['initialization'], seed=args.seed,
                external_weights_loaded=protocol['external_weights_loaded'],
                external_checkpoint_reads=int(args.from_stage_checkpoint is not None),
                query_slots_tiled=False, initial_module_sha256=initial_hashes,
                parameter_counts={name:sum(p.numel() for p in getattr(core,name).parameters())
                                  for name in ('reader','actor','lewm')}))
    config.update(version='21.0.4', model=asdict(core.config), recipe=SCHEMA,
                  horizon_training=protocol, smoke=args.canary,
                  initialization=protocol['initialization'], external_weights_loaded=protocol['external_weights_loaded'],
                  schedule=schedule, q_loss_scale=core.q_loss_scale.detach().cpu().tolist())
    config['training_data_contract'] = dict(config['training_data_contract'],
        schema='v21-native40-source-split-v1', fish_train_episodes=protocol['fish_train_episodes'],
        fish_development_episodes=protocol['fish_development_episodes'])
    config['training_data_contract'] = effective_data_contract(
        config['training_data_contract'], population_report)
    if population_report is not None:
        write_json(args.output/'training-population.json', population_report)
    write_json(args.output/'config.json', config)
    response_data = load_response_pack(args.response_pack, device)
    response_rng = np.random.default_rng(args.seed + 900)
    robot_rng = np.random.default_rng(args.seed + 950)        # 10/31: the robot blocks' draws, their own generator
    memory_rng = np.random.default_rng(args.seed + 1100)      # 17.166: picture-memory strides and view loss
    memory_counts = dict(rows=0, pictures=0, rows_losing_a_view=0)
    demo_rng = np.random.default_rng(args.seed + 960)         # 17.154: the Archive stage's demonstration rows
    started = time.monotonic()
    progress = dict(state='preparing', version='21.0.4', prediction_horizon=40, execution_horizon=5,
                    precision=args.precision, lewm_command_conditioning=args.lewm_command_conditioning)
    if args.scene_latent_lewm:
        progress.update(scene_normalization=core.lewm.scene_normalization,
                        scene_prediction_mode=core.lewm.scene_prediction_mode,
                        scene_lewm_architecture=core.lewm.architecture)
    write_json(args.output/'run-status.json', progress)
    training_variant = 0

    def lewm_pictures(state):
        """The LeWM's picture inputs of a prepared state."""
        kwargs = dict(patches=state.memory.features, patch_valid=state.memory.valid, memory=state.memory,
                      now_time=state.time)
        if args.lewm_controller:
            kwargs['controller'] = state.q.new_zeros(len(state.q), 2)       # 17.153: LeRobot recordings
        return kwargs

    def with_picture_memory(kind, rows, h, state, *, evaluation, window):
        """17.166: the state with the LeWM's memory of the last N pictures per camera in place of the chain's three
        (fcpg/v21_picture_memory.py): training rows draw their stride and view loss (memory_rng), evaluations use the
        evaluation stride and see every picture; Archive pictures come from the window pool when the rows have one,
        every other picture through the feature bank (17.170: the same values as the CPU source's reading)."""
        from fcpg.v21_picture_memory import archive_picture_memory, fish_picture_memory, view_loss_drops
        n, b = args.lewm_memory_pictures, len(rows)
        if evaluation:
            strides, drop = np.full(b, args.lewm_memory_eval_stride, np.int64), np.zeros((b, 2), np.int64)
        else:
            strides = memory_rng.choice(np.asarray(args.lewm_memory_strides, np.int64), b)
            drop = view_loss_drops(memory_rng, b, n, strides, args.lewm_memory_drop, args.lewm_memory_drop_seconds)
        bank = None if feature_bank is None else (lambda ids: feature_bank.pictures(kind, ids))
        if kind == 'archive':
            memory = archive_picture_memory(source, rows, h, n, strides=strides, drop=drop, device=device,
                                            features_for=bank if window is None else window.pictures)
        elif kind == 'fish':
            memory = fish_picture_memory(source, rows, h, n, strides=strides, drop=drop, device=device,
                                         features_for=bank)
        else:
            raise ValueError(f'no picture memory for {kind} rows')
        if not evaluation:
            memory_counts['rows'] += b
            memory_counts['pictures'] += int(memory.valid.sum())
            memory_counts['rows_losing_a_view'] += int((drop > 0).any(1).sum())
        return replace(state, memory=memory)

    def sam_targets(rows, chain, state, h):
        """17.163: the SAM 3.1 labels of these Archive rows' pictures (fcpg/v21_sam_labels.py chain_targets: the
        chain's endpoints checked by their capture times, the camera layout, each bank picture by its capture time)."""
        from fcpg.v21_sam_labels import chain_targets
        labels = chain_targets(source, sam_labels, rows, chain, state.memory, h, args.lewm_attention_heads)
        label_counts['rows'] += len(rows)
        label_counts['rows_with_target'] += int(labels['rows'].any(-1).sum())
        label_counts['bank_pictures'] += labels['bank_pictures']
        label_counts['labelled_bank_pictures'] += labels['labelled_bank_pictures']
        return labels

    def sam_losses(response, target, labels):
        """17.163: the weighted attention penalty and label-head loss of one Archive group (sam_step records them)."""
        from fcpg.v21_sam_labels import attention_penalty, future_label_loss
        total = 0.
        if args.lewm_attention_weight:
            penalty, mass = attention_penalty(response.attention_mass, labels['rows'])
            total = total + args.lewm_attention_weight * penalty
            sam_step.update(attention_penalty=round(float(penalty.detach()), 5),
                            attention_mass=None if mass is None else round(mass, 5))
        if args.lewm_label_head_weight:
            with compute_context(device, args.precision):
                label_loss, pictures = future_label_loss(core.lewm.label_head, response.hidden, target.readback_offsets,
                                                         labels['endpoint_times'], labels['h'], labels['coverage'],
                                                         labels['labelled'])
            total = total + args.lewm_label_head_weight * label_loss
            sam_step.update(label_loss=round(float(label_loss.detach()), 5), label_pictures=pictures)
        return total

    def prepare_host(kind, rows, h):
        """17.106: prepare()'s CPU part, (rows with a label, their labels, their chain without pictures), or
        (rows, None, None) when none has a label; built one batch ahead in a separate process (window reading)."""
        available = horizon_metadata(source, rows, kind, h)[-1][:, 0]
        rows = rows[available]
        if not len(rows):
            return rows, None, None
        return rows, horizon_targets(source, rows, kind, h), source.chain(rows, kind, images=False)

    def targets_on_device(targets):
        """HorizonTargets.to(device) through page-locked memory without waiting (17.171, the Actor trainer's 17.78c):
        a plain copy to the GPU made the CPU wait for the GPU's queued work eight times an update; the same values."""
        if device.type != 'cuda':
            return targets.to(device)
        move = lambda t: t.pin_memory().to(device, non_blocking=True)
        trace = targets.trace
        return replace(targets, trace=replace(trace, positions=move(trace.positions), offsets=move(trace.offsets),
                                              valid=move(trace.valid)),
                       command_offsets=move(targets.command_offsets), readback_offsets=move(targets.readback_offsets),
                       commands=move(targets.commands), readbacks=move(targets.readbacks), valid=move(targets.valid))

    def prepare(kind, rows, h, images, *, evaluation=False, input_audit=None, lewm_inputs_only=False, built=None,
                window=None):
        """built: prepare_host(kind, rows, h), made ahead; window: the WindowFeatures holding these rows (17.106)."""
        host = None
        if built is not None:
            rows, targets, host = built
            if targets is None:
                return None
            targets = targets_on_device(targets)
        else:
            # An old five-sample chunk can last less than 0.1 s at 60 Hz. If the
            # recording ends there, this context has no label on the new grid.
            available = horizon_metadata(source, rows, kind, h)[-1][:, 0]
            rows = rows[available]
            if not len(rows):
                return None
            targets = targets_on_device(horizon_targets(source, rows, kind, h))
        if images and feature_bank is not None:
            host = source.chain(rows, kind, images=False) if host is None else host
            # 17.173: the pictures stay float16 as stored (their float32 copies held the same values) in the chain and
            # the memory; the LeWM and FCPG's reader convert them where they read them
            features, checked = (feature_bank.load(rows, kind, context_frames=h+1, as_stored=True) if window is None
                                 else window.features(rows, h+1, as_stored=True))
            if not np.array_equal(checked, host.view_valid[:, :h+1].numpy()):
                raise ValueError('cached factual visual prefix identities differ')
            chain = replace(stage_chain_to_device(host, device), features=features.to(device))
        else:
            chain = staged.chain(rows, kind, images=images)
        if carry_gate is not None and lewm_inputs_only and not evaluation and host is not None:
            # section 17.120: the rows that hold something, their jaws' height gradient and whether they see the place
            from scripts.train_v21_fcpg_actor import carry_rows
            q_now = host.q[:, h].numpy()
            cue = None if host.object_relative is None else host.object_relative[:, h].numpy()
            held, gradient, _ = carry_rows(carry_gate, q_now, cue, np.repeat(q_now[:, None], 5, 1), shut=kind == 'fish')
            place = (np.zeros(len(rows), bool) if host.place_relative is None
                     else host.place_relative[:, h, 0].numpy() > .5)
            carry_counts['rows'] += len(rows)
            carry_counts['held'] += int(held.sum())
            carry_counts['held_seeing_place'] += int((held & place).sum())
            carry_box['rows'] = (torch.from_numpy(held).to(device), torch.from_numpy(gradient).float().to(device),
                                 torch.from_numpy(np.where(place, args.carry_place_factor, 1.).astype(np.float32)).to(device))
        original_chain = chain
        chain = augment_measurements(chain, rows, protocol=args.measurement_dropout,
                                     variant=training_variant, images=images, evaluation=evaluation)
        # Historical feedback represents inference-time forecasts, not an
        # additional dropout augmentation of the frozen factual input path.
        with temporary_eval(core.lewm), compute_context(device, args.precision):
            # LeWM stages and LeWM evaluation read only q / history / memory / time: skip the reader and the
            # per-block forecasts (fcpg/v21_lewm_inputs.py; the same values, tests/test_v21_lewm_inputs.py)
            state = (lewm_inputs(core, chain, h) if lewm_inputs_only and images else
                     factual_state(core, chain, h, images=images, train_reader=args.from_scratch and images))
        if args.lewm_memory_pictures and images:
            state = with_picture_memory(kind, rows, h, state, evaluation=evaluation, window=window)
        if input_audit is not None:
            as_list = lambda x: x.detach().cpu().tolist()
            input_audit.update(kind=kind, rows=rows.tolist(), context=h,
                factual_input_valid=as_list(original_chain.view_valid[:, :h+1]),
                augmented_input_valid=as_list(chain.view_valid[:, :h+1]),
                memory_valid=as_list(state.memory.valid),
                memory_capture_times=as_list(state.memory.image_times),
                memory_capture_q=as_list(state.memory.q_at_image),
                now=as_list(state.time),
                current_capture_times=as_list(chain.endpoint_times[:, h, None].expand(-1, 2)
                    if chain.capture_times is None else chain.capture_times[:, h]))
            if args.canary:
                # Two-update diagnostics record actual command/readback values,
                # not just a claim that both runs selected the same row IDs.
                input_audit['actual_targets'] = dict(
                    commands=as_list(targets.commands), readbacks=as_list(targets.readbacks),
                    command_offsets=as_list(targets.command_offsets),
                    readback_offsets=as_list(targets.readback_offsets),
                    trace_positions=as_list(targets.trace.positions),
                    trace_offsets=as_list(targets.trace.offsets), trace_valid=as_list(targets.trace.valid),
                    trace_complete=targets.trace.complete)
        if sam_labels is not None and kind == 'archive' and lewm_inputs_only and images and not evaluation:
            label_box['targets'] = sam_targets(rows, chain, state, h)
        if future_scenes is not None:
            # Future images are labels only. Load them AFTER constructing the
            # causal input state; never pass them to factual_state/memory.write.
            provider = evaluation_scenes if evaluation else future_scenes
            future = provider.load(rows, kind, h, device)
            if input_audit is not None:
                input_audit.update(future_ids=provider.identities(rows, kind, h).tolist(),
                    future_all40_valid=as_list(future.valid),
                    query_indices_one_based=(future.query_indices+1).tolist())
            spatial = None
            if fg_table is not None and kind == 'fish':
                prefix_ids = np.stack([horizon_metadata(source, rows, kind, j)[0]
                                       for j in range(h+1)], axis=1)
                prefix_times = (chain.endpoint_times[:, :h+1, None].expand(-1, -1, 2)
                                if chain.capture_times is None else chain.capture_times[:, :h+1])
                spatial = (
                    memory_fg_batch(fg_table, prefix_ids, prefix_times,
                                    chain.view_valid[:, :h+1], state.memory),
                    fg_batch(fg_table, provider.identities(rows, kind, h), device=device, image_valid=future.valid))
            return state, chain.text_ids, targets, future, spatial
        return state, chain.text_ids, targets

    @torch.no_grad()
    @development_mode(core)
    def evaluate(stage):
        use_actor = stage.startswith('actor')
        reports = {}
        # 17.124: a task pool adds its test episodes (report "task", Archive rows) and leaves the Archive report
        for name, kind in [('archive', 'archive'), ('fish', 'fish')] + ([('task', 'archive')] if task_pool else []):
            rows = task_pool['test_rows'] if name == 'task' else splits['development'][kind]
            if not len(rows):
                reports[name] = dict(examples=0, evaluated=False,
                    reason='all three available fish demonstrations are explicitly used for training')
                continue
            if name == 'archive':
                rng = np.random.default_rng(args.seed + 800)
                if task_pool is not None:
                    rows = np.setdiff1d(rows, task_pool['excluded_rows'])
                rows = np.sort(rng.choice(rows, min(len(rows), args.development_rows), replace=False))
            if args.canary:
                rows = rows[:2]
            entries = []
            scene_entries = []
            for h in range(4):
                for begin in range(0, len(rows), min(args.batch_size, 32)):
                    prepared = prepare(kind, rows[begin:begin+min(args.batch_size, 32)], h, True, evaluation=True,
                                       lewm_inputs_only=not use_actor)
                    if prepared is None:
                        continue
                    state, ids, target = prepared[:3]
                    with compute_context(device, args.precision):
                        if use_actor:
                            pred = core.act(state, ids, target.command_offsets).commands_position
                            actual = target.commands
                        else:
                            response = core.lewm(state.q, target.trace, state.history, target.readback_offsets,
                                                 **lewm_pictures(state))
                            pred = response.predicted_q_path[:, 1:]
                            if args.scene_latent_lewm:
                                future = prepared[3]
                                target_z = core.lewm.scene_encoder(future.features, future.valid)
                            actual = target.readbacks
                    if args.scene_latent_lewm:
                        _, _, scene_report = scene_prediction_loss(core.lewm, response, future, encoded_target=target_z)
                        if prepared[4] is not None:
                            _, spatial_report = spatial_supervision(core.lewm, response, target_z, future, *prepared[4])
                            scene_report.update(spatial_report)
                        if getattr(core.lewm, 'q_readout', None) is not None and kind == 'archive':
                            _, q_report = q_grounding(core.lewm, response, target_z, future,
                                state.memory.q_at_image[:, :, -1], response.observation_valid[:, :, -1])
                            scene_report.update(q_report)
                        scene_entries.append(scene_report)
                    entries.append(horizon_errors(pred, actual, target.valid))
            total = sum(e['examples'] for e in entries)
            reports[name] = dict(examples=total, full40_examples=sum(e['full40_examples'] for e in entries),
                **{key: sum(e[key] * e['examples'] for e in entries) / total for key in ('first5_mae', 'future_mae')})
            if scene_entries:
                scene_count = sum(e['scene_targets'] for e in scene_entries)
                persistence_count = sum(e['scene_persistence_targets'] for e in scene_entries)
                reports[name].update(scene_targets=scene_count,
                    scene_mse=sum(e['scene_mse'] * e['scene_targets'] for e in scene_entries) / max(scene_count, 1),
                    scene_persistence_mse=sum((e['scene_persistence_mse'] or 0) * e['scene_persistence_targets']
                                             for e in scene_entries) / max(persistence_count, 1),
                    scene_same_mask_mse=sum((e['scene_same_mask_mse'] or 0) * e['scene_persistence_targets']
                                            for e in scene_entries) / max(persistence_count, 1),
                    scene_std_mean=float(np.mean([e['scene_std_mean'] for e in scene_entries if e['scene_std_mean'] is not None])))
                baseline = reports[name]['scene_persistence_mse']
                reports[name]['scene_same_mask_relative_error'] = (
                    reports[name]['scene_same_mask_mse']/baseline if baseline else None)
                if any('scene_fg_prediction' in e for e in scene_entries):
                    spatial = {}
                    for key in ('scene_fg_current', 'scene_fg_future_encoding', 'scene_fg_prediction',
                                'scene_fg_persistence', 'scene_fg_prediction_same_mask'):
                        spatial[key] = {}
                        for c, view in enumerate(('wrist', 'column')):
                            values = [e[key][c] for e in scene_entries if key in e]
                            count = sum(v['pairs'] for v in values)
                            spatial[key][view] = dict(pairs=count, mean_pixel_error=(
                                sum(v['pixel_error_sum'] for v in values)/count if count else None))
                    reports[name]['scene_fg'] = dict(spatial,
                        label_kind='production-pseudo-label-not-independent-ground-truth',
                        evaluated_query_indices=evaluation_scenes.query_indices.tolist())
                if any('scene_q_prediction' in e for e in scene_entries):
                    grounding = {}
                    for key in ('scene_q_current', 'scene_q_future_encoding', 'scene_q_prediction',
                                'scene_q_persistence', 'scene_q_prediction_same_mask'):
                        grounding[key] = {}
                        for c, view in enumerate(('wrist', 'column')):
                            values = [e[key][c] for e in scene_entries if key in e and e[key][c]['pairs']]
                            count = sum(v['pairs'] for v in values)
                            grounding[key][view] = dict(pairs=count, mae_deg=(
                                sum(v['mae_deg'] * v['pairs'] for v in values)/count if count else None))
                    reports[name]['scene_q'] = dict(grounding, label_kind=Q_TARGET_KIND,
                        evaluated_query_indices=evaluation_scenes.query_indices.tolist())
        return reports

    def checkpoint(stage, step, optimizer, complete):
        payload = deepcopy({k: v for k, v in parent.items() if k not in ('model', 'optimizer', 'projector', 'rng')})
        payload.update(version='21.0.4', config=asdict(core.config), model=core.state_dict(),
            stage='fish' if complete and stage in ('actor_fish', 'lewm_fish') and (
                stage == 'actor_fish' or args.lewm_only) else stage,
            step=(parent.get('step', step) if complete and args.lewm_only and stage == 'lewm_fish' else step),
            complete=complete, smoke=args.canary, optimizer=optimizer.state_dict(),
            rng=rng_state(device), native40_protocol=protocol, stage_optimizer_step=step)
        payload['response_rng'] = response_rng.bit_generator.state
        if args.scene_latent_lewm:
            payload['scene_lewm_architecture'] = core.lewm.architecture
            payload['scene_normalization'] = core.lewm.scene_normalization
            payload['scene_prediction_mode'] = core.lewm.scene_prediction_mode
            payload['scene_fg_architecture'] = getattr(core.lewm, 'scene_fg_architecture', None)
            payload['scene_q_architecture'] = getattr(core.lewm, 'scene_q_architecture', None)
        if hasattr(core.lewm, 'command_conditioning'):
            payload['lewm_command_conditioning'] = core.lewm.command_conditioning
        if camera_mode == REVIEWED_TYPES:
            payload['lewm_camera_conditioning'] = camera_mode
        if getattr(core, 'temporal_training_contract', None) is not None:
            payload['temporal_training_contract'] = core.temporal_training_contract
            payload['temporal_feature_coverage'] = core.temporal_feature_coverage
        directory = args.output/('fish' if complete and (stage == 'actor_fish' or (args.lewm_only and stage == 'lewm_fish')) else stage)
        directory.mkdir(exist_ok=True)
        path = directory/('final.pt' if complete else 'latest.pt')
        temporary = path.with_suffix('.pt.tmp')
        torch.save(payload, temporary); temporary.replace(path)
        if args.locked_development_report is not None and not complete:
            import shutil
            immutable = directory/f'step-{step:05d}.pt'
            if immutable.exists():
                raise FileExistsError('do not overwrite an evaluated checkpoint')
            shutil.copy2(path, immutable)
            return immutable
        return path

    def evaluate_clock_checkpoint(stage, step, saved_path):
        if args.locked_development_report is None or args.canary:
            return
        from fcpg.v21_clock_evaluation import locked_development_rows, evaluate_core, short_training_gate
        locked = json.loads(args.locked_development_report.read_text())
        baseline = json.loads(args.clock_baseline_report.read_text())
        if source.recorded_clock is None:
            raise ValueError('clock repair evaluation requires completed corrected cache')
        rows = locked_development_rows(locked, splits['development'])
        with development_mode(core):
            metrics = evaluate_core(core, source, feature_bank, rows, device=device,
                                    precision=args.precision, context=locked['context'])
        gate = short_training_gate(metrics, baseline['models']['ln'], baseline['models']['parent'])
        destination = args.output/'clock-evaluation'/f'{stage}-{step:05d}.json'
        write_json(destination, dict(step=step, stage=stage, models=dict(ln=metrics), gate=gate,
            rows=locked['rows'], checkpoint=str(saved_path),
            cache_generation=source.recorded_clock.metadata['extraction_id']))
        print(json.dumps(dict(clock_evaluation=str(destination), step=step, gate=gate)), flush=True)

    try:
        for stage in protocol['training_order']:
            lewm_stage = stage.startswith('lewm')
            module = configure_training_stage(core, stage, from_scratch=args.from_scratch)
            if stage == 'lewm_fish' and args.fish_stage_seed is not None:
                # 17.127: the fish stage drawn again (dropout, SIGReg directions, retention draws; its rows come
                # from the same seed); the restored Archive stage set these from --seed
                random.seed(args.fish_stage_seed); np.random.seed(args.fish_stage_seed)
                torch.manual_seed(args.fish_stage_seed)
                response_rng.bit_generator.state = np.random.default_rng(args.fish_stage_seed + 900).bit_generator.state
                robot_rng.bit_generator.state = np.random.default_rng(args.fish_stage_seed + 950).bit_generator.state
            if stage == 'lewm_fish' and fish_start is not None:
                # 17.125: the fish stage starts where that run's ended (every LeWM weight replaced, loaded strictly)
                core.lewm.load_state_dict(fish_start.pop('lewm'))
                write_json(args.output/'lewm_fish-start.json', dict(section='17.125', **fish_start['record'],
                                                                    lewm_sha256=state_hash(core.state_dict(), 'lewm.')))
            parameters = [p for p in core.parameters() if p.requires_grad]
            learning_rate = (args.fish_learning_rate if stage.endswith('_fish') and args.fish_learning_rate else
                             args.learning_rate)
            optimizer = torch.optim.AdamW(parameters, lr=learning_rate,
                                          weight_decay=.01 if (args.from_scratch or args.full_patch_lewm or args.scene_latent_lewm) else 0.,
                                          fused=args.fused_optimizer)
            sigreg = None
            if lewm_stage and (args.full_patch_lewm or args.scene_latent_lewm):
                from fcpg.regularization import SIGReg
                sigreg = SIGReg(knots=17, num_proj=(args.scene_sigreg_projections if args.scene_latent_lewm else 1024)).to(device)
            stage_dir = args.output/('fish' if stage == 'actor_fish' else stage)
            final_path = stage_dir/'final.pt'
            if final_path.exists() and args.resume:
                saved = torch.load(final_path, map_location='cpu', weights_only=False)
                restore_training_checkpoint(core, saved, scene_sigreg_weight=(
                    args.scene_sigreg_weight if args.scene_latent_lewm else None),
                    scene_sampling=(args.scene_supervision, args.scene_sigreg_projections),
                    deterministic_training=args.deterministic_training,
                    current_runtime=protocol['reproducibility'],
                    scene_joint_gradient_routing=args.scene_joint_gradient_routing,
                    population_contract=population_contract, population_report=population_report)
                response_rng.bit_generator.state = saved['response_rng']
                continue
            if stage == 'lewm_archive' and args.lewm_archive_from is not None:
                # 17.123: that run's finished Archive stage, restored as a resume would, instead of training one
                saved = torch.load(args.lewm_archive_from, map_location='cpu', weights_only=False)
                checked = validate_archive_stage_source(saved, protocol, args.source_config,
                                                        {a.dest: a.default for a in parser()._actions})
                if (getattr(core, 'temporal_training_contract', None) is None
                        and saved.get('temporal_training_contract') is not None
                        and saved['temporal_training_contract'] == parent.get('temporal_training_contract')):
                    # checkpoint() starts from a copy of the parent, so a full-patch LeWM-only run saves the parent's
                    # temporal contract although its own core never had one (HL's did not; nor does this one)
                    saved = dict(saved, temporal_training_contract=None)
                    checked['parent_temporal_contract_in_checkpoint_not_the_runs'] = True
                restore_training_checkpoint(core, saved, scene_sigreg_weight=None,
                    scene_sampling=(args.scene_supervision, args.scene_sigreg_projections),
                    deterministic_training=args.deterministic_training,
                    current_runtime=protocol['reproducibility'],
                    scene_joint_gradient_routing=args.scene_joint_gradient_routing,
                    population_contract=population_contract, population_report=population_report)
                response_rng.bit_generator.state = saved['response_rng']
                write_json(args.output/'lewm_archive-from.json', dict(section='17.123', **checked,
                    **protocol['lewm_archive_stage_from'], lewm_sha256=state_hash(core.state_dict(), 'lewm.'),
                    restored='weights, RNG and response draws; this run did not train an Archive stage'))
                del saved
                continue
            resume_path = args.output/stage/'latest.pt'
            resume_step = 0
            if args.resume and resume_path.exists():
                saved = torch.load(resume_path, map_location='cpu', weights_only=False)
                restore_training_checkpoint(core, saved, optimizer, scene_sigreg_weight=(
                    args.scene_sigreg_weight if args.scene_latent_lewm else None),
                    scene_sampling=(args.scene_supervision, args.scene_sigreg_projections),
                    deterministic_training=args.deterministic_training,
                    current_runtime=protocol['reproducibility'],
                    scene_joint_gradient_routing=args.scene_joint_gradient_routing,
                    population_contract=population_contract, population_report=population_report)
                response_rng.bit_generator.state = saved['response_rng']
                resume_step = saved['step']
                write_json(args.output/'resumed-checkpoints'/f'{time.time_ns()}.json', dict(
                    checkpoint=str(resume_path), checkpoint_sha256=sha(resume_path), stage=stage, step=resume_step,
                    previous_precision=saved['native40_protocol']['schedule'].get('precision', 'fp32'),
                    current_precision=args.precision, optimizer_rng_sampler_and_coverage_restored=True))
            baseline_file = args.output/(stage+'-before.json')
            if not baseline_file.exists():
                write_json(args.output/'run-status.json', dict(progress, state='evaluating', stage=stage, phase='before'))
                write_json(baseline_file, evaluate(stage))
            groups_for_stage = ordered_groups(stage, splits, args,
                archive_episode_index=source.archive.endpoint_episode_index[source.archive.row_endpoint_index[:,0]],
                task_rows=task_pool['learn_rows'] if task_pool is not None else None)
            actor_input_bank = None
            if args.cache_frozen_actor_inputs and stage == 'actor_fish':
                from fcpg.v21_frozen_actor_inputs import FrozenActorInputBank
                groups_for_stage = list(groups_for_stage)
                populations = {}
                for step, (h, groups) in enumerate(groups_for_stage, 1):
                    if step > resume_step:
                        for kind, rows in groups:
                            populations.setdefault((kind, h), []).append(rows)
                actor_input_bank = FrozenActorInputBank(core)
                for (kind, h), blocks in sorted(populations.items()):
                    def cache_progress(done, total):
                        write_json(args.output/'run-status.json', dict(progress, state='caching_actor_inputs',
                            stage=stage, kind=kind, context=h, prepared_rows=done, total_rows=total,
                            completed_banks=len(actor_input_bank.banks), total_banks=len(populations),
                            frozen_actor_inputs=actor_input_bank.stats()))
                    actor_input_bank.build(kind, h, np.concatenate(blocks), prepare,
                                           batch_size=args.batch_size, progress=cache_progress)
                parity = []
                with torch.no_grad():
                    for (kind, h), blocks in sorted(populations.items()):
                        rows = blocks[0]
                        state, ids, targets = prepare(kind, rows, h, True)
                        expected = core.act(state, ids, targets.command_offsets).commands_position
                        cached = actor_input_bank.get(kind, h, rows)
                        actual = core.actor(**cached.inputs).commands_position
                        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-5)
                        parity.append(dict(kind=kind, context=h, examples=len(rows),
                            maximum_command_difference=float((actual-expected).abs().max())))
                write_json(args.output/'frozen-actor-input-cache.json', dict(
                    **actor_input_bank.stats(), actual_frozen_inputs=True, optimizer_schedule_unchanged=True,
                    parity=parity))
            window_features = builder = building = None
            if args.window_readers and stage == 'lewm_archive':
                # 17.106: the Actor trainer's window reading (17.79-17.82) and batch building (17.81) for the Archive
                # LeWM stage, on ordered_groups()' unchanged draws. Both kinds of process are forked here, readers
                # first, before any thread of this stage starts
                from scripts.train_v21_fcpg_actor import WindowFeatures, _start_building_process
                groups_for_stage = one_ahead(archive_windows(groups_for_stage, splits['train']['archive'],
                                                             args.window_readers))
                window_features = WindowFeatures(source, device, readers=args.window_readers)
                window_features.lookback = memory_lookback          # 17.166: 0 without a picture memory
                window_features.start_readers()
                _BUILDING['prepare'] = prepare_host
                builder = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context('fork'),
                                              initializer=_start_building_process)
                builder.submit(_start_building_process).result()      # forked now
                window_checked = False
            stage_start = time.monotonic()
            count = 0
            with (args.output/'training.jsonl').open('a') as log, (
                    (args.output/'paired-schedule.jsonl').open('a') if args.paired_schedule_audit
                    else nullcontext(None)) as audit_log:
                # 17.171: a LeWM update makes none of the model's GPU checks (fcpg.v21_checks.unchecked; James 「不要
                # 什么安全检查了 不要了」: each read a GPU value on the CPU, ~140 waits an update) and its losses and
                # gradient norm are read after the next update is queued (the Actor trainer's 17.80); nothing computed
                # changes. The one stop left: a loss or gradient norm that is not finite, seen when it is logged (every
                # update is logged before a checkpoint or an evaluation)
                defer = (device.type == 'cuda' and lewm_stage and actor_input_bank is None
                         and not args.scene_latent_lewm and audit_log is None)
                waiting = None

                def logged(entry):
                    """Read one update (its losses and gradient norm become numbers; not finite: the run stops) and
                    write its log line."""
                    record, pending, parts = entry
                    if pending is not None:
                        losses, (gradient_norm,) = pending.finish((parts, 1))
                        record.update(loss=sum(losses, 0.), gradient_norm=gradient_norm)
                    if not np.isfinite([record['loss'], record['gradient_norm']]).all():
                        raise RuntimeError(f"update {record['step']}: the loss or the gradient norm is not finite")
                    log.write(json.dumps(record)+'\n')
                    if record['step'] == 1 or record['step'] % 25 == 0:
                        log.flush(); write_json(args.output/'run-status.json', dict(progress, state='running', **record,
                            feature_cache=feature_bank.stats() if feature_bank is not None else None,
                            frozen_actor_inputs=actor_input_bank.stats() if actor_input_bank is not None else None,
                            **(dict(window_wait_s=round(window_features.waited, 1),
                                    windows_read_ahead=window_features.prefetched) if window_features is not None else {}),
                            gpu_allocated_bytes=torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0,
                            gpu_reserved_bytes=torch.cuda.memory_reserved(device) if device.type == 'cuda' else 0))
                        print(json.dumps(record), flush=True)

                for step, entry in enumerate(groups_for_stage, 1):
                    window = upcoming = None
                    if window_features is not None:
                        (h, groups, window), upcoming = entry
                    else:
                        h, groups = entry
                    count = step
                    if step <= resume_step:
                        continue
                    built = None
                    if window_features is not None:
                        (_, rows), = groups
                        if building is not None:
                            asked, building = building, None
                            if asked[0] != h or not np.array_equal(asked[1], rows):
                                raise ValueError('the batch built ahead is not this update\'s')
                            built = asked[2].result()
                        else:
                            built = prepare_host('archive', rows, h)
                        if upcoming is not None:
                            next_h, ((_, next_rows),), _ = upcoming
                            building = (next_h, next_rows, builder.submit(_build_in_process, 'archive', next_rows, next_h))
                        key, window_rows, last, following = window
                        if key not in window_features.resident:
                            window_features.prefetch([(key, window_rows)])
                            window_features.admit(key, window_rows)
                        window_features.prefetch(following)       # the disk always reads the next windows
                        if not window_checked and built[1] is not None:
                            # once per run: the window's pictures are the row cache's, bit for bit
                            mine, mine_valid = window_features.features(built[0], h+1)
                            reference, reference_valid = feature_bank.load(built[0], 'archive', context_frames=h+1)
                            if not torch.equal(mine, reference) or not np.array_equal(mine_valid, reference_valid):
                                raise ValueError('window features differ from the row cache')
                            write_json(args.output/'window-feature-check.json', dict(section='17.106', step=step,
                                context=h, rows=len(built[0]), equal_to_row_cache=True))
                            window_checked = True
                            del mine, reference
                    group_weights = [1 / len(groups)] * len(groups)
                    if stage == 'lewm_archive' and args.lewm_archive_demo_rows:
                        # 17.154: demonstrations 0 and 1 beside the Archive rows, at this update's context (their
                        # own group: not built ahead, not in the Archive window)
                        groups = [*groups, ('fish', demo_rng.choice(splits['train']['fish'],
                                                                    args.lewm_archive_demo_rows, replace=True))]
                        group_weights.append(args.lewm_archive_demo_weight)
                    training_variant = step
                    pending, losses = (Pending() if defer else None), []
                    update_checks = ExitStack()
                    if defer:
                        update_checks.enter_context(deferred(pending))
                        update_checks.enter_context(unchecked())
                    optimizer.zero_grad(set_to_none=True)
                    loss_value = 0.
                    scene_records = []
                    sam_step.clear()
                    pair_evidence = dict(stage=stage, step=step, context=h,
                        groups=[dict(kind=kind, rows=rows.tolist()) for kind, rows in groups],
                        rng_before=paired_rng_evidence(device), response_rng_before=response_rng.bit_generator.state,
                        masks=[], inputs=[]) if audit_log is not None else None
                    for group_index, (kind, rows) in enumerate(groups):
                        if actor_input_bank is not None:
                            cached = actor_input_bank.get(kind, h, rows)
                            with compute_context(device, args.precision):
                                action = core.actor(**cached.inputs).commands_position
                            loss = actor_loss(action, cached, cached.inputs['q_position'], core.q_loss_scale)
                        else:
                            input_record = {} if pair_evidence is not None else None
                            carry_box.clear()
                            label_box.clear()
                            prepared = prepare(kind, rows, h, True, input_audit=input_record,
                                               lewm_inputs_only=lewm_stage,
                                               built=built if group_index == 0 else None,
                                               window=window_features if group_index == 0 else None)
                            if prepared is None:
                                continue
                            if pair_evidence is not None:
                                pair_evidence['inputs'].append(input_record)
                            state, text_ids, target = prepared[:3]
                            if lewm_stage:
                                labels = label_box.pop('targets', None)        # 17.163: Archive rows with SAM labels
                                supervision = ({} if labels is None or not args.lewm_attention_weight else
                                               dict(attention_supervision=(args.lewm_attention_heads, labels['weights'])))
                                with compute_context(device, args.precision):
                                    response = core.lewm(
                                        state.q, target.trace, state.history, target.readback_offsets,
                                        **lewm_pictures(state), **supervision)
                                    if args.scene_latent_lewm:
                                        future = prepared[3]
                                        target_z = core.lewm.scene_encoder(future.features, future.valid)
                                        if args.canary:
                                            target_z.retain_grad()
                                        if pair_evidence is not None:
                                            pair_evidence['masks'].append(dict(kind=kind,
                                                observation=response.observation_valid.cpu().tolist(),
                                                # Shared eight-time mask; dense MSE coverage is logged separately.
                                                future=future.valid[:, future.regularization_slots()].cpu().tolist(),
                                                joint=target.valid.cpu().tolist()))
                                loss = lewm_loss(response, target, core.q_loss_scale)
                                if labels is not None:
                                    loss = loss + sam_losses(response, target, labels)
                                carry = carry_box.pop('rows', None)
                                if carry is not None:       # section 17.120: forecast jaws above the recorded ones
                                    held, gradient, factor = carry
                                    loss = loss + args.carry_forecast_weight * carry_forecast_penalty(
                                        response.predicted_q_path[:, 1:6].float(), target.readbacks[:, :5].float(),
                                        target.valid[:, :5], gradient, held, factor)
                                if args.scene_latent_lewm:
                                    joint_objective = args.scene_joint_weight * loss
                                    scene_loss, reg, scene_report = scene_prediction_loss(
                                        core.lewm, response, future, sigreg, encoded_target=target_z)
                                    scene_report.update(kind=kind, joint_loss=float(loss.detach()),
                                        sigreg_weight=args.scene_sigreg_weight,
                                        sigreg_projections=sigreg.num_proj,
                                        scene_supervision=args.scene_supervision,
                                        weighted_scene_sigreg=float((args.scene_sigreg_weight * reg).detach()))
                                    scene_records.append(scene_report)
                                    loss = scene_training_loss(scene_loss, reg, loss,
                                        sigreg_weight=args.scene_sigreg_weight, joint_weight=args.scene_joint_weight)
                                    representation_objective = scene_loss + args.scene_sigreg_weight * reg
                                    if prepared[4] is not None:
                                        spatial_loss, spatial_report = spatial_supervision(
                                            core.lewm, response, target_z, future, *prepared[4])
                                        loss = loss + args.scene_fg_weight * spatial_loss
                                        representation_objective = representation_objective + args.scene_fg_weight * spatial_loss
                                        scene_report.update(spatial_report, scene_fg_weight=args.scene_fg_weight,
                                            weighted_scene_fg_loss=float((args.scene_fg_weight*spatial_loss).detach()))
                                        if args.canary:
                                            scene_report['scene_fg_gradient_audit'] = audit_spatial_gradients(
                                                spatial_loss, core.lewm, response, target_z, future, *prepared[4])
                                    if args.archive_q_weight and kind == 'archive':
                                        # Label: capture q of the exact retained exposure each scene encodes.
                                        current_valid = response.observation_valid[:, :, -1]
                                        q_term, q_report = q_grounding(core.lewm, response, target_z, future,
                                            state.memory.q_at_image[:, :, -1], current_valid)
                                        loss = loss + args.archive_q_weight * q_term
                                        representation_objective = representation_objective + args.archive_q_weight * q_term
                                        scene_report.update(q_report, archive_q_weight=args.archive_q_weight,
                                            weighted_scene_q_loss=float((args.archive_q_weight*q_term).detach()))
                                        if args.canary:
                                            scene_report['scene_q_gradient_audit'] = audit_q_gradients(
                                                q_term, core.lewm, response, target_z, future, current_valid)
                                elif sigreg is not None:
                                    reg = lewm_sigreg_loss(response, sigreg)
                                    loss = loss + args.lewm_sigreg_weight * reg
                                    # #region agent log
                                    if step == 1:
                                        try:
                                            import json as _json, time as _time
                                            _layer = core.lewm.layers[0]
                                            with open('/home/james/feedback-corrected-grounding/.cursor/debug-55ff20.log', 'a') as _f:
                                                _f.write(_json.dumps({'sessionId': '55ff20', 'hypothesisId': 'sigreg-joint', 'location': 'train_v21_horizon40.py:lewm', 'message': 'joint loss plus sigreg', 'data': {'sigreg_weight': args.lewm_sigreg_weight, 'sigreg': round(float(reg.detach()), 6), 'pred_q_shape': list(response.predicted_q_path.shape), 'self_dropout': float(_layer.self_attn.dropout), 'cross_dropout': float(_layer.cross_attn.dropout), 'ff_dropout': 0.1}, 'timestamp': int(_time.time() * 1000)}) + '\n')
                                        except Exception:
                                            pass
                                    # #endregion
                            else:
                                with compute_context(device, args.precision):
                                    action = core.act(state, text_ids, target.command_offsets).commands_position
                                loss = actor_loss(action, target, state.q, core.q_loss_scale)
                        loss = loss * group_weights[group_index]
                        if not defer and not torch.isfinite(loss):   # 17.173: deferred, logged() stops on it (17.171)
                            raise ValueError('nonfinite native40 loss')
                        if lewm_stage and args.scene_latent_lewm:
                            if args.canary:
                                # Exclude earlier spatial audit probes from the
                                # subsequent actual-backward gradient check.
                                target_z.grad = None
                            scene_report['joint_scene_gradient_routing'] = backward_scene_loss(
                                loss, joint_objective/len(groups), representation_objective/len(groups),
                                core.lewm.scene_encoder, mode=args.scene_joint_gradient_routing,
                                audit=args.canary, retained_targets=(target_z,) if args.canary else ())
                        else:
                            loss.backward()
                        if defer:
                            losses.append(loss.detach())
                        else:
                            loss_value += float(loss.detach())
                        if args.scene_latent_lewm and args.canary:
                            if not future.valid.any() or target_z.grad is None or not target_z.grad[future.valid].abs().sum() > 0:
                                raise ValueError('scene canary: future observation target has no gradient')
                            if any(p.grad is not None for name in ('actor', 'reader') for p in getattr(core, name).parameters()):
                                raise ValueError('scene canary: frozen Actor/FCPG received gradients')
                            scene_report['future_encoder_gradient_verified'] = True
                            if args.lewm_command_conditioning == FULL_COMMANDS and step > 1:
                                gradient = core.lewm.command_projection[0].weight.grad
                                if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().sum() > 0:
                                    raise ValueError('command canary: complete command branch has no gradient')
                                scene_report['complete_command_gradient_verified'] = True
                    if lewm_stage and response_data is not None and not (
                            stage == 'lewm_fish' and args.lewm_fish_demos_only):     # 17.125: the demonstrations alone
                        from scripts.train_v21_local_response_repair import response_loss
                        size, batch = response_data
                        q, trace, history, qt, actual = batch(response_rng.integers(size, size=32))
                        with compute_context(device, args.precision):
                            setting = {}
                            if args.lewm_controller:          # 17.153: the service's planner, LeRobot's P 16
                                # [0., 1.] made on the GPU (17.173: the list copied to it made the CPU wait for all
                                # the queued GPU work, 43 ms of a b128 update)
                                setting['controller'] = torch.arange(2, dtype=q.dtype, device=q.device).expand(len(q), 2)
                            response = core.lewm(q, trace, history, qt, **setting)
                        retention = .5 * response_loss(response, actual, core.q_loss_scale)[0]
                        # q-only real response records have no visual labels.
                        # They cannot supply an observation-encoder SIGReg term.
                        if sigreg is not None and not args.scene_latent_lewm:
                            retention = retention + args.lewm_sigreg_weight * lewm_sigreg_loss(response, sigreg)
                        retention.backward()
                        if defer:
                            losses.append(retention.detach())
                        else:
                            loss_value += float(retention.detach())
                    robot_rows = 0
                    if lewm_stage and robot_blocks is not None and not (
                            stage == 'lewm_fish' and args.lewm_fish_demos_only) and not (
                            stage == 'lewm_archive' and args.robot_blocks_fish_only):     # 17.155
                        # 10/31: the robot's own blocks with their pictures, beside the response pack
                        from scripts.train_v21_local_response_repair import response_loss
                        picked = robot_rng.integers(robot_blocks.size, size=args.robot_block_rows)
                        q, trace, history, qt, actual, memory, now, _ = robot_blocks.batch(picked)
                        if args.lewm_memory_pictures and robot_memory is not None:
                            # 17.166: the session's stream (--robot-memory-pack), a share of rows losing one camera
                            b = len(picked)
                            lost = np.zeros((b, 2))
                            lose = memory_rng.random(b) < args.lewm_memory_drop
                            camera = memory_rng.integers(0, 2, b)
                            lost[np.flatnonzero(lose), camera[lose]] = memory_rng.uniform(
                                .5, args.lewm_memory_drop_seconds, b)[lose]
                            memory = robot_memory.memory(picked, now, args.lewm_memory_pictures, lost)
                        elif args.lewm_memory_pictures:    # 17.166: the snapshot's three pictures, newest slots
                            from fcpg.v21_picture_memory import from_visual_memory
                            memory = from_visual_memory(memory, args.lewm_memory_pictures)
                        with compute_context(device, args.precision):
                            pictures = dict(patches=memory.features, patch_valid=memory.valid, memory=memory,
                                            now_time=now)
                            if args.lewm_controller:          # 17.153: the session's arm P, the service's planner
                                pictures['controller'] = robot_blocks.controller(picked).to(q)
                            response = core.lewm(q, trace, history, qt, **pictures)
                        robot = args.robot_block_weight * response_loss(response, actual, core.q_loss_scale)[0]
                        if sigreg is not None and not args.scene_latent_lewm:
                            robot = robot + args.lewm_sigreg_weight * lewm_sigreg_loss(response, sigreg)
                        robot.backward()
                        if defer:
                            losses.append(robot.detach())
                        else:
                            loss_value += float(robot.detach())
                        robot_rows = len(picked)
                    gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=not defer)
                    optimizer.step()
                    if window is not None and window[2]:
                        window_features.release(window[0])     # its last rows are in this update: gone (as 17.82)
                    update_checks.close()
                    if defer:
                        pending.start(*losses, gradient)    # queued after this update, read after the next one
                    if audit_log is not None:
                        pair_evidence['rng_after'] = paired_rng_evidence(device)
                        pair_evidence['response_rng_after'] = response_rng.bit_generator.state
                        audit_log.write(json.dumps(pair_evidence)+'\n')
                        if step == 1 or step % 25 == 0:
                            audit_log.flush()
                    if not lewm_stage:
                        constrain_training_age_columns(core, optimizer)
                    record = dict(stage=stage, step=step, precision=args.precision,
                                  loss=loss_value, gradient_norm=None if defer else float(gradient),
                                  **({'demo_rows': len(groups[-1][1])}
                                     if stage == 'lewm_archive' and args.lewm_archive_demo_rows else {}),
                                  **({'robot_rows': robot_rows} if lewm_stage and robot_blocks is not None else {}),
                                  module_modes={name: getattr(core, name).training
                                                for name in ('lewm', 'actor', 'reader')},
                                  sigreg_feature_source=protocol['sigreg_feature_source'],
                                  scene_components=scene_records,
                                  **(dict(carry=dict(carry_counts)) if carry_gate is not None else {}),
                                  **(dict(sam=dict(sam_step, **label_counts)) if sam_labels is not None else {}),
                                  **(dict(picture_memory=dict(memory_counts)) if args.lewm_memory_pictures else {}),
                                  elapsed_s=time.monotonic()-started, stage_elapsed_s=time.monotonic()-stage_start)
                    if not defer:
                        logged((record, None, 0))
                    else:
                        if waiting is not None:
                            logged(waiting)
                        waiting = (record, pending, len(losses))
                        if step % args.checkpoint_every == 0 or step in args.keep_steps:
                            logged(waiting)         # read before anything is saved or evaluated
                            waiting = None
                    if step % args.checkpoint_every == 0:
                        saved_path = checkpoint(stage, step, optimizer, False)
                        evaluate_clock_checkpoint(stage, step, saved_path)
                    if step in args.keep_steps:
                        import shutil
                        kept = checkpoint(stage, step, optimizer, False)
                        immutable = kept.parent/f'step-{step:05d}.pt'
                        if immutable.exists():
                            raise FileExistsError('do not overwrite a kept checkpoint')
                        shutil.copy2(kept, immutable)
                        write_json(args.output/f'{stage}-step-{step:05d}.json', evaluate(stage))
                if waiting is not None:
                    logged(waiting)                 # the stage's last update
                    waiting = None
            if window_features is not None:
                builder.shutdown(wait=True, cancel_futures=True)
                window_features.close()
            if args.lewm_only and state_hash(core.state_dict(), 'actor.') != initial_hashes['actor']:
                raise ValueError('lewm-only training changed the Actor')
            if (not args.from_scratch or lewm_stage) and state_hash(core.state_dict(), 'reader.') != frozen_reader_sha:
                raise ValueError('frozen FCPG changed')
            write_json(args.output/'run-status.json', dict(progress, state='evaluating', stage=stage, phase='after'))
            write_json(args.output/(stage+'-after.json'), evaluate(stage))
            checkpoint(stage, count, optimizer, True)
        report = dict(progress, state='completed', elapsed_s=time.monotonic()-started,
            final_stage_optimizer_steps=count,
            checkpoint=str(args.output/'fish/final.pt'), frozen_fcpg_unchanged=not args.from_scratch,
            modules_changed={name:state_hash(core.state_dict(),name+'.') != initial_hashes[name]
                             for name in ('reader','actor','lewm')},
            validation_scope='offline archive development; fish development only when reserved in protocol; teacher-forced history',
            physical_success_verified=False, deployment_approved=False,
            max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
        if args.canary and args.scene_latent_lewm:
            saved = torch.load(args.output/'fish/final.pt', map_location='cpu', weights_only=False)
            reloaded = core_from_parent(saved)
            for name, value in core.state_dict().items():
                if not torch.equal(value.detach().cpu(), reloaded.state_dict()[name]):
                    raise ValueError('scene canary checkpoint roundtrip changed '+name)
            report['scene_checkpoint_strict_roundtrip_verified'] = True
            reloaded = reloaded.to(device).eval()
            with torch.no_grad(), development_mode(core):
                sample_domain = 'fish' if len(splits['development']['fish']) else 'archive'
                # 17.166: a picture-memory LeWM reads LeWM inputs only (the chain's state machine forecasts with three)
                sample = prepare(sample_domain, splits['development'][sample_domain][:2], 2, True,
                                 lewm_inputs_only=bool(args.lewm_memory_pictures))
                if sample is None:
                    raise ValueError('canary reload has no eligible fixed DEV examples')
                state, _, target = sample[:3]
                with compute_context(device, args.precision):
                    kwargs = lewm_pictures(state)
                    expected = core.lewm(state.q, target.trace, state.history, target.readback_offsets, **kwargs)
                    actual = reloaded.lewm(state.q, target.trace, state.history, target.readback_offsets, **kwargs)
                for field in ('predicted_q_path', 'predicted_scene', 'observation_scene', 'scene_anchor_valid'):
                    torch.testing.assert_close(getattr(actual, field), getattr(expected, field), atol=0, rtol=0)
                if fg_table is not None:
                    torch.testing.assert_close(reloaded.lewm.fg_readout(actual.predicted_scene.float()),
                        core.lewm.fg_readout(expected.predicted_scene.float()), atol=0, rtol=0)
                    report['scene_fg_forward_roundtrip_verified'] = True
                if getattr(core.lewm, 'q_readout', None) is not None:
                    torch.testing.assert_close(reloaded.lewm.q_readout(actual.predicted_scene.float()),
                        core.lewm.q_readout(expected.predicted_scene.float()), atol=0, rtol=0)
                    report['scene_q_forward_roundtrip_verified'] = True
            report['scene_checkpoint_forward_roundtrip_verified'] = True
            del reloaded
        write_json(args.output/'run-status.json', report)
        print(json.dumps(report), flush=True)
    except BaseException as error:
        write_json(args.output/'run-status.json', dict(progress, state='failed', error=type(error).__name__+': '+str(error)))
        raise


if __name__ == '__main__':
    main()
