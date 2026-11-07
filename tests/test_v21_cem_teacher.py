"""The alignment CEM as the Actor's teacher (fcpg/v21_cem_teacher.py, training overhaul doc section 17.110): the labels
are the robot CEM's own selections around the demonstrator's commands, the gripper stays the demonstrator's, and the
fish stage puts a label in together with the token shift it was made for. No robot, no data files."""
from dataclasses import dataclass, replace
import json

import numpy as np
import pytest
import torch

from fcpg.v21_alignment_cem import (ARM, IMAGE, MAX_OFFSET, OPEN_GRIP, SEARCH_JOINTS, AlignmentCEM, GeometricResponse,
                                    WristResponse)
from fcpg.v21_cem_teacher import (SCHEMA, SHIFT_LIMIT_PX, TeacherLabels, admission, draw_shifts, remembered,
                                  table_gate, teach)
from fcpg.v21_horizon_training import HorizonTargets
from fcpg.v21_models import CommandTrace

JACOBIAN = np.asarray([[-5.5, -1.5, -2.0, -1.2, 0.5],
                       [5.0, -2.3, -3.9, -3.6, -2.1]])          # the 10/11 fit, rounded (as test_v21_alignment_cem)
Q = np.asarray([-14., 15., 14., 55., -70., 30.])


def follows_commands(q):
    """An ideal LeWM: every read-back is the command before it."""
    return lambda commands: np.concatenate((np.repeat(q[None, None], len(commands), 0), np.asarray(commands)), 1)


def token(dx_px, dy_px, seen=1.):
    return np.asarray([seen, 1., dx_px / IMAGE, dy_px / IMAGE, .1, .05, 1., 0.])


def demonstration(q=Q, grip=31.):
    """The demonstrator's next five commands: a little further along, jaws open."""
    block = q[None] + np.linspace(.4, 2., 5)[:, None] * np.asarray([1., -.5, .5, .2, 0., 0.])[None]
    block[:, 5] = grip
    return block


@dataclass(frozen=True)
class NoGeometry(GeometricResponse):
    def local(self, q, offset_px):
        return None


@dataclass(frozen=True)
class Chain:
    object_relative: torch.Tensor


def test_the_first_shift_is_none_and_the_others_stay_inside_the_limit():
    shifts = draw_shifts(np.random.default_rng(3), 16, 60.)
    assert shifts.shape == (16, 2) and (shifts[0] == 0).all() and (np.abs(shifts) <= SHIFT_LIMIT_PX).all()
    assert (np.abs(shifts[1:]) > 0).all()
    assert np.array_equal(shifts, draw_shifts(np.random.default_rng(3), 16, 60.))
    with pytest.raises(ValueError):
        draw_shifts(np.random.default_rng(3), 0, 35.)


def test_the_label_is_the_cems_own_selection_with_the_demonstrators_gripper():
    response = WristResponse(JACOBIAN, 66, 'test')
    proposal = demonstration()
    shifts = np.asarray([[0., 0.], [40., 0.], [-40., 0.]])
    commands, usable, records = teach(Q, proposal, token(-30., -20.), follows_commands(Q), lambda c: True, response,
                                      shifts, seed=5)
    assert commands.shape == (3, 5, 6) and usable.all()
    assert np.array_equal(commands[:, :, 5], np.repeat(proposal[None, :, 5], 3, 0))    # the gripper is not taught
    for k, shift in enumerate(shifts):
        moved = token(-30., -20.)
        moved[2:4] += shift / IMAGE
        selected, record = AlignmentCEM(response, steer_approach=True, seed=5 + k).plan(
            Q, proposal, moved, follows_commands(Q), lambda c: True, commit=False)
        assert np.allclose(commands[k, :, :ARM], selected[:, :ARM])                  # exactly the robot planner's
        assert records[k]['selected_cost_px'] == pytest.approx(record['selected_cost_px'])
        assert records[k]['selected_cost_px'] < records[k]['proposal_cost_px']
        offset = commands[k, -1, :ARM] - proposal[-1, :ARM]
        assert (np.abs(offset[list(SEARCH_JOINTS)]) <= MAX_OFFSET + 1e-9).all() and abs(offset[4]) < 1e-12
    # the fish further right in the picture asks for a different pan than the fish further left
    assert commands[1, -1, 0] != pytest.approx(commands[2, -1, 0], abs=.5)


def test_a_shift_the_geometry_cannot_place_is_not_a_label():
    response = NoGeometry(np.r_[150., 112., 112., np.zeros(9)], kinematics=None, source='none')
    commands, usable, records = teach(Q, demonstration(), token(-30., -20.), follows_commands(Q), lambda c: True,
                                      response, np.zeros((2, 2)))
    assert not usable.any() and {r['reason'] for r in records} == {'geometry_unavailable'}
    assert np.array_equal(commands, np.repeat(demonstration()[None], 2, 0))


def test_only_the_open_jaw_fish_approach_is_taught():
    response = WristResponse(JACOBIAN, 66, 'test')
    with pytest.raises(ValueError, match='seen fish token and open jaws'):
        teach(Q, demonstration(), token(-30., -20., seen=0.), follows_commands(Q), lambda c: True, response,
              np.zeros((1, 2)))
    closed = Q.copy()
    closed[5] = OPEN_GRIP - 1.
    with pytest.raises(ValueError, match='seen fish token and open jaws'):
        teach(closed, demonstration(closed), token(-30., -20.), follows_commands(closed), lambda c: True, response,
              np.zeros((1, 2)))


def test_a_refused_selection_takes_the_robots_admission_path():
    response = WristResponse(JACOBIAN, 66, 'test')
    proposal = demonstration()
    # the gate admits only the demonstrator's own block: the CEM keeps it
    admitted = lambda c: bool(np.allclose(c, proposal))
    commands, usable, records = teach(Q, proposal, token(-30., -20.), follows_commands(Q), admitted, response,
                                      np.zeros((1, 2)))
    assert usable.all() and np.allclose(commands[0], proposal) and records[0]['reason'] == 'not_admitted'


def test_the_forecast_is_asked_once_per_block():
    calls = []

    def forecast(commands):
        calls.append(np.asarray(commands).copy())
        return follows_commands(Q)(commands)
    cached = remembered(forecast)
    block = demonstration()[None]
    first = cached(block)
    first[0, 0, 0] = 1e9                                   # a caller's change does not reach the memory
    assert np.array_equal(cached(block), follows_commands(Q)(block)) and len(calls) == 1
    cached(block + 1.)
    assert len(calls) == 2


def test_the_admission_is_the_services_chunk_audit():
    from scripts.serve_so101_prod_v17_7 import _audit_v177_chunk
    gate = table_gate(-10.)
    assert gate.physical_floor_mm == gate.planned_clearance_mm == gate.readback_clearance_mm == -10.
    home = np.asarray([-0.1, -97.0, 99.0, 61.9, -53.0, 31.5])
    admissible = admission(gate, home)
    for change in (np.zeros(6), np.asarray([5., 0., 0., 0., 0., 0.]), np.asarray([0., 40., -40., 0., 0., 0.]),
                   np.asarray([0., 90., 0., -60., 0., 0.])):
        block = home[None] + np.linspace(.2, 1., 5)[:, None] * change[None]
        assert admissible(block) == bool(_audit_v177_chunk(block.astype(np.float32), home.astype(np.float32),
                                                           clearance_gate=gate)['safe'])
    assert admissible(np.repeat(home[None], 5, 0))


def table(rows=(7, 9), k=3):
    n = len(rows)
    shifts = np.zeros((n, k, 2), np.float32)
    shifts[:, 1:] = np.asarray([[20., -10.], [-30., 5.]][:k - 1], np.float32)
    commands = np.zeros((n, k, 5, 6), np.float32)
    for i in range(n):
        for j in range(k):
            commands[i, j] = 100. * (i + 1) + 10. * j + np.arange(6)[None]
    return TeacherLabels(dict(rows=np.asarray(rows), contexts=np.asarray([2] * n, np.int8), shifts=shifts,
                              commands=commands, usable=np.ones((n, k), bool),
                              proposal_cost=np.ones((n, k), np.float32), selected_cost=np.zeros((n, k), np.float32)),
                         dict(split='train', smoke=False))


def targets_for(b):
    commands = torch.arange(b * 40 * 6, dtype=torch.float32).reshape(b, 40, 6)
    valid = torch.ones(b, 40, dtype=torch.bool)
    trace = CommandTrace(commands[:, :5].clone(), torch.zeros(b, 5), torch.ones(b, 5, dtype=torch.bool), 'recorded', True)
    offsets = torch.arange(40, dtype=torch.float64)[None].repeat(b, 1) * .1
    return HorizonTargets(trace, offsets, offsets + .1, commands, commands.clone(), valid, np.zeros(b, np.int64))


def chain_for(b):
    cue = torch.zeros(b, 5, 8)
    cue[:, :, 0] = 1.
    cue[:, :, 2] = .1
    cue[:, :, 3] = -.05
    return Chain(cue)


def test_relabel_puts_in_the_label_and_its_token_shift_for_labelled_rows_only():
    labels = table()
    rows = np.asarray([7, 8, 9])                                  # 8 has no label
    before_targets, before_chain = targets_for(3), chain_for(3)
    targets, chain, labelled, shifted = labels.relabel(rows, 2, before_targets, before_chain,
                                                       np.random.default_rng(0), 0.)
    assert labelled == 2 and shifted == 2
    for i, j in ((0, 0), (2, 1)):
        k = [kk for kk in range(1, 3) if np.allclose(targets.commands[i, :5, :ARM].numpy(),
                                                    labels.arrays['commands'][j, kk, :, :ARM])]
        assert len(k) == 1                                      # one of the shifted labels, arm joints only
        assert torch.equal(targets.commands[i, :5, 5], before_targets.commands[i, :5, 5])
        assert torch.equal(targets.commands[i, 5:], before_targets.commands[i, 5:])
        moved = before_chain.object_relative[i, 2, 2:4] + torch.as_tensor(labels.arrays['shifts'][j, k[0]] / IMAGE)
        assert torch.allclose(chain.object_relative[i, 2, 2:4], moved)
        others = [c for c in range(5) if c != 2]
        assert torch.equal(chain.object_relative[i, others], before_chain.object_relative[i, others])
        assert torch.equal(chain.object_relative[i, 2, [0, 1, 4, 5, 6, 7]],
                           before_chain.object_relative[i, 2, [0, 1, 4, 5, 6, 7]])
    assert torch.equal(targets.commands[1], before_targets.commands[1])
    assert torch.equal(chain.object_relative[1], before_chain.object_relative[1])
    # the inputs themselves are not changed
    assert torch.equal(before_targets.commands, targets_for(3).commands)


def test_relabel_keeps_the_fish_where_it_is_for_the_unshifted_share():
    labels = table()
    targets, chain, labelled, shifted = labels.relabel(np.asarray([7, 9]), 2, targets_for(2), chain_for(2),
                                                       np.random.default_rng(0), 1.)
    assert labelled == 2 and shifted == 0
    assert np.allclose(targets.commands[0, :5, :ARM].numpy(), labels.arrays['commands'][0, 0, :, :ARM])
    assert torch.equal(chain.object_relative, chain_for(2).object_relative)
    # another context, or a row whose first five targets are not all recorded, is left alone
    _, _, none, _ = labels.relabel(np.asarray([7, 9]), 1, targets_for(2), chain_for(2), np.random.default_rng(0), 0.)
    short = targets_for(2)
    short = replace(short, valid=short.valid.clone().index_fill_(1, torch.tensor([3]), False))
    _, _, cut, _ = labels.relabel(np.asarray([7, 9]), 2, short, chain_for(2), np.random.default_rng(0), 0.)
    assert none == 0 and cut == 0


def test_relabel_draws_both_shifted_and_unshifted_labels():
    labels = table(rows=tuple(range(200)))
    _, _, labelled, shifted = labels.relabel(np.arange(200), 2, targets_for(200), chain_for(200),
                                             np.random.default_rng(1), .5)
    assert labelled == 200 and 70 < shifted < 130


def test_a_table_round_trips_and_refuses_a_changed_file(tmp_path):
    labels = table()
    manifest = labels.save(tmp_path / 'labels')
    assert manifest['schema'] == SCHEMA and json.loads((tmp_path / 'labels' / 'manifest.json').read_text()) == manifest
    loaded = TeacherLabels.load(tmp_path / 'labels')
    assert all(np.array_equal(loaded.arrays[k], labels.arrays[k]) for k in labels.arrays)
    with np.load(tmp_path / 'labels' / 'labels.npz') as stored:
        arrays = {k: stored[k] for k in stored.files}
    arrays['commands'] = arrays['commands'] + 1.
    np.savez(tmp_path / 'labels' / 'labels.npz', **arrays)
    with pytest.raises(ValueError, match='not the file its manifest describes'):
        TeacherLabels.load(tmp_path / 'labels')


def test_a_table_refuses_a_pair_twice_or_a_first_shift():
    arrays = dict(table().arrays)
    with pytest.raises(ValueError, match='labelled twice'):
        TeacherLabels(dict(arrays, rows=np.asarray([7, 7])), {})
    shifts = arrays['shifts'].copy()
    shifts[0, 0] = 1.
    with pytest.raises(ValueError, match='first shift'):
        TeacherLabels(dict(arrays, shifts=shifts), {})
