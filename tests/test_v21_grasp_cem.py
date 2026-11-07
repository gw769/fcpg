"""Synthetic scoring/planning regressions; no robot, camera, or live API IO.

The image response is fitted from ten independent simulated observations.
Positive clearance and no-op physical checks are simulation fixtures only;
passing these tests supplies no physical alignment or grasp evidence.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.prod_v20_policy import empty_history
from fcpg.v21_cem_trace import CEMTrace
from fcpg.v21_grasp_cem import GraspCEMModel, plan_grasp_commands
from fcpg.v21_grasp_cost import (
    WristGraspCost,
    distance_alignment_scores,
    line_error_rad,
)
from fcpg.v21_lewm_cem import INADMISSIBLE_COST, MAX_STEP
from fcpg.v21_models import CommandTrace, V21LeWM
from fcpg.v21_visual_alignment import alignment_observation, fit_measured_response


SOURCE_Q = np.array([-5., -25., 35., 10., -65., 30.])
PROBE_SIZE = np.array([3., 3., 3., 2., 2.])
LOWER = np.array([-100.] * 5 + [0.])
UPPER = np.full(6, 100.)
JAW = np.array([[92., 112.], [132., 112.]])


def direction(degrees):
    angle = np.radians(degrees)
    return np.array([np.cos(angle), np.sin(angle)])


def simulated_observation(q=SOURCE_Q, *, angle_deg=30., stamp=1.):
    # This known image map generates the fit samples independently of the
    # fitted Jacobian and production cost/line-angle functions.
    delta = np.asarray(q) - SOURCE_Q
    center = [132. - 6. * delta[0] - delta[2] + .5 * delta[3],
              112. - 5. * delta[1] + .5 * delta[2] - delta[3]]
    body_axis = direction(90. + angle_deg + 8. * delta[4])
    record = dict(role='pickup', view='wrist', visible=True,
                  view_source='current_observation', source='operator_colour_body',
                  image_time_s=stamp, cog_px=center, axis_px=body_axis,
                  width_px=20., length_px=40.)
    return alignment_observation(record, q, JAW, image_time_s=stamp,
                                 q_time_s=stamp + .01, now_s=stamp + .02,
                                 max_age_s=.2)


def simulated_fit(*, stationary=None):
    pairs = []
    for joint, size in enumerate(PROBE_SIZE):
        for sign in (-1., 1.):
            stamp = 2. * len(pairs) + 1.
            q = SOURCE_Q.copy()
            q[joint] += sign * size
            before = simulated_observation(stamp=stamp)
            after = simulated_observation(q, stamp=stamp + 1.)
            if stationary in ('distance', 'both'):
                after = replace(after, target_center_px=before.target_center_px)
            if stationary in ('orientation', 'both'):
                after = replace(after, target_axis_px=before.target_axis_px)
            pairs.append((before, after))
    return fit_measured_response(pairs, unmodelled_gripper_tolerance=0.)


@pytest.fixture
def task_factory():
    response = simulated_fit()

    def make(angle_deg=30., **kwargs):
        return WristGraspCost(simulated_observation(angle_deg=angle_deg), response,
                              **kwargs)

    return make


def fake_state():
    return SimpleNamespace(q=torch.tensor(SOURCE_Q, dtype=torch.float32)[None],
                           history=empty_history(1))


def simulated_physical_check(q, commands):
    """Deliberate no-op: no real geometry or physical evidence is supplied."""


def simulated_clearance(q):
    """Positive clearance only inside this synthetic planning environment."""
    return 30.


class FakeLeWM:
    def __init__(self, gain=.8):
        self.gain = gain
        self.calls = []

    def __call__(self, q, trace, history, query):
        trace.validate(len(q), max_commands=256)
        assert trace.positions.shape == (len(q), 40, 6)
        assert trace.source == 'planned' and trace.complete
        assert trace.valid.all()
        assert history.commands_position.shape == (len(q), 3, 5, 6)
        assert not history.valid.any()
        expected_query = torch.arange(1, 41, dtype=query.dtype)[None] * .1
        torch.testing.assert_close(query, expected_query.expand_as(query))
        torch.testing.assert_close(trace.offsets, query - query[:, :1])
        gain = self.gain(q, trace.positions) if callable(self.gain) else self.gain
        future = q[:, None] + gain * (trace.positions - q[:, None])
        result = SimpleNamespace(predicted_q_path=torch.cat((q[:, None], future), 1),
                                 q_uncertainty=torch.ones_like(future))
        self.calls.append(SimpleNamespace(commands=trace.positions.detach().clone(),
                                          predicted=result.predicted_q_path.detach().clone()))
        return result


def model_for(task, lewm=None, **kwargs):
    return GraspCEMModel(FakeLeWM() if lewm is None else lewm, fake_state(), task,
                         simulated_physical_check, lower=LOWER, upper=UPPER,
                         clearance=kwargs.pop('clearance', simulated_clearance),
                         **kwargs)


def constant_future(*, steps=40, j1=0., j5=0.):
    path = np.repeat(SOURCE_Q[None], steps + 1, axis=0)
    path[1:, 0] += j1
    path[1:, 4] += j5
    return path


def rollout(model, deltas):
    endpoints = model.q[:, :5] + model.q.new_tensor(deltas)
    return model.rollout_embeddings(None, None, endpoints[None, :, None])


def run_plan(task, lewm, *, trace=None, physical_check=simulated_physical_check,
             actor=None, dt=None, clearance=simulated_clearance):
    state = fake_state()
    if actor is None:
        actor = state.q[:, None].expand(1, 40, 6).clone()
        # The seed suggests moving away, worsening roll, and closing the jaw.
        # A valid alignment/approach plan must still search for improvement.
        actor[:, :, 0] -= 1.
        actor[:, :, 4] += 1.
        actor[:, :, 5] = 0.
    return plan_grasp_commands(
        lewm, state, actor, torch.full((1, 5), .1) if dt is None else dt,
        physical_check, task=task, lower=LOWER, upper=UPPER,
        clearance=clearance, generator=torch.Generator().manual_seed(71),
        trace=trace, population=24, elites=6, iterations=3)


@pytest.mark.parametrize('first,second,expected', [
    (0., 0., 0.), (0., 180., 0.), (180., 0., 0.), (0., 90., 90.),
    (89., -89., 2.), (-89., 89., 2.),
])
def test_unoriented_line_error(first, second, expected):
    actual = np.degrees(line_error_rad(direction(first), direction(second)))
    assert abs(actual) == pytest.approx(expected, abs=1e-12)
    assert -90. <= actual <= 90.


@pytest.mark.parametrize('swap_jaws', [False, True])
@pytest.mark.parametrize('swap_fish_axis', [False, True])
def test_jaw_and_fish_endpoint_swaps_preserve_both_costs(swap_jaws, swap_fish_axis):
    jaw = JAW[::-1] if swap_jaws else JAW
    axis = direction(120.) * (-1. if swap_fish_axis else 1.)
    score = distance_alignment_scores([115., 116.], jaw, axis)
    assert score['distance_px'] == pytest.approx(5.)
    assert score['angle_error_deg'] == pytest.approx(30.)
    assert score['distance_score'] == pytest.approx(5. / 224.)
    assert score['alignment_score'] == pytest.approx(30. / 90.)


@pytest.mark.parametrize('fish_angle,expected', [(0., 90.), (90., 0.),
                                               (180., 90.), (270., 0.)])
def test_blue_line_is_perpendicular_to_fish_long_axis(fish_angle, expected):
    score = distance_alignment_scores([112., 112.], JAW, direction(fish_angle))
    assert score['angle_error_deg'] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('bad', [[0., 0.], [np.nan, 1.], [np.inf, 0.], [1., 2., 3.]])
def test_invalid_line_direction_fails_closed(bad):
    with pytest.raises(ValueError, match='finite and nondegenerate'):
        line_error_rad(bad, [1., 0.])
    with pytest.raises(ValueError, match='finite and nondegenerate'):
        line_error_rad([1., 0.], bad)


def test_fit_uses_ten_independent_simulated_motion_samples(task_factory):
    response = task_factory().response
    assert response.sample_count == 10
    assert response.design_condition == pytest.approx(1.)
    np.testing.assert_allclose(response.reference_q, SOURCE_Q)
    np.testing.assert_allclose(response.measured_support, PROBE_SIZE)
    np.testing.assert_allclose(response.observed_radius, PROBE_SIZE)
    np.testing.assert_allclose(response.jacobian,
        [[-6., 0., -1., .5, 0.], [0., -5., .5, -1., 0.],
         [0., 0., 0., 0., np.radians(8.)]], atol=1e-12)


@pytest.mark.parametrize('steps', [5, 40])
def test_cost_is_exactly_weighted_distance_plus_alignment(task_factory, steps):
    task = task_factory(distance_weight=2., alignment_weight=3.)
    score = task.evaluate(constant_future(steps=steps, j1=1., j5=-1.))
    assert score['distance_cost'] == pytest.approx(2. * 14. / 224.)
    assert score['alignment_cost'] == pytest.approx(3. * 22. / 90.)
    assert score['cost'] == pytest.approx(score['distance_cost'] + score['alignment_cost'])
    np.testing.assert_allclose(score['distance_px'], [20.] + [14.] * steps)
    np.testing.assert_allclose(score['angle_error_deg'], [30.] + [22.] * steps)
    np.testing.assert_allclose(score['predicted_centres_px'][1:], [[126., 112.]] * steps)
    assert score['prediction_source'] == 'lewm_actual_q_via_measured_moving_wrist_response'


def test_native40_weights_first_five_and_remaining_35_equally(task_factory):
    path = constant_future(j1=1., j5=-1.)
    path[6:, 0] = SOURCE_Q[0] + 2.
    path[6:, 4] = SOURCE_Q[4] - 2.
    score = task_factory().evaluate(path)
    assert score['distance_cost'] == pytest.approx((14. + 8.) / 2. / 224.)
    assert score['alignment_cost'] == pytest.approx((22. + 14.) / 2. / 90.)


def test_each_visual_term_changes_ranking_independently(task_factory):
    task = task_factory()
    hold = task.evaluate(constant_future())
    nearer = task.evaluate(constant_future(j1=1.))
    aligned = task.evaluate(constant_future(j5=-1.))
    assert nearer['cost'] < hold['cost'] and aligned['cost'] < hold['cost']
    assert nearer['alignment_cost'] == pytest.approx(hold['alignment_cost'])
    assert aligned['distance_cost'] == pytest.approx(hold['distance_cost'])


def test_wrist_cost_rejects_half_turn_hidden_by_unoriented_line_score(task_factory):
    task = task_factory()
    flipped_axis = -task.observation.target_axis_px
    source = distance_alignment_scores(
        task.observation.target_center_px, task.observation.jaw_fm_px,
        task.observation.target_axis_px)
    flipped = distance_alignment_scores(
        task.observation.target_center_px, task.observation.jaw_fm_px,
        flipped_axis)
    assert flipped['angle_error_deg'] == pytest.approx(source['angle_error_deg'])

    jacobian = task.response.jacobian.copy()
    jacobian[2] = [0., 0., 0., 0., np.pi]
    task = replace(task, response=replace(task.response, jacobian=jacobian))
    with pytest.raises(ValueError, match='half-turn|branch|90'):
        task.evaluate(constant_future(j5=1.))


def test_wrist_cost_rejects_discontinuous_signed_angle_forecast(task_factory):
    task = task_factory(0.)
    jacobian = task.response.jacobian.copy()
    jacobian[2] = [0., 0., 0., 0., np.radians(80.)]
    task = replace(task, response=replace(task.response, jacobian=jacobian))
    path = constant_future()
    path[1, 4] += 1.
    path[2:, 4] -= 1.
    with pytest.raises(ValueError, match='continuous|angle step'):
        task.evaluate(path)


@pytest.mark.parametrize('stationary', ['distance', 'orientation', 'both'])
def test_stationary_image_fit_cannot_establish_two_term_support(stationary):
    response = simulated_fit(stationary=stationary)
    with pytest.raises(ValueError, match='identify both distance and orientation'):
        WristGraspCost(simulated_observation(), response)


@pytest.mark.parametrize('changes', [
    {'sample_count': 9}, {'design_condition': 101.},
    {'residual_rms': np.array([4.01, 0., 0.])},
    {'residual_rms': np.array([0., 0., np.radians(5.01)])},
    {'measured_support': np.array([0., 3., 3., 2., 2.])},
    {'observed_radius': np.array([3., 3., 3., 0., 2.])},
])
def test_unsupported_response_fit_is_rejected(task_factory, changes):
    task = task_factory()
    with pytest.raises(ValueError, match='lacks support|fit quality'):
        replace(task, response=replace(task.response, **changes))


@pytest.mark.parametrize('field,message', [
    ('observed_radius', 'measured pose support'),
    ('measured_support', 'local measured displacement'),
])
def test_entire_forecast_must_stay_in_each_measured_support(task_factory, field, message):
    task = task_factory()
    support = getattr(task.response, field).copy()
    support[0] = .5
    task = replace(task, response=replace(task.response, **{field: support}))
    path = constant_future()
    path[-1, 0] += .6
    with pytest.raises(ValueError, match=message):
        task.evaluate(path)


@pytest.mark.parametrize('delta', [-.3, .3])
def test_forecast_cannot_close_or_open_the_unmodelled_gripper(task_factory, delta):
    path = constant_future()
    path[-1, 5] += delta
    with pytest.raises(ValueError, match='gripper motion'):
        task_factory().evaluate(path)


def test_forecast_leaving_real_image_content_is_rejected(task_factory):
    task = task_factory()
    task = replace(task, observation=replace(task.observation,
                                            target_center_px=np.array([222., 112.])))
    with pytest.raises(ValueError, match='real image content'):
        task.evaluate(constant_future(j1=-1.))


@pytest.mark.parametrize('rows', [5, 40, 42])
def test_cost_requires_source_plus_native_five_or_40_responses(task_factory, rows):
    with pytest.raises(ValueError, match='q0..q5 or q0..q40'):
        task_factory().evaluate(np.repeat(SOURCE_Q[None], rows, axis=0))


def test_cost_receipt_preserves_the_synthetic_evidence_boundary(task_factory):
    receipt = task_factory().receipt()
    assert receipt['score_terms'] == ['distance', 'blue_white_alignment']
    assert receipt['angle_period_deg'] == 180
    assert receipt['angle_error_range_deg'] == [0, 90]
    assert receipt['blue_axis'] == 'perpendicular_to_fish_long_axis'
    assert receipt['sample_count'] == 10 and receipt['gripper_held']
    assert receipt['source_distance_px'] == pytest.approx(20.)
    assert receipt['source_angle_error_deg'] == pytest.approx(30.)
    assert receipt['learned_pixel_prediction'] is False
    assert receipt['physical_alignment_verified'] is False
    assert receipt['physical_grasp_verified'] is False


def test_endpoint_expansion_is_smooth_first_five_then_hold_with_open_jaw(task_factory):
    model = model_for(task_factory(0.))
    endpoint = model.q[:, :5] + model.q.new_tensor([[1., -1., .5, 1., -1.]])
    commands = model.expand(endpoint)
    assert commands.shape == (1, 40, 6)
    increments = torch.diff(torch.cat((model.q[:, None], commands[:, :5]), 1), dim=1)
    torch.testing.assert_close(increments, increments[:, :1].expand_as(increments))
    torch.testing.assert_close(commands[:, 4, :5], endpoint)
    torch.testing.assert_close(commands[:, 5:], commands[:, 4:5].expand(-1, 35, -1))
    torch.testing.assert_close(commands[:, :, 5], model.q[:, 5:6].expand(-1, 40))
    assert (increments.abs() <= increments.new_tensor(MAX_STEP)).all()


@pytest.mark.parametrize('angle,alignment_first', [(30., True), (0., False)])
def test_unaligned_search_is_pure_roll_and_aligned_search_can_translate(task_factory, angle,
                                                                     alignment_first):
    model = model_for(task_factory(angle))
    endpoints = model.q[:, None, None, :5] + 1.
    before = endpoints.clone()
    constrained = model.constrain(endpoints)
    assert model.alignment_first is alignment_first
    torch.testing.assert_close(endpoints, before)
    if alignment_first:
        torch.testing.assert_close(constrained[..., :4], model.q[:, None, None, :4])
    else:
        torch.testing.assert_close(constrained, endpoints)
    torch.testing.assert_close(constrained[..., 4], endpoints[..., 4])


@pytest.mark.parametrize('height,message', [(19.9, 'clearance before CEM'),
                                          (np.nan, 'clearance is unavailable')])
def test_rotation_requires_known_clearance_in_the_simulation(task_factory, height, message):
    with pytest.raises(ValueError, match=message):
        model_for(task_factory(), clearance=lambda q: height)


def test_alignment_gate_rejects_translation_and_loss_of_clearance(task_factory):
    model = model_for(task_factory(),
                      clearance=lambda q: 30. + float(q[4] - SOURCE_Q[4]))
    path = model.q.expand(5, -1).clone()
    assert model.path_admitted(path, command=True)
    path[:, 0] += .01
    assert not model.path_admitted(path, command=True)
    assert model.path_admitted(path)
    path[:, 0] = model.q[0, 0] + .26
    assert not model.path_admitted(path)
    path[:, 0] = model.q[0, 0]
    path[:, 4] -= 1.
    assert not model.path_admitted(path, command=True)
    assert not model.path_admitted(path)


def test_raw_cost_ranking_follows_reversed_predictions_not_command_direction(task_factory):
    model = model_for(task_factory(0.))
    commands = model.expand(model.q[:, :5] + model.q.new_tensor(
        [[1., 0., 0., 0., 0.], [-1., 0., 0., 0., 0.]]))
    model._commands = commands
    reversed_path = torch.cat((model.q[:, None].expand(2, 1, 6),
                               2. * model.q[:, None] - commands), 1)[None]
    costs = model.goal_cost(reversed_path, None)[0]
    assert costs[1] < costs[0] < INADMISSIBLE_COST
    assert model.task.evaluate(constant_future(j1=1.))['cost'] < model.task.evaluate(
        constant_future(j1=-1.))['cost']
    # Cost evaluation is isolated here. Actual candidate admission must reject
    # BOTH wrong-direction trajectories, as checked by the next regression.


def test_rollout_rejects_reversed_response_even_when_visual_cost_improves(task_factory):
    trace = CEMTrace()
    model = model_for(task_factory(0.), FakeLeWM(gain=-1.), trace=trace)
    path = rollout(model, [[1., 0., 0., 0., 0.], [-1., 0., 0., 0., 0.]])
    assert torch.isnan(path).all()
    assert (model.goal_cost(path, None) == INADMISSIBLE_COST).all()
    batches = [event['values'] for event in trace.events
               if event['event'] == 'grasp_lewm40_batch']
    assert batches and not batches[0]['prefix_response_admitted'].any()


def test_admitted_ranking_uses_actual_response_magnitude_not_command_magnitude(task_factory):
    def gain(q, commands):
        large = commands[:, 4, 0] - q[:, 0] > 1.5
        return torch.where(large, .2, 1.)[:, None, None]

    model = model_for(task_factory(0.), FakeLeWM(gain=gain))
    path = rollout(model, [[1., 0., 0., 0., 0.], [2., 0., 0., 0., 0.]])
    assert torch.isfinite(path).all()
    assert path[0, 0, 5, 0] - model.q[0, 0] == pytest.approx(1.)
    assert path[0, 1, 5, 0] - model.q[0, 0] == pytest.approx(.4)
    costs = model.goal_cost(path, None)[0]
    assert costs[0] < costs[1] < INADMISSIBLE_COST
    # The larger requested displacement would have won command-only scoring.
    task = model.task
    assert task.evaluate(constant_future(j1=2.))['cost'] < task.evaluate(
        constant_future(j1=1.))['cost']


def test_rollout_requires_a_single_native40_forecast_not_a_five_step_preview(task_factory):
    def short_forecast(q, trace, history, query):
        response = FakeLeWM()(q, trace, history, query)
        return SimpleNamespace(predicted_q_path=response.predicted_q_path[:, :6],
                               q_uncertainty=response.q_uncertainty[:, :5])

    model = model_for(task_factory(0.), short_forecast)
    with pytest.raises(ValueError, match='native40 LeWM response required'):
        rollout(model, [[1., 0., 0., 0., 0.]])


@pytest.mark.parametrize('invalid', ['late_nan', 'wrong_source', 'zero_sigma', 'nan_sigma'])
def test_invalid_native40_response_is_inadmissible(task_factory, invalid):
    def bad_forecast(q, trace, history, query):
        response = FakeLeWM()(q, trace, history, query)
        if invalid == 'late_nan':
            response.predicted_q_path[:, -1, 0] = float('nan')
        elif invalid == 'wrong_source':
            response.predicted_q_path[:, 0, 0] += .1
        else:
            response.q_uncertainty[:, -1, 0] = 0. if invalid == 'zero_sigma' else float('nan')
        return response

    model = model_for(task_factory(0.), bad_forecast)
    path = rollout(model, [[1., 0., 0., 0., 0.]])
    assert torch.isnan(path).all()
    assert model.goal_cost(path, None).item() == INADMISSIBLE_COST


@pytest.mark.parametrize('failure,message', [
    ('angle', 'first five responses worsen alignment'),
    ('retreat', 'first five responses retreat from target'),
    ('approach', 'approach loses line alignment'),
])
def test_good_lookahead_cannot_hide_bad_executed_prefix(task_factory, failure, message):
    model = model_for(task_factory(0. if failure == 'approach' else 30.))
    path = constant_future(j1=1., j5=0. if failure == 'approach' else -2.)
    if failure == 'angle':
        path[1:6, 4] = SOURCE_Q[4] + .1
    elif failure == 'retreat':
        path[1:6, 0] = SOURCE_Q[0] - 1.
    else:
        path[3, 4] = SOURCE_Q[4] + 2.
    with pytest.raises(ValueError, match=message):
        model.evaluate(torch.tensor(path))


@pytest.mark.parametrize('angle,phase', [(30., 'align_open_jaw'), (0., 'approach_open_jaw')])
def test_full_cem_returns_exactly_five_improving_open_jaw_commands(task_factory, angle, phase):
    lewm, trace, physical_calls = FakeLeWM(), CEMTrace(), []

    def physical_check(q, commands):
        # Recording no-op in simulation, including the final uncached checks.
        physical_calls.append(commands.detach().clone())

    selected, receipt = run_plan(task_factory(angle), lewm, trace=trace,
                                 physical_check=physical_check)
    assert selected.shape == (1, 5, 6)
    assert selected.is_contiguous() and torch.isfinite(selected).all()
    torch.testing.assert_close(selected[:, :, 5], torch.full((1, 5), 30.))
    increments = torch.diff(torch.cat((fake_state().q[:, None], selected), 1), dim=1)
    torch.testing.assert_close(increments, increments[:, :1].expand_as(increments),
                               atol=1e-5, rtol=1e-5)
    assert (increments.abs() <= selected.new_tensor(MAX_STEP) + 1e-6).all()
    assert receipt['phase'] == phase
    assert receipt['inferred_steps'] == receipt['cem_forecast_steps'] == 40
    assert receipt['execute_steps'] == 5 and not receipt['cem_bypassed']
    assert not receipt['small_step_extension_enabled']
    assert receipt['ranking_state_source'] == 'lewm_predicted_readback'
    assert receipt['candidate_parameterization'] == 'five_step_endpoint_then_hold_to_40'
    assert receipt['selected_scores']['cost'] < receipt['hold_scores']['cost']
    assert not receipt['physical_alignment_verified'] and not receipt['physical_grasp_verified']
    if angle:
        torch.testing.assert_close(selected[:, :, :4], fake_state().q[:, None, :4].expand(-1, 5, -1))
        assert selected[0, -1, 4] < SOURCE_Q[4]
        assert receipt['selected_scores']['angle_error_deg'][5] < receipt['hold_scores']['angle_error_deg'][5] - .25
    else:
        assert receipt['selected_scores']['distance_px'][5] < receipt['hold_scores']['distance_px'][5]
        assert max(receipt['selected_scores']['angle_error_deg'][1:6]) <= 15.
    assert lewm.calls and receipt['evaluated_candidates'] == sum(len(call.commands) for call in lewm.calls)
    for call in lewm.calls:
        torch.testing.assert_close(call.commands[:, 5:], call.commands[:, 4:5].expand(-1, 35, -1))
        torch.testing.assert_close(call.commands[:, :, 5], torch.full_like(call.commands[:, :, 5], 30.))
        if angle:
            torch.testing.assert_close(call.commands[:, :, :4], fake_state().q[:, None, :4].expand(len(call.commands), 40, -1))
    predicted = torch.tensor(receipt['predicted_q_path40'])
    assert predicted.shape == (41, 6)
    torch.testing.assert_close(physical_calls[-12], selected)
    torch.testing.assert_close(physical_calls[-6], predicted[None, 1:6])
    for start in (-12, -6):
        for joint in range(5):
            stalled = physical_calls[start].clone()
            stalled[:, :, joint] = SOURCE_Q[joint]
            torch.testing.assert_close(physical_calls[start + joint + 1], stalled)
    payload = trace.payload()
    assert payload['complete'] and payload['error'] is None and not payload['executable']
    assert payload['rng_before']['generator_state'] is not None
    assert payload['rng_after']['generator_state'] is not None
    events = payload['events']
    assert events[0]['event'] == 'planning_input' and events[-1]['event'] == 'selected'
    assert {'cem_iteration', 'grasp_lewm40_batch', 'grasp_candidate_scores'} <= {event['event'] for event in events}
    torch.testing.assert_close(events[-1]['values']['commands'], selected)
    score_events = [event['values'] for event in events if event['event'] == 'grasp_candidate_scores']
    for event in score_events:
        assert event['commands'].shape[1:] == (40, 6)
        for component in event['components']:
            if 'cost' in component:
                assert component['cost'] == pytest.approx(component['distance_cost'] + component['alignment_cost'])


def test_stationary_lewm_cannot_claim_visual_optimization(task_factory):
    trace = CEMTrace()
    with pytest.raises(RuntimeError, match='insensitive'):
        run_plan(task_factory(0.), FakeLeWM(gain=0.), trace=trace)
    payload = trace.payload()
    assert payload['complete'] is False and 'insensitive' in payload['error']
    assert all(event['event'] != 'selected' for event in payload['events'])


def test_trace_finishes_when_initial_clearance_rejects_alignment(task_factory):
    trace = CEMTrace()
    lewm = FakeLeWM()
    with pytest.raises(ValueError, match='clearance before CEM'):
        run_plan(task_factory(), lewm, trace=trace, clearance=lambda q: 19.)
    payload = trace.payload()
    assert payload['complete'] is False and 'clearance before CEM' in payload['error']
    assert payload['rng_before'] is not None and payload['rng_after'] is not None
    assert not lewm.calls
    assert [event['event'] for event in payload['events']] == ['planning_input']


@pytest.mark.parametrize('bad', ['actor5', 'dt40', 'wrong_interval'])
def test_planner_enforces_native40_input_and_five_point_execution_grid(task_factory, bad):
    actor = fake_state().q[:, None].expand(1, 5 if bad == 'actor5' else 40, 6).clone()
    dt = torch.full((1, 40 if bad == 'dt40' else 5), .2 if bad == 'wrong_interval' else .1)
    message = 'native40 Actor' if bad == 'actor5' else 'five commands'
    with pytest.raises(ValueError, match=message):
        run_plan(task_factory(), FakeLeWM(), actor=actor, dt=dt)


def test_real_native40_lewm_prefix_matches_runtime_five_step_forecast(task_factory):
    # A real CPU model checks causality; neither response admission nor the
    # production trace constructor is replaced to make planning tests pass.
    state = fake_state()
    model = model_for(task_factory(0.))
    commands = model.expand(state.q[:, :5] + state.q.new_tensor([[1., -1., .5, 0., -1.]]))
    dt = torch.full((1, 40), .1)
    query = dt.double().cumsum(-1)
    offsets = torch.cat((torch.zeros(1, 1, dtype=torch.float64), query[:, :-1]), -1)
    full_trace = CommandTrace(commands, offsets, torch.ones(1, 40, dtype=torch.bool), 'planned', True)
    prefix_trace = CommandTrace.policy(commands[:, :5], dt[:, :5])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        lewm = V21LeWM(horizon=40).eval()
        with torch.no_grad():
            native = lewm(state.q, full_trace, state.history, query)
            prefix = lewm(state.q, prefix_trace, state.history, query[:, :5])
            altered_commands = commands.clone()
            altered_commands[:, 5:, :4] += 2.
            changed_tail = lewm(state.q, replace(full_trace, positions=altered_commands),
                                 state.history, query)
    torch.testing.assert_close(native.predicted_q_path[:, :6], prefix.predicted_q_path,
                               atol=2e-5, rtol=1e-6)
    torch.testing.assert_close(native.q_uncertainty[:, :5], prefix.q_uncertainty,
                               atol=2e-5, rtol=1e-6)
    torch.testing.assert_close(native.predicted_q_path[:, :6], changed_tail.predicted_q_path[:, :6],
                               atol=2e-5, rtol=1e-6)
