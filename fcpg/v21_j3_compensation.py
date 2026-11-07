"""One fixed J3 compensation recipe; preparation never authorizes execution."""
from copy import deepcopy

import numpy as np

from .v21_j3_range_profile import range_offsets


def _array(value, shape, label):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(label + ' must be finite numeric data') from error
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(label + ' must be finite with shape ' + str(shape))
    return result.copy()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _recipe(old_source, source_q):
    old = _array(old_source, (6,), 'old_source')
    current = _array(source_q, (6,), 'source_q')
    original = old.copy()
    original[2] -= 10.
    _require(original[2] <= current[2] <= old[2],
             'current J3 is outside the old source to original target interval')
    other = [0, 1, 3, 4, 5]
    _require(np.max(np.abs(current[other] - old[other])) <= .2,
             'other actual axes differ from the old source by more than 0.2')
    goal = current.copy()
    goal[2] = old[2] - 12.3
    _require(abs(current[2] - goal[2]) <= 5.,
             'remaining compensation displacement exceeds 5 calibrated units')
    commands = np.repeat(current[None], 5, axis=0)
    commands[:, 2] = np.maximum(goal[2], current[2] - 2.5 * np.arange(1, 6))
    lower, upper = np.array([-100.] * 5 + [0.]), np.full(6, 100.)
    for positions in (old, current, original, goal, commands):
        _require(np.all(positions >= lower) and np.all(positions <= upper),
                 'fixed compensation leaves original calibrated position bounds')
    _require(np.max(np.abs(np.diff(np.vstack((current, commands)), axis=0))) <= 2.5 + 1e-10,
             'compensation exceeds the fixed per-step bound')
    _require(np.array_equal(commands[-1], goal), 'five commands cannot reach the fixed compensation goal')
    return dict(schema='v21-j3-compensation-recipe-v1',
        old_source=old.tolist(), source_q=current.tolist(), original_target=original.tolist(),
        compensated_goal=goal.tolist(), commands=commands.tolist(), command_budget=5,
        period_s=.1, maximum_step=2.5, maximum_remaining_displacement=5.,
        original_amplitude=10., compensated_amplitude=12.3,
        p=16, gain_writes=0, position_unit='lerobot_calibrated_position',
        requires_expanded_amplitude_authorization=True, automatic_retry=False,
        model_history_imported=False)


def prepare_j3_compensation(range10_report, current_actual_q):
    """Validate a saved completed range10 report and prepare exactly five goals.

    Structural validation cannot prove a dict came from hardware; the caller
    must supply the existing actual range10 receipt and fresh actual position.
    The original target remains old q0 minus 10; the command goal is minus 12.3.
    """
    report = range10_report
    _require(isinstance(report, dict), 'completed range10 report required')
    _require(report.get('schema') == 'v21-j3-range-diagnostic-v1'
             and report.get('completed') is True and report.get('amplitude') == 10
             and report.get('command_budget') == 10 and report.get('period_s') == .1
             and report.get('offsets') == list(range_offsets(10))
             and report.get('p') == 16 and report.get('gain_writes') == 0
             and report.get('drive_attempted') is True
             and report.get('model_history_imported') is False
             and report.get('configuration_unchanged_verified') is True,
             'saved report is not the completed unchanged P16 range10 diagnostic')
    drive = report.get('drive_result')
    _require(isinstance(drive, dict), 'range10 actual drive receipt required')
    _require(drive.get('completed') is True and drive.get('action_source') == 'j3_range_diagnostic'
             and drive.get('diagnostic_total_amplitude') == 10
             and drive.get('not_model_transition') is True and drive.get('model_feedback_label') is False
             and drive.get('original_geometry_admitted') is True
             and drive.get('protective_hold_after_q10') is True
             and drive.get('gain_writes') == 0 and drive.get('maximum_lateness_s') == .03
             and drive.get('normal_follow', {}).get('budget_s') == .4,
             'range10 actual drive contract differs')
    for key in ('policy_commands_attempted', 'policy_commands_returned',
                'policy_goal_reads_attempted', 'policy_goal_reads_returned', 'policy_goal_registers_verified'):
        _require(type(drive.get(key)) is int and drive[key] == 10, 'incomplete range10 ' + key)
    old = _array(drive.get('source_q_position'), (6,), 'old actual q0')
    measured = _array(drive.get('measured_q_path'), (11, 6), 'range10 actual readbacks')
    expected = np.repeat(old[None], 10, axis=0)
    expected[:, 2] += np.asarray(range_offsets(10))
    for label, value in (('report commands', report.get('commands_position')),
                         ('drive commands', drive.get('commands_position'))):
        _require(np.allclose(_array(value, (10, 6), label), expected, atol=1e-4, rtol=0),
                 'range10 commands differ from actual q0 and fixed offsets')
    _require(np.array_equal(old, measured[0]), 'range10 actual q0 differs from first readback')
    qt = _array(drive.get('readback_times_s'), (11,), 'range10 readback times')
    ct = _array(drive.get('command_times_s'), (10,), 'range10 command times')
    wt = _array(drive.get('write_completed_times_s'), (10,), 'range10 write times')
    _require(np.all(np.diff(qt) > 0) and np.all(np.diff(ct) > 0)
             and np.all((qt[:-1] <= ct) & (ct <= wt) & (wt <= qt[1:])),
             'range10 command and actual readback clocks do not correspond')
    goals = drive.get('policy_goal_readbacks', [])
    _require(len(goals) == 10 and all(isinstance(item, dict)
             and type(item.get('command_index')) is int and item['command_index'] == i
             for i, item in enumerate(goals)), 'range10 ordered Goal evidence is missing')
    return _recipe(old, current_actual_q)


def validate_j3_compensation_recipe(recipe):
    """Rebuild the only supported recipe; reject altered targets or commands."""
    _require(isinstance(recipe, dict), 'fixed compensation recipe required')
    expected = _recipe(recipe.get('old_source'), recipe.get('source_q'))
    for key, value in expected.items():
        if key in ('old_source', 'source_q', 'original_target', 'compensated_goal', 'commands'):
            actual = _array(recipe.get(key), np.asarray(value).shape, key)
            _require(np.array_equal(actual, np.asarray(value)), 'fixed compensation recipe differs: ' + key)
        else:
            _require(type(recipe.get(key)) is type(value) and recipe[key] == value,
                     'fixed compensation recipe differs: ' + key)
    return deepcopy(expected)


def bind_j3_compensation_wire(recipe, present_raw, *, normalize_raw, encode_positions):
    """Bind to actual encoder cells using the owner's pure local conversions.

    Conversion callbacks must do no IO. Ideal target/goal/commands stay separate
    from the exact float32 commands that map into the approved raw cells.
    """
    fixed = validate_j3_compensation_recipe(recipe)
    raw = _array(present_raw, (6,), 'actual present_raw')
    _require(np.equal(raw, np.floor(raw)).all(), 'actual present_raw must be integer cells')
    raw = raw.astype(np.int64)
    current = _array(normalize_raw(raw.copy()), (6,), 'normalized actual cells')
    _require(np.allclose(current, fixed['source_q'], atol=1e-4, rtol=0),
             'source actual position differs from the actual raw cells')
    next_raw = raw.copy()
    next_raw[2] += 1
    j3_tick = float(_array(normalize_raw(next_raw), (6,), 'J3 next encoder cell')[2] - current[2])
    _require(j3_tick > 0 and np.isfinite(j3_tick), 'fixed J3 requires positive calibrated encoder direction')
    max_ticks = int(np.floor(2.5 / j3_tick + 1e-9))
    _require(max_ticks > 0, 'J3 encoder cannot express the bounded step')
    ideal_goal = np.asarray(fixed['compensated_goal'])
    raw_goal = _array(encode_positions(ideal_goal.copy()), (6,), 'encoded compensation goal')
    _require(np.equal(raw_goal, np.floor(raw_goal)).all(), 'encoded goal must be integer cells')
    terminal = raw.copy()
    terminal[2] = int(raw_goal[2])
    # Truncation can land below the fixed maximum amplitude; select the nearest
    # representable J3 position on the original-source side of the ideal goal.
    for _ in range(3):
        represented = _array(normalize_raw(terminal.copy()), (6,), 'represented goal')
        if represented[2] >= ideal_goal[2]:
            break
        terminal[2] += 1
    represented = _array(normalize_raw(terminal.copy()), (6,), 'represented goal')
    _require(ideal_goal[2] <= represented[2] < ideal_goal[2] + j3_tick + 1e-9,
             'wire goal exceeds the fixed compensation bound or misses its nearest cell')
    raw_commands = np.repeat(raw[None], 5, axis=0)
    raw_commands[:, 2] = np.maximum(terminal[2], raw[2] - max_ticks * np.arange(1, 6))
    _require(np.array_equal(raw_commands[-1], terminal), 'five bounded raw steps cannot reach the wire goal')
    wire_commands = []
    represented_commands = []
    for cells in raw_commands:
        position = _array(normalize_raw(cells.copy()), (6,), 'wire cell position')
        # Mid-cell float32 values survive truncation for every unchanged axis.
        upper_cell = _array(normalize_raw(cells + 1), (6,), 'wire cell upper boundary')
        command = ((position + upper_cell) * .5).astype(np.float32)
        actual = _array(encode_positions(command.copy()), (6,), 'float32 encoded command')
        _require(np.array_equal(actual, cells), 'float32 command does not preserve approved encoder cells')
        _require(command[2] >= ideal_goal[2], 'float32 J3 command exceeds the fixed compensation amplitude')
        wire_commands.append(command.tolist())
        represented_commands.append(position.tolist())
    actual_path = np.vstack((current, represented_commands))
    _require(np.max(np.abs(np.diff(actual_path, axis=0))) <= 2.5 + 1e-9,
             'actual encoder command step exceeds 2.5 calibrated units')
    fixed['wire'] = dict(source_raw=raw.tolist(), commands_raw=raw_commands.tolist(),
        commands=wire_commands, represented_commands=represented_commands,
        goal_raw=terminal.tolist(), represented_goal=represented.tolist(),
        j3_position_per_tick=j3_tick, maximum_j3_step_ticks=max_ticks,
        other_axes_raw_unchanged=True, verified_float32_roundtrip=True)
    return fixed


def verify_j3_compensation_wire(recipe, *, normalize_raw, encode_positions):
    """Rebuild and check the wire binding against the same owner's conversion."""
    _require(isinstance(recipe, dict) and isinstance(recipe.get('wire'), dict),
             'compensation requires actual encoder-bound wire commands')
    rebuilt = bind_j3_compensation_wire(recipe, recipe['wire'].get('source_raw'),
        normalize_raw=normalize_raw, encode_positions=encode_positions)
    _require(recipe['wire'] == rebuilt['wire'], 'compensation wire recipe differs from actual encoder binding')
    return rebuilt
