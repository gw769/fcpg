"""Auditable sampling probabilities from an explicitly supplied training subset.

Weak response is a training-label stratum, never a policy input or an inference
decision. No unselected row is read, and no hardware or model is constructed.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import torch


SAMPLING_MODES = ("equal_run", "equal_sample", "weak_response_balanced", "response_ratio_balanced")
WEAK_RESPONSE_JOINT_INDEX = 2
WEAK_RESPONSE_THRESHOLD = .25
LOW_RESPONSE_RATIO = .25


def response_features(row):
    """Measured training strata; never used to change targets or live gates."""
    q0 = _array(row, "source_q_position", (6,))
    commands = _array(row, "commands_position", (5, 6))
    measured = _array(row, "measured_q_path", (6, 6))
    if not np.allclose(q0, measured[0], atol=1e-4, rtol=0):
        raise ValueError("training measured_q_path source differs from q0")
    commanded = commands[:, WEAK_RESPONSE_JOINT_INDEX] - q0[WEAK_RESPONSE_JOINT_INDEX]
    observed = measured[1:, WEAK_RESPONSE_JOINT_INDEX] - q0[WEAK_RESPONSE_JOINT_INDEX]
    peak = float(commanded[np.argmax(np.abs(commanded))])
    command_motion = abs(peak)
    actual_motion = float(np.abs(observed).max())
    # Include q0 so opposite-only movement has zero positive response, not a
    # misleading absolute response ratio. Zero-size commands have no ratio.
    directional_motion = max(0., float((observed * np.sign(peak)).max()))
    ratio = directional_motion / command_motion if command_motion > 0. else None
    driven = command_motion > WEAK_RESPONSE_THRESHOLD
    legacy_weak = driven and actual_motion <= WEAK_RESPONSE_THRESHOLD
    low_ratio = driven and ratio <= LOW_RESPONSE_RATIO
    return dict(max_command_j3_displacement=command_motion,
                max_actual_j3_displacement=actual_motion,
                commanded_peak_j3_delta=peak,
                directional_j3_response=directional_motion,
                directional_response_ratio=ratio,
                legacy_weak_response=bool(legacy_weak),
                low_ratio_response=bool(low_ratio),
                ratio_or_absolute_weak_response=bool(legacy_weak or low_ratio))


def _indices(train_indices, row_count):
    if isinstance(train_indices, torch.Tensor):
        if train_indices.device.type != "cpu":
            raise ValueError("training indices must be on CPU")
        train_indices = train_indices.detach().numpy()
    values = np.asarray(train_indices)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iu":
        raise ValueError("training indices must be a nonempty one-dimensional integer sequence")
    if values.min() < 0 or values.max() >= row_count:
        raise ValueError("training index out of range")
    if np.unique(values).size != values.size:
        raise ValueError("training indices must be unique")
    return values.astype(np.int64, copy=False)


def _array(row, key, shape):
    try:
        value = np.asarray(row[key], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"training row requires numeric {key}") from error
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"training {key} must have shape {shape} and finite values")
    return value


def response_sampling(rows, train_indices, mode):
    """Return CPU float64 probabilities in train_indices order and JSON metadata.

    ``equal_run`` exactly retains the previous trainer's operation order:
    NumPy float64 1/run_count for each selected row, then divide by that ordered
    array's sum. Thus an unchanged torch Generator produces identical draws.

    The balanced mode gives weak and non-weak strata equal probability mass,
    with uniform samples within each. If only one stratum exists, it uses that
    stratum uniformly and explicitly records the missing class. Every count,
    label-derived criterion and probability uses only the selected rows.
    """
    if mode not in SAMPLING_MODES:
        raise ValueError(f"unknown response sampling mode: {mode}")
    indices = _indices(train_indices, len(rows))
    runs, weak, command_motion, actual_motion, features = [], [], [], [], []
    for index in indices:
        row = rows[int(index)]
        try:
            run = Path(row["path"]).parent.name
        except (KeyError, TypeError) as error:
            raise ValueError("training row requires a run path") from error
        if not run:
            raise ValueError("training row requires a named parent run directory")
        values = response_features(row)
        features.append(values)
        command_delta = values['max_command_j3_displacement']
        actual_delta = values['max_actual_j3_displacement']
        runs.append(run)
        command_motion.append(command_delta)
        actual_motion.append(actual_delta)
        weak.append(values['ratio_or_absolute_weak_response'] if mode == 'response_ratio_balanced'
                    else values['legacy_weak_response'])

    runs = np.asarray(runs)
    weak = np.asarray(weak, dtype=bool)
    counts = Counter(runs)
    weak_count = int(weak.sum())
    if mode == "equal_run":
        weights = np.array([1. / counts[run] for run in runs], dtype=np.float64)
    elif mode == "equal_sample":
        weights = np.ones(len(indices), dtype=np.float64)
    else:
        weights = np.empty(len(indices), dtype=np.float64)
        if weak_count:
            weights[weak] = 1. / weak_count
        if weak_count < len(indices):
            weights[~weak] = 1. / (len(indices) - weak_count)
    probabilities = weights / weights.sum()
    strata = {}
    for name, mask in (("weak_response", weak), ("non_weak_response", ~weak)):
        strata[name] = dict(count=int(mask.sum()), probability_mass=float(probabilities[mask].sum()))
    missing = [name for name, values in strata.items() if values["count"] == 0]
    run_statistics = {}
    for run in sorted(counts):
        mask = runs == run
        run_statistics[str(run)] = dict(
            count=int(mask.sum()), probability_mass=float(probabilities[mask].sum()),
            weak_response_count=int((mask & weak).sum()),
            non_weak_response_count=int((mask & ~weak).sum()),
            weak_response_probability_mass=float(probabilities[mask & weak].sum()),
            non_weak_response_probability_mass=float(probabilities[mask & ~weak].sum()),
        )
    metadata = dict(
        schema="v21-response-sampling-v1", mode=mode, scope="train_indices_only",
        training_samples=len(indices), train_indices=indices.tolist(),
        probability_sum=float(probabilities.sum()), probabilities=probabilities.tolist(),
        weak_response_definition=dict(
            joint="J3", joint_index_zero_based=WEAK_RESPONSE_JOINT_INDEX,
            position_unit="lerobot_calibrated_position", threshold=WEAK_RESPONSE_THRESHOLD,
            command="max_t(abs(command_t - q0)) > threshold",
            actual="max_t(abs(measured_q_path[1:] - q0)) <= threshold",
            whole_trajectory=True, q0_range_filter=False,
        ),
        strata=strata, runs=run_statistics, missing_strata=missing,
        balance_fallback=("uniform_over_only_observed_stratum"
                          if mode in ("weak_response_balanced", "response_ratio_balanced") and missing else None),
        samples=[dict(train_index=int(index), run=str(runs[i]), weak_response=bool(weak[i]),
                      max_command_j3_displacement=command_motion[i],
                      max_actual_j3_displacement=actual_motion[i],
                      probability=float(probabilities[i]))
                 for i, index in enumerate(indices)],
    )
    if mode == 'response_ratio_balanced':
        metadata['schema'] = 'v21-response-sampling-v2'
        metadata['weak_response_definition'].update(
            actual='legacy absolute weak response OR directional_peak/command_peak <= ratio_threshold',
            ratio_threshold=LOW_RESPONSE_RATIO,
            directional_peak='max(0, max_t((measured_q_path[1:] - q0) * sign(commanded_peak)))',
            purpose='training sampling only; physical response acceptance and targets unchanged')
        for sample, values in zip(metadata['samples'], features):
            sample.update(values)
    return torch.tensor(probabilities, dtype=torch.float64, device="cpu"), metadata
