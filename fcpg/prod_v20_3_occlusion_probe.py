"""V20.3 occlusion-persistence probe on geometry-available development chains.

The one behavioural change of V20.3 is that losing images does not erase the
body estimate: the last visually corrected mechanical state follows the ACTUAL
joint readback. This module measures that claim against matched estimators on
the same chains, sharing one H0 belief and the same stored measurements:

  proprioceptive_memory  V20.3 production completion path with images dropped
  independent_endpoint   same path, cross-tip covariance removed at fusion
  pixel_offset_carry     V20.1 geometry propagation: last visual pair plus the
                         FK pixel displacement, i.e. a constant pixel offset
  frozen_pixels          previous belief carried in pixel space with time-scaled
                         covariance (V20.1 behaviour without camera geometry)
  nominal_prior          calibrated projection of the current actual q, no vision
  memoryless_fusion      current image fused with the nominal prior, no memory
  full_vision            the same chain with every image (reference)

``pixel_offset_carry`` is the fair competitor to the mechanical-reference
memory. The two differ only where a correction's pixel effect changes with
depth or camera model; with an affine pixel fit they are expected to nearly
coincide, and a null difference there is a result, not a failure.

Evidence boundary: on real fish chains there is no per-endpoint fingertip
label and the camera mapping is an affine pixel fit, not metric extrinsics.
Distances are therefore measured to the full-vision filtered MAP pair and to
the current raw measurement, i.e. to what vision would have said, not to
physical ground truth. Supplied ``target_fm_px``/``point_valid`` labels (e.g.
synthetic chains) additionally produce true EPE/NLL/coverage rows. Per-row
values are returned so confidence intervals can be clustered by recording
episode; adjacent overlapping windows are not independent trials. Nothing here
selects checkpoints, opens held episodes or reaches hardware. Passing this
probe is not anti-fog or real-occlusion success.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import torch
from torch import Tensor

from .prod_v20_1_belief import pair_nll
from .prod_v20_3 import detach_actual
from .prod_v20_3_evaluation import _Sums, _map_pair, _restore_model_mode, observer_metrics
from .prod_v20_3_independent_endpoint import independent_endpoint_fusion
from .prod_v20_policy import append_completed, empty_history

VIEWS = ("wrist", "column")
HORIZONS = 5
ESTIMATORS = ("proprioceptive_memory", "independent_endpoint", "pixel_offset_carry",
              "frozen_pixels", "nominal_prior", "memoryless_fusion")
PER_ROW_METRICS = ("map_pair_distance_to_full_vision_px", "map_pair_distance_to_raw_measurement_px")


def _schedule(rows):
    return torch.tensor(rows, dtype=torch.bool)


# Image availability [horizon, view]; H0 is always observed so every estimator
# starts from the same visually corrected belief. H4 is the recovery endpoint.
DEFAULT_OCCLUSION_SCHEDULES = {
    "both_h1_h3": _schedule([(1, 1), (0, 0), (0, 0), (0, 0), (1, 1)]),
    "wrist_h1_h3": _schedule([(1, 1), (0, 1), (0, 1), (0, 1), (1, 1)]),
    "column_h1_h3": _schedule([(1, 1), (1, 0), (1, 0), (1, 0), (1, 1)]),
}


def _validate_chain(chain: Mapping):
    q, commands = chain["q_position"], chain["commands_position"]
    paths, intervals = chain["measured_q_paths_position"], chain["time_deltas"]
    observations, valid = chain["measurements"], chain["view_valid"]
    batch = len(q)
    if q.shape != (batch, HORIZONS, 6) or paths.shape != (batch, 4, 6, 6) or commands.shape != (batch, 4, 5, 6) or len(observations) != HORIZONS:
        raise ValueError("occlusion probe requires exactly one H0..H4 chain per row")
    if intervals.shape != (batch, 4, 5) or not (intervals > 0).all() or not torch.isfinite(intervals).all():
        raise ValueError("occlusion probe intervals must be finite positive seconds")
    if not torch.allclose(paths[:, :, 0], q[:, :-1], atol=1e-4, rtol=0) or not torch.allclose(paths[:, :, -1], q[:, 1:], atol=1e-4, rtol=0):
        raise ValueError("occlusion probe q/image/readback chain is not aligned")
    valid = torch.as_tensor(valid, device=q.device, dtype=torch.bool)
    if valid.shape != (batch, HORIZONS, 2):
        raise ValueError("view_valid must be [B,5,2]")
    if chain.get("geometry") is None:
        raise ValueError("occlusion probe requires camera geometry")
    initial = chain.get("initial_time_s", q.new_zeros(batch))
    return q, commands, paths, intervals, observations, valid, chain["geometry"], initial


def _validate_schedule(name, available, device):
    schedule = torch.as_tensor(available, device=device)
    if schedule.dtype != torch.bool or schedule.shape != (HORIZONS, 2):
        raise ValueError(f"occlusion schedule {name!r} must be a bool [5,2] image-availability tensor")
    if not schedule[0].all():
        raise ValueError(f"occlusion schedule {name!r} must observe both H0 images; all estimators share one H0 belief")
    return schedule


def schedule_status(available: Tensor) -> dict[str, list[str]]:
    status = {}
    for view, name in enumerate(VIEWS):
        rows = []
        for horizon in range(HORIZONS):
            if not available[horizon, view]:
                rows.append("occluded")
            elif horizon and not available[horizon - 1, view]:
                rows.append("recovered")
            else:
                rows.append("visible")
        status[name] = rows
    return status


def occluded_measurement(measurement, available):
    """Remove every image-derived field of unavailable views.

    Mirrors the runtime's image-less state: no scene tokens, features, field
    or center response, and zero visibility so completion never treats the
    dropped view as an accepted visual capture. Available views are untouched.
    """
    mask = torch.as_tensor(available, device=measurement.belief.log_weights.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask[None].expand(len(measurement.belief.log_weights), -1)
    if mask.shape != measurement.belief.log_weights.shape[:2]:
        raise ValueError("availability must be [2] or [B,2]")

    def blank(value):
        return torch.where(mask.reshape(*mask.shape, *([1] * (value.ndim - 2))), value, torch.zeros_like(value))

    belief = replace(measurement.belief, visible_probability=blank(measurement.belief.visible_probability))
    return replace(measurement, belief=belief, scene_tokens=blank(measurement.scene_tokens),
                   component_features=blank(measurement.component_features),
                   pooling_map=blank(measurement.pooling_map),
                   coarse_center_log_posterior=blank(measurement.coarse_center_log_posterior))


def _shared_initial(core, q, observations, valid, geometry, initial_time):
    first = observations[0]
    if not getattr(core, "INITIAL_CONSUMES_RAW_MEASUREMENT", False):
        first = core.observer.assimilate(first, geometry.make_prior(q[:, 0]), valid[:, 0])
    return core.initial_from_measurement(first, q[:, 0], valid[:, 0], initial_time, geometry=geometry)


def _current_update(core, raw, prior, valid, q, geometry, history=None):
    if getattr(core, "INITIAL_CONSUMES_RAW_MEASUREMENT", False):
        return core.observer.update(raw, prior, valid, q_position=q, actual_history=history,
                                    mechanical_prior=geometry.make_prior(q))
    return core.observer.assimilate(raw, prior, valid)


@torch.no_grad()
def replay_states(core, chain: Mapping, available: Tensor, *, frozen_pixels: bool = False) -> list:
    """Production H0..H4 replay with the given image availability.

    ``frozen_pixels=True`` keeps the shared geometry-corrected H0 belief but
    completes without geometry, so the belief is carried in pixel space with
    time-scaled covariance exactly as V20.1 does when no camera mapping exists.
    """
    q, commands, paths, intervals, observations, valid, geometry, initial = _validate_chain(chain)
    available = _validate_schedule("replay", available, q.device)
    state = _shared_initial(core, q, observations, valid, geometry, initial)
    completion_geometry = None if frozen_pixels else geometry
    states = [state]
    for horizon in range(HORIZONS - 1):
        prediction = core.predict(state, commands[:, horizon], intervals[:, horizon], completion_geometry)
        effective = valid[:, horizon + 1] & available[horizon + 1]
        time = state.endpoint_time_s + intervals[:, horizon].sum(-1)
        state = detach_actual(core.complete_from_measurement(state, prediction,
            occluded_measurement(observations[horizon + 1], available[horizon + 1]),
            paths[:, horizon], effective, time, geometry=completion_geometry))
        states.append(state)
    return states


@torch.no_grad()
def pixel_offset_carry(core, chain: Mapping, available: Tensor) -> list:
    """V20.1 geometry propagation as an estimator: last visual pair + FK pixel delta.

    Uses the geometry adapter's inherited ``propagate`` (constant pixel
    residual, per-transition process variance) and the same V20.3 image
    update when a view is available. It shares the geometry-corrected H0 belief.
    """
    q, commands, paths, intervals, observations, valid, geometry, initial = _validate_chain(chain)
    available = _validate_schedule("pixel_offset_carry", available, q.device)
    belief = _shared_initial(core, q, observations, valid, geometry, initial).observation.belief
    beliefs = [belief]
    history = empty_history(len(q), device=q.device)
    for horizon in range(1, HORIZONS):
        source = occluded_measurement(observations[horizon - 1], available[horizon - 1])
        history = append_completed(history, commands[:, horizon - 1], paths[:, horizon - 1],
                                   source.scene_tokens.mean(-2), intervals[:, horizon - 1])
        prior = geometry.propagate(belief, q[:, horizon - 1], q[:, horizon])
        effective = valid[:, horizon] & available[horizon]
        belief = _current_update(core, occluded_measurement(observations[horizon], available[horizon]),
                                 prior, effective, q[:, horizon], geometry, history).belief
        beliefs.append(belief)
    return beliefs


def _map_component(belief, value):
    index = belief.log_weights.argmax(-1)
    suffix = value.shape[3:]
    gathered = value.gather(2, index.reshape(*index.shape, 1, *([1] * len(suffix))).expand(-1, -1, 1, *suffix))
    return gathered.squeeze(2)


def _belief_metrics(sums: _Sums, per_row: dict, prefix: str, belief, reference_map: Tensor, raw_map: Tensor | None,
                    raw_valid: Tensor, elapsed: Tensor | None):
    batch = len(belief.log_weights)
    map_pair = _map_pair(belief)
    to_full = (map_pair - reference_map).norm(dim=-1).mean(-1)
    to_raw = None if raw_map is None else (map_pair - raw_map).norm(dim=-1).mean(-1)
    nll = pair_nll(belief, reference_map, torch.ones(batch, 2, 2, dtype=torch.bool, device=map_pair.device),
                   ordered=True, reduction="none", identity_aware=True)
    log_volume = .5 * torch.linalg.slogdet(_map_component(belief, belief.state_covariance)).logabsdet
    visible = _map_component(belief, belief.visible_probability).mean(-1)
    for view, name in enumerate(VIEWS):
        sums.add(f"{prefix}/{name}/map_pair_distance_to_full_vision_px", to_full[:, view])
        if to_raw is not None:
            sums.add(f"{prefix}/{name}/map_pair_distance_to_raw_measurement_px", to_raw[:, view], raw_valid[:, view])
        sums.add(f"{prefix}/{name}/full_vision_map_nll", nll[:, view])
        sums.add(f"{prefix}/{name}/map_component_log_volume", log_volume[:, view])
        sums.add(f"{prefix}/{name}/map_visible_mass", visible[:, view])
        if elapsed is not None:
            sums.add(f"{prefix}/{name}/elapsed_since_visual_capture_s", elapsed[:, view])
        # Per-row values let a caller cluster confidence intervals by episode;
        # invalid raw views are recorded as None, never as zero distance.
        per_row[f"{prefix}/{name}/map_pair_distance_to_full_vision_px"] = to_full[:, view].double().cpu().tolist()
        if to_raw is not None:
            per_row[f"{prefix}/{name}/map_pair_distance_to_raw_measurement_px"] = [
                float(value) if ok else None for value, ok in zip(to_raw[:, view].double().cpu().tolist(), raw_valid[:, view].cpu().tolist())]


def _memory_elapsed(states, initial, frozen_pixels):
    elapsed = []
    for state in states:
        if frozen_pixels or state.correction_time_s is None:
            elapsed.append((state.endpoint_time_s - initial)[:, None].expand(-1, 2))
        else:
            elapsed.append(state.endpoint_time_s[:, None] - state.correction_time_s)
    return elapsed


@torch.no_grad()
@_restore_model_mode
def occlusion_probe(core, chain: Mapping, *, schedules: Mapping[str, Tensor] | None = None,
                    coverage_samples: int = 256, seed: int = 20100) -> dict:
    """Matched estimators under scheduled image loss; development evidence only."""
    q, commands, paths, intervals, observations, valid, geometry, initial = _validate_chain(chain)
    schedules = DEFAULT_OCCLUSION_SCHEDULES if schedules is None else schedules
    schedules = {name: _validate_schedule(name, available, q.device) for name, available in schedules.items()}
    if not schedules:
        raise ValueError("at least one occlusion schedule is required")
    batch = len(q)
    labeled_chain = "target_fm_px" in chain and "point_valid" in chain

    def labeled(belief, horizon):
        return observer_metrics(replace(observations[horizon], belief=belief), chain["target_fm_px"][:, horizon],
                                chain["point_valid"][:, horizon], chain.get("ordered", True),
                                seed=seed, coverage_samples=coverage_samples)

    every_image = torch.ones(HORIZONS, 2, dtype=torch.bool, device=q.device)
    full_states = replay_states(core, chain, every_image)
    reference = [_map_pair(state.observation.belief) for state in full_states]
    raw_spatial_available = not getattr(core.observer, "raw_feature_cache", False)
    raw = [_map_pair(observation.belief) if raw_spatial_available else None for observation in observations]
    sums, per_row, labels, unavailable = _Sums(), {}, {}, {}
    if labeled_chain:
        for horizon in range(HORIZONS):
            labels[f"full_vision/h{horizon}"] = labeled(full_states[horizon].observation.belief, horizon)
            if raw_spatial_available:
                labels[f"raw_measurement/h{horizon}"] = labeled(observations[horizon].belief, horizon)
    for name, available in schedules.items():
        beliefs = {}
        elapsed = {}
        memory_states = replay_states(core, chain, available)
        beliefs["proprioceptive_memory"] = [state.observation.belief for state in memory_states]
        elapsed["proprioceptive_memory"] = _memory_elapsed(memory_states, initial, False)
        try:
            with independent_endpoint_fusion(core):
                independent_states = replay_states(core, chain, available)
        except NotImplementedError as error:
            unavailable["independent_endpoint"] = str(error)
        else:
            beliefs["independent_endpoint"] = [state.observation.belief for state in independent_states]
            elapsed["independent_endpoint"] = _memory_elapsed(independent_states, initial, False)
        if hasattr(geometry, "propagate"):
            beliefs["pixel_offset_carry"] = pixel_offset_carry(core, chain, available)
        else:
            unavailable["pixel_offset_carry"] = "geometry adapter has no V20.1 pixel propagation"
        frozen_states = replay_states(core, chain, available, frozen_pixels=True)
        beliefs["frozen_pixels"] = [state.observation.belief for state in frozen_states]
        elapsed["frozen_pixels"] = _memory_elapsed(frozen_states, initial, True)
        beliefs["nominal_prior"] = [geometry.make_prior(q[:, horizon]) for horizon in range(HORIZONS)]
        beliefs["memoryless_fusion"] = [
            _current_update(core, occluded_measurement(observations[horizon], available[horizon]),
                            geometry.make_prior(q[:, horizon]), valid[:, horizon] & available[horizon],
                            q[:, horizon], geometry).belief
            for horizon in range(HORIZONS)]
        for horizon in range(HORIZONS):
            effective = valid[:, horizon] & available[horizon]
            for view, view_name in enumerate(VIEWS):
                sums.add(f"occlusion/{name}/h{horizon}/{view_name}/image_available_fraction", effective[:, view].float())
            for estimator, sequence in beliefs.items():
                belief = sequence[horizon]
                _belief_metrics(sums, per_row, f"occlusion/{name}/{estimator}/h{horizon}", belief, reference[horizon],
                                raw[horizon], valid[:, horizon], elapsed.get(estimator, [None] * HORIZONS)[horizon])
                if labeled_chain:
                    labels[f"{name}/{estimator}/h{horizon}"] = labeled(belief, horizon)
    return dict(
        evidence="training-development", unseen_claim=False, chain_rows=batch,
        geometry_kind=getattr(geometry, "geometry_kind", "unavailable"),
        metric_geometry=bool(getattr(geometry, "metric_3d", False)),
        estimators=[name for name in ESTIMATORS if name not in unavailable],
        estimators_unavailable=unavailable,
        schedules={name: dict(image_available=available.cpu().tolist(), status=schedule_status(available))
                   for name, available in schedules.items()},
        metrics=sums.result(), per_row=per_row, labeled=labels,
        protocol=dict(
            reference=("full-vision filtered MAP pair" + (" and current raw measurement MAP pair" if raw_spatial_available else "") + " at the same endpoint; "
                      "physical ground truth only where target_fm_px labels are supplied"),
            raw_measurement_available=raw_spatial_available,
            raw_measurement_limit="raw feature placeholders have no spatial interpretation" if not raw_spatial_available else None,
            shared_h0="every estimator starts from the same geometry-corrected H0 belief and never sees a dropped image",
            proprioceptive_memory="V20.3 completion: actual-q localization through episode correction memory; vision corrects when present",
            independent_endpoint="same completion with cross-tip covariance removed at fusion; inference-time matched intervention, not a retrained point baseline",
            pixel_offset_carry="V20.1 geometry propagation: last visual pair plus FK pixel displacement (constant pixel offset), same image update when available",
            frozen_pixels="completion without geometry: previous belief carried in pixel space with time-scaled covariance (V20.1 without camera mapping)",
            nominal_prior="calibrated projection of the current actual q; no visual correction",
            memoryless_fusion="current image fused once with the nominal prior; no temporal memory",
            geometry_caveat="fish chains use an affine pixel fit, not metric extrinsics; archive chains have no geometry and are not covered",
            mechanical_vs_pixel_offset="with an affine pixel fit, proprioceptive_memory and pixel_offset_carry are expected to nearly coincide; "
                                       "their difference is only demonstrable under metric camera geometry",
            per_row="per-row distances for episode-clustered confidence intervals; overlapping windows are not independent trials",
            labels_available=labeled_chain, coverage_samples=coverage_samples, coverage_seed=seed,
            anti_fog_or_real_occlusion_claim=False, held_episode_access=False, robot_commands_sent=False))


__all__ = ["DEFAULT_OCCLUSION_SCHEDULES", "ESTIMATORS", "PER_ROW_METRICS", "VIEWS", "occlusion_probe",
           "pixel_offset_carry", "replay_states", "occluded_measurement", "schedule_status"]
