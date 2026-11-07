"""Explicit RAW observer reuse, never old feedback/Actor completion reuse."""
from dataclasses import asdict

from .prod_v20_1 import V201Config, VERSION as PARENT_VERSION
from .prod_v20_1_transfer import validate_observer_checkpoint as validate_parent
from .prod_v20_3 import VERSION, CONTRACT, OBSERVATION_CONTRACT, POSITION_UNIT


def validate_observer_checkpoint(payload, config, schedule):
    if payload.get("version") == PARENT_VERSION:
        parent = dict(asdict(config))
        parent["state_representation"] = V201Config().state_representation
        return validate_parent(payload, V201Config(**parent), schedule)
    expected = asdict(config)
    actual = dict(payload.get("config", {}))
    if actual.get("actor_mode") in ("full", "no_innovation"):
        actual["actor_mode"] = expected["actor_mode"]
    old_schedule, new_schedule = dict(payload.get("schedule", {})), asdict(schedule)
    for key in ("sigreg_weight", "sigreg_recipe"):
        old_schedule.pop(key, None)
        new_schedule.pop(key, None)
    if (payload.get("version") != VERSION or payload.get("contract") != CONTRACT
            or payload.get("position_unit") != POSITION_UNIT or actual != expected
            or old_schedule != new_schedule or payload.get("stage") != "refit"
            or payload.get("step") != schedule.real_updates
            or payload.get("metrics", {}).get("complete") is not True or not payload.get("run_id")):
        raise ValueError("observer reuse requires completed formal matching raw measurement")
    return payload


def import_observer_only(core, payload, schedule):
    validate_observer_checkpoint(payload, core.config, schedule)
    core.observer.load_state_dict({key.removeprefix("observer."): value
        for key, value in payload["model"].items() if key.startswith("observer.")}, strict=True)
    return dict(source_version=payload["version"], source_run_id=payload["run_id"],
        imported_modules=["observer"], reused_raw_cache_contract=OBSERVATION_CONTRACT,
        arm_actor_optimizer_rng_imported=False, feedback_changed_requires_downstream_training=True)
