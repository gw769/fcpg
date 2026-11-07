"""Explicit perception-only import across the ARM SIGReg training change.

No old ARM/Actor weights, optimizer, RNG or completion flags are imported.
The observer implementation and raw-cache contract have not changed.
"""
from dataclasses import asdict

from .prod_v20_1 import VERSION, CONTRACT, OBSERVATION_CONTRACT, POSITION_UNIT


def validate_observer_checkpoint(payload, config, schedule):
    expected = asdict(config)
    actual = dict(payload.get("config", {}))
    if actual.get("actor_mode") in ("full", "no_innovation"):
        actual["actor_mode"] = expected["actor_mode"]
    expected_schedule = asdict(schedule)
    actual_schedule = dict(payload.get("schedule", {}))
    # These fields only govern ARM, which is NOT imported.
    for key in ("sigreg_weight", "sigreg_recipe"):
        expected_schedule.pop(key, None)
        actual_schedule.pop(key, None)
    if (payload.get("version") != VERSION
            or payload.get("contract") not in (OBSERVATION_CONTRACT, CONTRACT)
            or payload.get("position_unit") != POSITION_UNIT
            or actual != expected or actual_schedule != expected_schedule
            or payload.get("stage") != "refit"
            or payload.get("step") != schedule.real_updates
            or payload.get("metrics", {}).get("complete") is not True
            or not payload.get("run_id")):
        raise ValueError("perception import requires the completed formal V20.1 refit contract")
    return payload


def import_observer_only(core, payload, schedule):
    validate_observer_checkpoint(payload, core.config, schedule)
    state = {key.removeprefix("observer."): value for key, value in payload["model"].items()
             if key.startswith("observer.")}
    core.observer.load_state_dict(state, strict=True)
    return dict(source_run_id=payload["run_id"], source_stage="refit", source_step=payload["step"],
                imported_modules=["observer"], reused_raw_cache_contract=OBSERVATION_CONTRACT,
                arm_actor_optimizer_rng_imported=False)
