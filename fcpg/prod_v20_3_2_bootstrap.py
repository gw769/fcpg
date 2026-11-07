"""Reuse a completed V20.3.2 CAD observer, never a refit or policy checkpoint."""
from dataclasses import asdict
import math
import random

import numpy as np
import torch


def validate_pre_refit_checkpoint(payload, core, schedule, stage="fcpg"):
    if stage not in ("appearance", "fcpg"):
        raise ValueError("only appearance/CAD reuse is supported")
    expected = dict(version=core.CHECKPOINT_VERSION, contract=core.CHECKPOINT_CONTRACT,
                    position_unit=core.CHECKPOINT_POSITION_UNIT, config=asdict(core.config),
                    schedule=asdict(schedule), stage=stage)
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"CAD source checkpoint {name} differs")
    steps = (math.ceil(schedule.appearance_endpoints / schedule.effective_batch) if stage == "appearance"
             else math.ceil(schedule.cad_rows / schedule.effective_batch) * schedule.cad_passes)
    if payload.get("step") != steps or payload.get("metrics", {}).get("complete") is not True:
        raise ValueError("CAD source checkpoint is incomplete or a smoke result")
    return payload


def import_cad_observer(core, payload, schedule, *, restore_rng=True):
    validate_pre_refit_checkpoint(payload, core, schedule)
    prefix = "observer."
    observer = {name[len(prefix):]: value for name, value in payload["model"].items() if name.startswith(prefix)}
    core.observer.load_state_dict(observer, strict=True)
    # No optimizer and no ARM, Actor or control-encoding weights are imported.
    if restore_rng:
        state = payload["rng"]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state["cuda"]:
            torch.cuda.set_rng_state_all(state["cuda"])
    return dict(imported="observer-only-after-CAD", source_stage="fcpg", source_step=payload["step"],
                optimizer_imported=False, arm_actor_imported=False, rng_restored=restore_rng)
