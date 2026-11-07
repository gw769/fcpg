"""Incremental training contracts for Prod-V17.8.1.

The expensive V17.8 Measurement, belief cache, LeWM and 5.12M-example Actor
pretraining are not repeated.  This module strictly migrates one completed
V17.8 Actor checkpoint, then fine-tunes the Actor and the new soft-object
branch with equal archive replay and event-balanced fish examples.

Contact labels and forward kinematics exist only in this optimizer.  They are
sampling/supervision metadata and never enter the deployed Actor signature.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from fcpg import prod_v17_7_training as _v177
from fcpg import prod_v17_8_training as _v178
from fcpg.prod_v17_8 import V178RawHistory
from fcpg.prod_v17_8_1 import (
    ARCHITECTURE_CONTRACT,
    VERSION,
    V1781Core,
)
from fcpg.prod_v17_8_1_contact import (
    CONTACT_EVENTS,
    ContactEvent,
    ContactEventTable,
    EventFKCallback,
    DifferentiableToolAxisCallback,
    build_contact_event_table,
    contact_event_metrics,
    event_balanced_epoch_indices,
    training_tool_axis_cosine_loss,
)
from fcpg.prod_v18_1_data import TRANSITIONS, V181FishArchive


SEED = 17_810
BASE_VERSION = _v178.VERSION
BASE_ARCHITECTURE_CONTRACT = _v178.ARCHITECTURE_CONTRACT
DEFAULT_INCREMENTAL_UPDATES = 1_000
DEFAULT_ARCHIVE_BATCH = 64
DEFAULT_FISH_BATCH = 64
DEFAULT_MICROBATCH = 32
DEFAULT_TOOL_AXIS_WEIGHT = 0.10
CHECKPOINT_EVERY = 250
GRADIENT_CLIP = 1.0
REQUIRED_BASE_ARCHIVE_EXPOSURES = 5_120_000
REQUIRED_FISH_REPLAY_UPDATES = _v178.ACTOR_FISH_REPLAY_UPDATES
REQUIRED_REPLAY_PAIR_EXPOSURES = 256_000


class V1781TrainingError(RuntimeError):
    """Raised before an invalid incremental run can change model weights."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _new_parameter_prefix(key: str) -> bool:
    return key.startswith("object_head.") or key.startswith(
        (
            "actor.object_",
            "actor.correction_",
        )
    )


def _base_config_matches(parent: Mapping[str, Any], core: V1781Core) -> bool:
    expected = asdict(core.config)
    for key in (
        "object_query_dim",
        "object_attention_temperature",
        "correction_layers",
    ):
        expected.pop(key, None)
    return dict(parent) == expected


def migrate_v178_actor_checkpoint(
    path: str | Path,
    core: V1781Core,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Strictly load all V17.8 keys and leave only V17.8.1 keys initialized.

    The parent checkpoint contains the complete core state even though its
    component is named ``actor``.  A partial parent or an unexpected key is
    rejected.  The zero-initialized correction head makes the migrated model
    command-identical to its parent until incremental optimization begins.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise V1781TrainingError(f"V17.8 parent checkpoint is absent: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise V1781TrainingError("V17.8 parent checkpoint must be a mapping")
    if (
        payload.get("version") != BASE_VERSION
        or payload.get("architecture") != BASE_ARCHITECTURE_CONTRACT
        or payload.get("component") != "actor"
        or payload.get("seed") != _v178.SEED
        or not isinstance(payload.get("config"), Mapping)
        or not isinstance(payload.get("model"), Mapping)
        or not isinstance(payload.get("metrics"), Mapping)
    ):
        raise V1781TrainingError("V17.8 parent checkpoint contract differs")
    if not _base_config_matches(payload["config"], core):
        raise V1781TrainingError("V17.8 parent model config differs")
    metrics = payload["metrics"]
    if (
        metrics.get("run_kind") != "full"
        or metrics.get("archive_pair_exposures")
        != REQUIRED_BASE_ARCHIVE_EXPOSURES
        or metrics.get("expected_archive_pair_exposures")
        != REQUIRED_BASE_ARCHIVE_EXPOSURES
        or metrics.get("archive_contract_complete") is not True
        or metrics.get("fish_archive_replay_updates")
        != REQUIRED_FISH_REPLAY_UPDATES
        or int(metrics.get("archive_replay_exposures", 0))
        < REQUIRED_REPLAY_PAIR_EXPOSURES
        or int(metrics.get("fish_pair_exposures", 0))
        < REQUIRED_REPLAY_PAIR_EXPOSURES
    ):
        raise V1781TrainingError(
            "V17.8 parent must be a completed full Actor training run"
        )
    child_state = core.state_dict()
    parent_state = payload["model"]
    expected_parent_keys = {
        key for key in child_state if not _new_parameter_prefix(key)
    }
    if set(parent_state) != expected_parent_keys:
        raise V1781TrainingError(
            "V17.8 parent migration did not cover exactly the old model keys"
        )
    for key, value in parent_state.items():
        reference = child_state[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != reference.shape
            or value.dtype != reference.dtype
        ):
            raise V1781TrainingError(
                f"V17.8 parent tensor contract differs for {key}"
            )
    incompatible = core.load_state_dict(payload["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    if unexpected or not missing or not all(_new_parameter_prefix(key) for key in missing):
        raise V1781TrainingError(
            "V17.8 parent migration did not cover exactly the new V17.8.1 keys"
        )
    if bool(torch.count_nonzero(core.actor.correction_head.weight)) or bool(
        torch.count_nonzero(core.actor.correction_head.bias)
    ):
        raise V1781TrainingError("new correction head is not zero initialized")
    core.to(device)
    return {
        "parent": str(source),
        "parent_step": int(payload.get("step", 0)),
        "parent_run_kind": str(metrics["run_kind"]),
        "parent_archive_pair_exposures": int(
            metrics["archive_pair_exposures"]
        ),
        "parent_fish_replay_updates": int(
            metrics["fish_archive_replay_updates"]
        ),
        "parent_archive_replay_exposures": int(
            metrics["archive_replay_exposures"]
        ),
        "parent_fish_pair_exposures": int(metrics["fish_pair_exposures"]),
        "loaded_parent_keys": int(len(payload["model"])),
        "new_initialized_keys": missing,
    }


def save_v1781_checkpoint(
    path: str | Path,
    core: V1781Core,
    *,
    step: int,
    metrics: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": VERSION,
        "architecture": ARCHITECTURE_CONTRACT,
        "component": "actor",
        "seed": SEED,
        "step": int(step),
        "config": asdict(core.config),
        "model": core.state_dict(),
        "metrics": dict(metrics),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_v1781_checkpoint(
    path: str | Path,
    core: V1781Core,
    *,
    device: torch.device,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise V1781TrainingError(f"V17.8.1 checkpoint is absent: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != VERSION
        or payload.get("architecture") != ARCHITECTURE_CONTRACT
        or payload.get("component") != "actor"
        or payload.get("seed") != SEED
        or payload.get("config") != asdict(core.config)
        or not isinstance(payload.get("model"), Mapping)
    ):
        raise V1781TrainingError("V17.8.1 checkpoint contract differs")
    core.load_state_dict(payload["model"], strict=True)
    core.to(device)
    return payload


@dataclass(frozen=True)
class V1781FishEventIndex:
    """Canonical fish pair ordinals and their de-duplicated event table."""

    table: ContactEventTable
    ordinals: np.ndarray

    def __post_init__(self) -> None:
        value = np.asarray(self.ordinals)
        if value.shape != (len(self.table),) or value.dtype.kind not in "iu":
            raise V1781TrainingError("fish event ordinals must match the event table")
        if bool((value < 0).any()):
            raise V1781TrainingError("fish event ordinals must be non-negative")


def build_fish_event_index(
    fish: V181FishArchive,
    event_fk: EventFKCallback,
    *,
    batch_size: int = 512,
) -> V1781FishEventIndex:
    """Audit, de-duplicate and classify all optimizer-visible fish chunks."""

    if int(batch_size) <= 0:
        raise V1781TrainingError("fish event-index batch_size must be positive")
    factual = _v178.as_v178_fish_archive(fish)
    ordinals = np.asarray(factual.pair_ordinals("train"), dtype=np.int64)
    episode_rows: list[np.ndarray] = []
    frame_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    command_rows: list[np.ndarray] = []
    for selected in _v178._batches(ordinals, int(batch_size)):
        pair = factual.pair_batch(selected, partition="train")
        episode_rows.append(np.asarray(pair.episode, dtype=np.int64))
        frame_rows.append(np.asarray(pair.source_frame, dtype=np.int64))
        state_rows.append(np.asarray(pair.current_q, dtype=np.float32) * 100.0)
        command_rows.append(
            np.asarray(pair.target_commands, dtype=np.float32) * 100.0
        )
    table = build_contact_event_table(
        np.concatenate(episode_rows),
        np.concatenate(frame_rows),
        np.concatenate(state_rows),
        np.concatenate(command_rows),
        event_fk,
    )
    canonical = ordinals[np.asarray(table.original_indices, dtype=np.int64)]
    return V1781FishEventIndex(table=table, ordinals=canonical)


def _event_names(table: ContactEventTable, rows: np.ndarray) -> list[ContactEvent]:
    result: list[ContactEvent] = []
    for index in np.asarray(rows, dtype=np.int64).tolist():
        value = table.events[int(index)]
        if value is None:
            raise V1781TrainingError("balanced fish selection contains an unclassified row")
        result.append(value)
    return result


def _checkpoint_metrics(
    *,
    step: int,
    loss: float,
    archive_exposures: int,
    fish_exposures: int,
    event_exposures: Mapping[str, int],
    event_index: V1781FishEventIndex,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "steps": int(step),
        "loss": float(loss),
        "archive_replay_exposures": int(archive_exposures),
        "fish_event_balanced_exposures": int(fish_exposures),
        "fish_event_exposures": dict(event_exposures),
        "fish_unique_event_audit": contact_event_metrics(event_index.table),
        "archive_fish_loss_weight": [0.5, 0.5],
        "contact_event_and_fk_are_training_only": True,
        "settled_clock_is_runtime_only_until_real_settle_training_exists": True,
        "parent": dict(parent),
    }


def train_v1781_incremental_actor(
    core: V1781Core,
    archive: Any,
    fish: V181FishArchive,
    archive_beliefs: Any,
    archive_features: Any,
    fish_source: Any,
    identity: Any,
    task_cache: Any,
    output_root: str | Path,
    *,
    device: torch.device,
    vocabulary: Any,
    event_index: V1781FishEventIndex,
    parent_report: Mapping[str, Any],
    tool_axis_callback: DifferentiableToolAxisCallback,
    updates: int = DEFAULT_INCREMENTAL_UPDATES,
    archive_batch: int = DEFAULT_ARCHIVE_BATCH,
    fish_batch: int = DEFAULT_FISH_BATCH,
    microbatch: int = DEFAULT_MICROBATCH,
    learning_rate: float = _v178.ACTOR_FISH_REPLAY_LR,
    tool_axis_weight: float = DEFAULT_TOOL_AXIS_WEIGHT,
    smoke: bool = False,
) -> dict[str, Any]:
    """Fine-tune with 50% generic replay and 50% balanced contact events."""

    if smoke:
        updates = 2
        archive_batch = fish_batch = microbatch = min(2, int(microbatch))
    integers = (updates, archive_batch, fish_batch, microbatch)
    if any(isinstance(value, bool) or int(value) <= 0 for value in integers):
        raise V1781TrainingError("incremental sizes must be positive integers")
    if archive_batch % microbatch or fish_batch % microbatch:
        raise V1781TrainingError("incremental batches must divide into microbatches")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise V1781TrainingError("learning_rate must be finite and positive")
    if not math.isfinite(float(tool_axis_weight)) or float(tool_axis_weight) < 0.0:
        raise V1781TrainingError("tool_axis_weight must be finite and non-negative")

    output = Path(output_root).expanduser().resolve()
    stage = output / "actor"
    latest = stage / "latest.pt"
    final = stage / "final.pt"
    if final.is_file():
        payload = load_v1781_checkpoint(final, core, device=device)
        return dict(payload.get("metrics", {}))

    core.to(device)
    core.measurement.eval()
    core.grounding_encoder.eval()
    core.lewm.eval()
    trainable = _v178._set_trainable(
        core,
        list(core.actor.parameters())
        + list(core.innovation_encoder.parameters())
        + list(core.object_head.parameters()),
    )
    optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate))
    resume = 0
    resumed_metrics: dict[str, Any] | None = None
    if latest.is_file():
        payload = load_v1781_checkpoint(latest, core, device=device)
        resume = int(payload.get("step", 0))
        resumed_metrics = dict(payload.get("metrics", {}))
        if isinstance(payload.get("optimizer"), Mapping):
            optimizer.load_state_dict(payload["optimizer"])
    if resume < 0 or resume > int(updates):
        raise V1781TrainingError("incremental checkpoint step is outside this run")
    if resume > 0 and (
        resumed_metrics is None
        or resumed_metrics.get("parent") != dict(parent_report)
        or resumed_metrics.get("archive_replay_exposures")
        != resume * int(archive_batch)
        or resumed_metrics.get("fish_event_balanced_exposures")
        != resume * int(fish_batch)
    ):
        raise V1781TrainingError(
            "incremental checkpoint does not match this parent/batch contract"
        )
    if resume == int(updates):
        if resumed_metrics is None:
            raise V1781TrainingError("completed incremental checkpoint lacks metrics")
        save_v1781_checkpoint(
            final,
            core,
            step=resume,
            metrics=resumed_metrics,
        )
        _write_json(output / "train-metrics.json", resumed_metrics)
        return resumed_metrics

    loss_scale = torch.as_tensor(
        _v178.compute_joint_delta_loss_scale(archive), device=device
    )
    row_source = vocabulary.row_index(archive)
    factual_fish = _v178.as_v178_fish_archive(fish)
    total_fish = int(updates) * int(fish_batch)
    samples_per_event = math.ceil(total_fish / len(CONTACT_EVENTS))
    schedule = event_balanced_epoch_indices(
        event_index.table,
        samples_per_event=samples_per_event,
        seed=SEED,
    )[:total_fish]
    if len(schedule) != total_fish:
        raise V1781TrainingError("balanced fish schedule length differs")

    started = time.monotonic()
    last_loss = float("nan")
    archive_exposures = resume * int(archive_batch)
    fish_exposures = resume * int(fish_batch)
    event_exposures = {event.value: 0 for event in CONTACT_EVENTS}
    for index in schedule[: resume * int(fish_batch)]:
        event = event_index.table.events[int(index)]
        if event is not None:
            event_exposures[event.value] += 1

    axis_module = tool_axis_callback
    if isinstance(axis_module, torch.nn.Module):
        axis_module.to(device)
        axis_module.requires_grad_(False)
        axis_module.eval()

    for step in range(resume, int(updates)):
        horizon = step % TRANSITIONS
        # A stateless per-step seed makes an interrupted run sample-identical
        # after restoring latest.pt; no hidden NumPy RNG cursor is required.
        generator = np.random.default_rng(SEED + 1 + step)
        archive_rows = generator.integers(
            0, int(archive.rows), size=int(archive_batch), dtype=np.int64
        )
        selected_event_rows = schedule[
            step * int(fish_batch) : (step + 1) * int(fish_batch)
        ]
        selected_ordinals = event_index.ordinals[selected_event_rows]
        selected_events = _event_names(event_index.table, selected_event_rows)

        optimizer.zero_grad(set_to_none=True)
        weighted_loss = torch.zeros((), device=device)
        for micro in _v178._batches(archive_rows, int(microbatch)):
            with _v178._autocast(device):
                scene, update, q, history = _v177._archive_cached_actor_context(
                    core,
                    archive,
                    archive_beliefs,
                    archive_features,
                    micro,
                    horizon,
                    device=device,
                    row_source=row_source,
                    collect_history=True,
                )
                prediction = core.act(
                    scene,
                    update.grounded,
                    update.innovation.innovation,
                    q,
                    history=history,
                    task_embedding=task_cache.for_rows(micro, device=device),
                )
                target = _v178._tensor(
                    _v178._archive_target(archive, micro, horizon), device
                )
                loss = _v178.actor_action_loss(
                    prediction.desired_commands, target, q, loss_scale
                ).total
                weight = 0.5 * len(micro) / float(archive_batch)
                term = loss * weight
            term.backward()
            weighted_loss = weighted_loss + term.detach()

        ordinal_positions = np.arange(len(selected_ordinals), dtype=np.int64)
        for fish_horizon in range(TRANSITIONS):
            active = ordinal_positions[selected_ordinals % TRANSITIONS == fish_horizon]
            for micro_positions in _v178._batches(active, int(microbatch)):
                micro_ordinals = selected_ordinals[micro_positions]
                micro_events = [selected_events[int(value)] for value in micro_positions]
                with _v178._autocast(device):
                    pair, scene, update, q, history = _v177._fish_actor_context(
                        core,
                        identity,
                        factual_fish,
                        fish_source,
                        micro_ordinals,
                        device=device,
                        vocabulary=vocabulary,
                        collect_history=True,
                    )
                    prediction = core.act(
                        scene,
                        update.grounded,
                        update.innovation.innovation,
                        q,
                        history=history,
                        task_embedding=task_cache.fish(
                            len(micro_ordinals), device=device
                        ),
                    )
                    target = _v178._tensor(
                        _v177._fish_commands_degrees(pair), device
                    )
                    action = _v178.actor_action_loss(
                        prediction.desired_commands, target, q, loss_scale
                    ).total
                    pose_rows = torch.as_tensor(
                        [
                            event in {
                                ContactEvent.VERTICAL_APPROACH,
                                ContactEvent.CLOSE,
                            }
                            for event in micro_events
                        ],
                        device=device,
                        dtype=torch.bool,
                    )
                    if bool(pose_rows.any()) and float(tool_axis_weight) > 0.0:
                        pose_valid = pose_rows[:, None].expand(-1, target.shape[1])
                        pose = training_tool_axis_cosine_loss(
                            prediction.desired_commands,
                            target,
                            tool_axis_callback,
                            valid=pose_valid,
                        ).loss
                        action = action + float(tool_axis_weight) * pose
                    weight = 0.5 * len(micro_ordinals) / float(fish_batch)
                    term = action * weight
                term.backward()
                weighted_loss = weighted_loss + term.detach()

        torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
        optimizer.step()
        last_loss = float(weighted_loss.cpu())
        archive_exposures += int(archive_batch)
        fish_exposures += int(fish_batch)
        for event in selected_events:
            event_exposures[event.value] += 1
        completed = step + 1
        if completed == 1 or completed % 25 == 0 or completed == int(updates):
            elapsed = max(time.monotonic() - started, 1.0e-6)
            print(
                json.dumps(
                    {
                        "stage": "v1781-event-balanced-incremental",
                        "step": completed,
                        "total": int(updates),
                        "loss": last_loss,
                        "elapsed_s": round(elapsed, 1),
                        "eta_s": round(
                            (int(updates) - completed) * elapsed / completed, 1
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        metrics = _checkpoint_metrics(
            step=completed,
            loss=last_loss,
            archive_exposures=archive_exposures,
            fish_exposures=fish_exposures,
            event_exposures=event_exposures,
            event_index=event_index,
            parent=parent_report,
        )
        if completed % CHECKPOINT_EVERY == 0 or completed == int(updates):
            save_v1781_checkpoint(
                latest,
                core,
                step=completed,
                metrics=metrics,
                optimizer=optimizer,
            )

    save_v1781_checkpoint(
        final,
        core,
        step=int(updates),
        metrics=metrics,
    )
    _write_json(output / "train-metrics.json", metrics)
    return metrics


__all__ = [
    "ARCHITECTURE_CONTRACT",
    "BASE_ARCHITECTURE_CONTRACT",
    "BASE_VERSION",
    "CHECKPOINT_EVERY",
    "DEFAULT_ARCHIVE_BATCH",
    "DEFAULT_FISH_BATCH",
    "DEFAULT_INCREMENTAL_UPDATES",
    "DEFAULT_MICROBATCH",
    "DEFAULT_TOOL_AXIS_WEIGHT",
    "REQUIRED_BASE_ARCHIVE_EXPOSURES",
    "REQUIRED_FISH_REPLAY_UPDATES",
    "REQUIRED_REPLAY_PAIR_EXPOSURES",
    "SEED",
    "VERSION",
    "V1781FishEventIndex",
    "V1781TrainingError",
    "build_fish_event_index",
    "load_v1781_checkpoint",
    "migrate_v178_actor_checkpoint",
    "save_v1781_checkpoint",
    "train_v1781_incremental_actor",
]
