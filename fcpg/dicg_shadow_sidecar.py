"""Process-isolated DICG production-shadow client.

Parent process API is ``submit`` / ``status`` / ``close`` only.  Keep-newest
lives in a parent-side pending slot; a tiny flush thread retries
``put_nowait`` onto the spawn ``Queue(maxsize=1)``.  The parent never
``get()``s that queue (a 150KiB column in the feeder pipe would race).
This flush thread is not a second infer worker and does not pair frames.

This module's top level must stay free of ``torch``, ``mujoco``,
``fcpg.dicg_shadow``, and ``fcpg.dicg_shadow_geometry``.  Checkpoint / ViT /
geometry loading happens only inside the child via ``DICGShadowEngine``.
The child must not construct ``DICGShadowRunner`` (that would add a second
thread queue).
"""

from __future__ import annotations

import atexit
import inspect
import json
import multiprocessing
import os
import queue
import signal
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np


SCHEMA = "fcpg.dicg_production_shadow.v1"
METRIC_SPACE = "uncalibrated_hypothesis_logit"
CONTROL_INFLUENCE = "none"
PLANNER_DATAFLOW_INFLUENCE = "none"
COLUMN_VIEW = "column"
IMAGE_HW = 224
JOINT_DIM = 6
QUEUE_MAXSIZE = 1
DEFAULT_TOP_K = 5
DEFAULT_LATENCY_BUDGET_S = 0.10
_WORKER_POLL_S = 0.05
_CLOSE_JOIN_S = 5.0
_TERMINATE_JOIN_S = 1.0
START_METHOD = "spawn"

IPC_PAYLOAD_KEYS = frozenset({"timestamp_s", "state", "column"})
SKIP_INSUFFICIENT_ARM_DQ = "insufficient_arm_dq"
SKIP_INSUFFICIENT_EXCITATION = "insufficient_excitation"

PLANNER_CONSUMABLE_STATUS_KEYS = frozenset(
    {
        "top1_index",
        "top1_logit",
        "topk_indices",
        "topk_logits",
        "hypothesis_logits",
        "selected_chunk",
        "selected_index",
        "command",
        "certificate",
        "planner_score",
        "consistency_score",
        "q_corrected_px",
        "absolute_tip_px",
    }
)

FORBIDDEN_PUBLIC_METHODS = frozenset(
    {
        "score",
        "rank",
        "predict",
        "rollout",
        "top1",
        "logits",
        "plan",
        "command",
        "select",
    }
)

PHASE2_WIRING = (
    "DICGShadowEngine is the threadless load+infer+jsonl core in "
    "fcpg/dicg_shadow.py; DICGShadowRunner remains the in-process wrapper.",
    "Default _child_main calls engine.safe_process_pair on the same "
    "single inbound Queue(maxsize=1).  Never construct DICGShadowRunner in the child.",
    "scripts/serve_so101_realtime_fcpg_lewm.py _attach spawns this sidecar "
    "and must not call DICGShadowGeometry.from_files in the 8765 process.",
    "JSONL may carry uncalibrated logits; status() must still strip "
    "PLANNER_CONSUMABLE_STATUS_KEYS and must not RPC the child.",
    "Honesty stays process_isolated=true; in-process runner honesty stays false.",
)

__all__ = [
    "CONTROL_INFLUENCE",
    "DICGShadowSidecar",
    "DICGShadowSidecarError",
    "IPC_PAYLOAD_KEYS",
    "METRIC_SPACE",
    "PHASE2_WIRING",
    "PLANNER_CONSUMABLE_STATUS_KEYS",
    "PLANNER_DATAFLOW_INFLUENCE",
    "QUEUE_MAXSIZE",
    "SCHEMA",
    "START_METHOD",
    "honesty_fields",
    "offer_nonblocking",
]


class DICGShadowSidecarError(RuntimeError):
    """Raised only for invalid parent-side constructor arguments."""


def honesty_fields() -> dict[str, Any]:
    return {
        "shadow_only": True,
        "control_influence": CONTROL_INFLUENCE,
        "planner_dataflow_influence": PLANNER_DATAFLOW_INFLUENCE,
        "process_isolated": True,
        "resource_contention_possible": True,
        "timing_influence_possible": True,
        "heldout_gate_passed": False,
        "live_ready": False,
        "interpretation_licensed": False,
        "physical_tip_accuracy_claim": False,
        "uncertainty_calibrated": False,
        "metric_space": METRIC_SPACE,
    }


def _stamp_honesty(record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(record) if record is not None else {}
    payload.update(honesty_fields())
    return payload


def offer_nonblocking(mailbox: Any, item: Any) -> str:
    """Put ``item`` on a maxsize=1 queue.  Drop the queued predecessor if full.

    Returns ``accepted``, ``replaced``, or ``dropped``.  Never blocks.
    """

    try:
        mailbox.put_nowait(item)
        return "accepted"
    except queue.Full:
        pass
    except Exception:
        return "dropped"
    try:
        mailbox.get_nowait()
    except queue.Empty:
        pass
    except Exception:
        return "dropped"
    try:
        mailbox.put_nowait(item)
        return "replaced"
    except Exception:
        return "dropped"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _as_state(value: Any) -> np.ndarray:
    array = np.array(value, dtype=np.float32, copy=True)
    if array.shape != (JOINT_DIM,):
        raise ValueError(f"q readback must be shape {(JOINT_DIM,)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("state contains non-finite values")
    return array


def _copy_ipc_payload(observation: Any) -> dict[str, Any]:
    if isinstance(observation, Mapping):
        timestamp = observation.get("timestamp_s")
        state = observation.get("state")
        frames = observation.get("frames")
        column = observation.get("column")
        if column is None and isinstance(frames, Mapping):
            column = frames.get(COLUMN_VIEW)
    else:
        timestamp = getattr(observation, "timestamp_s", None)
        state = getattr(observation, "state", None)
        frames = getattr(observation, "frames", None)
        column = getattr(observation, "column", None)
        if column is None and isinstance(frames, Mapping):
            column = frames.get(COLUMN_VIEW)
    if timestamp is None or not np.isfinite(float(timestamp)):
        raise ValueError("observation timestamp_s must be finite")
    joints = _as_state(state)
    image = np.array(column, copy=True)
    if image.shape != (IMAGE_HW, IMAGE_HW, 3) or image.dtype != np.uint8:
        raise ValueError("column frame must be uint8 [224,224,3]")
    return {
        "timestamp_s": float(timestamp),
        "state": joints,
        "column": image,
    }


def _strip_planner_keys(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in PLANNER_CONSUMABLE_STATUS_KEYS
    }


def _path_text(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser())


def _child_config(
    *,
    checkpoint_path: str | Path,
    vit_checkpoint_path: str | Path,
    log_path: str | Path,
    dataset_contract: str | Path | None,
    observer_bundle: str | Path | None,
    mujoco_model: str | Path | None,
    extrinsic_index: int | str | None,
    extrinsic_group_key: str | None,
    device: str,
    top_k: int,
    latency_budget_s: float,
) -> dict[str, Any]:
    return {
        "checkpoint_path": _path_text(checkpoint_path),
        "vit_checkpoint_path": _path_text(vit_checkpoint_path),
        "log_path": _path_text(log_path),
        "dataset_contract": _path_text(dataset_contract),
        "observer_bundle": _path_text(observer_bundle),
        "mujoco_model": _path_text(mujoco_model),
        "extrinsic_index": extrinsic_index,
        "extrinsic_group_key": extrinsic_group_key,
        "device": str(device),
        "top_k": int(top_k),
        "latency_budget_s": float(latency_budget_s),
        "parent_pid": int(os.getpid()),
        "phase": 2,
        "probe_path": None,
    }


def _counter_status(snapshot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    snap = dict(snapshot) if snapshot is not None else {}
    reasons = snap.get("skip_reasons")
    if not isinstance(reasons, Mapping):
        reasons = {}
    return {
        "pairs_ok": int(snap.get("pairs_ok", 0) or 0),
        "pairs_failed": int(snap.get("pairs_failed", 0) or 0),
        "pairs_skipped": int(snap.get("pairs_skipped", 0) or 0),
        "log_errors": int(snap.get("log_errors", 0) or 0),
        "over_budget": int(snap.get("over_budget", 0) or 0),
        "last_skip_reason": snap.get("last_skip_reason"),
        "skip_reasons": {
            str(key): int(value)
            for key, value in reasons.items()
        },
    }


def _publish_status(status_out: Any, record: Mapping[str, Any]) -> None:
    offer_nonblocking(status_out, _stamp_honesty(_strip_planner_keys(record)))


def _append_jsonl(log_path: str | None, record: Mapping[str, Any]) -> bool:
    if not log_path:
        return False
    payload = _stamp_honesty({"schema": SCHEMA, **dict(record)})
    payload["schema"] = SCHEMA
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_jsonable(payload), ensure_ascii=True, allow_nan=False)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        return True
    except Exception:
        return False


def _required_stack_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    names = (
        "checkpoint_path",
        "vit_checkpoint_path",
        "dataset_contract",
        "observer_bundle",
        "mujoco_model",
    )
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for name in names:
        raw = config.get(name)
        if raw is None:
            missing.append(name)
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_file():
            missing.append(name)
            continue
        resolved[name] = path
    if missing:
        raise RuntimeError(
            "refusing to start: child stack files missing or not files: "
            + ", ".join(missing)
        )
    return resolved


def _load_production_stack(config: Mapping[str, Any]) -> Any:
    """Child-only load.  Imports the threadless engine here.  Never starts a runner."""

    paths = _required_stack_paths(config)
    from fcpg.dicg_shadow import DICGShadowEngine

    return DICGShadowEngine.from_paths(
        checkpoint_path=paths["checkpoint_path"],
        vit_checkpoint_path=paths["vit_checkpoint_path"],
        log_path=str(config.get("log_path") or paths["checkpoint_path"]),
        dataset_contract=paths["dataset_contract"],
        observer_bundle=paths["observer_bundle"],
        mujoco_model=paths["mujoco_model"],
        extrinsic_index=config.get("extrinsic_index"),
        extrinsic_group_key=config.get("extrinsic_group_key"),
        device=str(config.get("device", "cpu")),
        top_k=int(config.get("top_k") or DEFAULT_TOP_K),
        latency_budget_s=float(config.get("latency_budget_s") or DEFAULT_LATENCY_BUDGET_S),
        honesty_fields=honesty_fields,
    )


def _child_ignore_parent_signals() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _parent_is_gone(parent_pid: int) -> bool:
    if parent_pid <= 0:
        return False
    try:
        os.kill(parent_pid, 0)
    except OSError:
        return True
    return False


def _child_main(
    config: dict[str, Any],
    inbound: Any,
    status_out: Any,
    stop: Any,
) -> None:
    """Default child: one inbound queue, optional engine load, no Runner thread."""

    _child_ignore_parent_signals()
    parent_pid = int(config.get("parent_pid") or 0)
    engine: Any | None = None
    loaded = False
    load_error: str | None = None
    try:
        if bool(config.get("skip_production_load")):
            loaded = False
        else:
            engine = _load_production_stack(config)
            loaded = True
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
        _publish_status(
            status_out,
            {
                "ready": False,
                "loaded": False,
                "engine_wired": False,
                "worker_alive": True,
                "load_error": load_error,
                **_counter_status(),
            },
        )
        return

    _publish_status(
        status_out,
        {
            "ready": True,
            "loaded": loaded,
            "engine_wired": engine is not None,
            "worker_alive": True,
            "load_error": None,
            "phase": 2,
            **_counter_status(),
        },
    )
    previous: dict[str, Any] | None = None
    counters = _counter_status()
    while not stop.is_set():
        if _parent_is_gone(parent_pid):
            break
        try:
            item = inbound.get(timeout=_WORKER_POLL_S)
        except queue.Empty:
            continue
        except Exception:
            break
        if not isinstance(item, Mapping) or set(item) != IPC_PAYLOAD_KEYS:
            continue
        if previous is None:
            previous = dict(item)
            continue
        if engine is not None:
            try:
                engine.safe_process_pair(previous, item)
            except Exception:
                pass
            counters = _counter_status(engine.snapshot())
        else:
            record = {
                "ok": True,
                "record_kind": "phase1_pair_only",
                "engine_wired": False,
                "timestamp_t_s": float(previous["timestamp_s"]),
                "timestamp_tp1_s": float(item["timestamp_s"]),
                "delta_q": [
                    float(left)
                    for left in (
                        np.asarray(item["state"], dtype=np.float32)
                        - np.asarray(previous["state"], dtype=np.float32)
                    ).tolist()
                ],
            }
            if not _append_jsonl(config.get("log_path"), record):
                counters["log_errors"] = int(counters.get("log_errors", 0) or 0) + 1
            else:
                counters["pairs_ok"] = int(counters.get("pairs_ok", 0) or 0) + 1
        _publish_status(
            status_out,
            {
                "ready": True,
                "loaded": loaded,
                "engine_wired": engine is not None,
                "worker_alive": True,
                **counters,
            },
        )
        previous = dict(item)


def child_target_ready_loop(
    config: dict[str, Any],
    inbound: Any,
    status_out: Any,
    stop: Any,
) -> None:
    """Test child: skip production load, pair inbound payloads, write JSONL."""

    config = dict(config)
    config["skip_production_load"] = True
    _child_main(config, inbound, status_out, stop)


def child_target_load_fail(
    config: dict[str, Any],
    inbound: Any,
    status_out: Any,
    stop: Any,
) -> None:
    """Test child: fail closed on load without touching the parent process."""

    del inbound, stop
    _child_ignore_parent_signals()
    _publish_status(
        status_out,
        {
            "ready": False,
            "loaded": False,
            "engine_wired": False,
            "load_error": "injected load failure",
            "pairs_ok": 0,
            "pairs_failed": 0,
            "log_errors": 0,
        },
    )


def child_target_os_exit(
    config: dict[str, Any],
    inbound: Any,
    status_out: Any,
    stop: Any,
) -> None:
    """Test child: fatal abort after ready.  Must not kill the parent."""

    del config, inbound, stop
    _child_ignore_parent_signals()
    _publish_status(
        status_out,
        {
            "ready": True,
            "loaded": False,
            "engine_wired": False,
            "pairs_ok": 0,
            "pairs_failed": 0,
            "log_errors": 0,
        },
    )
    os._exit(1)


def child_target_stuck(
    config: dict[str, Any],
    inbound: Any,
    status_out: Any,
    stop: Any,
) -> None:
    """Test child: stay alive and never read inbound (queue stays full)."""

    del config, inbound
    _child_ignore_parent_signals()
    _publish_status(
        status_out,
        {
            "ready": True,
            "loaded": False,
            "engine_wired": False,
            "pairs_ok": 0,
            "pairs_failed": 0,
            "log_errors": 0,
        },
    )
    while not stop.is_set():
        time.sleep(_WORKER_POLL_S)


def child_target_record_keys(
    config: dict[str, Any],
    inbound: Any,
    status_out: Any,
    stop: Any,
) -> None:
    """Test child: persist the exact inbound key set for the IPC whitelist."""

    _child_ignore_parent_signals()
    _publish_status(
        status_out,
        {
            "ready": True,
            "loaded": False,
            "engine_wired": False,
            "pairs_ok": 0,
            "pairs_failed": 0,
            "log_errors": 0,
        },
    )
    while not stop.is_set():
        try:
            item = inbound.get(timeout=_WORKER_POLL_S)
        except queue.Empty:
            continue
        except Exception:
            break
        if not isinstance(item, Mapping):
            continue
        probe = {
            "keys": sorted(item.keys()),
            "state_shape": list(np.asarray(item.get("state")).shape)
            if "state" in item
            else None,
            "column_shape": list(np.asarray(item.get("column")).shape)
            if "column" in item
            else None,
            "column_dtype": str(np.asarray(item.get("column")).dtype)
            if "column" in item
            else None,
        }
        path = config.get("probe_path")
        if path:
            try:
                Path(path).write_text(
                    json.dumps(probe, ensure_ascii=True), encoding="utf-8"
                )
            except Exception:
                pass
        _publish_status(
            status_out,
            {
                "ready": True,
                "loaded": False,
                "engine_wired": False,
                "last_ipc_keys": list(probe["keys"]),
                **_counter_status(),
            },
        )


def child_target_skip_telemetry(
    config: dict[str, Any],
    inbound: Any,
    status_out: Any,
    stop: Any,
) -> None:
    """Test child: publish skip/over_budget counters without loading the stack."""

    del config, inbound
    _child_ignore_parent_signals()
    _publish_status(
        status_out,
        {
            "ready": True,
            "loaded": True,
            "engine_wired": True,
            "worker_alive": True,
            **_counter_status(
                {
                    "pairs_ok": 0,
                    "pairs_failed": 0,
                    "pairs_skipped": 2,
                    "log_errors": 0,
                    "over_budget": 0,
                    "last_skip_reason": SKIP_INSUFFICIENT_ARM_DQ,
                    "skip_reasons": {
                        SKIP_INSUFFICIENT_ARM_DQ: 2,
                        SKIP_INSUFFICIENT_EXCITATION: 0,
                    },
                }
            ),
        },
    )
    while not stop.is_set():
        time.sleep(_WORKER_POLL_S)


class DICGShadowSidecar:
    """Spawn-only parent client.  Public API: submit, status, close."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        vit_checkpoint_path: str | Path,
        log_path: str | Path,
        dataset_contract: str | Path | None = None,
        observer_bundle: str | Path | None = None,
        mujoco_model: str | Path | None = None,
        extrinsic_index: int | str | None = None,
        extrinsic_group_key: str | None = None,
        device: str = "cpu",
        top_k: int = DEFAULT_TOP_K,
        latency_budget_s: float = DEFAULT_LATENCY_BUDGET_S,
        child_target: Callable[..., Any] | None = None,
        ready_timeout_s: float = 2.0,
        skip_production_load: bool = False,
        probe_path: str | Path | None = None,
    ) -> None:
        label = str(device).strip()
        if label not in {"cpu", "cuda:0"}:
            raise DICGShadowSidecarError(
                f"device must be 'cpu' or 'cuda:0', got {device!r}"
            )
        budget = float(latency_budget_s)
        if not np.isfinite(budget) or budget <= 0.0:
            raise DICGShadowSidecarError("latency_budget_s must be finite and positive")
        timeout = float(ready_timeout_s)
        if not np.isfinite(timeout) or timeout < 0.0:
            raise DICGShadowSidecarError("ready_timeout_s must be finite and non-negative")

        self._device = label
        self._top_k = max(1, int(top_k))
        self._latency_budget_s = budget
        self._checkpoint_path = str(Path(checkpoint_path).expanduser())
        self._vit_path = str(Path(vit_checkpoint_path).expanduser())
        self._log_path = str(Path(log_path).expanduser())
        self._lock = threading.Lock()
        self._closed = False
        self._submitted = 0
        self._dropped = 0
        self._submit_errors = 0
        self._child_status: dict[str, Any] = {}
        self._pending: dict[str, Any] | None = None
        self._inbound = None
        self._status_out = None
        self._stop = None
        self._flush_stop = threading.Event()
        self._flush_thread: threading.Thread | None = None
        self._proc = None

        target = child_target if child_target is not None else _child_main
        if not callable(target) or inspect.isfunction(target) is False:
            raise DICGShadowSidecarError("child_target must be a top-level function")
        if target.__module__ in {"__main__", ""}:
            raise DICGShadowSidecarError(
                "child_target must be importable from a real module for spawn"
            )

        ctx = multiprocessing.get_context(START_METHOD)
        config = _child_config(
            checkpoint_path=checkpoint_path,
            vit_checkpoint_path=vit_checkpoint_path,
            log_path=log_path,
            dataset_contract=dataset_contract,
            observer_bundle=observer_bundle,
            mujoco_model=mujoco_model,
            extrinsic_index=extrinsic_index,
            extrinsic_group_key=extrinsic_group_key,
            device=label,
            top_k=self._top_k,
            latency_budget_s=budget,
        )
        config["skip_production_load"] = bool(skip_production_load)
        config["probe_path"] = _path_text(probe_path)
        inbound = ctx.Queue(maxsize=QUEUE_MAXSIZE)
        status_out = ctx.Queue(maxsize=QUEUE_MAXSIZE)
        stop = ctx.Event()
        proc = ctx.Process(
            target=target,
            args=(config, inbound, status_out, stop),
            name="dicg-shadow-sidecar",
            daemon=False,
        )
        self._inbound = inbound
        self._status_out = status_out
        self._stop = stop
        self._proc = proc
        try:
            proc.start()
        except Exception as exc:
            self._closed = True
            raise DICGShadowSidecarError(
                f"failed to spawn sidecar child ({type(exc).__name__}: {exc})"
            ) from exc
        atexit.register(self.close)
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="dicg-shadow-sidecar-flush",
            daemon=True,
        )
        self._flush_thread.start()
        self._wait_for_first_status(timeout)

    def _try_flush(self) -> None:
        """Push the newest pending snapshot if the inbound slot is free.

        Never ``get()`` the inbound ``mp.Queue``: a 150KiB column can sit in
        the feeder pipe, and a parent ``get_nowait`` then races to ``Full``.
        """

        with self._lock:
            if self._closed:
                return
            item = self._pending
            mailbox = self._inbound
        if item is None or mailbox is None:
            return
        try:
            mailbox.put_nowait(item)
        except queue.Full:
            return
        except Exception:
            return
        with self._lock:
            if self._pending is item:
                self._pending = None

    def _flush_loop(self) -> None:
        while not self._flush_stop.is_set():
            self._try_flush()
            time.sleep(_WORKER_POLL_S)

    def submit(self, observation: Any) -> bool:
        try:
            with self._lock:
                if self._closed:
                    return False
                proc = self._proc
            if proc is None or not proc.is_alive():
                return False
            snapshot = _copy_ipc_payload(observation)
            with self._lock:
                if self._pending is not None:
                    self._dropped += 1
                self._pending = snapshot
                self._submitted += 1
            self._try_flush()
            return True
        except Exception:
            with self._lock:
                self._submit_errors += 1
            return False

    def status(self) -> dict[str, Any]:
        self._try_flush()
        self._drain_status()
        proc = self._proc
        alive = bool(proc is not None and proc.is_alive())
        with self._lock:
            child = dict(self._child_status)
            payload = {
                "closed": bool(self._closed),
                "device": self._device,
                "queue_maxsize": QUEUE_MAXSIZE,
                "start_method": START_METHOD,
                "submitted": int(self._submitted),
                "dropped": int(self._dropped),
                "submit_errors": int(self._submit_errors),
                "pairs_ok": int(child.get("pairs_ok", 0) or 0),
                "pairs_failed": int(child.get("pairs_failed", 0) or 0),
                "pairs_skipped": int(child.get("pairs_skipped", 0) or 0),
                "last_skip_reason": child.get("last_skip_reason"),
                "skip_reasons": dict(child.get("skip_reasons") or {}),
                "log_errors": int(child.get("log_errors", 0) or 0),
                "over_budget": int(child.get("over_budget", 0) or 0),
                "worker_alive": alive,
                "child_pid": None if proc is None else proc.pid,
                "ready": bool(child.get("ready", False)),
                "loaded": bool(child.get("loaded", False)),
                "engine_wired": bool(child.get("engine_wired", False)),
                "load_error": child.get("load_error"),
                "checkpoint_path": self._checkpoint_path,
                "vit_path": self._vit_path,
                "log_path": self._log_path,
                "phase": 2,
            }
        return _stamp_honesty(_strip_planner_keys(payload))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._flush_stop.set()
        flush = self._flush_thread
        if flush is not None and flush.is_alive():
            flush.join(timeout=0.5)
        stop = self._stop
        proc = self._proc
        if stop is not None:
            try:
                stop.set()
            except Exception:
                pass
        if proc is not None and proc.is_alive():
            proc.join(timeout=_CLOSE_JOIN_S)
        if proc is not None and proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass
            proc.join(timeout=_TERMINATE_JOIN_S)
        if proc is not None and proc.is_alive():
            try:
                proc.kill()
            except Exception:
                pass
            proc.join(timeout=_TERMINATE_JOIN_S)
        self._close_mailbox(self._inbound)
        self._close_mailbox(self._status_out)

    def _close_mailbox(self, mailbox: Any) -> None:
        if mailbox is None:
            return
        closer = getattr(mailbox, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
        cancel = getattr(mailbox, "cancel_join_thread", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass

    def _drain_status(self) -> None:
        mailbox = self._status_out
        if mailbox is None:
            return
        latest: dict[str, Any] | None = None
        while True:
            try:
                item = mailbox.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            if isinstance(item, Mapping):
                latest = dict(item)
        if latest is None:
            return
        with self._lock:
            self._child_status = _stamp_honesty(_strip_planner_keys(latest))

    def _wait_for_first_status(self, timeout_s: float) -> None:
        deadline = time.perf_counter() + float(timeout_s)
        while time.perf_counter() < deadline:
            self._drain_status()
            with self._lock:
                if self._child_status:
                    return
            proc = self._proc
            if proc is None or not proc.is_alive():
                return
            time.sleep(0.01)
        self._drain_status()
