"""Strict V20.3 hardware transaction adapter.

This module owns no serial connection.  A caller supplies the established
lease-owning IO object and an endpoint capture callback.  The old V17
``execute_v177_chunk`` is deliberately not used: it adds a hidden settled
endpoint after q5 which cannot be attributed to the V20.3 forecast.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np
import torch


class V203TransactionBridge:
    def __init__(self, runtime, io, *, capture_endpoint: Callable, read_q: Callable | None = None,
                 write_q: Callable | None = None, halt: Callable | None = None,
                 physical_check: Callable | None = None, period_s: float = .2,
                 clock: Callable = time.monotonic, wait: Callable = time.sleep,
                 cancelled: Callable[[], bool] | None = None):
        if period_s <= 0 or not np.isfinite(period_s):
            raise ValueError("V20.3 transaction period must be finite and positive")
        self.runtime, self.io, self.capture_endpoint = runtime, io, capture_endpoint
        self.read_q = read_q or io._read_state_only
        self.write_q = write_q or io._write_position_command
        self.halt = halt or (lambda: None)
        self.physical_check = physical_check or (lambda _q, _a: None)
        self.period_s, self.clock, self.wait = float(period_s), clock, wait
        self.cancelled = cancelled or (lambda: False)
        self.lock = threading.RLock()
        self.last_endpoint = None
        self.last_prediction = None
        self.last_forecast = None
        self.last_execution = None
        self.last_attempt = None

    def set_endpoint(self, endpoint):
        """Install a freshly captured endpoint for the next explicit plan."""
        with self.lock:
            self.last_endpoint = endpoint
            return endpoint

    def plan(self, endpoint):
        with self.lock:
            if self.runtime.pending is not None:
                self.runtime.discard_pending_for_replan()
            if getattr(self.runtime, "state", None) is None or not hasattr(self.runtime, "refresh_endpoint"):
                self.runtime.observe_initial(endpoint)
            else:
                self.runtime.refresh_endpoint(endpoint)
            self.last_endpoint = endpoint
            prediction = self.runtime.plan(torch.full((1, 5), self.period_s,
                                                        device=endpoint.q_position.device))
            self.last_prediction = prediction
            forecast = self.runtime.status().get("forecast")
            if forecast is not None:
                forecast = dict(forecast)
                forecast["transaction_id"] = float(prediction.source_time_s.item())
                self.last_forecast = forecast
            return self.runtime.status()

    def execute(self):
        with self.lock:
            if self.last_prediction is None:
                raise RuntimeError("V20.3 requires a fresh plan before execution")
            self.last_attempt = dict(completed=False, commands_attempted=0, commands_returned=0,
                                     phase="preflight", timings_s={})
            try:
                if self.cancelled():
                    raise RuntimeError("five-command transaction cancelled")
                q0 = np.asarray(self.read_q(), dtype=np.float32)
                if q0.shape != (6,) or not np.isfinite(q0).all():
                    raise ValueError("initial hardware readback must be finite [6]")
                # Audit the already predicted path before binding its timing
                # epoch.  The clearance audit can take tens of milliseconds;
                # doing it after rebase would make q0 appear late even though
                # no hardware command had been sent.
                preview_commands = self.runtime.pending.commands_position.detach().cpu().numpy()[0]
                self.physical_check(q0, preview_commands)
                q0 = np.asarray(self.read_q(), dtype=np.float32)
                if q0.shape != (6,) or not np.isfinite(q0).all():
                    raise ValueError("pre-command hardware readback must be finite [6]")
                # The rebase performs one model prediction to bind the pending
                # action to the actual q epoch.  Its timestamp must be taken
                # from the clock sample immediately before that prediction.
                source = float(self.clock())
                self.last_attempt["source_time_s"] = source
                q0_t = torch.as_tensor(q0, device=self.runtime.core.config.q_loss_scale.device
                                        if hasattr(self.runtime.core.config, "q_loss_scale") else self.runtime.pending.source_q_position.device)[None]
                # Rebase checks that the arm did not move while the model was
                # inferring.  The timestamp is the actual pre-command read.
                self.runtime.rebase_pending_execution_epoch(q0_t, timestamp_s=source)
                after_prediction = float(self.clock())
                self.last_attempt["timings_s"]["rebase_prediction"] = after_prediction - source
                # Preserve the actual readback time. Slow inference must fail
                # the existing deadline; relabeling q0 after inference would
                # hide latency and manufacture a synchronous transaction.
                commands = self.runtime.begin_execution()
                self.last_attempt["timings_s"]["begin_execution"] = float(self.clock()) - after_prediction
                # Capture the rebased prediction actually bound to this
                # transaction.  Once ``runtime.complete`` clears ``pending``
                # the dashboard still needs this exact LeWM path and Actor
                # targets to diagnose ordering and per-step errors.
                forecast = self.runtime.pending
                self.last_forecast = {
                    "source": "lewm_prediction_not_actual_readback",
                    "transaction_id": float(forecast.source_time_s.item()),
                    "source_time_s": forecast.source_time_s.detach().cpu().tolist(),
                    "source_q_position": forecast.source_q_position.detach().cpu().tolist(),
                    "commands_position": forecast.commands_position.detach().cpu().tolist(),
                    "time_deltas_s": forecast.time_deltas.detach().cpu().tolist(),
                    "predicted_q_path": forecast.predicted_q_path.detach().cpu().tolist(),
                }
                path, timestamps = [q0.copy()], [source]
                previous = q0.copy()
                for index, command in enumerate(commands.detach().cpu().numpy()[0]):
                    self.last_attempt["phase"] = f"q{index}"
                    boundary = source + index * self.period_s
                    delay = boundary - self.clock()
                    if delay > 0:
                        self.wait(delay)
                    if self.cancelled():
                        raise RuntimeError("five-command transaction cancelled")
                    if self.clock() - boundary > self.runtime.max_endpoint_skew_s:
                        self.last_attempt["timings_s"]["missed_deadline_by"] = float(self.clock()) - boundary
                        raise RuntimeError(f"q{index} command deadline missed")
                    if index:
                        measured = np.asarray(self.read_q(), dtype=np.float32)
                        if measured.shape != (6,) or not np.isfinite(measured).all():
                            raise ValueError(f"q{index} readback invalid")
                        path.append(measured.copy()); timestamps.append(float(self.clock()))
                    self.last_attempt["commands_attempted"] += 1
                    actual = np.asarray(self.write_q(command, previous), dtype=np.float32)
                    self.last_attempt["commands_returned"] += 1
                    if actual.shape != (6,) or not np.allclose(actual, command, atol=1e-4, rtol=0):
                        raise RuntimeError("hardware changed a V20.3 command")
                    previous = actual.copy()
                boundary = source + 5 * self.period_s
                delay = boundary - self.clock()
                if delay > 0:
                    self.wait(delay)
                if self.cancelled():
                    raise RuntimeError("five-command transaction cancelled")
                q5 = np.asarray(self.read_q(), dtype=np.float32)
                if q5.shape != (6,) or not np.isfinite(q5).all():
                    raise ValueError("q5 readback invalid")
                path.append(q5.copy()); timestamps.append(float(self.clock()))
                self.last_attempt["phase"] = "endpoint-observation"
                endpoint = self.capture_endpoint(q5.copy(), timestamps[-1])
                measured = torch.as_tensor(np.stack(path), device=commands.device, dtype=commands.dtype)[None]
                readback_times = torch.as_tensor(np.asarray(timestamps), device=commands.device, dtype=torch.float64)[None]
                status = self.runtime.complete(endpoint, commands.detach().clone(), measured, readback_times)
                self.last_endpoint, self.last_execution = endpoint, {
                    "transaction_id": source,
                    "source_time_s": forecast.source_time_s.detach().cpu().tolist(),
                    "source_q_position": forecast.source_q_position.detach().cpu().tolist(),
                    "commands_position": forecast.commands_position.detach().cpu().tolist(),
                    "time_deltas_s": forecast.time_deltas.detach().cpu().tolist(),
                    "predicted_q_path": forecast.predicted_q_path.detach().cpu().tolist(),
                    "measured_q_path": measured.detach().cpu().tolist(),
                    "readback_times_s": readback_times.detach().cpu().tolist(),
                }
                self.last_prediction = None
                self.last_attempt.update(completed=True, phase="complete")
                return status
            except BaseException as error:
                self.last_attempt["error"] = f"{type(error).__name__}: {error}"
                self.runtime.stop()
                self.last_prediction = None
                try:
                    self.halt()
                finally:
                    raise

    def stop(self):
        with self.lock:
            self.runtime.stop()
            self.last_prediction = None
            self.halt()

    def status(self):
        with self.lock:
            runtime = self.runtime.status()
            if runtime.get("forecast") is None and self.last_forecast is not None:
                runtime["forecast"] = self.last_forecast
            if self.last_execution is not None:
                runtime["last_execution"] = self.last_execution
            return {"controller": {"connected": True, "ready": self.last_prediction is not None,
                                    "capabilities": {"plan": True, "execute": self.last_prediction is not None,
                                                      "home": True, "stop": True}},
                    "runtime": runtime,
                    "last_execution": self.last_execution, "last_attempt": self.last_attempt}

    def handle(self, name, payload=None):
        if name == "plan":
            return self.plan(self.last_endpoint if self.last_endpoint is not None else
                             (_ for _ in ()).throw(RuntimeError("bridge needs a current endpoint")))
        if name == "execute":
            return self.execute()
        if name == "stop":
            self.stop(); return self.status()
        raise RuntimeError(f"V20.3 bridge operation is not wired: {name}")

    def close(self):
        self.stop()
