"""Persistent video decoders isolated from the PyTorch/CUDA process.

The sidecar builder imports PyTorch, Hugging Face, PyAV and OpenCV in one long
running job.  Forking that process can inherit locks owned by vanished ATen or
OpenMP threads and has produced deterministic ``futex_wait_queue`` stalls.
This module instead starts plain Python subprocesses over Unix socket pairs.
Each worker imports only the archived-video reader and remains alive for the
whole build.  No worker forks the model process or imports torch.
"""

from __future__ import annotations

import argparse
import atexit
import os
import queue
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from threading import Lock
from typing import Any, Iterable


WORKER_PROTOCOL = "so101-isolated-video-decode-v1"
WORKER_START_TIMEOUT_SECONDS = 30.0
WORKER_STOP_TIMEOUT_SECONDS = 5.0


@dataclass
class _Worker:
    process: subprocess.Popen[bytes]
    connection: Connection
    lock: Lock
    broken: bool = False


class IsolatedVideoDecodePool:
    """A persistent, bounded pool of torch-free video subprocesses."""

    def __init__(self, *, workers: int) -> None:
        if workers <= 0:
            raise ValueError("isolated video decoder count must be positive")
        self._workers: list[_Worker] = []
        self._available: queue.Queue[_Worker] = queue.Queue(maxsize=workers)
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="isolated-video-decode",
        )
        self._closed = False
        try:
            for _ in range(workers):
                worker = self._start_worker()
                self._workers.append(worker)
                self._available.put_nowait(worker)
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    @staticmethod
    def _start_worker() -> _Worker:
        parent_socket, child_socket = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        )
        child_fd = child_socket.fileno()
        module = Path(__file__).resolve()
        environment = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = "1"
        process = subprocess.Popen(
            (sys.executable, str(module), "--worker-fd", str(child_fd)),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(child_fd,),
            env=environment,
        )
        child_socket.close()
        connection = Connection(parent_socket.detach())
        worker = _Worker(process=process, connection=connection, lock=Lock())
        if not connection.poll(WORKER_START_TIMEOUT_SECONDS):
            IsolatedVideoDecodePool._terminate(worker)
            raise TimeoutError("isolated video decoder did not become ready")
        try:
            status, payload = connection.recv()
        except (EOFError, OSError) as exc:
            IsolatedVideoDecodePool._terminate(worker)
            raise RuntimeError("isolated video decoder exited during startup") from exc
        if status != "ready" or payload.get("protocol") != WORKER_PROTOCOL:
            IsolatedVideoDecodePool._terminate(worker)
            raise RuntimeError(f"invalid isolated video decoder handshake: {status!r}")
        if bool(payload.get("torch_loaded")):
            IsolatedVideoDecodePool._terminate(worker)
            raise RuntimeError("isolated video decoder imported torch")
        return worker

    @property
    def worker_pids(self) -> tuple[int, ...]:
        return tuple(worker.process.pid for worker in self._workers)

    def _run_one(self, payload: Any, *, deadline: float) -> Any:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("isolated video decode batch exceeded its deadline")
        try:
            worker = self._available.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("no isolated video decoder became available") from exc
        reusable = False
        try:
            with worker.lock:
                if worker.broken or worker.process.poll() is not None:
                    raise RuntimeError("isolated video decoder is not alive")
                worker.connection.send(payload)
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not worker.connection.poll(remaining):
                    worker.broken = True
                    self._terminate(worker)
                    path = payload[0] if isinstance(payload, tuple) and payload else "unknown"
                    raise TimeoutError(
                        "isolated video decode timed out for path " + str(path)
                    )
                status, result = worker.connection.recv()
                if status != "ok":
                    worker.broken = True
                    self._terminate(worker)
                    raise RuntimeError(
                        "isolated video decoder failed:\n" + str(result)
                    )
                reusable = True
                return result
        except (EOFError, BrokenPipeError, OSError) as exc:
            worker.broken = True
            self._terminate(worker)
            raise RuntimeError("isolated video decoder connection failed") from exc
        finally:
            if reusable and not self._closed:
                self._available.put(worker)

    def map_ordered(
        self,
        payloads: Iterable[Any],
        *,
        timeout: float,
    ) -> list[Any]:
        """Decode all jobs concurrently and return them in submission order."""

        if self._closed:
            raise RuntimeError("isolated video decoder pool is closed")
        jobs = tuple(payloads)
        if timeout <= 0:
            raise ValueError("isolated video decode timeout must be positive")
        if not jobs:
            return []
        deadline = time.monotonic() + float(timeout)
        futures: dict[Future[Any], int] = {
            self._executor.submit(self._run_one, job, deadline=deadline): index
            for index, job in enumerate(jobs)
        }
        ordered: list[Any | None] = [None] * len(jobs)
        try:
            for future in as_completed(futures, timeout=timeout):
                ordered[futures[future]] = future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        if any(value is None for value in ordered):
            raise RuntimeError("isolated video decoder omitted a result")
        return ordered

    @staticmethod
    def _terminate(worker: _Worker) -> None:
        try:
            worker.connection.close()
        except OSError:
            pass
        if worker.process.poll() is None:
            worker.process.terminate()
            try:
                worker.process.wait(timeout=WORKER_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait(timeout=WORKER_STOP_TIMEOUT_SECONDS)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        for worker in self._workers:
            if worker.broken or worker.process.poll() is not None:
                self._terminate(worker)
                continue
            acknowledged = False
            try:
                worker.connection.send(None)
                if worker.connection.poll(WORKER_STOP_TIMEOUT_SECONDS):
                    status, _ = worker.connection.recv()
                    acknowledged = status == "stopped"
            except (EOFError, BrokenPipeError, OSError):
                pass
            if acknowledged:
                worker.connection.close()
                try:
                    worker.process.wait(timeout=WORKER_STOP_TIMEOUT_SECONDS)
                    continue
                except subprocess.TimeoutExpired:
                    pass
            self._terminate(worker)
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    def __enter__(self) -> "IsolatedVideoDecodePool":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _worker_main(fd: int) -> int:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from fcpg.lerobot_archive import decode_video_frames_letterboxed_job

    connection = Connection(fd)
    connection.send(
        (
            "ready",
            {
                "protocol": WORKER_PROTOCOL,
                "pid": os.getpid(),
                "torch_loaded": "torch" in sys.modules,
            },
        )
    )
    try:
        while True:
            payload = connection.recv()
            if payload is None:
                connection.send(("stopped", None))
                return 0
            try:
                result = decode_video_frames_letterboxed_job(payload)
                connection.send(("ok", result))
            except BaseException:
                connection.send(("error", traceback.format_exc()))
    except EOFError:
        return 0
    finally:
        connection.close()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-fd", type=int, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(_worker_main(_argument_parser().parse_args().worker_fd))
