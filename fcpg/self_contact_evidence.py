"""Frame-indexed visual evidence for ``self_contact_seek_v3``.

Historical ``self_contact_seek_v1`` / ``v2`` sidecars stay readable and
are not rewritten.  A future live session must emit this schema.  Images
are stored so command-0 jaw gap can be reviewed offline.  A disk
preflight failure refuses before the first Goal write.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fcpg.self_contact_v1 import PLANNING_CANVAS_HW, is_planning_224_hw

EVIDENCE_SCHEMA_VERSION = 3
SEEK_SCHEMA_V3 = "self_contact_seek_v3"
FRAME_INDEX_DTYPE = np.int32
PROBE_FILENAME = ".self_contact_evidence_probe"
DEFAULT_MIN_FREE_BYTES = 32 * 1024 * 1024
PLANNING_BYTES_PER_FRAME = 2 * 224 * 224 * 3  # column + wrist planning224


class EvidenceStoreError(RuntimeError):
    """Fail-closed evidence refusal.  Not a contact verdict."""


def evidence_schema_fields() -> dict[str, Any]:
    return {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "seek_schema": SEEK_SCHEMA_V3,
        "images_persisted_complete": False,
        "per_view_roi": {"column": None, "wrist": None},
        "pre_motion_handshake": None,
    }


def reference_frame_hash(frame: np.ndarray) -> str:
    rgb = np.ascontiguousarray(np.asarray(frame), dtype=np.uint8)
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def reference_pair_hash(column: np.ndarray, wrist: np.ndarray) -> str:
    payload = (reference_frame_hash(column) + ":" + reference_frame_hash(wrist)).encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def estimate_evidence_bytes(
    n_frames: int,
    *,
    include_native: bool = False,
    native_hw: tuple[int, int] = (480, 640),
) -> int:
    planning = int(n_frames) * PLANNING_BYTES_PER_FRAME
    native = 0
    if include_native:
        height, width = int(native_hw[0]), int(native_hw[1])
        native = int(n_frames) * 2 * height * width * 3
    return planning + native + 2 * 1024 * 1024


def preflight_evidence_store(
    output_dir: str | Path,
    *,
    n_frames_budget: int = 400,
    include_native: bool = False,
    min_free_bytes: int | None = None,
) -> dict[str, Any]:
    """Probe the destination before any motion command.  Fail closed."""

    root = Path(output_dir)
    needed = estimate_evidence_bytes(n_frames_budget, include_native=include_native)
    floor = int(min_free_bytes if min_free_bytes is not None else DEFAULT_MIN_FREE_BYTES)
    required = max(needed, floor)
    evidence: dict[str, Any] = {
        "ok": False,
        "path": str(root),
        "required_bytes": required,
        "before_first_command": True,
    }
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        evidence["reason"] = f"disk_preflight_mkdir:{exc}"
        return evidence
    probe = root / PROBE_FILENAME
    try:
        payload = b"self_contact_evidence_v3_probe\n" + os.urandom(64)
        probe.write_bytes(payload)
        readback = probe.read_bytes()
        if readback != payload:
            evidence["reason"] = "disk_preflight_readback_mismatch"
            return evidence
        probe.unlink()
    except OSError as exc:
        evidence["reason"] = f"disk_preflight_write:{exc}"
        return evidence
    try:
        stat = os.statvfs(root)
        free = int(stat.f_bavail) * int(stat.f_frsize)
        evidence["free_bytes"] = free
        if free < required:
            evidence["reason"] = "disk_preflight_insufficient_space"
            return evidence
    except OSError as exc:
        evidence["reason"] = f"disk_preflight_statvfs:{exc}"
        return evidence
    evidence["ok"] = True
    evidence["reason"] = None
    return evidence


def _as_rgb(frame: Any) -> np.ndarray:
    rgb = np.asarray(frame)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {getattr(rgb, 'shape', None)}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def empty_frame_row(index: int, *, phase: str | None = None) -> dict[str, Any]:
    return {
        "index": int(index),
        "frame_id": int(index),
        "phase": phase,
        "timestamp_s": None,
        "command": None,
        "present": None,
        "views": {
            "column": {"planning_224": False, "native": False},
            "wrist": {"planning_224": False, "native": False},
        },
    }


class SeekEvidenceRecorder:
    """In-memory then fail-closed npz writer.  Lengths must stay aligned."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        n_frames_budget: int = 400,
        include_native: bool = False,
        require_disk: bool = False,
    ) -> None:
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.n_frames_budget = int(n_frames_budget)
        self.include_native = bool(include_native)
        self.require_disk = bool(require_disk)
        self.preflight: dict[str, Any] | None = None
        self.column_planning: list[np.ndarray] = []
        self.wrist_planning: list[np.ndarray] = []
        self.column_native: list[np.ndarray | None] = []
        self.wrist_native: list[np.ndarray | None] = []
        self.timestamps: list[float] = []
        self.frame_ids: list[int] = []
        self.commands: list[float | None] = []
        self.presents: list[float | None] = []
        self.phases: list[str] = []
        self.column_canvas: list[dict[str, Any]] = []
        self.wrist_canvas: list[dict[str, Any]] = []
        self.column_letterbox: list[dict[str, Any]] = []
        self.wrist_letterbox: list[dict[str, Any]] = []
        self.telemetry: list[dict[str, Any]] = []
        self.persist_error: str | None = None
        self.images_persisted_complete = False
        self.paths: dict[str, str] = {}

    def run_preflight(self) -> dict[str, Any]:
        if self.output_dir is None:
            if self.require_disk:
                self.preflight = {
                    "ok": False,
                    "reason": "disk_preflight_missing_output_dir",
                    "before_first_command": True,
                }
                return self.preflight
            self.preflight = {
                "ok": True,
                "reason": None,
                "in_memory_only": True,
                "before_first_command": True,
            }
            return self.preflight
        self.preflight = preflight_evidence_store(
            self.output_dir,
            n_frames_budget=self.n_frames_budget,
            include_native=self.include_native,
        )
        return self.preflight

    def __len__(self) -> int:
        return len(self.frame_ids)

    def append(
        self,
        *,
        column_planning: np.ndarray,
        wrist_planning: np.ndarray,
        timestamp_s: float,
        phase: str,
        command: float | None,
        present: float | None,
        frame_id: int | None = None,
        column_native: np.ndarray | None = None,
        wrist_native: np.ndarray | None = None,
        column_canvas: Mapping[str, Any] | None = None,
        wrist_canvas: Mapping[str, Any] | None = None,
        column_letterbox: Mapping[str, Any] | None = None,
        wrist_letterbox: Mapping[str, Any] | None = None,
        telemetry: Mapping[str, Any] | None = None,
    ) -> None:
        col = _as_rgb(column_planning)
        wri = _as_rgb(wrist_planning)
        if not is_planning_224_hw(col.shape) or not is_planning_224_hw(wri.shape):
            raise EvidenceStoreError(
                f"planning frames must be {PLANNING_CANVAS_HW}, "
                f"got column={col.shape} wrist={wri.shape}"
            )
        index = len(self.frame_ids)
        self.column_planning.append(np.ascontiguousarray(col))
        self.wrist_planning.append(np.ascontiguousarray(wri))
        self.column_native.append(
            None if column_native is None else _as_rgb(column_native)
        )
        self.wrist_native.append(None if wrist_native is None else _as_rgb(wrist_native))
        self.timestamps.append(float(timestamp_s))
        self.frame_ids.append(index if frame_id is None else int(frame_id))
        self.commands.append(None if command is None else float(command))
        self.presents.append(None if present is None else float(present))
        self.phases.append(str(phase))
        self.column_canvas.append(dict(column_canvas or {}))
        self.wrist_canvas.append(dict(wrist_canvas or {}))
        self.column_letterbox.append(dict(column_letterbox or {}))
        self.wrist_letterbox.append(dict(wrist_letterbox or {}))
        self.telemetry.append({} if telemetry is None else dict(telemetry))

    def length_contract(self) -> dict[str, Any]:
        n = len(self.frame_ids)
        lengths = {
            "frame_ids": n,
            "column_planning": len(self.column_planning),
            "wrist_planning": len(self.wrist_planning),
            "timestamps": len(self.timestamps),
            "commands": len(self.commands),
            "presents": len(self.presents),
            "phases": len(self.phases),
            "telemetry": len(self.telemetry),
        }
        aligned = all(value == n for value in lengths.values())
        return {"ok": aligned, "n": n, "lengths": lengths}

    def arrays(self) -> dict[str, np.ndarray]:
        contract = self.length_contract()
        if not contract["ok"]:
            raise EvidenceStoreError(
                f"evidence length mismatch: {contract['lengths']}"
            )
        payload: dict[str, np.ndarray] = {
            "column_planning_224": np.stack(self.column_planning, axis=0)
            if self.column_planning
            else np.zeros((0, 224, 224, 3), dtype=np.uint8),
            "wrist_planning_224": np.stack(self.wrist_planning, axis=0)
            if self.wrist_planning
            else np.zeros((0, 224, 224, 3), dtype=np.uint8),
            "timestamps": np.asarray(self.timestamps, dtype=np.float64),
            "frame_ids": np.asarray(self.frame_ids, dtype=FRAME_INDEX_DTYPE),
            "commands": np.asarray(
                [np.nan if item is None else float(item) for item in self.commands],
                dtype=np.float32,
            ),
            "presents": np.asarray(
                [np.nan if item is None else float(item) for item in self.presents],
                dtype=np.float32,
            ),
            "phases": np.asarray(self.phases, dtype=object),
        }
        if any(frame is not None for frame in self.column_native):
            payload["column_native"] = _stack_optional(self.column_native)
        if any(frame is not None for frame in self.wrist_native):
            payload["wrist_native"] = _stack_optional(self.wrist_native)
        return payload

    def persist(self) -> dict[str, Any]:
        contract = self.length_contract()
        if not contract["ok"]:
            self.images_persisted_complete = False
            self.persist_error = f"evidence_length_mismatch:{contract['lengths']}"
            return self.summary()
        if int(contract["n"]) < 1:
            self.images_persisted_complete = False
            self.persist_error = self.persist_error or "evidence_empty"
            if self.output_dir is None and self.require_disk:
                self.persist_error = "evidence_dir_missing"
            return self.summary()
        if self.output_dir is None:
            self.images_persisted_complete = bool(contract["ok"]) and not self.require_disk
            if self.require_disk:
                self.persist_error = "evidence_dir_missing"
                self.images_persisted_complete = False
            return self.summary()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            npz_path = self.output_dir / "self_contact.npz"
            arrays = self.arrays()
            np.savez_compressed(npz_path, **arrays)
            meta = {
                "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                "n_frames": int(contract["n"]),
                "keys": sorted(arrays),
                "column_canvas": self.column_canvas,
                "wrist_canvas": self.wrist_canvas,
                "column_letterbox": self.column_letterbox,
                "wrist_letterbox": self.wrist_letterbox,
                "telemetry": self.telemetry,
                "images_persisted_complete": True,
            }
            meta_path = self.output_dir / "self_contact_evidence.json"
            meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            self.paths = {
                "npz": str(npz_path),
                "evidence_json": str(meta_path),
            }
            self.images_persisted_complete = True
            self.persist_error = None
        except OSError as exc:
            self.images_persisted_complete = False
            self.persist_error = f"evidence_persist:{exc}"
        return self.summary()

    def summary(self) -> dict[str, Any]:
        contract = self.length_contract()
        return {
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "images_persisted_complete": bool(self.images_persisted_complete),
            "persist_error": self.persist_error,
            "n_frames": int(contract["n"]),
            "length_contract": contract,
            "preflight": self.preflight,
            "paths": dict(self.paths),
            "require_disk": self.require_disk,
        }


def _stack_optional(frames: Sequence[np.ndarray | None]) -> np.ndarray:
    ready = [frame for frame in frames if frame is not None]
    if not ready:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)
    shape = ready[0].shape
    stacked = []
    for frame in frames:
        if frame is None:
            stacked.append(np.zeros(shape, dtype=np.uint8))
        else:
            stacked.append(frame)
    return np.stack(stacked, axis=0)


def validate_images_persisted_complete(
    record: Mapping[str, Any],
    *,
    require_disk: bool = True,
) -> dict[str, Any]:
    """Third-session instrument success is illegal without complete images."""

    complete = bool(record.get("images_persisted_complete"))
    version = record.get("evidence_schema_version")
    reasons: list[str] = []
    if version not in {EVIDENCE_SCHEMA_VERSION, str(EVIDENCE_SCHEMA_VERSION)}:
        reasons.append("evidence_schema_version_missing")
    if not complete:
        reasons.append("images_not_persisted_complete")
    if require_disk and not (record.get("evidence_paths") or record.get("paths")):
        reasons.append("evidence_paths_missing")
    return {
        "ok": not reasons,
        "instrument_success_allowed": not reasons,
        "reasons": reasons,
    }


def load_indexed_frame(
    store: Mapping[str, np.ndarray] | np.lib.npyio.NpzFile,
    index: int,
    *,
    view: str,
    canvas: str = "planning_224",
) -> np.ndarray:
    key = {
        ("column", "planning_224"): "column_planning_224",
        ("wrist", "planning_224"): "wrist_planning_224",
        ("column", "native"): "column_native",
        ("wrist", "native"): "wrist_native",
    }[(str(view), str(canvas))]
    stack = np.asarray(store[key])
    if index < 0 or index >= len(stack):
        raise IndexError(f"{key} has no frame {index}")
    return np.asarray(stack[index])
