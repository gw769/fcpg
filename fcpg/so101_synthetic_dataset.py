"""Storage and sampling helpers for synthetic SO-101 FCPG supervision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from fcpg.so101_synthetic_grounding import SO101CameraDrift


DATASET_FORMAT = "so101-synthetic-fcpg-v2"
LEGACY_DATASET_FORMAT = "so101-synthetic-fcpg-v1"
ARRAY_NAMES = (
    "rgb",
    "states",
    "anchors_xyz",
    "q_nominal",
    "q_true",
    "valid",
    "drift",
    "camera_index",
    "camera_context",
    "drift_variant",
    "split",
    "source_index",
    "source_row",
    "episode",
    "task",
)


def balanced_counts(total: int, groups: int) -> np.ndarray:
    if total <= 0 or groups <= 0:
        raise ValueError("total and groups must be positive")
    counts = np.full(groups, total // groups, dtype=np.int64)
    counts[: total % groups] += 1
    return counts


def sample_camera_drift(
    rng: np.random.Generator,
    *,
    clean_probability: float,
    max_azimuth_degrees: float,
    max_elevation_degrees: float,
    max_distance_meters: float,
) -> SO101CameraDrift:
    if not 0.0 <= clean_probability <= 1.0:
        raise ValueError("clean_probability must lie in [0, 1]")
    if min(
        max_azimuth_degrees,
        max_elevation_degrees,
        max_distance_meters,
    ) < 0:
        raise ValueError("camera drift limits must be non-negative")
    if float(rng.random()) < clean_probability:
        return SO101CameraDrift()
    azimuth = float(rng.uniform(-max_azimuth_degrees, max_azimuth_degrees))
    if max_azimuth_degrees > 1.0 and abs(azimuth) < 1.0:
        azimuth = float(np.copysign(1.0, azimuth if azimuth else rng.choice((-1, 1))))
    return SO101CameraDrift(
        azimuth_degrees=azimuth,
        elevation_degrees=float(
            rng.uniform(-max_elevation_degrees, max_elevation_degrees)
        ),
        distance_meters=float(
            rng.uniform(-max_distance_meters, max_distance_meters)
        ),
    )


def clean_partner_indices(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Map every augmented row to its clean row for the same view and state."""

    required = (
        "source_index",
        "source_row",
        "camera_index",
        "drift_variant",
        "episode",
        "task",
        "split",
    )
    if any(name not in arrays for name in required):
        raise ValueError("dataset lacks paired clean/drift identity arrays")
    length = len(arrays["source_row"])
    if any(len(arrays[name]) != length for name in required):
        raise ValueError("paired identity arrays have inconsistent lengths")
    clean_by_key: dict[tuple[int, int, int], int] = {}
    for index in range(length):
        if int(arrays["drift_variant"][index]) != 0:
            continue
        key = (
            int(arrays["source_index"][index]),
            int(arrays["source_row"][index]),
            int(arrays["camera_index"][index]),
        )
        if key in clean_by_key:
            raise ValueError(f"duplicate clean row for identity {key}")
        clean_by_key[key] = index
    partners = np.empty(length, dtype=np.int64)
    for index in range(length):
        key = (
            int(arrays["source_index"][index]),
            int(arrays["source_row"][index]),
            int(arrays["camera_index"][index]),
        )
        if key not in clean_by_key:
            raise ValueError(f"missing clean row for identity {key}")
        partner = clean_by_key[key]
        for name in ("episode", "task", "split"):
            if int(arrays[name][index]) != int(arrays[name][partner]):
                raise ValueError(f"paired rows disagree on {name}")
        partners[index] = partner
    return partners


class SO101SyntheticFCPGDataset:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        dataset_format = self.manifest.get("format")
        if dataset_format not in (DATASET_FORMAT, LEGACY_DATASET_FORMAT):
            raise ValueError(f"unsupported SO101 synthetic dataset: {self.root}")
        array_names = (
            ARRAY_NAMES
            if dataset_format == DATASET_FORMAT
            else tuple(name for name in ARRAY_NAMES if name != "drift_variant")
        )
        self.arrays = {
            name: np.load(self.root / f"{name}.npy", mmap_mode="r")
            for name in array_names
        }
        expected = int(self.manifest["samples"])
        for name, values in self.arrays.items():
            if len(values) != expected:
                raise ValueError(f"{name}.npy length disagrees with manifest")
        if self.arrays["rgb"].dtype != np.uint8:
            raise ValueError("rgb.npy must use uint8")
        if self.arrays["rgb"].shape[1:] != (
            int(self.manifest["height"]),
            int(self.manifest["width"]),
            3,
        ):
            raise ValueError("rgb.npy shape disagrees with manifest")
        if self.arrays["q_nominal"].shape[1:] != (3, 2):
            raise ValueError("q_nominal.npy must contain three 2D anchors")
        if self.arrays["q_true"].shape != self.arrays["q_nominal"].shape:
            raise ValueError("q_true.npy must match q_nominal.npy")
        if self.arrays["valid"].shape[1:] != (3,):
            raise ValueError("valid.npy must contain three anchor masks")
        if self.arrays["camera_context"].shape[1:] != (5,):
            raise ValueError("camera_context.npy must contain five geometry values")

    def indices(self, split: str) -> np.ndarray:
        codes = {"train": 0, "val": 1}
        if split not in codes:
            raise ValueError("split must be 'train' or 'val'")
        return np.flatnonzero(np.asarray(self.arrays["split"]) == codes[split])

    def __len__(self) -> int:
        return int(self.manifest["samples"])


__all__ = [
    "ARRAY_NAMES",
    "DATASET_FORMAT",
    "SO101SyntheticFCPGDataset",
    "balanced_counts",
    "clean_partner_indices",
    "sample_camera_drift",
]
