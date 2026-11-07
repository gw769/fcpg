"""Exact completed archive prefixes for the existing V20.3.2 real labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .prod_v20_data import V20DataError
from .prod_v20_3_sources import V203Sources
from .prod_v20_3_2_history import ObserverMotionHistory


def recover_labeled_history(archive, real, records, temporal_root=None):
    """Match entry/endpoint to a recorded chain, using only its causal prefix.

    Multiple chains can contain the same endpoint. Prefer the longest available
    prefix, then the smallest row. This selection uses identity only, never
    image labels, action quality or a future state. No cross-episode stitching.
    """
    count = len(real["q_position"])
    arrays = dict(
        history_commands_position=np.zeros((count, 3, 5, 6), np.float32),
        history_readbacks_position=np.zeros((count, 3, 6, 6), np.float32),
        history_time_deltas=np.zeros((count, 3, 5), np.float32),
        history_valid=np.zeros((count, 3), bool),
        history_source_endpoint=np.full((count, 3), -1, np.int64),
    )
    identities = {}
    for record in records:
        for kind, field in enumerate(("query_offset", "target_offset")):
            key = str(record["entry_id"]), kind
            if key in identities:
                raise V20DataError("ambiguous labeled history entry/endpoint")
            identities[key] = int(record["episode_index"]), int(record[field])

    endpoint_pairs = np.rec.fromarrays(
        (archive.endpoint_episode_index, archive.endpoint_local_frame_index),
        names=("episode", "frame"))
    order = np.argsort(endpoint_pairs)
    sorted_pairs = endpoint_pairs[order]
    if len(sorted_pairs) > 1 and np.any(sorted_pairs[1:] == sorted_pairs[:-1]):
        raise V20DataError("ambiguous archive endpoint identity")
    best_row = np.full(archive.endpoint_count, -1, np.int64)
    best_horizon = np.zeros(archive.endpoint_count, np.int64)
    for horizon in range(1, 5):
        endpoint, first_row = np.unique(archive.row_endpoint_index[:, horizon], return_index=True)
        best_row[endpoint], best_horizon[endpoint] = first_row, horizon

    matched = 0
    for row in range(count):
        if not real["q_available"][row]:
            continue
        key = str(real["entry_id"][row]), int(real["endpoint_kind"][row])
        if key not in identities:
            continue
        query = np.array([identities[key]], dtype=endpoint_pairs.dtype)[0]
        address = np.searchsorted(sorted_pairs, query)
        if address == len(order) or sorted_pairs[address] != query:
            continue
        endpoint = int(order[address])
        horizon = int(best_horizon[endpoint])
        if horizon == 0:
            continue
        chain = archive.chains([int(best_row[endpoint])])
        if int(chain.endpoint_index[0, horizon]) != endpoint:
            raise V20DataError("labeled history ended at a different image")
        if not np.allclose(chain.q_position[0, horizon], real["q_position"][row], atol=.02, rtol=0):
            raise V20DataError("labeled history does not end at the recovered actual q")
        begin = max(0, horizon - 3)
        size = horizon - begin
        slot = slice(3 - size, 3)
        arrays["history_commands_position"][row, slot] = chain.commands_position[0, begin:horizon]
        arrays["history_readbacks_position"][row, slot] = chain.measured_q_paths_position[0, begin:horizon]
        arrays["history_time_deltas"][row, slot] = np.diff(chain.readback_time_offsets_s[0, begin:horizon], axis=-1)
        arrays["history_source_endpoint"][row, slot] = chain.endpoint_index[0, begin:horizon]
        arrays["history_valid"][row, slot] = True
        matched += 1
    archive_matches = matched
    if temporal_root is not None:
        _recover_from_original_grid(arrays, real, identities, Path(temporal_root))
    valid = arrays["history_valid"]
    matched = int(valid.any(1).sum())
    audit = dict(rows=count, rows_with_actual_history=matched, completed_transitions=int(valid.sum()),
                 archive_prefix_rows=archive_matches, original_grid_prefix_rows=matched - archive_matches,
                 valid_labeled_views_with_history=(real["view_valid"] & valid.any(1)[:, None]).sum(0).tolist(),
                 rows_without_matched_history=count - matched,
                 prefix_lengths=np.bincount(valid.sum(1), minlength=4).tolist(),
                 selection="same episode/frame; archive prefix or original recorded temporal grid; no future actions",
                 unavailable="explicit empty history; no fabricated commands or readbacks",
                 evidence="training-fit", unseen_claim=False)
    return arrays, audit


def _recover_from_original_grid(arrays, real, identities, root):
    """Recover labels absent from the sparse 320K endpoint bank.

    The original index records its actual sampled frame grid, states, actions
    and declared source fps. Use its preceding grid entries rather than assume
    a frame stride or invent command/readback values. Historical RGB is neither
    required nor fabricated: this feeds only the observer's motion context.
    """
    def mmap(name):
        return np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    metadata = json.loads((root / "manifest.json").read_text())
    states, commands = mmap("states"), mmap("actions")
    offsets, window_episode = mmap("episode_offsets"), mmap("window_episode")
    grids = [mmap(name) for name in ("window_image_indices", "window_future_action_indices", "window_future_indices")]
    canonical = {}
    for row in range(len(real["q_position"])):
        if not real["q_available"][row] or arrays["history_valid"][row].any():
            continue
        key = str(real["entry_id"][row]), int(real["endpoint_kind"][row])
        if key not in identities:
            continue
        episode, frame = identities[key]
        record = metadata["episodes"][episode]
        start, stop = int(offsets[episode]), int(offsets[episode + 1])
        if start != int(record["frame_start"]) or stop != int(record["frame_stop"]):
            raise V20DataError("original history episode offsets disagree")
        fps = float(record["manifest_row"]["fps_declared"])
        if not np.isfinite(fps) or fps <= 0 or not 0 <= frame < stop - start:
            raise V20DataError("invalid original history frame/fps")
        if episode not in canonical:
            rows = np.flatnonzero(window_episode == episode)
            canonical[episode] = np.unique(np.concatenate([np.asarray(grid[rows]).reshape(-1) for grid in grids]))
        grid = canonical[episode]
        current = np.searchsorted(grid, frame)
        if current == len(grid) or grid[current] != frame:
            continue
        chunks = min(3, int(current) // 5)
        if chunks == 0:
            continue
        previous = grid[current - chunks * 5:current + 1].astype(np.int64)
        if previous[0] < 0 or previous[-1] >= stop - start:
            raise V20DataError("original history crosses an episode boundary")
        actual = np.asarray(states[start + previous], np.float32)
        sent = np.asarray(commands[start + previous[:-1]], np.float32)
        if not np.isfinite(actual).all() or not np.isfinite(sent).all():
            raise V20DataError("nonfinite original factual history")
        if not np.allclose(actual[-1], real["q_position"][row], atol=.02, rtol=0):
            raise V20DataError("original history does not end at recovered actual q")
        for index in range(chunks):
            slot = 3 - chunks + index
            cursor = index * 5
            arrays["history_commands_position"][row, slot] = sent[cursor:cursor + 5]
            arrays["history_readbacks_position"][row, slot] = actual[cursor:cursor + 6]
            arrays["history_time_deltas"][row, slot] = np.diff(previous[cursor:cursor + 6]) / fps
            arrays["history_valid"][row, slot] = True


class V2032Sources(V203Sources):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("seed", 20000)
        super().__init__(*args, **kwargs)
        records = json.loads(Path(self.paths.real_recovery_panel).read_text())["records"]
        arrays, self.real_history_audit = recover_labeled_history(self.archive, self.real, records, self.paths.temporal_root)
        self.real.update(arrays)

    def _attach_history(self, batch):
        names = ("history_commands_position", "history_readbacks_position", "history_time_deltas", "history_valid")
        batch["actual_history"] = ObserverMotionHistory(*(torch.as_tensor(batch[name], device=self.device) for name in names))
        return batch

    def real_batch(self, *args, **kwargs):
        batch = self._attach_history(super().real_batch(*args, **kwargs))
        ids = args[0] if args else kwargs.get("indices")
        if ids is not None:
            # Exercise H0/H1/H2/H3 on the SAME labeled current endpoint. Keep
            # the most recent factual suffix; never synthesize a transition.
            history = batch["actual_history"]
            # View selection uses ordinal % 2. Pair adjacent ordinals before
            # choosing length, so EACH view receives all four lengths.
            lengths = torch.as_tensor((np.asarray(ids) // 2) % 4, device=self.device)
            valid = history.valid & (torch.arange(3, device=self.device)[None] >= 3 - lengths[:, None])
            def masked(value):
                return torch.where(valid.reshape(*valid.shape, *([1] * (value.ndim - 2))), value, torch.zeros_like(value))
            batch["actual_history"] = ObserverMotionHistory(masked(history.commands_position), masked(history.readbacks_position),
                masked(history.time_deltas), valid)
        return batch

    def real_evaluation_batch(self, *args, **kwargs):
        return self._attach_history(super().real_evaluation_batch(*args, **kwargs))

    def preflight(self, **kwargs):
        report = super().preflight(**kwargs)
        report["observer_history_refit"] = self.real_history_audit
        if not self.real_history_audit["valid_labeled_views_with_history"][1]:
            raise V20DataError("V20.3.2 Column history refit has no matched labeled completed trajectories")
        return report
