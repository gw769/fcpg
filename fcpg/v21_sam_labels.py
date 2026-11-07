"""SAM 3.1 labels of the Archive pictures for the LeWM (11/01, section 17.163; James 「訓練的時候用sam3分割的，為什麼沒有加
attention的懲罰呢」「254萬張圖片 每張圖片對應的關節角度都別搞錯了，要輸入模型的」).

The table (scripts/build_v21_sam_label_table.py) is laid out like the DINOv3-L cache it labels: per Archive partition
(temporal-train, temporal-dev), in the cache's row order and its view order, coverage [rows, 2 views, 2 kinds (object,
robot), PATCHES] uint8 (0-255: how much of each 16 x 16 patch of the 224 letterbox the grasped object / the robot covers)
and labelled [rows, 2]. It is opened against the trainer's own caches: each partition's endpoint_index must be the
cache's bit for bit, and a label only where the cache has a valid picture. So a label follows exactly the cache row the
trainer reads the picture's tokens and capture q from (fcpg/v21_clock_cache.py), and the camera layout moves it like
them (fcpg/v21_archive_camera_layout.py remap).
The LeWM's picture bank is the memory's last three exposures per camera (fcpg/v21_memory.py); bank_labels() finds each
bank picture's chain picture by its capture time (the memory copies it from the chain, so equality is exact) and refuses
a bank picture that matches none. attention_weights() gives the supervised heads their targets per key: the first half
of them the object's coverage, the other half the robot's (one kind for every head would let them all settle on the
arm, by far the larger area); attention_penalty() asks each such head for at least TARGET of its attention there: the
mean over layers, heads with a target and queries of max(0, log TARGET - log(EPS + mass)), so a head that already
looks there is left alone and the joints' forecast keeps the rest; future_label_loss() asks the label head for the
coverage of the chain's later pictures at the read-back nearest to their time.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .v21_encoder import PATCHES

SCHEMA = 'v21-sam31-labels-v1'
KINDS = ('object', 'robot')
EPS = 1e-4
TARGET = .5


class SamLabelTable:
    def __init__(self, root, caches):
        root = Path(root)
        self.root = root
        self.manifest = json.loads((root / 'manifest.json').read_text())
        if self.manifest.get('schema') != SCHEMA or self.manifest.get('complete') is not True:
            raise ValueError('incomplete or unknown SAM label table')
        self.parts = []
        for cache in caches:
            name = Path(cache.root).name
            endpoint = np.load(root / name / 'endpoint_index.npy', mmap_mode='r')
            if not np.array_equal(endpoint, np.asarray(cache.endpoint_index)):
                raise ValueError(f'SAM label rows are not the DINO cache rows: {name}')
            coverage = np.load(root / name / 'coverage.npy', mmap_mode='r')
            labelled = np.load(root / name / 'labelled.npy', mmap_mode='r')
            if (coverage.shape != (len(endpoint), 2, len(KINDS), PATCHES) or coverage.dtype != np.uint8
                    or labelled.shape != (len(endpoint), 2) or labelled.dtype != np.bool_):
                raise ValueError(f'SAM label table shapes differ: {name}')
            if (np.asarray(labelled) & ~np.asarray(cache.view_valid, bool)).any():
                raise ValueError(f'SAM labels on pictures the cache has no valid picture for: {name}')
            self.parts.append((name, np.asarray(endpoint), coverage, labelled))

    def get(self, ids):
        """Endpoint ids [...] -> coverage [..., 2, 2, PATCHES] uint8 and labelled [..., 2] bool, cache view order."""
        ids = np.asarray(ids, np.int64)
        flat = ids.reshape(-1)
        coverage = np.zeros((len(flat), 2, len(KINDS), PATCHES), np.uint8)
        labelled = np.zeros((len(flat), 2), bool)
        done = np.zeros(len(flat), bool)
        for _, endpoint, part_coverage, part_labelled in self.parts:
            index = np.searchsorted(endpoint, flat)
            match = (index < len(endpoint)) & (endpoint[np.minimum(index, len(endpoint) - 1)] == flat) & ~done
            if match.any():
                rows = index[match]
                order = np.argsort(rows, kind='stable')          # memory-mapped rows read in ascending order
                where = np.flatnonzero(match)[order]
                coverage[where] = part_coverage[rows[order]]
                labelled[where] = part_labelled[rows[order]]
                done |= match
        if not done.all():
            raise ValueError(f'endpoints outside the SAM label table: {flat[~done][:10].tolist()}')
        return coverage.reshape(*ids.shape, 2, len(KINDS), PATCHES), labelled.reshape(*ids.shape, 2)


def chain_targets(source, table, rows, chain, memory, h, heads):
    """The SAM 3.1 labels of prepared Archive chain rows at context h (the trainer's chain and the LeWM's memory, on
    one device): the chain's endpoints are source.archive.row_endpoint_index[rows], checked by their recorded capture
    times (they must be the chain's wherever it has a valid picture), their labels moved by the camera layout like
    the tokens and capture q, never on a picture the trainer has no valid picture for; then the bank's labels by
    capture time. -> dict(weights [B, heads, keys], rows [B, heads] (attention_weights), coverage [B, J, 2, 2, PATCHES]
    and labelled [B, J, 2] of every chain picture, endpoint_times [B, J], h, bank_pictures, labelled_bank_pictures)."""
    ids = np.asarray(source.archive.row_endpoint_index[rows], np.int64)
    _, times, _ = source.recorded_clock.observations(ids, 'archive')
    valid = chain.view_valid.cpu().numpy()
    if not np.array_equal(np.where(valid, times, 0.), np.where(valid, chain.capture_times.cpu().numpy(), 0.)):
        raise ValueError("the label endpoints are not the chain's (capture times differ)")
    coverage, labelled = table.get(ids)
    roles = source.archive_camera_roles
    coverage, labelled = roles.remap(coverage, ids), roles.remap(labelled, ids)
    if (labelled & ~valid).any():
        raise ValueError('a SAM label on a picture the trainer has no valid picture for')
    device = memory.image_times.device
    coverage = torch.from_numpy(np.ascontiguousarray(coverage)).to(device)
    labelled = torch.from_numpy(np.ascontiguousarray(labelled)).to(device)
    bank, kept = bank_labels(memory.image_times, memory.valid, chain.capture_times[:, :h + 1],
                             chain.view_valid[:, :h + 1], coverage[:, :h + 1], labelled[:, :h + 1])
    weights, has = attention_weights(bank, kept, heads)
    return dict(weights=weights, rows=has, coverage=coverage, labelled=labelled, endpoint_times=chain.endpoint_times,
                h=h, bank_pictures=int(memory.valid.sum()), labelled_bank_pictures=int(kept.sum()))


def bank_labels(image_times, bank_valid, capture_times, view_valid, coverage, labelled):
    """The labels of the LeWM's picture bank. image_times [B,2,3] float64 and bank_valid [B,2,3] (the memory's);
    capture_times [B,J,2], view_valid [B,J,2], coverage [B,J,2,2,PATCHES], labelled [B,J,2]: the chain's first J
    pictures in the trainer's camera order. -> coverage [B,2,3,2,PATCHES] and labelled [B,2,3]: slot (view v, s) holds
    the chain picture of view v whose capture time is the slot's."""
    same = ((capture_times.transpose(1, 2)[:, :, None, :] == image_times[..., None])
            & view_valid.transpose(1, 2)[:, :, None, :])                                  # [B, 2, 3, J]
    found = same.any(-1)
    if (bank_valid & ~found).any():
        raise ValueError('a picture of the LeWM bank matches no chain picture by capture time')
    pick = same.float().argmax(-1)                                                          # [B, 2, 3], the first
    by_view = coverage.transpose(1, 2)                                                      # [B, 2, J, 2, P]
    gathered = by_view.gather(2, pick[..., None, None].expand(-1, -1, -1, *by_view.shape[3:]))
    kept = labelled.transpose(1, 2).gather(2, pick) & bank_valid & found
    return torch.where(kept[..., None, None], gathered, torch.zeros_like(gathered)), kept


def attention_weights(coverage, labelled, heads):
    """Bank coverage [B,2,3,2,PATCHES] uint8 and labelled [B,2,3] -> per supervised head its target per key
    [B, heads, 6 PATCHES] in 0-1 (heads[:heads // 2] the object's coverage, the others the robot's; the bank's key
    order) and which (row, head) has any [B, heads]."""
    per_kind = torch.where(labelled[..., None, None], coverage.float() / 255., torch.zeros((), device=coverage.device))
    per_kind = per_kind.permute(0, 3, 1, 2, 4).reshape(len(coverage), len(KINDS), -1)      # [B, kind, 6 P]
    kind = torch.tensor([0] * (heads // 2) + [1] * (heads - heads // 2), device=coverage.device)
    weights = per_kind[:, kind]
    return weights, weights.amax(-1) > 0


def attention_penalty(mass, rows, target=TARGET):
    """mass [layers, B, heads, queries] (ARMOutput.attention_mass), rows [B, heads] (attention_weights) -> the mean of
    max(0, log target - log(EPS + mass)) over layers, the (row, head) pairs with a target and queries (0 when there
    is none), and the mean mass of those pairs (None when none)."""
    if not bool(rows.any()):
        return mass.new_zeros(()), None
    selected = mass[:, rows]
    return (np.log(target) - (selected + EPS).log()).clamp_min(0).mean(), float(selected.detach().mean())


def future_label_loss(label_head, hidden, readback_offsets, endpoint_times, h, coverage, labelled, tolerance=.05):
    """The label head's logits at the read-back nearest to each later chain picture j > h (within `tolerance` s of
    its endpoint time) against that picture's coverage / 255 (binary cross-entropy), on the cameras with a label.
    hidden [B, Q, width] (ARMOutput.hidden), readback_offsets [B, Q] s after now, endpoint_times [B, J] float64,
    coverage [B, J, 2, 2, PATCHES] uint8, labelled [B, J, 2] -> (mean loss, labelled pictures used)."""
    b = len(hidden)
    total, count, pictures = hidden.new_zeros((), dtype=torch.float32), 0, 0
    for j in range(h + 1, coverage.shape[1]):
        after = (endpoint_times[:, j] - endpoint_times[:, h]).to(readback_offsets)
        gap = (readback_offsets - after[:, None]).abs()
        nearest = gap.argmin(1)
        use = labelled[:, j] & (gap.gather(1, nearest[:, None])[:, 0] <= tolerance)[:, None]      # [B, 2]
        if not bool(use.any()):
            continue
        logits = label_head(hidden[torch.arange(b, device=hidden.device), nearest]).float()
        logits = logits.view(b, 2, len(KINDS), PATCHES)
        loss = F.binary_cross_entropy_with_logits(logits, coverage[:, j].float() / 255., reduction='none')
        mask = use[..., None, None].expand_as(loss)
        total = total + loss[mask].sum()
        count += int(mask.sum())
        pictures += int(use.sum())
    return (total / count if count else total), pictures
