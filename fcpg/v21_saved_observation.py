"""Recover a saved REAL view with its original q/clock; never recover forecasts."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .v21_memory import VisualObservation, clock


@dataclass(frozen=True)
class SavedView:
    observation: VisualObservation
    source_q: torch.Tensor
    source_time: torch.Tensor
    rgb: np.ndarray
    age_s: float
    source_path: str


def load_saved_column(path, *, now, device='cpu'):
    """Read the V21 service's actual-observation NPZ, not a model checkpoint.

    The current process must use the same host monotonic-clock epoch. A future
    timestamp is rejected; an old timestamp is retained, not silently clamped.
    Caller must not infer an unrecorded command history from this one image.
    """
    clock('current probe time', now, (1,))
    required = {'rgb':(2,224,224,3), 'dino_tokens':(1,2,256,384), 'q_position':(1,6),
                'q_at_image':(1,2,6), 'image_times_s':(1,2), 'q_time_s':(1,)}
    with np.load(Path(path), allow_pickle=False) as archive:
        if 'measurement_available' in archive:
            availability = archive['measurement_available']
            if availability.shape != (1,2) or availability.dtype != np.bool_ or not availability[0,1]:
                raise ValueError('this endpoint did not contain a real Column observation')
        data = {}
        for name, shape in required.items():
            if name not in archive or archive[name].shape != shape:
                raise ValueError(f'saved actual observation lacks aligned {name}')
            value = np.array(archive[name], copy=True)
            if not np.isfinite(value).all(): raise ValueError(f'nonfinite saved {name}')
            data[name] = value
    if data['rgb'].dtype != np.uint8:
        raise ValueError('saved RGB must remain uint8')
    if any(data[key].dtype != np.float64 for key in ('image_times_s','q_time_s')):
        raise ValueError('saved actual clocks must be float64')
    image_time = float(data['image_times_s'][0,1]); source_time = float(data['q_time_s'][0])
    if not 0 <= image_time <= source_time <= now.item():
        raise ValueError('saved view is from the future or a different clock epoch')
    valid = torch.tensor([[False,True]],device=device)
    tensor = lambda key: torch.as_tensor(data[key],device=device)
    observation = VisualObservation(tensor('dino_tokens'), tensor('q_at_image'), tensor('image_times_s'), valid)
    return SavedView(observation,tensor('q_position'),tensor('q_time_s'),data['rgb'][1],
                     now.item()-image_time,str(Path(path).resolve()))
