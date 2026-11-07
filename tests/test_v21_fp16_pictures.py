"""17.173: the LeWM trainer's pictures stay float16 as stored (ScratchDinoFeatureBank.load and the window pool's
features, as_stored=True): the memory's shift without its extra copy gives _shifted_in's values, a float16 memory holds
the float32 memory's values, and the full-patch LeWM forecasts the same from float16 pictures as from their float32
copies, on the GPU under bf16 autocast too (where feature_in now reads them in bfloat16 at once; width 256 makes
feature_in a Linear with either encoder, as DINOv3-L's 1024 -> 384 is)."""
from dataclasses import replace

import pytest
import torch

from fcpg.v21_encoder import DIM, PATCHES
from fcpg.v21_memory import VisualMemory, VisualObservation, _shift, _shifted_in
from v21_lewm_helpers import setup


def test_the_shift_without_the_copy_gives_the_same_values():
    torch.manual_seed(0)
    for shape in ((4, 2, 3, PATCHES, 8), (4, 2, 3, 6), (4, 2, 3)):
        old = torch.randn(shape).half()
        new = torch.randn(shape[:2] + shape[3:]).half()
        select = (torch.rand(4, 2) < .5).reshape(4, 2, *([1] * (len(shape) - 2)))
        got = _shift(old, new, select)
        assert got.dtype == old.dtype and torch.equal(got, _shifted_in.plain(old, new, select))
        assert got.data_ptr() not in (old.data_ptr(), new.data_ptr())


def test_a_float16_memory_holds_the_float32_memorys_values():
    torch.manual_seed(1)
    b = 3
    half, full = VisualMemory.empty(b, dtype=torch.float16), VisualMemory.empty(b)
    for i in range(4):          # one more exposure than the memory holds
        picture = torch.randn(b, 2, PATCHES, DIM).half()
        q, mask = torch.randn(b, 2, 6), torch.rand(b, 2) < .7
        times = torch.full((b, 2), float(i), dtype=torch.float64)
        now = torch.full((b,), float(i), dtype=torch.float64)
        half, inserted = half.write(VisualObservation(picture, q, times, mask), now)
        full, also = full.write(VisualObservation(picture.float(), q, times, mask), now)
        assert torch.equal(inserted, also)
    assert half.features.dtype == torch.float16 and torch.equal(half.features.float(), full.features)
    for name in ('q_at_image', 'image_times', 'valid'):
        assert torch.equal(getattr(half, name), getattr(full, name))


def _on(value, device):
    """A record of tensors (memory, history, command trace) on the device."""
    return type(value)(**{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in vars(value).items()})


@pytest.mark.parametrize('cuda', [False, pytest.param(True, marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='bf16 autocast on the GPU'))])
def test_the_lewm_forecasts_the_same_from_float16_pictures(cuda):
    core, state, _, targets, _ = setup(width=256)
    device = 'cuda' if cuda else 'cpu'
    lewm = core.lewm.to(device)
    assert isinstance(lewm.feature_in, torch.nn.Linear)
    memory = _on(replace(state.memory, features=state.memory.features.half().float()), device)  # what float16 holds
    trace, history = _on(targets.trace, device), _on(state.history, device)

    def forecast(patches):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=cuda):
            return lewm(state.q.to(device), trace, history, targets.readback_offsets.to(device), patches=patches,
                        patch_valid=memory.valid, memory=memory, now_time=state.time.to(device)).predicted_q_path

    plain = forecast(memory.features)
    assert torch.equal(forecast(memory.features.half()), plain) and torch.isfinite(plain).all()
