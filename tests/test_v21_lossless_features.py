import numpy as np
import pytest

from fcpg.v21_lossless_features import LosslessFeatures


@pytest.mark.parametrize('shuffle', [False, True])
def test_lossless_preserves_all_half_bits_and_noncontiguous_shape(shuffle):
    # Includes signed zero, subnormal, NaN payload and infinities; compare bits, not floats.
    original = np.arange(65536, dtype=np.uint16).view(np.float16).reshape(256, 256).T
    codec = LosslessFeatures(shuffle=shuffle)
    restored = codec.decode(codec.encode(original))
    assert restored.shape == original.shape
    assert restored.tobytes() == original.tobytes()


def test_int8_zero_and_full_range_round_trip():
    codec = LosslessFeatures()
    values = np.arange(-128, 128, dtype=np.int8).reshape(8, 32)
    assert np.array_equal(codec.decode(codec.encode(values)), values)
    zeros = np.zeros((2, 256, 384), np.float16)
    assert np.array_equal(codec.decode(codec.encode(zeros)), zeros)


def test_invalid_dtype_truncated_data_and_header_are_rejected():
    codec = LosslessFeatures()
    with pytest.raises(ValueError):
        codec.encode(np.ones(8, dtype=np.float64))
    payload = codec.encode(np.arange(1024, dtype=np.float16))
    with pytest.raises(ValueError):
        codec.decode(payload[:-1])
    with pytest.raises(ValueError):
        codec.decode(b'wrong')
