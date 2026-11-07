"""fcpg/v21_live_service.exposure_tokens (training overhaul doc section 17.56): a learned finder gets its own encoder's
tokens; the default (DINOv2) process cannot serve a DINOv3-L finder."""
from types import SimpleNamespace

import pytest

from fcpg.v21_encoder import ENCODER, encoder_of
from fcpg.v21_live_service import exposure_tokens


def test_each_finder_reads_its_own_encoders_tokens():
    owner = SimpleNamespace(last_dino_tokens='dinov2 tokens', last_model_tokens='model tokens')
    assert exposure_tokens(owner, 'dinov2_vits14') == 'dinov2 tokens'
    if ENCODER == 'dinov2_vits14':
        with pytest.raises(ValueError, match='V21_ENCODER=dinov3_vitl16'):
            exposure_tokens(owner, 'dinov3_vitl16')
    else:
        assert exposure_tokens(owner, ENCODER) == 'model tokens'


def test_the_encoder_of_a_token_shape():
    assert encoder_of(256, 384) == 'dinov2_vits14' and encoder_of(196, 1024) == 'dinov3_vitl16'
    with pytest.raises(ValueError):
        encoder_of(196, 384)
