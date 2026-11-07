"""fcpg/v21_object_finder.py (training overhaul doc section 17.34)."""
import numpy as np
import torch

from fcpg.v21_object_finder import (CELL_PX, OUT, ObjectFinder, coverage_centre, geometry, letterbox_coverage, loss,
                                    patch_weights, pool, reference, snap_aspect, targets)


def test_a_quarter_size_label_lands_where_the_letterbox_puts_it():
    # a 448 x 592 labeller picture (fit of a 480 x 640 camera): 4:3, content 224 x 168 from row 28
    small = np.zeros((112, 148), bool)
    small[:, :] = True
    cover = letterbox_coverage(small, (448, 592))
    assert snap_aspect(448, 592) == 4 / 3
    rows = np.flatnonzero(cover.max(1) > .5)
    assert rows[0] == 28 and rows[-1] == 28 + 168 - 1
    assert cover.max(0).min() > .5                      # the long side fills the width


def test_a_centred_square_keeps_its_centre_and_area():
    mask = np.zeros((224, 224), bool)
    mask[100:124, 90:130] = True                      # 24 x 40 px around (110, 112)
    cover64, cover16 = targets(letterbox_coverage(mask))
    assert abs(cover64.sum() * CELL_PX ** 2 - mask.sum()) < 1.
    cx, cy = coverage_centre(cover64)
    assert abs(cx - 110.) < .5 and abs(cy - 112.) < .5
    assert cover16.shape == (16, 16) and abs(cover16.sum() * 14 ** 2 - mask.sum()) < 1.


def test_geometry_of_a_probability_map():
    prob = np.zeros((OUT, OUT))
    prob[30:34, 20:40] = .9                             # a horizontal bar
    g = geometry(prob)
    assert g['found'] and g['cells'] == 80
    assert abs(g['centre_px'][0] - 30. * CELL_PX) < 1e-6 and abs(g['centre_px'][1] - 32. * CELL_PX) < 1e-6
    assert g['length_px'] > g['width_px'] and abs(np.cos(2 * g['theta']) - 1.) < 1e-6
    assert not geometry(np.full((OUT, OUT), .4))['found']


def test_reference_is_the_mean_token_under_the_outline():
    tokens = np.zeros((2, 256, 384), np.float32)
    tokens[0, 5] = 1.
    tokens[1, 7] = 3.
    cover = np.zeros((2, 16, 16), np.float32)
    cover[0].reshape(-1)[5] = 1.
    cover[1].reshape(-1)[7] = 1.
    assert np.allclose(reference(tokens, cover), 2.)
    assert reference(tokens, np.zeros((2, 16, 16))) is None


def test_the_network_and_its_loss():
    torch.manual_seed(0)
    model = ObjectFinder()
    tokens, ref = torch.randn(3, 256, 384), torch.randn(3, 384)
    logits = model(tokens, ref)
    assert logits.shape == (3, OUT, OUT)
    target = torch.zeros(3, OUT, OUT)
    target[0, 10:20, 10:20] = 1.
    value = loss(logits, target, torch.tensor([True, False, False]))
    value.backward()
    assert torch.isfinite(value)
    plain = ObjectFinder(conditional=False)
    assert plain(tokens).shape == (3, OUT, OUT)


def test_pool_and_patch_weights():
    tokens = torch.zeros(1, 256, 384)
    tokens[0, 17] = 2.                                  # patch (1, 1)
    prob = torch.zeros(1, OUT, OUT)
    prob[0, 4:8, 4:8] = .9                              # exactly patch (1, 1) on the 64 x 64 grid
    w = patch_weights(prob)
    assert w.shape == (1, 16, 16) and float(w[0, 1, 1]) == 1. and float(w.sum()) == 1.
    assert torch.allclose(pool(tokens, w), torch.full((1, 384), 2.))
    assert torch.allclose(pool(tokens, torch.zeros(1, 16, 16)), torch.zeros(1, 384))


def test_the_presence_head():
    torch.manual_seed(0)
    model = ObjectFinder(conditional=False, presence=True)
    assert model.config == dict(width=192, blocks=3, conditional=False, presence=True)
    logit = model.presence_logit(torch.randn(4, 384), torch.randn(4, 384))
    assert logit.shape == (4,)
    old = ObjectFinder(conditional=False)
    assert 'presence' not in old.config and not hasattr(old, 'presence_net')


def test_a_dinov3_l_finder_reads_its_own_tokens_and_dinov2_configs_stay(tmp_path):
    # section 17.56: the finder records its encoder; DINOv2 finders keep the config they always had
    from fcpg.v21_object_finder import LearnedColumnFinder, load, save
    assert ObjectFinder(conditional=False).config == dict(width=192, blocks=3, conditional=False)
    torch.manual_seed(0)
    model = ObjectFinder(conditional=True, presence=False, encoder='dinov3_vitl16').eval()
    assert model.config['encoder'] == 'dinov3_vitl16' and (model.grid, model.dim) == (14, 1024)
    tokens, ref = torch.randn(2, 196, 1024), torch.randn(2, 1024)
    logits = model(tokens, ref)
    assert logits.shape == (2, OUT, OUT)
    w = patch_weights(torch.sigmoid(logits), grid=14)
    assert w.shape == (2, 14, 14) and pool(tokens, w).shape == (2, 1024)
    _, cover = targets(np.ones((224, 224), np.float32), 14)
    assert cover.shape == (14, 14) and np.allclose(cover, 1.)
    assert reference(np.ones((1, 196, 1024), np.float32), np.ones((1, 14, 14))).shape == (1024,)
    save(tmp_path / 'column.pt', model, column_reference=np.ones(1024, np.float32))
    again, _ = load(tmp_path / 'column.pt')
    with torch.no_grad():
        assert again.encoder == 'dinov3_vitl16' and torch.equal(again(tokens, ref), model(tokens, ref))
    column = LearnedColumnFinder(tmp_path / 'column.pt')
    assert column.encoder == 'dinov3_vitl16' and column.describe()['encoder'] == 'dinov3_vitl16'
    assert column.record(np.zeros((196, 1024), np.float32))['reason'] == 'no_column_tokens'
    assert column.record(np.random.default_rng(0).standard_normal((196, 1024)).astype(np.float32))['view'] == 'column'
    try:
        column.record(np.ones((256, 384), np.float32))
    except ValueError as error:
        assert 'dinov3_vitl16' in str(error)
    else:
        raise AssertionError('DINOv2 tokens must be refused by a DINOv3-L finder')


def test_patch_weights_on_the_dinov2_grid_are_the_exact_four_by_four_cells():
    import torch.nn.functional as F
    prob = torch.rand(3, OUT, OUT)
    assert torch.equal(patch_weights(prob), F.avg_pool2d((prob >= .5).float()[:, None], 4)[:, 0])
