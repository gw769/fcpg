"""fcpg/v21_text_finder.py (James 10/28: the task text names the thing; DINOv3's text alignment, no SAM): the object
phrase keeps its colour, Meta's file splits into the vision head and the text tower, a patch's score is the phrase's
share of the softmax over the phrase and the contrast names (10/31, Meta's zero-shot segmentation recipe), the names'
embeddings are cached under the tower's fingerprint, the reference is the mean served token of the best-scoring patches,
the wrist and column finders run on that reference instead of the fish one, and keeping the encoder's prefix tokens
leaves the model's tokens bit for bit."""
from pathlib import Path

import numpy as np
import pytest
import torch

from fcpg import v21_text_finder as tf

L_WEIGHTS = Path('/home/james/so101-c4-step2/models/dinov3/vit_large_patch16_dinov3.lvd1689m/model.safetensors')
needs_repo = pytest.mark.skipif(not (tf.DEFAULT_REPO / 'dinov3').is_dir() or not tf.DEFAULT_BPE.is_file(),
                                reason='no DINOv3 code or BPE vocabulary on this machine')


def test_the_object_phrase_keeps_its_colour():
    assert tf.object_phrase('Pick up the red fish toy and place it in the cardboard box') == 'red fish toy'
    assert tf.object_phrase('Pick up the blue tube and place it in the cardboard box') == 'blue tube'
    with pytest.raises(ValueError):
        tf.object_phrase('pick up and place it')


def test_the_reference_is_the_mean_served_token_of_the_best_patches():
    served = np.random.default_rng(0).standard_normal((196, 1024)).astype(np.float32)
    cosines = np.linspace(-.1, .1, 196)
    cosines[[7, 40, 99, 150]] = [.5, .45, .4, .35]
    reference, record = tf.text_reference(served, cosines, top=4)
    assert record['top'] == [7, 40, 99, 150] and record['peak'] == .5
    assert np.allclose(reference, served[[7, 40, 99, 150]].mean(0)) and reference.dtype == np.float32


def test_meta_s_file_splits_into_the_head_and_the_text_tower():
    head, text = tf.split_state({'visual_model.head.blocks.0.x': 1, 'text_model.backbone.y': 2, 'logit_scale': 3})
    assert head == {'blocks.0.x': 1} and text == {'backbone.y': 2}
    with pytest.raises(ValueError, match='unexpected'):
        tf.split_state({'visual_model.head.a': 1, 'text_model.b': 2, 'visual_model.backbone.c': 3})
    with pytest.raises(ValueError, match='lacks'):
        tf.split_state({'visual_model.head.a': 1})


def _small_parts():
    """The real vision head and tokenizer with a small text tower of the same classes (the real one is 0.5 B)."""
    tf._import_dinov3(tf.DEFAULT_REPO)
    from dinov3.eval.text.text_tower import build_text_model
    from dinov3.eval.text.text_transformer import TextTransformer
    from dinov3.eval.text.tokenizer import get_tokenizer
    from dinov3.eval.text.vision_tower import VisionHead
    torch.manual_seed(0)
    head = VisionHead(tf.DIM, tf.EMBED, tf.HEADS, tf.HEAD_BLOCKS, .3, True, True, False)
    head.init_weights()
    text = build_text_model(tf.EMBED, None, False, 0, False, 0., 'argmax', True, backbone=TextTransformer(
        context_length=77, vocab_size=49408, dim=64, num_heads=4, num_layers=2, ffn_ratio=4, is_causal=True,
        ls_init_value=None, dropout_prob=0.))
    text.init_weights()
    return head.eval(), text.eval(), get_tokenizer(str(tf.DEFAULT_BPE))


def test_a_patch_wins_by_beating_the_other_names_not_by_its_raw_cosine():
    """Meta's per-pixel softmax over the names: a patch close to every name (background) loses to a patch that is the
    phrase rather than the other names, although its raw cosine to the phrase is higher (the 10/28 scorer's failure)."""
    shared, phrase_own, other_own = torch.eye(4)[0], torch.eye(4)[1], torch.eye(4)[2:]
    texts = torch.nn.functional.normalize(torch.stack([shared + phrase_own, shared + other_own[0],
                                                       shared + other_own[1]]), dim=-1)
    background = torch.nn.functional.normalize(shared, dim=0)
    thing = torch.nn.functional.normalize(phrase_own - .5 * shared, dim=0)
    features = torch.stack([background, thing])
    assert float(features[0] @ texts[0]) > float(features[1] @ texts[0])
    scores = tf.class_scores(features, texts)
    assert float(scores[1]) > float(scores[0]) and abs(float(scores[0]) - 1 / 3) < 1e-6
    assert tf.contrast_names('Box ', ('wall', 'box', 'cardboard box')) == ['wall', 'cardboard box']
    assert len(tf.ADE20K) == 150 and len(tf.TEMPLATES) == 80 and all('{0}' in t for t in tf.TEMPLATES)


@needs_repo
def test_the_names_are_embedded_once_and_cached_under_the_tower_fingerprint(tmp_path, monkeypatch):
    _, text, tokenizer = _small_parts()
    first = tf.name_embeddings(text, tokenizer, ['table', 'wall'], cache=tmp_path)
    assert first.shape == (2, tf.HALF) and len(list(tmp_path.glob('dinotxt-names-*.npy'))) == 1
    assert torch.allclose(first.norm(dim=-1), torch.ones(2), atol=1e-5)
    monkeypatch.setattr(tf, 'phrase_embedding', lambda *a, **k: (_ for _ in ()).throw(AssertionError('recomputed')))
    assert torch.equal(tf.name_embeddings(text, tokenizer, ['table', 'wall'], cache=tmp_path), first)
    monkeypatch.undo()
    with torch.no_grad():
        next(text.parameters()).add_(1.)                       # other weights: another key, not the cached file
    tf.name_embeddings(text, tokenizer, ['table', 'wall'], cache=tmp_path)
    assert len(list(tmp_path.glob('dinotxt-names-*.npy'))) == 2


@needs_repo
def test_the_scorer_gives_every_patch_a_score_and_the_picture_a_reference(tmp_path):
    scorer = tf.TextScorer('blue tube', parts=_small_parts(), contrast=('table', 'wall', 'robot', 'Blue tube'),
                           cache=tmp_path)
    assert scorer.text.shape == (tf.HALF,) and abs(float(scorer.text.norm()) - 1.) < 1e-5
    assert scorer.contrast_names == ['table', 'wall', 'robot'] and scorer.texts.shape == (4, tf.HALF)
    rng = np.random.default_rng(1)
    prefix = rng.standard_normal((5, 1024)).astype(np.float32)
    served = rng.standard_normal((196, 1024)).astype(np.float32)
    cosines, scores = scorer.cosines(prefix, served), scorer.scores(prefix, served)
    assert cosines.shape == scores.shape == (196,) and np.all(np.abs(cosines) <= 1 + 1e-5)
    assert np.all((scores > 0) & (scores < 1))
    reference, record = scorer.reference(prefix, served)
    assert reference.shape == (1024,) and record['phrase'] == 'blue tube' and len(record['top']) == tf.TOP
    assert record['top'] == [int(i) for i in np.argsort(-scores, kind='stable')[:tf.TOP]]
    assert np.allclose(reference, served[record['top']].mean(0)) and np.isclose(record['peak'], scores.max())
    assert np.isclose(record['peak_cosine'], cosines[record['top'][0]])
    with pytest.raises(ValueError):
        scorer.cosines(prefix[:4], served)
    with pytest.raises(ValueError, match='no tokens'):
        scorer.reference(np.zeros_like(prefix), served)
    described = scorer.describe()
    assert described['phrase'] == 'blue tube' and described['sam_used'] is False and described['contrast'] == 3


class _Scorer:
    phrase = 'blue tube'

    def __init__(self, value):
        self.value, self.seen = float(value), []

    def reference(self, prefix, served):
        self.seen.append(np.asarray(prefix).shape)
        return np.full(1024, self.value, np.float32), dict(peak=.4, top=[1], phrase=self.phrase)

    def describe(self):
        return dict(finder='dinotxt', phrase=self.phrase)


def test_the_wrist_finder_runs_on_the_text_reference_not_the_fish_one():
    from fcpg.v21_object_finder import ObjectFinder
    torch.manual_seed(0)
    model = ObjectFinder(conditional=True, encoder='dinov3_vitl16').eval()
    references = []
    forward = model.forward

    def spy(tokens, reference=None):
        references.append(reference.clone())
        return forward(tokens, reference)
    model.forward = spy
    served = np.random.default_rng(2).standard_normal((196, 1024)).astype(np.float32)
    encoder = tf.TextWristEncoder.__new__(tf.TextWristEncoder)     # the projection fit is not under test here
    scorer = _Scorer(3.)
    encoder.model, encoder.device, encoder.timings, encoder.reference = model, torch.device('cpu'), [], None
    encoder.tokens_of, encoder.prefix_of, encoder.scorer = (lambda: served), (lambda: np.ones((5, 1024))), scorer
    record, info = encoder.find_record(None)
    assert scorer.seen == [(5, 1024)] and torch.equal(references[0], torch.full((1, 1024), 3.))
    assert info['text']['phrase'] == 'blue tube' and encoder.last_text['peak'] == .4
    assert record is None or record['text_phrase'] == 'blue tube'


def test_the_wrist_reference_comes_from_the_column_picture_of_the_exposure():
    # 11/03: made from the wrist picture itself the reference found a fish in nearly every picture without one
    from fcpg.v21_object_finder import ObjectFinder
    torch.manual_seed(0)
    model = ObjectFinder(conditional=True, encoder='dinov3_vitl16').eval()
    rng = np.random.default_rng(2)
    wrist, column = (rng.standard_normal((196, 1024)).astype(np.float32) for _ in range(2))
    seen = []

    class Scorer(_Scorer):
        def reference(self, prefix, served):
            seen.append((float(np.asarray(prefix)[0, 0]), served is column or np.array_equal(served, column)))
            return super().reference(prefix, served)
    encoder = tf.TextWristEncoder.__new__(tf.TextWristEncoder)
    encoder.model, encoder.device, encoder.timings, encoder.reference = model, torch.device('cpu'), [], None
    encoder.tokens_of, encoder.prefix_of, encoder.scorer = (lambda: wrist), (lambda: np.ones((5, 1024))), Scorer(3.)
    encoder.reference_of = lambda: (np.full((5, 1024), 2.), column)
    record, info = encoder.find_record(None)
    assert seen == [(2., True)] and info['text']['reference_view'] == 'column'
    # no column picture this exposure: the wrist picture's own
    encoder.reference_of = lambda: (np.zeros((5, 1024)), np.zeros((196, 1024), np.float32))
    record, info = encoder.find_record(None)
    assert seen[-1] == (1., False) and info['text']['reference_view'] == 'wrist'


def test_the_column_finder_outlines_with_the_text_reference(tmp_path):
    from fcpg.v21_object_finder import ObjectFinder, save
    torch.manual_seed(0)
    save(tmp_path / 'column.pt', ObjectFinder(conditional=True, encoder='dinov3_vitl16').eval(),
         column_reference=np.ones(1024, np.float32))
    scorer = _Scorer(2.)
    column = tf.TextColumnFinder(tmp_path / 'column.pt', scorer=scorer, prefix_of=lambda: np.ones((5, 1024)))
    assert column.reference is None
    assert column.record(np.zeros((196, 1024), np.float32))['reason'] == 'no_column_tokens' and not scorer.seen
    out = column.record(np.random.default_rng(0).standard_normal((196, 1024)).astype(np.float32))
    assert scorer.seen == [(5, 1024)] and out['view'] == 'column' and out['text']['phrase'] == 'blue tube'
    assert torch.equal(column.reference, torch.full((1, 1024), 2.))
    assert column.describe()['finder'] == 'learned_column+dinotxt'


@pytest.mark.skipif(not L_WEIGHTS.is_file(), reason='no DINOv3-L weights on this machine')
def test_the_prefix_tokens_leave_the_model_tokens_bit_for_bit():
    from fcpg.v21_live_encoder import dinov3_int8_with_prefix
    from scripts.build_v21_dinov3_features import dinov3_int8, load_dinov3
    model = load_dinov3(L_WEIGHTS, 'cpu')
    pictures = np.random.default_rng(3).integers(0, 255, (2, 224, 224, 3), dtype=np.uint8)
    tokens, scales = dinov3_int8(model, pictures, 'cpu', 2)
    again, again_scales, prefix = dinov3_int8_with_prefix(model, pictures, 'cpu', 2)
    assert np.array_equal(tokens, again) and np.array_equal(scales, again_scales)
    assert prefix.shape == (2, 5, 1024) and prefix.dtype == np.float16 and np.isfinite(prefix).all()
