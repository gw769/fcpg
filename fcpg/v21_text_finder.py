"""The task text names what to grasp (James 10/28: 「改文字，然后dino3去识别这个物体」, 「改代码 不用sam3. 任务指令要改阿」).

DINOv3's own text alignment (dino.txt: Meta's dinov3_vitl16_dinotxt_tet1280d20h24l, a text tower and a two-block vision
head trained on the frozen DINOv3-L/16 this system already runs; DINOv3 paper, arXiv 2508.10104) decides, for every 16-px
patch of the live letterbox224 picture, how much it is the task's object phrase. No SAM, no colour rule, nothing
trained by us:
- object_phrase(): the task text's first run of words that are neither function nor action words, colour words kept
  ("Pick up the blue tube and place it in the cardboard box" -> "blue tube"); SAM 3 was never given colours
  (fcpg/v21_sam3_finder.object_prompts), dino.txt reads them;
- each name's embedding is the patch half of the text features, normalised, averaged over the 80 TEMPLATES and
  normalised again, as Meta's zero-shot segmentation notebook makes its class embeddings;
- the vision head reads the last-layer tokens of the very forward pass the model's tokens come from, its five prefix
  tokens (class + 4 registers) included (fcpg/v21_live_encoder.ModelTokens.last_prefix), as VisionTower does;
- a patch's score is the phrase's share of a softmax over the phrase and the 150 ADE20K names (fcpg/v21_text_vocab.py)
  on that patch, as the notebook decides each pixel: the phrase only has to beat the other names there. The raw cosine
  to the phrase alone (the 10/28 scorer) ranked table and background patches first: training overhaul doc 17.149,
  "blue tube" 0 of 15 pictures, then 12 of 15 with this recipe on the same pictures (exploratory; a fresh check gates
  the robot). The names' embeddings are made once and cached in DEFAULT_ROOT under the text tower's fingerprint;
- TextScorer.reference(): the mean served token of the TOP best-scoring patches is the reference vector the learned
  conditional finders (training overhaul doc sections 17.34 and 17.51) take in place of the fish reference their model
  files carry, so they outline the named thing with their own masks (TextWristEncoder, TextColumnFinder); the scores
  and the phrase's cosines are recorded with every find for the offline check (analysis/v21/text_finder_robot_check.py).
The weights come from Meta's DINOv3 release (James 10/31: the public Hugging Face copy PIA-SPACE-LAB/
dinov3_vitl16_dinotxt_vision_head_and_text_encoder, SHA-256 equal to Meta's file names; the backbone file equals the
system's DINOv3-L, robot-20261030/verify_dinotxt_b_20261031.py), the BPE vocabulary from dl.fbaipublicfiles.com, the
code from github.com/facebookresearch/dinov3; all three live in DEFAULT_ROOT on .102.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .v21_object_finder import LearnedColumnFinder, LearnedObjectRelativeEncoder
from .v21_text_vocab import ADE20K, TEMPLATES

DEFAULT_ROOT = Path('/home/james/so101-c4-step2/models/dinov3-text')
DEFAULT_REPO = DEFAULT_ROOT / 'dinov3-repo'
DEFAULT_BPE = DEFAULT_ROOT / 'bpe_simple_vocab_16e6.txt.gz'
DEFAULT_WEIGHTS = DEFAULT_ROOT / 'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth'
DIM, HEADS, PREFIX = 1024, 16, 5            # DINOv3-L/16: width, heads, class + 4 register tokens
EMBED = 2048                                # dino.txt: class half + patch half
HALF = EMBED // 2
HEAD_BLOCKS = 2
TOP = 4                                     # patches whose mean served token is the reference
SOURCE = 'dinotxt_task_phrase'
SCORING = 'phrase share of a softmax over the phrase and the ADE20K names per patch (Meta dinotxt segmentation notebook)'


def object_phrase(task):
    """The object phrase of a task text, colour words kept ("Pick up the red fish toy and place it in the cardboard
    box" -> "red fish toy")."""
    from .v21_sam3_finder import ACTION_WORDS, FUNCTION_WORDS
    for part in re.split(r'[|,.;:!?()]', str(task).lower()):
        run = []
        for word in re.findall(r"[a-z][a-z'-]*", part) + [None]:
            if word is None or word in FUNCTION_WORDS or word in ACTION_WORDS:
                if run:
                    return ' '.join(run)
            else:
                run.append(word)
    raise ValueError(f'no object phrase in the task text {task!r}')


def _import_dinov3(repo):
    repo = str(Path(repo).expanduser().resolve())
    if not Path(repo, 'dinov3', 'eval', 'text', 'vision_tower.py').is_file():
        raise FileNotFoundError(f'no DINOv3 code at {repo}')
    if repo not in sys.path:
        sys.path.insert(0, repo)


def build_parts(repo=DEFAULT_REPO):
    """The vision head and the text tower of dinov3_vitl16_dinotxt_tet1280d20h24l (dinov3/hub/dinotxt.py), untrained."""
    _import_dinov3(repo)
    from dinov3.eval.text.text_tower import build_text_model
    from dinov3.eval.text.text_transformer import TextTransformer
    from dinov3.eval.text.vision_tower import VisionHead
    head = VisionHead(DIM, EMBED, HEADS, HEAD_BLOCKS, .3, True, True, False)
    text = build_text_model(EMBED, None, False, 0, False, 0., 'argmax', True, backbone=TextTransformer(
        context_length=77, vocab_size=49408, dim=1280, num_heads=20, num_layers=24, ffn_ratio=4, is_causal=True,
        ls_init_value=None, dropout_prob=0.))
    return head, text


def split_state(state):
    """Meta's vision-head-and-text-encoder file -> (head state, text state); every key must belong to one of them
    (the logit scale and bias excepted), and the file must not carry a backbone."""
    head, text, other = {}, {}, []
    for key, value in state.items():
        if key.startswith('visual_model.head.'):
            head[key[len('visual_model.head.'):]] = value
        elif key.startswith('text_model.'):
            text[key[len('text_model.'):]] = value
        elif key not in ('logit_scale', 'logit_bias'):
            other.append(key)
    if other:
        raise ValueError(f'unexpected keys in the dino.txt file: {other[:5]} ({len(other)})')
    if not head or not text:
        raise ValueError('the dino.txt file lacks the vision head or the text tower')
    return head, text


def load_dinotxt(weights=DEFAULT_WEIGHTS, bpe=DEFAULT_BPE, repo=DEFAULT_REPO):
    """-> (vision head, text tower, tokenizer) on the CPU in eval mode, every weight of the file loaded strictly."""
    head, text = build_parts(repo)
    head_state, text_state = split_state(torch.load(str(weights), map_location='cpu', weights_only=True))
    head.load_state_dict(head_state, strict=True)
    text.load_state_dict(text_state, strict=True)
    from dinov3.eval.text.tokenizer import get_tokenizer
    return head.eval(), text.eval(), get_tokenizer(str(bpe))


@torch.no_grad()
def phrase_embedding(text_tower, tokenizer, phrase, templates=TEMPLATES):
    """The patch half of the phrase's text features [HALF], unit length (the notebook's recipe)."""
    tokens = tokenizer.tokenize([t.format(phrase) for t in templates])
    features = text_tower(tokens.to(next(text_tower.parameters()).device)).float()
    if features.shape[-1] != EMBED:
        raise ValueError(f'text features {tuple(features.shape)}, expected [.., {EMBED}]')
    return F.normalize(F.normalize(features[:, HALF:], dim=-1).mean(0), dim=-1)


def contrast_names(phrase, names=ADE20K):
    """The names a phrase has to beat on a patch: every contrast name but the phrase itself (stripped, lower case)."""
    own = str(phrase).strip().lower()
    return [n for n in names if n.strip().lower() != own]


def tower_fingerprint(text_tower):
    """A short fingerprint of the text tower's weights, so cached name embeddings never outlive the weights. Sums in
    double precision, so the key is the same on every device (a float32 sum's last digits depend on the GPU)."""
    h = hashlib.sha1()
    with torch.no_grad():
        for name, value in text_tower.state_dict().items():
            v = value.double()
            h.update(f'{name}{tuple(v.shape)}{float(v.sum()):.6e}{float(v.abs().sum()):.6e}'.encode())
    return h.hexdigest()[:16]


def name_embeddings(text_tower, tokenizer, names, templates=TEMPLATES, cache=None):
    """[len(names), HALF] (CPU): phrase_embedding of every name. With a cache folder, kept there as
    dinotxt-names-<key>.npy, the key made of the tower's fingerprint, the names and the templates."""
    names, templates = list(names), list(templates)
    key = hashlib.sha1(json.dumps(dict(tower=tower_fingerprint(text_tower), names=names, templates=templates))
                       .encode()).hexdigest()[:16]
    path = None if cache is None else Path(cache) / f'dinotxt-names-{key}.npy'
    if path is not None and path.is_file():
        out = np.load(path)
        if out.shape == (len(names), HALF):
            return torch.as_tensor(out)
    out = torch.stack([phrase_embedding(text_tower, tokenizer, n, templates).cpu() for n in names]).float()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f'{path.stem}.part{os.getpid()}.npy')
        np.save(temporary, out.numpy())
        os.replace(temporary, path)
    return out


@torch.no_grad()
def patch_features(head, prefix, patches):
    """prefix [n, 5, DIM] and patches [n, P, DIM] of one forward pass -> the head's patch features [n, P, HALF], unit."""
    x = torch.cat([prefix, patches], 1).float()
    return F.normalize(head(x)[:, PREFIX:].float(), dim=-1)


@torch.no_grad()
def patch_cosines(head, prefix, patches, text):
    """prefix [n, 5, DIM] and patches [n, P, DIM] of one forward pass -> cosines [n, P] of the head's patch features to
    the phrase embedding text [HALF]."""
    return patch_features(head, prefix, patches) @ text.float()


def class_scores(features, texts):
    """features [.., P, HALF] (unit) and texts [C, HALF] (the phrase first, then the contrast names) -> [.., P]: the
    phrase's share of a softmax over the names on every patch, as Meta's notebook decides each pixel (raw cosines)."""
    return (features @ texts.float().T).softmax(-1)[..., 0]


def text_reference(served, scores, top=TOP):
    """served tokens [P, DIM] (what the finders read) and their scores [P] -> (reference [DIM] float32, record)."""
    served = np.asarray(served, np.float32)
    scores = np.asarray(scores, np.float64)
    order = np.argsort(-scores, kind='stable')[:int(top)]
    record = dict(peak=float(scores[order[0]]), top_mean=float(scores[order].mean()),
                  median=float(np.median(scores)), top=[int(i) for i in order])
    return served[order].mean(0).astype(np.float32), record


class TextScorer:
    """The phrase's and the contrast names' embeddings (made once; the names' cached) and the vision head on the device;
    reference(prefix, served) per picture."""

    def __init__(self, phrase, *, weights=DEFAULT_WEIGHTS, bpe=DEFAULT_BPE, repo=DEFAULT_REPO, device='cpu',
                 top=TOP, parts=None, contrast=ADE20K, cache=DEFAULT_ROOT):
        head, text_tower, tokenizer = load_dinotxt(weights, bpe, repo) if parts is None else parts
        self.phrase, self.top = str(phrase), int(top)
        self.weights, self.bpe, self.repo = str(weights), str(bpe), str(repo)
        self.device = torch.device(device)
        self.contrast_names = contrast_names(self.phrase, contrast)
        text_tower = text_tower.to(self.device)    # 0.5 B parameters, needed once
        self.text = phrase_embedding(text_tower, tokenizer, self.phrase).to(self.device)
        self.contrast = name_embeddings(text_tower, tokenizer, self.contrast_names, cache=cache).to(self.device)
        del text_tower
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.texts = torch.cat([self.text[None], self.contrast])
        self.head = head.to(self.device).eval()
        self.timings = []

    def features(self, prefix, served):
        p = torch.as_tensor(np.asarray(prefix, np.float32), device=self.device)[None]
        x = torch.as_tensor(np.asarray(served, np.float32), device=self.device)[None]
        if p.shape[1:] != (PREFIX, DIM) or x.shape[-1] != DIM:
            raise ValueError(f'the text finder needs the prefix [{PREFIX}, {DIM}] and patch tokens [P, {DIM}], '
                             f'got {tuple(p.shape[1:])} and {tuple(x.shape[1:])}')
        return patch_features(self.head, p, x)[0]

    def cosines(self, prefix, served):
        """The phrase's raw cosine per patch [P] (recorded; not what ranks the patches)."""
        return (self.features(prefix, served) @ self.text).cpu().numpy()

    def scores(self, prefix, served):
        """The phrase's share of the softmax over the phrase and the contrast names per patch [P]."""
        return class_scores(self.features(prefix, served), self.texts).cpu().numpy()

    def reference(self, prefix, served):
        started = time.perf_counter()
        if not np.any(prefix) or not np.any(served):
            raise ValueError('no tokens for this picture: no text reference')
        features = self.features(prefix, served)
        scores = class_scores(features, self.texts).cpu().numpy()
        cosines = (features @ self.text).cpu().numpy()
        reference, record = text_reference(served, scores, self.top)
        record.update(phrase=self.phrase, peak_cosine=float(cosines[record['top'][0]]),
                      cosine_median=float(np.median(cosines)), model_s=round(time.perf_counter() - started, 4))
        self.timings.append(record)
        del self.timings[:-256]
        return reference, record

    def describe(self):
        return dict(finder='dinotxt', phrase=self.phrase, weights=self.weights, bpe=self.bpe, repo=self.repo,
                    templates=len(TEMPLATES), contrast=len(self.contrast_names), scoring=SCORING,
                    top=self.top, source=SOURCE, sam_used=False, colour_used=False,
                    reference='mean served DINOv3-L token of the top patches by the class-softmax score')


class TextWristEncoder(LearnedObjectRelativeEncoder):
    """The learned wrist encoder (fcpg/v21_object_finder.py) with the task phrase's reference from each exposure in
    place of the fish reference in its model file; projection, conventions and place cue are the learned encoder's.
    reference_of (11/03): () -> (prefix, tokens) of the picture the reference is made from, the column picture of the
    same exposure: it shows the whole table, so the phrase's best match there is the object named, and the wrist finder
    looks for that object. Made from the wrist picture itself, the reference was the best match of whatever the wrist
    saw, and a fish was found in nearly every picture without one (the table, a monitor, a red bottle cap; with the
    column's reference almost none, and the fish where it was: robot-20261103/cross-view-reference-rollc-20261103.txt).
    Without reference_of, or while the column picture has no tokens, the wrist picture's own."""

    def __init__(self, *args, scorer, prefix_of, reference_of=None, **kwargs):
        self.scorer, self.prefix_of, self.reference_of = scorer, prefix_of, reference_of
        super().__init__(*args, **kwargs)
        if not self.model.config['conditional']:
            raise ValueError('the task-text reference needs a conditional finder')
        self.reference = None                  # never the fish's: made from every exposure below
        self.last_text = None

    def find_record(self, wrist_rgb_letterbox224):
        served = np.asarray(self.tokens_of(), np.float32)
        prefix, source, view = self.prefix_of(), served, 'wrist'
        reference_of = getattr(self, 'reference_of', None)
        if reference_of is not None:
            other_prefix, other = reference_of()
            if other_prefix is not None and other is not None and np.any(other_prefix) and np.any(other):
                prefix, source, view = other_prefix, other, 'column'
        reference, text = self.scorer.reference(np.asarray(prefix, np.float32), np.asarray(source, np.float32))
        text = dict(text, reference_view=view)
        self.reference = torch.as_tensor(reference, device=self.device)[None]
        self.last_text = text
        record, info = super().find_record(wrist_rgb_letterbox224)
        info['text'] = text
        if record is not None:
            record = dict(record, text_phrase=self.scorer.phrase, text_peak=text['peak'])
        return record, info

    def describe(self):
        return dict(super().describe(), finder='learned+dinotxt', text=self.scorer.describe(),
                    reference_view=('the column picture of the same exposure (the wrist picture\'s own while it has no '
                                    'tokens)' if getattr(self, 'reference_of', None) is not None else 'the wrist picture'))


class TextColumnFinder(LearnedColumnFinder):
    """The learned column finder with the task phrase's reference from each column picture (above)."""

    def __init__(self, model_path, *, scorer, prefix_of, device='cpu'):
        super().__init__(model_path, device=device)
        self.scorer, self.prefix_of = scorer, prefix_of
        self.reference, self.last_text = None, None

    def record(self, tokens, image_time_s=None):
        served = np.asarray(tokens, np.float32)
        text = None
        if np.any(served):
            reference, text = self.scorer.reference(np.asarray(self.prefix_of(), np.float32), served)
            self.reference = torch.as_tensor(reference, device=self.device)[None]
        self.last_text = text
        if self.reference is None:             # no column picture yet: nothing to outline
            return super().record(np.zeros_like(served), image_time_s)
        out = super().record(served, image_time_s)
        return dict(out, text=text) if text is not None else out

    def describe(self):
        return dict(super().describe(), finder='learned_column+dinotxt', text=self.scorer.describe())
