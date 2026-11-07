"""A learned object finder on the system's own DINO patches: no SAM and no colour at inference (training overhaul doc
section 17.34; James 10/19: SAM only labels the training data, the model and the robot run without it).

Input: the frozen DINO wrist tokens the system already computes (DINOv2-S/14 [256, 384]: 16 x 16 patches of the
letterbox224 picture; a finder made with encoder='dinov3_vitl16' reads the DINOv3-L tokens [196, 1024] of the same picture,
section 17.56, and records that in its config) and a reference vector for the thing to find, the mean token under that thing's outline in pictures where the
robot holds it (Archive: the same grasp's held pictures; the fish: demonstrations 0 and 1). Output: mask logits on a
64 x 64 grid over the letterbox224 picture (cells of 3.5 px). geometry() turns a probability map into found / centre /
extents in letterbox224 pixels. SAM 3 only made the training outlines (Archive labels, analysis/v21/archive_outline_
label.py; the fish, analysis/v21/demo_fish_masks.py).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .v21_encoder import ENCODERS

DEFAULT_ENCODER = 'dinov2_vits14'
GRID, DIM = ENCODERS[DEFAULT_ENCODER]      # a finder's own grid and width: ObjectFinder.grid / .dim
SIDE = 224
OUT = 64
CELL_PX = SIDE / OUT
FOUND = .5                 # a cell belongs to the thing at probability >= FOUND
MIN_CELLS = 2              # and the thing is found when at least MIN_CELLS cells do
ASPECTS = (4 / 3, 16 / 9, 16 / 10, 5 / 4, 3 / 2, 1.)
SNAP = .02                 # a picture whose aspect is this close to a common camera aspect has that aspect


class _Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=1)
        self.norm = nn.GroupNorm(8, width)

    def forward(self, x):
        return x + F.gelu(self.norm(self.conv(x)))


class ObjectFinder(nn.Module):
    """tokens [B, grid * grid, dim] (+ reference [B, dim]) -> mask logits [B, 64, 64]; grid and dim of the encoder the
    finder reads (DINOv2-S/14 16 x 16 x 384 unless made with encoder='dinov3_vitl16', 14 x 14 x 1024; section 17.56).
    conditional=False ignores the reference. presence=True (section 17.35) adds presence_logit(pooled, reference): is
    the thing under the mask the one the reference describes (pooled = pool(tokens, patch weights of the mask))."""

    def __init__(self, width=192, blocks=3, conditional=True, presence=False, encoder=DEFAULT_ENCODER):
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f'unknown encoder {encoder!r}')
        self.encoder = encoder
        self.grid, self.dim = ENCODERS[encoder]
        self.config = dict(width=int(width), blocks=int(blocks), conditional=bool(conditional))
        if presence:
            self.config['presence'] = True
        if encoder != DEFAULT_ENCODER:          # DINOv2 finders keep the config they always had
            self.config['encoder'] = encoder
        dim = self.dim
        self.norm_tokens = nn.LayerNorm(dim)
        self.norm_reference = nn.LayerNorm(dim)
        inputs = 3 * dim + 1 if conditional else dim
        self.embed = nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, width))
        self.context = nn.Sequential(*[_Block(width) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv2d(width, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 1, 1))
        if presence:
            self.norm_pooled = nn.LayerNorm(dim)
            self.presence_net = nn.Sequential(nn.Linear(3 * dim + 1, width), nn.GELU(), nn.Linear(width, width),
                                              nn.GELU(), nn.Linear(width, 1))

    def presence_logit(self, pooled, reference):
        p, r = self.norm_pooled(pooled), self.norm_reference(reference)
        cos = F.cosine_similarity(pooled, reference, dim=-1, eps=1e-6)[:, None]
        return self.presence_net(torch.cat((p, r, p * r, cos), -1))[:, 0]

    def forward(self, tokens, reference=None):
        b = tokens.shape[0]
        t = self.norm_tokens(tokens)
        if self.config['conditional']:
            if reference is None:
                raise ValueError('a conditional finder needs the reference vector')
            r = self.norm_reference(reference)[:, None, :].expand_as(t)
            cos = F.cosine_similarity(tokens, reference[:, None, :], dim=-1, eps=1e-6)[..., None]
            x = torch.cat((t, r, t * r, cos), -1)
        else:
            x = t
        x = self.embed(x).transpose(1, 2).reshape(b, -1, self.grid, self.grid)
        x = self.context(x)
        x = F.interpolate(x, size=(OUT, OUT), mode='bilinear', align_corners=False)
        return self.head(x)[:, 0]


def pool(tokens, weights):
    """The mean token under patch weights: tokens [B, grid * grid, dim], weights [B, grid, grid] -> [B, dim] (zeros
    when empty)."""
    w = weights.reshape(weights.shape[0], -1, 1).to(tokens.dtype)
    return (w * tokens).sum(1) / w.sum(1).clamp_min(1e-6)


def area_pool(cover, grid=GRID):
    """Area fractions on the 64 x 64 grid [B, 64, 64] -> their mean over each patch of a grid x grid encoder [B, grid,
    grid] (DINOv2: exact 4 x 4 cells; DINOv3-L's 14 x 14 patches do not divide 64: adaptive bins of 4-5 cells)."""
    x = cover.float()[:, None]
    return (F.avg_pool2d(x, OUT // grid) if OUT % grid == 0 else F.adaptive_avg_pool2d(x, grid))[:, 0]


def patch_weights(prob, threshold=FOUND, grid=GRID):
    """A mask on the 64 x 64 grid (probability >= threshold) as area weights of the grid x grid patches: [B, 64, 64]
    -> [B, grid, grid]."""
    return area_pool(prob >= threshold, grid)


def loss(logits, target, positive):
    """Per-cell cross entropy against the outline's area fractions, plus a soft overlap (Dice) term on pictures that show
    the thing; positive [B] bool."""
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none').mean((1, 2))
    p = torch.sigmoid(logits.float())
    inter = (p * target).sum((1, 2))
    dice = 1. - (2. * inter + 1.) / (p.sum((1, 2)) + target.sum((1, 2)) + 1.)
    return (bce + positive.float() * dice).mean()


def snap_aspect(h, w):
    a = float(w) / float(h)
    for c in ASPECTS + tuple(1. / v for v in ASPECTS[:-1]):
        if abs(a / c - 1.) < SNAP:
            return c
    return a


def letterbox_coverage(mask, picture_hw=None):
    """A mask over a picture (any resolution with that picture's aspect, e.g. the outline labeller's quarter size; the
    aspect from picture_hw when given) -> float [224, 224] coverage in the letterbox the DINO caches used
    (fcpg/lerobot_archive.letterbox_rgb: long side 224, centred)."""
    import cv2
    mask = np.asarray(mask, np.float32)
    h, w = picture_hw if picture_hw is not None else mask.shape
    a = snap_aspect(h, w)
    if a >= 1:
        cw, ch = SIDE, max(1, int(round(SIDE / a)))
    else:
        cw, ch = max(1, int(round(SIDE * a))), SIDE
    out = np.zeros((SIDE, SIDE), np.float32)
    left, top = (SIDE - cw) // 2, (SIDE - ch) // 2
    out[top:top + ch, left:left + cw] = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_AREA)
    return out


def targets(coverage224, grid=GRID):
    """(area fractions [64, 64], patch coverage [grid, grid]) of a [224, 224] coverage."""
    import cv2
    c = np.asarray(coverage224, np.float32)
    return (cv2.resize(c, (OUT, OUT), interpolation=cv2.INTER_AREA),
            cv2.resize(c, (grid, grid), interpolation=cv2.INTER_AREA))


def reference(tokens, coverage):
    """Mean token under the outline: tokens [n, grid * grid, dim], coverage [n, grid, grid] -> [dim] float32, or None
    when the outlines cover less than one patch in all."""
    w = np.asarray(coverage, np.float64).reshape(len(coverage), -1)
    total = w.sum()
    if total < 1.:
        return None
    acc = np.zeros(np.shape(tokens)[-1], np.float64)
    for k in range(len(w)):
        if w[k].sum() > 0:
            acc += (w[k][:, None] * np.asarray(tokens[k], np.float64)).sum(0)
    return (acc / total).astype(np.float32)


def geometry(prob):
    """A probability map [64, 64] -> dict(found, cells, peak, score, centre_px, length_px, width_px, theta) in letterbox224
    pixels (centre as (x, y)); centre and extents only when found."""
    prob = np.asarray(prob, np.float64)
    keep = prob >= FOUND
    cells = int(keep.sum())
    out = dict(found=cells >= MIN_CELLS, cells=cells, peak=float(prob.max()))
    if not out['found']:
        return out
    ys, xs = np.nonzero(keep)
    w = prob[ys, xs]
    x, y = (xs + .5) * CELL_PX, (ys + .5) * CELL_PX
    cx, cy = float((w * x).sum() / w.sum()), float((w * y).sum() / w.sum())
    cov = np.cov(np.stack((x - cx, y - cy)), aweights=w, bias=True)
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, 1]
    out.update(score=float(w.mean()), centre_px=(cx, cy), length_px=float(np.sqrt(12. * max(values[1], 0.))),
               width_px=float(np.sqrt(12. * max(values[0], 0.))), theta=float(np.arctan2(major[1], major[0])))
    return out


def coverage_centre(cover64):
    """Area-weighted centre (x, y) in letterbox224 pixels of a [64, 64] area-fraction map, or None when empty."""
    c = np.asarray(cover64, np.float64)
    if c.sum() <= 0:
        return None
    ys, xs = np.mgrid[:OUT, :OUT]
    return (float(((xs + .5) * CELL_PX * c).sum() / c.sum()), float(((ys + .5) * CELL_PX * c).sum() / c.sum()))


def save(path, model, **record):
    torch.save(dict(config=model.config, state=model.state_dict(), **record), path)


def load(path, device='cpu'):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = ObjectFinder(**blob['config']).to(device).eval()
    model.load_state_dict(blob['state'])
    return model, blob


SOURCE = 'learned_dino_finder'     # the record source of the service's wrist object token with this finder


def _encoder_base():
    from .v21_sam3_finder import Sam3ObjectRelativeEncoder
    return Sam3ObjectRelativeEncoder


class LearnedObjectRelativeEncoder(_encoder_base()):
    """Drop-in for the colour encoder (and the SAM 3 one, fcpg/v21_sam3_finder.Sam3ObjectRelativeEncoder, whose jaw
    projection, token conventions and score field it keeps): the outline comes from the learned finder on the frozen
    DINO tokens of the very exposure being encoded (tokens_of(): TrialOwner.capture sets last_dino_tokens before it
    encodes), with the task's reference vector made offline from its demonstrations. No SAM, no colour, no network.
    score_source: 'dino_pickup' for the colour token layout ('jaw', H1b), 'one' for the outline cue ('hold'), as the
    Archive cues had it."""

    def __init__(self, model_path, wrist_reference, kinematics, *, convention, tokens_of, hold_px=None,
                 reference=None, score_source=None, dino_tracker=None, device='cpu', place_finder=None, holding=None):
        from .so101_camera_calibration import ProjectionFit
        from .v21_sam3_finder import CONVENTIONS
        import json
        from pathlib import Path
        if convention not in CONVENTIONS:
            raise ValueError(f'unknown token convention {convention!r}')
        score_source = (dict(jaw='dino_pickup', hold='one')[convention] if score_source is None
                        else str(score_source))
        if score_source not in ('dino_pickup', 'one'):
            raise ValueError("the learned finder's score field is 'dino_pickup' or 'one'")
        if score_source == 'dino_pickup' and not callable(getattr(dino_tracker, 'locate_by_frozen_features', None)):
            raise ValueError("score source 'dino_pickup' needs the DINO tracker of the colour-era encoder")
        if convention == 'hold' and (hold_px is None or not np.isfinite(np.asarray(hold_px, np.float64)).all()):
            raise ValueError("the 'hold' convention needs the demonstrations' hold reference")
        self.device = torch.device(device)
        self.model, blob = load(model_path, self.device)
        reference = blob.get('fish_reference') if reference is None else reference
        if (self.model.config['conditional'] or self.model.config.get('presence')) and reference is None:
            raise ValueError('this finder needs the task reference vector')
        self.reference = None if reference is None else torch.as_tensor(np.asarray(reference, np.float32),
                                                                         device=self.device)[None]
        self.model_path, self.tokens_of = str(model_path), tokens_of
        self.encoder = self.model.encoder          # tokens_of() gives this encoder's tokens (the service picks them)
        self.score_source, self.dino_tracker = score_source, dino_tracker
        self.convention, self.anchor_word, self.threshold = convention, None, FOUND
        self.prompts, self.rule, self.mode, self.client, self.stream = [], 'learned', 'learned', None, 'wrist'
        self.hold_px = None if hold_px is None else np.asarray(hold_px, np.float64).reshape(2)
        calibration = json.loads(Path(wrist_reference).read_text(encoding='utf-8'))
        fits = calibration['fits']
        row = fits['wrist'] if isinstance(fits, dict) else next(r for r in fits if r.get('view') == 'wrist')
        self.wrist_fit = ProjectionFit('wrist', tuple(row['feature_names']),
                                       np.asarray(row['coefficients'], np.float64), float(row['ridge']))
        self.kinematics, self.wrist_reference, self.raw_frame = kinematics, str(wrist_reference), None
        self._colour_model = None
        self.tracker = None
        self.timings = []
        # section 17.117: where the held thing goes, from the learned box finder on the same exposure's tokens
        if (place_finder is None) != (holding is None):
            raise ValueError('the place cue needs both the box finder and the holding test')
        if place_finder is not None and (convention != 'hold' or place_finder.encoder != self.encoder):
            raise ValueError("the place cue is the 'hold' convention on this encoder's own tokens")
        self.place_finder, self.holding = place_finder, holding
        self.clear_observation()

    def encode_place(self, wrist_rgb_letterbox224, q):
        """Where the held thing goes (training overhaul doc section 17.117): the learned box finder on the wrist tokens
        of this very exposure, in this encoder's 'hold' convention against the same reference, score 1 (as the tables
        give it); zeros while nothing is held (the Archive taught the place cue during the carry only) and when the
        box is not found."""
        if self.place_finder is None:
            raise RuntimeError('this encoder was made without the box finder: no place cue')
        self.last_place_record = None
        if not self.holding():
            return np.zeros(8, np.float32)
        record = self.place_finder.record(np.asarray(self.tokens_of(), np.float32), 'wrist')
        self.last_place_record = record
        if not record.get('visible'):
            return np.zeros(8, np.float32)
        from .v21_sam3_finder import token
        return token(dict(cog_px=record['cog_px'], theta_image=record['theta_image'], length_px=record['length_px'],
                          width_px=record['width_px'], score=1.), self.convention, hold_px=self.hold_px)

    @torch.no_grad()
    def find_record(self, wrist_rgb_letterbox224):
        import time
        started = time.perf_counter()
        tokens = np.asarray(self.tokens_of(), np.float32)
        expected = (self.model.grid ** 2, self.model.dim)
        if tokens.shape != expected:
            raise ValueError(f'the learned finder ({self.encoder}) needs the wrist tokens {list(expected)}, '
                             f'got {tokens.shape}')
        x = torch.as_tensor(tokens, device=self.device)[None]
        logits = self.model(x, self.reference if self.model.config['conditional'] else None)
        prob = torch.sigmoid(logits.float())
        g = geometry(prob[0].cpu().numpy())
        presence = None
        if g['found'] and self.model.config.get('presence'):
            pooled = pool(x, patch_weights(prob, grid=self.model.grid))
            presence = float(torch.sigmoid(self.model.presence_logit(pooled, self.reference).float())[0])
        found = g['found'] and (presence is None or presence >= FOUND)
        info = dict(picture='dino_tokens', found=found, mask_found=g['found'], presence=presence, cells=g['cells'],
                    peak=g['peak'], word_scores={}, model_s=round(time.perf_counter() - started, 4))
        self.timings.append(info)
        del self.timings[:-256]
        if not found:
            return None, info
        record = dict(cog_px=[float(v) for v in g['centre_px']], theta_image=float(g['theta']),
                      length_px=float(g['length_px']), width_px=float(g['width_px']), score=float(g['score']),
                      finder_score=float(g['score']), cells=int(g['cells']), source=SOURCE, available=True)
        return record, info

    def clear_observation(self):
        self.last_record = self.last_policy_record = self.last_find = self.last_found = None
        self.last_place_record = None
        self.last_diagnostic_vector = np.zeros(8, np.float32)
        self.track_reset_error = None

    def describe(self):
        return dict(fields=['available', 'score', 'dx', 'dy', 'length', 'width', 'cos2theta', 'sin2theta'],
                    place=None if self.place_finder is None else dict(
                        finder=self.place_finder.describe(), view='wrist', convention=self.convention,
                        while_='the gripper holds the thing (section 17.117)', score=1.),
                    finder='learned', model=self.model_path, model_config=dict(self.model.config), encoder=self.encoder,
                    convention=self.convention, hold_px=None if self.hold_px is None else self.hold_px.tolist(),
                    score_source=self.score_source, found_probability=FOUND, found_cells=MIN_CELLS,
                    wrist_reference=self.wrist_reference, coordinate_space='letterbox224/224',
                    policy_source=SOURCE, colour_used=False, sam_used=False)


COLUMN_SOURCE = 'learned_dino_finder_column'   # the service's column fish record with the column finder (17.51)


class LearnedColumnFinder:
    """The column fish of the live grasp target (training overhaul doc section 17.51): the learned finder on the frozen
    DINO tokens of the column picture of the exposure being planned on (the service's last_dino_tokens[1]), with the
    column fish reference its model file carries (scripts/train_v21_object_finder.py --column-pairs: the fish on the
    table as the column camera saw it in demonstrations 0 and 1). No SAM, no colour, no network. record() gives the
    column pick-up record fcpg/v21_place_goal.LivePlaceGoal.observe_fish takes (source COLUMN_SOURCE, centre in
    letterbox224 pixels), or a not-visible one."""

    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model, blob = load(model_path, self.device)
        reference = blob.get('column_reference')
        if reference is None or not self.model.config['conditional']:
            raise ValueError('the column finder is a conditional finder trained with --column-pairs (section 17.51)')
        self.reference = torch.as_tensor(np.asarray(reference, np.float32), device=self.device)[None]
        self.model_path = str(model_path)
        self.encoder = self.model.encoder          # record() takes this encoder's column tokens
        self.timings = []

    @torch.no_grad()
    def record(self, tokens, image_time_s=None):
        import time
        started = time.perf_counter()
        tokens = np.asarray(tokens, np.float32)
        expected = (self.model.grid ** 2, self.model.dim)
        if tokens.shape != expected:
            raise ValueError(f'the column finder ({self.encoder}) needs the column tokens {list(expected)}, '
                             f'got {tokens.shape}')
        stamp = {} if image_time_s is None else dict(image_time_s=float(image_time_s))
        base = dict(source=COLUMN_SOURCE, role='pickup', view='column', colour_used=False, sam_used=False,
                    picture_source='dino_tokens', **stamp)
        if not np.any(tokens):           # the column picture was not valid for this exposure (no tokens were made)
            return dict(base, visible=False, reason='no_column_tokens')
        logits = self.model(torch.as_tensor(tokens, device=self.device)[None], self.reference)
        g = geometry(torch.sigmoid(logits.float())[0].cpu().numpy())
        info = dict(cells=g['cells'], peak=g['peak'], model_s=round(time.perf_counter() - started, 4))
        self.timings.append(info)
        del self.timings[:-256]
        if not g['found']:
            return dict(base, visible=False, reason='learned_finder_found_nothing', finder=info)
        centre = [float(v) for v in g['centre_px']]
        return dict(base, visible=True, cog_px=centre, centroid_px=list(centre),
                    axis_px=[float(np.cos(g['theta'])), float(np.sin(g['theta']))], length_px=float(g['length_px']),
                    width_px=float(g['width_px']), score=float(g['score']), theta_image=float(g['theta']),
                    cells=int(g['cells']), finder=info)

    def describe(self):
        return dict(finder='learned_column', model=self.model_path, model_config=dict(self.model.config),
                    encoder=self.encoder,
                    source=COLUMN_SOURCE, found_probability=FOUND, found_cells=MIN_CELLS,
                    coordinate_space='letterbox224/224', colour_used=False, sam_used=False)


BOX_SOURCE = 'learned_dino_box_finder'   # the service's drop-zone records with the box finder (section 17.57)
BOX_VIEWS = ('wrist', 'column')


class LearnedBoxFinder:
    """The box (the drop zone) in the wrist and the column picture of an exposure (training overhaul doc section 17.57):
    the learned conditional finder on that exposure's own tokens of the view, with the view's box reference its model
    file carries (scripts/train_v21_object_finder.py --box-pairs: the box as the wrist and the column camera saw it in
    demonstrations 0 and 1). No SAM, no colour, no frozen-feature prototype and no hand-set similarity bar: the box is
    found where the finder's own mask has at least MIN_CELLS cells at probability FOUND, as every learned finder here.
    record(tokens, view) gives the drop-zone record the service's targets carry (the frozen-DINOv2 tracker's schema,
    fcpg/prod_v17_7_dino_targets._visible_payload / _not_visible; source BOX_SOURCE, centre in letterbox224 pixels):
    fcpg/v21_alignment_cem reads it to align shut jaws to the box, for the gripper rule's release and for
    released_over_box; the service's drop goal reads the column one."""

    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model, blob = load(model_path, self.device)
        if not self.model.config['conditional'] or any(blob.get(f'{v}_box_reference') is None for v in BOX_VIEWS):
            raise ValueError('the box finder is a conditional finder trained with --box-pairs (section 17.57)')
        self.references = {v: torch.as_tensor(np.asarray(blob[f'{v}_box_reference'], np.float32),
                                               device=self.device)[None] for v in BOX_VIEWS}
        self.model_path = str(model_path)
        self.encoder = self.model.encoder          # record() takes this encoder's tokens
        self.timings = []
        self.last_prob = {}                        # view -> the last map record() made (section 17.118: the box solid)

    @torch.no_grad()
    def record(self, tokens, view, image_time_s=None):
        import time
        if view not in BOX_VIEWS:
            raise ValueError(f'unknown view {view!r}')
        started = time.perf_counter()
        tokens = np.asarray(tokens, np.float32)
        expected = (self.model.grid ** 2, self.model.dim)
        if tokens.shape != expected:
            raise ValueError(f'the box finder ({self.encoder}) needs the {view} tokens {list(expected)}, '
                             f'got {tokens.shape}')
        stamp = {} if image_time_s is None else dict(image_time_s=float(image_time_s))
        base = dict(role='drop_zone', view=view, source=BOX_SOURCE, separation=0., colour_used=False, sam_used=False,
                    picture_source='dino_tokens', **stamp)
        empty = dict(visible=False, centroid_px=None, axis_px=None, width_px=None, length_px=None, score=-1.)
        if not np.any(tokens):           # the picture was not valid for this exposure (no tokens were made)
            self.last_prob.pop(view, None)
            return dict(base, **empty, reason=f'no_{view}_tokens')
        logits = self.model(torch.as_tensor(tokens, device=self.device)[None], self.references[view])
        prob = torch.sigmoid(logits.float())[0].cpu().numpy()
        self.last_prob[view] = prob
        g = geometry(prob)
        info = dict(cells=g['cells'], peak=g['peak'], model_s=round(time.perf_counter() - started, 4))
        self.timings.append(info)
        del self.timings[:-256]
        if not g['found']:
            return dict(base, **empty, reason='learned_box_finder_found_nothing', finder=info)
        centre = [float(v) for v in g['centre_px']]
        return dict(base, visible=True, cog_px=centre, centroid_px=list(centre),
                    axis_px=[float(np.cos(g['theta'])), float(np.sin(g['theta']))], length_px=float(g['length_px']),
                    width_px=float(g['width_px']), score=float(g['score']), theta_image=float(g['theta']),
                    cells=int(g['cells']), finder=info)

    def describe(self):
        return dict(finder='learned_box', model=self.model_path, model_config=dict(self.model.config),
                    encoder=self.encoder, views=list(BOX_VIEWS), source=BOX_SOURCE, found_probability=FOUND,
                    found_cells=MIN_CELLS, coordinate_space='letterbox224/224', colour_used=False, sam_used=False)
