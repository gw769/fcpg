"""Colour-free live finder for the fish (training overhaul doc section 17.32; James 10/18 "肯定不要用颜色", 10/19 "继续").

SAM 3, named with the object word of the task text, finds the thing to grasp in the camera pictures. It runs in its own
process (scripts/serve_v21_sam3_finder.py: the sam3-tf environment on the .102 A100) and the service asks it over a
local Unix socket at every capture. This module is the service's side: the socket client, the geometry, and the pieces
the service swaps in:
- Sam3ObjectRelativeEncoder replaced the operator colour model's encoder (gone from the code, 10/31). Its
  wrist object token is what the Actor reads and what fcpg/v21_alignment_cem.py aligns to, admits the close with and
  checks released_over_box with;
- column_pickup_record gives the column pick-up record that
  fcpg/v21_place_goal.LivePlaceGoal.observe_fish takes the grasp target of 17.14 models from;
- the operator colour model this finder replaced is gone from the code (10/31).
The box stays the DINO registration (fcpg/v21_target_tracker.py), which uses no colour.

What SAM 3 is asked for: the object words of the task text (object_prompts: the first run of words that are neither
function, colour nor action words, whole and word by word; "Pick up the red fish toy ..." -> "fish toy", "fish", "toy"),
all on one vision pass. On the 10/16 records "fish" alone missed the fish once it sat between the jaws (a close-up of
its head), where "toy" found it (doc 17.32). One mask is kept (choose): RULES 'best' = the best score over the words,
'first' = the first word, in that order, that finds anything.

The token: [available, score, dx, dy, length, width, cos 2theta, sin 2theta] in letterbox224 / 224, from the chosen
SAM 3 mask at or above THRESHOLD (analysis/v21/demo_fish_cues.py cut the demonstrations' cues the T2 models learned
from with "fish" alone and the same threshold). Two conventions, by what the model learned:
- 'jaw' (the colour token's layout, fcpg/v21_object_relative.encode_vector; H1b and every model before T2): dx, dy from
  the jaw-pair midpoint of the wrist projection, theta against the jaw line. The CEM's target (0, 0) is then the fish
  centre at that midpoint, as on 10/16-10/17;
- 'hold' (the outline cue of fcpg/v21_outline_cue.py; T2, T2b): dx, dy from where the fish sits in the jaws of our
  training demonstrations 0 and 1 (the mean of their held centres; the exam demonstration 3 is not used), theta
  against the picture's x axis.
The score field, too, as the model's training wrote it (SCORE_SOURCES; both Actors move with it, doc 17.32):
- 'dino_pickup' ('jaw'): the colour-era token's score was not colour: the frozen DINO features' confirmation score of
  the registered pickup in the same wrist picture (the old colour encoder asked
  VisualTargetTracker.locate_by_frozen_features('pickup', 'wrist', letterbox, jaw_pair_px)), and 1 when that does not
  see the object (232 robot states of 10/16: equal to 1e-4 in all 114 it saw, 1 in all 118 others,
  analysis/v21/token_score_source_check.py). The same call on the same picture, here with no colour anywhere;
- 'detection' ('hold'): the demonstrations' fish cues carry SAM 3's best "fish" score at THRESHOLD on that picture
  (analysis/v21/demo_fish_cues.py), the Archive cues 1 for an accepted outline (fcpg/v21_outline_cue.py): the anchor
  word's best score on this picture when it has one, else 1 (the track alone has the fish).
SAM 3 sees the camera's raw picture of the same exposure when the service has it (the demonstrations' cues were cut
from 448-row pictures), else the letterbox.
"""
import json
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np

import re

DIM = 8
PROMPT = 'fish'                 # the demonstrations' cues (analysis/v21/demo_fish_cues.py) used this word alone
THRESHOLD = .3                  # analysis/v21/demo_fish_cues.py
ANCHOR_SCORE = .5               # a detection this sure (re)starts the live track (doc 17.32: 151 of 167 such on the 10/16
                                # records within 15 px of the colour record's centre, the rest 16-42 px)
CONVENTIONS = ('jaw', 'hold')
RULES = ('best', 'first')
MODES = ('track', 'detect')
SCORE_SOURCES = ('dino_pickup', 'detection', 'one')      # the token's score field; see the module docstring
DEFAULT_SCORE_SOURCE = dict(jaw='dino_pickup', hold='detection')
# analysis/v21/archive_outline_sam3.py FUNCTION_WORDS and COLOUR_WORDS (tests/test_v21_sam3_finder.py checks they stay
# equal; that module's imports are too heavy for the service), and the verbs a task starts with
FUNCTION_WORDS = set('a an the and or but of to into onto on in inside within at from with without by for up down off '
                     'out over under above below near next it its them they their this that these those all each every '
                     'both other another some any one two three four five six seven eight nine ten first second third '
                     'fourth then than so as is are be been being please there here where which who'.split())
COLOUR_WORDS = set('red green blue yellow white black grey gray brown orange purple pink violet cyan magenta beige gold '
                   'golden silver light dark'.split())
ACTION_WORDS = set('pick place put grab take move push pull lift insert stack drop hold carry set get bring wipe pour '
                   'open close press turn slide fold hang throw toss sort'.split())


def object_prompts(task):
    """The object words of a task text: its first run of words that are neither function, colour nor action words,
    whole and then word by word ("Pick up the red fish toy and place it in the cardboard box" -> ['fish toy', 'fish',
    'toy']). Colour words never reach SAM 3."""
    for part in re.split(r'[|,.;:!?()]', task.lower()):
        run = []
        for word in re.findall(r"[a-z][a-z'-]*", part) + [None]:
            if word is None or word in FUNCTION_WORDS or word in COLOUR_WORDS or word in ACTION_WORDS:
                if run:
                    return list(dict.fromkeys([' '.join(run)] + run))
            else:
                run.append(word)
    raise ValueError(f'no object word in the task text {task!r}')
SOURCE = 'sam3_task_word'       # the record source: fcpg/v21_place_goal.LivePlaceGoal takes column fish records with it
MAGIC = b'S3F1'
MAX_HEADER = 1 << 20
IMAGE = 224.
MIN_PIXELS = 4
LETTERBOX_MATCH = 6.            # mean abs grey-level difference above which the raw picture is not this letterbox


# ---- the wire: MAGIC, two little-endian uint32 lengths, a JSON header, a binary payload ----

def send_message(sock, header, payload=b''):
    head = json.dumps(header, allow_nan=False).encode()
    sock.sendall(MAGIC + struct.pack('<II', len(head), len(payload)) + head + payload)


def _read_exact(sock, n):
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(min(1 << 20, n - len(out)))
        if not chunk:
            raise ConnectionError('finder connection closed')
        out += chunk
    return bytes(out)


def receive_message(sock):
    if _read_exact(sock, 4) != MAGIC:
        raise ConnectionError('not a finder message')
    head_len, payload_len = struct.unpack('<II', _read_exact(sock, 8))
    if head_len > MAX_HEADER or payload_len > 64 << 20:
        raise ConnectionError('finder message too large')
    return json.loads(_read_exact(sock, head_len)), _read_exact(sock, payload_len)


def pack_masks(masks):
    """[n, h, w] bool -> bytes (np.packbits over the flattened masks)."""
    masks = np.asarray(masks, bool)
    return np.packbits(masks.reshape(len(masks), -1), axis=1).tobytes() if len(masks) else b''


def unpack_masks(payload, n, h, w):
    if n == 0:
        return np.zeros((0, h, w), bool)
    per = (h * w + 7) // 8
    bits = np.frombuffer(payload, np.uint8)
    if bits.size != n * per:
        raise ValueError('mask payload size mismatch')
    return np.unpackbits(bits.reshape(n, per), axis=1, count=h * w).reshape(n, h, w).astype(bool)


class FinderClient:
    """The service's connection to scripts/serve_v21_sam3_finder.py. find() -> ([(mask [h, w] bool, score)], info);
    one request at a time, reconnecting once when the connection was lost."""

    def __init__(self, path, timeout_s=3.):
        self.path, self.timeout_s = str(Path(path).expanduser()), float(timeout_s)
        self._sock = None
        self._lock = threading.Lock()
        self.last_info = None

    def _connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_s)
        sock.connect(self.path)
        self._sock = sock

    def close(self):
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                finally:
                    self._sock = None

    def _ask(self, header, payload):
        if self._sock is None:
            self._connect()
        send_message(self._sock, header, payload)
        return receive_message(self._sock)

    def request(self, header, payload=b''):
        with self._lock:
            for attempt in (0, 1):
                try:
                    return self._ask(header, payload)
                except (ConnectionError, OSError):
                    if self._sock is not None:
                        try:
                            self._sock.close()
                        except OSError:
                            pass
                        self._sock = None
                    if attempt:
                        raise

    def describe(self):
        reply, _ = self.request(dict(op='describe'))
        return reply

    def _find(self, rgb, header):
        rgb = np.ascontiguousarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('the finder takes uint8 RGB [h, w, 3]')
        started = time.monotonic()
        reply, payload = self.request(dict(dict(op='find'), shape=list(rgb.shape), **header), rgb.tobytes())
        self.last_reply = reply
        if not reply.get('ok'):
            raise RuntimeError(f"finder error: {reply.get('error')}")
        n, h, w = (int(v) for v in reply['mask_shape'])
        masks = unpack_masks(payload, n, h, w)
        scores = [float(s) for s in reply.get('scores') or [1.] * n]
        owners = [int(i) for i in (reply.get('prompt_index') or [0] * n)]
        self.last_info = dict(round_trip_s=time.monotonic() - started, model_s=reply.get('model_s'), found=n,
                              threshold=float(header['threshold']))
        return masks, scores, owners, self.last_info

    def find(self, rgb, prompt=PROMPT, threshold=THRESHOLD):
        """One prompt -> ([(mask, score)], info)."""
        masks, scores, _, info = self._find(rgb, dict(prompt=str(prompt), threshold=float(threshold)))
        return list(zip(masks, scores)), dict(info, prompt=prompt)

    def track(self, rgb, stream, prompts, threshold=THRESHOLD, *, anchor_word=PROMPT, anchor_score=ANCHOR_SCORE):
        """The finder's live track for one camera stream (scripts/serve_v21_sam3_finder.Finder.track) ->
        (mask or None, info with the track's source and the words' best detection scores)."""
        prompts = [str(p) for p in prompts]
        masks, _, _, info = self._find(rgb, dict(op='track', stream=str(stream), prompts=prompts,
                                                 threshold=float(threshold), anchor_word=str(anchor_word),
                                                 anchor_score=float(anchor_score)))
        reply = self.last_reply
        return (masks[0] if len(masks) else None), dict(info, track=reply.get('track'),
                                                         word_scores=reply.get('word_scores'))

    def track_reset(self, stream):
        reply, _ = self.request(dict(op='track_reset', stream=str(stream)))
        if not reply.get('ok'):
            raise RuntimeError(f"finder error: {reply.get('error')}")

    def find_many(self, rgb, prompts, threshold=THRESHOLD):
        """Several prompts on one picture (one vision pass in the finder) -> ([(mask, score, prompt)], info)."""
        prompts = [str(p) for p in prompts]
        masks, scores, owners, info = self._find(rgb, dict(prompts=prompts, threshold=float(threshold)))
        return [(m, s, prompts[o]) for m, s, o in zip(masks, scores, owners)], dict(info, prompts=prompts)


# ---- geometry ----

def mask_geometry(mask):
    """(centre_x, centre_y, length, width, theta) of a boolean mask in its own pixels (centres at +0.5), as
    fcpg/v21_outline_cue.mask_geometry; None below MIN_PIXELS."""
    from .v21_outline_cue import mask_geometry as outline_geometry
    return outline_geometry(mask)


def contour_px(mask, transform, limit=64):
    """The mask's outer outline as letterbox224 points (display only), at most `limit` points."""
    import cv2
    found, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found:
        return []
    outline = max(found, key=cv2.contourArea).reshape(-1, 2).astype(np.float64) + .5
    if len(outline) > limit:
        outline = outline[np.linspace(0, len(outline) - 1, limit).round().astype(int)]
    return [[float(x), float(y)] for x, y in transform(outline)]


class PictureTransform:
    """Maps the finder's picture to letterbox224: the identity for the letterbox itself, letterbox_v1 for a raw picture
    (fcpg/letterbox_geometry.py, the geometry the service's letterbox comes from)."""

    def __init__(self, picture_hw, raw):
        self.h, self.w = (int(v) for v in picture_hw)
        self.raw = bool(raw)
        if self.raw:
            from .letterbox_geometry import letterbox_geometry
            self.geometry = letterbox_geometry(self.w, self.h, int(IMAGE))
            self.scale = float(IMAGE) / max(self.h, self.w)
        else:
            if (self.h, self.w) != (int(IMAGE), int(IMAGE)):
                raise ValueError('a letterbox picture is 224 x 224')
            self.geometry, self.scale = None, 1.

    def __call__(self, points_xy):
        points = np.asarray(points_xy, np.float64).reshape(-1, 2)
        return points if self.geometry is None else np.asarray(self.geometry.raw_to_planning(points), np.float64)


def record_from_mask(mask, score, transform, *, image_time_s=None):
    """A tracker-style record of the mask in letterbox224 pixels (cog_px, axis_px, length_px, width_px, contour_px), or
    None for an empty mask."""
    found = mask_geometry(mask)
    if found is None:
        return None
    cx, cy, length, width, theta = found
    centre = transform([(cx, cy)])[0]
    return dict(source=SOURCE, visible=True, cog_px=[float(centre[0]), float(centre[1])],
                centroid_px=[float(centre[0]), float(centre[1])],
                axis_px=[float(np.cos(theta)), float(np.sin(theta))],
                length_px=float(length * transform.scale), width_px=float(width * transform.scale),
                score=float(score), area_px=int(np.count_nonzero(mask)), theta_image=float(theta),
                contour_px=contour_px(mask, transform), picture_hw=[transform.h, transform.w],
                picture_raw=transform.raw, colour_used=False,
                **({} if image_time_s is None else dict(image_time_s=float(image_time_s))))


def token(record, convention, *, jaw_pair_px=None, hold_px=None):
    """The eight numbers of a record (letterbox224 pixels) in one convention; zeros for no record."""
    out = np.zeros(DIM, np.float32)
    if record is None:
        return out
    if convention not in CONVENTIONS:
        raise ValueError(f'unknown token convention {convention!r}')
    centre = np.asarray(record['cog_px'], np.float64)
    theta = float(record['theta_image'])
    if convention == 'jaw':
        pair = np.asarray(jaw_pair_px, np.float64).reshape(2, 2)
        jaw = pair[1] - pair[0]
        if not np.isfinite(pair).all() or np.linalg.norm(jaw) < 1e-6:
            return out
        origin = pair.mean(0)
        theta -= float(np.arctan2(jaw[1], jaw[0]))
    else:
        origin = np.asarray(hold_px, np.float64).reshape(2)
    values = (1., float(np.clip(record['score'], 0., 1.)), *((centre - origin) / IMAGE),
              record['length_px'] / IMAGE, record['width_px'] / IMAGE, np.cos(2 * theta), np.sin(2 * theta))
    if not np.isfinite(values).all():
        return out
    out[:] = values
    return out


def best(found):
    """The best-scoring entry ((mask, score, ...)) with a usable mask, or None."""
    usable = [f for f in found if np.count_nonzero(f[0]) >= MIN_PIXELS]
    return max(usable, key=lambda f: f[1]) if usable else None


def choose(found, prompts, rule):
    """One (mask, score, prompt) of find_many's answer: 'best' = the best score over all words, 'first' = the best of
    the first word in `prompts` order that found anything; None when nothing usable."""
    if rule == 'best':
        return best(found)
    if rule != 'first':
        raise ValueError(f'unknown rule {rule!r}')
    for prompt in prompts:
        chosen = best([f for f in found if f[2] == prompt])
        if chosen is not None:
            return chosen
    return None


def word_scores(found, prompts):
    """{prompt: best score or None} for the record."""
    out = {}
    for prompt in prompts:
        chosen = best([f for f in found if f[2] == prompt])
        out[prompt] = None if chosen is None else round(float(chosen[1]), 4)
    return out


def matching_raw(raw, letterbox):
    """The raw picture (RGB) of the same exposure as `letterbox`, or None: the raw frame letterboxed must equal the
    letterbox up to resampling (also tried with the channels swapped, in case the camera layer keeps BGR)."""
    if raw is None:
        return None
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[2] != 3 or raw.dtype != np.uint8:
        return None
    from .letterbox_geometry import letterbox_rgb_image
    target = np.asarray(letterbox, np.float32)
    for candidate in (raw, raw[..., ::-1]):
        candidate = np.ascontiguousarray(candidate)
        if float(np.abs(letterbox_rgb_image(candidate).astype(np.float32) - target).mean()) <= LETTERBOX_MATCH:
            return candidate
    return None


def demo_hold_reference(cue_report_path, demonstrations=(0, 1)):
    """Where the fish sits in the jaws, letterbox224 pixels: the mean of the held centres (reference_px) of our
    training demonstrations' cues (analysis/v21/demo_fish_cues.py report). Their pictures (picture_hw) are the camera
    content of the 224 x 224 letterbox (4:3, bars top and bottom) resized to them (analysis/v21/demo_outline_feed.py
    unbar + fit), so a point maps back as x * 224 / w, y * 168 / h + 28."""
    import io
    data = Path(cue_report_path).read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        report = json.loads(str(z['report']))
    inner = int(round(IMAGE * 3 / 4))
    top = (int(IMAGE) - inner) // 2
    points = []
    for demo in demonstrations:
        if int(demo) in (2, 3):
            raise ValueError('the hold reference comes from the training demonstrations 0 and 1 only')
        item = report[str(int(demo))]
        h, w = (float(v) for v in item['picture_hw'])
        x, y = (float(v) for v in item['reference_px'])
        points.append((x * IMAGE / w, y * inner / h + top))
    return np.mean(np.asarray(points, np.float64), axis=0)


class Sam3ObjectRelativeEncoder:
    """The wrist object encoder with SAM 3 (and, through LearnedObjectRelativeEncoder, the learned finder's base class):
    encode(wrist_letterbox224, q) -> token, jaw_pair(q), clear_observation(), describe(), last_record,
    last_policy_record, last_diagnostic_vector (the 'jaw' token, for the display and assistance cue). The jaw projection
    is the wrist calibration's (ProjectionFit on the wrist reference file). raw_frame: a callable returning the camera's raw
    wrist picture of the exposure being encoded (TrialOwner.frames['wrist'] is set before encode is called).
    score_source: SCORE_SOURCES, by default the convention's (DEFAULT_SCORE_SOURCE); 'dino_pickup' needs dino_tracker,
    the colour-era encoder's VisualTargetTracker (its frozen-feature locate only; it holds no colour)."""

    def __init__(self, client, wrist_reference, kinematics, *, convention, hold_px=None, prompts=(PROMPT,),
                 rule='best', threshold=THRESHOLD, raw_frame=None, mode='track', anchor_word=PROMPT,
                 anchor_score=ANCHOR_SCORE, stream='wrist', score_source=None, dino_tracker=None):
        from .so101_camera_calibration import ProjectionFit
        if convention not in CONVENTIONS:
            raise ValueError(f'unknown token convention {convention!r}')
        if rule not in RULES:
            raise ValueError(f'unknown rule {rule!r}')
        score_source = DEFAULT_SCORE_SOURCE[convention] if score_source is None else str(score_source)
        if score_source not in SCORE_SOURCES:
            raise ValueError(f'unknown score source {score_source!r}')
        if score_source == 'dino_pickup' and not callable(getattr(dino_tracker, 'locate_by_frozen_features', None)):
            raise ValueError("score source 'dino_pickup' needs a frozen-feature DINO tracker")
        self.score_source, self.dino_tracker = score_source, dino_tracker
        if convention == 'hold' and (hold_px is None or not np.isfinite(np.asarray(hold_px, np.float64)).all()):
            raise ValueError("the 'hold' convention needs the demonstrations' hold reference")
        prompts = [str(p) for p in prompts]
        if not prompts or any(w in COLOUR_WORDS for p in prompts for w in p.split()):
            raise ValueError('prompts must name the object without colour words')
        if mode not in MODES or (mode == 'track' and anchor_word not in prompts):
            raise ValueError("mode is 'track' (with its anchor word among the prompts) or 'detect'")
        self.client, self.convention, self.prompts, self.rule = client, convention, prompts, rule
        self.threshold, self.mode, self.anchor_word = float(threshold), mode, str(anchor_word)
        self.anchor_score, self.stream = float(anchor_score), str(stream)
        self.hold_px = None if hold_px is None else np.asarray(hold_px, np.float64).reshape(2)
        calibration = json.loads(Path(wrist_reference).read_text(encoding='utf-8'))
        fits = calibration['fits']
        row = fits['wrist'] if isinstance(fits, dict) else next(r for r in fits if r.get('view') == 'wrist')
        self.wrist_fit = ProjectionFit('wrist', tuple(row['feature_names']),
                                       np.asarray(row['coefficients'], np.float64), float(row['ridge']))
        self.kinematics = kinematics
        self.wrist_reference = str(wrist_reference)
        self.raw_frame = raw_frame
        self.tracker = None
        self.timings = []
        self.clear_observation()

    def jaw_pair(self, q):
        from .so101_camera_calibration import predict_projection
        q = np.asarray(q, np.float32)
        world, local = self.kinematics.forward_anchor_frames(q[None])
        return np.asarray(predict_projection(self.wrist_fit, world, local, np.asarray([q[-1]]))[0, 1:3], np.float64)

    def picture(self, wrist_rgb_letterbox224):
        """(picture, transform, picture source) for SAM 3: the matching raw picture when there is one."""
        letterbox = np.ascontiguousarray(wrist_rgb_letterbox224)
        raw = None
        if self.raw_frame is not None:
            try:
                raw = matching_raw(self.raw_frame(), letterbox)
            except Exception:           # a missing raw frame is ordinary; the letterbox is always there
                raw = None
        if raw is not None:
            return raw, PictureTransform(raw.shape[:2], raw=True), 'raw_same_exposure'
        return letterbox, PictureTransform(letterbox.shape[:2], raw=False), 'letterbox224'

    def find_record(self, wrist_rgb_letterbox224):
        if self.mode == 'track':
            return self.track_record(wrist_rgb_letterbox224)
        picture, transform, source = self.picture(wrist_rgb_letterbox224)
        found, info = self.client.find_many(picture, self.prompts, self.threshold)
        chosen = choose(found, self.prompts, self.rule)
        info = dict(info, picture=source, candidates=len(found), word_scores=word_scores(found, self.prompts))
        self.timings.append(info)
        del self.timings[:-256]
        self.last_found = (found, transform)
        if chosen is None:
            return None, info
        record = record_from_mask(chosen[0], chosen[1], transform)
        if record is not None:
            record.update(picture_source=source, candidates=len(found), prompt=chosen[2], rule=self.rule,
                          word_scores=info['word_scores'])
        return record, info

    def track_record(self, wrist_rgb_letterbox224):
        """The live track's mask on this picture (the finder anchors it on a detection of anchor_word at or above
        anchor_score and follows it with the SAM 3 video tracker in between), as a record with score 1 (an accepted
        outline, as the Archive outlines of fcpg/v21_outline_cue.py)."""
        picture, transform, source = self.picture(wrist_rgb_letterbox224)
        mask, info = self.client.track(picture, self.stream, self.prompts, self.threshold,
                                       anchor_word=self.anchor_word, anchor_score=self.anchor_score)
        info = dict(info, picture=source)
        self.timings.append(info)
        del self.timings[:-256]
        self.last_found = ([] if mask is None else [(mask, 1., self.anchor_word)], transform)
        if mask is None:
            return None, info
        record = record_from_mask(mask, 1., transform)
        if record is not None:
            record.update(picture_source=source, track=info.get('track'), word_scores=info.get('word_scores'),
                          prompt=self.anchor_word, mode='track')
        return record, info

    def score(self, record, info, wrist_rgb_letterbox224, pair):
        """record['score'] as the model's training wrote the token's score field (SCORE_SOURCES), and where it came
        from in record['score_source']."""
        source = self.score_source
        if source == 'dino_pickup':
            dino = self.dino_tracker.locate_by_frozen_features(
                'pickup', 'wrist', np.ascontiguousarray(wrist_rgb_letterbox224), jaw_pair_px=pair)
            dino = dino if isinstance(dino, dict) else {}
            seen = bool(dino.get('visible')) and dino.get('score') is not None
            record['dino_pickup'] = dict(visible=bool(dino.get('visible')), score=dino.get('score'))
            value, origin = (float(dino['score']), 'dino_pickup') if seen else (1., 'dino_unseen')
        elif source == 'detection':
            found = (info.get('word_scores') or {}).get(self.anchor_word)
            value, origin = ((float(found), 'detection') if found is not None and found >= self.threshold
                             else (1., 'track_only'))
        else:
            value, origin = 1., 'one'
        record['score'], record['score_source'] = value, origin

    def encode(self, wrist_rgb_letterbox224, q):
        record, info = self.find_record(wrist_rgb_letterbox224)
        pair = self.jaw_pair(q)
        if record is not None:
            self.score(record, info, wrist_rgb_letterbox224, pair)
        self.last_find = info
        self.last_record = record
        self.last_policy_record = record
        self.last_diagnostic_vector = token(record, 'jaw', jaw_pair_px=pair)
        if record is None:
            return np.zeros(DIM, np.float32)
        return token(record, self.convention, jaw_pair_px=pair, hold_px=self.hold_px)

    def encode_place(self, wrist_rgb_letterbox224, q):
        raise RuntimeError('the SAM 3 finder gives no place token; models that read one are not run with it')

    def clear_observation(self):
        """Forget this observation and, in track mode, the live track (the service calls it at HOME and when the wrist
        view is lost). A finder that cannot be reached here is reported by the next capture, not by HOME."""
        self.last_record = self.last_policy_record = self.last_find = self.last_found = None
        self.last_diagnostic_vector = np.zeros(DIM, np.float32)
        self.track_reset_error = None
        if getattr(self, 'mode', None) == 'track':
            try:
                self.client.track_reset(self.stream)
            except Exception as exc:
                self.track_reset_error = f'{type(exc).__name__}: {exc}'

    def describe(self):
        return dict(fields=['available', 'score', 'dx', 'dy', 'length', 'width', 'cos2theta', 'sin2theta'],
                    finder='sam3', prompts=list(self.prompts), rule=self.rule, threshold=self.threshold,
                    mode=self.mode, anchor_word=self.anchor_word, anchor_score=self.anchor_score,
                    convention=self.convention, hold_px=None if self.hold_px is None else self.hold_px.tolist(),
                    score_source=self.score_source,
                    wrist_reference=self.wrist_reference, coordinate_space='letterbox224/224',
                    policy_source=SOURCE, colour_used=False)


def column_pickup_record(client, column_letterbox224, raw_column=None, *, prompts=(PROMPT,), rule='best',
                         threshold=THRESHOLD, image_time_s=None):
    """The column pick-up record (targets['targets']['pickup']['column']) from SAM 3: the chosen mask's centre in
    letterbox224 pixels with source SOURCE, or a not-visible record."""
    letterbox = np.ascontiguousarray(column_letterbox224)
    raw = matching_raw(raw_column, letterbox)
    picture = letterbox if raw is None else raw
    transform = PictureTransform(picture.shape[:2], raw=raw is not None)
    found, info = client.find_many(picture, prompts, threshold)
    chosen = choose(found, prompts, rule)
    record = None if chosen is None else record_from_mask(chosen[0], chosen[1], transform, image_time_s=image_time_s)
    if record is None:
        record = dict(source=SOURCE, visible=False, reason='sam3_found_nothing', colour_used=False,
                      **({} if image_time_s is None else dict(image_time_s=float(image_time_s))))
    else:
        record.update(prompt=chosen[2])
    record.update(role='pickup', view='column', rule=rule, word_scores=word_scores(found, prompts),
                  finder=dict(info, candidates=len(found)),
                  picture_source='raw_same_exposure' if raw is not None else 'letterbox224')
    return record
