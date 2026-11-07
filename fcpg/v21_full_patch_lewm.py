"""Action-conditioned LeWM that keeps every DINO patch.

Frozen ViT-S features are 384-d, so the bank is [B, 1536, 384]: two cameras,
three frames, 256 patches. Nothing pools those patches into one vector.
Forty future queries attend to each other causally and cross-attend to the
1536 visual keys. FCPG grounding is a per-frame FiLM from body state, not a
second visual summary.
With DINOv3-L (fcpg/v21_encoder.py, 196 patches of 1024) every patch first passes
feature_in (Linear 1024 -> 384) and the bank is [B, 1176, 384]; for DINOv2 feature_in
is the identity and nothing changes.
controller_input=True (10/31, 17.153): the arm's control setting as two inputs, 0 or 1 each:
[shoulder_lift and elbow_flex at P 32 (fcpg/v21_arm_gain.py), commands from the service's
planner (five per block) instead of a LeRobot recording's command every frame]. Each adds a
learned vector (controller_weight, from 0) to every query; such a LeWM must be given them.
balanced_inputs=True (11/01, 17.160 after the 17.159 audit): every input on the scale of the others where it meets them.
The picture bank passes a LayerNorm in every layer before the picture keys and values (norm_m; prod_v20's decoder had
one, memory_norm: the v21 bank met LayerNorm'd queries of ~1 per element at ~0.28 and the picture attention stayed close
to an average); the query input also reads (active command - joints) / increment, computed in FP32 before any
autocast (the -1..1 positions alone left the difference to the model, and BF16 rounded them to up to a third of a
small command); the read-back times are read as t / TIME_SCALE and their steps as dt / STEP_SCALE (in raw seconds the
4-second queries' time input was 2-3 times the joints'). Such a LeWM has norm_m in every layer and a 20-wide query_in.
command_symlog=True (11/01, 17.161 after the 17.160 audit; with balanced_inputs only): the command distance in increments
passes DreamerV3's symlog, sign(x) log(1 + |x|) (Hafner et al. 2023): small distances stay as they are, large ones
shrink (20 increments -> 3.0), because on the 4-second demonstration queries the plain distance reached L2 35 (p95 80),
ten times the joints'. Weights of the same shapes: the LeWM keeps a buffer command_symlog that the loaders find.
Attention supervision (11/01, 17.163; James 「訓練的時候用sam3分割的，為什麼沒有加attention的懲罰呢」): forward(...,
attention_supervision=(heads, weights)) also returns, per layer, how much picture attention the first `heads` heads give
the labelled patches, sum over keys of p(key) x weight (weights [B, keys] for every head or [B, heads, keys] one per
head, 0-1: the SAM 3.1 coverage of the object or the robot), with the scores _attend's scaled_dot_product_attention
uses (the same projections and the padding),
in FP32 and without the attention dropout; the trainer turns it into a penalty (fcpg/v21_sam_labels.py). Nothing else
changes and inference never passes it. label_head=True (17.163): one more linear readout of the final hidden state,
2 cameras x (object, robot) x 196 patch logits, for the trainer's future-label prediction; made after every other
weight from zeros (no RNG draw); unused at inference, and the loaders find it (architecture_from_state).
patch_encoder_layers=N (11/01, 17.164; James 「dino是凍結的」「你不要陷入誤區」「推理不用sam」): the LeWM's own picture
encoder. With DINO frozen, the only trainable picture path was a linear map and FiLM per patch; patches never read each
other, nor the other camera, nor the other two pictures. N pre-LN self-attention layers (_PatchEncoderLayer) now run over
every readable token of the bank (both cameras, all three pictures, after FiLM) before the query layers read it, trained
by the same loss. Their residual outputs start at zero (a new encoder passes the bank unchanged), they are built last on
a copy of the RNG state (every other weight and every later draw are the LeWM's without them), a row without a picture
keeps its one zero token, and the loaders find them (architecture_from_state).
memory_pictures=N, memory_tokens=K (11/01, 17.166; James 「3張我覺得不太足夠，萬一是30張呢」「直接30張」「不會抄人家的方案馬」):
the last N pictures each camera took instead of three (fcpg/v21_picture_memory.py PictureMemory: right-aligned per
camera, oldest -> newest, slots before a camera's first picture empty). As the established long-memory designs do
(RT-1: TokenLearner, 8 tokens per image; NativeMEM / PAM: learned queries pool each frame into one or two cached
tokens; SlowFast: the latest frames in detail, the many others compact): every picture is pooled by K learned queries
into K tokens from what was known when it was taken (its patches, their positions, its camera, the joints at the
exposure), so a picture's tokens are made once; at each forecast they are moved by how long ago it was taken and how
far the joints went since (memory_film, the same 19 inputs as the per-frame FiLM); each camera's newest picture also
keeps all its patches (the per-frame path, its FiLM). The queries read both (a learned kind vector tells them apart).
The loss is unchanged; the frame-slot vectors are unused (time is
read from the capture times only, whatever the cadence). When every row of a call is one memory expanded (the
planner's candidates: stride 0 along the batch), the bank is made once and every row reads it.
"""
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .prod_v20_policy import DEFAULT_LOWER, DEFAULT_UPPER, _PositionModule
from .v21_checks import checking, fail_if
from .v21_memory import finite, finite_where, clock, pose_position_basis
from .v21_models import ARMOutput
from .v21_command_memory import COMMAND_MODES, LEGACY_COMMANDS, FULL_COMMANDS, CausalCommandRead
from .v21_camera_types import LEGACY_SLOTS, REVIEWED_TYPES, validate_camera_types
from .v21_encoder import DIM as FEATURE_DIM, PATCHES

DIM = 384          # default width; 17.106: wider / deeper LeWMs pass width= and layers=
CONTROLLER_INPUTS = ('arm_p32', 'planner')     # 17.153: controller inputs, in this order
HEADS = 8
LAYERS = 4
QUERY_INPUTS, BALANCED_QUERY_INPUTS = 14, 20     # 17.160: + (active command - joints) / increment
TIME_SCALE = 4.     # 17.160: s, the 40-step forecast (one fixed scale: the robot's 0.5 s and the demonstrations' 4 s
                    # keep their real lengths)
STEP_SCALE = .1     # 17.160: s, one read-back step
POSITION_CODE = 32  # concat_inputs: the width of a patch's learned position code and of a camera's code
NUMBERS = 19        # concat_inputs: a picture's numbers, read with each of its patches (joints as taken, joints now, their
                    # change, log age)
ALIGN_ZEROS = 5     # concat_inputs: zeros after the numbers, so the bank's width is a multiple of 8 and the GPU's
                    # tensor cores project it (James 11/02 「对齐阿」: 1107 wide took 1.8 times as long on the A100)
BANK_INPUTS = FEATURE_DIM + 2 * POSITION_CODE + NUMBERS + ALIGN_ZEROS  # concat_inputs: a patch as the query layers
                    # read it: its DINO features (normalized once, no weights), its position and camera codes, its
                    # picture's numbers and the zeros
POSE_POSITION = 72  # pose_position (11/03, James 「AOK阿」): each patch's position basis (x, y, sin and cos of pi x and
                    # pi y: 6) times its picture's joints as taken and their change since (12), joined to the bank: the
                    # numbers alone add the same amount to every patch of a picture in a key (one linear layer), so the
                    # joints as taken could weigh whole pictures, never which patch of one; 1112 + 72 = 1184, still a
                    # multiple of 8


@dataclass(frozen=True)
class SupervisedARMOutput(ARMOutput):
    """17.163: the forecast of a call with attention supervision and the supervised heads' picture-attention mass on
    the labelled patches, [layers, B, heads, queries] (a subclass: ARMOutput and its subclasses keep their fields)."""
    attention_mass: torch.Tensor | None = None


def architecture_from_state(state, prefix=''):
    """{width, layers} of a full-patch LeWM from its weights (a checkpoint's model state; prefix 'lewm.' for a whole
    core). Old checkpoints give the defaults, so every loader can build the model the weights were trained in."""
    query = state.get(f'{prefix}query')
    width = int(query.shape[-1]) if query is not None else DIM
    marks = {key[len(prefix) + len('layers.'):].split('.', 1)[0] for key in state if key.startswith(f'{prefix}layers.')}
    layers = 1 + max(int(mark) for mark in marks) if marks else LAYERS
    found = dict(width=width, layers=layers)
    if f'{prefix}controller_weight' in state:  # 17.153: a LeWM that reads the arm's control setting
        found['controller_input'] = True
    if f'{prefix}layers.0.norm_m.weight' in state:     # 17.160: a LeWM whose inputs are balanced
        found['balanced_inputs'] = True
    if f'{prefix}command_symlog' in state:              # 17.161: its command distance through symlog
        found['command_symlog'] = True
    if f'{prefix}label_head.weight' in state:           # 17.163: a LeWM trained with the future-label readout
        found['label_head'] = True
    encoder = {key[len(prefix) + len('patch_encoder.'):].split('.', 1)[0] for key in state
               if key.startswith(f'{prefix}patch_encoder.')}
    if encoder:                                         # 17.164: the LeWM's own picture encoder
        found['patch_encoder_layers'] = 1 + max(int(mark) for mark in encoder)
    if f'{prefix}memory_pictures' in state:             # 17.166: the memory of the last N pictures per camera
        found['memory_pictures'] = int(state[f'{prefix}memory_pictures'])
        if f'{prefix}memory_queries' in state:          # its pool (a concatenating LeWM has none)
            found['memory_tokens'] = int(state[f'{prefix}memory_queries'].shape[0])
    if f'{prefix}concat_inputs' in state:               # 11/02: concatenated inputs (balanced, with the control setting)
        found.update(concat_inputs=True, controller_input=True, balanced_inputs=True)
    if f'{prefix}pose_position' in state:               # 11/03: the bank also joins patch position x joints products
        found['pose_position'] = True
    return found


def _autocasting():
    return torch.is_autocast_enabled('cuda') if hasattr(torch, 'get_autocast_dtype') else torch.is_autocast_enabled()


def _autocast_dtype():
    return torch.get_autocast_dtype('cuda') if hasattr(torch, 'get_autocast_dtype') else torch.get_autocast_gpu_dtype()


def _attend(attention, query, memory, *, causal=False, readable=None, training=False):
    """17.106: an nn.MultiheadAttention's function on its own weights, without the copies its batch-first path makes of
    the projected picture keys (four or more of [B, 1176, width] per layer and per pass): q, k and v are projected
    separately and read as [B, heads, tokens, width / heads] views by scaled_dot_product_attention, the self-attention
    is causal by is_causal, readable (bool, True = may be read) replaces the key padding mask, and the attention weights
    get the module's dropout. The same values as the module up to rounding (tests/test_v21_full_patch_lewm_size.py);
    at width 768 with 8 layers and batch 128 the query layers train in 196 ms instead of 309 (10/26, A100)."""
    width = attention.embed_dim
    weight, bias = attention.in_proj_weight, attention.in_proj_bias
    if attention._qkv_same_embed_dim:
        weights = [weight[i * width:(i + 1) * width] for i in range(3)]
    else:                   # concat_inputs: keys and values of a wider input (kdim / vdim), each its own projection
        weights = [attention.q_proj_weight, attention.k_proj_weight, attention.v_proj_weight]
    q, k, v = (F.linear(x, w, bias[i * width:(i + 1) * width])
               .unflatten(-1, (attention.num_heads, width // attention.num_heads)).transpose(1, 2)
               for i, (x, w) in enumerate(zip((query, memory, memory), weights)))
    mask = readable
    if (training and mask is not None and not causal and not attention._qkv_same_embed_dim and k.shape[0] == q.shape[0]
            and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and _flash_capable(q.device)):
        out = _flash_read(q, k, v, mask, attention.dropout, (width // attention.num_heads) ** -.5)
        return attention.out_proj(out.transpose(1, 2).flatten(2))
    if k.shape[0] == 1 and q.shape[0] > 1:     # 17.166: one bank shared by every row (views, nothing copied)
        k, v = k.expand(q.shape[0], -1, -1, -1), v.expand(q.shape[0], -1, -1, -1)
        if mask is not None:
            mask = mask.expand(q.shape[0], *mask.shape[1:])
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal,
                                         dropout_p=attention.dropout if training else 0.)
    return attention.out_proj(out.transpose(1, 2).flatten(2))


@lru_cache(maxsize=None)
def _flash_capable(device):
    """Whether FlashAttention runs on this GPU (the A100 does, the robot's V100 does not): one tiny call tells."""
    try:
        x = torch.zeros(1, 1, 1, 8, device=device, dtype=torch.bfloat16)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            F.scaled_dot_product_attention(x, x, x)
        return True
    except RuntimeError:
        return False


def _flash_read(q, k, v, readable, dropout_p, scale):
    """The training read of a concatenating LeWM's bank by FlashAttention, which takes no key padding mask (James 11/02
    「用那个FlashAttention加速也可以阿」: the masked kernel's backward through the 11,760 patches took 62 ms a layer on the
    A100, this 13): one more channel, 1 on the queries and 0 (readable) or -30000 (padding) on the keys, gives every
    padding key exp(-30000 x scale) = 0, as the mask does, and zeros take the heads to a multiple of 8 channels; the
    same values up to rounding (tests/test_v21_concat_inputs.py). q [B, heads, queries, head], k and v [B, heads, keys,
    head], readable [B, 1, 1, keys] (True = may be read, at least one per row) -> [B, heads, queries, head]."""
    head = q.shape[-1]
    pad = (head + 8) // 8 * 8 - head
    shut = torch.where(readable, 0., -30000.).to(k.dtype).transpose(-1, -2).expand(-1, k.shape[1], -1, -1)
    q = torch.cat((q, q.new_ones(*q.shape[:-1], 1), q.new_zeros(*q.shape[:-1], pad - 1)), -1)
    k = torch.cat((k, shut, k.new_zeros(*k.shape[:-1], pad - 1)), -1)
    v = torch.cat((v, v.new_zeros(*v.shape[:-1], pad)), -1)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=scale)[..., :head]


def _attention_mass(attention, query, memory, readable, heads, weights):
    """17.163: the picture attention the first `heads` heads of a cross-attention give weights (0-1 per key; [B, keys]
    for every head or [B, heads, keys] one per head): sum over keys of p(key) x weight -> [B, heads, queries]. p is
    _attend's: the same q / k projections (only these heads' rows of in_proj), / sqrt(head width), the unreadable keys
    shut; in FP32, no dropout."""
    width, count = attention.embed_dim, attention.num_heads
    size = width // count
    if not 1 <= heads <= count:
        raise ValueError(f'attention supervision takes 1-{count} heads')
    weight, bias = attention.in_proj_weight, attention.in_proj_bias
    q = F.linear(query, weight[:heads * size], bias[:heads * size]).float()
    k = F.linear(memory, weight[width:width + heads * size], bias[width:width + heads * size]).float()
    q = q.unflatten(-1, (heads, size)).transpose(1, 2)
    k = k.unflatten(-1, (heads, size)).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) / size ** .5
    p = scores.masked_fill(~readable[:, None, None], float('-inf')).softmax(-1)
    weights = weights.float()[:, None, None] if weights.ndim == 2 else weights.float()[:, :, None]
    return (p * weights).sum(-1)


def _one_readable_token(bank, padding):
    """A row whose keys are all padding keeps its first key readable and every key zero: one neutral token. The rows
    are set without asking whether any is empty (17.171: asking made the CPU wait for the GPU); the same values."""
    empty = padding.all(dim=1)
    padding = padding.clone()
    padding[:, 0] &= ~empty
    return torch.where(empty[:, None, None], 0., bank), padding


class _QueryLayer(nn.Module):
    def __init__(self, dim=DIM, heads=HEADS, direct=False, memory_norm=False, memory_dim=None):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.norm_c = nn.LayerNorm(dim)
        # memory_dim (concat_inputs): the bank's tokens are [token, numbers], wider than the queries, and the keys and
        # values project them as they are
        wide = {} if memory_dim is None else dict(kdim=memory_dim, vdim=memory_dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True, **wide)
        self.norm_f = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.direct = direct        # 17.106: _attend() on the same weights instead of the modules' own forward
        if memory_norm:             # 17.160: the picture bank on the queries' scale before its keys and values
            self.norm_m = nn.LayerNorm(dim if memory_dim is None else memory_dim)

    def forward(self, query, memory, causal, memory_pad, supervise=None):
        """supervise (17.163): (heads, weights [B, keys]) -> (the query, the heads' attention mass [B, heads, queries])."""
        qn = self.norm_q(query)
        if self.direct:             # causal is always the full lower triangle here (V21FullPatchLeWM.forward)
            query = query + _attend(self.self_attn, qn, qn, causal=True, training=self.training)
        else:
            query = query + self.self_attn(qn, qn, qn, attn_mask=causal, need_weights=False)[0]
        if hasattr(self, 'norm_m'):
            memory = self.norm_m(memory)
        qn = self.norm_c(query)
        mass = None
        if self.direct:
            if supervise is not None:
                mass = _attention_mass(self.cross_attn, qn, memory, ~memory_pad, *supervise)
            query = query + _attend(self.cross_attn, qn, memory, readable=~memory_pad[:, None, None],
                                    training=self.training)
        else:
            if supervise is not None:
                raise ValueError('attention supervision reads a directly attending LeWM only')
            query = query + self.cross_attn(
                qn, memory, memory, key_padding_mask=memory_pad, need_weights=False)[0]
        hidden = self.ff[1](self.ff[0](self.norm_f(query)))
        hidden = F.dropout(hidden, 0.1, self.training)
        hidden = F.dropout(self.ff[2](hidden), 0.1, self.training)
        return query + hidden if supervise is None else (query + hidden, mass)


class _PatchEncoderLayer(nn.Module):
    """17.164: one pre-LN self-attention layer of the picture bank: every token reads every readable token (both
    cameras, all three pictures), then a feed-forward layer, the dropout of _QueryLayer. The attention's and the
    feed-forward's outputs start at zero, so a new layer passes the bank unchanged."""

    def __init__(self, dim=DIM, heads=HEADS):
        super().__init__()
        self.norm_a = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.norm_f = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        with torch.no_grad():
            for weight in (self.attn.out_proj.weight, self.attn.out_proj.bias, self.ff[2].weight, self.ff[2].bias):
                weight.zero_()

    def forward(self, bank, readable):
        """bank [B, keys, dim]; readable [B, keys], True = a token of a picture that was taken."""
        normed = self.norm_a(bank)
        bank = bank + _attend(self.attn, normed, normed, readable=readable[:, None, None], training=self.training)
        hidden = F.dropout(self.ff[1](self.ff[0](self.norm_f(bank))), 0.1, self.training)
        return bank + F.dropout(self.ff[2](hidden), 0.1, self.training)


class _PicturePool(nn.Module):
    """17.166: K learned queries read one picture's patch tokens (pre-LN cross-attention, then a feed-forward layer, both
    residual on the queries; the query layers' dropout): its K memory tokens. Pictures are independent of each other."""

    def __init__(self, dim=DIM, heads=HEADS, tokens=8):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_p = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.norm_f = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, queries, patches):
        """queries [K, dim] (learned); patches [M, PATCHES, dim], one picture per row -> [M, K, dim]."""
        query = queries[None].expand(len(patches), -1, -1)
        query = query + _attend(self.attn, self.norm_q(query), self.norm_p(patches), training=self.training)
        hidden = F.dropout(self.ff[1](self.ff[0](self.norm_f(query))), 0.1, self.training)
        return query + F.dropout(self.ff[2](hidden), 0.1, self.training)


class V21FullPatchLeWM(_PositionModule):
    """40 action/time queries read an uncompressed 1536-patch visual bank."""

    reads_visual_memory = True

    def __init__(self, *, increment_scale=None, horizon=40, command_conditioning=LEGACY_COMMANDS, width=DIM,
                 layers=LAYERS, controller_input=False, balanced_inputs=False,
                 command_symlog=False, label_head=False, patch_encoder_layers=0, memory_pictures=0, memory_tokens=8,
                 concat_inputs=False, pose_position=False):
        super().__init__(DEFAULT_LOWER, DEFAULT_UPPER, increment_scale)
        if horizon != 40:
            raise ValueError('full-patch LeWM is the 40-step model')
        if (isinstance(width, bool) or not isinstance(width, int) or width < HEADS or width % HEADS
                or isinstance(layers, bool) or not isinstance(layers, int) or layers < 1):
            raise ValueError(f'full-patch LeWM width must be a positive multiple of {HEADS}, layers at least 1')
        # concat_inputs (James 11/02 「能concat的就concat 然后改代码」「能合并的就合并网络 不要把linear放在外面」「不用30张
        # 压缩后的 直接都给lewm」, a picture-memory LeWM with the balanced and the control-setting inputs only; Fig. 1 c):
        # inputs are concatenated and go as they are into the network that reads them, in place of the added position,
        # camera and kind vectors, the scale-and-shift (FiLM), the pool and the added step query, history and control
        # setting: the query layers read every patch of every picture of the memory as [DINO features, position code,
        # camera code, its picture's 19 numbers] (no pooling) and start each step from [learned step query, step inputs,
        # recent motion, control setting] (their input layer, query_in). See _concat_memory_bank and forward. The
        # default builds every earlier LeWM unchanged.
        if type(concat_inputs) is not bool or (concat_inputs and not (memory_pictures and controller_input
                                                                     and balanced_inputs is True)):
            raise ValueError('concat_inputs is an explicit boolean for a picture-memory LeWM with the balanced and the '
                             'control-setting inputs')
        if type(pose_position) is not bool or (pose_position and not concat_inputs):
            raise ValueError('pose_position is an explicit boolean for a concatenating LeWM (concat_inputs)')
        bank_inputs = BANK_INPUTS + (POSE_POSITION if pose_position else 0)
        self.horizon = horizon
        self.width, self.layer_count, self.heads = width, layers, HEADS
        if not concat_inputs:
            self.feature_in = nn.Identity() if FEATURE_DIM == width else nn.Linear(FEATURE_DIM, width)
            self.patch_xy = nn.Parameter(torch.randn(1, 1, 1, PATCHES, width) * .02)
            self.camera = nn.Parameter(torch.randn(1, 2, 1, 1, width) * .02)
            self.frame = nn.Parameter(torch.randn(1, 1, 3, 1, width) * .02)
            self.film = nn.Sequential(nn.Linear(19, width), nn.GELU(), nn.Linear(width, width * 2))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        if concat_inputs:     # the layers' input: [learned step query, step inputs, recent motion, control setting]
            self.query_in = nn.Linear(width + BALANCED_QUERY_INPUTS + 303 + len(CONTROLLER_INPUTS), width)
        else:
            self.history = nn.Sequential(nn.Linear(303, width), nn.GELU(), nn.Linear(width, width))
            self.query_in = nn.Linear(BALANCED_QUERY_INPUTS if balanced_inputs is True else QUERY_INPUTS, width)
        self.query = nn.Parameter(torch.randn(1, horizon, width) * .02)
        # 17.106: a resized LeWM attends directly (_attend); the deployed default size keeps the modules' own path,
        # and a LeWM that reads the control setting or balanced inputs attends directly at every size, as it was trained
        if not all(isinstance(flag, bool) for flag in (controller_input, balanced_inputs, command_symlog, label_head)):
            raise ValueError('controller_input, balanced_inputs, command_symlog and label_head must be explicit '
                             'booleans')
        if command_symlog and not balanced_inputs:
            raise ValueError('command_symlog shapes the command distance of a balanced_inputs LeWM')
        if (isinstance(patch_encoder_layers, bool) or not isinstance(patch_encoder_layers, int)
                or not 0 <= patch_encoder_layers <= 8):
            raise ValueError('patch_encoder_layers is a whole number from 0 to 8')
        whole = lambda x, low, high: not isinstance(x, bool) and isinstance(x, int) and low <= x <= high
        if not whole(memory_pictures, 0, 64) or not whole(memory_tokens, 1, 32):
            raise ValueError('memory_pictures is a whole number from 0 to 64, memory_tokens from 1 to 32')
        if memory_pictures and (label_head or patch_encoder_layers):
            raise ValueError('a picture-memory LeWM has no label head or patch encoder')
        self.balanced_inputs = balanced_inputs
        if command_symlog:          # 17.161: a marker in the weights (the shapes are the same), no RNG draw
            self.register_buffer('command_symlog', torch.ones(()))
        direct = ((width, layers) != (DIM, LAYERS) or controller_input or balanced_inputs
                  or bool(memory_pictures))
        # concat_inputs: no per-layer bank norm (its scale and shift would be the keys' and values' own projection);
        # the DINO features are normalized once in _concat_memory_bank, as FCPG's reader does
        self.layers = nn.ModuleList(_QueryLayer(width, HEADS, direct, memory_norm=balanced_inputs and not concat_inputs,
                                                memory_dim=bank_inputs if concat_inputs else None)
                                    for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.response = nn.Linear(width, 6)
        self.scale = nn.Linear(width, 6)
        nn.init.normal_(self.response.weight, std=.005)
        nn.init.zeros_(self.response.bias)
        self.command_conditioning = LEGACY_COMMANDS
        self.enable_command_conditioning(command_conditioning)
        self.camera_conditioning = LEGACY_SLOTS
        self.controller_input = controller_input
        if controller_input and not concat_inputs:
            # 17.153: created after every other weight and without drawing from the RNG (the same seed gives the same
            # LeWM), from 0
            self.controller_weight = nn.Parameter(self.query_in.weight.new_zeros(len(CONTROLLER_INPUTS), width))
        if label_head:
            # 17.163: the same way; logits of 2 cameras x (object, robot) x PATCHES from the final hidden state
            self.label_head = torch.nn.utils.skip_init(nn.Linear, width, 4 * PATCHES).to(self.query_in.weight)
            with torch.no_grad():
                self.label_head.weight.zero_()
                self.label_head.bias.zero_()
        if patch_encoder_layers:
            # 17.164: last, on a copy of the RNG state: the global stream is the same as without the encoder afterwards
            with torch.random.fork_rng(devices=[]):
                self.patch_encoder = nn.ModuleList(_PatchEncoderLayer(width, HEADS)
                                                   for _ in range(patch_encoder_layers))
            self.patch_encoder.to(self.query_in.weight)
        if memory_pictures and concat_inputs:
            # every patch of every picture: [DINO features, position code, camera code, its picture's numbers]
            # (_concat_memory_bank); no pool, so memory_tokens plays no part
            self.register_buffer('memory_pictures', torch.tensor(memory_pictures))
            self.register_buffer('concat_inputs', torch.ones(()))
            if pose_position:       # a marker in the weights, as concat_inputs (the keys' widths say it as well)
                self.register_buffer('pose_position', torch.ones(()))
            self.memory_position = nn.Parameter(torch.randn(PATCHES, POSITION_CODE) * .02)
            self.memory_camera = nn.Parameter(torch.randn(2, POSITION_CODE) * .02)
        elif memory_pictures:
            # 17.166: last; the pool's queries, its layer, the per-forecast FiLM (from 0) and the two kind vectors
            self.register_buffer('memory_pictures', torch.tensor(memory_pictures))
            self.memory_queries = nn.Parameter(torch.randn(memory_tokens, width) * .02)
            self.memory_pool = _PicturePool(width, HEADS, memory_tokens)
            self.memory_film = nn.Sequential(nn.Linear(19, width), nn.GELU(), nn.Linear(width, width * 2))
            nn.init.zeros_(self.memory_film[-1].weight)
            nn.init.zeros_(self.memory_film[-1].bias)
            self.memory_kind = nn.Parameter(torch.randn(2, width) * .02)    # [newest patches, pooled tokens]
            for module in (self.memory_pool, self.memory_film):
                module.to(self.query_in.weight)

    def architecture(self):
        found = dict(width=self.width, layers=self.layer_count, heads=self.heads)
        if self.controller_input:
            found['controller_input'] = True
        if getattr(self, 'balanced_inputs', False):
            found['balanced_inputs'] = True
        if hasattr(self, 'command_symlog'):
            found['command_symlog'] = True
        if hasattr(self, 'label_head'):
            found['label_head'] = True
        if hasattr(self, 'patch_encoder'):
            found['patch_encoder_layers'] = len(self.patch_encoder)
        if hasattr(self, 'memory_pictures'):
            found['memory_pictures'] = int(self.memory_pictures)
            if hasattr(self, 'memory_queries'):     # a concatenating LeWM has no pool
                found['memory_tokens'] = int(self.memory_queries.shape[0])
        if hasattr(self, 'concat_inputs'):
            found['concat_inputs'] = True
        if hasattr(self, 'pose_position'):
            found['pose_position'] = True
        return found

    def enable_camera_conditioning(self, mode):
        if mode not in (LEGACY_SLOTS, REVIEWED_TYPES):
            raise ValueError('unsupported camera conditioning')
        if mode == self.camera_conditioning:
            return
        if self.camera_conditioning != LEGACY_SLOTS:
            raise ValueError('cannot discard a trained camera type branch')
        # No RNG consumption or change to old slot embeddings/scene targets.
        # Unknown has NO learned parameter, avoiding an unknown-domain token.
        self.camera_type_embedding = nn.Parameter(self.query_in.weight.new_zeros(2, self.width))
        self.camera_conditioning = mode

    def enable_command_conditioning(self, mode):
        if mode not in COMMAND_MODES:
            raise ValueError('unsupported LeWM command conditioning')
        if mode == self.command_conditioning:
            return
        if self.command_conditioning != LEGACY_COMMANDS:
            raise ValueError('cannot discard a trained command branch')
        self.command_projection = nn.Sequential(nn.Linear(20, self.width), nn.GELU(), nn.Linear(self.width, self.width))
        self.command_readers = nn.ModuleList(CausalCommandRead(self.width, self.heads) for _ in range(self.layer_count))
        self.command_projection.to(self.query_in.weight)
        self.command_readers.to(self.query_in.weight)
        self.command_conditioning = mode

    def _command_bank(self, trace, q, query_offsets):
        # Each token sees only this write and earlier writes. In particular,
        # do not encode duration UNTIL the next write (a future dependency).
        valid = trace.valid
        commands = torch.where(valid[..., None], trace.positions, 0.).to(q)
        times = torch.where(valid, trace.offsets, 0.)
        increments = torch.diff(torch.cat((q[:, None], commands), 1), dim=1)
        preceding_dt = torch.diff(torch.cat((times.new_zeros(len(q), 1), times), 1), dim=1).clamp_min(0)
        features = torch.cat((self.normalize_position(commands), (commands-q[:, None])/self.increment_scale,
            increments/self.increment_scale, torch.log1p(times).to(q)[..., None],
            torch.log1p(preceding_dt).to(q)[..., None]), -1)
        bank = self.command_projection(torch.where(valid[..., None], features, 0.))
        eligible = valid[:, None] & (times[:, None] < query_offsets[:, :, None] - 1e-9)
        # Some legal queries precede the first write. A neutral extra token is
        # visible ONLY to those queries, never an arbitrary unmasked command.
        bank = torch.cat((bank, bank.new_zeros(len(q), 1, self.width)), 1)
        eligible = torch.cat((eligible, ~eligible.any(-1, keepdim=True)), -1)
        forbidden = (~eligible)[:, None].expand(-1, self.heads, -1, -1)
        return bank, forbidden.reshape(len(q)*self.heads, query_offsets.shape[1], bank.shape[1])

    def _visual_bank(self, patches, patch_valid, q_now, memory, now_time):
        b = patches.shape[0]
        if patches.shape != (b, 2, 3, PATCHES, FEATURE_DIM):
            raise ValueError(f'visual bank must be finite DINO patches [B,2,3,{PATCHES},{FEATURE_DIM}]')
        if checking():
            fail_if(~torch.isfinite(patches).all(), f'visual bank must be finite DINO patches [B,2,3,{PATCHES},{FEATURE_DIM}]')
        if patch_valid is None:
            patch_valid = torch.ones(b, 2, 3, dtype=torch.bool, device=patches.device)
        elif patch_valid.shape != (b, 2, 3) or patch_valid.dtype != torch.bool:
            raise ValueError('patch valid mask must be bool [B,2,3]')
        # 17.173: under autocast the linear feature_in reads its input in the autocast type, so the pictures go to it in
        # that type at once (float16 -> bfloat16 rounds as float16 -> float32 -> bfloat16 does), without a float32 copy
        to = (_autocast_dtype() if _autocasting() and isinstance(self.feature_in, nn.Linear)
              else self.query_in.weight.dtype)
        tokens = self.feature_in(patches.to(dtype=to, device=self.query_in.weight.device))
        tokens = tokens + self.patch_xy + self.camera + self.frame
        if self.camera_conditioning == REVIEWED_TYPES:
            types = validate_camera_types(getattr(memory, 'camera_types', None), b).to(tokens.device)
            table = torch.cat((tokens.new_zeros(1, self.width), self.camera_type_embedding), 0)
            tokens = tokens + table[types][:, :, None, None]
        if memory is None or now_time is None:
            q_cap = torch.zeros(b, 2, 3, 6, device=tokens.device, dtype=tokens.dtype)
            age = torch.zeros(b, 2, 3, device=tokens.device, dtype=tokens.dtype)
        else:
            q_cap = memory.q_at_image.to(device=tokens.device, dtype=tokens.dtype)
            if q_cap.shape != (b, 2, 3, 6):
                raise ValueError('capture q must be [B,2,3,6]')
            age = (now_time.to(device=tokens.device, dtype=torch.float64)[:, None, None]
                   - memory.image_times.to(device=tokens.device, dtype=torch.float64)).clamp(0, 30).to(tokens.dtype)
        q_now_n = self.normalize_position(q_now.to(tokens))
        q_cap_n = self.normalize_position(q_cap)
        film_in = torch.cat((
            q_cap_n,
            q_now_n[:, None, None, :].expand(-1, 2, 3, -1),
            q_now_n[:, None, None, :] - q_cap_n,
            torch.log1p(age)[..., None]), -1)
        gamma, beta = self.film(film_in).chunk(2, dim=-1)
        tokens = (1 + gamma)[..., None, :] * tokens + beta[..., None, :]
        bank = tokens.reshape(b, 6 * PATCHES, self.width)
        padding = ~patch_valid.to(device=bank.device).reshape(b, 6).repeat_interleave(PATCHES, dim=1)
        return _one_readable_token(bank, padding)

    def _memory_bank(self, patches, patch_valid, q_now, memory, now_time):
        """17.166: the bank of a picture-memory LeWM, [B, 2 x PATCHES + 2 x N x K, width], and its padding: each
        camera's newest picture in all its patches (moved by the per-frame FiLM), then the K pooled tokens of every
        picture (pooled from its patches, positions and camera only; moved by memory_film to this forecast)."""
        b, n = patches.shape[0], int(self.memory_pictures)
        if patches.shape != (b, 2, n, PATCHES, FEATURE_DIM):
            raise ValueError(f'picture memory must be DINO patches [B,2,{n},{PATCHES},{FEATURE_DIM}]')
        if patch_valid is None or patch_valid.shape != (b, 2, n) or patch_valid.dtype != torch.bool:
            raise ValueError(f'picture memory valid mask must be bool [B,2,{n}]')
        if memory is None or now_time is None:
            raise ValueError('a picture-memory LeWM reads the capture times and joints (memory) and now_time')
        weight = self.query_in.weight
        valid = patch_valid.to(weight.device)
        if checking():
            fail_if((valid[..., :-1] & ~valid[..., 1:]).any(),
                    "picture memory is right-aligned: a camera's pictures fill its newest slots")
        q_cap = memory.q_at_image.to(device=weight.device, dtype=weight.dtype)
        times = memory.image_times.to(device=weight.device, dtype=torch.float64)
        if q_cap.shape != (b, 2, n, 6) or times.shape != (b, 2, n):
            raise ValueError(f'picture memory capture q / times must be [B,2,{n},6] / [B,2,{n}]')
        finite_where('picture memory capture q', q_cap, valid)
        finite_where('picture memory capture times', times, valid)
        age = now_time.to(device=weight.device, dtype=torch.float64)[:, None, None] - times
        if checking():
            fail_if(((age < -1e-6) & valid).any(), 'a picture of the memory was taken after the forecast starts')
        age = age.clamp(0, 120)
        q_cap_n, q_now_n = self.normalize_position(q_cap), self.normalize_position(q_now.to(weight))
        film_in = torch.cat((q_cap_n, q_now_n[:, None, None].expand(-1, 2, n, -1), q_now_n[:, None, None] - q_cap_n,
                             torch.log1p(age).to(weight.dtype)[..., None]), -1)                      # [B, 2, N, 19]
        if hasattr(self, 'concat_inputs'):
            return self._concat_memory_bank(patches, valid, film_in, b, n)
        # every picture taken: K tokens, pooled from what was known when it was taken
        k = int(self.memory_queries.shape[0])
        taken = valid.reshape(-1).nonzero().squeeze(1)
        picked = patches.reshape(-1, PATCHES, FEATURE_DIM).index_select(0, taken.to(patches.device))
        if checking():
            fail_if(~torch.isfinite(picked).all(), 'picture memory: a taken picture has nonfinite patches')
        camera = (torch.arange(b * 2 * n, device=weight.device) // n % 2).index_select(0, taken)
        tokens_k = weight.new_zeros(b * 2 * n, k, self.width)
        if len(taken):
            tokens = self.feature_in(picked.to(device=weight.device, dtype=weight.dtype))
            tokens = tokens + self.patch_xy[0, 0] + self.camera[0, :, 0].index_select(0, camera)  # [taken, P, w]
            pooled = self.memory_pool(self.memory_queries, tokens)                                  # [taken, K, w]
            tokens_k = tokens_k.to(pooled.dtype).index_copy(0, taken, pooled)
        tokens_k = tokens_k.reshape(b, 2, n, k, -1)
        gamma, beta = self.memory_film(film_in).chunk(2, dim=-1)
        tokens_k = (1 + gamma)[..., None, :] * tokens_k + beta[..., None, :] + self.memory_kind[1]
        # each camera's newest picture in all its patches, as the per-frame path reads a picture
        newest = self.feature_in(patches[:, :, -1].to(device=weight.device, dtype=weight.dtype))
        newest = newest + self.patch_xy[:, 0] + self.camera[:, :, 0]
        gamma, beta = self.film(film_in[:, :, -1]).chunk(2, dim=-1)
        newest = (1 + gamma)[..., None, :] * newest + beta[..., None, :] + self.memory_kind[0]
        bank = torch.cat((newest.reshape(b, 2 * PATCHES, -1), tokens_k.reshape(b, 2 * n * k, -1)), 1)
        padding = torch.cat((~valid[:, :, -1].repeat_interleave(PATCHES, dim=1),
                             ~valid.reshape(b, 2 * n).repeat_interleave(k, dim=1)), 1)
        return _one_readable_token(bank, padding)      # as _visual_bank

    def _concat_memory_bank(self, patches, valid, condition, b, n):
        """concat_inputs: every patch of every picture of the memory, as the query layers read it (no pooling, no
        projection of its own): [its DINO features normalized (no weights, as FCPG's reader), its position code, its
        camera code, the 19 numbers of `condition` for its picture (joints as taken, joints now, their change, log
        age), ALIGN_ZEROS zeros] -> [B, 2 x N x PATCHES, BANK_INPUTS], made in autocast's dtype when autocasting
        (2 x 30 x 196 patches: one copy, which every layer's keys and values read), and its padding (the patches of a
        slot without a picture); a row without any picture keeps one zero token."""
        weight = self.query_in.weight
        dtype = _autocast_dtype() if weight.is_cuda and _autocasting() else weight.dtype
        features = F.layer_norm(patches.to(device=weight.device, dtype=torch.float32), (FEATURE_DIM,)).to(dtype)
        codes = torch.cat((self.memory_position[None, None].expand(2, n, -1, -1),
                           self.memory_camera[:, None, None].expand(-1, n, PATCHES, -1)), -1)   # [2, N, P, 64]
        numbers = F.pad(condition, (0, ALIGN_ZEROS))                                           # [B, 2, N, 24]
        bank = torch.cat((features, codes[None].expand(b, -1, -1, -1, -1).to(dtype),
                          numbers[..., None, :].expand(-1, -1, -1, PATCHES, -1).to(dtype)), -1)
        if hasattr(self, 'pose_position'):
            # 11/03: each patch's position basis x its picture's joints as taken and their change since
            basis = pose_position_basis().to(device=weight.device, dtype=torch.float32)          # [PATCHES, 6]
            pose = torch.cat((condition[..., :6], condition[..., 12:18]), -1).to(basis)          # [B, 2, N, 12]
            products = (basis[:, :, None] * pose[..., None, None, :]).flatten(-2)               # [B, 2, N, P, 72]
            bank = torch.cat((bank, products.to(dtype)), -1)
        padding = ~valid.repeat_interleave(PATCHES, dim=2).reshape(b, 2 * n * PATCHES)
        return _one_readable_token(bank.reshape(b, 2 * n * PATCHES, self._bank_inputs()), padding)

    def _bank_inputs(self):
        """concat_inputs: the width of a bank token (BANK_INPUTS, + POSE_POSITION with pose_position)."""
        return BANK_INPUTS + (POSE_POSITION if hasattr(self, 'pose_position') else 0)

    def _encode_patches(self, bank, padding):
        """17.164: the bank through the LeWM's own picture encoder; a row without a picture keeps its one zero token."""
        readable = ~padding
        for layer in self.patch_encoder:
            bank = layer(bank, readable)
        empty = readable.sum(1) == 1        # _visual_bank's row without a picture: token 0 alone (a picture has PATCHES)
        return torch.where(empty[:, None, None], 0., bank) if empty.any() else bank

    def forward(self, q_position, trace, history, query_offsets, *, patches=None, patch_valid=None,
                memory=None, now_time=None, scene_context=None, controller=None, attention_supervision=None):
        b = len(q_position)
        if attention_supervision is not None and hasattr(self, 'memory_pictures'):
            raise ValueError('attention supervision reads the three-picture bank only')
        if attention_supervision is not None:
            # 17.163: (heads, weights [B, 6 * PATCHES] or [B, heads, 6 * PATCHES] in 0-1, the bank's key order)
            # -> output.attention_mass
            heads, weights = attention_supervision
            if patches is None or not all(layer.direct for layer in self.layers):
                raise ValueError('attention supervision needs pictures and a directly attending LeWM')
            finite('attention supervision weights', weights,
                   (b, 6 * PATCHES) if weights.ndim == 2 else (b, heads, 6 * PATCHES))
            if (weights < 0).any() or (weights > 1).any():
                raise ValueError('attention supervision weights are 0-1')
        if controller is not None and not getattr(self, 'controller_input', False):
            raise ValueError('this LeWM was not built to read the arm\'s control setting')
        if getattr(self, 'controller_input', False):
            # 17.153: no silent default; the trainer, the exams and the service say which setting every row had
            if controller is None:
                raise ValueError(f'this LeWM reads the arm\'s control setting: pass controller [B, '
                                 f'{len(CONTROLLER_INPUTS)}] {CONTROLLER_INPUTS}')
            finite('controller setting', controller, (b, len(CONTROLLER_INPUTS)))
            if checking():
                fail_if(~((controller == 0) | (controller == 1)).all(), 'controller inputs are 0 or 1')
        h = query_offsets.shape[1] if query_offsets.ndim == 2 else 0
        if h not in (5, 15, self.horizon):
            raise ValueError('full-patch LeWM expects 5, 15 or 40 readbacks')
        finite('current q', q_position, (b, 6))
        trace.validate(b, max_commands=256)
        clock('future readback offsets', query_offsets, (b, h))
        if checking():
            fail_if((query_offsets <= 0).any() | (query_offsets.diff(dim=1) <= 0).any(),
                    'future readback times must be positive and ordered')
        q = q_position.to(self.query_in.weight)
        active = trace.active_at(query_offsets, q_position).to(q)
        qt = query_offsets.to(q)
        qdt = torch.diff(torch.cat((qt.new_zeros(b, 1), qt), -1), dim=1)
        inputs = [self.normalize_position(q)[:, None].expand(-1, h, -1), self.normalize_position(active)]
        if getattr(self, 'balanced_inputs', False):
            # 17.160: how far each command is from the joints, in increments (FP32 here; autocast casts it in
            # query_in with its own relative precision), and the times on a fixed 0-1 scale
            distance = (active - q[:, None]) / self.increment_scale.to(q)
            if hasattr(self, 'command_symlog'):         # 17.161: DreamerV3's symlog, in FP32 like the distance
                distance = torch.sign(distance) * torch.log1p(distance.abs())
            inputs += [distance, torch.stack((qt / TIME_SCALE, qdt / STEP_SCALE), -1)]
        else:
            inputs.append(torch.stack((qt, qdt), -1))
        if hasattr(self, 'concat_inputs'):
            # each step enters the layers as [learned step query, the step's inputs, recent motion, control setting]
            motion = self._motion(history, q).flatten(1).to(q)
            query = self.query_in(torch.cat((self.query[:, :h].expand(b, -1, -1).to(q), torch.cat(inputs, -1).to(q),
                                             motion[:, None].expand(-1, h, -1),
                                             controller.to(q)[:, None].expand(-1, h, -1)), -1))
        else:
            query = self.query[:, :h] + self.query_in(torch.cat(inputs, -1))
            motion = self._motion(history, q).flatten(1)
            query = query + self.history(motion)[:, None]
            if getattr(self, 'controller_input', False):
                query = query + (controller.to(query) @ self.controller_weight.to(query))[:, None]
        if scene_context is not None:
            finite('learned observation scene context', scene_context, (b, self.width))
            query = query + scene_context[:, None]
        direct = self.layers[0].direct
        if patches is None and direct:
            # 17.106: the empty bank as its one readable token (the same reads: every other token is padding)
            bank = query.new_zeros(b, 1, self._bank_inputs() if hasattr(self, 'concat_inputs') else self.width)
            pad = torch.zeros(b, 1, device=query.device, dtype=torch.bool)
        elif patches is None:
            bank = query.new_zeros(b, 6 * PATCHES, self.width)
            pad = torch.ones(b, 6 * PATCHES, device=query.device, dtype=torch.bool)
            pad[:, 0] = False
        elif hasattr(self, 'memory_pictures'):
            # 17.166: every row one memory expanded (the planner's candidates)? then the bank once for all of them
            shared = b > 1 and all(x is not None and x.stride(0) == 0 for x in (patches, patch_valid, q_position, now_time))
            if shared:
                first = SimpleNamespace(q_at_image=memory.q_at_image[:1], image_times=memory.image_times[:1])
                bank, pad = self._memory_bank(patches[:1], patch_valid[:1], q[:1], first, now_time[:1])
            else:
                bank, pad = self._memory_bank(patches, patch_valid, q, memory, now_time)
        else:
            bank, pad = self._visual_bank(patches, patch_valid, q, memory, now_time)
            if hasattr(self, 'patch_encoder'):
                bank = self._encode_patches(bank, pad)
        if direct and bank.is_cuda and _autocasting():
            # 17.106: the bank in autocast's dtype once, not once per layer inside every projection (the same values;
            # the layers' gradients for it now add up in that dtype before their one conversion back)
            bank = bank.to(_autocast_dtype())
        commands = command_mask = None
        if self.command_conditioning == FULL_COMMANDS:
            commands, command_mask = self._command_bank(trace, q, query_offsets)
        causal = torch.ones(h, h, device=query.device, dtype=torch.bool).triu(1)
        masses = []
        for index, layer in enumerate(self.layers):
            if commands is not None:
                query = self.command_readers[index](query, commands, command_mask)
            if attention_supervision is None:
                query = layer(query, bank, causal, pad)
            else:
                query, mass = layer(query, bank, causal, pad, supervise=(heads, weights))
                masses.append(mass)
        hidden = self.norm(query)
        reference = active
        predicted = reference + self.response(hidden) * self.increment_scale
        sigma = (F.softplus(self.scale(hidden).float()) + 1e-4) * self.increment_scale
        path = torch.cat((q_position[:, None], predicted.to(q_position)), 1)
        if attention_supervision is not None:
            return SupervisedARMOutput(path, sigma.to(q_position), hidden, regularization_hidden=query,
                                       attention_mass=torch.stack(masses))
        return ARMOutput(path, sigma.to(q_position), hidden, regularization_hidden=query)
