"""Classic 192D Actor and command-timestamp-conditioned LeWM for V21 research."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from .prod_v20_policy import (
    DEFAULT_LOWER, DEFAULT_UPPER, _PositionModule, _Encoder, _Decoder,
    _validate_history, project_joint_bounds,
)
from .v21_checks import checking, fail_if
from .v21_compiled import region
from .v21_memory import finite, finite_where, clock
from .v21_encoder import DIM as FEATURE_DIM, PATCHES


def admit_untrained_scene_projections(state, initialized):
    """Drop the old 192-d patch bottleneck and keep the 384-d vision tower at init."""
    prefix = "lewm." if any(str(key).startswith("lewm.") for key in state) else ""
    for key in list(state):
        bare = key[5:] if str(key).startswith("lewm.") else str(key)
        if bare == "scene_gate" or bare.startswith(("patch_in.", "patch_attn.")):
            del state[key]
    for key, value in initialized.items():
        bare = key[5:] if str(key).startswith("lewm.") else str(key)
        if not bare.startswith(("vision_layers.", "vision_read.")):
            continue
        full = prefix + bare
        if full not in state:
            state[full] = value.detach().cpu().clone()
        elif tuple(state[full].shape) != tuple(value.shape):
            raise ValueError(f"{full} shape {tuple(state[full].shape)} is not {tuple(value.shape)}")
    return state


@dataclass(frozen=True)
class CommandTrace:
    positions: torch.Tensor       # [B,M,6], all recorded/planned commands
    offsets: torch.Tensor         # [B,M], float64 seconds from q0
    valid: torch.Tensor           # [B,M], right padding
    source: str                   # planned | recorded
    complete: bool                # no knowingly omitted command entries

    def validate(self, batch, *, max_commands=128):
        if self.positions.ndim != 3 or self.positions.shape[0] != batch or self.positions.shape[-1] != 6:
            raise ValueError("command trace must be [B,M,6]")
        m = self.positions.shape[1]
        if max_commands not in (128, 256) or not 1 <= m <= max_commands or self.valid.shape != (batch, m) or self.valid.dtype != torch.bool:
            raise ValueError("invalid command trace padding")
        if checking():
            fail_if(~self.valid[:, 0].all() | ((~self.valid[:, :-1]) & self.valid[:, 1:]).any(),
                    "command trace requires a nonempty valid prefix")
        if self.source not in ("planned", "recorded") or self.complete is not True:
            raise ValueError("complete planned or recorded command trace required")
        # section 17.80: the checks of the valid commands, written without selecting them
        finite_where("valid commands", self.positions, self.valid)
        if self.offsets.shape != self.valid.shape:
            raise ValueError("command clock shape differs")
        finite_where("command offsets", self.offsets, self.valid)
        if self.offsets.dtype != torch.float64:
            raise ValueError("command offsets: timestamps must be float64 seconds")
        if checking():
            fail_if(((self.offsets < 0) & self.valid).any(), "negative command offset")
        both = self.valid[:, 1:] & self.valid[:, :-1]
        if checking():
            fail_if((both & (self.offsets.diff(dim=1) <= 0)).any(), "command times must increase strictly")

    def active_at(self, query_times, q0):
        """Command immediately BEFORE each readback; an exactly tied new write is later."""
        b, m = self.valid.shape
        eligible = self.valid[:, None] & (self.offsets[:, None] < query_times[:, :, None] - 1e-9)
        indices = torch.where(eligible, torch.arange(m, device=q0.device), -1).amax(-1)
        clean = torch.where(self.valid[..., None], self.positions, torch.zeros_like(self.positions))
        result = clean.gather(1, indices.clamp_min(0)[..., None].expand(-1, -1, 6))
        return torch.where((indices >= 0)[..., None], result, q0[:, None])

    @classmethod
    def policy(cls, commands, dt):
        b = len(commands)
        finite("policy commands", commands, (b, 5, 6)); finite("intervals", dt, (b, 5))
        if (dt <= 0).any():
            raise ValueError("positive command intervals required")
        times = torch.cat((torch.zeros(b, 1, device=dt.device, dtype=torch.float64),
                           dt.double().cumsum(-1)[:, :-1]), -1)
        return cls(commands, times, torch.ones(b, 5, device=dt.device, dtype=torch.bool), "planned", True)


@dataclass(frozen=True)
class ARMOutput:
    predicted_q_path: torch.Tensor
    q_uncertainty: torch.Tensor
    hidden: torch.Tensor
    # Optional training representation; keep the existing normalized hidden
    # and all inference consumers unchanged.
    regularization_hidden: torch.Tensor | None = None


class V21LeWM(_PositionModule):
    """2-layer Transformer. A query cannot read later commands, even indirectly."""
    def __init__(self, *, increment_scale=None, horizon=5):
        super().__init__(DEFAULT_LOWER, DEFAULT_UPPER, increment_scale)
        if type(horizon) is not int or horizon not in (5, 40):
            raise ValueError('LeWM horizon must be 5 or 40')
        self.horizon = horizon
        self.joint = nn.Linear(1, 192)
        self.joint_ids = nn.Parameter(torch.randn(1, 6, 192) * .02)
        self.history_projection = nn.Sequential(nn.Linear(101, 192), nn.GELU(), nn.Linear(192, 192))
        self.history_ids = nn.Parameter(torch.randn(1, 3, 192) * .02)
        # Full DINO patches stay 384-d tokens. No spatial pool and no 384→192 squeeze.
        # DINOv3-L patches (fcpg/v21_encoder.py) first pass Linear 1024 -> 384; for DINOv2 this is the identity.
        self.feature_in = nn.Identity() if FEATURE_DIM == 384 else nn.Linear(FEATURE_DIM, 384)
        self.vision_layers = nn.ModuleList(_Encoder(384, 6) for _ in range(4))
        self.vision_read = nn.ModuleList(
            nn.MultiheadAttention(192, 6, kdim=384, vdim=384, batch_first=True) for _ in range(2))
        for reader in self.vision_read:
            nn.init.zeros_(reader.out_proj.weight)
            nn.init.zeros_(reader.out_proj.bias)
        self.command_projection = nn.Sequential(nn.Linear(20, 192), nn.GELU(), nn.Linear(192, 192))
        self.time_projection = nn.Linear(2, 192)
        self.query = nn.Parameter(torch.randn(1, horizon, 192) * .02)
        self.layers = nn.ModuleList([_Encoder(), _Encoder()])
        self.norm = nn.LayerNorm(192)
        self.response = nn.Linear(192, 6)
        self.scale = nn.Linear(192, 6)
        nn.init.normal_(self.response.weight, std=.005); nn.init.zeros_(self.response.bias)

    def _vision_tokens(self, patches, patch_valid):
        """Keep every patch. Shape in [B,2,3,PATCHES,FEATURE_DIM], out [B, 6 * PATCHES, 384]."""
        b = patches.shape[0]
        if patches.shape != (b, 2, 3, PATCHES, FEATURE_DIM) or not torch.isfinite(patches).all():
            raise ValueError(f'LeWM patches must be finite [B,2,3,{PATCHES},{FEATURE_DIM}]')
        if patch_valid is None:
            patch_valid = torch.ones(b, 2, 3, dtype=torch.bool, device=patches.device)
        elif patch_valid.shape != (b, 2, 3) or patch_valid.dtype != torch.bool:
            raise ValueError('patch valid mask must be bool [B,2,3]')
        weight = self.vision_layers[0].norm1.weight
        tokens = self.feature_in(patches.reshape(b, 6 * PATCHES, FEATURE_DIM).to(dtype=weight.dtype, device=weight.device))
        padding = ~patch_valid.to(device=tokens.device).reshape(b, 6).repeat_interleave(PATCHES, dim=1)
        empty = padding.all(dim=1)
        if empty.any():
            padding = padding.clone()
            padding[empty, 0] = False
        for layer in self.vision_layers:
            tokens = layer(tokens, padding=padding)
        # #region agent log
        try:
            import json as _json, time as _time
            with open('/home/james/feedback-corrected-grounding/.cursor/debug-666150.log', 'a') as _f:
                _f.write(_json.dumps({'sessionId': '666150', 'hypothesisId': 'V', 'location': 'v21_models.py:_vision_tokens', 'message': 'vision tokens kept', 'data': {'shape': list(tokens.shape), 'dim': int(tokens.shape[-1]), 'pooled': False}, 'timestamp': int(_time.time() * 1000)}) + '\n')
        except Exception:
            pass
        # #endregion
        return tokens, padding

    def forward(self, q_position, trace, history, query_offsets, *, patches=None, patch_valid=None,
                memory=None, now_time=None):
        b = len(q_position)
        h = query_offsets.shape[1] if query_offsets.ndim == 2 else 0
        if h not in (5, self.horizon):
            raise ValueError('readback horizon differs from the trained LeWM capacity')
        del memory, now_time
        finite("current q", q_position, (b, 6))
        trace.validate(b, max_commands=256 if h == 40 else 128)
        _validate_history(history, b)
        clock("future readback offsets", query_offsets, (b, h))
        if (query_offsets <= 0).any() or (query_offsets.diff(dim=1) <= 0).any():
            raise ValueError("future readback times must be positive and ordered")
        if (trace.valid & (trace.offsets >= query_offsets[:, -1:])).any():
            raise ValueError("command trace extends beyond final readback")
        q = q_position.to(self.joint.weight)
        commands = torch.where(trace.valid[..., None], trace.positions, torch.zeros_like(trace.positions)).to(q)
        ct = torch.where(trace.valid, trace.offsets, torch.zeros_like(trace.offsets))
        increments = torch.diff(torch.cat((q[:, None], commands), 1), dim=1)
        previous_dt = torch.diff(torch.cat((ct.new_zeros(b, 1), ct), -1), dim=1).clamp_min(0)
        values = torch.cat((self.normalize_position(commands), (commands-q[:, None])/self.increment_scale,
                            increments/self.increment_scale, torch.log1p(ct).to(q)[..., None],
                            torch.log1p(previous_dt).to(q)[..., None]), -1)
        values = torch.where(trace.valid[..., None], values, torch.zeros_like(values))
        joints = self.joint(self.normalize_position(q)[..., None]) + self.joint_ids
        historic = self.history_projection(self._motion(history, q)) + self.history_ids
        vision = None
        vision_pad = None
        if patches is not None:
            vision, vision_pad = self._vision_tokens(patches, patch_valid)
        qt = query_offsets.to(q)
        qdt = torch.diff(torch.cat((qt.new_zeros(b, 1), qt), -1), dim=1)
        query = self.query[:, :h] + self.time_projection(torch.stack((qt, qdt), -1))
        tokens = torch.cat((joints, historic, self.command_projection(values), query), 1)
        m = commands.shape[1]; length = 9 + m + h
        allow = torch.zeros(b, length, length, device=q.device, dtype=torch.bool)
        allow[:, :, :9] = True
        allow[:, 9:9+m, 9:9+m] = ct[:, None, :] <= ct[:, :, None]
        allow[:, 9+m:, 9:9+m] = ct[:, None, :] < query_offsets[:, :, None] - 1e-9
        allow[:, 9+m:, 9+m:] = torch.ones(h, h, device=q.device, dtype=torch.bool).tril()
        padding = torch.cat((torch.zeros(b, 6, device=q.device, dtype=torch.bool),
                             ~history.valid, ~trace.valid,
                             torch.zeros(b, h, device=q.device, dtype=torch.bool)), -1)
        causal = (~allow)[:, None].expand(-1, 6, -1, -1).reshape(b*6, length, length)
        for layer, reader in zip(self.layers, self.vision_read):
            tokens = layer(tokens, padding=padding, causal=causal)
            if vision is not None:
                tokens = tokens + reader(tokens, vision, vision, key_padding_mask=vision_pad, need_weights=False)[0]
        hidden = self.norm(tokens[:, -h:])
        reference = trace.active_at(query_offsets, q_position).to(q)
        predicted = reference + self.response(hidden) * self.increment_scale
        sigma = (F.softplus(self.scale(hidden).float()) + 1e-4) * self.increment_scale
        return ARMOutput(torch.cat((q_position[:, None], predicted.to(q_position)), 1),
                         sigma.to(q_position), hidden)


class _WideDecoder(nn.Module):
    """The prod_v20 decoder block (same parts and names) at another width."""
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, query, memory, causal, memory_padding):
        normalized = self.norm1(query)
        query = query + self.self_attn(normalized, normalized, normalized, attn_mask=causal, need_weights=False)[0]
        mem = self.memory_norm(memory)
        query = query + self.cross_attn(self.norm2(query), mem, mem, key_padding_mask=memory_padding, need_weights=False)[0]
        return query + self.ffn(self.norm3(query))


class _DropoutEncoder(nn.Module):
    """Section 17.75 (actor_layers 'v2'): the _Encoder block with dropout on attention weights and both residual
    branches."""
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.drop = nn.Dropout(dropout)

    @region         # section 17.84: one compiled program per layer call in a training update (fcpg/v21_compiled.py)
    def forward(self, x, padding=None):
        normalized = self.norm1(x)
        x = x + self.drop(self.attn(normalized, normalized, normalized, key_padding_mask=padding, need_weights=False)[0])
        return x + self.drop(self.ffn(self.norm2(x)))


class _DropoutDecoder(_WideDecoder):
    """Section 17.75 (actor_layers 'v2'): the _WideDecoder block with dropout; causal=None lets every command query
    attend to every other (parallel decoding of the whole chunk, as OpenVLA-OFT)."""
    def __init__(self, dim, heads, dropout):
        super().__init__(dim, heads)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    @region         # section 17.84, as _DropoutEncoder.forward
    def forward(self, query, memory, causal, memory_padding):
        normalized = self.norm1(query)
        query = query + self.drop(self.self_attn(normalized, normalized, normalized, attn_mask=causal,
                                                 need_weights=False)[0])
        mem = self.memory_norm(memory)
        query = query + self.drop(self.cross_attn(self.norm2(query), mem, mem, key_padding_mask=memory_padding,
                                                  need_weights=False)[0])
        return query + self.drop(self.ffn(self.norm3(query)))


@dataclass(frozen=True)
class PolicyOutput:
    commands_position: torch.Tensor
    increments: torch.Tensor
    # training doc section 17.46: the absolute jaw opening the auxiliary head predicts (training loss only), or None
    jaw_absolute: torch.Tensor | None = None


PLACE_GOAL_INIT_SEED = 1707   # fcpg/v21_place_goal.py projection; its own generator (training doc section 17.7)


class V21Actor(_PositionModule):
    """Paired memory/update tokens + q + native text tokens; no MiniLM dependency."""
    def __init__(self, vocabulary_size, *, increment_scale=None, max_text_tokens=64, object_relative=False, horizon=5,
                 history_inputs=True, gripper_separate=False, queries=8, width=192, place_relative=False,
                 place_goal=False, gripper_relative=False, gripper_absolute_aux=False, layers='v1', text_tokens=True,
                 object_on_pose=False):
        super().__init__(DEFAULT_LOWER, DEFAULT_UPPER, increment_scale)
        if type(queries) is not int or queries not in (8, 32) or type(width) is not int or width not in (192, 384):
            raise ValueError('Actor reads 8 or 32 FCPG tokens per view and is 192 or 384 wide')
        # section 17.75: 'v2' = 4 context layers instead of 1, command queries attending in both directions, dropout
        # 0.1 (_DropoutEncoder, _DropoutDecoder); 'v1' builds exactly the earlier modules and parameter names.
        if layers not in ('v1', 'v2'):
            raise ValueError("Actor layers are 'v1' or 'v2'")
        self.layer_version = layers
        # same doc, section 8 (L1): a longer FCPG summary per view and/or a wider Actor with the same layers.
        # FCPG tokens stay 192-d; width 192 and 8 queries is every earlier model.
        self.queries_per_view, self.width = queries, width
        if type(horizon) is not int or horizon not in (5, 40):
            raise ValueError('Actor horizon must be 5 or 40')
        if type(history_inputs) is not bool or type(gripper_separate) is not bool:
            raise ValueError('history_inputs and gripper_separate must be explicit booleans')
        if gripper_separate and history_inputs:
            raise ValueError('the motion history carries the jaw reading; gripper_separate needs history_inputs=False')
        # same doc, section 8: True = the jaw opening (joint 6) is handled on its own. The Actor does not read it,
        # so it cannot move the arm, and the sixth output is the opening to command (in the normalised joint
        # range), not a change added to the reading. The five arm joints stay "pose + changes".
        self.gripper_separate = gripper_separate
        # section 17.39 (K2): the separate jaw output becomes "current reading + output x half range" instead of
        # "range centre + output x half range". The reading is still not an input: only how the sixth output becomes
        # a command changes, so a squeeze below the contact width can be learned whatever the object's width.
        if type(gripper_relative) is not bool or (gripper_relative and not gripper_separate):
            raise ValueError('gripper_relative is an explicit boolean and needs gripper_separate')
        self.gripper_relative = gripper_relative
        # section 17.46: an extra output that predicts the absolute jaw command ("range centre + output x half range")
        # from the same features, trained beside the relative one and never used as a command. It keeps the pressure
        # an absolute jaw puts on reading the jaws in the wrist picture (17.43-17.44: K6 lost half of H1b's sight steering).
        if type(gripper_absolute_aux) is not bool or (gripper_absolute_aux and not gripper_relative):
            raise ValueError('gripper_absolute_aux is an explicit boolean and needs gripper_relative')
        self.gripper_absolute_aux = gripper_absolute_aux
        self.horizon = horizon
        # docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md (B): False = the Actor does not see how the arm just
        # moved (per-frame motion tokens, execution feedback); it acts from the views, pose, text and object cue.
        # The unused projections stay so parameter names (and the temporal age-column contract) are unchanged.
        self.history_inputs = history_inputs
        self.uses_object_relative = bool(object_relative)
        if self.uses_object_relative:
            # Wrist object-vs-jaw grounding (8 numbers). Zero output init: the token starts as zeros. Once seen it
            # still takes attention from the other tokens, so a warm start is close to, not exactly, the model
            # before (training overhaul doc section 17).
            self.object_relative = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
            nn.init.zeros_(self.object_relative[2].weight); nn.init.zeros_(self.object_relative[2].bias)
        # 11/03 (V21Config.object_on_pose): the same projection added to the pose token while the cue is seen, as the
        # place cue below, instead of a token of its own; same parameters
        if type(object_on_pose) is not bool or (object_on_pose and not self.uses_object_relative):
            raise ValueError('object_on_pose is an explicit boolean and needs the object token')
        self.object_on_pose = object_on_pose
        # docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md section 17: where the object goes (the registered drop zone
        # seen in the wrist picture) against the jaws, the same 8 numbers, its own projection with zero output init,
        # ADDED to the pose token and only while seen. Not a token of its own: even a zero token takes attention from
        # the others once LayerNorm and attention biases are trained, so a warm start would not be the plain model.
        if type(place_relative) is not bool or (place_relative and not self.uses_object_relative):
            raise ValueError('place_relative is an explicit boolean and needs the object token')
        self.uses_place_relative = place_relative
        if self.uses_place_relative:
            self.place_relative = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
            nn.init.zeros_(self.place_relative[2].weight); nn.init.zeros_(self.place_relative[2].bias)
        # same doc, section 17.7 (the B route): the place goal against the jaws in the arm's polar frame
        # (fcpg/v21_place_goal.py), the same kind of zero-output projection added to the pose token while available.
        # Drawn from its own generator, so every other module starts exactly as in the same seed's model without it.
        if type(place_goal) is not bool:
            raise ValueError('place_goal is an explicit boolean')
        self.uses_place_goal = place_goal
        if self.uses_place_goal:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(PLACE_GOAL_INIT_SEED)
                self.place_goal = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
            nn.init.zeros_(self.place_goal[2].weight); nn.init.zeros_(self.place_goal[2].bias)
        if vocabulary_size < 4:
            raise ValueError("vocabulary must include four reserved tokens")
        # James 11/02 「fcpg和actor 不要处理了文字了 没用 它看不懂的」: False = no task-word tokens in the context (the
        # object finder still reads the task text); True builds exactly the earlier modules
        if type(text_tokens) is not bool:
            raise ValueError('text_tokens is an explicit boolean')
        self.text_tokens = text_tokens
        if text_tokens:
            self.text = nn.Embedding(vocabulary_size, width, padding_idx=0)
            self.text_position = nn.Parameter(torch.randn(1, max_text_tokens, width) * .02)
        self.pair = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, width), nn.GELU(), nn.Linear(width, width))
        self.flags = nn.Linear(5, width)
        self.motion_context = nn.Sequential(nn.Linear(28, width), nn.GELU(), nn.Linear(width, width))
        self.view = nn.Parameter(torch.randn(1, 2, 1, width) * .02)
        self.q_projection = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
        self.q_error = nn.Sequential(nn.Linear(37, width), nn.GELU(), nn.Linear(width, width))
        if layers == 'v2':
            self.context = nn.ModuleList([_DropoutEncoder(width, width // 32, .1) for _ in range(4)])
            self.layers = nn.ModuleList([_DropoutDecoder(width, width // 32, .1) for _ in range(4)])
        else:
            self.context = _Encoder() if width == 192 else _Encoder(width, width // 32)
            self.layers = nn.ModuleList([_Decoder() if width == 192 else _WideDecoder(width, width // 32) for _ in range(4)])
        self.query = nn.Parameter(torch.randn(1, horizon, width) * .02)
        self.command_time = nn.Linear(2, width)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 6)
        nn.init.normal_(self.output.weight, std=.005); nn.init.zeros_(self.output.bias)
        if gripper_absolute_aux:
            self.jaw_absolute = nn.Linear(width, 1)
            nn.init.normal_(self.jaw_absolute.weight, std=.005); nn.init.zeros_(self.jaw_absolute.bias)

    def forward(self, grounding, visual_error, q_position, text_ids, *, command_offsets,
                motion_context, motion_available, memory_available, measurement_pending,
                visual_feedback_available, visual_feedback_age, q_error, feedback_offsets,
                feedback_available, feedback_age, object_relative=None, place_relative=None, place_goal=None):
        b = len(q_position)
        finite("grounding", grounding, (b, 2, self.queries_per_view, 192)); finite("visual update", visual_error, grounding.shape)
        finite("q", q_position, (b, 6)); finite("q error", q_error, (b, 5, 6))
        finite("feedback age", feedback_age, (b,))
        finite("visual feedback age", visual_feedback_age, (b, 2))
        clock("actual feedback offsets", feedback_offsets, (b, 5))
        h = command_offsets.shape[1] if command_offsets.ndim == 2 else 0
        if h not in (5, self.horizon):
            raise ValueError('command horizon differs from the trained Actor capacity')
        clock("Actor command offsets", command_offsets, (b, h))
        if checking():
            fail_if((command_offsets[:, 0].abs() > 1e-9).any() | (command_offsets.diff(dim=1) <= 0).any()
                    | (command_offsets < 0).any() | (visual_feedback_age < 0).any(),
                    "Actor needs ordered command times starting at zero and nonnegative ages")
        if checking():
            fail_if((feedback_available[:, None] & (feedback_offsets <= 0)).any(), "actual feedback times must be positive")
        finite("plain motion context", motion_context, (b, 2, 3, 28))
        if motion_available.shape != (b, 2, 3) or motion_available.dtype != torch.bool:
            raise ValueError("actual frame motion availability must be bool [B,2,3]")
        masks = (memory_available, measurement_pending, visual_feedback_available)
        if any(x.shape != (b, 2) or x.dtype != torch.bool for x in masks):
            raise ValueError("separate observation/memory/update masks required")
        if feedback_available.shape != (b,) or feedback_available.dtype != torch.bool:
            raise ValueError("invalid execution feedback metadata")
        if checking():
            fail_if((feedback_age < 0).any(), "invalid execution feedback metadata")
        reads_text = getattr(self, 'text_tokens', True)
        if reads_text and (text_ids.ndim != 2 or text_ids.shape[0] != b
                           or not 1 <= text_ids.shape[1] <= self.text_position.shape[1] or text_ids.dtype != torch.long):
            raise ValueError("text token ids are invalid or silently truncated")
        if reads_text and checking():
            fail_if((text_ids < 0).any() | (text_ids >= self.text.num_embeddings).any() | ~(text_ids != 0).any(-1).all(),
                    "text token ids are invalid or silently truncated")
        reference = self.pair[1].weight
        g = grounding.to(reference)
        ev = torch.where(visual_feedback_available[..., None, None], visual_error, torch.zeros_like(visual_error)).to(g)
        executed = feedback_available if self.history_inputs else torch.zeros_like(feedback_available)
        flags = torch.cat((torch.stack((*masks, executed[:, None].expand(-1, 2)), -1).to(g),
                           torch.log1p(visual_feedback_age).to(g)[..., None]), -1)
        tracks = self.pair(torch.cat((g, ev), -1)) + self.flags(flags)[:, :, None] + self.view
        pose = self.normalize_position(q_position.to(g))
        if self.gripper_separate:
            pose = torch.cat((pose[:, :5], torch.zeros_like(pose[:, 5:])), -1)
        qtoken = self.q_projection(pose)[:, None]
        relative = None
        if self.uses_object_relative:
            relative = torch.zeros(b, 8, device=g.device, dtype=g.dtype) if object_relative is None else object_relative.to(g)
            finite("object-relative grounding", relative, (b, 8))
            if getattr(self, 'object_on_pose', False):
                qtoken = qtoken + self.object_relative(relative)[:, None] * (relative[:, :1] > 0).to(g)[:, None]
        if self.uses_place_relative:
            place = torch.zeros(b, 8, device=g.device, dtype=g.dtype) if place_relative is None else place_relative.to(g)
            finite("place-relative grounding", place, (b, 8))
            qtoken = qtoken + self.place_relative(place)[:, None] * (place[:, :1] > 0).to(g)[:, None]
        elif place_relative is not None:
            raise ValueError("a place token was given to an Actor without the place input")
        if getattr(self, 'uses_place_goal', False):
            goal = torch.zeros(b, 8, device=g.device, dtype=g.dtype) if place_goal is None else place_goal.to(g)
            finite("place goal", goal, (b, 8))
            qtoken = qtoken + self.place_goal(goal)[:, None] * (goal[:, :1] > 0).to(g)[:, None]
        elif place_goal is not None:
            raise ValueError("a place goal was given to an Actor without the goal input")
        eq = torch.where(feedback_available[:, None, None], q_error, torch.zeros_like(q_error)).to(g)
        err = torch.cat(((eq/self.increment_scale).flatten(1), feedback_available[:, None].to(g),
                         torch.log1p(feedback_age).to(g)[:, None],
                         torch.log1p(feedback_offsets).to(g)), -1)
        error_token = self.q_error(err)[:, None]
        if reads_text:
            text = self.text(text_ids) + self.text_position[:, :text_ids.shape[1]]
            text_padding = text_ids == 0
        else:
            text = g.new_zeros(b, 0, self.width)
            text_padding = torch.zeros(b, 0, device=g.device, dtype=torch.bool)
        if self.history_inputs:
            motion = self.motion_context(motion_context.to(g)).flatten(1, 2)
            motion_padding = ~motion_available.flatten(1)
            fixed = [qtoken, error_token]
        else:
            motion = g.new_zeros(b, 0, self.width)
            motion_padding = torch.zeros(b, 0, device=g.device, dtype=torch.bool)
            fixed = [qtoken]
        fixed_padding = torch.zeros(b, len(fixed), device=g.device, dtype=torch.bool)
        if self.uses_object_relative and not getattr(self, 'object_on_pose', False):
            fixed.append(self.object_relative(relative)[:, None])
            # The token exists only when the wrist object grounding is available;
            # archive rows (no registered object) attend exactly as before.
            fixed_padding = torch.cat((fixed_padding, (relative[:, 0] <= 0)[:, None]), 1)
        context = torch.cat((tracks.flatten(1, 2), motion, *fixed, text), 1)
        # Missing CURRENT image never erases a still-available memory track.
        padding = torch.cat((~memory_available[..., None].expand(-1, -1, self.queries_per_view).flatten(1),
                             motion_padding, fixed_padding, text_padding), 1)
        if getattr(self, 'layer_version', 'v1') == 'v2':
            for layer in self.context:
                context = layer(context, padding=padding)
            causal = None
        else:
            context = self.context(context, padding=padding)
            causal = torch.ones(h, h, device=g.device, dtype=torch.bool).triu(1)
        dt = command_offsets.diff(dim=1, prepend=command_offsets.new_zeros(b, 1))
        query = self.query[:, :h] + self.command_time(torch.stack((command_offsets, dt), -1).to(g))
        for layer in self.layers:
            query = layer(query, context, causal, padding)
        features = self.norm(query)
        increments = self.output(features).to(q_position)
        absolute = None
        if getattr(self, 'gripper_absolute_aux', False):
            absolute = (self.position_center[5].to(increments)
                        + self.jaw_absolute(features).to(q_position) * self.position_half_range[5].to(increments))
        if self.gripper_separate:
            arm = q_position[:, None, :5] + increments[..., :5].cumsum(1)
            origin = (q_position[:, None, 5:] if getattr(self, 'gripper_relative', False)
                      else self.position_center[5].to(increments))
            jaw = origin + increments[..., 5:] * self.position_half_range[5].to(increments)
            desired = torch.cat((arm, jaw), -1)
            # consumers read "pose + summed increments"; keep that identity for the jaw as well
            increments = torch.diff(torch.cat((q_position[:, None], desired), 1), dim=1)
        else:
            desired = q_position[:, None] + increments.cumsum(1)
        return PolicyOutput(project_joint_bounds(desired, self.position_lower, self.position_upper), increments,
                            absolute)
