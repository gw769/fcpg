"""Standalone LeWM, faithful to Maes et al. 2026 (github.com/lucas-maes/le-wm, MIT licence).

FeedForward, Attention, ConditionalBlock, Transformer, Embedder, MLP and ARPredictor are
copied from le-wm `module.py` (Copyright (c) 2026 Lucas Maes, MIT); only formatting and
two marked arguments differ (`Attention(causal=...)` for the patch encoder, and the
action Embedder's smoothing width, which the original sets equal to its 10-wide input).

Changes our data force, and nothing else:
1. Input is the frozen DINOv2 patch bank [256, 384] of each image, not pixels. A small
   transformer with a CLS token stands in for the from-scratch ViT-tiny; CLS goes through
   the original projector MLP with BatchNorm.
2. Two cameras per frame. One shared encoder with a learned camera embedding gives one
   embedding per camera; the predictor's frame token is their concatenation. A learned
   token stands in for a camera without an exposure. Missing exposures never enter
   BatchNorm, the prediction loss or SIGReg (SIGReg counts shrink; nothing is zero-filled).
3. A sequence is the frames [t-1.0, t-0.5, t, t+0.5, ..., t+4.0] s (three chain endpoints
   plus the sparse8 futures). Action block k is the five 10 Hz commands between frame k
   and k+1 (frameskip 5, as the original's Push-T data).

Unchanged: next-embedding MSE with gradients into the targets (no stop-gradient, no EMA),
teacher forcing over 4-frame windows (history 3, num_preds 1), step-wise SIGReg on every
encoded frame, causal AdaLN-zero predictor, BatchNorm projector and pred_proj, and the
planning cost (MSE between the last predicted embedding and the goal embedding).
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ARCHITECTURE = 'v21-lewm-standalone-v1'
FRAMES = 11          # t-1.0 ... t+4.0 s every 0.5 s
CONTEXT_FRAMES = 3   # t-1.0, t-0.5, t
HISTORY = 3
BLOCK = 5            # 10 Hz commands per 0.5 s action block
JOINTS = 6


# ---------------------------------------------------------------- copied from le-wm module.py
def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention; `causal` is fixed per module (added argument)."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, causal=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.causal = causal
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
                       if project_out else nn.Identity())

    def forward(self, x):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        b, t, _ = x.shape
        q, k, v = (y.reshape(b, t, self.heads, -1).transpose(1, 2) for y in self.to_qkv(x).chunk(3, dim=-1))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=self.causal)
        return self.to_out(out.transpose(1, 2).reshape(b, t, -1))


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard (unconditioned) block; the patch encoder uses it non-causally."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, causal=True):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, causal=causal)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    """Conditional transformer as in le-wm (input/cond/output projections, final norm)."""

    def __init__(self, input_dim, hidden_dim, output_dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
                                    for _ in range(depth))
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()

    def forward(self, x, c):
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        return self.output_proj(self.norm(x))


class Embedder(nn.Module):
    """Action encoder. le-wm uses smoothed_dim=10 for a 10-wide Push-T block; here the
    smoothing width equals the 30-wide block so it is not an extra bottleneck."""

    def __init__(self, input_dim, smoothed_dim, emb_dim, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(nn.Linear(smoothed_dim, mlp_scale * emb_dim), nn.SiLU(),
                                   nn.Linear(mlp_scale * emb_dim, emb_dim))

    def forward(self, x):
        x = self.patch_embed(x.float().permute(0, 2, 1)).permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, norm_fn=nn.LayerNorm, act_fn=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 norm_fn(hidden_dim) if norm_fn is not None else nn.Identity(),
                                 act_fn(), nn.Linear(hidden_dim, output_dim or input_dim))

    def forward(self, x):
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(self, *, num_frames, depth, heads, mlp_dim, input_dim, hidden_dim, output_dim=None,
                 dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(input_dim, hidden_dim, output_dim or input_dim, depth, heads,
                                       dim_head, mlp_dim, dropout)

    def forward(self, x, c):
        x = x + self.pos_embedding[:, :x.size(1)]
        return self.transformer(self.dropout(x), c)
# ----------------------------------------------------------------------------------------------


class PatchEncoder(nn.Module):
    """DINO patch bank of one image -> CLS -> projector MLP with BatchNorm (as le-wm)."""

    def __init__(self, *, cameras=2, patch_dim=384, depth=2, heads=6, dim_head=64, mlp_dim=1536,
                 embed_dim=192, projector_hidden=2048):
        super().__init__()
        self.input_norm = nn.LayerNorm(patch_dim)
        self.position = nn.Parameter(torch.randn(1, 256, patch_dim) * .02)
        self.camera = nn.Parameter(torch.randn(cameras, 1, patch_dim) * .02)
        self.cls = nn.Parameter(torch.randn(1, 1, patch_dim) * .02)
        self.blocks = nn.ModuleList(Block(patch_dim, heads, dim_head, mlp_dim, causal=False) for _ in range(depth))
        self.norm = nn.LayerNorm(patch_dim)
        self.projector = MLP(patch_dim, projector_hidden, embed_dim, norm_fn=nn.BatchNorm1d)
        self.embed_dim = embed_dim
        self.cameras = cameras
        self.recompute = True
        # PatchDropout (training only, off by default): fraction of the 256 patch tokens each
        # image keeps. Not part of the model config, so checkpoints and evaluation are unchanged.
        self.keep_fraction = 1.

    def forward(self, patches, valid):
        """patches [..., C, 256, 384]; valid [..., C] bool -> [..., C, D] (zeros where invalid)."""
        if valid.dtype != torch.bool or patches.shape[:-2] != valid.shape or valid.shape[-1] != self.cameras:
            raise ValueError('patch encoder requires patches [..., C, 256, 384] and bool validity [..., C]')
        out = torch.zeros(*valid.shape, self.embed_dim, device=patches.device, dtype=torch.float32)
        if not valid.any():
            return out
        if self.training and int(valid.sum()) < 2:
            raise ValueError('BatchNorm projector needs at least two valid exposures in training')
        camera = torch.nonzero(valid)[:, -1]
        tokens = self.input_norm(patches[valid].to(self.position.dtype)) + self.position + self.camera[camera]
        if self.training and self.keep_fraction < 1.:
            # Each image keeps its own random subset of patches, with their own positions.
            # keep_fraction == 1 draws no random numbers, so default training is unchanged.
            keep = max(1, int(round(tokens.shape[1] * self.keep_fraction)))
            order = torch.rand(tokens.shape[:2], device=tokens.device).argsort(1)[:, :keep]
            tokens = tokens.gather(1, order[..., None].expand(-1, -1, tokens.shape[-1]))
        x = torch.cat((self.cls.expand(len(tokens), -1, -1).to(tokens.dtype), tokens), 1)
        for block in self.blocks:
            # Recompute block activations in backward: same values and gradients, far less memory
            # (thousands of 257-token images per step).
            x = (checkpoint(block, x, use_reentrant=False)
                 if self.recompute and self.training and torch.is_grad_enabled() else block(x))
        z = self.projector(self.norm(x[:, 0]))
        return out.index_put((valid,), z.float())


class StandaloneLeWM(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, *, cameras=2, embed_dim=192, encoder_depth=2, history=HISTORY,
                 action_dim=BLOCK * JOINTS, depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1,
                 inverse_dynamics=False):
        super().__init__()
        self.config = dict(cameras=cameras, embed_dim=embed_dim, encoder_depth=encoder_depth, history=history,
                           action_dim=action_dim, depth=depth, heads=heads, dim_head=dim_head,
                           mlp_dim=mlp_dim, dropout=dropout)
        frame_dim = cameras * embed_dim
        # 2026-10-12 (docs/V21_LEWM_JEPA_RETRAIN_20261012.md): an optional training-only head that reads
        # the command change of one 0.5 s block back from the two frames around it. Off by default; the
        # config key exists only when it is on, so every earlier checkpoint loads and compares unchanged.
        self.inverse = None
        if inverse_dynamics:
            self.config['inverse_dynamics'] = True
            self.inverse = nn.Sequential(nn.Linear(3 * frame_dim, 768), nn.GELU(), nn.Linear(768, 768), nn.GELU(),
                                         nn.Linear(768, action_dim))
        self.encoder = PatchEncoder(cameras=cameras, depth=encoder_depth, embed_dim=embed_dim)
        self.missing = nn.Parameter(torch.zeros(cameras, embed_dim))
        self.action_encoder = Embedder(action_dim, action_dim, frame_dim)
        self.predictor = ARPredictor(num_frames=history, depth=depth, heads=heads, mlp_dim=mlp_dim,
                                     input_dim=frame_dim, hidden_dim=frame_dim, output_dim=frame_dim,
                                     dim_head=dim_head, dropout=dropout)
        self.pred_proj = MLP(frame_dim, 2048, frame_dim, norm_fn=nn.BatchNorm1d)
        self.cameras, self.embed_dim, self.history = cameras, embed_dim, history

    def encode(self, patches, valid):
        return self.encoder(patches, valid)

    def frame_tokens(self, emb, valid):
        return torch.where(valid[..., None], emb, self.missing.to(emb)).flatten(-2)

    def predict(self, emb, valid, act_emb):
        """emb [B, T<=history, C, D], valid [B, T, C], act_emb [B, T, C*D] -> next embeddings [B, T, C, D].

        Position j sees frames and actions 0..j only (causal), as in le-wm.
        """
        b, t = emb.shape[:2]
        if t > self.history or act_emb.shape[:2] != (b, t):
            raise ValueError('predictor input must be at most `history` frames with one action each')
        hidden = self.predictor(self.frame_tokens(emb, valid), act_emb)
        return self.pred_proj(hidden.reshape(b * t, -1)).reshape(b, t, self.cameras, self.embed_dim).float()

    def rollout(self, history_emb, history_valid, act_emb):
        """Autoregressive rollout (le-wm `rollout`): history [B, 3, C, D] of actual frames,
        act_emb [B, 3 + n - 1, C*D] -> n predicted frames [B, n, C, D]."""
        steps = act_emb.shape[1] - history_emb.shape[1] + 1
        if steps < 1:
            raise ValueError('rollout needs at least one future action block')
        emb, valid = history_emb, history_valid
        predictions = []
        for _ in range(steps):
            k = emb.shape[1]
            window = slice(max(0, k - self.history), k)
            nxt = self.predict(emb[:, window], valid[:, window], act_emb[:, window])[:, -1:]
            predictions.append(nxt)
            emb = torch.cat((emb, nxt), 1)
            valid = torch.cat((valid, torch.ones_like(valid[:, :1])), 1)
        return torch.cat(predictions, 1)


def inverse_dynamics_loss(model, emb, valid, delta_blocks, block_valid):
    """Mean squared error of the command change read back from consecutive frames.

    emb [B, S, C, D] (with gradient), valid [B, S, C], delta_blocks [B, S-1, 30] (the five commands of
    block j minus the command just before it, scaled), block_valid [B, S-1]. Block j lies between frame j
    and frame j+1; it is scored when both frames have at least one exposure and its commands are complete.
    A missing camera enters as the learned `missing` token, as in the predictor input.
    """
    if model.inverse is None:
        raise ValueError('this model has no inverse-dynamics head')
    b, s = emb.shape[:2]
    if valid.shape[:2] != (b, s) or delta_blocks.shape[:2] != (b, s - 1) or block_valid.shape != (b, s - 1):
        raise ValueError('sequence, validity and command blocks disagree')
    tokens = model.frame_tokens(emb.float(), valid)                     # [B, S, C*D]
    before, after = tokens[:, :-1], tokens[:, 1:]
    scored = valid[:, :-1].any(-1) & valid[:, 1:].any(-1) & block_valid
    if not scored.any():
        raise ValueError('no scored command block in the batch')
    guess = model.inverse(torch.cat((before, after, after - before), -1)[scored])
    target = delta_blocks[scored].float()
    loss = (guess.float() - target).square().mean()
    with torch.no_grad():
        spread = (target - target.mean(0)).square().mean()
        diag = dict(idm_mse=float(loss), idm_scored=int(scored.sum()),
                    idm_explained=float(1 - loss / spread) if spread > 0 else None)
    return loss, diag


def window_indices(frames, device=None):
    """4-frame teacher-forcing windows (history 3, num_preds 1) over a sequence."""
    if frames < HISTORY + 1:
        raise ValueError('sequence shorter than one window')
    return torch.arange(frames - HISTORY, device=device)[:, None] + torch.arange(HISTORY + 1, device=device)[None]


def teacher_forced_loss(model, emb, valid, act_emb, block_valid):
    """le-wm loss over every 4-frame window; returns (loss, diagnostics).

    A target camera is scored when it has an exposure, every input frame up to the
    predicting position has at least one exposure, and every action block up to it is
    complete (padded commands past an episode end are zeros, not actions). Targets keep
    their gradient.
    """
    b, s, c, d = emb.shape
    if valid.shape != (b, s, c) or act_emb.shape[:2] != (b, s - 1) or block_valid.shape != (b, s - 1):
        raise ValueError('sequence, validity and action blocks disagree')
    idx = window_indices(s, emb.device)                       # [W, 4]
    w = idx.shape[0]
    win, win_valid = emb[:, idx], valid[:, idx]               # [B, W, 4, C, D]
    ctx = win[:, :, :HISTORY].reshape(b * w, HISTORY, c, d)
    ctx_valid = win_valid[:, :, :HISTORY].reshape(b * w, HISTORY, c)
    ctx_act = act_emb[:, idx[:, :HISTORY]].reshape(b * w, HISTORY, -1)
    ctx_block = block_valid[:, idx[:, :HISTORY]].reshape(b * w, HISTORY)
    target = win[:, :, 1:].reshape(b * w, HISTORY, c, d)
    target_valid = win_valid[:, :, 1:].reshape(b * w, HISTORY, c)
    pred = model.predict(ctx, ctx_valid, ctx_act)
    admissible = (ctx_valid.any(-1) & ctx_block).int().cumprod(1).bool()
    scored = target_valid & admissible[..., None]
    if not scored.any():
        raise ValueError('no scored target in the batch')
    error = (pred - target.float()).square().mean(-1)
    loss = error[scored].mean()
    with torch.no_grad():
        previous = ctx.float()                                # persistence: the frame before the target
        persistence = (previous - target.float()).square().mean(-1)
        paired = scored & ctx_valid
        diag = dict(pred_mse=float(loss), scored=int(scored.sum()), windows=w,
                    pred_mse_paired=float(error[paired].mean()) if paired.any() else None,
                    persistence_mse=float(persistence[paired].mean()) if paired.any() else None)
    return loss, diag


def latent_goal_cost(predicted_last, goal, goal_valid):
    """le-wm planning cost: MSE between the last predicted embedding and the goal embedding,
    averaged over cameras that have a goal exposure. predicted_last [N, C, D], goal [C, D]."""
    error = (predicted_last.float() - goal.float()[None]).square().mean(-1)
    keep = goal_valid.to(error.device)[None].expand_as(error)
    return (error * keep).sum(-1) / keep.sum(-1).clamp_min(1)
