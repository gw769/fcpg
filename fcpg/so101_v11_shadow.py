"""Non-actuating V11 shadow runner for production SO-101 deployment.

Loads the V11 (SO101V10) checkpoint and runs a forward pass alongside every
production plan call.  The result is logged for monitoring; no V11 command
ever reaches the robot.

Input contract (must match v10/data.py V10TrajectoryDataset exactly)
---------------------------------------------------------------------
* RGB:               uint8 → float32 / 255.0   (NO ImageNet mean/std)
* state_history:     LiveObservation.state is LeRobot raw units
                     (arm ~[-100,100], gripper ~[0,100]). The model sees
                     ``state / JOINT_SCALE`` with JOINT_SCALE=100, same as
                     ``v10/data.py`` / ``v10/live_pilot.py``.
* past_action_chunks: ``action_landmarks`` — intervals from history_offsets
                     pairs, 5 linspaced deque indices, observed states as
                     the commanded-action proxy, then / JOINT_SCALE.
* camera_context:    CAMERA_CONTEXT constant [(1,0,0,0,0),(0,1,0,0,0)],
                     same as training (v10/data.py:46-49).
* anchor_xyz_history: FK on raw LeRobot units, metres, no /100.
* outputs:           action chunks and future_state are unscaled (*100)
                     back to LeRobot units before the hardware supervisor.

Design constraints
------------------
* NO control output reaches the robot from this module.
* history_offsets come from the checkpoint V10Config (V11 long history
  is (-60, ..., 0); V11.1/V11.3 are (-5, 0)).  The runner requires
  len(observations) >= 1 - min(offsets) and skip-returns otherwise.
* Only planner.fits and planner.kinematics are borrowed from the planner.
  State normalization is NOT taken from the planner (planner has its own
  per-joint stats). V11 uses the fixed /100 scale from v10/data.py.
* All tensors are computed without grad.  Any error is caught and returned
  as {"shadow_ok": False, "error": ...} — never propagates into control.
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from v10.canonical_actor_sample import action_landmarks, scale_joints, unscale_joints
from v10.data import JOINT_SCALE

if TYPE_CHECKING:
    from fcpg.so101_realtime_planner import SO101RealtimePlanner

logger = logging.getLogger(__name__)

V11_6_FAMILY = "v11_6"
V12_FAMILY = "v12"
V11_6_RANK_USES = "v11_6_2view"
V11_MEMORY_GOAL_FAMILIES = ("v11_3", "v11_4", "v11_5", "v11_6", "v12")
# V12 co-trains the same two-view architecture on a broader dynamics pool, so it
# loads and ranks exactly as v11_6 does; only the training data differs.
TWO_VIEW_TASK_CLASS_FAMILIES = (V11_6_FAMILY, V12_FAMILY)

_DEFAULT_HISTORY_OFFSETS: tuple[int, ...] = (-60, -30, -15, -10, -5, 0)
_VIEWS = ("wrist", "column")

# camera_context constant from v10/data.py:46-49
_CAMERA_CONTEXT = np.asarray(
    ((1.0, 0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 0.0)),
    dtype=np.float32,
)  # shape [V=2, 5]

# Safety thresholds in *normalized* units (raw / JOINT_SCALE).
# 0.35 ≈ 35°; the hardware supervisor still applies max_command_jump (8°).
# Do not raise these to paper over a missing /100.
_MAX_ARM_JUMP = 0.35
_MAX_GRIPPER_JUMP = 0.35


def v11_constructor_kwargs(
    arch: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Build SO101V10 kwargs from a checkpoint architecture block."""

    family = arch.get("family")
    if family in TWO_VIEW_TASK_CLASS_FAMILIES and int(getattr(config, "num_views", 0)) != 2:
        raise ValueError(
            f"{family} requires num_views=2, got {getattr(config, 'num_views', None)}"
        )
    kwargs: dict[str, Any] = {}
    if family in V11_MEMORY_GOAL_FAMILIES:
        kwargs = {
            "structured_goal": True,
            "structured_goal_source": "episode_terminal_state",
            "goal_query_residual": False,
            "task_class_conditioning": family in TWO_VIEW_TASK_CLASS_FAMILIES,
        }
    elif family == "v11_2" or arch.get("structured_goal") == "target_future_state_h4":
        kwargs = {
            "structured_goal": True,
            "structured_goal_source": "target_future_state_h4",
            "goal_query_residual": True,
        }
    # The residual bound carries no parameters, so a checkpoint trained under one
    # bound loads silently under another and the rollout it produces live is not
    # the rollout it was trained to produce. Read it back from the checkpoint.
    mode = arch.get("state_residual_mode")
    if mode is not None:
        kwargs["state_residual_mode"] = str(mode)
        steps = arch.get("state_residual_steps")
        if steps is not None:
            kwargs["state_residual_steps"] = float(steps)
    return kwargs


def load_v11_policy(
    checkpoint_path: str | Path,
    vit_checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[Any, Any, str | None, dict[str, Any]]:
    """Load a V11 family policy. Does not require a planner or HTTP server."""

    from v10.config import V10Config
    from v10.model import SO101V10

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    raw_cfg = dict(ckpt["config"])
    for key in ("history_offsets", "future_offsets"):
        if isinstance(raw_cfg.get(key), list):
            raw_cfg[key] = tuple(raw_cfg[key])
    config = V10Config(**raw_cfg)
    arch = dict(ckpt.get("architecture") or {})
    family = arch.get("family")
    goal_kwargs = v11_constructor_kwargs(arch, config)
    has_vit = any("vit." in key for key in ckpt["model"].keys())
    if has_vit:
        try:
            model = SO101V10(
                config=config,
                vit_checkpoint=str(vit_checkpoint_path),
                **goal_kwargs,
            )
        # OSError covers missing and unreadable; UnpicklingError covers a
        # truncated or corrupt ViT file, which otherwise escaped as a bare
        # "invalid load key" that never named the file at fault.
        except (ValueError, OSError, TypeError, pickle.UnpicklingError) as exc:
            # This used to fall back to a bare vit_hf grafted onto an
            # uninitialised model: no requires_grad_(False), no eval pin, no
            # 5,501,376-parameter assertion, and no policy/bridge/corrector at
            # all -- so the strict load_state_dict below could not have
            # succeeded anyway. Refuse the unfrozen ViT explicitly instead of
            # failing later with an unrelated key mismatch.
            raise ValueError(
                f"frozen V9 ViT init failed for {checkpoint_path}: "
                f"vit_checkpoint={vit_checkpoint_path} ({type(exc).__name__}: "
                f"{exc}). Refusing to substitute an unfrozen ViT built outside "
                "FrozenV9ViT's freeze / eval / 5,501,376-parameter checks."
            ) from exc
        model.load_state_dict(ckpt["model"])
    else:
        model = SO101V10(
            config=config,
            vit_checkpoint=str(vit_checkpoint_path),
            **goal_kwargs,
        )
        model.load_state_dict(ckpt["model"])
    contract = ckpt.get("action_contract")
    if contract:
        model.set_action_contract(
            torch.as_tensor(contract["step_cap"], dtype=torch.float32),
            torch.as_tensor(contract["lower"], dtype=torch.float32),
            torch.as_tensor(contract["upper"], dtype=torch.float32),
        )
    model.eval().to(torch.device(device))
    return model, config, family, arch


def _rgb_to_float(frames_uint8: np.ndarray) -> torch.Tensor:
    """[T, 224, 224, 3] uint8 → [T, 3, 224, 224] float32 / 255.0.

    Matches training: result[name] = result[name].float().div_(255.0)
    NO ImageNet mean/std subtraction.
    """
    t = torch.from_numpy(frames_uint8.copy()).permute(0, 3, 1, 2).float()
    return t.div_(255.0)


def v11_family_constructor_kwargs(
    family: str | None,
    architecture: dict[str, Any] | None = None,
) -> dict[str, object]:
    """SO101V10 kwargs for a V11 family. LeWM never receives goal/task_class."""

    arch = architecture or {}
    if family in V11_MEMORY_GOAL_FAMILIES:
        return {
            "structured_goal": True,
            "structured_goal_source": "episode_terminal_state",
            "goal_query_residual": False,
            "task_class_conditioning": family in TWO_VIEW_TASK_CLASS_FAMILIES,
        }
    if family == "v11_2" or arch.get("structured_goal") == "target_future_state_h4":
        return {
            "structured_goal": True,
            "structured_goal_source": "target_future_state_h4",
            "goal_query_residual": True,
        }
    return {}


def _action_landmarks(
    obs_states: np.ndarray,
    current_idx: int,
    offsets: tuple[int, ...],
) -> np.ndarray:
    """Replicate V10TrajectoryDataset._action_landmarks using deque states.

    Observed states stand in for commanded actions, then / JOINT_SCALE.
    """
    return action_landmarks(obs_states, current_idx, offsets)


def decode_v11_action_chunk(
    action_chunks_np: np.ndarray,
    current_state_raw: np.ndarray,
    *,
    raw_steps_per_chunk: int = 5,
    state_dim: int = 6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Unscale the first predicted chunk and score jumps in normalized units.

    ``action_chunks_np`` is the model output ([H, 30], already /100).
    ``current_state_raw`` is live LeRobot units. The returned chunk is *100
    so ``HardwareSO101IO.execute_chunk`` can send it.
    """
    first_norm = np.asarray(action_chunks_np[0], dtype=np.float32).reshape(
        raw_steps_per_chunk, state_dim
    )
    cur_norm = scale_joints(current_state_raw)
    jumps = np.abs(first_norm - cur_norm[None, :])
    max_arm_jump = float(jumps[:, :5].max())
    max_gripper_jump = float(jumps[:, 5].max())
    nan_inf = not np.all(np.isfinite(first_norm))
    first_live = unscale_joints(first_norm)
    return first_live, {
        "max_arm_joint_jump": max_arm_jump,
        "max_gripper_jump": max_gripper_jump,
        "max_arm_joint_jump_deg": max_arm_jump * float(JOINT_SCALE),
        "max_gripper_jump_deg": max_gripper_jump * float(JOINT_SCALE),
        "nan_inf": nan_inf,
        "arm_safe": (not nan_inf) and max_arm_jump <= _MAX_ARM_JUMP,
        "gripper_safe": (not nan_inf) and max_gripper_jump <= _MAX_GRIPPER_JUMP,
        "first_chunk_norm": first_norm,
    }


class V11ShadowRunner:
    """Run V11 in shadow mode alongside the production planner.

    Parameters
    ----------
    checkpoint_path:
        Path to the V11 best checkpoint.
    vit_checkpoint_path:
        Path to the frozen V9 ViT init checkpoint.
    planner:
        The live production planner.  Used for fits and kinematics only.
    device:
        Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        vit_checkpoint_path: str | Path,
        planner: "SO101RealtimePlanner",
        *,
        device: str = "cuda:0",
    ) -> None:
        self.device = torch.device(device)
        model, config, family, arch = load_v11_policy(
            checkpoint_path,
            vit_checkpoint_path,
            device=device,
        )
        goal_kwargs = v11_constructor_kwargs(arch, config)
        self.model = model
        self.config = config
        self.family = family
        self.architecture = arch
        self.rank_uses = (
            V11_6_RANK_USES if family in TWO_VIEW_TASK_CLASS_FAMILIES else None
        )
        self.history_offsets = tuple(config.history_offsets)
        self.min_obs = 1 - min(self.history_offsets)
        self.live_goal_proxy = (
            "current_state" if goal_kwargs.get("structured_goal") else None
        )

        # FK geometry from planner — read-only references, never modified here.
        # Live state is LeRobot raw; the model sees raw / JOINT_SCALE.
        self.fits       = planner.fits
        self.kinematics = planner.kinematics

        logger.info(
            "V11ShadowRunner ready: family=%s num_views=%d task_class=%s "
            "rank_uses=%s obs_horizon=%d history=%s action_horizon=%d "
            "live_goal_proxy=%s device=%s",
            family,
            int(config.num_views),
            bool(goal_kwargs.get("task_class_conditioning")),
            self.rank_uses,
            config.obs_horizon,
            self.history_offsets,
            config.action_horizon,
            self.live_goal_proxy,
            device,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def rank_prediction_steps(self) -> int:
        """Chunk states a rank rollout genuinely predicts, before any padding.

        ``rollout_candidate_raw_states`` pads its output up to whatever horizon
        the planner asks for by repeating the final predicted state, so the
        length of what it returns says nothing about how far this model saw.
        The policy emits exactly one state per configured future offset, and
        that prefix is the only part of the rollout a cost function may read as
        a prediction: the tail is the same state over again, which is
        indistinguishable from a genuinely stationary trajectory.
        """

        return len(tuple(self.config.future_offsets))

    def predict(
        self,
        observations: deque,
        current_state: np.ndarray,
        goal_state_raw: np.ndarray | None = None,
        task_class: int | None = None,
    ) -> dict[str, Any]:
        """Run one V11 shadow forward pass.

        Parameters
        ----------
        observations:
            Runtime's ``self.observations`` deque (maxlen=71).
            Each entry is a LiveObservation with .state in LeRobot raw
            units and .frames as uint8 [224,224,3].
        current_state:
            Current arm state [6] in LeRobot raw units.
        goal_state_raw:
            Optional structured-goal proprio in the same raw units.
            Default is ``current_state`` (deploy proxy for
            ``episode_terminal_state``).

        Returns
        -------
        dict with shadow result.  ``first_chunk`` is LeRobot raw units.
        ``shadow_ok`` is False when the pass was skipped or errored.
        """
        try:
            return self._predict_inner(
                observations,
                current_state,
                goal_state_raw=goal_state_raw,
                task_class=task_class,
            )
        except Exception as exc:
            logger.warning("V11 shadow prediction raised: %s", exc, exc_info=True)
            return {"shadow_ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _predict_inner(
        self,
        observations: deque,
        current_state: np.ndarray,
        goal_state_raw: np.ndarray | None = None,
        task_class: int | None = None,
    ) -> dict[str, Any]:
        n_obs = len(observations)
        if n_obs < self.min_obs:
            return {
                "shadow_ok": False,
                "reason": f"insufficient_history:{n_obs}<{self.min_obs}",
            }

        T   = self.config.obs_horizon
        V   = len(_VIEWS)                    # 2
        rsc = self.config.raw_steps_per_chunk   # 5
        sd  = self.config.state_dim             # 6
        offsets = self.history_offsets

        # ---- 1. Sample history from deque --------------------------------
        obs_list = list(observations)
        # offset k → list index (n_obs - 1 + k) — same as deque[k-1]
        history_indices = [n_obs - 1 + off for off in offsets]

        states_hist_raw = np.stack(
            [obs_list[i].state for i in history_indices], axis=0
        ).astype(np.float32)  # [T, 6] LeRobot raw; FK stays in this space
        states_hist = scale_joints(states_hist_raw)

        frames_hist = {
            view: np.stack(
                [obs_list[i].frames[view] for i in history_indices], axis=0
            )  # [T, 224, 224, 3] uint8
            for view in _VIEWS
        }

        # ---- 2. FK projection for anchor_xyz and q_nominal ---------------
        from scripts.evaluate_so101_real_shadow_fcpg_lewm import project_history

        base_xyz, nominal, valid = project_history(
            states_hist_raw,
            self.fits,
            self.kinematics,
            image_size=(224, 224),
        )
        # base_xyz: [T, A, 3] metres — no normalization, matches training

        # Guard: model requires finite anchor_xyz
        if not np.all(np.isfinite(base_xyz)):
            finite_mask = np.isfinite(base_xyz).all(axis=-1)  # [T, A]
            base_xyz = np.where(np.isfinite(base_xyz), base_xyz, 0.0)
            for view in _VIEWS:
                valid[view] = valid[view] & finite_mask

        # ---- 3. Build tensors --------------------------------------------
        # RGB history [1, T, V, 3, 224, 224] — /255, no ImageNet
        rgb_views = [
            _rgb_to_float(frames_hist[view]).to(self.device)  # [T, 3, H, W]
            for view in _VIEWS
        ]
        rgb_history = torch.stack(rgb_views, dim=1).unsqueeze(0)

        # state_history [1, T, 6] — raw / JOINT_SCALE
        state_tensor = (
            torch.from_numpy(states_hist).unsqueeze(0).to(self.device)
        )

        # anchor_xyz_history [1, T, A, 3]
        anchor_xyz = (
            torch.from_numpy(base_xyz).unsqueeze(0).to(self.device)
        )

        # q_nominal_px [1, T, V, A, 2]
        q_nominal_np = np.stack(
            [nominal[view] for view in _VIEWS], axis=1
        )  # [T, V, A, 2]
        q_nominal_t = (
            torch.from_numpy(q_nominal_np).unsqueeze(0).to(self.device)
        )

        # anchor_valid [1, T, V, A]
        anchor_valid_np = np.stack(
            [valid[view] for view in _VIEWS], axis=1
        ).astype(bool)  # [T, V, A]
        anchor_valid_t = (
            torch.from_numpy(anchor_valid_np).unsqueeze(0).to(self.device)
        )

        # view_valid [1, T, V]
        view_valid_t = anchor_valid_t.any(dim=-1)

        # Guard: model raises if no valid view for any batch item
        if not view_valid_t.any().item():
            return {"shadow_ok": False, "reason": "no_valid_view"}

        # camera_context [1, T, V, 5] — constant, matches training
        cam_np = np.broadcast_to(_CAMERA_CONTEXT[None], (T, V, 5)).copy()
        cam_ctx = (
            torch.from_numpy(cam_np).unsqueeze(0).to(self.device)
        )

        # ---- 4. past_action_chunks [1, obs_horizon-1, rsc*sd] ------------
        # Replicate _action_landmarks: one interval per consecutive offset pair.
        all_states = np.stack([obs.state for obs in obs_list], axis=0)  # [n_obs, 6]
        pac_np = _action_landmarks(all_states, n_obs - 1, offsets)
        past_action_t = (
            torch.from_numpy(pac_np).unsqueeze(0).to(self.device)
        )

        goal_state = None
        goal_raw = current_state if goal_state_raw is None else goal_state_raw
        if getattr(self.model.policy, "structured_goal", False):
            # Training used episode_terminal_state. Live may pass a planner
            # approach pose; otherwise current_state is the documented proxy.
            goal_state = (
                torch.from_numpy(scale_joints(goal_raw))
                .unsqueeze(0)
                .to(self.device)
            )

        # ---- 5. Forward pass (no grad) -----------------------------------
        with torch.no_grad():
            grounded = self.model.encode_grounded_history(
                rgb=rgb_history,
                q_nominal_px=q_nominal_t,
                anchor_xyz_history=anchor_xyz,
                state_history=state_tensor,
                camera_context=cam_ctx,
                anchor_valid=anchor_valid_t,
                view_valid=view_valid_t,
            )
            tag = None
            if getattr(self.model.policy, "task_class_conditioning", False):
                if task_class is None:
                    return {"shadow_ok": False, "reason": "v11_6_requires_task_class"}
                tag = torch.as_tensor(
                    [int(task_class)], device=self.device, dtype=torch.long
                )
            prediction = self.model.policy(
                grounded.context,
                grounded.anchor_tokens,
                grounded.anchor_valid,
                grounded.view_valid,
                grounded.q_corrected_px,
                anchor_xyz,
                state_tensor,
                past_action_t,
                forced_dynamics_action=None,
                goal_state=goal_state,
                task_class=tag,
            )

        # ---- 6. Decode output -------------------------------------------
        # prediction.action_chunks: [1, future_horizon=4, rsc*sd] where rsc=5, sd=6
        # prediction.future_state: [1, future_horizon=4, 6] already / JOINT_SCALE
        action_chunks_np = prediction.action_chunks[0].cpu().numpy()  # [4, 30]
        future_state_np = prediction.future_state[0].cpu().numpy()    # [4, 6]
        first_chunk, safety = decode_v11_action_chunk(
            action_chunks_np,
            current_state,
            raw_steps_per_chunk=rsc,
            state_dim=sd,
        )
        predicted_next_state = unscale_joints(future_state_np[0])
        shadow_pass = (
            (not safety["nan_inf"])
            and safety["arm_safe"]
            and safety["gripper_safe"]
        )

        return {
            "shadow_ok": True,
            "v11_shadow_pass": shadow_pass,
            "nan_inf": safety["nan_inf"],
            "max_arm_joint_jump": round(safety["max_arm_joint_jump"], 3),
            "max_gripper_jump": round(safety["max_gripper_jump"], 3),
            "max_arm_joint_jump_deg": round(safety["max_arm_joint_jump_deg"], 3),
            "max_gripper_jump_deg": round(safety["max_gripper_jump_deg"], 3),
            "arm_safe": safety["arm_safe"],
            "gripper_safe": safety["gripper_safe"],
            "predicted_next_state": predicted_next_state.tolist(),
            "first_chunk": first_chunk.astype(np.float32),
            "first_chunk_endpoint": first_chunk[-1].tolist(),
            "joint_scale": float(JOINT_SCALE),
            "n_obs_used": n_obs,
            "family": self.family,
            "task_class": None if tag is None else int(task_class),
            "num_views": int(self.config.num_views),
            "rank_uses": self.rank_uses,
            "goal_in_lewm": False,
            "history_offsets": list(offsets),
            "live_goal_proxy": (
                "supplied_goal" if goal_state_raw is not None else self.live_goal_proxy
            ),
            "view_valid_current": view_valid_t[0, -1].cpu().numpy().tolist(),
        }

    def rollout_candidate_raw_states(
        self,
        observations: deque,
        prefixes: np.ndarray,
        *,
        horizon: int,
    ) -> np.ndarray | None:
        """Goal-free 2-view LeWM rollout. Returns raw LeRobot states [K, H, 6]."""

        from fcpg.so101_v11_6_lewm_rank import (
            batched_lewm_canonical_forward,
            snapshot_rank_observations,
        )

        obs_snap = snapshot_rank_observations(observations)
        values = np.asarray(prefixes, dtype=np.float32).copy()
        if values.ndim != 3 or values.shape[-1] != 6:
            raise ValueError("prefixes must have shape [K, R, 6]")
        horizon_out = int(horizon)
        if horizon_out < 1:
            raise ValueError("horizon must be positive")
        # Encode uses shared MuJoCo FK (already locked). Do not hold the CUDA
        # rank lock here: IK holds the FK lock and a nested opposite order
        # deadlocks under ThreadingHTTPServer.
        encoded = self._encode_live_history(obs_snap)
        if encoded is None:
            return None
        action_horizon = int(self.config.action_horizon)
        if values.shape[1] < action_horizon:
            pad = np.repeat(values[:, -1:, :], action_horizon - values.shape[1], axis=1)
            action_raw = np.concatenate((values, pad), axis=1)
        else:
            action_raw = values[:, :action_horizon]
        action = torch.as_tensor(
            scale_joints(action_raw), device=self.device, dtype=torch.float32
        )
        n_cand = int(action.shape[0])

        def _expand(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.expand(n_cand, *tensor.shape[1:]).contiguous()

        future, _latent = batched_lewm_canonical_forward(
            self.model.policy,
            action=action,
            per_view_context=_expand(encoded["context"]),
            per_view_anchor_tokens=_expand(encoded["anchor_tokens"]),
            anchor_valid=_expand(encoded["anchor_valid"]),
            view_valid=_expand(encoded["view_valid"]),
            q_corrected_px=_expand(encoded["q_corrected_px"]),
            anchor_xyz_history=_expand(encoded["anchor_xyz"]),
            state_history=_expand(encoded["state"]),
            past_action_chunks=_expand(encoded["past_action"]),
        )
        future_raw = unscale_joints(future.float().cpu().numpy())
        # The declared offset count is what the planner clamps its terminal cost
        # to, so it has to match the tensor rather than merely describe it. If
        # these ever disagree the padding below is mis-sized and the cost reads
        # duplicated states while believing they are predictions -- fail closed.
        declared = self.rank_prediction_steps
        if future_raw.shape[1] != declared:
            raise RuntimeError(
                f"LeWM predicted {future_raw.shape[1]} chunk states but declares "
                f"{declared} future offsets {tuple(self.config.future_offsets)}; "
                "refusing to pad a rollout whose real length is unknown"
            )
        if future_raw.shape[1] >= horizon_out:
            return future_raw[:, :horizon_out].astype(np.float32)
        pad_state = np.repeat(
            future_raw[:, -1:, :], horizon_out - future_raw.shape[1], axis=1
        )
        return np.concatenate((future_raw, pad_state), axis=1).astype(np.float32)

    def _encode_live_history(self, observations: deque) -> dict[str, torch.Tensor] | None:
        n_obs = len(observations)
        if n_obs < self.min_obs:
            return None
        T = self.config.obs_horizon
        V = len(_VIEWS)
        offsets = self.history_offsets
        obs_list = list(observations)
        history_indices = [n_obs - 1 + off for off in offsets]
        states_hist_raw = np.stack(
            [obs_list[i].state for i in history_indices], axis=0
        ).astype(np.float32)
        frames_hist = {
            view: np.stack(
                [obs_list[i].frames[view] for i in history_indices], axis=0
            )
            for view in _VIEWS
        }
        from scripts.evaluate_so101_real_shadow_fcpg_lewm import project_history

        base_xyz, nominal, valid = project_history(
            states_hist_raw,
            self.fits,
            self.kinematics,
            image_size=(224, 224),
        )
        if not np.all(np.isfinite(base_xyz)):
            finite_mask = np.isfinite(base_xyz).all(axis=-1)
            base_xyz = np.where(np.isfinite(base_xyz), base_xyz, 0.0)
            for view in _VIEWS:
                valid[view] = valid[view] & finite_mask
        rgb_views = [
            _rgb_to_float(frames_hist[view]).to(self.device) for view in _VIEWS
        ]
        rgb_history = torch.stack(rgb_views, dim=1).unsqueeze(0)
        state_tensor = torch.from_numpy(scale_joints(states_hist_raw)).unsqueeze(0).to(
            self.device
        )
        anchor_xyz = torch.from_numpy(base_xyz).unsqueeze(0).to(self.device)
        q_nominal_np = np.stack([nominal[view] for view in _VIEWS], axis=1)
        q_nominal_t = torch.from_numpy(q_nominal_np).unsqueeze(0).to(self.device)
        anchor_valid_np = np.stack([valid[view] for view in _VIEWS], axis=1).astype(bool)
        anchor_valid_t = torch.from_numpy(anchor_valid_np).unsqueeze(0).to(self.device)
        view_valid_t = anchor_valid_t.any(dim=-1)
        if not view_valid_t.any().item():
            return None
        cam_np = np.broadcast_to(_CAMERA_CONTEXT[None], (T, V, 5)).copy()
        cam_ctx = torch.from_numpy(cam_np).unsqueeze(0).to(self.device)
        all_states = np.stack([obs.state for obs in obs_list], axis=0)
        pac_np = _action_landmarks(all_states, n_obs - 1, offsets)
        past_action_t = torch.from_numpy(pac_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            grounded = self.model.encode_grounded_history(
                rgb=rgb_history,
                q_nominal_px=q_nominal_t,
                anchor_xyz_history=anchor_xyz,
                state_history=state_tensor,
                camera_context=cam_ctx,
                anchor_valid=anchor_valid_t,
                view_valid=view_valid_t,
            )
        return {
            "context": grounded.context,
            "anchor_tokens": grounded.anchor_tokens,
            "anchor_valid": grounded.anchor_valid,
            "view_valid": grounded.view_valid,
            "q_corrected_px": grounded.q_corrected_px,
            "anchor_xyz": anchor_xyz,
            "state": state_tensor,
            "past_action": past_action_t,
        }


__all__ = [
    "V11ShadowRunner",
    "V11_6_FAMILY",
    "V12_FAMILY",
    "TWO_VIEW_TASK_CLASS_FAMILIES",
    "V11_6_RANK_USES",
    "decode_v11_action_chunk",
    "load_v11_policy",
    "v11_constructor_kwargs",
    "v11_family_constructor_kwargs",
]
