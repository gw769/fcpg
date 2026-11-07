"""No-hardware live adapter and offline gates for Prod-V17.7.

This module owns frozen DINO extraction, the calibrated fish-rig projection,
and diagnostic-only policy probes.  It deliberately owns no camera device,
serial port, robot transport, clearance policy, or task target.  Fish boxes
used by the offline counterfactual are image edits performed by the checker;
they are never passed to :class:`V177Core` or the live runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from fcpg.prod_v17_6 import _project_action_chunk
from fcpg.prod_v17_7 import (
    ANCHORS,
    BELIEF_CLASSES,
    COMMAND_STEPS,
    GRID_CELLS,
    GRID_SIZE,
    IMAGE_SIZE,
    PATH_STEPS,
    STATE_DIM,
    VIEWS,
    VIEW_ORDER,
    V177Config,
    V177Core,
)
from fcpg.prod_v17_7_runtime import V177RuntimeController
from fcpg.prod_v17_7_training import (
    ARCHITECTURE_CONTRACT,
    SEED,
    VERSION,
    V177GeometryIdentityProvider,
)
from fcpg.prod_v18_1_training import extract_dinov2_tokens, load_frozen_dinov2


POSITION_UNIT = "lerobot_calibrated_normalized_position"
FISH_SOURCE_INDEX = 129
EXPECTED_MODEL_STATE_KEYS = 363
EXPECTED_ACTOR_CONTEXT = (
    "current_scene",
    "dynamics_grounding",
    "posterior_pooled_appearance",
    "innovation",
    "current_q",
)
MIN_ANALYTIC_SPAN_PX = 8.0
MIN_BELIEF_TO_ANALYTIC_SPAN = 0.25
MIN_ANALYTIC_TRANSPORT_PX = 4.0
MIN_FINAL_TRANSPORT_PX = 1.0
MIN_TRANSPORT_COSINE = 0.5


class V177LiveError(RuntimeError):
    """Raised before malformed live data reaches V17.7 or hardware code."""


def load_strict_v177_full_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device | str,
) -> tuple[V177Core, dict[str, Any]]:
    """Load exactly the completed V17.7 ``full`` Actor checkpoint.

    A generic training loader checks version/component and strict state keys,
    but deployment additionally fixes the seed, mode, complete dataclass
    configuration, and expected 363-key architecture.  A truncated or
    partially copied checkpoint therefore fails before DINO or runtime setup.
    """

    target = torch.device(device)
    source = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise V177LiveError(f"V17.7 checkpoint is absent: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as error:
        raise V177LiveError(f"V17.7 checkpoint cannot be read: {source}") from error
    if not isinstance(payload, dict):
        raise V177LiveError("V17.7 checkpoint must be a mapping")
    expected_metadata = {
        "version": VERSION,
        "architecture": ARCHITECTURE_CONTRACT,
        "component": "actor",
        "seed": SEED,
    }
    observed = {name: payload.get(name) for name in expected_metadata}
    if observed != expected_metadata:
        raise V177LiveError(
            f"V17.7 checkpoint metadata differs: observed={observed} "
            f"expected={expected_metadata}"
        )
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise V177LiveError("V17.7 checkpoint config must be a mapping")
    config_fields = {field.name for field in fields(V177Config)}
    if set(raw_config) != config_fields:
        missing = sorted(config_fields - set(raw_config))
        extra = sorted(set(raw_config) - config_fields)
        raise V177LiveError(
            f"V17.7 checkpoint config keys differ: missing={missing} extra={extra}"
        )
    try:
        config = V177Config(**raw_config)
        config.validate()
    except Exception as error:
        raise V177LiveError("V17.7 checkpoint config is invalid") from error
    if config.actor_mode != "full":
        raise V177LiveError("live policy requires the V17.7 full Actor")
    if not config.analytic_transport or not config.geometry_prior_fusion:
        raise V177LiveError(
            "live policy requires analytic transport and H0 geometry-prior fusion"
        )
    if int(config.source_vocabulary) <= FISH_SOURCE_INDEX:
        raise V177LiveError("V17.7 source vocabulary does not contain fish index 129")

    state = payload.get("model")
    if not isinstance(state, dict):
        raise V177LiveError("V17.7 checkpoint model must be a state mapping")
    model = V177Core(config)
    expected_keys = set(model.state_dict())
    if len(expected_keys) != EXPECTED_MODEL_STATE_KEYS:
        raise V177LiveError(
            f"local V17.7 architecture has {len(expected_keys)} keys, expected "
            f"{EXPECTED_MODEL_STATE_KEYS}"
        )
    if len(state) != EXPECTED_MODEL_STATE_KEYS or set(state) != expected_keys:
        missing = sorted(expected_keys - set(state))
        extra = sorted(set(state) - expected_keys)
        raise V177LiveError(
            "V17.7 checkpoint is not the complete 363-key model: "
            f"count={len(state)} missing={missing[:8]} extra={extra[:8]}"
        )
    for name, value in state.items():
        if not torch.is_tensor(value):
            raise V177LiveError(f"checkpoint state {name!r} is not a tensor")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise V177LiveError(f"checkpoint state {name!r} contains non-finite values")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as error:
        raise V177LiveError("V17.7 checkpoint failed strict model loading") from error
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("actor_mode") != "full":
        raise V177LiveError("V17.7 checkpoint metrics do not identify the full Actor")
    source_conditioning = metrics.get("source_conditioning")
    actor_context = metrics.get("actor_context")
    if (
        metrics.get("run_kind") != "full"
        or metrics.get("innovation_encoder_trained") is not True
        or metrics.get("raw_actor_history_input") is not False
        or not isinstance(actor_context, (list, tuple))
        or tuple(actor_context) != EXPECTED_ACTOR_CONTEXT
        or not isinstance(source_conditioning, dict)
        or source_conditioning.get("fish_index") != FISH_SOURCE_INDEX
    ):
        raise V177LiveError("V17.7 checkpoint deployment contract is incomplete")
    model.requires_grad_(False).to(target).eval()
    return model, payload


def _inward_only_action_bounds(
    current: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    if (
        current.shape != (STATE_DIM,)
        or lower.shape != (STATE_DIM,)
        or upper.shape != (STATE_DIM,)
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
    ):
        raise V177LiveError("action bounds must be finite ordered six-vectors")
    effective_lower = lower.copy()
    effective_upper = upper.copy()
    effective_lower[:5] = np.minimum(effective_lower[:5], current[:5])
    effective_upper[:5] = np.maximum(effective_upper[:5], current[:5])
    return effective_lower, effective_upper


def _nonmonotonic_recovery_joints(
    commands: np.ndarray,
    current: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tolerance: float = 1.0e-4,
) -> tuple[int, ...]:
    commands = np.asarray(commands, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    if commands.shape != (COMMAND_STEPS, STATE_DIM):
        raise V177LiveError("commands must be [5,6]")
    if not all(np.all(np.isfinite(value)) for value in (commands, current, lower, upper)):
        raise V177LiveError("recovery audit contains non-finite values")
    previous = np.maximum.reduce((lower[:5] - current[:5], current[:5] - upper[:5], np.zeros(5)))
    tracked = previous > float(tolerance)
    failed = np.zeros(5, dtype=bool)
    for row in commands:
        violation = np.maximum.reduce((lower[:5] - row[:5], row[:5] - upper[:5], np.zeros(5)))
        failed |= tracked & (violation > previous + float(tolerance))
        previous = violation
    return tuple(np.flatnonzero(failed).tolist())


def belief_mean_px(log_belief: torch.Tensor) -> torch.Tensor:
    """Conditional-visible posterior mean, for reporting only."""

    probability = torch.softmax(torch.as_tensor(log_belief).float(), dim=-1)
    if probability.ndim != 4 or tuple(probability.shape[1:]) != (
        VIEWS,
        ANCHORS,
        BELIEF_CLASSES,
    ):
        raise V177LiveError("belief must be [B,2,2,3137]")
    visible = probability[..., :GRID_CELLS]
    visible = visible / visible.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    axis = torch.arange(GRID_SIZE, device=visible.device, dtype=visible.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    centers = torch.stack((xx, yy), dim=-1).reshape(GRID_CELLS, 2) * 4.0 + 1.5
    return torch.einsum("...c,cd->...d", visible, centers)


def _vector_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    numerator = np.sum(first * second, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 1.0e-8,
    ).astype(np.float32)


def translate_image_region(
    image: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
    *,
    dx: int,
) -> np.ndarray:
    """Translate one annotated image crop horizontally for an offline probe.

    The annotation controls only this image edit.  No coordinate or mask is
    returned to the model.  The vacated rectangle is filled by the median RGB
    value of a narrow ring around the box before the crop is pasted.
    """

    source = np.asarray(image)
    if source.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or source.dtype != np.uint8:
        raise V177LiveError("counterfactual image must be uint8 [224,224,3]")
    x0, y0, x1, y1 = map(int, box_xyxy)
    if not (0 <= x0 < x1 <= IMAGE_SIZE and 0 <= y0 < y1 <= IMAGE_SIZE):
        raise V177LiveError("counterfactual box must lie inside the image")
    destination_x0 = x0 + int(dx)
    destination_x1 = x1 + int(dx)
    if not (0 <= destination_x0 < destination_x1 <= IMAGE_SIZE):
        raise V177LiveError("counterfactual translation leaves the image")
    output = source.copy()
    crop = source[y0:y1, x0:x1].copy()
    ring_x0, ring_x1 = max(0, x0 - 4), min(IMAGE_SIZE, x1 + 4)
    ring_y0, ring_y1 = max(0, y0 - 4), min(IMAGE_SIZE, y1 + 4)
    neighborhood = source[ring_y0:ring_y1, ring_x0:ring_x1]
    outside = np.ones(neighborhood.shape[:2], dtype=bool)
    outside[
        y0 - ring_y0 : y1 - ring_y0,
        x0 - ring_x0 : x1 - ring_x0,
    ] = False
    ring = neighborhood[outside]
    if not len(ring):
        raise V177LiveError("counterfactual box needs a surrounding background ring")
    fill = np.median(ring, axis=0).astype(np.uint8)
    output[y0:y1, x0:x1] = fill
    output[y0:y1, destination_x0:destination_x1] = crop
    return output


class V177LivePolicy:
    """Own frozen perception and the actual-endpoint V17.7 policy clock."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        dino_repository: str | Path,
        dino_weights: str | Path,
        real_calibration: str | Path,
        device: torch.device | str,
        action_lower_position: np.ndarray | None = None,
        action_upper_position: np.ndarray | None = None,
        source_index: int = FISH_SOURCE_INDEX,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise V177LiveError("CUDA was requested but is unavailable")
        if int(source_index) != FISH_SOURCE_INDEX:
            raise V177LiveError("live fish policy requires source_index=129")
        model, payload = load_strict_v177_full_checkpoint(
            checkpoint, device=self.device
        )
        if (action_lower_position is None) != (action_upper_position is None):
            raise V177LiveError("deployment action bounds must be supplied together")
        if action_lower_position is None:
            lower = model.actor.action_lower.detach().float().cpu().numpy()
            upper = model.actor.action_upper.detach().float().cpu().numpy()
        else:
            lower = np.asarray(action_lower_position, dtype=np.float32)
            upper = np.asarray(action_upper_position, dtype=np.float32)
        if (
            lower.shape != (STATE_DIM,)
            or upper.shape != (STATE_DIM,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
        ):
            raise V177LiveError("deployment action bounds are invalid")

        self.identity = V177GeometryIdentityProvider(
            real_calibration, device=self.device
        )
        self.runtime = V177RuntimeController(
            model,
            device=self.device,
            source_index=FISH_SOURCE_INDEX,
            analytic_transport_provider=self.identity.analytic_transport,
        )
        self.backbone = load_frozen_dinov2(
            dino_repository, dino_weights, device=self.device
        )
        self.action_lower_position = lower.copy()
        self.action_upper_position = upper.copy()
        self.effective_action_lower_position = lower.copy()
        self.effective_action_upper_position = upper.copy()
        self._install_action_bounds(lower, upper)
        self.action_source = "v17_7_camera_aware_predictive_belief_full_controller"
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.source_index = FISH_SOURCE_INDEX
        self.seed = int(payload["seed"])
        self.training_step = int(payload.get("step", -1))
        self.last_proposal: dict[str, Any] | None = None
        self.last_completion: dict[str, Any] | None = None
        # Rendering-only snapshot of the latest grounded belief.  Keeping it
        # outside ``last_proposal`` avoids placing the full 56x56 maps in the
        # status JSON, and it is never read by the runtime or Actor.
        self._render_log_posterior: np.ndarray | None = None

    def bind_source_state_encoder(
        self, encoder: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        if not callable(encoder):
            raise V177LiveError("source state encoder must be callable")

        def tensor_encoder(state: torch.Tensor) -> torch.Tensor:
            values = state[0].detach().float().cpu().numpy().copy()
            return torch.from_numpy(np.ascontiguousarray(np.asarray(encoder(values))))

        self.runtime.bind_source_state_encoder(tensor_encoder)

    def bind_command_admission_provider(
        self, provider: Callable[[np.ndarray, np.ndarray], None]
    ) -> None:
        """Bind the deployment's reject-only physical gate before inference."""

        if not callable(provider):
            raise V177LiveError("command admission provider must be callable")

        def tensor_provider(
            source_q: torch.Tensor, commands: torch.Tensor
        ) -> None:
            source = source_q[0].detach().float().cpu().numpy().copy()
            block = commands[0].detach().float().cpu().numpy().copy()
            result = provider(source, block)
            if result is not None:
                raise V177LiveError(
                    "command admission provider must return None or raise"
                )

        self.runtime.bind_command_admission_provider(tensor_provider)

    def _install_action_bounds(self, lower: np.ndarray, upper: np.ndarray) -> None:
        actor = self.runtime.model.actor
        with torch.no_grad():
            actor.action_lower.copy_(
                torch.from_numpy(np.asarray(lower, dtype=np.float32)).to(actor.action_lower)
            )
            actor.action_upper.copy_(
                torch.from_numpy(np.asarray(upper, dtype=np.float32)).to(actor.action_upper)
            )

    def _endpoint_tensors(
        self, frames: Mapping[str, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if set(frames) != set(VIEW_ORDER):
            raise V177LiveError(
                f"live frames must contain exactly {list(VIEW_ORDER)}"
            )
        ordered: list[np.ndarray] = []
        for view in VIEW_ORDER:
            frame = np.asarray(frames[view])
            if frame.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or frame.dtype != np.uint8:
                raise V177LiveError(
                    f"{view} must be uint8 [{IMAGE_SIZE},{IMAGE_SIZE},3]"
                )
            ordered.append(frame)
        images = np.ascontiguousarray(np.stack(ordered))
        encoded = extract_dinov2_tokens(
            self.backbone, images, device=self.device, batch_size=VIEWS
        )
        tokens = torch.from_numpy(np.ascontiguousarray(encoded)).to(
            self.device, dtype=torch.float32
        )
        rgb = torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2)
        valid = torch.ones(VIEWS, dtype=torch.bool, device=self.device)
        return rgb, tokens, valid

    def _geometry_prior(
        self, state: np.ndarray, valid: torch.Tensor
    ) -> torch.Tensor:
        q = torch.from_numpy(np.ascontiguousarray(state))[None].to(
            self.device, dtype=torch.float32
        )
        return self.identity.broad_log_prior(
            q,
            valid.reshape(1, VIEWS),
            sigma_px=float(self.runtime.model.config.geometry_prior_sigma_px),
        )

    def _proposal_diagnostics(
        self, proposal: Any, source_state: np.ndarray
    ) -> dict[str, Any]:
        grounded = proposal.filter_update.grounded
        statistics = grounded.belief_statistics[0].float().cpu().numpy()
        means = belief_mean_px(grounded.log_posterior)[0].cpu().numpy()
        geometry_px, geometry_valid = self.identity.pixels(
            torch.from_numpy(np.ascontiguousarray(source_state))[None].to(
                self.device, dtype=torch.float32
            )
        )
        geometry = geometry_px[0].float().cpu().numpy()
        geometry_valid_np = geometry_valid[0].cpu().numpy()
        belief_span = np.linalg.norm(means[:, 0] - means[:, 1], axis=-1).astype(np.float32)
        geometry_span = np.linalg.norm(
            geometry[:, 0] - geometry[:, 1], axis=-1
        ).astype(np.float32)
        collapse = (
            geometry_valid_np.all(axis=-1)
            & (geometry_span > MIN_ANALYTIC_SPAN_PX)
            & (belief_span < MIN_BELIEF_TO_ANALYTIC_SPAN * geometry_span)
        )

        analytic = proposal.analytic_shift_px[0].float().cpu().numpy()
        analytic_valid = proposal.analytic_valid[0].cpu().numpy()
        final = proposal.prediction.transport_shift_px[0].float().cpu().numpy()
        learned = np.where(analytic_valid[:, None, None], final - analytic, final)
        cosine = _vector_cosine(analytic, final)
        measurement_log = (
            proposal.measurement.anchor_hypothesis_log_likelihood.float()
        )
        measurement = measurement_log[0].exp().cpu().numpy()
        measurement_means = belief_mean_px(measurement_log)[0].cpu().numpy()
        measurement_span = np.linalg.norm(
            measurement_means[:, 0] - measurement_means[:, 1], axis=-1
        ).astype(np.float32)
        normalized = np.allclose(
            np.exp(
                proposal.filter_update.grounded.log_posterior[0]
                .float()
                .cpu()
                .numpy()
            ).sum(axis=-1),
            1.0,
            atol=2.0e-4,
            rtol=0.0,
        )
        return {
            "source_index": int(proposal.source_index[0].item()),
            "geometry_prior_used": bool(proposal.geometry_prior_applied),
            "geometry_anchor_px": geometry,
            "geometry_anchor_valid": geometry_valid_np,
            "geometry_anchor_span_px": geometry_span,
            "belief_mean_px": means,
            "belief_span_px": belief_span,
            "belief_collapse": collapse,
            "belief_visible_mass": statistics[..., 0],
            "belief_absent_mass": statistics[..., 1],
            "belief_entropy": statistics[..., 2],
            "measurement_absent_mass": measurement[..., GRID_CELLS],
            # Raw exchangeable measurement diagnostics precede permutation
            # assignment and H0 geometry-prior fusion.  They make it possible
            # to distinguish visual collapse from a bad/stale q prior without
            # feeding any diagnostic coordinates back into control.
            "measurement_hypothesis_mean_px": measurement_means,
            "measurement_hypothesis_span_px": measurement_span,
            "identity_assignment": proposal.assigned_measurement.assignment[0]
            .float()
            .cpu()
            .numpy(),
            "analytic_transport_px": analytic,
            "analytic_valid": analytic_valid,
            "learned_residual_px": learned,
            "final_transport_px": final,
            "transport_sigma_px": proposal.prediction.transport_sigma_px[0]
            .float()
            .cpu()
            .numpy(),
            "analytic_transport_norm_px": np.linalg.norm(analytic, axis=-1),
            "final_transport_norm_px": np.linalg.norm(final, axis=-1),
            "transport_cosine": cosine,
            "belief_finite": bool(
                np.all(
                    np.isfinite(
                        grounded.log_posterior[0].float().cpu().numpy()
                    )
                )
            ),
            "belief_normalized": bool(normalized),
        }

    def projected_geometry(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(state, dtype=np.float32)
        if q.shape == (STATE_DIM,):
            q = q[None]
        if q.ndim != 2 or q.shape[1] != STATE_DIM or not np.all(np.isfinite(q)):
            raise V177LiveError("projection state must be finite [6] or [B,6]")
        px, valid = self.identity.pixels(
            torch.from_numpy(np.ascontiguousarray(q)).to(
                self.device, dtype=torch.float32
            )
        )
        return px.float().cpu().numpy(), valid.cpu().numpy()

    @torch.inference_mode()
    def propose(
        self,
        frames: Mapping[str, np.ndarray],
        current_q_position: np.ndarray,
        *,
        replace_pending: bool = False,
    ) -> dict[str, Any]:
        state = np.asarray(current_q_position, dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
            raise V177LiveError("current robot state must be finite [6]")
        self.last_proposal = None
        self._render_log_posterior = None
        if self.runtime.has_pending_chunk:
            if not replace_pending:
                raise V177LiveError("an unexecuted V17.7 proposal is pending")
            preserve_actual_endpoint = self.runtime.completed_chunks > 0
            self.runtime.cancel_pending_proposal()
            # A replaced H0 preview is explicitly reobserved from its caller's
            # fresh images.  Once a real transition has completed, however,
            # resetting here would erase H1+ actual measurement/readback
            # memory; preserve it and let propose_from_latest verify current q.
            if not preserve_actual_endpoint:
                self.runtime.reset()
        effective_lower, effective_upper = _inward_only_action_bounds(
            state, self.action_lower_position, self.action_upper_position
        )
        self._install_action_bounds(effective_lower, effective_upper)
        self.effective_action_lower_position = effective_lower
        self.effective_action_upper_position = effective_upper

        started = time.perf_counter()
        q = torch.from_numpy(np.ascontiguousarray(state)).to(
            self.device, dtype=torch.float32
        )
        used_latest = self.runtime.has_latest_actual
        try:
            if used_latest:
                proposal = self.runtime.propose_from_latest(q)
            else:
                rgb, tokens, valid = self._endpoint_tensors(frames)
                proposal = self.runtime.observe_and_propose(
                    tokens,
                    valid,
                    q,
                    rgb=rgb,
                    geometry_log_prior=self._geometry_prior(state, valid),
                )
            commands = proposal.actor_output.commands[0].float().cpu().numpy().astype(np.float32)
            predicted = proposal.prediction.predicted_q_path[0].float().cpu().numpy().astype(np.float32)
            if commands.shape != (COMMAND_STEPS, STATE_DIM) or not np.all(np.isfinite(commands)):
                raise V177LiveError("V17.7 produced an invalid command chunk")
            if predicted.shape != (PATH_STEPS, STATE_DIM) or not np.all(np.isfinite(predicted)):
                raise V177LiveError("V17.7 produced an invalid predicted q path")
            if np.any(commands < effective_lower[None] - 1.0e-4) or np.any(
                commands > effective_upper[None] + 1.0e-4
            ):
                raise V177LiveError("V17.7 output violates deployment joint bounds")
            reversals = _nonmonotonic_recovery_joints(
                commands,
                state,
                self.action_lower_position,
                self.action_upper_position,
            )
            if reversals:
                raise V177LiveError(
                    "V17.7 reverses outward before guarded recovery on arm joint(s) "
                    + ",".join(map(str, reversals))
                )
            actor = self.runtime.model.actor
            projected = _project_action_chunk(
                torch.from_numpy(state)[None],
                torch.from_numpy(commands)[None],
                actor.step_cap.detach().cpu(),
                actor.action_lower.detach().cpu(),
                actor.action_upper.detach().cpu(),
            )
            fixed_point_error = float(
                torch.max(torch.abs(projected - torch.from_numpy(commands)[None])).item()
            )
            if fixed_point_error > 1.0e-4:
                raise V177LiveError("V17.7 command is not a projector fixed point")
            increments = np.diff(np.concatenate((state[None], commands), axis=0), axis=0)
            record: dict[str, Any] = {
                "action_source": self.action_source,
                "position_unit": POSITION_UNIT,
                "commands_position": commands,
                "predicted_q_path_position": predicted,
                "pending_generation": int(proposal.generation),
                "maximum_observed_step_per_joint_position": np.max(
                    np.abs(increments), axis=0
                ),
                "projector_fixed_point_error_position": fixed_point_error,
                "effective_action_lower": effective_lower,
                "effective_action_upper": effective_upper,
                "inference_s": time.perf_counter() - started,
                "memory_chunks": self.runtime.completed_chunks,
                "used_latest_actual_endpoint": used_latest,
            }
            record.update(self._proposal_diagnostics(proposal, state))
            self._render_log_posterior = (
                proposal.filter_update.grounded.log_posterior[0]
                .detach()
                .float()
                .cpu()
                .numpy()
                .copy()
            )
            self.last_proposal = record
            return record
        except BaseException:
            self.runtime.cancel_pending_proposal()
            self.last_proposal = None
            self._render_log_posterior = None
            raise

    def render_log_posterior(self) -> np.ndarray | None:
        """Return a defensive copy for diagnostics-only image rendering."""

        if self._render_log_posterior is None:
            return None
        return self._render_log_posterior.copy()

    @torch.inference_mode()
    def complete(
        self,
        frames: Mapping[str, np.ndarray] | None = None,
        measured_q_path_position: np.ndarray | None = None,
        *,
        delta_t_s: float,
        actual_endpoint_state_position: np.ndarray | None = None,
        generation: int | None = None,
        commands_position: np.ndarray | None = None,
        next_frames: Mapping[str, np.ndarray] | None = None,
        next_q_position: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if frames is not None and next_frames is not None:
            raise V177LiveError("supply frames or next_frames, not both")
        observed_frames = frames if frames is not None else next_frames
        if observed_frames is None or measured_q_path_position is None:
            raise V177LiveError("completion requires next frames and measured path")
        if actual_endpoint_state_position is not None and next_q_position is not None:
            raise V177LiveError("supply one endpoint state alias")
        endpoint = (
            actual_endpoint_state_position
            if actual_endpoint_state_position is not None
            else next_q_position
        )
        if generation is not None and int(generation) != self.runtime.pending_generation:
            raise V177LiveError("completion generation differs from pending proposal")
        if commands_position is not None:
            pending = self.runtime.pending_commands
            supplied = np.asarray(commands_position, dtype=np.float32)
            if pending is None or supplied.shape != (COMMAND_STEPS, STATE_DIM):
                raise V177LiveError("completion command chunk is invalid")
            if not np.allclose(
                supplied,
                pending[0].float().cpu().numpy(),
                atol=1.0e-5,
                rtol=0.0,
            ):
                raise V177LiveError("completion commands differ from pending proposal")
        path = np.asarray(measured_q_path_position, dtype=np.float32)
        if path.shape != (PATH_STEPS, STATE_DIM) or not np.all(np.isfinite(path)):
            raise V177LiveError("measured path must be finite [6,6]")
        if not np.isfinite(delta_t_s) or float(delta_t_s) <= 0.0:
            raise V177LiveError("transition duration must be positive and finite")
        rgb, tokens, valid = self._endpoint_tensors(observed_frames)
        kwargs: dict[str, Any] = {"actual_rgb": rgb}
        if endpoint is not None:
            endpoint_array = np.asarray(endpoint, dtype=np.float32)
            if endpoint_array.shape != (STATE_DIM,) or not np.all(np.isfinite(endpoint_array)):
                raise V177LiveError("actual endpoint state must be finite [6]")
            kwargs["actual_endpoint_q"] = torch.from_numpy(
                np.ascontiguousarray(endpoint_array)
            ).to(self.device, dtype=torch.float32)
        update = self.runtime.complete_and_reobserve(
            tokens,
            valid,
            torch.from_numpy(np.ascontiguousarray(path)).to(
                self.device, dtype=torch.float32
            ),
            **kwargs,
        )
        innovation = update.innovation
        record = {
            "belief_innovation_rms": float(
                innovation.innovation.float().square().mean().sqrt().item()
            ),
            "state_innovation_rms_position": float(
                innovation.state_residual.float().square().mean().sqrt().item()
            ),
            "belief_surprise_mean": float(
                innovation.belief_surprise.float().mean().item()
            ),
            "belief_surprise_max": float(
                innovation.belief_surprise.float().amax().item()
            ),
            "belief_absent_mass": update.grounded.belief_statistics[0, ..., 1]
            .float()
            .cpu()
            .numpy(),
            "belief_entropy": update.grounded.belief_statistics[0, ..., 2]
            .float()
            .cpu()
            .numpy(),
            "transition_duration_s": float(delta_t_s),
            "memory_chunks": self.runtime.completed_chunks,
        }
        self.last_completion = record
        return record

    def status(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "architecture": ARCHITECTURE_CONTRACT,
            "actor_mode": "full",
            "position_unit": POSITION_UNIT,
            "source_index": self.source_index,
            "view_order": list(VIEW_ORDER),
            "checkpoint": self.checkpoint,
            "seed": self.seed,
            "training_step": self.training_step,
            "predictive_belief_grounding": True,
            "camera_aware_transport": True,
            "analytic_transport": bool(self.runtime.model.config.analytic_transport),
            "physical_admission_before_lewm": bool(
                self.runtime.has_bound_command_admission_provider
            ),
            "geometry_prior_fusion": bool(
                self.runtime.model.config.geometry_prior_fusion
            ),
            "control_uses_point_coordinates": False,
            "deployment_action_lower": self.action_lower_position,
            "deployment_action_upper": self.action_upper_position,
            "effective_action_lower": self.effective_action_lower_position,
            "effective_action_upper": self.effective_action_upper_position,
            "pending_chunk": self.runtime.has_pending_chunk,
            "pending_generation": self.runtime.pending_generation,
            "memory_chunks": self.runtime.completed_chunks,
            "last_proposal": self.last_proposal,
            "last_completion": self.last_completion,
        }

    def reset(self) -> None:
        self.runtime.reset()
        self.last_proposal = None
        self.last_completion = None
        self._render_log_posterior = None


def offline_gate_summary(
    baseline: Mapping[str, Any],
    *,
    current_geometry_px: np.ndarray,
    terminal_geometry_px: np.ndarray,
    geometry_valid: np.ndarray,
    fish_centers_px: np.ndarray,
    repeated_commands: np.ndarray,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    counterfactual_dx_px: float,
) -> dict[str, Any]:
    """Evaluate the fixed pre-hardware gates from already computed records."""

    analytic = np.asarray(baseline["analytic_transport_px"], dtype=np.float32)
    final = np.asarray(baseline["final_transport_px"], dtype=np.float32)
    analytic_valid = np.asarray(baseline["analytic_valid"], dtype=bool)
    analytic_norm = np.linalg.norm(analytic, axis=-1)
    final_norm = np.linalg.norm(final, axis=-1)
    cosine = np.asarray(baseline["transport_cosine"], dtype=np.float32)
    transport_required = analytic_valid[:, None] & (
        analytic_norm > MIN_ANALYTIC_TRANSPORT_PX
    )
    transport_ok = (~transport_required) | (
        (final_norm > MIN_FINAL_TRANSPORT_PX) & (cosine > MIN_TRANSPORT_COSINE)
    )

    current_geometry = np.asarray(current_geometry_px, dtype=np.float32)
    terminal_geometry = np.asarray(terminal_geometry_px, dtype=np.float32)
    valid = np.asarray(geometry_valid, dtype=bool)
    fish = np.asarray(fish_centers_px, dtype=np.float32)
    column_index = VIEW_ORDER.index("column")
    if fish.shape == (2,):
        column_fish = fish
    elif fish.shape == (VIEWS, 2):
        column_fish = fish[column_index]
    else:
        raise V177LiveError("fish center must be column [2] or per-view [2,2]")
    if not np.all(np.isfinite(column_fish)):
        raise V177LiveError("column fish center must be finite")
    if current_geometry.shape != (VIEWS, ANCHORS, 2) or terminal_geometry.shape != (
        VIEWS,
        ANCHORS,
        2,
    ):
        raise V177LiveError("projected geometry must be [2,2,2]")
    if valid.shape != (VIEWS, ANCHORS):
        raise V177LiveError("projected geometry validity must be [2,2]")
    current_midpoint = current_geometry.mean(axis=1)
    terminal_midpoint = terminal_geometry.mean(axis=1)
    current_distance = float(
        np.linalg.norm(current_midpoint[column_index] - column_fish)
    )
    terminal_distance = float(
        np.linalg.norm(terminal_midpoint[column_index] - column_fish)
    )
    terminal_in_frame = bool(
        np.all(np.isfinite(terminal_geometry))
        and np.all(terminal_geometry >= 0.0)
        and np.all(terminal_geometry < IMAGE_SIZE)
    )
    direction_ok = bool(
        valid[column_index].all()
        and terminal_in_frame
        and terminal_distance < current_distance
    )

    base_commands = np.asarray(baseline["commands_position"], dtype=np.float32)
    repeat = np.asarray(repeated_commands, dtype=np.float32)
    left_commands = np.asarray(left["commands_position"], dtype=np.float32)
    right_commands = np.asarray(right["commands_position"], dtype=np.float32)
    repeat_noise = float(np.max(np.abs(base_commands - repeat)))
    response = float(np.max(np.abs(right_commands - left_commands)))
    action_threshold = max(1.0e-3, 3.0 * repeat_noise)

    baseline_path = np.asarray(
        baseline["command_geometry_path_px"], dtype=np.float32
    )
    repeated_path = np.asarray(
        baseline["repeated_command_geometry_path_px"], dtype=np.float32
    )
    left_path = np.asarray(left["command_geometry_path_px"], dtype=np.float32)
    right_path = np.asarray(right["command_geometry_path_px"], dtype=np.float32)
    expected_path_shape = (COMMAND_STEPS, VIEWS, ANCHORS, 2)
    if any(
        value.shape != expected_path_shape
        for value in (baseline_path, repeated_path, left_path, right_path)
    ):
        raise V177LiveError("command FK paths must be [5,2,2,2]")
    repeat_path_noise = float(np.max(np.abs(repeated_path - baseline_path)))
    path_response = float(np.max(np.abs(right_path - left_path)))
    path_threshold = max(0.25, 3.0 * repeat_path_noise)

    left_terminal = np.asarray(left["terminal_geometry_px"], dtype=np.float32).mean(axis=1)
    right_terminal = np.asarray(right["terminal_geometry_px"], dtype=np.float32).mean(axis=1)
    terminal_response_x = right_terminal[:, 0] - left_terminal[:, 0]
    repeated_terminal = np.asarray(
        baseline.get("repeated_terminal_geometry_px", terminal_geometry),
        dtype=np.float32,
    ).mean(axis=1)
    repeat_terminal_noise = float(
        np.max(np.abs(repeated_terminal - terminal_midpoint))
    )
    terminal_threshold = max(0.25, 3.0 * repeat_terminal_noise)
    response_direction_ok = bool(
        terminal_response_x[column_index] * np.sign(float(counterfactual_dx_px))
        > terminal_threshold
    )
    projected_shift = terminal_geometry - current_geometry
    analytic_projection_error = np.max(
        np.abs(analytic - projected_shift), axis=-1
    )
    analytic_projection_ok = bool(
        np.all((~analytic_valid[:, None]) | (analytic_projection_error <= 1.0e-4))
    )
    visible = np.asarray(baseline["belief_visible_mass"], dtype=np.float32)
    absent = np.asarray(baseline["belief_absent_mass"], dtype=np.float32)
    gates = {
        "belief_finite": bool(baseline["belief_finite"]),
        "belief_normalized": bool(baseline["belief_normalized"]),
        "visibility_probability_legal": bool(
            np.all(np.isfinite(visible))
            and np.all(np.isfinite(absent))
            and np.all(visible >= 0.0)
            and np.all(visible <= 1.0)
            and np.all(absent >= 0.0)
            and np.all(absent <= 1.0)
            and np.allclose(visible + absent, 1.0, atol=2.0e-4, rtol=0.0)
        ),
        "anchor_not_collapsed": bool(
            not np.asarray(baseline["belief_collapse"], dtype=bool).any()
        ),
        "analytic_matches_fk_projection": analytic_projection_ok,
        "transport_matches_analytic": bool(transport_ok.all()),
        "terminal_midpoints_in_frame": terminal_in_frame,
        "terminal_moves_toward_fish": direction_ok,
        "counterfactual_changes_action": bool(response > action_threshold),
        "counterfactual_changes_fk_path": bool(path_response > path_threshold),
        "counterfactual_direction_matches": response_direction_ok,
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "metrics": {
            "transport_required": transport_required,
            "transport_ok": transport_ok,
            "analytic_projection_error_px": analytic_projection_error,
            "column_fish_center_px": column_fish,
            "current_column_fish_distance_px": current_distance,
            "terminal_column_fish_distance_px": terminal_distance,
            "fk_midpoint_progress_px": current_distance - terminal_distance,
            "repeat_action_noise_max": repeat_noise,
            "counterfactual_action_response_max": response,
            "counterfactual_action_threshold": action_threshold,
            "repeat_fk_path_noise_max_px": repeat_path_noise,
            "counterfactual_fk_path_response_max_px": path_response,
            "counterfactual_fk_path_threshold_px": path_threshold,
            "counterfactual_terminal_response_x_px": terminal_response_x,
            "repeat_terminal_noise_max_px": repeat_terminal_noise,
            "counterfactual_terminal_threshold_px": terminal_threshold,
        },
    }


__all__ = [
    "EXPECTED_MODEL_STATE_KEYS",
    "EXPECTED_ACTOR_CONTEXT",
    "FISH_SOURCE_INDEX",
    "POSITION_UNIT",
    "V177LiveError",
    "V177LivePolicy",
    "belief_mean_px",
    "load_strict_v177_full_checkpoint",
    "offline_gate_summary",
    "translate_image_region",
]
