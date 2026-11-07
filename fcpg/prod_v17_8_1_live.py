"""No-hardware live adapter for Prod-V17.8.1.

This module deliberately owns no camera, serial port, HTTP server or HOME
implementation.  It adapts one completed V17.8.1 checkpoint to the established
V17.7 hardware service while preserving V17.8.1's task, history and settled
endpoint clocks.

The deployment-facing command path is::

    Actor -> absolute joint-limit projection -> physical admission
          -> analytic transport -> LeWM

Physical admission is reject-only.  It cannot rewrite an Actor command.  A
completed transaction records the timed q0..q5 readback path separately from
the later image-synchronous settled state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields
from pathlib import Path
import json
import math
import time
from typing import Any

import numpy as np
import torch

from fcpg.prod_v17_7_live import (
    FISH_SOURCE_INDEX,
    MIN_ANALYTIC_SPAN_PX,
    MIN_BELIEF_TO_ANALYTIC_SPAN,
    _inward_only_action_bounds,
    _nonmonotonic_recovery_joints,
    _vector_cosine,
    belief_mean_px,
)
from fcpg.prod_v17_8 import (
    ANCHORS,
    BELIEF_CLASSES,
    COMMAND_STEPS,
    GRID_CELLS,
    IMAGE_SIZE,
    PATH_STEPS,
    STATE_DIM,
    TASK_EMBEDDING_DIM,
    VIEWS,
    VIEW_ORDER,
)
from fcpg.prod_v17_8_1 import (
    ARCHITECTURE_CONTRACT,
    VERSION,
    V1781Config,
    V1781Core,
)
from fcpg.prod_v17_8_1_contact import CONTACT_EVENTS
from fcpg.prod_v17_8_1_runtime import (
    V1781RuntimeController,
    V1781RuntimeError,
)
from fcpg.prod_v17_8_1_training import (
    DEFAULT_ARCHIVE_BATCH,
    DEFAULT_FISH_BATCH,
    DEFAULT_INCREMENTAL_UPDATES,
    REQUIRED_BASE_ARCHIVE_EXPOSURES,
    REQUIRED_FISH_REPLAY_UPDATES,
    REQUIRED_REPLAY_PAIR_EXPOSURES,
    SEED,
)
from fcpg.prod_v17_8_runtime import _batch_one
from fcpg.prod_v17_8_training import (
    ARCHITECTURE_CONTRACT as V178_ARCHITECTURE_CONTRACT,
    FISH_TASK_TEXT,
    TASK_ENCODER,
    VERSION as V178_VERSION,
    V178GeometryIdentityProvider,
)
from fcpg.prod_v18_1_training import extract_dinov2_tokens, load_frozen_dinov2


# The V17.7 label contains the word ``normalized`` even though this stack and
# its persisted data use the public LeRobot calibrated-position values.  Keep
# one truthful unit string at the new boundary without changing old artifacts.
POSITION_UNIT = "lerobot_calibrated_position"
EXPECTED_MODEL_STATE_KEYS = 417
EXPECTED_PARENT_MODEL_STATE_KEYS = 355
EXPECTED_INCREMENTAL_ARCHIVE_EXPOSURES = (
    DEFAULT_INCREMENTAL_UPDATES * DEFAULT_ARCHIVE_BATCH
)
EXPECTED_INCREMENTAL_FISH_EXPOSURES = (
    DEFAULT_INCREMENTAL_UPDATES * DEFAULT_FISH_BATCH
)
TASK_EMBEDDING_POOLING = "attention_mask_mean_then_l2"


class V1781LiveError(RuntimeError):
    """Raised before malformed live data reaches hardware-facing code."""


def _normal_task_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def _require_completed_metrics(metrics: object) -> None:
    if not isinstance(metrics, Mapping):
        raise V1781LiveError("V17.8.1 checkpoint metrics are absent")
    event_exposures = metrics.get("fish_event_exposures")
    expected_events = {event.value for event in CONTACT_EVENTS}
    parent = metrics.get("parent")
    event_counts_valid = bool(
        isinstance(event_exposures, Mapping)
        and set(event_exposures) == expected_events
    )
    if event_counts_valid:
        try:
            event_counts_valid = all(
                not isinstance(event_exposures[name], bool)
                and int(event_exposures[name]) > 0
                for name in expected_events
            ) and sum(int(event_exposures[name]) for name in expected_events) == (
                EXPECTED_INCREMENTAL_FISH_EXPOSURES
            )
        except (TypeError, ValueError):
            event_counts_valid = False
    try:
        loss_is_finite = math.isfinite(float(metrics.get("loss")))
    except (TypeError, ValueError):
        loss_is_finite = False
    if (
        metrics.get("steps") != DEFAULT_INCREMENTAL_UPDATES
        or not loss_is_finite
        or metrics.get("archive_replay_exposures")
        != EXPECTED_INCREMENTAL_ARCHIVE_EXPOSURES
        or metrics.get("fish_event_balanced_exposures")
        != EXPECTED_INCREMENTAL_FISH_EXPOSURES
        or metrics.get("archive_fish_loss_weight") != [0.5, 0.5]
        or metrics.get("contact_event_and_fk_are_training_only") is not True
        or metrics.get("settled_clock_is_runtime_only_until_real_settle_training_exists")
        is not True
        or not event_counts_valid
        or not isinstance(parent, Mapping)
        or parent.get("parent_step") != 24_000
        or parent.get("parent_run_kind") != "full"
        or parent.get("parent_archive_pair_exposures")
        != REQUIRED_BASE_ARCHIVE_EXPOSURES
        or parent.get("parent_fish_replay_updates")
        != REQUIRED_FISH_REPLAY_UPDATES
        or int(parent.get("parent_archive_replay_exposures", 0))
        < REQUIRED_REPLAY_PAIR_EXPOSURES
        or int(parent.get("parent_fish_pair_exposures", 0))
        < REQUIRED_REPLAY_PAIR_EXPOSURES
        or parent.get("loaded_parent_keys") != EXPECTED_PARENT_MODEL_STATE_KEYS
    ):
        raise V1781LiveError(
            "V17.8.1 checkpoint is not a completed formal incremental Actor"
        )


def load_fish_task_embedding_cache(path: str | Path) -> torch.Tensor:
    """Load the exact fish row from a completed V17.8 task cache.

    A bare vector is intentionally not accepted.  The cache must retain the
    training text vocabulary and MiniLM/pooling metadata, making the live task
    binding explicit without loading MiniLM in the hardware process.
    """

    root = Path(path).expanduser().resolve()
    required = {
        "config": root / "config.json",
        "embeddings": root / "embeddings.npy",
        "index": root / "task_index.npy",
        "names": root / "task_names.json",
    }
    if not root.is_dir() or any(not value.is_file() for value in required.values()):
        raise V1781LiveError(
            "task embedding path must be a complete V17.8 task-cache directory"
        )
    try:
        config = json.loads(required["config"].read_text(encoding="utf-8"))
        names = json.loads(required["names"].read_text(encoding="utf-8"))
        embeddings = np.load(required["embeddings"], allow_pickle=False)
        row_index = np.load(required["index"], allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise V1781LiveError("V17.8 task embedding cache is unreadable") from error
    if not isinstance(config, dict) or not isinstance(names, list):
        raise V1781LiveError("V17.8 task embedding metadata is malformed")
    expected = {
        "version": V178_VERSION,
        "contract": V178_ARCHITECTURE_CONTRACT,
        "encoder": TASK_ENCODER,
        "pooling": TASK_EMBEDDING_POOLING,
        "embedding_dim": TASK_EMBEDDING_DIM,
        "complete": True,
    }
    observed = {key: config.get(key) for key in expected}
    if observed != expected:
        raise V1781LiveError(
            f"V17.8 task embedding metadata differs: observed={observed} "
            f"expected={expected}"
        )
    if config.get("task_texts") != names:
        raise V1781LiveError("task embedding names differ from cache config")
    if names.count(FISH_TASK_TEXT) != 1:
        raise V1781LiveError("task cache must contain the exact fish task once")
    if (
        embeddings.dtype != np.float32
        or embeddings.shape != (len(names), TASK_EMBEDDING_DIM)
        or row_index.dtype != np.int64
        or row_index.ndim != 1
        or int(config.get("rows", -1)) != len(names)
        or int(config.get("row_count", -1)) != len(row_index)
        or bool((row_index < 0).any())
        or bool((row_index >= len(names)).any())
        or not bool(np.isfinite(embeddings).all())
    ):
        raise V1781LiveError("V17.8 task embedding arrays violate their contract")
    norms = np.linalg.norm(embeddings, axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 2.0e-4:
        raise V1781LiveError("V17.8 task embeddings are not unit-L2 normalized")
    fish = np.asarray(embeddings[names.index(FISH_TASK_TEXT)], dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(fish.copy()))


def load_strict_v1781_full_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device | str,
) -> tuple[V1781Core, dict[str, Any]]:
    """Load exactly one completed 417-key V17.8.1 full-stack checkpoint."""

    target = torch.device(device)
    source = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise V1781LiveError(f"V17.8.1 checkpoint is absent: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as error:
        raise V1781LiveError(
            f"V17.8.1 checkpoint cannot be read: {source}"
        ) from error
    if not isinstance(payload, dict):
        raise V1781LiveError("V17.8.1 checkpoint must be a mapping")
    expected_metadata = {
        "version": VERSION,
        "architecture": ARCHITECTURE_CONTRACT,
        "component": "actor",
        "seed": SEED,
        "step": DEFAULT_INCREMENTAL_UPDATES,
    }
    observed = {name: payload.get(name) for name in expected_metadata}
    if observed != expected_metadata:
        raise V1781LiveError(
            f"V17.8.1 checkpoint metadata differs: observed={observed} "
            f"expected={expected_metadata}"
        )
    # ``latest.pt`` at step 1000 also has complete-looking metadata.  The
    # optimizer is deliberately absent from the immutable deployment final.
    if "optimizer" in payload or "cursor" in payload:
        raise V1781LiveError("V17.8.1 live loading requires immutable final.pt")

    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise V1781LiveError("V17.8.1 checkpoint config must be a mapping")
    config_fields = {field.name for field in fields(V1781Config)}
    if set(raw_config) != config_fields:
        missing = sorted(config_fields - set(raw_config))
        extra = sorted(set(raw_config) - config_fields)
        raise V1781LiveError(
            "V17.8.1 checkpoint config keys differ: "
            f"missing={missing} extra={extra}"
        )
    try:
        config = V1781Config(**raw_config)
        config.validate()
    except Exception as error:
        raise V1781LiveError("V17.8.1 checkpoint config is invalid") from error
    if config != V1781Config():
        raise V1781LiveError(
            "V17.8.1 live loading requires the frozen production model config"
        )
    if (
        config.actor_mode != "full"
        or not config.analytic_transport
        or not config.geometry_prior_fusion
        or int(config.raw_history_steps) != 3
        or int(config.source_vocabulary) <= FISH_SOURCE_INDEX
    ):
        raise V1781LiveError("V17.8.1 deployment model contract is incomplete")

    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise V1781LiveError("V17.8.1 checkpoint model must be a state mapping")
    model = V1781Core(config)
    expected = model.state_dict()
    if len(expected) != EXPECTED_MODEL_STATE_KEYS:
        raise V1781LiveError(
            f"local V17.8.1 architecture has {len(expected)} keys, expected "
            f"{EXPECTED_MODEL_STATE_KEYS}"
        )
    if len(state) != EXPECTED_MODEL_STATE_KEYS or set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise V1781LiveError(
            "V17.8.1 checkpoint is not the complete 417-key model: "
            f"count={len(state)} missing={missing[:8]} extra={extra[:8]}"
        )
    for name, reference in expected.items():
        value = state[name]
        if (
            not torch.is_tensor(value)
            or value.shape != reference.shape
            or value.dtype != reference.dtype
        ):
            raise V1781LiveError(
                f"V17.8.1 checkpoint tensor contract differs for {name!r}"
            )
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise V1781LiveError(
                f"V17.8.1 checkpoint state {name!r} contains non-finite values"
            )
    _require_completed_metrics(payload.get("metrics"))
    try:
        model.load_state_dict(state, strict=True)
    except Exception as error:  # pragma: no cover - tensor checks precede this
        raise V1781LiveError(
            "V17.8.1 checkpoint failed strict model loading"
        ) from error
    model.requires_grad_(False).to(target).eval()
    return model, payload


class V1781LiveRuntimeController(V1781RuntimeController):
    """V17.8.1 runtime with the V17.7 hardware-service binding surface."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._physical_admission_provider: (
            Callable[[torch.Tensor, torch.Tensor], None] | None
        ) = None
        super().__init__(*args, **kwargs)

    @property
    def has_bound_source_state_encoder(self) -> bool:
        return self._source_state_encoder is not None

    @property
    def has_bound_command_admission_provider(self) -> bool:
        return self._physical_admission_provider is not None

    def bind_command_admission_provider(
        self, provider: Callable[[torch.Tensor, torch.Tensor], None]
    ) -> None:
        if not callable(provider):
            raise V1781RuntimeError("command admission provider must be callable")
        if self._physical_admission_provider is not None:
            raise V1781RuntimeError("command admission provider is already bound")
        self._require_pre_observation_binding("command admission provider")
        self._physical_admission_provider = provider

    def source_states_match(
        self, reference: torch.Tensor, measured: torch.Tensor
    ) -> bool:
        """Use the same numeric metric later enforced by settled history.

        Encoder cells remain available through ``source_state_cell_delta`` for
        diagnostics, but can never approve a q0 that the actual-history clock
        would reject after commands have already been sent.
        """

        expected = _batch_one(
            reference,
            (STATE_DIM,),
            device=self.device,
            name="reference source state",
            floating=True,
        )
        actual = _batch_one(
            measured,
            (STATE_DIM,),
            device=self.device,
            name="measured source state",
            floating=True,
        )
        return bool(
            torch.allclose(
                actual,
                expected,
                atol=self.source_state_atol_position,
                rtol=0.0,
            )
        )

    def _analytic_inputs(
        self, source_q: torch.Tensor, commands: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        provider = self._physical_admission_provider
        if provider is None:
            raise V1781RuntimeError(
                "physical command admission is not bound before LeWM"
            )
        try:
            result = provider(
                source_q.detach().clone(), commands.detach().clone()
            )
        except Exception as error:
            raise V1781RuntimeError(
                "physical command admission rejected Actor output"
            ) from error
        if result is not None:
            raise V1781RuntimeError(
                "command admission provider must return None or raise"
            )
        # This call computes analytic transport and then invokes LeWM in the
        # inherited proposal builder, so admission is strictly before both.
        return super()._analytic_inputs(source_q, commands)

    def cancel_pending_proposal(self, *, generation: int | None = None) -> None:
        # Compatibility with V177HardwareService's display cancellation.  It
        # can only cancel the one currently pending generation.
        selected = self.pending_generation if generation is None else generation
        if selected is None:
            raise V1781RuntimeError("no proposed chunk is waiting for cancellation")
        super().cancel_pending_proposal(generation=int(selected))


def _object_centers_px(attention: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(attention).float()
    if tuple(value.shape) != (1, VIEWS, 2, 16, 16):
        raise V1781LiveError("object attention must be [1,2,2,16,16]")
    axis = torch.arange(16, device=value.device, dtype=value.dtype) * 14.0 + 6.5
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)
    return torch.einsum("bvrhw,hwd->bvrd", value, grid)


class V1781LivePolicy:
    """Own frozen perception and one actual settled-endpoint policy clock."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        dino_repository: str | Path,
        dino_weights: str | Path,
        real_calibration: str | Path,
        device: torch.device | str,
        task_text: str = FISH_TASK_TEXT,
        task_embedding: torch.Tensor | None = None,
        action_lower_position: np.ndarray | None = None,
        action_upper_position: np.ndarray | None = None,
        source_index: int = FISH_SOURCE_INDEX,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise V1781LiveError("CUDA was requested but is unavailable")
        normalized_task = _normal_task_text(task_text)
        if normalized_task != FISH_TASK_TEXT:
            raise V1781LiveError(
                "fish live policy requires the exact frozen training task text"
            )
        if int(source_index) != FISH_SOURCE_INDEX:
            raise V1781LiveError("live fish policy requires source_index=129")
        model, payload = load_strict_v1781_full_checkpoint(
            checkpoint, device=self.device
        )
        if (action_lower_position is None) != (action_upper_position is None):
            raise V1781LiveError("deployment action bounds must be supplied together")
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
            raise V1781LiveError("deployment action bounds are invalid")

        self.identity = V178GeometryIdentityProvider(
            real_calibration, device=self.device
        )
        self.runtime = V1781LiveRuntimeController(
            model,
            normalized_task,
            device=self.device,
            source_index=FISH_SOURCE_INDEX,
            task_embedding=task_embedding,
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
        self.action_source = "v17_8_1_task_settled_predictive_belief_controller"
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.source_index = FISH_SOURCE_INDEX
        self.seed = int(payload["seed"])
        self.training_step = int(payload["step"])
        self.task_text = normalized_task
        self.task_encoder = TASK_ENCODER
        self.task_embedding_source = (
            "runtime_minilm" if task_embedding is None else "training_task_cache"
        )
        self.last_proposal: dict[str, Any] | None = None
        self.last_completion: dict[str, Any] | None = None
        self._render_log_posterior: np.ndarray | None = None
        self._render_object_attention: np.ndarray | None = None

    def bind_source_state_encoder(
        self, encoder: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        if not callable(encoder):
            raise V1781LiveError("source state encoder must be callable")

        def tensor_encoder(state: torch.Tensor) -> torch.Tensor:
            values = state[0].detach().float().cpu().numpy().copy()
            return torch.from_numpy(
                np.ascontiguousarray(np.asarray(encoder(values)))
            )

        self.runtime.bind_source_state_encoder(tensor_encoder)

    def bind_command_admission_provider(
        self, provider: Callable[[np.ndarray, np.ndarray], None]
    ) -> None:
        if not callable(provider):
            raise V1781LiveError("command admission provider must be callable")

        def tensor_provider(
            source_q: torch.Tensor, commands: torch.Tensor
        ) -> None:
            source = source_q[0].detach().float().cpu().numpy().copy()
            block = commands[0].detach().float().cpu().numpy().copy()
            result = provider(source, block)
            if result is not None:
                raise V1781LiveError(
                    "command admission provider must return None or raise"
                )

        self.runtime.bind_command_admission_provider(tensor_provider)

    def _install_action_bounds(self, lower: np.ndarray, upper: np.ndarray) -> None:
        actor = self.runtime.model.actor
        if hasattr(actor, "step_cap"):
            raise V1781LiveError("V17.8.1 Actor unexpectedly contains a step cap")
        with torch.no_grad():
            actor.action_lower.copy_(
                torch.from_numpy(np.asarray(lower, dtype=np.float32)).to(
                    actor.action_lower
                )
            )
            actor.action_upper.copy_(
                torch.from_numpy(np.asarray(upper, dtype=np.float32)).to(
                    actor.action_upper
                )
            )

    def _endpoint_tensors(
        self, frames: Mapping[str, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if set(frames) != set(VIEW_ORDER):
            raise V1781LiveError(
                f"live frames must contain exactly {list(VIEW_ORDER)}"
            )
        ordered: list[np.ndarray] = []
        for view in VIEW_ORDER:
            frame = np.asarray(frames[view])
            if frame.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or frame.dtype != np.uint8:
                raise V1781LiveError(
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

    @torch.inference_mode()
    def _object_diagnostics(self, proposal: Any) -> dict[str, Any]:
        valid = proposal.measurement.view_valid
        scene = proposal.measurement.scene_tokens * valid[:, :, None, None].to(
            proposal.measurement.scene_tokens
        )
        objects = self.runtime.model.object_head(
            scene, self.runtime.task_embedding, valid
        )
        attention = objects.dashboard_attention_maps.float()
        centers = _object_centers_px(attention)
        self._render_object_attention = attention[0].cpu().numpy().copy()
        return {
            "task_object_attention": self._render_object_attention,
            "task_object_mean_px": centers[0].cpu().numpy(),
            "task_object_roles": ["pickup", "drop"],
        }

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
        belief_span = np.linalg.norm(means[:, 0] - means[:, 1], axis=-1)
        geometry_span = np.linalg.norm(geometry[:, 0] - geometry[:, 1], axis=-1)
        collapse = (
            geometry_valid_np.all(axis=-1)
            & (geometry_span > MIN_ANALYTIC_SPAN_PX)
            & (belief_span < MIN_BELIEF_TO_ANALYTIC_SPAN * geometry_span)
        )
        analytic = proposal.analytic_shift_px[0].float().cpu().numpy()
        analytic_valid = proposal.analytic_valid[0].cpu().numpy()
        final = proposal.prediction.transport_shift_px[0].float().cpu().numpy()
        learned = np.where(analytic_valid[:, None, None], final - analytic, final)
        measurement_log = proposal.measurement.anchor_hypothesis_log_likelihood.float()
        measurement = measurement_log[0].exp().cpu().numpy()
        measurement_means = belief_mean_px(measurement_log)[0].cpu().numpy()
        result = {
            "source_index": int(proposal.source_index[0].item()),
            "geometry_prior_used": bool(proposal.geometry_prior_applied),
            "geometry_anchor_px": geometry,
            "geometry_anchor_valid": geometry_valid_np,
            "geometry_anchor_span_px": geometry_span.astype(np.float32),
            "belief_mean_px": means,
            "belief_span_px": belief_span.astype(np.float32),
            "belief_collapse": collapse,
            "belief_visible_mass": statistics[..., 0],
            "belief_absent_mass": statistics[..., 1],
            "belief_entropy": statistics[..., 2],
            "measurement_absent_mass": measurement[..., GRID_CELLS],
            "measurement_hypothesis_mean_px": measurement_means,
            "measurement_hypothesis_span_px": np.linalg.norm(
                measurement_means[:, 0] - measurement_means[:, 1], axis=-1
            ).astype(np.float32),
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
            "transport_cosine": _vector_cosine(analytic, final),
            "belief_finite": bool(
                np.isfinite(grounded.log_posterior[0].float().cpu().numpy()).all()
            ),
            "belief_normalized": bool(
                np.allclose(
                    grounded.log_posterior[0].float().cpu().numpy().astype(np.float64),
                    torch.log_softmax(
                        grounded.log_posterior[0].float(), dim=-1
                    ).cpu().numpy().astype(np.float64),
                    atol=2.0e-4,
                    rtol=0.0,
                )
            ),
        }
        result.update(self._object_diagnostics(proposal))
        return result

    def projected_geometry(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(state, dtype=np.float32)
        if q.shape == (STATE_DIM,):
            q = q[None]
        if q.ndim != 2 or q.shape[1] != STATE_DIM or not np.all(np.isfinite(q)):
            raise V1781LiveError("projection state must be finite [6] or [B,6]")
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
            raise V1781LiveError("current robot state must be finite [6]")
        self.last_proposal = None
        self._render_log_posterior = None
        self._render_object_attention = None
        if self.runtime.has_pending_chunk:
            if not replace_pending:
                raise V1781LiveError("an unexecuted V17.8.1 proposal is pending")
            preserve_actual_endpoint = self.runtime.completed_chunks > 0
            self.runtime.cancel_pending_proposal()
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
            output = proposal.actor_output
            commands = output.commands[0].float().cpu().numpy().astype(np.float32)
            desired = output.desired_commands[0].float().cpu().numpy()
            predicted = (
                proposal.prediction.predicted_q_path[0]
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            if commands.shape != (COMMAND_STEPS, STATE_DIM) or not np.isfinite(commands).all():
                raise V1781LiveError("V17.8.1 produced an invalid command chunk")
            if predicted.shape != (PATH_STEPS, STATE_DIM) or not np.isfinite(predicted).all():
                raise V1781LiveError("V17.8.1 produced an invalid predicted q path")
            if np.any(commands < effective_lower[None] - 1.0e-4) or np.any(
                commands > effective_upper[None] + 1.0e-4
            ):
                raise V1781LiveError("V17.8.1 output violates deployment joint bounds")
            reversals = _nonmonotonic_recovery_joints(
                commands,
                state,
                self.action_lower_position,
                self.action_upper_position,
            )
            if reversals:
                raise V1781LiveError(
                    "V17.8.1 reverses outward before physical-bound recovery on "
                    "arm joint(s) " + ",".join(map(str, reversals))
                )
            projected = np.clip(
                desired, effective_lower[None], effective_upper[None]
            )
            fixed_point_error = float(np.max(np.abs(projected - commands)))
            if fixed_point_error > 1.0e-4:
                raise V1781LiveError(
                    "V17.8.1 command is not an absolute-limit projection"
                )
            increments = np.diff(
                np.concatenate((state[None], commands), axis=0), axis=0
            )
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
                "task_text": self.task_text,
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
            if self.runtime.has_pending_chunk:
                self.runtime.cancel_pending_proposal()
            self.last_proposal = None
            self._render_log_posterior = None
            self._render_object_attention = None
            raise

    def render_log_posterior(self) -> np.ndarray | None:
        if self._render_log_posterior is None:
            return None
        return self._render_log_posterior.copy()

    def render_object_attention(self) -> np.ndarray | None:
        if self._render_object_attention is None:
            return None
        return self._render_object_attention.copy()

    @torch.inference_mode()
    def complete(
        self,
        frames: Mapping[str, np.ndarray] | None = None,
        measured_q_path_position: np.ndarray | None = None,
        *,
        timed_duration_s: float | None = None,
        settle_duration_s: float | None = None,
        settled_state_position: np.ndarray | None = None,
        generation: int | None = None,
        commands_position: np.ndarray | None = None,
        # Compatibility aliases are accepted only when the explicit settle
        # duration is also present, so the old q5-only server cannot silently
        # masquerade as a V17.8.1 settled transaction.
        delta_t_s: float | None = None,
        actual_endpoint_state_position: np.ndarray | None = None,
        next_frames: Mapping[str, np.ndarray] | None = None,
        next_q_position: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if frames is not None and next_frames is not None:
            raise V1781LiveError("supply frames or next_frames, not both")
        observed_frames = frames if frames is not None else next_frames
        if observed_frames is None or measured_q_path_position is None:
            raise V1781LiveError("completion requires settled frames and measured path")
        if timed_duration_s is not None and delta_t_s is not None:
            raise V1781LiveError("supply timed_duration_s or delta_t_s, not both")
        timed = timed_duration_s if timed_duration_s is not None else delta_t_s
        endpoints = [
            value
            for value in (
                settled_state_position,
                actual_endpoint_state_position,
                next_q_position,
            )
            if value is not None
        ]
        if len(endpoints) != 1:
            raise V1781LiveError("completion requires exactly one settled state")
        if timed is None or not math.isfinite(float(timed)) or float(timed) <= 0.0:
            raise V1781LiveError("timed duration must be positive and finite")
        if (
            settle_duration_s is None
            or not math.isfinite(float(settle_duration_s))
            or float(settle_duration_s) < 0.0
        ):
            raise V1781LiveError(
                "explicit non-negative settle duration is required"
            )
        selected_generation = (
            self.runtime.pending_generation if generation is None else generation
        )
        pending = self.runtime.pending_commands
        if selected_generation is None or pending is None:
            raise V1781LiveError("no pending V17.8.1 proposal exists")
        commands = (
            pending[0].float().cpu().numpy()
            if commands_position is None
            else np.asarray(commands_position, dtype=np.float32)
        )
        if commands.shape != (COMMAND_STEPS, STATE_DIM) or not np.isfinite(commands).all():
            raise V1781LiveError("completion commands must be finite [5,6]")
        if not np.allclose(
            commands,
            pending[0].float().cpu().numpy(),
            atol=1.0e-5,
            rtol=0.0,
        ):
            raise V1781LiveError("completion commands differ from pending proposal")
        path = np.asarray(measured_q_path_position, dtype=np.float32)
        if path.shape != (PATH_STEPS, STATE_DIM) or not np.isfinite(path).all():
            raise V1781LiveError("measured path must be finite [6,6]")
        endpoint = np.asarray(endpoints[0], dtype=np.float32)
        if endpoint.shape != (STATE_DIM,) or not np.isfinite(endpoint).all():
            raise V1781LiveError("settled state must be finite [6]")

        rgb, tokens, valid = self._endpoint_tensors(observed_frames)
        update = self.runtime.complete_and_reobserve(
            tokens,
            valid,
            torch.from_numpy(np.ascontiguousarray(commands)).to(
                self.device, dtype=torch.float32
            ),
            torch.from_numpy(np.ascontiguousarray(path)).to(
                self.device, dtype=torch.float32
            ),
            generation=int(selected_generation),
            actual_rgb=rgb,
            settled_q=torch.from_numpy(np.ascontiguousarray(endpoint)).to(
                self.device, dtype=torch.float32
            ),
            timed_duration_s=float(timed),
            settle_duration_s=float(settle_duration_s),
        )
        innovation = update.innovation
        settled = self.runtime.latest_settled_state_innovation()
        record = {
            "belief_innovation_rms": float(
                innovation.innovation.float().square().mean().sqrt().item()
            ),
            "state_innovation_rms_position": float(
                innovation.state_residual.float().square().mean().sqrt().item()
            ),
            "settled_state_innovation_rms_position": float(
                settled.settled_endpoint_residual.float().square().mean().sqrt().item()
            ),
            "settle_delta_position": settled.settle_delta[0].float().cpu().numpy(),
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
            "timed_duration_s": float(timed),
            "settle_duration_s": float(settle_duration_s),
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
            "task_text": self.task_text,
            "task_encoder": self.task_encoder,
            "task_embedding_source": self.task_embedding_source,
            "predictive_belief_grounding": True,
            "camera_aware_transport": True,
            "settled_endpoint_clock": True,
            "task_conditioned_object_readout": True,
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
        self.runtime.reset(discard_pending=True)
        self.last_proposal = None
        self.last_completion = None
        self._render_log_posterior = None
        self._render_object_attention = None


__all__ = [
    "EXPECTED_MODEL_STATE_KEYS",
    "FISH_TASK_TEXT",
    "POSITION_UNIT",
    "V1781LiveError",
    "V1781LivePolicy",
    "V1781LiveRuntimeController",
    "load_strict_v1781_full_checkpoint",
    "load_fish_task_embedding_cache",
]
