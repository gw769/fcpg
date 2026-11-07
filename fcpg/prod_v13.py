"""Prod-V13: end-to-end FCPG-H v4.6 + the V12.4 LeWM carrier.

The maintained FCPG-H endpoint produces a discrete, content-canonical exact
assignment.  The standalone router retains a straight-through relaxation for
audits, but the current Prod-V13 training cache contains generic point slots,
not physical fingertips.  Prod-V13 therefore detaches this boundary:

* the forward value is the exact v4.6 candidate gather;
* a softmax restricted to the same immutable query family remains available
  for diagnostics but cannot send a carrier gradient into FCPG-H; and
* an endpoint-valid queried primary is a hard measurement with zero routing
  gradient, so a task loss cannot move an observed point.

The old V12.4 pixel-residual corrector and reliability heads are not used.
The frozen V12.4 ViT still supplies the only visual backbone; its projectors,
anchor bridge and *goal-free LeWM carrier* are initialized from the step-20000
endpoint and train while the 100,224-parameter PartGraph remains frozen.

V12 checkpoints contain an actor and a LeWM in one shared policy object.  That
does not make the actor stored in the V12.4 step-20000 file the production
actor.  Live deployment selects a separate complete V12.1 ``actor_best@4750``
checkpoint and a separate complete LeWM checkpoint.  Prod-V13 preserves that
dual-carrier contract:

* the V12.1 actor is a frozen external action proposer and is never loaded into
  this trainable module;
* the pure actor output/goal/task parameters embedded in V12.4 are frozen and
  their action output is ignored; and
* the shared memory/action-decoder parameters remain part of the LeWM carrier,
  because the frozen V12.4 architecture conditions dynamics on the decoder's
  latent even when the physical action is supplied explicitly.

Consequently a Prod-V13 checkpoint is not an actor checkpoint and must never
replace the frozen V12.1 production actor.  CoTracker remains an external
frozen candidate proposer and is not part of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from fcpg.canonical_query_family_part_graph_v4_6 import (
    content_canonical_candidate_order,
)
from fcpg.fcpg_h_candidate_first import PART_COUNT, TIP_PART_INDICES
from fcpg.fcpg_h_v4 import load_frozen_fcpg_h_v4
from fcpg.offline_pipeline_manifest import sha256_file
from fcpg.permutation_exact_part_graph_v4_5 import (
    PermutationExactPartGraphTrackAssociation,
)
from fcpg.query_family_constrained_part_graph_v4_3 import (
    build_query_family_support,
    validate_candidate_query_owner,
)
from fcpg.real_rgb_candidate_interface_v4_6 import (
    FROZEN_V12_4_SHA256,
)
from v10.config import FORMAT as V12_BASE_FORMAT
from v10.config import V10Config
from v10.lewm_canonical_forward import lewm_canonical_output
from v10.model import GroundedHistory, SO101V10


PROD_V13_FORMAT = "so101-prod-v13-fcpg-h-v4-6-v12-4-lewm-e2e-v4"
PROD_V13_CHECKPOINT_FORMAT = "so101-prod-v13-frozen-grounder-checkpoint-v2"
PROD_V13_METHOD_REVISION = (
    "prod-v13-frozen-v4-6-grounder-lewm-carrier-v4"
)
PROD_V13_ROUTING_TEMPERATURE = 1.0
FROZEN_V12_4_STEP = 20_000
FROZEN_PRODUCTION_ACTOR_STEP = 4_750
FROZEN_PRODUCTION_ACTOR_SHA256 = (
    "415eb6a3a89d63d5a70463fde48177374f0e69bc44d690cc54780791f5145562"
)
FCPGH_V4_6_PARAMETER_COUNT = 100_224
# The serialized policy state has another 18 scalar buffer entries
# (step_cap/action_lower/action_upper); these counts deliberately cover only
# optimizable nn.Parameters.
V12_POLICY_PARAMETER_COUNT = 4_673_121
V12_POLICY_SHARED_PARAMETER_COUNT = 3_506_112
V12_POLICY_LEWM_ONLY_PARAMETER_COUNT = 1_087_899
V12_POLICY_ACTOR_ONLY_PARAMETER_COUNT = 79_110
# The v4.6 grounder has 100,224 parameters.  The current cache contains generic
# CoTracker query slots rather than physical fingertip labels, so those weights
# are immutable.  Only the V12.4 carrier subset is optimized.
PROD_V13_TRAINABLE_PARAMETER_COUNT = 5_339_761
PROD_V13_GROUNDER_UPDATE_POLICY = (
    "frozen_exact_v4_6_generic_query_targets_are_diagnostic_only"
)


def prod_v13_state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash a named tensor state independently of serialization metadata."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()

# These are the parameters whose forward values are used only to produce the
# bundled actor output or to inject actor-only goal/task intent.  The canonical
# LeWM API disables goal/task conditioning and supplies the physical action, so
# none of these parameters lies on the LeWM loss path.
_V12_ACTOR_ONLY_PREFIXES = (
    "delta_head.",
    "goal_projection.",
    "goal_type_embedding",
    "task_class_embedding.",
    "task_class_type_embedding",
)

# The remaining policy parameters not listed here or above are genuinely
# shared: notably memory_encoder and action_decoder.  The latter is not an
# accidental actor leak; V12.4 was trained with dynamics_conditioning set to
# actor_latent, so its decoder latent is part of the fixed LeWM architecture.
_V12_LEWM_ONLY_PREFIXES = (
    "anchor_residual_head.",
    "chunk_action_projection.",
    "context_residual_head.",
    "dynamics.",
    "dynamics_input.",
    "horizon_embedding",
    "pixel_residual_head.",
    "state_residual_head.",
)


class _HardForwardSoftBackward(torch.autograd.Function):
    """Return an exact one-hot forward value and the soft path's gradient."""

    @staticmethod
    def forward(ctx, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        del ctx, soft
        return hard

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        del ctx
        return None, gradient


@dataclass(frozen=True)
class ProdV13TipRoute:
    """One flattened view/history row of exact-forward tip routing."""

    logits: torch.Tensor
    soft_assignment: torch.Tensor
    straight_through_assignment: torch.Tensor
    hard_candidate_index: torch.Tensor
    protected_primary: torch.Tensor
    tip_px: torch.Tensor
    center_px: torch.Tensor
    anchor_px: torch.Tensor
    anchor_valid: torch.Tensor
    canonical_candidate_order: torch.Tensor


@dataclass(frozen=True)
class ProdV13GroundedHistory:
    """V12.4-compatible grounded history plus auditable association state."""

    grounded: GroundedHistory
    association_logits: torch.Tensor
    soft_tip_assignment: torch.Tensor
    straight_through_tip_assignment: torch.Tensor
    hard_tip_candidate_index: torch.Tensor
    protected_primary: torch.Tensor
    canonical_candidate_order: torch.Tensor


@dataclass(frozen=True)
class ProdV13LeWMOutput:
    """Dynamics-only public output; the bundled V12.4 actor stays private."""

    dynamics_latent: torch.Tensor
    future_context: torch.Tensor
    future_state: torch.Tensor
    future_anchor_xyz: torch.Tensor
    future_anchor_px: torch.Tensor


def _validate_router_inputs(
    logits: torch.Tensor,
    candidate_tracks_px: torch.Tensor,
    candidate_valid: torch.Tensor,
    candidate_source: torch.Tensor,
    candidate_query_owner: torch.Tensor,
    primary_candidate_index: torch.Tensor,
) -> tuple[int, int, int]:
    if logits.ndim != 3 or logits.shape[1] != PART_COUNT:
        raise ValueError("logits must be [N,6,M]")
    rows, _, candidates = logits.shape
    if candidate_tracks_px.ndim != 4 or candidate_tracks_px.shape[:2] != (
        rows,
        candidates,
    ):
        raise ValueError("candidate_tracks_px must be [N,M,L,2]")
    if candidate_tracks_px.shape[-1] != 2 or candidate_tracks_px.shape[2] < 2:
        raise ValueError("candidate tracks require L>=2 and xy coordinates")
    time = int(candidate_tracks_px.shape[2])
    if candidate_valid.shape != (rows, candidates, time):
        raise ValueError("candidate_valid must be [N,M,L]")
    if candidate_valid.dtype != torch.bool:
        raise ValueError("candidate_valid must be boolean")
    if candidate_source.shape != (rows, candidates):
        raise ValueError("candidate_source must be [N,M]")
    owner = validate_candidate_query_owner(
        candidate_query_owner, candidate_source
    )
    if primary_candidate_index.shape != (rows, len(TIP_PART_INDICES)):
        raise ValueError("primary_candidate_index must be [N,2]")
    if primary_candidate_index.dtype != torch.long:
        raise ValueError("primary_candidate_index must be int64")
    if bool(
        (
            (primary_candidate_index < 0)
            | (primary_candidate_index >= candidates)
        ).any()
    ):
        raise ValueError("primary candidate index is outside the bank")
    if bool(primary_candidate_index[:, 0].eq(primary_candidate_index[:, 1]).any()):
        raise ValueError("fixed and moving primaries must be distinct")
    primary_owner = owner.gather(1, primary_candidate_index)
    expected_owner = torch.arange(
        len(TIP_PART_INDICES),
        dtype=torch.long,
        device=logits.device,
    )[None].expand_as(primary_owner)
    if not bool(primary_owner.eq(expected_owner).all()):
        raise ValueError("each primary must belong to its supplied query family")
    for name, value in (
        ("logits", logits),
        ("candidate_tracks_px", candidate_tracks_px),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")
    return rows, candidates, time


def straight_through_query_family_route(
    logits: torch.Tensor,
    candidate_tracks_px: torch.Tensor,
    candidate_valid: torch.Tensor,
    candidate_source: torch.Tensor,
    candidate_query_owner: torch.Tensor,
    primary_candidate_index: torch.Tensor,
    *,
    temperature: float = PROD_V13_ROUTING_TEMPERATURE,
) -> ProdV13TipRoute:
    """Route two tips with exact v4.6 forward values and soft gradients.

    Query-owner supports for fixed and moving tips are disjoint.  Consequently
    their rows of the full six-part exact assignment cannot compete with each
    other or with the four owner=-1 auxiliary rows.  Selecting the first
    maximum in content-canonical order is therefore exactly the public v4.6
    tip result, including exact-score tie handling.
    """

    rows, candidates, time = _validate_router_inputs(
        logits,
        candidate_tracks_px,
        candidate_valid,
        candidate_source,
        candidate_query_owner,
        primary_candidate_index,
    )
    if not torch.isfinite(torch.tensor(float(temperature))) or temperature <= 0:
        raise ValueError("routing temperature must be finite and positive")
    endpoint_valid = candidate_valid[:, :, -1]
    allowed_all = build_query_family_support(
        candidate_query_owner, endpoint_valid
    )
    # A full v4.6 decode needs four different owner=-1 candidates.  Enforce the
    # deployment contract even though only the two public tips are routed here.
    auxiliary_support = allowed_all[:, TIP_PART_INDICES[0] - 1]
    if bool(auxiliary_support.sum(dim=1).lt(PART_COUNT - len(TIP_PART_INDICES)).any()):
        raise ValueError("not enough owner=-1 candidates for the six-part exact solve")

    tip_rows = torch.as_tensor(
        TIP_PART_INDICES, dtype=torch.long, device=logits.device
    )
    tip_logits = logits.index_select(1, tip_rows)
    tip_allowed = allowed_all.index_select(1, tip_rows)
    scaled = tip_logits.float() / float(temperature)
    soft = scaled.masked_fill(~tip_allowed, float("-inf")).softmax(dim=-1)
    if not bool(torch.isfinite(soft).all()):
        raise ValueError("soft query-family assignment is non-finite")

    order = content_canonical_candidate_order(
        candidate_tracks_px,
        candidate_valid,
        candidate_source,
        candidate_query_owner,
    )
    canonical_logits = tip_logits.gather(
        2, order[:, None].expand(-1, len(TIP_PART_INDICES), -1)
    )
    canonical_allowed = tip_allowed.gather(
        2, order[:, None].expand(-1, len(TIP_PART_INDICES), -1)
    )
    canonical_selected = canonical_logits.masked_fill(
        ~canonical_allowed, float("-inf")
    ).argmax(dim=-1)
    hard_index = order.gather(1, canonical_selected)
    protected = endpoint_valid.gather(1, primary_candidate_index)
    hard_index = torch.where(protected, primary_candidate_index, hard_index)
    hard = F.one_hot(hard_index, num_classes=candidates).to(dtype=soft.dtype)
    straight_through = _HardForwardSoftBackward.apply(hard, soft)
    # A valid primary is a physical measurement, not a differentiable policy
    # choice.  It remains one-hot in both forward and backward passes.
    straight_through = torch.where(
        protected[..., None], hard.detach(), straight_through
    )

    endpoint_px = candidate_tracks_px[:, :, time - 1].float()
    exact_tip = endpoint_px.gather(
        1, hard_index[..., None].expand(-1, -1, 2)
    )
    # Do not form the public hard measurement through a one-hot matrix
    # multiplication.  CUDA autocast may execute that contraction in bf16 and
    # round an otherwise exact measured coordinate.  The forward value is the
    # literal gather; only the backward value follows the soft assignment.
    soft_tip = torch.einsum("ntm,nmd->ntd", soft, endpoint_px)
    tip_px = _HardForwardSoftBackward.apply(exact_tip, soft_tip)
    tip_px = torch.where(protected[..., None], exact_tip.detach(), tip_px)
    center_px = tip_px.mean(dim=1)
    anchor_px = torch.cat((center_px[:, None], tip_px), dim=1)
    selected_valid = endpoint_valid.gather(1, hard_index)
    anchor_valid = torch.cat(
        (selected_valid.all(dim=1, keepdim=True), selected_valid), dim=1
    )
    if not bool(anchor_valid.all()):
        raise AssertionError("Prod-V13 routed an endpoint-invalid candidate")
    # The straight-through forward value must be an exact measured gather.
    if not torch.equal(tip_px.detach(), exact_tip.detach()):
        raise AssertionError("straight-through route changed the hard forward value")
    return ProdV13TipRoute(
        logits=logits,
        soft_assignment=soft,
        straight_through_assignment=straight_through,
        hard_candidate_index=hard_index,
        protected_primary=protected,
        tip_px=tip_px,
        center_px=center_px,
        anchor_px=anchor_px,
        anchor_valid=anchor_valid,
        canonical_candidate_order=order,
    )


def complete_query_family_support_mask(
    candidate_valid: torch.Tensor,
    candidate_query_owner: torch.Tensor,
) -> torch.Tensor:
    """Return rows where global/fixed/moving endpoint support all exist."""
    if candidate_valid.ndim < 2 or candidate_valid.dtype != torch.bool:
        raise ValueError("candidate_valid must be boolean [...,M,T]")
    if candidate_query_owner.shape != candidate_valid.shape[:-1]:
        raise ValueError("candidate_query_owner must align with [...,M]")
    if candidate_query_owner.dtype != torch.long:
        raise ValueError("candidate_query_owner must be int64")
    if bool(
        ((candidate_query_owner < -1) | (candidate_query_owner > 1)).any()
    ):
        raise ValueError("candidate query owner must lie in {-1,0,1}")
    endpoint_valid = candidate_valid[..., -1]
    complete = torch.ones(
        endpoint_valid.shape[:-1],
        dtype=torch.bool,
        device=endpoint_valid.device,
    )
    for family in (-1, 0, 1):
        complete &= (
            endpoint_valid & candidate_query_owner.eq(family)
        ).any(dim=-1)
    return complete


def masked_part_assignment_cross_entropy(
    logits: torch.Tensor,
    target_candidate_index: torch.Tensor,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """The single Prod-V13 grounding objective, with explicit target masking."""

    if logits.ndim < 3 or logits.shape[-2] != PART_COUNT:
        raise ValueError("logits must end in [6,M]")
    expected = logits.shape[:-1]
    if target_candidate_index.shape != expected:
        raise ValueError("target_candidate_index must match logits without M")
    if target_valid.shape != expected or target_valid.dtype != torch.bool:
        raise ValueError("target_valid must be bool and match assignment targets")
    if target_candidate_index.dtype != torch.long:
        raise ValueError("target_candidate_index must be int64")
    candidates = int(logits.shape[-1])
    selected = target_candidate_index[target_valid]
    if selected.numel() == 0:
        raise ValueError("association objective has no valid target")
    if bool(((selected < 0) | (selected >= candidates)).any()):
        raise ValueError("a valid assignment target is outside the candidate bank")
    return F.cross_entropy(logits[target_valid], selected)


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix) for prefix in prefixes)


def v12_policy_role_inventory(policy: nn.Module) -> dict[str, Any]:
    """Partition the V12 policy into shared, LeWM-only, and actor-only state."""

    roles: dict[str, dict[str, list[int]]] = {
        "shared_actor_lewm": {},
        "lewm_only": {},
        "actor_only_frozen": {},
    }
    for name, parameter in policy.named_parameters():
        if _matches_prefix(name, _V12_ACTOR_ONLY_PREFIXES):
            role = "actor_only_frozen"
        elif _matches_prefix(name, _V12_LEWM_ONLY_PREFIXES):
            role = "lewm_only"
        else:
            role = "shared_actor_lewm"
        roles[role][name] = list(parameter.shape)
    counts = {
        role: sum(policy.get_parameter(name).numel() for name in names)
        for role, names in roles.items()
    }
    expected = {
        "shared_actor_lewm": V12_POLICY_SHARED_PARAMETER_COUNT,
        "lewm_only": V12_POLICY_LEWM_ONLY_PARAMETER_COUNT,
        "actor_only_frozen": V12_POLICY_ACTOR_ONLY_PARAMETER_COUNT,
    }
    if counts != expected or sum(counts.values()) != V12_POLICY_PARAMETER_COUNT:
        raise ValueError(
            "V12 policy role inventory drifted: "
            f"counts={counts} expected={expected}"
        )
    return {"counts": counts, "parameter_names": roles}


def validate_production_actor_checkpoint(path: str | Path) -> dict[str, Any]:
    """Validate the frozen V12.1 actor reference without loading it into V13."""

    resolved = Path(path).expanduser().resolve()
    digest = sha256_file(resolved)
    if digest != FROZEN_PRODUCTION_ACTOR_SHA256:
        raise ValueError("Prod-V13 requires the frozen V12.1 actor_best@4750 file")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    architecture = payload.get("architecture")
    if (
        payload.get("format") != V12_BASE_FORMAT
        or int(payload.get("step", -1)) != FROZEN_PRODUCTION_ACTOR_STEP
        or not isinstance(architecture, dict)
        or architecture.get("family") != "v12"
        or architecture.get("actor_pool") != "fish_only"
        or architecture.get("lewm_pool") != "broad_plus_fish"
    ):
        raise ValueError("production actor endpoint identity or architecture drifted")
    return {
        "path": str(resolved),
        "sha256": digest,
        "step": int(payload["step"]),
        "family": architecture["family"],
        "role": "frozen_external_action_proposer",
        "loaded_into_prod_v13": False,
        "optimizer_membership": False,
    }


def _checkpoint_config(payload: dict[str, Any]) -> V10Config:
    raw = payload.get("config")
    if not isinstance(raw, dict):
        raise ValueError("V12.4 checkpoint has no config mapping")
    values = dict(raw)
    values["history_offsets"] = tuple(int(x) for x in values["history_offsets"])
    values["future_offsets"] = tuple(int(x) for x in values["future_offsets"])
    config = V10Config(**values)
    config.validate()
    if config.num_views != 2 or config.obs_horizon != 2:
        raise ValueError("Prod-V13 requires the actual two-view/two-frame V12.4 endpoint")
    return config


def _load_v12_4(path: Path, *, device: torch.device) -> tuple[SO101V10, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if sha256_file(resolved) != FROZEN_V12_4_SHA256:
        raise ValueError("Prod-V13 requires the frozen V12.4 step-20000 file")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    architecture = payload.get("architecture")
    if (
        payload.get("format") != V12_BASE_FORMAT
        or int(payload.get("step", -1)) != FROZEN_V12_4_STEP
        or not isinstance(architecture, dict)
        or architecture.get("family") != "v12"
        or architecture.get("lewm_sigreg_weight") != 0.02
    ):
        raise ValueError("V12.4 endpoint identity or architecture drifted")
    config = _checkpoint_config(payload)
    model = SO101V10(
        config=config,
        vit_checkpoint=resolved,
        structured_goal=True,
        structured_goal_source="episode_terminal_state",
        goal_query_residual=False,
        task_class_conditioning=True,
        state_residual_mode=str(architecture.get("state_residual_mode", "uniform")),
        state_residual_steps=float(architecture.get("state_residual_steps", 1.0)),
    ).to(device)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("V12.4 checkpoint has no model state")
    model.load_state_dict(state, strict=True)
    contract = payload.get("action_contract")
    if not isinstance(contract, dict):
        raise ValueError("V12.4 checkpoint has no action contract")
    model.set_action_contract(
        contract["step_cap"], contract["lower"], contract["upper"]
    )
    return model, payload


class ProdV13(nn.Module):
    """Frozen v4.6 association feeding a trainable V12.4 LeWM carrier.

    The public forward remains a single composed graph.  Training deliberately
    stops gradients at the FCPG-H measurement boundary because the current
    Prod-V13 cache labels generic meta-tracking points, not robot fingertips.
    """

    def __init__(
        self,
        *,
        v12: SO101V10,
        grounder: PermutationExactPartGraphTrackAssociation,
        production_actor_reference: dict[str, Any],
    ) -> None:
        super().__init__()
        self.v12 = v12
        self.grounder = grounder
        self.production_actor_reference = dict(production_actor_reference)
        self._configure_trainability()

    @classmethod
    def from_frozen_endpoints(
        cls,
        *,
        production_actor_checkpoint: str | Path,
        v12_checkpoint: str | Path,
        fcpg_checkpoint: str | Path,
        device: str | torch.device,
    ) -> "ProdV13":
        target = torch.device(device)
        actor_reference = validate_production_actor_checkpoint(
            production_actor_checkpoint
        )
        v12, _ = _load_v12_4(Path(v12_checkpoint), device=target)
        fcpg_v4, _ = load_frozen_fcpg_h_v4(
            fcpg_checkpoint, device=target, trainable=False
        )
        return cls(
            v12=v12,
            grounder=fcpg_v4.grounder,
            production_actor_reference=actor_reference,
        )

    def _configure_trainability(self) -> None:
        self.requires_grad_(False)
        self.grounder.requires_grad_(False)
        for module in (
            self.v12.context_projector,
            self.v12.patch_projector,
            self.v12.bridge,
            self.v12.policy,
        ):
            module.requires_grad_(True)
        # V12.4 stores a structured-goal actor shell in the same policy object.
        # The deployed actor is a different complete checkpoint.  Freeze every
        # pure actor parameter here; shared memory/action-decoder parameters are
        # retained because canonical V12.4 LeWM actually consumes them.
        inventory = v12_policy_role_inventory(self.v12.policy)
        for name in inventory["parameter_names"]["actor_only_frozen"]:
            self.v12.policy.get_parameter(name).requires_grad_(False)
        self.v12.vit.requires_grad_(False)
        self.v12.corrector.requires_grad_(False)
        self.v12.anchor_reliability_head.requires_grad_(False)
        self.v12.view_reliability_head.requires_grad_(False)
        if self.trainable_parameter_count != PROD_V13_TRAINABLE_PARAMETER_COUNT:
            raise ValueError(
                "Prod-V13 trainable parameter inventory drifted: "
                f"{self.trainable_parameter_count:,} != "
                f"{PROD_V13_TRAINABLE_PARAMETER_COUNT:,}"
            )

    def train(self, mode: bool = True) -> "ProdV13":
        super().train(mode)
        self.grounder.eval()
        self.v12.vit.eval()
        self.v12.corrector.eval()
        self.v12.anchor_reliability_head.eval()
        self.v12.view_reliability_head.eval()
        return self

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def grounder_trainable(self) -> bool:
        return any(parameter.requires_grad for parameter in self.grounder.parameters())

    def parameter_contract(self) -> dict[str, Any]:
        trainable = {
            name: list(parameter.shape)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        frozen = {
            name: list(parameter.shape)
            for name, parameter in self.named_parameters()
            if not parameter.requires_grad
        }
        policy_roles = v12_policy_role_inventory(self.v12.policy)
        return {
            "format": PROD_V13_FORMAT,
            "method_revision": PROD_V13_METHOD_REVISION,
            "grounder_update_policy": PROD_V13_GROUNDER_UPDATE_POLICY,
            "trainable_parameters": sum(
                self.get_parameter(name).numel() for name in trainable
            ),
            "frozen_parameters": sum(
                self.get_parameter(name).numel() for name in frozen
            ),
            "trainable_parameter_names": trainable,
            "frozen_parameter_names": frozen,
            "unused_frozen_v12_modules": [
                "corrector",
                "anchor_reliability_head",
                "view_reliability_head",
            ],
            "dual_carrier_contract": {
                "production_actor": self.production_actor_reference,
                "production_actor_loaded_into_training_module": False,
                "v12_4_role": "goal_free_lewm_carrier",
                "v12_4_bundled_actor_output_used": False,
                "v12_4_bundled_actor_checkpoint_authorized": False,
                "shared_action_decoder_reason": (
                    "V12.4 dynamics_conditioning=actor_latent; decoder latent is "
                    "part of canonical LeWM while its action output is ignored"
                ),
            },
            "v12_policy_roles": policy_roles,
        }

    def load_trainable_state(
        self, state: Mapping[str, torch.Tensor]
    ) -> None:
        """Restore only the LeWM-carrier state over exact frozen endpoints."""

        expected = {
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if set(state) != expected:
            raise ValueError("Prod-V13 trainable checkpoint names drifted")
        with torch.no_grad():
            for name, value in state.items():
                parameter = self.get_parameter(name)
                if parameter.shape != value.shape or parameter.dtype != value.dtype:
                    raise ValueError(
                        f"Prod-V13 trainable checkpoint {name} schema drifted"
                    )
                parameter.copy_(value.to(parameter.device))

    def _validate_history_inputs(
        self,
        rgb: torch.Tensor,
        candidate_tracks_px: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_source: torch.Tensor,
        candidate_query_owner: torch.Tensor,
        primary_candidate_index: torch.Tensor,
        part_motion: torch.Tensor,
        anchor_xyz_history: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if rgb.ndim != 6:
            raise ValueError("rgb must be [B,H,V,3,224,224]")
        batch, history, views, channels, height, width = rgb.shape
        config = self.v12.config
        if (history, views, channels, height, width) != (
            config.obs_horizon,
            config.num_views,
            3,
            config.image_size,
            config.image_size,
        ):
            raise ValueError("RGB does not match the frozen V12.4 contract")
        candidates, event_time = self._validate_history_metadata(
            batch=batch,
            history=history,
            views=views,
            candidate_tracks_px=candidate_tracks_px,
            candidate_valid=candidate_valid,
            candidate_source=candidate_source,
            candidate_query_owner=candidate_query_owner,
            primary_candidate_index=primary_candidate_index,
            part_motion=part_motion,
            anchor_xyz_history=anchor_xyz_history,
            view_valid=view_valid,
        )
        return batch, history, views, candidates, event_time

    @staticmethod
    def _validate_history_metadata(
        *,
        batch: int,
        history: int,
        views: int,
        candidate_tracks_px: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_source: torch.Tensor,
        candidate_query_owner: torch.Tensor,
        primary_candidate_index: torch.Tensor,
        part_motion: torch.Tensor,
        anchor_xyz_history: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> tuple[int, int]:
        if candidate_tracks_px.ndim != 6 or candidate_tracks_px.shape[:3] != (
            batch,
            history,
            views,
        ):
            raise ValueError("candidate_tracks_px must be [B,H,V,M,L,2]")
        candidates, event_time = candidate_tracks_px.shape[3:5]
        if candidate_tracks_px.shape[-1] != 2 or event_time < 2:
            raise ValueError("candidate event tracks require L>=2 and xy")
        if candidate_valid.shape != (
            batch,
            history,
            views,
            candidates,
            event_time,
        ):
            raise ValueError("candidate_valid shape drifted")
        if candidate_source.shape != (batch, history, views, candidates):
            raise ValueError("candidate_source shape drifted")
        if candidate_query_owner.shape != (batch, history, views, candidates):
            raise ValueError("candidate_query_owner shape drifted")
        if primary_candidate_index.shape != (batch, history, views, 2):
            raise ValueError("primary_candidate_index shape drifted")
        if part_motion.shape != (batch, history, event_time, PART_COUNT, 44):
            raise ValueError("part_motion must be [B,H,L,6,44]")
        if anchor_xyz_history.shape != (batch, history, 3, 3):
            raise ValueError("anchor_xyz_history must be [B,H,3,3]")
        if view_valid.shape != (batch, history, views) or view_valid.dtype != torch.bool:
            raise ValueError("view_valid must be bool [B,H,V]")
        return candidates, event_time

    def _route_and_bridge_cached(
        self,
        *,
        vit_cls_history: torch.Tensor,
        vit_patch_history: torch.Tensor,
        candidate_tracks_px: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_source: torch.Tensor,
        candidate_query_owner: torch.Tensor,
        primary_candidate_index: torch.Tensor,
        part_motion: torch.Tensor,
        anchor_xyz_history: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> ProdV13GroundedHistory:
        """Run the trainable path from immutable raw ViT tokens.

        Cache builders store the frozen ViT output *before* the V12.4
        projectors.  Consequently this path removes video decoding and the
        frozen ViT from every optimizer step without freezing either projector
        or changing the RGB forward value.
        """

        if vit_cls_history.ndim != 4:
            raise ValueError("vit_cls_history must be [B,H,V,192]")
        batch, history, views, latent = vit_cls_history.shape
        config = self.v12.config
        if (history, views, latent) != (
            config.obs_horizon,
            config.num_views,
            config.latent_dim,
        ):
            raise ValueError("cached ViT CLS tokens do not match V12.4")
        if vit_patch_history.shape != (
            batch,
            history,
            views,
            256,
            config.latent_dim,
        ):
            raise ValueError("vit_patch_history must be [B,H,V,256,192]")
        if not vit_cls_history.is_floating_point() or not vit_patch_history.is_floating_point():
            raise ValueError("cached ViT tokens must be floating point")
        if not bool(torch.isfinite(vit_cls_history).all()) or not bool(
            torch.isfinite(vit_patch_history).all()
        ):
            raise ValueError("cached ViT tokens contain non-finite values")
        candidates, event_time = self._validate_history_metadata(
            batch=batch,
            history=history,
            views=views,
            candidate_tracks_px=candidate_tracks_px,
            candidate_valid=candidate_valid,
            candidate_source=candidate_source,
            candidate_query_owner=candidate_query_owner,
            primary_candidate_index=primary_candidate_index,
            part_motion=part_motion,
            anchor_xyz_history=anchor_xyz_history,
            view_valid=view_valid,
        )
        rows = batch * history * views
        tracks = candidate_tracks_px.reshape(rows, candidates, event_time, 2)
        valid = candidate_valid.reshape(rows, candidates, event_time)
        source = candidate_source.reshape(rows, candidates)
        owner = candidate_query_owner.reshape(rows, candidates)
        primary = primary_candidate_index.reshape(rows, 2)
        motion = (
            part_motion[:, :, None]
            .expand(-1, -1, views, -1, -1, -1)
            .reshape(rows, event_time, PART_COUNT, 44)
        )
        row_valid = view_valid.reshape(rows)

        # Nominal means the supplied primary measurement, not a calibrated
        # projection.  It remains available for bridge compatibility even when
        # the primary's endpoint-valid bit is false.
        primary_track = tracks.gather(
            1,
            primary[:, :, None, None].expand(-1, -1, event_time, 2),
        )
        nominal_tip = primary_track[:, :, -1]
        nominal_anchor = torch.cat(
            (nominal_tip.mean(dim=1, keepdim=True), nominal_tip), dim=1
        )

        # Association runs under CUDA bf16 autocast while cached tracks are
        # fp16.  Keep the public/audited routing tensors in fp32 so indexed
        # writes never depend on whichever low-precision dtype autocast chose
        # for the scorer.
        logits = torch.zeros(
            (rows, PART_COUNT, candidates),
            dtype=torch.float32,
            device=tracks.device,
        )
        soft = torch.zeros(
            (rows, 2, candidates), dtype=torch.float32, device=tracks.device
        )
        straight = torch.zeros_like(soft)
        hard_index = primary.clone()
        protected = torch.zeros((rows, 2), dtype=torch.bool, device=tracks.device)
        corrected = nominal_anchor.float()
        route_valid = torch.zeros((rows, 3), dtype=torch.bool, device=tracks.device)
        canonical_order = torch.arange(
            candidates, dtype=torch.long, device=tracks.device
        )[None].expand(rows, -1)

        routing_valid = complete_query_family_support_mask(valid, owner)
        selected_rows = torch.nonzero(
            row_valid & routing_valid, as_tuple=False
        ).flatten()
        if selected_rows.numel():
            # Generic query slots are useful for exercising the composed
            # measurement->LeWM path, but they are not fingertip supervision.
            # Keep the exact v4.6 measurement immutable and detach both its
            # logits and the straight-through route from carrier gradients.
            with torch.no_grad():
                association = self.grounder(
                    tracks.index_select(0, selected_rows),
                    valid.index_select(0, selected_rows),
                    source.index_select(0, selected_rows),
                    motion.index_select(0, selected_rows),
                )
                route = straight_through_query_family_route(
                    association.logits,
                    tracks.index_select(0, selected_rows),
                    valid.index_select(0, selected_rows),
                    source.index_select(0, selected_rows),
                    owner.index_select(0, selected_rows),
                    primary.index_select(0, selected_rows),
                )
            logits = logits.index_copy(0, selected_rows, route.logits.float())
            soft = soft.index_copy(0, selected_rows, route.soft_assignment)
            straight = straight.index_copy(
                0, selected_rows, route.straight_through_assignment
            )
            hard_index = hard_index.index_copy(
                0, selected_rows, route.hard_candidate_index
            )
            protected = protected.index_copy(
                0, selected_rows, route.protected_primary
            )
            corrected = corrected.index_copy(0, selected_rows, route.anchor_px)
            route_valid = route_valid.index_copy(
                0, selected_rows, route.anchor_valid
            )
            canonical_order = canonical_order.index_copy(
                0, selected_rows, route.canonical_candidate_order
            )

        flat_context = self.v12.context_projector(
            vit_cls_history.reshape(rows, config.latent_dim)
        )
        flat_patches = self.v12.patch_projector(
            vit_patch_history.reshape(rows, 256, config.latent_dim)
        )
        view_batch = batch * views
        nominal_history = nominal_anchor.reshape(batch, history, views, 3, 2)
        corrected_history = corrected.reshape(batch, history, views, 3, 2)
        effective_valid = route_valid.reshape(batch, history, views, 3)
        effective_valid &= view_valid[..., None]
        bridge = self.v12.bridge(
            flat_context.reshape(batch, history, views, -1)
            .permute(0, 2, 1, 3)
            .reshape(view_batch, history, -1),
            flat_patches.reshape(batch, history, views, 256, -1)
            .permute(0, 2, 1, 3, 4)
            .reshape(view_batch, history, 256, -1),
            nominal_history.permute(0, 2, 1, 3, 4).reshape(
                view_batch, history, 3, 2
            ),
            corrected_history.permute(0, 2, 1, 3, 4).reshape(
                view_batch, history, 3, 2
            ),
            anchor_xyz_history[:, None]
            .expand(-1, views, -1, -1, -1)
            .reshape(view_batch, history, 3, 3),
            effective_valid.permute(0, 2, 1, 3).reshape(
                view_batch, history, 3
            ),
            image_size=(self.v12.config.image_size, self.v12.config.image_size),
        )
        context = bridge.context_embedding.reshape(
            batch, views, history, -1
        ).permute(0, 2, 1, 3)
        anchor_tokens = bridge.anchor_tokens.reshape(
            batch, views, history, 3, -1
        ).permute(0, 2, 1, 3, 4)
        grounded = GroundedHistory(
            context=context,
            patch_tokens=flat_patches.reshape(
                batch, history, views, 256, self.v12.config.latent_dim
            ),
            anchor_tokens=anchor_tokens,
            q_corrected_px=corrected_history,
            anchor_valid=effective_valid,
            view_valid=view_valid,
            anchor_reliability_logit=torch.zeros(
                batch,
                history,
                views,
                3,
                device=vit_cls_history.device,
                dtype=context.dtype,
            ),
            view_reliability_logit=torch.zeros(
                batch,
                history,
                views,
                device=vit_cls_history.device,
                dtype=context.dtype,
            ),
        )
        return ProdV13GroundedHistory(
            grounded=grounded,
            association_logits=logits.reshape(
                batch, history, views, PART_COUNT, candidates
            ),
            soft_tip_assignment=soft.reshape(
                batch, history, views, 2, candidates
            ),
            straight_through_tip_assignment=straight.reshape(
                batch, history, views, 2, candidates
            ),
            hard_tip_candidate_index=hard_index.reshape(
                batch, history, views, 2
            ),
            protected_primary=protected.reshape(batch, history, views, 2),
            canonical_candidate_order=canonical_order.reshape(
                batch, history, views, candidates
            ),
        )

    def encode_grounded_cached(
        self,
        *,
        vit_cls_history: torch.Tensor,
        vit_patch_history: torch.Tensor,
        candidate_tracks_px: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_source: torch.Tensor,
        candidate_query_owner: torch.Tensor,
        primary_candidate_index: torch.Tensor,
        part_motion: torch.Tensor,
        anchor_xyz_history: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> ProdV13GroundedHistory:
        """Public no-RGB training path over a SHA-bound frozen-ViT cache."""

        return self._route_and_bridge_cached(
            vit_cls_history=vit_cls_history,
            vit_patch_history=vit_patch_history,
            candidate_tracks_px=candidate_tracks_px,
            candidate_valid=candidate_valid,
            candidate_source=candidate_source,
            candidate_query_owner=candidate_query_owner,
            primary_candidate_index=primary_candidate_index,
            part_motion=part_motion,
            anchor_xyz_history=anchor_xyz_history,
            view_valid=view_valid,
        )

    def encode_grounded_history(
        self,
        *,
        rgb: torch.Tensor,
        candidate_tracks_px: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_source: torch.Tensor,
        candidate_query_owner: torch.Tensor,
        primary_candidate_index: torch.Tensor,
        part_motion: torch.Tensor,
        anchor_xyz_history: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> ProdV13GroundedHistory:
        """Reference RGB path; numerically matched to ``encode_grounded_cached``."""

        self._validate_history_inputs(
            rgb,
            candidate_tracks_px,
            candidate_valid,
            candidate_source,
            candidate_query_owner,
            primary_candidate_index,
            part_motion,
            anchor_xyz_history,
            view_valid,
        )
        hidden = self.v12.vit(rgb.flatten(0, 2))
        batch, history, views = rgb.shape[:3]
        return self._route_and_bridge_cached(
            vit_cls_history=hidden[:, 0].reshape(batch, history, views, -1),
            vit_patch_history=hidden[:, 1:].reshape(
                batch, history, views, 256, -1
            ),
            candidate_tracks_px=candidate_tracks_px,
            candidate_valid=candidate_valid,
            candidate_source=candidate_source,
            candidate_query_owner=candidate_query_owner,
            primary_candidate_index=primary_candidate_index,
            part_motion=part_motion,
            anchor_xyz_history=anchor_xyz_history,
            view_valid=view_valid,
        )

    def lewm(
        self,
        grounded: ProdV13GroundedHistory,
        *,
        anchor_xyz_history: torch.Tensor,
        state_history: torch.Tensor,
        past_action_chunks: torch.Tensor,
        action: torch.Tensor,
    ) -> ProdV13LeWMOutput:
        """Run goal-free V12.4 LeWM; never expose its bundled actor output."""

        value = grounded.grounded
        output = lewm_canonical_output(
            self.v12.policy,
            per_view_context=value.context,
            per_view_anchor_tokens=value.anchor_tokens,
            anchor_valid=value.anchor_valid,
            view_valid=value.view_valid,
            q_corrected_px=value.q_corrected_px,
            anchor_xyz_history=anchor_xyz_history,
            state_history=state_history,
            past_action_chunks=past_action_chunks,
            action=action,
        )
        # The underlying V12 class computes an action tensor because its actor
        # and LeWM share memory/decoder state.  Do not expose that tensor from
        # Prod-V13: deployment actions come exclusively from the independently
        # frozen V12.1 actor carrier.
        return ProdV13LeWMOutput(
            dynamics_latent=output.dynamics_latent,
            future_context=output.future_context,
            future_state=output.future_state,
            future_anchor_xyz=output.future_anchor_xyz,
            future_anchor_px=output.future_anchor_px,
        )


def load_prod_v13_checkpoint(
    checkpoint: str | Path,
    *,
    production_actor_checkpoint: str | Path,
    v12_checkpoint: str | Path,
    fcpg_checkpoint: str | Path,
    device: str | torch.device,
    require_fixed_endpoint: bool = True,
) -> tuple[ProdV13, dict[str, Any]]:
    """Load a complete Prod-V13 endpoint without tensor grafting.

    The serialized file intentionally contains only the 5.34M trainable
    tensors.  All three immutable carrier files must be supplied and match the
    SHA bindings written by the trainer.
    """

    resolved = Path(checkpoint).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if (
        payload.get("format") != PROD_V13_CHECKPOINT_FORMAT
        or payload.get("prod_v13_format") != PROD_V13_FORMAT
        or payload.get("method_revision") != PROD_V13_METHOD_REVISION
        or payload.get("grounder_update_policy")
        != PROD_V13_GROUNDER_UPDATE_POLICY
        or bool(payload.get("actor_loss"))
        or bool(payload.get("sealed_test_read"))
    ):
        raise ValueError("Prod-V13 checkpoint identity/safety contract drifted")
    step = int(payload.get("step", -1))
    fixed = int(payload.get("fixed_endpoint_step", -1))
    if step <= 0 or fixed <= 0 or (require_fixed_endpoint and step != fixed):
        raise ValueError("Prod-V13 checkpoint is not its registered fixed endpoint")
    bindings = {
        "production_actor_reference_sha256": sha256_file(
            production_actor_checkpoint
        ),
        "v12_checkpoint_sha256": sha256_file(v12_checkpoint),
        "fcpg_initialization_sha256": sha256_file(fcpg_checkpoint),
    }
    for name, actual in bindings.items():
        if payload.get(name) != actual:
            raise ValueError(f"Prod-V13 checkpoint {name} binding mismatch")
    state = payload.get("trainable_state")
    if not isinstance(state, Mapping):
        raise ValueError("Prod-V13 checkpoint lacks trainable state")
    if payload.get("trainable_state_sha256") != prod_v13_state_dict_sha256(state):
        raise ValueError("Prod-V13 checkpoint trainable-state SHA mismatch")
    model = ProdV13.from_frozen_endpoints(
        production_actor_checkpoint=production_actor_checkpoint,
        v12_checkpoint=v12_checkpoint,
        fcpg_checkpoint=fcpg_checkpoint,
        device=device,
    )
    frozen_grounder_sha = prod_v13_state_dict_sha256(model.grounder.state_dict())
    if payload.get("frozen_grounder_state_sha256") != frozen_grounder_sha:
        raise ValueError("Prod-V13 frozen grounder state binding mismatch")
    if payload.get("parameter_contract") != model.parameter_contract():
        raise ValueError("Prod-V13 checkpoint parameter contract drifted")
    model.load_trainable_state(state)
    return model.eval(), payload


__all__ = [
    "FCPGH_V4_6_PARAMETER_COUNT",
    "FROZEN_PRODUCTION_ACTOR_SHA256",
    "FROZEN_PRODUCTION_ACTOR_STEP",
    "PROD_V13_FORMAT",
    "PROD_V13_CHECKPOINT_FORMAT",
    "PROD_V13_METHOD_REVISION",
    "PROD_V13_GROUNDER_UPDATE_POLICY",
    "PROD_V13_ROUTING_TEMPERATURE",
    "PROD_V13_TRAINABLE_PARAMETER_COUNT",
    "ProdV13",
    "ProdV13GroundedHistory",
    "ProdV13LeWMOutput",
    "ProdV13TipRoute",
    "complete_query_family_support_mask",
    "load_prod_v13_checkpoint",
    "masked_part_assignment_cross_entropy",
    "straight_through_query_family_route",
    "prod_v13_state_dict_sha256",
    "v12_policy_role_inventory",
    "validate_production_actor_checkpoint",
]
