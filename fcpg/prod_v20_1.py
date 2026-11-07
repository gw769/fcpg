"""V20.1 joint-pair feedback controller. No hardware side effects.

V20 data/policy primitives are reused, but learned modules are newly initialized
and checkpoint/cache contracts are distinct. Calibrated positions are NOT degrees.
"""
from dataclasses import dataclass, fields, replace
import math
from typing import Any

import torch
from torch import nn

from .prod_v20_policy import V20Actor, ActualHistory, empty_history, append_completed
from .prod_v20_1_arm import V201LeWM, ARMProjector
from .prod_v20_1_belief import JointPairBelief
from .prod_v20_1_observer import ObserverOutput, V201PairObserver

VERSION = "20.1.0"
OBSERVATION_CONTRACT = "mechanical-joint-belief-feedback-v1"
CONTRACT = "mechanical-joint-belief-feedback-v2-arm-sigreg"
POSITION_UNIT = "lerobot_calibrated_position"
VIEW_ORDER = ("wrist", "column")
INNOVATION_MODES = ("full", "belief_only", "q_only", "neither")


@dataclass(frozen=True)
class V201Config:
    seed: int = 20100
    components: int = 4
    dimension: int = 192
    history_steps: int = 3
    command_steps: int = 5
    task_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    position_unit: str = POSITION_UNIT
    actor_mode: str = "full"
    pooling_channels: int = 8
    state_representation: str = "center-halfvector-full-joint-covariance"

    def __post_init__(self):
        if (self.components, self.dimension, self.history_steps, self.command_steps, self.pooling_channels) != (4,192,3,5,8):
            raise ValueError("V20.1 architecture is fixed")
        if self.position_unit != POSITION_UNIT or self.actor_mode not in ("full", "no_innovation"):
            raise ValueError("invalid V20.1 configuration")


def detach_belief(belief):
    return belief.detach()


def canonical_belief(belief: JointPairBelief) -> JointPairBelief:
    """A coordinate chart only: swapping endpoint names cannot change control.

    Lexicographically choose the sign of gauge-odd quantities. Covariance and
    visibility break ties at zero projected span, without imposing an opening
    floor or declaring either screen side to be physical F.
    """
    identity=belief.identity_probability
    odd=torch.cat((belief.state_mean[...,2:],
                   belief.state_covariance[...,:2,2:].flatten(-2),
                   belief.visible_probability[...,1:2]-belief.visible_probability[...,:1],
                   2*identity[...,None]-1),-1)
    nonzero=odd!=0
    index=nonzero.to(torch.int64).argmax(-1,keepdim=True)
    first=odd.gather(-1,index).squeeze(-1)
    flip=nonzero.any(-1)&(first<0)
    sign=belief.state_mean.new_tensor([1,1,-1,-1])
    return replace(belief,
        state_mean=torch.where(flip[...,None],belief.state_mean*sign,belief.state_mean),
        state_covariance=torch.where(flip[...,None,None],belief.state_covariance*sign[:,None]*sign[None,:],belief.state_covariance),
        visible_probability=torch.where(flip[...,None],belief.visible_probability.flip(-1),belief.visible_probability),
        identity_probability=torch.where(flip,1-identity,identity))


def belief_statistics(belief: JointPairBelief) -> torch.Tensor:
    # Full 4x4 covariance retains translation/aperture cross-tip dependence.
    belief=canonical_belief(belief)
    identity = getattr(belief, "identity_probability", None)
    if identity is None:
        identity = torch.full_like(belief.log_weights, .5)
    return torch.cat((belief.state_mean.float()/224.,
                      belief.state_covariance.float().flatten(-2)/(224.**2),
                      belief.visible_probability.float(), belief.log_weights.float().exp()[...,None],
                      identity.float()[...,None]), -1)  # 24


def matched_prior_statistics(actual: JointPairBelief, prior: JointPairBelief, *, return_covariance=False):
    """Gauge-aligned prior responsibilities; never average opposite jaw labels.

    This is a diagnostic moment projection, not the observer's measurement
    update. Mixture-number permutations do not change the matched statistics.
    """
    actual,prior=canonical_belief(actual),canonical_belief(prior)
    observed = actual.state_mean.double()[..., :, None, :]
    means = prior.state_mean.double()[..., None, :, :]
    covariance = prior.state_covariance.double()[..., None, :, :, :]
    signs = means.new_tensor([1,1,-1,-1])
    swapped = means * signs
    swap = (observed-swapped).square().sum(-1) < (observed-means).square().sum(-1)
    means = torch.where(swap[...,None], swapped, means)
    covariance = torch.where(swap[...,None,None], covariance*signs[:,None]*signs[None,:], covariance)
    total = covariance + actual.state_covariance.double()[..., :, None, :, :]
    delta = observed-means
    solved = torch.linalg.solve(total, delta[...,None]).squeeze(-1)
    score = prior.log_weights.double()[...,None,:] - .5*((delta*solved).sum(-1)+torch.linalg.slogdet(total).logabsdet)
    weights = score.softmax(-1)
    mean = (weights[...,None]*means).sum(-2)
    centered = means-mean[...,None,:]
    cov = (weights[...,None,None]*(covariance+centered[..., :,None]*centered[...,None,:])).sum(-3)
    visibility = prior.visible_probability.double()[...,None,:,:].expand(*weights.shape,2)
    visibility = torch.where(swap[...,None], visibility.flip(-1), visibility)
    visibility = (weights[...,None]*visibility).sum(-2)
    identity = getattr(prior,"identity_probability",None)
    if identity is None:
        identity = torch.full_like(prior.log_weights,.5)
    identity = identity[...,None,:].expand_as(weights)
    identity = torch.where(swap,1-identity,identity)
    # A matched mixture weight has no categorical label identity; expected
    # mass is reported separately from covariance and spatial residuals.
    mass = (weights*prior.log_weights.exp()[...,None,:]).sum(-1)
    statistics = torch.cat((mean/224., cov.flatten(-2)/224.**2, visibility,
                            mass[...,None], (weights*identity).sum(-1)[...,None]),-1).float()
    # The neural summary is fp32; never reconstruct an SPD covariance from
    # that rounded summary for another inverse/logdet computation.
    return (statistics, cov) if return_covariance else statistics


class V201ControlEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.grounding = nn.Sequential(nn.LayerNorm(408),nn.Linear(408,192),nn.GELU(),nn.Linear(192,192))
        self.belief_error = nn.Sequential(nn.Linear(25,192),nn.GELU(),nn.Linear(192,192))
        self.q_error = nn.Sequential(nn.Linear(30,192),nn.GELU(),nn.Linear(192,192))
        self.availability = nn.Linear(2,192)

    def forward(self, observation, prior, measured_q_path, predicted_q_path, q_scale,
                *, innovation_mode="full", view_valid=None):
        if innovation_mode not in INNOVATION_MODES:
            raise ValueError("unknown pre-encoding intervention")
        stats = belief_statistics(observation.belief)
        batch = len(stats)
        grounding = self.grounding(torch.cat((observation.component_features.float(),stats),-1))
        matched = matched_cov = None
        if prior is not None:
            matched, matched_cov = matched_prior_statistics(observation.belief, prior, return_covariance=True)
        residual = torch.zeros_like(stats) if prior is None else stats-matched
        # Per-component Gaussian differential entropy is not mixture entropy.
        log_volume = .5*torch.linalg.slogdet(observation.belief.state_covariance.double()).logabsdet
        if prior is None:
            log_volume_delta = torch.zeros_like(log_volume)
        else:
            log_volume_delta = log_volume-.5*torch.linalg.slogdet(matched_cov).logabsdet
        path = stats.new_zeros(batch,30)
        if measured_q_path is not None:
            if predicted_q_path is None or measured_q_path.shape != (batch,6,6) or predicted_q_path.shape != (batch,6,6):
                raise ValueError("innovation requires paired actual/predicted q0..q5")
            path = ((measured_q_path[:,1:].float()-predicted_q_path[:,1:].float())/q_scale).flatten(1)
        valid = torch.ones(batch,2,device=stats.device,dtype=torch.bool) if view_valid is None else view_valid.bool()
        present = stats.new_full((batch,2,4,1),float(prior is not None))
        availability = torch.cat((present,valid[...,None,None].expand(-1,-1,4,1).to(stats)), -1)
        spatial = torch.cat((residual,log_volume_delta[...,None].to(stats)),-1)
        # Masks precede both learned branches; multiplying their outputs also
        # removes biases, while the SAME availability metadata remains visible.
        spatial_enabled = innovation_mode in ("full","belief_only") and prior is not None
        q_enabled = innovation_mode in ("full","q_only") and measured_q_path is not None
        spatial = torch.where(valid[...,None,None],spatial,torch.zeros_like(spatial))
        belief_token = self.belief_error(spatial*float(spatial_enabled))*float(spatial_enabled)
        q_token = self.q_error(path*float(q_enabled))*float(q_enabled)
        innovation = belief_token+q_token[:,None,None,:]+self.availability(availability)
        return grounding,innovation


@dataclass(frozen=True)
class V201ActualState:
    observation: ObserverOutput
    q_position: torch.Tensor
    history: ActualHistory
    grounding_tracks: torch.Tensor
    innovation_tracks: torch.Tensor
    endpoint_time_s: torch.Tensor
    view_valid: torch.Tensor
    prior: JointPairBelief | None = None
    measured_q_path: torch.Tensor | None = None
    predicted_q_path: torch.Tensor | None = None


@dataclass(frozen=True)
class V201Prediction:
    commands_position: torch.Tensor
    predicted_q_path: torch.Tensor
    q_uncertainty: torch.Tensor
    prior: JointPairBelief
    source_q_position: torch.Tensor
    source_time_s: torch.Tensor
    time_deltas: torch.Tensor
    arm_hidden: torch.Tensor | None = None  # factual training only; not actual memory


class V201Core(nn.Module):
    def __init__(self, config=V201Config()):
        super().__init__()
        self.config=config
        self.observer=V201PairObserver()
        self.lewm=V201LeWM()
        self.actor=V20Actor()
        self.control_encoding=V201ControlEncoding()
        self.register_buffer("q_loss_scale",torch.ones(6))
        # Training-only readout after all original modules: their seeded
        # initialization remains unchanged. Runtime never invokes this head.
        self.arm_sigreg_projector=ARMProjector()

    def freeze_for_stage(self,stage):
        self.requires_grad_(False).eval()
        if stage in ("appearance","fcpg","refit"):
            self.observer.requires_grad_(True).train()
        elif stage=="lewm":
            self.lewm.requires_grad_(True).train()
            self.arm_sigreg_projector.requires_grad_(True).train()
        elif stage in ("actor","dense","fish"):
            self.actor.requires_grad_(True).train()
            self.control_encoding.requires_grad_(True).train()
        else:
            raise ValueError("unknown training stage")

    def initial(self,rgb,dino_tokens,q_position,view_valid,endpoint_time_s,geometry=None):
        prior=geometry.make_prior(q_position) if geometry is not None else None
        observation=self.observer(rgb,dino_tokens,view_valid,prior=prior)
        return self.initial_from_measurement(observation,q_position,view_valid,endpoint_time_s)

    def initial_from_measurement(self,observation,q_position,view_valid,endpoint_time_s):
        grounding,innovation=self.control_encoding(observation,None,None,None,self.q_loss_scale,view_valid=view_valid)
        return V201ActualState(observation,q_position,empty_history(len(q_position),device=q_position.device,dtype=q_position.dtype),grounding,innovation,endpoint_time_s,view_valid)

    def encode_innovation(self,state,mode="full"):
        return self.control_encoding(state.observation,state.prior,state.measured_q_path,state.predicted_q_path,
                                     self.q_loss_scale,innovation_mode=mode,view_valid=state.view_valid)[1]

    def act(self,state,task_embedding,innovation_mode=None):
        mode=innovation_mode or ("neither" if self.config.actor_mode=="no_innovation" else "full")
        innovation=state.innovation_tracks if mode=="full" else self.encode_innovation(state,mode)
        return self.actor(state.observation.scene_tokens,state.grounding_tracks,innovation,state.q_position,state.history,task_embedding,view_valid=state.view_valid)

    def predict(self,state,commands_position,time_deltas,geometry=None,*,return_arm_hidden=False):
        hidden=None
        if return_arm_hidden:
            output,hidden=self.lewm.forward_with_hidden(state.q_position,commands_position,state.history,time_deltas)
        else:
            output=self.lewm(state.q_position,commands_position,state.history,time_deltas)
        with torch.no_grad():
            if geometry is None:
                # Unknown camera does not become fake pixel motion. Carry
                # uncertainty growth rather than frozen overconfident pixels.
                belief=state.observation.belief.detach()
                prior=replace(belief,state_covariance=belief.state_covariance+torch.eye(4,device=belief.state_mean.device)*4.)
            else:
                prior=geometry.propagate(state.observation.belief.detach(),state.q_position,
                                         output.predicted_q_path[:,-1].detach(),output.q_uncertainty[:,-1].detach())
        return V201Prediction(commands_position,output.predicted_q_path,output.q_uncertainty,prior,state.q_position,state.endpoint_time_s,time_deltas,hidden)

    def complete_from_measurement(self,previous,prediction,measurement,measured_q_path,view_valid,endpoint_time_s,actual_time_deltas=None):
        if not torch.allclose(prediction.source_q_position,previous.q_position,atol=1e-4,rtol=0) or not torch.allclose(prediction.source_time_s,previous.endpoint_time_s,atol=1e-6,rtol=0):
            raise ValueError("prediction belongs to a different actual endpoint")
        if measured_q_path.shape != (len(previous.q_position),6,6) or not torch.allclose(measured_q_path[:,0],previous.q_position,atol=1e-4,rtol=0):
            raise ValueError("actual path must be aligned q0..q5")
        observation=self.observer.assimilate(measurement,prediction.prior,view_valid)
        grounding,innovation=self.control_encoding(observation,prediction.prior,measured_q_path,prediction.predicted_q_path,self.q_loss_scale,view_valid=view_valid)
        elapsed=prediction.time_deltas if actual_time_deltas is None else actual_time_deltas
        if elapsed.shape != prediction.time_deltas.shape or not torch.isfinite(elapsed).all() or not (elapsed>0).all():
            raise ValueError("actual time intervals invalid")
        history=append_completed(previous.history,prediction.commands_position,measured_q_path,previous.observation.scene_tokens.mean(-2),elapsed)
        return V201ActualState(observation,measured_q_path[:,-1],history,grounding,innovation,endpoint_time_s,view_valid,prediction.prior,measured_q_path,prediction.predicted_q_path)


def detach_actual(state):
    def detached(value):
        if isinstance(value,torch.Tensor):
            return value.detach()
        if isinstance(value,JointPairBelief):
            return value.detach()
        if hasattr(value,"__dataclass_fields__"):
            return replace(value,**{field.name:detached(getattr(value,field.name)) for field in fields(value)})
        return value
    return detached(state)
