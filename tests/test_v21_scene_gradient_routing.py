from copy import deepcopy

import pytest
import torch
from torch import nn

from fcpg.v21_scene_gradient_routing import (
    COUPLED, ISOLATED, backward_scene_loss, saved_scene_gradient_routing)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scene_encoder = nn.Sequential(nn.Linear(4,8),nn.GELU(),nn.Linear(8,4))
        self.dynamics = nn.Linear(4,4)
        self.joint = nn.Linear(4,2)
        self.sigma = nn.Linear(4,2)
        self.visual = nn.Linear(4,4)

    def losses(self,x,y):
        current,future=self.scene_encoder(x),self.scene_encoder(y)
        future.retain_grad()
        hidden=self.dynamics(current)
        prediction=self.visual(hidden)
        joint=self.joint(hidden).square().mean()+.05*self.sigma(hidden).exp().mean()
        representation=(prediction-future).square().mean()+.02*(current.square().mean()+future.square().mean())
        return joint+representation,joint,representation,future


@pytest.mark.parametrize('mode',[COUPLED,ISOLATED])
@pytest.mark.parametrize('preexisting',[False,True])
def test_only_joint_encoder_contribution_is_removed_with_accumulation(mode,preexisting):
    torch.manual_seed(81)
    model=Toy();x,y=torch.randn(3,4),torch.randn(3,4)
    total,joint,representation,future=model.losses(x,y)
    named=list(model.named_parameters());params=[p for _,p in named]
    total_grads=torch.autograd.grad(total,params,retain_graph=True,allow_unused=True)
    representation_grads=torch.autograd.grad(representation,params,retain_graph=True,allow_unused=True)
    before=[torch.randn_like(p) if preexisting else torch.zeros_like(p) for p in params]
    for p,g in zip(params,before):
        p.grad=g.clone() if preexisting else None
    future.grad=None
    rng=torch.get_rng_state().clone()
    expected_future=torch.autograd.grad(total,future,retain_graph=True)[0]
    future.grad=None
    report=backward_scene_loss(total,joint,representation,model.scene_encoder,mode=mode,audit=True,
                               retained_targets=(future,))
    for (name,p),prior,all_g,visual_g in zip(named,before,total_grads,representation_grads):
        chosen=visual_g if mode==ISOLATED and name.startswith('scene_encoder.') else all_g
        expected=prior+(torch.zeros_like(p) if chosen is None else chosen)
        torch.testing.assert_close(p.grad,expected,rtol=2e-5,atol=2e-7)
    assert future.grad is not None and future.grad.abs().sum()>0
    torch.testing.assert_close(future.grad,expected_future)
    assert torch.equal(rng,torch.get_rng_state())
    assert not any(p._backward_hooks for p in params)
    assert report['joint_encoder_raw_gradient_l2']>0
    if mode==ISOLATED:
        assert report['blocked_joint_increment_max_abs']==0
        assert report['encoder_representation_gradient_verified']


def test_default_coupled_is_exact_legacy_backward_and_does_not_draw_rng():
    torch.manual_seed(51)
    a=Toy(); b=deepcopy(a);x,y=torch.randn(3,4),torch.randn(3,4)
    la=a.losses(x,y);lb=b.losses(x,y)
    rng=torch.get_rng_state().clone()
    la[0].backward()
    backward_scene_loss(*lb[:3],b.scene_encoder)
    for p,q in zip(a.parameters(),b.parameters()):
        assert torch.equal(p.grad,q.grad)
    assert torch.equal(rng,torch.get_rng_state())


def test_protocol_missing_means_coupled_but_cannot_hide_a_mismatch():
    assert saved_scene_gradient_routing({})==COUPLED
    assert saved_scene_gradient_routing({'scene_joint_gradient_routing':ISOLATED,
        'schedule':{'scene_joint_gradient_routing':ISOLATED}})==ISOLATED
    with pytest.raises(ValueError,match='disagree'):
        saved_scene_gradient_routing({'scene_joint_gradient_routing':ISOLATED,
                                     'schedule':{'scene_joint_gradient_routing':COUPLED}})
    with pytest.raises(ValueError,match='unsupported'):
        saved_scene_gradient_routing({'scene_joint_gradient_routing':'detach-everything'})


def test_temporary_parameter_hooks_are_removed_on_backward_error():
    model=Toy()
    total,joint,representation,_=model.losses(torch.randn(2,4),torch.randn(2,4))
    with pytest.raises(RuntimeError):
        backward_scene_loss(total,joint.detach(),representation,model.scene_encoder,mode=ISOLATED)
    assert not any(p._backward_hooks for p in model.parameters())


def test_routing_change_cannot_be_resumed_as_a_resource_only_change():
    from fcpg.v21_scene_lewm import SCENE_ARCHITECTURE,SCENE_SIGREG_SOURCE
    from scripts.train_v21_horizon40 import (TRAINING_MODE_POLICY,canonical_protocol,
        validate_resource_continuation,validate_training_semantics)
    old=dict(training_mode_policy=TRAINING_MODE_POLICY,sigreg_feature_source=SCENE_SIGREG_SOURCE,
             scene_lewm_architecture=SCENE_ARCHITECTURE,schedule=dict(scene_latent_lewm=True),device={})
    new=deepcopy(old)
    new['scene_joint_gradient_routing']=COUPLED
    new['schedule']['scene_joint_gradient_routing']=COUPLED
    assert canonical_protocol(old)==canonical_protocol(new)
    new['scene_joint_gradient_routing']=ISOLATED
    new['schedule']['scene_joint_gradient_routing']=ISOLATED
    with pytest.raises(ValueError,match='cannot change'):
        validate_resource_continuation(old,new)
    with pytest.raises(ValueError,match='routing changed'):
        validate_training_semantics(old,full_patch_lewm=False,scene_latent_lewm=True,
                                    scene_joint_gradient_routing=ISOLATED)


@pytest.mark.parametrize('precision',['fp32','bf16'])
def test_real_full_patch_native40_keeps_future_and_visual_dynamics_gradients(precision):
    from contextlib import nullcontext
    from fcpg.prod_v20_policy import empty_history
    from fcpg.v21_models import CommandTrace
    from fcpg.v21_scene_lewm import V21SceneLeWM,FP32_RESIDUAL_SCENE
    from fcpg.v21_command_memory import FULL_COMMANDS
    torch.set_num_threads(2);torch.manual_seed(55)
    model=V21SceneLeWM(normalization='layernorm',scene_prediction_mode=FP32_RESIDUAL_SCENE,
                      command_conditioning=FULL_COMMANDS).train()
    # A warm-started scene-context is not the all-zero new-branch initializer.
    with torch.no_grad():
        model.scene_context.weight.normal_(std=.01)
    q=torch.randn(2,6)
    trace=CommandTrace.policy(q[:,None].repeat(1,5,1),torch.full((2,5),.1))
    times=(torch.arange(40,dtype=torch.float64)[None]+1).repeat(2,1)*.1
    patches=torch.randn(2,2,3,256,384)
    future_features=torch.randn(2,40,2,256,384)
    with torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False) if precision=='bf16' else nullcontext():
        output=model(q,trace,empty_history(2),times,patches=patches)
        target=model.scene_encoder(future_features,torch.ones(2,40,2,dtype=torch.bool))
    output.observation_scene.retain_grad();target.retain_grad()
    joint=output.predicted_q_path.float().square().mean()
    representation=(output.predicted_scene.float()-target.float()).square().mean()+.02*target.float().square().mean()
    params=list(model.parameters()); encoder_ids={id(p) for p in model.scene_encoder.parameters()}
    gj=torch.autograd.grad(joint,params,retain_graph=True,allow_unused=True)
    gv=torch.autograd.grad(representation,params,retain_graph=True,allow_unused=True)
    expected_target=torch.autograd.grad(representation,target,retain_graph=True)[0]
    target.grad=None;output.observation_scene.grad=None
    rng=torch.get_rng_state().clone()
    report=backward_scene_loss(joint+representation,joint,representation,model.scene_encoder,
        mode=ISOLATED,audit=True,retained_targets=(target,output.observation_scene))
    assert report['joint_encoder_raw_gradient_l2']>0
    assert report['blocked_joint_increment_max_abs']==0
    for p,j,v in zip(params,gj,gv):
        if j is None and v is None:
            assert p.grad is None
            continue
        expected=torch.zeros_like(p) if v is None else v
        if id(p) not in encoder_ids and j is not None:
            expected=expected+j
        torch.testing.assert_close(p.grad,expected,rtol=2e-5,atol=2e-7)
    torch.testing.assert_close(target.grad,expected_target,rtol=0,atol=0)
    assert (target.grad.float().abs().sum((0,3))>0).all()  # all40 x both cameras
    assert output.observation_scene.grad.abs().sum()>0
    assert model.layers[0].cross_attn.in_proj_weight.grad.abs().sum()>0
    assert torch.equal(rng,torch.get_rng_state())
