"""Loss-specific training routing; never changes a model's forward function.

Only the joint objective's contribution to scene_encoder parameters is blocked.
Current/future visual targets, SIGReg and F/G retain their original graph.
This is not stop-gradient on observations, a new loss, or a deployment policy.
"""
import torch

COUPLED = 'coupled-v1'
ISOLATED = 'isolate-scene-encoder-v1'
SCENE_GRADIENT_ROUTING = (COUPLED, ISOLATED)


def saved_scene_gradient_routing(protocol):
    schedule = protocol.get('schedule', {})
    mode = protocol.get('scene_joint_gradient_routing', schedule.get('scene_joint_gradient_routing', COUPLED))
    if mode not in SCENE_GRADIENT_ROUTING:
        raise ValueError('unsupported scene joint gradient routing')
    if schedule.get('scene_joint_gradient_routing', mode) != mode:
        raise ValueError('scene joint gradient routing fields disagree')
    return mode


def backward_scene_loss(total, joint, representation, encoder, *, mode=COUPLED, audit=False,
                        retained_targets=()):
    """Accumulate one domain's weighted losses, preserving prior .grad values.

Coupled default is exactly the legacy single backward. Isolation performs two
backwards on ONE forward/dropout graph. Temporary parameter hooks zero ONLY
the joint contribution, then are removed before representation.backward().
No encoder activation/future target is detached and no RNG is consumed here.
Global gradient clipping still runs later, once after both domains/retention;
its changed norm can legitimately affect the scaling of other parameters.
"""
    if mode not in SCENE_GRADIENT_ROUTING:
        raise ValueError('unsupported scene joint gradient routing')
    if mode == COUPLED and not audit:
        total.backward()
        return dict(mode=mode, joint_encoder_isolated=False)
    params = tuple(p for p in encoder.parameters() if p.requires_grad)
    if not params or any(loss.ndim or not torch.isfinite(loss) for loss in (total,joint,representation)):
        raise ValueError('routing requires finite scalar losses and a trainable scene encoder')
    report = dict(mode=mode, joint_encoder_isolated=mode==ISOLATED)
    if audit:
        # Parameter .grad is untouched by autograd.grad, but retain_grad()
        # targets can collect probe gradients. Restore those before the real
        # backward so a future-target check cannot pass on probe evidence.
        prior = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]
        retained_prior = [None if t.grad is None else t.grad.detach().clone() for t in retained_targets]
        try:
            raw_joint = torch.autograd.grad(joint,params,retain_graph=True,allow_unused=True)
            expected = torch.autograd.grad(representation if mode==ISOLATED else total,
                                           params,retain_graph=True,allow_unused=True)
        finally:
            for tensor, gradient in zip(retained_targets,retained_prior):
                tensor.grad = gradient
        report['joint_encoder_raw_gradient_l2'] = sum(
            float(g.detach().double().square().sum()) for g in raw_joint if g is not None)**.5
        del raw_joint
    if mode == COUPLED:
        total.backward()
    else:
        handles = []
        try:
            for parameter in params:
                handles.append(parameter.register_hook(lambda grad: torch.zeros_like(grad)))
            joint.backward(retain_graph=True)
        finally:
            for handle in handles:
                handle.remove()
        if audit:
            difference = max(float((p.grad-previous).abs().max()) if p.grad is not None
                             else float(previous.abs().max()) for p,previous in zip(params,prior))
            if difference != 0:
                raise ValueError('joint loss changed scene-encoder accumulated gradients')
            report['blocked_joint_increment_max_abs'] = difference
        representation.backward()
    if audit:
        maximum = 0.
        for p,previous,g in zip(params,prior,expected):
            wanted = previous if g is None else previous+g
            actual = torch.zeros_like(p) if p.grad is None else p.grad
            torch.testing.assert_close(actual,wanted,rtol=2e-5,atol=2e-7)
            maximum = max(maximum,float((actual-wanted).abs().max()))
        report.update(encoder_gradient_max_abs_difference=maximum,
                      encoder_representation_gradient_verified=mode==ISOLATED,
                      encoder_expected_gradient_verified=True)
    return report
