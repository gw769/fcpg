"""Independent-endpoint fusion: the matched baseline for the correlated pair.

The V20.1 draft's first proposition states what an estimator with independent
fingertip endpoints cannot represent: the cross-tip covariance that separates
a common-translation error from an aperture error. This module removes exactly
that cross-tip block, at the fusion boundary only, so the same trained
measurement network, the same actual-readback correction memory, the same ARM
forecast, history and Actor inputs can be compared with and without correlated
pair uncertainty.

This is an inference-time diagnostic intervention, not a retrained point
baseline and not a new model. No parameter, cache or checkpoint changes.
Endpoint means and per-tip marginal covariances are preserved exactly; only
the (F,M) cross block is zeroed.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

from .prod_v20_1_belief import JointPairBelief, pair_from_means
from .prod_v20_3_observer import V203PairObserver, fuse_v203_pair_beliefs


def independent_endpoint_belief(belief: JointPairBelief) -> JointPairBelief:
    """Same endpoint means, weights, visibility and per-tip marginals; zero cross-tip covariance."""
    covariance = belief.endpoint_joint_covariance.clone()
    covariance[..., :2, 2:] = 0
    covariance[..., 2:, :2] = 0
    return pair_from_means(belief.means_px, joint_covariance=covariance,
                           visible_probability=belief.visible_probability,
                           log_weights=belief.log_weights,
                           identity_probability=belief.identity_probability)


def assimilate_independent_endpoints(observer: V203PairObserver, measurement, prior, view_valid):
    """V20.3 visibility-aware update with block-diagonal prior, measurement and posterior.

    The posterior is re-pooled from the current spatial field under the
    block-diagonal belief, exactly as the production update re-pools under the
    correlated belief. Invalid views keep the (block-diagonal) prior.
    """
    measurement = replace(measurement, belief=independent_endpoint_belief(measurement.belief))
    if prior is None:
        return observer.repool(measurement, measurement.belief, view_valid)
    belief = fuse_v203_pair_beliefs(measurement.belief, independent_endpoint_belief(prior), view_valid,
                                    prior_floor=observer.prior_floor)
    return observer.repool(measurement, independent_endpoint_belief(belief), view_valid)


@contextmanager
def independent_endpoint_fusion(core):
    """Temporarily route ``core.observer.assimilate`` through the block-diagonal update.

    Everything else (actual-q localization memory, ARM forecast, innovation,
    Actor) runs the unchanged production path, so any difference is
    attributable to cross-tip covariance alone. The original method is
    restored even when the wrapped evaluation raises.
    """
    observer = core.observer
    if getattr(observer, "raw_feature_cache", False):
        raise NotImplementedError("the raw-field conditioned observer has no independent-endpoint fusion baseline")
    if not isinstance(observer, V203PairObserver):
        raise TypeError("independent-endpoint fusion requires the V20.3 observer")
    if type(observer).assimilate is not V203PairObserver.assimilate:
        raise NotImplementedError("observer subclass overrides assimilation; the matched baseline is defined for V20.3 only")
    if "assimilate" in observer.__dict__:
        raise RuntimeError("observer assimilation is already overridden")

    def assimilate(measurement, prior, view_valid):
        return assimilate_independent_endpoints(observer, measurement, prior, view_valid)

    observer.__dict__["assimilate"] = assimilate
    try:
        yield core
    finally:
        del observer.__dict__["assimilate"]


__all__ = ["independent_endpoint_belief", "assimilate_independent_endpoints", "independent_endpoint_fusion"]
