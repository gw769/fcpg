from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.v21_episode_sampling import EpisodeBatchSampler
from scripts.train_v21_horizon40 import ordered_groups


def setup(mode='iid-v1'):
    args=SimpleNamespace(seed=21004,lewm_fish_updates=30,actor_fish_updates=30,
        fish_batch_size=32,canary=False,lewm_archive_only_diagnostic=False,archive_batch_sampling=mode)
    splits={'train':{'fish':np.arange(20,40),'archive':np.arange(10)}}
    episodes=np.array([0,1,1,1,2,2,2,2,2,2])
    return args,splits,episodes


def test_iid_mode_is_exact_legacy_sequence():
    args,splits,episodes=setup()
    rng=np.random.default_rng(args.seed+1)
    for h,groups in ordered_groups('lewm_fish',splits,args):
        for domain,rows in groups:
            assert np.array_equal(rows,rng.choice(splits['train'][domain],16,replace=True))


def test_episode_grouping_keeps_fish_and_context_identical():
    args,splits,episodes=setup()
    old=list(ordered_groups('lewm_fish',splits,args))
    args.archive_batch_sampling='episode-v1'
    new=list(ordered_groups('lewm_fish',splits,args,archive_episode_index=episodes))
    for (ha,a),(hb,b) in zip(old,new):
        assert ha==hb
        assert a[0][0]==b[0][0]=='fish'
        assert np.array_equal(a[0][1],b[0][1])
        assert len(np.unique(episodes[b[1][1]]))==1
        assert np.isin(b[1][1],splits['train']['archive']).all()


def test_episode_weighting_preserves_uniform_row_marginals():
    args,splits,episodes=setup()
    sampler=EpisodeBatchSampler(splits['train']['archive'],episodes)
    rng=np.random.default_rng(13)
    samples=np.concatenate([sampler.draw(rng,16) for _ in range(15000)])
    freq=np.bincount(samples,minlength=10)/len(samples)
    assert np.max(np.abs(freq-.1)) < .009


def test_grouping_does_not_guess_missing_episode_identity():
    args,splits,episodes=setup('episode-v1')
    with pytest.raises(ValueError): list(ordered_groups('lewm_fish',splits,args))
    args.lewm_archive_only_diagnostic=True
    with pytest.raises(ValueError): list(ordered_groups('lewm_fish',splits,args,archive_episode_index=episodes))


def test_sampler_rejects_duplicate_or_outside_train_rows():
    with pytest.raises(ValueError): EpisodeBatchSampler([1,1],[0,0])
    with pytest.raises(ValueError): EpisodeBatchSampler([2],[0,0])


def test_episode_pair_matches_the_fish_half_and_keeps_fish_paired():
    args,splits,episodes=setup()
    old=list(ordered_groups('lewm_fish',splits,args))
    args.archive_batch_sampling='episode-pair-v1'
    new=list(ordered_groups('lewm_fish',splits,args,archive_episode_index=episodes))
    assert len(old)==len(new)==30
    for (ha,a),(hb,b) in zip(old,new):
        assert ha==hb and a[0][0]==b[0][0]=='fish'
        assert np.array_equal(a[0][1],b[0][1])
        kind,rows=b[1]
        assert kind=='archive' and len(rows)==16
        assert np.isin(rows,splits['train']['archive']).all()
        first,second=episodes[rows[:8]],episodes[rows[8:]]
        assert len(np.unique(first))==len(np.unique(second))==1
        assert first[0]!=second[0]


def test_one_episode_request_is_the_legacy_draw():
    sampler=EpisodeBatchSampler(np.arange(10),np.array([0,1,1,1,2,2,2,2,2,2]))
    a,b=np.random.default_rng(5),np.random.default_rng(5)
    for _ in range(50):
        assert np.array_equal(sampler.draw(a,16),sampler.draw_episodes(b,16,1))


def test_episode_pair_keeps_episode_marginals_near_row_share():
    rng=np.random.default_rng(0)
    sizes=rng.integers(20,120,size=300)
    episodes=np.repeat(np.arange(300),sizes)
    sampler=EpisodeBatchSampler(np.arange(len(episodes)),episodes)
    batches=np.stack([sampler.draw_episodes(rng,16,2) for _ in range(40000)])
    picks=episodes[batches[:,[0,8]]].ravel()  # the two recordings chosen per batch
    observed=np.bincount(picks,minlength=300)
    expected=len(picks)*sizes/sizes.sum()
    # Unbiased picks give chi2 ~ 299 +/- 24.5; a size-blind sampler gives thousands.
    assert ((observed-expected)**2/expected).sum() < 299+5*24.5
    assert np.abs(observed/expected-1).mean() < .08


def test_episode_pair_rejects_impossible_batches():
    rng=np.random.default_rng(0)
    with pytest.raises(ValueError): EpisodeBatchSampler(np.arange(4),np.zeros(4,int)).draw_episodes(rng,16,2)
    with pytest.raises(ValueError): EpisodeBatchSampler(np.arange(10),np.arange(10)).draw_episodes(rng,15,2)
    with pytest.raises(ValueError): EpisodeBatchSampler(np.arange(10),np.arange(10)).draw_episodes(rng,16,0)
