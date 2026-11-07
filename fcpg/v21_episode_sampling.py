"""Archive batch correlation samplers; distinct episode pairs can bias row marginals.

`draw` preserves exactly 1/N per row. `draw_episodes(..., episodes=2)` does
so only approximately when every recording has a small share of the pool.
For two recordings with 90/10 rows, each supplies half a batch (50/50), not
90/10. These are different row marginals; no sampling behavior is changed here.
"""
import numpy as np

IID = 'iid-v1'
EPISODE = 'episode-v1'
# The fish half of every mixed batch is 16 rows from its two TRAIN recordings.
# This mode gives the Archive half the same composition: two recordings x 8.
EPISODE_PAIR = 'episode-pair-v1'
MODES = (IID, EPISODE, EPISODE_PAIR)
EPISODES_PER_BATCH = {EPISODE: 1, EPISODE_PAIR: 2}


def episode_pair_row_probabilities(row_counts, *, max_attempts=10000):
    """Per-draw probability of *one row* in each recording for episode-pair-v1.

    For p_i=n_i/N and an unlimited rejection budget, inclusion is
        p_i * (1 + sum_{j != i} p_j/(1-p_j)).
    With the sampler's finite budget M, multiply each first-recording term
    by A_i=1-p_i**(M-1), then condition on success=sum(p_i*A_i).
    It supplies half the draws, uniformly within its n_i rows; divide the
    conditional inclusion by 2*n_i. dot(row_counts, result) is 1.
    This is the normalized marginal of a uniformly chosen batch slot, not the
    probability a row appears at least once in a batch. The sampler's finite
    rejection-budget error case produces no batch; the returned probabilities
    explicitly condition on a batch being produced.
    """
    counts = np.asarray(row_counts)
    if (counts.ndim != 1 or len(counts) < 2 or counts.dtype.kind not in 'iu'
            or (counts <= 0).any()):
        raise ValueError('at least two positive integer recording row counts required')
    if type(max_attempts) is not int or max_attempts < 2:
        raise ValueError('at least two draw attempts required to produce a pair')
    counts = counts.astype(np.float64)
    p = counts / counts.sum()
    if (p >= 1).any() or (p <= 0).any():
        raise ValueError('recording proportions exceed float64 resolution')
    accepted = -np.expm1((max_attempts - 1) * np.log(p))
    success = np.dot(p, accepted)
    terms = p * accepted / (1. - p)
    inclusion = p * (accepted + terms.sum() - terms) / success
    return inclusion / (2. * counts)


class EpisodeBatchSampler:
    def __init__(self, train_rows, row_episode_index):
        self.rows = np.asarray(train_rows, np.int64)
        episodes = np.asarray(row_episode_index)
        if (self.rows.ndim != 1 or not len(self.rows) or len(np.unique(self.rows)) != len(self.rows)
                or episodes.ndim != 1 or (self.rows < 0).any() or (self.rows >= len(episodes)).any()):
            raise ValueError('unique eligible TRAIN rows and episode identities required')
        self.row_episodes = episodes[self.rows]
        order = np.argsort(self.row_episodes, kind='stable')
        boundaries = np.flatnonzero(np.diff(self.row_episodes[order]))+1
        self.by_episode = {int(self.row_episodes[ix[0]]): self.rows[ix]
                           for ix in np.split(order, boundaries)}

    def draw(self, rng, size):
        if type(size) is not int or size <= 0:
            raise ValueError('positive batch size required')
        # P(episode)=n_episode/N, P(row|episode)=1/n_episode => P(row)=1/N.
        ep = int(self.row_episodes[rng.integers(len(self.rows))])
        return rng.choice(self.by_episode[ep], size=size, replace=True)

    def draw_episodes(self, rng, size, episodes, *, max_attempts=10000):
        """`episodes` distinct recordings, size/episodes rows from each.

        One episode is exactly `draw` (same RNG use). For more, each recording
        is picked with probability proportional to its eligible rows and a
        repeat is redrawn. The row marginal is approximately 1/N only when
        each recording has a small pool share. With unequal large shares it
        is biased: 90/10 rows in two recordings supply 50/50 of the batch.
        `episode_pair_row_probabilities` computes the exact two-recording
        marginal without changing draws or consuming RNG.
        """
        if type(episodes) is not int or episodes < 1:
            raise ValueError('positive integer episode count required')
        if episodes == 1:
            return self.draw(rng, size)
        if type(size) is not int or size <= 0 or size % episodes:
            raise ValueError('batch size must split evenly across the episodes')
        if len(self.by_episode) < episodes:
            raise ValueError('not enough distinct TRAIN episodes for one batch')
        chosen = []
        for _ in range(max_attempts):
            ep = int(self.row_episodes[rng.integers(len(self.rows))])
            if ep not in chosen:
                chosen.append(ep)
                if len(chosen) == episodes:
                    break
        else:
            raise ValueError('could not draw distinct TRAIN episodes')
        per = size // episodes
        return np.concatenate([rng.choice(self.by_episode[ep], size=per, replace=True) for ep in chosen])
