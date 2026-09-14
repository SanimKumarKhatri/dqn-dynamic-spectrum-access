"""
Round-robin channel environment for dynamic spectrum access,
following Wang, Liu, Gomes & Krishnamachari (arXiv:1802.06958),
Section VII-B: single good channel, round-robin switching.

Dynamics
--------
Exactly one channel is good at any time. If the good channel at time t
is k, then at time t+1:
    with probability p       -> channel (k + 1) mod N becomes good
    with probability (1 - p) -> channel k remains good
This is the paper's fixed-pattern switching with single-channel subsets
activated in a sequential order.

Observation (partial)
---------------------
The agent does NOT observe the full state. It only learns whether the
channel it selected was good or bad. The state fed to the DQN is the
concatenation of the last M per-slot observation vectors, each of
length N:

    x_t = [o_{t-1}, o_{t-2}, ..., o_{t-M}]    (length M * N)

where o_tau is a length-N vector with
    +1 at the selected channel if it was good,
    -1 at the selected channel if it was bad,
     0 elsewhere.
This is exactly the encoding used in the paper (Section VII-A).

Reward
------
+1 if the selected channel was good, -1 otherwise. Matches Section III.
"""

import numpy as np


class RoundRobinChannelEnv:
    """
    n_channels : int
        Number of channels N. Paper uses N = 16.
    p : float in [0, 1]
        Probability that the good channel advances by one slot each
        time step. Paper's Fig. 4 sweeps p in {0.75, 0.80, 0.85, 0.90, 0.95}.
    history_len : int or None
        Number of past slots M used to build the state. Paper uses
        M = N. If None, defaults to n_channels.
    seed : int or None
        Seed for the internal RNG (reproducibility).
    """

    def __init__(self, n_channels = 16, p = 0.9, history_len=None, seed=None):
        self.n_channels = n_channels
        self.p = p
        self.history_len = history_len if history_len is not None else n_channels
        self.rng = np.random.RandomState(seed)

        self.good_channel = 0
        self.history = []

    def reset(self):
        self.good_channel = int(self.rng.randint(self.n_channels))
        self.history = [np.zeros(self.n_channels, dtype=np.float32) for _ in range(self.history_len)]
        return self._get_state()

    def _get_state(self):
        """
        Concatenate observation vectors, most recent first
        """
        parts = [self.history[-(i + 1)] for i in range(self.history_len)]
        return np.concatenate(parts).astype(np.float32)

    def step(self, action: int):
        assert 0 <= action < self.n_channels, f"invalid action{action}"

        observation = 1 if action == self.good_channel else 0
        reward = 1.0 if observation == 1 else -1.0

        # Build per-slot observation vector: +1 (good), -1 (bad), 0 elsewhere
        observation_vec = np.zeros(self.n_channels, dtype=np.float32)
        observation_vec[action] = 1.0 if observation == 1 else -1.0

        # Update history
        self.history.pop(0)
        self.history.append((observation_vec))

        # Advance the good channel stochastically: with probability p, move to k+1
        prev_good_channel = self.good_channel
        if self.rng.rand() < self.p:
            self.good_channel = (self.good_channel + 1) % self.n_channels
    
        next_state = self._get_state()
        done = False
        info = {
            "good_channel_at_action": prev_good_channel,
            "good_channel": self.good_channel,}

        return next_state, reward, done, info

    @property
    def state_dim(self):
        return self.history_len * self.n_channels

    @property
    def action_dim(self):
        return self.n_channels


if __name__ == "__main__":
    # Quick sanity check
    N = 16
    P = 0.9

    env = RoundRobinChannelEnv(n_channels=N, p=P, seed=0)
    state = env.reset()
    print(f"state_dim={env.state_dim}, actual state shape={state.shape}, action_dim={env.action_dim}, p={P}")

    # Oracle test
    env.reset()
    total_reward = 0.0
    good = env.good_channel
    for _ in range(1000):
        _, reward, _, info = env.step(good)
        total_reward += reward
        good = info["good_channel"]
    print(f"Oracle avg reward (should be ~1.0): {total_reward / 1000:.3f}")

    # Random agent test
    env.reset()
    total_reward = 0.0
    for _ in range(1000):
        action = env.rng.randint(N)
        _, reward, _, _ = env.step(action)
        total_reward += reward
    expected = 2/N - 1
    print(f"Random avg reward (should be ~{expected:.3f}): {total_reward / 1000:.3f}")