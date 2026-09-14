"""
Whittle Index heuristic (Section IV-B), following Wang, Liu, Gomes &
Krishnamachari (arXiv:1802.06958).
"""
import numpy as np

def estimate_transition_matrix(env, channel: int, horizon: int):
    """
    Sense `channel` for `horizon` consecutive slots and MLE-fit its
    2-state transition matrix from the resulting good/bad sequence.
    Returns (p01, p11).
    """
    obs_seq = []
    for _ in range(horizon):
        _, reward, _, _ = env.step(channel)
        obs_seq.append(1 if reward > 0 else 0)

    n00 = n01 = n10 = n11 = 0
    for prev, nxt in zip(obs_seq[:-1], obs_seq[1:]):
        if prev == 0 and nxt == 0:
            n00 += 1
        elif prev == 0 and nxt == 1:
            n01 += 1
        elif prev == 1 and nxt == 0:
            n10 += 1
        else:
            n11 += 1

    # Fall back to 0.5 if a state was never visited during estimation
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.5
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.5
    return p01, p11


class WhittleIndexPolicy:
    def __init__(self, n_channels: int, seed: int = 0):
        self.n_channels = n_channels
        self.rng = np.random.RandomState(seed) 
        self.p01 = np.zeros(n_channels)
        self.p11 = np.zeros(n_channels)
        self.pi1 = np.zeros(n_channels)   # stationary P(good) per channel
        self.lam = np.zeros(n_channels)   # eigenvalue p11 - p01 per channel

        self.last_state = np.zeros(n_channels, dtype=int)  # last observed state per channel
        self.tau = np.zeros(n_channels, dtype=int)          # slots since last observed

    def fit(self, env, mle_horizon: int = 10_000):
        """Estimation phase: MLE-fit each channel's transition matrix in isolation."""
        for k in range(self.n_channels):
            p01, p11 = estimate_transition_matrix(env, k, mle_horizon)
            self.p01[k] = p01
            self.p11[k] = p11
            p10 = 1 - p11
            self.pi1[k] = p01 / (p01 + p10) if (p01 + p10) > 0 else 0.5
            self.lam[k] = p11 - p01

        self.last_state[:] = 0
        self.tau[:] = 10_000

    def _belief(self, channel: int) -> float:
        s = self.last_state[channel]
        tau = self.tau[channel]
        indicator = 1 if s == 1 else 0
        return self.pi1[channel] + (indicator - self.pi1[channel]) * (self.lam[channel] ** tau)

    def select_action(self) -> int:
        beliefs = [self._belief(k) for k in range(self.n_channels)]
        noise = self.rng.rand(self.n_channels) * 1e-6
        return int(np.argmax(beliefs + noise))

    def update(self, action: int, reward: float):
        """Call after each env.step() with the action taken and reward received."""
        observed_state = 1 if reward > 0 else 0
        self.last_state[action] = observed_state
        self.tau[action] = 0
        for k in range(self.n_channels):
            if k != action:
                self.tau[k] += 1


if __name__ == "__main__":
    from envs.channel_env import RoundRobinChannelEnv

    N = 16
    P = 0.9
    MLE_HORIZON = 10_000
    EVAL_STEPS = 5000

    env = RoundRobinChannelEnv(n_channels=N, p=P, seed=0)
    env.reset()

    policy = WhittleIndexPolicy(n_channels=N, seed=0)
    policy.fit(env, mle_horizon=MLE_HORIZON)

    print(f"Estimated p01 (bad->good), first 4 channels: {policy.p01[:4]}")
    print(f"Estimated p11 (good->good), first 4 channels: {policy.p11[:4]}")
    print(f"True env p (advance probability): {P}")

    env.reset()
    total_reward = 0.0
    for _ in range(EVAL_STEPS):
        action = policy.select_action()
        _, reward, _, _ = env.step(action)
        policy.update(action, reward)
        total_reward += reward

    print(f"Whittle-equivalent avg reward over {EVAL_STEPS} steps: {total_reward / EVAL_STEPS:.3f}")