"""
Optimal policy for fixed-pattern channel switching (Theorem 1, Section VI),
following Wang, Liu, Gomes & Krishnamachari (arXiv:1802.06958).

Algorithm (Theorem 1 / Algorithm 1 & 2):
  - p >= 0.5 (channel likely to advance each step):
        after a GOOD observation -> advance to the next channel
        after a BAD  observation -> stay on the current channel
  - p < 0.5 (channel likely to stay each step):
        after a GOOD observation -> stay on the current channel
        after a BAD  observation -> advance to the next channel
"""

class OptimalPolicy:
    def __init__(self, n_channels: int, p: float):
        self.n_channels = n_channels
        self.p = p
        self.advance_on_good = (p >= 0.5)
        self.current_channel = None

    def reset(self, initial_good_channel: int):
        """
        Per Theorem 1's assumptions: the initial activated channel is
        known a-priori.
        """
        self.current_channel = initial_good_channel

    def select_action(self) -> int:
        return self.current_channel

    def update(self, reward: float):
        """Call after each env.step() with the reward received."""
        was_good = reward > 0
        if was_good == self.advance_on_good:
            self.current_channel = (self.current_channel + 1) % self.n_channels

if __name__ == "__main__":
    import numpy as np
    from envs.channel_env import RoundRobinChannelEnv

    N = 16
    N_STEPS = 5000

    # Sweep matches the paper's Fig. 4: p in {0.75, 0.80, 0.85, 0.90, 0.95}
    for p in [0.75, 0.80, 0.85, 0.90, 0.95]:
        vals = []
        for seed in range(5):
            env = RoundRobinChannelEnv(n_channels=N, p=p, seed=seed)
            env.reset()
            policy = OptimalPolicy(n_channels=N, p=p)
            policy.reset(env.good_channel)

            total_reward = 0.0
            for _ in range(N_STEPS):
                action = policy.select_action()
                _, reward, _, _ = env.step(action)
                policy.update(reward)
                total_reward += reward

            avg_reward = total_reward / N_STEPS
            expected = 2 * p - 1
            print(f"p={p:.2f}  avg reward={avg_reward:.3f}  (expected {expected:.3f})")
            vals.append(avg_reward)
        print(f"p={p:.2f}  avg={np.mean(vals):.3f} +- {np.std(vals):.3f}")