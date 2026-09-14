"""
DQN agent for dynamic multichannel access (Section V, architecture from
Section VII-A / Table I), following Wang, Liu, Gomes & Krishnamachari
(arXiv:1802.06958).

Architecture (Section VII-A):
  - Fully connected network, 2 hidden layers x 200 neurons, ReLU
  - Input: state = concatenation of past M=N observation vectors
  - Output: Q-value for each of the N channels

Hyperparameters (Table I):
  epsilon = 0.1, minibatch size = 32, optimizer = Adam,
  learning rate = 1e-4, experience replay size = 1,000,000, gamma = 0.9

Target network: the frozen target network theta^- from Mnih et al. (2013),
which the paper cites as [5] for its DQN. Synced from the online network
every `target_update_every` gradient steps.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 200,        # Section VII-A architecture
        lr: float = 1e-4,              # Table I
        gamma: float = 0.9,            # Table I
        epsilon: float = 0.1,          # Table I
        buffer_size: int = 1_000_000,  # Table I
        batch_size: int = 32,          # Table I
        target_update_every: int = 500,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.q_network = QNetwork(state_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Frozen target network theta^- (Mnih et al. 2013).
        self.target_network = QNetwork(state_dim, action_dim, hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.target_update_every = target_update_every
        self._train_steps = 0

        self.replay_buffer = ReplayBuffer(buffer_size, seed=seed)
        self.rng = np.random.RandomState(seed)

    def select_action(self, state: np.ndarray) -> int:
        """epsilon-greedy: random action w.p. epsilon, else argmax Q (Section VII-A)."""
        if self.rng.rand() < self.epsilon:
            return int(self.rng.randint(self.action_dim))
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def store(self, state, action, reward, next_state):
        self.replay_buffer.push(state, action, reward, next_state)

    def train_step(self):
        """One gradient step on a random minibatch from replay. Returns loss, or
        None if the buffer doesn't have enough samples yet."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states)
        actions_t = torch.from_numpy(actions)
        rewards_t = torch.from_numpy(rewards)
        next_states_t = torch.from_numpy(next_states)

        # y = r + gamma * max_a' Q(x', a'; theta^-), using the frozen
        # target network theta^- (Mnih et al. 2013).
        with torch.no_grad():
            next_q = self.target_network(next_states_t)
            max_next_q = next_q.max(dim=1)[0]
            targets = rewards_t + self.gamma * max_next_q

        q_values = self.q_network(states_t)
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        loss = nn.functional.mse_loss(q_selected, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Sync target network periodically.
        self._train_steps += 1
        if self._train_steps % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()


if __name__ == "__main__":
    from envs.channel_env import RoundRobinChannelEnv

    N = 16
    P = 0.9
    TRAIN_STEPS = 20_000
    EVAL_STEPS = 5_000
    LOG_EVERY = 2_000

    env = RoundRobinChannelEnv(n_channels=N, p=P, seed=0)
    state = env.reset()

    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, seed=0)

    running_reward = 0.0
    for t in range(1, TRAIN_STEPS + 1):
        action = agent.select_action(state)
        next_state, reward, _, _ = env.step(action)
        agent.store(state, action, reward, next_state)
        agent.train_step()
        state = next_state
        running_reward += reward

        if t % LOG_EVERY == 0:
            print(f"step {t:6d}  avg reward (last {LOG_EVERY}): "
                  f"{running_reward / LOG_EVERY:+.3f}")
            running_reward = 0.0

    # Greedy evaluation.
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    eval_reward = 0.0
    for _ in range(EVAL_STEPS):
        action = agent.select_action(state)
        state, reward, _, _ = env.step(action)
        eval_reward += reward
    agent.epsilon = saved_eps

    print(f"\nEval avg reward over {EVAL_STEPS} steps (greedy): "
          f"{eval_reward / EVAL_STEPS:+.3f}")
    print("Compare to optimal policy at p=0.9: +0.791 per step "
          "(see policies/optimal_policy.py)")