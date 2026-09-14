"""
Reproduces Fig. 4 from Wang et al. (arXiv:1802.06958), Section VII-B.

Single good channel, round-robin switching. Compares:

  - DQN               (trained from scratch per p)
  - Optimal           (Theorem 1, genie with known initial channel)
  - Whittle Index     (per-channel MLE + independent-channel Whittle)
  - Random            (uniform channel selection)

Evaluation protocol
-------------------
Because gamma = 0.9, the discounted return sum gamma^t r_t is dominated
by the first ~30 steps (sum gamma^t over t=0..29 is 9.58 of the ~10
effective horizon). To avoid the discounted return being a measurement
of cold-start transient rather than steady-state policy quality, we run
each policy for BURN_IN steps unscored, then accumulate discounted
reward over SCORE_STEPS. This is the closest we can get to the paper's
steady-state Fig. 4 values (5 to 9 across p) without changing gamma.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt

from envs.channel_env import RoundRobinChannelEnv
from agents.dqn_agent import DQNAgent
from policies.optimal_policy import OptimalPolicy
from policies.whittle_index import WhittleIndexPolicy

# Configuration -------------------------------------------#

P_VALUES = [0.75, 0.80, 0.85, 0.90, 0.95]
N_CHANNELS = 16
GAMMA = 0.9

N_EPISODES = 100
EPISODE_LENGTH = 1000

BURN_IN = 200           # steps run unscored before measuring
SCORE_STEPS = 1000      # steps over which discounted return is computed

SEEDS = [0, 1, 2]

MLE_HORIZON = 10_000        # paper Section VII-B

RESULTS_JSON = "results/exp1_round_robin.json"
FIGURE_PATH = "results/figures/exp1_round_robin.png"

# Helpers -------------------------------------------#

def discounted_return(rewards, gamma):
    r = np.asarray(rewards, dtype=np.float64)
    discounts = gamma ** np.arange(len(r))
    return float(np.dot(discounts, r))


def make_env(p, seed):
    return RoundRobinChannelEnv(n_channels=N_CHANNELS, p=p, seed=seed)


# per-policy evaluation ------------------------------------------- #

def eval_dqn(agent, p, seed_offset, burn_in=BURN_IN, score_steps=SCORE_STEPS):
    """Continuous trajectory: burn-in unscored, then measure discounted return."""
    saved_eps = agent.epsilon
    agent.epsilon = 0.0

    env = make_env(p, seed=seed_offset)
    state = env.reset()
    for _ in range(burn_in):
        action = agent.select_action(state)
        state, _, _, _ = env.step(action)

    rewards = []
    for _ in range(score_steps):
        action = agent.select_action(state)
        state, reward, _, _ = env.step(action)
        rewards.append(reward)

    agent.epsilon = saved_eps
    return discounted_return(rewards, GAMMA)


def eval_optimal(p, seed_offset, score_steps=SCORE_STEPS):
    """Theorem 1 policy with genie initialization (initial channel known)."""
    env = make_env(p, seed=seed_offset)
    env.reset()
    policy = OptimalPolicy(n_channels=N_CHANNELS, p=p)
    policy.reset(env.good_channel)   # genie: matches Theorem 1's assumption

    rewards = []
    for _ in range(score_steps):
        action = policy.select_action()
        _, reward, _, _ = env.step(action)
        policy.update(reward)
        rewards.append(reward)
    return discounted_return(rewards, GAMMA)


def eval_whittle(p, fit_seed, eval_seed, burn_in=BURN_IN,
                 score_steps=SCORE_STEPS):
    fit_env = make_env(p, seed=fit_seed)
    fit_env.reset()
    policy = WhittleIndexPolicy(n_channels=N_CHANNELS, seed=fit_seed)
    policy.fit(fit_env, mle_horizon=MLE_HORIZON)

    env = make_env(p, seed=eval_seed)
    env.reset()
    policy.last_state[:] = 0
    policy.tau[:] = MLE_HORIZON

    for _ in range(burn_in):
        action = policy.select_action()
        _, _, _, _ = env.step(action)
        policy.update(action, 0.0)  # dummy update to advance tau
    # reset tracking after burn-in
    policy.last_state[:] = 0
    policy.tau[:] = MLE_HORIZON

    rewards = []
    for _ in range(score_steps):
        action = policy.select_action()
        _, reward, _, _ = env.step(action)
        policy.update(action, reward)
        rewards.append(reward)
    return discounted_return(rewards, GAMMA)


def eval_random(p, seed_offset, burn_in=BURN_IN, score_steps=SCORE_STEPS):
    env = make_env(p, seed=seed_offset)
    env.reset()
    rng = np.random.RandomState(seed_offset)
    for _ in range(burn_in):
        action = int(rng.randint(N_CHANNELS))
        _, _, _, _ = env.step(action)

    rewards = []
    for _ in range(score_steps):
        action = int(rng.randint(N_CHANNELS))
        _, reward, _, _ = env.step(action)
        rewards.append(reward)
    return discounted_return(rewards, GAMMA)


# DQN training ---------------------------------------------------- #

def train_dqn(p, seed, n_episodes=N_EPISODES, episode_length=EPISODE_LENGTH,
              verbose=True):
    env = make_env(p, seed=seed)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                     seed=seed)

    ep_avg = []
    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        for _ in range(episode_length):
            action = agent.select_action(state)
            next_state, reward, _, _ = env.step(action)
            agent.store(state, action, reward, next_state)
            agent.train_step()
            state = next_state
            ep_reward += reward
        ep_avg.append(ep_reward / episode_length)
        if verbose and (ep + 1) % 20 == 0:
            recent = np.mean(ep_avg[-20:])
            print(f"  ep {ep+1:4d}  avg per-step reward (last 20): {recent:+.3f}")
    return agent

# Main --------------------------------------------------------------- #

def main():
    os.makedirs("results/figures", exist_ok=True)
    results = {}

    for p in P_VALUES:
        print(f"\n=== p = {p:.2f} ===")
        results[p] = {}

        # DQN ---
        dqn_returns = []
        for seed in SEEDS:
            print(f"  seed {seed}: training DQN "
                  f"({N_EPISODES} episodes x {EPISODE_LENGTH} steps = "
                  f"{N_EPISODES * EPISODE_LENGTH:,} steps)...")
            agent = train_dqn(p, seed)
            r = eval_dqn(agent, p, seed_offset=10_000 + 1000 * seed)
            print(f"      eval discounted return: {r:+.3f}")
            dqn_returns.append(r)
        results[p]["dqn"] = [float(np.mean(dqn_returns)),
                             float(np.std(dqn_returns))]

        # Optimal ---
        r = eval_optimal(p, seed_offset=20_000)
        results[p]["optimal"] = [float(r), 0.0]

        # Whittle Index heuristic ---
        r = eval_whittle(p, fit_seed=30_000, eval_seed=40_000)
        results[p]["whittle"] = [float(r), 0.0]

        # Random ---
        r = eval_random(p, seed_offset=50_000)
        results[p]["random"] = [float(r), 0.0]

        # per-p summary ---
        print(f"  Summary at p = {p:.2f}:")
        for name, (m, _) in results[p].items():
            print(f"      {name:8s}: {m:+.3f}")

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_JSON}")

    # plot (Fig. 4 reproduction) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    p_arr = np.array(P_VALUES)

    def series(name):
        return np.array([results[p][name][0] for p in P_VALUES])

    ax.plot(p_arr, series("dqn"), marker="o", color="red", label="DQN")
    ax.plot(p_arr, series("optimal"), marker="s", color="green",
            label="Optimal policy (known dynamics)")
    ax.plot(p_arr, series("whittle"), marker="^", color="blue",
            label="Whittle index heuristic")
    ax.plot(p_arr, series("random"), marker="x", color="gray",
            label="Random")

    ax.set_xlabel("Probability that the following channel is good (p)")
    ax.set_ylabel("Average discounted reward")
    ax.set_title("Fig. 4 reproduction: single good channel, round robin")
    ax.set_xticks(P_VALUES)
    ax.set_xticklabels([f"{p:.2f}" for p in P_VALUES])
    ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=9,
        )
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"Saved figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()