"""
Reproduces Fig. 4 from Wang et al. (arXiv:1802.06958), Section VII-B.

Compares average discounted reward vs. switching probability p in the
single-good-channel round-robin setting, for four policies:

  - DQN               (trained from scratch per p)
  - Optimal           (Theorem 1, genie with known initial channel)
  - Whittle Index     (per-channel MLE + independent-channel Whittle)
  - Random            (uniform channel selection)

Discounted return convention: sum_{t=0}^{T-1} gamma^t * r_t, with
gamma = 0.9 and T = EVAL_HORIZON. Averaged over EVAL_EPISODES eval
episodes. This matches the paper's Fig. 4 y-axis scale: with gamma = 0.9,
the optimal policy's expected return at per-step reward r_bar is
r_bar / (1 - gamma) = 10 * r_bar, so the plotted values run 5 -> 9
across p in {0.75, ..., 0.95}.
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

TRAIN_STEPS = 30_000        
LOG_EVERY = 10_000

EVAL_EPISODES = 20
EVAL_HORIZON = 1_000

SEEDS = [0]                

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

def eval_dqn(agent, p, seed_offset):
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    returns = []
    for ep in range(EVAL_EPISODES):
        env = make_env(p, seed=seed_offset + ep)
        state = env.reset()
        rewards = []
        for _ in range(EVAL_HORIZON):
            action = agent.select_action(state)
            state, reward, _, _ = env.step(action)
            rewards.append(reward)
        returns.append(discounted_return(rewards, GAMMA))
    agent.epsilon = saved_eps
    return np.array(returns)


def eval_optimal(p, seed_offset):
    returns = []
    for ep in range(EVAL_EPISODES):
        env = make_env(p, seed=seed_offset + ep)
        env.reset()
        policy = OptimalPolicy(n_channels=N_CHANNELS, p=p)
        policy.reset(env.good_channel)   # genie: knows the true initial channel
        rewards = []
        for _ in range(EVAL_HORIZON):
            action = policy.select_action()
            _, reward, _, _ = env.step(action)
            policy.update(reward)
            rewards.append(reward)
        returns.append(discounted_return(rewards, GAMMA))
    return np.array(returns)


def eval_whittle(p, fit_seed, eval_seed_offset):
    # Fit on a separate env, then eval on fresh envs.
    fit_env = make_env(p, seed=fit_seed)
    fit_env.reset()
    policy = WhittleIndexPolicy(n_channels=N_CHANNELS, seed=fit_seed)
    policy.fit(fit_env, mle_horizon=MLE_HORIZON)

    returns = []
    for ep in range(EVAL_EPISODES):
        env = make_env(p, seed=eval_seed_offset + ep)
        env.reset()
        # Reset tracking; keep the fitted model.
        policy.last_state[:] = 0
        policy.tau[:] = MLE_HORIZON

        rewards = []
        for _ in range(EVAL_HORIZON):
            action = policy.select_action()
            _, reward, _, _ = env.step(action)
            policy.update(action, reward)
            rewards.append(reward)
        returns.append(discounted_return(rewards, GAMMA))
    return np.array(returns)


def eval_random(p, seed_offset):
    returns = []
    for ep in range(EVAL_EPISODES):
        env = make_env(p, seed=seed_offset + ep)
        env.reset()
        rng = np.random.RandomState(seed_offset + ep)
        rewards = []
        for _ in range(EVAL_HORIZON):
            action = int(rng.randint(N_CHANNELS))
            _, reward, _, _ = env.step(action)
            rewards.append(reward)
        returns.append(discounted_return(rewards, GAMMA))
    return np.array(returns)


# DQN training ---------------------------------------------------- #

def train_dqn(p, seed, train_steps=TRAIN_STEPS, verbose=True):
    env = make_env(p, seed=seed)
    state = env.reset()
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, seed=seed)

    running = 0.0
    for t in range(1, train_steps + 1):
        action = agent.select_action(state)
        next_state, reward, _, _ = env.step(action)
        agent.store(state, action, reward, next_state)
        agent.train_step()
        state = next_state
        running += reward

        if verbose and t % LOG_EVERY == 0:
            print(f"      step {t:6d}  avg reward (last {LOG_EVERY}): {running / LOG_EVERY:+.3f}")
            running = 0.0
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
            print(f"  seed {seed}: training DQN ({TRAIN_STEPS} steps)...")
            agent = train_dqn(p, seed)
            r = eval_dqn(agent, p, seed_offset=10_000 + 1000 * seed)
            print(f"      eval discounted return: {r.mean():+.3f} ± {r.std():.3f}")
            dqn_returns.append(r)
        dqn_returns = np.concatenate(dqn_returns)
        results[p]["dqn"] = [float(dqn_returns.mean()), float(dqn_returns.std())]

        # Optimal ---
        r = eval_optimal(p, seed_offset=20_000)
        results[p]["optimal"] = [float(r.mean()), float(r.std())]

        # Whittle Index heuristic ---
        r = eval_whittle(p, fit_seed=30_000, eval_seed_offset=40_000)
        results[p]["whittle"] = [float(r.mean()), float(r.std())]

        # Random ---
        r = eval_random(p, seed_offset=50_000)
        results[p]["random"] = [float(r.mean()), float(r.std())]

        # per-p summary ---
        print(f"  Summary at p = {p:.2f}:")
        for name, (m, s) in results[p].items():
            print(f"      {name:8s}: {m:+.3f} ± {s:.3f}")

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_JSON}")

    # plot (Fig. 4 reproduction) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    p_arr = np.array(P_VALUES)

    def series(name):
        m = np.array([results[p][name][0] for p in P_VALUES])
        s = np.array([results[p][name][1] for p in P_VALUES])
        return m, s

    dqn_m, dqn_s = series("dqn")
    opt_m, opt_s = series("optimal")
    wh_m, wh_s = series("whittle")
    rand_m, rand_s = series("random")

    ax.errorbar(p_arr, dqn_m, yerr=dqn_s, marker="o", color="red",
                label="DQN", capsize=4)
    ax.errorbar(p_arr, opt_m, yerr=opt_s, marker="s", color="green",
                label="Optimal policy (known dynamics)", capsize=4)
    ax.errorbar(p_arr, wh_m, yerr=wh_s, marker="^", color="blue",
                label="Whittle index heuristic", capsize=4)
    ax.errorbar(p_arr, rand_m, yerr=rand_s, marker="x", color="gray",
                label="Random", capsize=4)

    ax.set_xlabel("Probability that the following channel is good (p)")
    ax.set_ylabel("Average discounted reward")
    ax.set_title("Fig. 4 reproduction: single good channel, round robin")
    ax.set_xticks(P_VALUES)
    ax.set_xticklabels([f"{p:.2f}" for p in P_VALUES])
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"Saved figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()