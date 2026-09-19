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

EPISODE_LENGTH = 1000

BURN_IN = 200           # steps run unscored before measuring
SCORE_STEPS = 1000      # steps over which discounted return is computed
#STEPS_PER_P = {0.75: 2_000_000, 0.80: 1_200_000, 0.85: 1_000_000,
#               0.90: 700_000, 0.95: 500_000}

STEPS_PER_P = {0.75: 2_100_000, 0.80: 2_000_000, 0.85: 1_800_000,
               0.90: 1_500_000, 0.95: 600_000}

SEEDS = [0] #[0, 1, 2]

N_EVAL_TRAJECTORIES = 10    # for every policy, incl. DQN's greedy eval

MLE_HORIZON = 10_000        # paper Section VII-B

RESULTS_JSON = "results/exp1_round_robin_3.json"
FIGURE_PATH = "results/figures/exp1_round_robin_3.png"

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

    returns = []
    for k in range(N_EVAL_TRAJECTORIES):
        env = make_env(p, seed=seed_offset + k)
        state = env.reset()
        for _ in range(burn_in):
            action = agent.select_action(state)
            state, _, _, _ = env.step(action)

        rewards = []
        for _ in range(score_steps):
            action = agent.select_action(state)
            state, reward, _, _ = env.step(action)
            rewards.append(reward)
        returns.append(discounted_return(rewards, GAMMA))

    agent.epsilon = saved_eps
    return np.array(returns)


def eval_optimal(p, seed_offset, score_steps=SCORE_STEPS):
    """Theorem 1 policy with genie initialization (initial channel known)."""
    returns = []

    for k in range(N_EVAL_TRAJECTORIES):
        env = make_env(p, seed=seed_offset + k)
        env.reset()
        policy = OptimalPolicy(n_channels=N_CHANNELS, p=p)
        policy.reset(env.good_channel)   # genie: matches Theorem 1's assumption

        for _ in range(BURN_IN):
            action = policy.select_action()
            _, reward, _, _ = env.step(action)
            policy.update(reward)
        rewards = []
        for _ in range(score_steps):
            action = policy.select_action()
            _, reward, _, _ = env.step(action)
            policy.update(reward)
            rewards.append(reward)
        returns.append(discounted_return(rewards, GAMMA))
    return np.array(returns)

def eval_whittle(p, fit_seed, eval_seed, burn_in=BURN_IN,
                 score_steps=SCORE_STEPS):
    fit_env = make_env(p, seed=fit_seed)
    fit_env.reset()
    fitted = WhittleIndexPolicy(n_channels=N_CHANNELS, seed=fit_seed)
    fitted.fit(fit_env, mle_horizon=MLE_HORIZON)

    returns = []

    for k in range(N_EVAL_TRAJECTORIES):
        policy = WhittleIndexPolicy(n_channels=N_CHANNELS, seed=eval_seed + k)
        policy.p01 = fitted.p01.copy()
        policy.p11 = fitted.p11.copy()
        policy.pi1 = fitted.pi1.copy()
        policy.lam = fitted.lam.copy()
        policy.last_state[:] = 0
        policy.tau[:] = MLE_HORIZON

        env = make_env(p, seed=eval_seed + k)
        env.reset()
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
        returns.append(discounted_return(rewards, GAMMA))
    return np.array(returns)

def eval_random(p, seed_offset, burn_in=BURN_IN, score_steps=SCORE_STEPS):
    returns = []
    for k in range(N_EVAL_TRAJECTORIES):
        env = make_env(p, seed=seed_offset + k)
        env.reset()
        rng = np.random.RandomState(seed_offset + k)
        for _ in range(burn_in):
            action = int(rng.randint(N_CHANNELS))
            _, _, _, _ = env.step(action)

        rewards = []
        for _ in range(score_steps):
            action = int(rng.randint(N_CHANNELS))
            _, reward, _, _ = env.step(action)
            rewards.append(reward)
        returns.append(discounted_return(rewards, GAMMA))
    return np.array(returns)


# DQN training ---------------------------------------------------- #

def train_dqn(p, seed, episode_length=EPISODE_LENGTH,
              verbose=True):

    total_steps = STEPS_PER_P[p]
    n_episodes = total_steps // episode_length
    log_every = max(1, n_episodes // 20)

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

        if verbose and (ep + 1) % log_every == 0:
            recent = np.mean(ep_avg[-log_every:])
            pct = 100 * (ep + 1) / n_episodes
            print(f"  ep {ep+1:5d}/{n_episodes} ({pct:4.0f}%)  "
                  f"avg per-step reward (last {log_every}): {recent:+.3f}")
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
            total_steps = STEPS_PER_P[p]
            print(f"  seed {seed}: training DQN ({total_steps:,} steps)...")
            agent = train_dqn(p, seed)
            r = eval_dqn(agent, p, seed_offset=10_000 + 1000 * seed)
            print(f"      eval discounted return: {r.mean():+.3f} ± {r.std():.3f}")
            dqn_returns.append(r)
        results[p]["dqn"] = [float(np.mean(dqn_returns)),
                             float(np.std(dqn_returns))]

        # Optimal ---
        r = eval_optimal(p, seed_offset=20_000)
        results[p]["optimal"] = [float(r.mean()), float(r.std())]

        # Whittle Index heuristic ---
        r = eval_whittle(p, fit_seed=30_000, eval_seed=40_000)
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