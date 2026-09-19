"""
Convergence check for the DQN agent at a single p value.

Trains continuously for TOTAL_STEPS, pausing every EVAL_EVERY steps to run
a short greedy evaluation. Plots the resulting learning curve against
the Optimal policy's known average reward at this p, so you can see
where/if DQN actually plateaus, rather than guessing an episode count.

This script is a diagnostic, not part of the final replication numbers.
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from envs.channel_env import RoundRobinChannelEnv
from agents.dqn_agent import DQNAgent
from policies.optimal_policy import OptimalPolicy

# Configuration -------------------------------------------------------- #

N_CHANNELS = 16
P = 0.75
SEED = 0

TOTAL_STEPS = 200_000
EVAL_EVERY = 5_000
EVAL_TRAJECTORIES = 3
EVAL_STEPS_PER_TRAJECTORY = 2_000
EVAL_SEED_BASE = 9_000 

# Helpers ---------------------------------------------------------------- #

def optimal_reference_reward(p, n_channels=N_CHANNELS, steps=5000, seed=123):
    """Quick reference: Optimal policy's avg reward at this p, for the plot's
    horizontal line. Not the discounted-return metric used elsewhere -- just
    a plain running average, matching this script's DQN eval metric below."""
    env = RoundRobinChannelEnv(n_channels=n_channels, p=p, seed=seed)
    env.reset()
    policy = OptimalPolicy(n_channels=n_channels, p=p)
    policy.reset(env.good_channel)
    total = 0.0
    for _ in range(steps):
        action = policy.select_action()
        _, reward, _, _ = env.step(action)
        policy.update(reward)
        total += reward
    return total / steps


def greedy_eval(agent, p, n_channels=N_CHANNELS,
                 n_trajectories=EVAL_TRAJECTORIES,
                 steps_per_trajectory=EVAL_STEPS_PER_TRAJECTORY,
                 seed_base=EVAL_SEED_BASE):
    """Average plain reward over several short greedy trajectories on
    FRESH environments (not the training env), so this measures the
    learned policy, not a lucky continuation of the training trajectory."""
    saved_eps = agent.epsilon
    agent.epsilon = 0.0

    trajectory_avgs = []
    for k in range(n_trajectories):
        env = RoundRobinChannelEnv(n_channels=n_channels, p=p, seed=seed_base + k)
        state = env.reset()
        total = 0.0
        for _ in range(steps_per_trajectory):
            action = agent.select_action(state)
            state, reward, _, _ = env.step(action)
            total += reward
        trajectory_avgs.append(total / steps_per_trajectory)

    agent.epsilon = saved_eps
    return float(np.mean(trajectory_avgs)), float(np.std(trajectory_avgs))


# Main -------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=float, default=P, help="Switching probability")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS, help="Total training steps")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
 
    p = args.p
    total_steps = args.steps
    eval_every = args.eval_every
    seed = args.seed
    os.makedirs("results/figures", exist_ok=True)
    results_json = f"results/convergence_check_p{p:.2f}_{total_steps}_{seed}.json"
    figure_path = f"results/figures/convergence_check_p{p:.2f}_{total_steps}_{seed}.png"

    print(f"Computing Optimal policy reference reward at p={p}...")
    optimal_avg = optimal_reference_reward(p)
    print(f"Optimal policy avg reward: {optimal_avg:.3f}\n")

    env = RoundRobinChannelEnv(n_channels=N_CHANNELS, p=P, seed=SEED)
    state = env.reset()
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, seed=SEED)

    history = {"step": [], "eval_mean": [], "eval_std": []}

    start_time = time.time()
    for t in range(1, total_steps + 1):
        action = agent.select_action(state)
        next_state, reward, _, _ = env.step(action)
        agent.store(state, action, reward, next_state)
        agent.train_step()
        state = next_state
 
        if t % eval_every == 0:
            eval_mean, eval_std = greedy_eval(agent, p)
            elapsed = time.time() - start_time
            steps_per_sec = t / elapsed
            print(f"step {t:7,d}  eval avg reward: {eval_mean:+.3f} ± {eval_std:.3f}  "
                  f"(optimal: {optimal_avg:+.3f})  [{steps_per_sec:.0f} steps/s]")
            history["step"].append(t)
            history["eval_mean"].append(eval_mean)
            history["eval_std"].append(eval_std)
 
    with open(results_json, "w") as f:
        json.dump({"p": p, "optimal_avg": optimal_avg, "history": history}, f, indent=2)
    print(f"\nSaved results to {results_json}")

    torch.save(agent.q_network.state_dict(),
               f"results/dqn_p{p:.2f}_{total_steps}_{seed}.pt")
    print(f"Saved model weights to results/dqn_p{p:.2f}_{total_steps}_{seed}.pt")

    # Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.array(history["step"])
    means = np.array(history["eval_mean"])
    stds = np.array(history["eval_std"])
 
    ax.plot(steps, means, marker="o", color="red", label="DQN (greedy eval)")
    ax.fill_between(steps, means - stds, means + stds, color="red", alpha=0.15)
    ax.axhline(optimal_avg, color="green", linestyle="--", label="Optimal policy")
 
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Average reward (greedy eval)")
    ax.set_title(f"DQN convergence check at p={p}")
    ax.legend()
    ax.grid(alpha=0.3)
 
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    print(f"Saved figure to {figure_path}")

if __name__ == "__main__":
    main()