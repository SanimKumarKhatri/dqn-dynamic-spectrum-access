"""
Policy extraction diagnostic (Section VII-B residual-gap analysis).

Loads a trained DQN checkpoint and runs greedy at p=0.75, logging
(action, true good channel, reward) every step. Reports:
  (a) fraction of steps on the true good channel
  (b1) after a +1: how often the next action is (chosen+1) mod N   [Theorem 1: advance]
  (b2) after a -1: how often the next action equals the chosen one  [Theorem 1: stay]
  (c)  failure decomposition: failures where the action was the
       policy-correct one (unavoidable, env didn't advance) vs.
       policy-wrong (true DQN error)
"""
import numpy as np
import torch

from envs.channel_env import RoundRobinChannelEnv
from agents.dqn_agent import QNetwork

N = 16
P = 0.75
CHECKPOINT = "results/dqn_p0.75_2000000_1.pt"
EVAL_SEED = 777
STEPS = 2000
BURN_IN = 100

net = QNetwork(state_dim=N * N, action_dim=N)
net.load_state_dict(torch.load(CHECKPOINT))
net.eval()

env = RoundRobinChannelEnv(n_channels=N, p=P, seed=EVAL_SEED)
state = env.reset()

actions, goods, rewards = [], [], []
with torch.no_grad():
    for _ in range(STEPS):
        q = net(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        action = int(q.argmax(dim=1).item())
        state, reward, _, info = env.step(action)
        actions.append(action)
        goods.append(info["good_channel_at_action"])  # true good channel at action time
        rewards.append(reward)

actions = np.array(actions); goods = np.array(goods); rewards = np.array(rewards)
sl = slice(BURN_IN, STEPS)

on_good        = float(np.mean(actions[sl] == goods[sl]))
pos            = rewards[sl][:-1] == 1.0
neg            = rewards[sl][:-1] == -1.0
advance_after_pos = float(np.mean(actions[sl][1:][pos]  == (actions[sl][:-1][pos]  + 1) % N))
stay_after_neg    = float(np.mean(actions[sl][1:][neg]  ==  actions[sl][:-1][neg]))
fails          = rewards[sl][1:] == -1.0
expected_act   = np.where(rewards[sl][:-1] == 1.0,
                          (actions[sl][:-1] + 1) % N,
                          actions[sl][:-1])
unavoidable    = float(np.mean(actions[sl][1:][fails] == expected_act[fails]))
policy_error   = 1.0 - unavoidable

print(f"(a) steps on true good channel : {on_good:.3f}   (optimal: {P:.2f})")
print(f"(b1) advance after +1          : {advance_after_pos:.3f}   (optimal: 1.000)")
print(f"(b2) stay after -1             : {stay_after_neg:.3f}   (optimal: 1.000)")
print(f"(c) failures that were policy-correct (unavoidable): {unavoidable:.3f}")
print(f"    failures that were policy errors                : {policy_error:.3f}  (optimal: 0.000)")
print(f"    overall failure rate       : {float(np.mean(rewards[sl] == -1.0)):.3f}   (optimal: {1-P:.2f})")