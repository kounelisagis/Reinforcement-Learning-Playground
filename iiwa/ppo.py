from environment import IiwaEnv
from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import time
from torch.nn.functional import softplus
from torch.distributions import Normal


env = IiwaEnv(enable_graphics=False)
observe_dim = 14
action_dim = 7
action_range = t.Tensor([[1.4835298, 1.4835298, 1.7453293, 1.3089969, 2.268928, 2.3561945, 2.3561945]])
max_episodes = 300
max_steps = 400
solved_reward = 0
solved_repeat = 5


class Actor(t.nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.mu_head = nn.Linear(16, action_num)
        self.sigma_head = nn.Linear(16, action_num)

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))

        mu = self.mu_head(a)
        # check if mu contains only nans
        if t.isnan(mu).any():
            mu = t.zeros_like(mu, requires_grad=True)

        # check if sigma contains only nans
        sigma = softplus(self.sigma_head(a))
        if t.isnan(sigma).any():
            sigma = t.ones_like(sigma, requires_grad=True)

        dist = Normal(mu, sigma)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()

        # remapping the distribution
        act_log_prob = dist.log_prob(act)
        act_tanh = t.tanh(act)
        act = act_tanh * action_range

        # the distribution remapping process used in the original essay.
        act_log_prob -= t.log(action_range * (1 - act_tanh.pow(2)) + 1e-6)
        act_log_prob = act_log_prob.sum(1, keepdim=True)

        return act, act_log_prob, act_entropy


class Critic(t.nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc1 = t.nn.Linear(state_dim, 16)
        self.fc2 = t.nn.Linear(16, 16)
        self.fc3 = t.nn.Linear(16, 1)

    def forward(self, state):
        v = t.relu(self.fc1(state))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)

        return v


def train_ppo():
    actor = Actor(observe_dim, action_dim)
    critic = Critic(observe_dim)

    ppo = PPO(actor, critic, t.optim.Adam, nn.MSELoss(reduction="sum"))

    reward_fulfilled = 0
    smoothed_total_reward = 0.
    all_rewards = []

    start_time = time.time()

    for episode in range(max_episodes):
        total_reward = 0.
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
        tmp_observations = []

        while not terminal and step <= max_steps:
            with t.no_grad():
                if episode < 50:
                    action = ((2.0 * t.rand(1, 7) - 1.0) * action_range)
                else:
                    action = ppo.act({"state": state})[0]

                state, reward, terminal = env.step(action)
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward

                tmp_observations.append(
                    {
                        "state": {"state": state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )
            step += 1


        ppo.store_episode(tmp_observations)
        ppo.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        all_rewards.append(total_reward)

        # episode >= 50 means not solved by random policy
        if episode >= 50 and smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                break
        else:
            reward_fulfilled = 0

    if reward_fulfilled >= solved_repeat:
        logger.info("Environment solved!")

    end_time = time.time()
    logger.info(f"Time needed: {end_time - start_time:.2f} seconds")

    return all_rewards, end_time - start_time, reward_fulfilled >= solved_repeat
