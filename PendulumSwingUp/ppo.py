from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
from environment import PendulumEnv
from torch.nn.functional import softplus
from torch.distributions import Normal
import time

# configurations
env = PendulumEnv(enable_graphics=False)
observe_dim = 3
action_num = 1
action_range = 2.5
max_episodes = 1000
max_steps = 500
solved_reward = -250
solved_repeat = 3


# model definition
class Actor(nn.Module):
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
        sigma = softplus(self.sigma_head(a))

        dist = Normal(mu, sigma)
        act = action if action is not None else dist.sample()

        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act)
        act = t.tanh(act)*action_range

        return act, act_log_prob, act_entropy


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        v = t.relu(self.fc1(state))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)
        return v


def train_ppo():
    actor = Actor(observe_dim, action_num)
    critic = Critic(observe_dim)

    ppo = PPO(
        actor,
        critic,
        t.optim.Adam,
        nn.MSELoss(reduction="sum"),
        gradient_max = 100.,
        batch_size = 512,
        actor_learning_rate = 1e-3,
        critic_learning_rate = 1e-3,
    )

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    all_rewards = []

    start_time = time.time()

    for episode in range(max_episodes):
        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
        tmp_observations = []
        frame_skip = 0

        while not terminal and step <= max_steps:
            with t.no_grad():
                old_state = state

                if frame_skip == 0:
                    if episode < 50:
                        action = (2.0*t.rand(1, 1) - 1.0) * action_range
                    else:
                        action = ppo.act({"state": old_state})[0]
                    previous_action = action
                    frame_skip += 1
                elif frame_skip < 45:
                    action = previous_action
                    frame_skip += 1
                else:
                    action = previous_action
                    frame_skip = 0


                state, reward, terminal = env.step([[action.item()]])
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward

                tmp_observations.append(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )
            step += 1

        # update
        ppo.store_episode(tmp_observations)
        ppo.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f} terminal={terminal}")

        all_rewards.append(total_reward)

        if episode >= 75 and smoothed_total_reward > solved_reward:
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
