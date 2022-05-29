from environment import ContinuousMountainCarEnv
from machin.frame.algorithms import TD3
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import time

# configurations
env = ContinuousMountainCarEnv()
observe_dim = 2
action_dim = 1
action_range = 1
max_episodes = 600
max_steps = 500
solved_reward = 80
solved_repeat = 5


# model definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.tanh(self.fc3(a)) * self.action_range
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q


def train_td3():
    actor = Actor(observe_dim, action_dim, action_range)
    actor_t = Actor(observe_dim, action_dim, action_range)
    critic = Critic(observe_dim, action_dim)
    critic_t = Critic(observe_dim, action_dim)
    critic2 = Critic(observe_dim, action_dim)
    critic2_t = Critic(observe_dim, action_dim)

    td3 = TD3(
        actor,
        actor_t,
        critic,
        critic_t,
        critic2,
        critic2_t,
        t.optim.Adam,
        nn.MSELoss(reduction="sum"),
    )

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    all_rewards = []

    start_time = time.time()

    for episode in range(max_episodes):
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        frame_skip = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
        tmp_observations = []

        while not terminal and step <= max_steps:
            with t.no_grad():
                old_state = state

                if episode < 200:
                    action = 2*t.rand(size=(1,1)) - 1
                    if step == 0:
                        previous_action = action
                    # frame skipping -> use the same action for 30 consecutive steps
                    if frame_skip < 30:
                        action = previous_action
                        frame_skip += 1
                    else:
                        previous_action = action
                        frame_skip = 0

                else:
                    action = td3.act({"state": old_state})
                # print(action)
                state, reward, terminal, _ = env.step(action.numpy())
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

        td3.store_episode(tmp_observations)
        # update, update more if episode is longer (slow solution), else less
        if episode >= 150:
            for _ in range(step):
                td3.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        all_rewards.append(total_reward)

        # episode >= 200 means not solved by random policy
        if episode >= 200 and smoothed_total_reward > solved_reward:
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
