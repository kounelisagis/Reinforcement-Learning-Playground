from td3 import train_td3
from ppo import train_ppo
import matplotlib.pyplot as plt
import numpy as np

all_rewards_list, time_list = [], []
counter = 0

for i in range(5):
    all_rewards, time, solved = train_td3()
    # all_rewards, time, solved = train_ppo()
    all_rewards_list.append(all_rewards)
    time_list.append(time)


def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

y, error = tolerant_mean(all_rewards_list)
plt.plot(np.arange(len(y))+1, y, 'b-', label='mean')
plt.fill_between(np.arange(len(y))+1, y - error, y + error, color='b', alpha=0.2)

plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('TD3')
plt.legend()
plt.show()


fig, ax = plt.subplots(1, 1)
ax.boxplot(time_list)
ax.set_ylabel('Time (seconds)')
ax.set_title('TD3')

plt.show()
