from fetch_stack import FetchStackEnv
import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
env = FetchStackEnv()
obs = env.reset()
# while (np.argmax(obs['desired_goal'][3:]) != 1 or env.task_mode != 1):
#     obs = env.reset()
# for i in range(env.n_object):
#     print('object%d:' % i, obs['observation'][3 + 3*i: 3 + 3 * (i+1)])
# g_idx = np.argmax(obs['desired_goal'][3:])
# obs['observation'][3 + 3 * g_idx:3 + 3 * g_idx + 2] = obs['observation'][3:5]
# obs['observation'][3 + 3 * g_idx + 2] = obs['observation'][5] + env.size_object[2] * 3
# one_hot = np.zeros(env.n_object)
# one_hot[g_idx] = 1
# env.goal = np.concatenate([obs['observation'][3 + 3 * g_idx: 6 + 3 * g_idx], one_hot])
# r = env.compute_reward(obs['observation'], env.goal, None)
# print(r)
for i in range(100):
    ax.cla()
    ax.imshow(env.render(mode='rgb_array'))
    ax.set_title('g_idx=%d' % np.argmax(obs['desired_goal'][3:]))
    plt.pause(0.1)
    action = np.random.randn(4)
    obs, reward, done, info = env.step(action)

