import os, imageio, shutil
import numpy as np
import matplotlib.pyplot as plt
from test_her import ENTRY_POINT
from stable_baselines.her import HER
from stable_baselines.her.utils import KEY_ORDER

'''
This file aims to see the value fn along a hand-tuned trajectory.
'''
# Reset the environment to hard configuration.
env_name = 'FetchPushWallObstacle-v1'
kwargs = dict(random_box=False, heavy_obstacle=True)
env = ENTRY_POINT[env_name](**kwargs)

# Pick a load_path.
load_path = 'logs/FetchPushWallObstacle-v1_heavy_purerandom/her_sac'
load_path = 'logs/FetchPushWallObstacle-v1_heavy_purerandom_offset0/her_sac'
model = HER.load(os.path.join(load_path, 'model_42.zip'), env=env)

obs = env.reset()
env.goal[0] = 1.2
env.goal[1] = obs['achieved_goal'][1]
obs['desired_goal'] = env.goal
# Design action sequence. First, push the obstacle away; then, push the box to the goal
batch_obs = []
imgs = []
greedy = True
for i in range(50):
    if not greedy:
        if i <= 6:
            action = obs['observation'][6:9] - np.asarray([0, 0.2, 0]) - obs['observation'][0:3]
        elif i <= 20:
            action = np.asarray([0.0, 1.0, 0.0])
        elif obs['observation'][0] < obs['achieved_goal'][0] + 0.1 and i <= 27:
            action = obs['observation'][3:6] + np.asarray([0.1, 0.0, 0.1]) - obs['observation'][0:3]
        elif i <= 32:
            action = obs['observation'][3:6] + np.asarray([0.15, 0, -0.05]) - obs['observation'][0:3]
        elif obs['observation'][3] - obs['desired_goal'][0] > 0.05:
            action = np.asarray([-1.0, 0.0, 0.0])
        else:
            action = np.asarray([1., 0, 1.])
    else:
        if obs['observation'][0] < obs['achieved_goal'][0] and i <= 4:
            action = obs['achieved_goal'] + np.array([0.0, 0, 0.1]) - obs['observation'][0:3]
        elif i <= 9:
            action = obs['achieved_goal'] + np.array([0.15, 0, -0.05]) - obs['observation'][0:3]
        else:
            action = np.asarray([-1.0, 0.0, 0.0])
    print(i, action)
    action /= np.max(np.abs(action))
    action = np.concatenate((action, [0.]))
    batch_obs.append(np.concatenate([obs[key] for key in KEY_ORDER]))
    obs, _, _, _ = env.step(action)
    img = env.render(mode='rgb_array')
    imgs.append(img)
    # plt.imshow(img)
    # plt.pause(0.1)
batch_obs = np.asarray(batch_obs)

sac_model = model.model
feed_dict = {
            sac_model.observations_ph: batch_obs,
}
# TODO: make animation
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
values = sac_model.sess.run(sac_model.step_ops[6], feed_dict)
for i in range(values.shape[0]):
    print(i, values[i, :])
# exit()
os.makedirs('temp')
for i in range(values.shape[0]):
    ax.cla()
    ax2.cla()
    ax.plot(values[:i, 0])
    ax.set_xlim(0, values.shape[0] - 1)
    ax.set_ylim(min(values), max(values))
    ax.set_xlabel('steps')
    ax.set_ylabel('value fn')
    ax2.imshow(imgs[i])
    plt.savefig(os.path.join('temp', str(i) + '.png'))
    plt.pause(0.1)
# plt.show()
gif_imgs = []
for i in range(values.shape[0]):
    img = plt.imread(os.path.join('temp', str(i) + '.png'))
    gif_imgs.append(img)
imageio.mimsave('greedy_offset0.gif', gif_imgs)
shutil.rmtree('temp')
