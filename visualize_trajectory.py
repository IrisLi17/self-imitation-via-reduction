import os, imageio, shutil
import numpy as np
import matplotlib.pyplot as plt
from test_her import ENTRY_POINT
from stable_baselines.her import HER
from stable_baselines.her.utils import KEY_ORDER

'''
This file aims to see the value fn along a hand-tuned trajectory.
'''

# Pick a load_path.
load_path = 'logs/FetchPushWallObstacle-v1_heavy_purerandom/her_sac2'
# load_path = 'logs/FetchPushWallObstacle-v1_heavy_purerandom_offset0/her_sac'
# load_path = 'logs/FetchPushWallObstacle-v1_heavy_purerandom_hidev/her_sac'
# load_path = 'logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac/custom'

# Reset the environment to hard configuration.
if 'FetchPushWallObstacle-v4' in load_path:
    env_name = 'FetchPushWallObstacle-v4'
    from push_wall_obstacle import FetchPushWallObstacleEnv_v4
    ENTRY_POINT['FetchPushWallObstacle-v4'] = FetchPushWallObstacleEnv_v4
else:
    env_name = 'FetchPushWallObstacle-v1'
kwargs = dict(random_box=False, heavy_obstacle=True)
if not 'FetchPushWallObstacle-v4' in load_path:
    kwargs['hide_velocity'] = ('hidev' in load_path)
env = ENTRY_POINT[env_name](**kwargs)

model = HER.load(os.path.join(load_path, 'model_89.zip'), env=env)

'''
#####
if load_path == 'logs/FetchPushWallObstacle-v1_heavy_purerandom_hidev/her_sac':
    obs_array = np.load('free_hidev_obs.npy')[20, :]
elif load_path == 'logs/FetchPushWallObstacle-v1_heavy_purerandom/her_sac':
    obs_array = np.load('free_heavypurerandom_obs.npy')[35, :]
print(obs_array)
from visualize_obstacle import plot_value_obstaclepos
plot_value_obstaclepos(obs_array, model.model, load_path)
exit()
#####
'''

free = False
greedy = True
obs = env.reset()
env.goal[0] = 1.2
if not free:
    env.goal[1] = obs['achieved_goal'][1]
else:
    env.goal[1] = 0.75 # hidev89 can push the obstacle with box
    if load_path == 'logs/FetchPushWallObstacle-v1_heavy_purerandom_hidev/her_sac':
        env.goal[1] = 0.85
    elif load_path == 'logs/FetchPushWallObstacle-v1_heavy_purerandom/her_sac':
        env.goal[1] = 0.8
        env.goal[0] = 1.17
    elif load_path == 'logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac':
        env.goal[1] = 0.8
obs['desired_goal'] = env.goal
print(obs)
# Design action sequence. First, push the obstacle away; then, push the box to the goal
batch_obs = []
imgs = []
for i in range(50):
    if free:
        action, _ = model.predict(obs)
    elif not greedy:
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
            action = obs['achieved_goal'][:3] + np.array([0.0, 0, 0.1]) - obs['observation'][0:3]
        elif i <= 9:
            action = obs['achieved_goal'][:3] + np.array([0.15, 0, -0.05]) - obs['observation'][0:3]
        else:
            action = np.asarray([-1.0, 0.0, 0.0])
    print(i, action)
    if not free:
        action /= np.max(np.abs(action))
        action = np.concatenate((action, [0.]))
    batch_obs.append(np.concatenate([obs[key] for key in KEY_ORDER]))
    obs, _, _, _ = env.step(action)
    img = env.render(mode='rgb_array')
    imgs.append(img)
    plt.imshow(img)
    plt.pause(0.1)
batch_obs = np.asarray(batch_obs)
# np.save('free_universe_obs.npy', batch_obs)
# np.save('free_universe_img.npy', imgs)

sac_model = model.model
feed_dict = {
            sac_model.observations_ph: batch_obs,
}
# TODO: make animation
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
values = sac_model.sess.run(sac_model.step_ops[6], feed_dict)
# np.save('free_universe_value.npy', values)
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
plt.show()
gif_imgs = []
for i in range(values.shape[0]):
    img = plt.imread(os.path.join('temp', str(i) + '.png'))
    gif_imgs.append(img)
# imageio.mimsave('greedy_universe.gif', gif_imgs)
shutil.rmtree('temp')
