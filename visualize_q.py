import gym, os
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from test_her import ENTRY_POINT
from stable_baselines.her import HER


env_name = 'FetchPushWallObstacle-v1'
kwargs = dict(random_box=False, heavy_obstacle=True)
load_path = 'logs/FetchPushWallObstacle-v1_heavy_random/her_sac2/' # U shape valley
load_path = 'logs/FetchPushWallObstacle-v1_heavy_determine/her_sac2/' # 2 peak regions
load_path = 'logs/FetchPushWallObstacle-v1_heavy_purerandom/her_sac/' # V chair, the minimum point is *not* expected
# load_path = 'logs/FetchPushWallObstacle-v1_heavy_mix0.7/her_sac' # ?
# load_path = 'logs/FetchPushWallObstacle-v1_heavy_random_hack/her_sac'
# kwargs = dict(random_box=False, heavy_obstacle=False)
# load_path = 'logs/FetchPushWallObstacle-v1_light_random/her_sac2/' # flat in ypos
# load_path = 'logs/FetchPushWallObstacle-v1_light_determine/her_sac2/' # W shape, rough
# load_path = 'logs/FetchPushWallObstacle-v1_light_purerandom/her_sac/' # flat
# load_path = 'logs/FetchPushWallObstacle-v1_light_random_hack/her_sac/' # flat
gym.register(env_name, entry_point=ENTRY_POINT[env_name], max_episode_steps=50, kwargs=kwargs)
env = gym.make(env_name)
model = HER.load(os.path.join(load_path, 'model_40.zip'), env=env)
env = model.env

if os.path.exists('batch_obs.npy'):
    batch_obs = np.load('batch_obs.npy')
    batch_actions = np.load('batch_actions.npy')
    imgs = np.load('batch_imgs.npy')
    _batch_obs = []
    _batch_actions = []
    _imgs = []
    # for j in range(13, 16, 1):
    #     for i in range(25):
    #         obs = batch_obs[j].copy()
    #         obs[7] = 0.5 + 0.02 * i # obstacle ypos
    #         obs[13] = obs[7] - obs[1] # rel pos
    #         _batch_obs.append(obs)
    #         _batch_actions.append(batch_actions[j].copy())
    #         _imgs.append(imgs[j].copy())
    obstacle_xpos, obstacle_ypos = np.meshgrid(np.linspace(1.0, 1.6, 21), np.linspace(0.4, 1.1, 21))
    grid_shape = obstacle_xpos.shape
    _obstacle_xpos = obstacle_xpos.reshape(-1, 1)
    _obstacle_ypos = obstacle_ypos.reshape(-1, 1)
    batch_obs = np.tile(batch_obs[13], (_obstacle_xpos.shape[0], 1))
    # batch_obs = np.tile(batch_obs[8], (_obstacle_xpos.shape[0], 1))
    batch_obs[:, 6] = _obstacle_xpos[:, 0]
    batch_obs[:, 7] = _obstacle_ypos[:, 0]
    batch_obs[:, 12] = batch_obs[:, 6] - batch_obs[:, 0]
    batch_obs[:, 13] = batch_obs[:, 7] - batch_obs[:, 1]
    batch_obs[:, -2] = 0.75
    # batch_actions = np.tile(np.array([-1.0, 0.0, 0.0, 0.0]), (_obstacle_xpos.shape[0], 1))
    batch_actions = np.tile(batch_actions[13], (_obstacle_xpos.shape[0], 1))
    imgs = np.tile(imgs[13], (_obstacle_xpos.shape[0], 1, 1, 1))
    # imgs = np.tile(imgs[8], (_obstacle_xpos.shape[0], 1, 1, 1))
    # batch_obs = np.asarray(_batch_obs)
    # batch_actions = np.asarray(_batch_actions)
    # imgs = np.asarray(_imgs)
else:
    batch_obs = []
    batch_actions = []
    imgs = []
    obs = env.reset()
    while (obs[-3] - env.env.pos_wall[0]) * (obs[3] - env.env.pos_wall[0]) > 0:
        obs = env.reset()
    for t in range(50):
        action, _ = model.predict(obs)
        batch_obs.append(obs)
        batch_actions.append(action)
        obs, reward, done, _ = env.step(action)
        imgs.append(env.render(mode='rgb_array'))
    batch_obs = np.asarray(batch_obs)
    batch_actions = np.asarray(batch_actions)
    np.save('batch_obs.npy', batch_obs)
    np.save('batch_actions.npy', batch_actions)
    np.save('batch_imgs.npy', np.asarray(imgs))

sac_model = model.model
feed_dict = {
            sac_model.observations_ph: batch_obs,
            sac_model.actions_ph: batch_actions,
}
# TODO: make animation
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111, projection='3d')
for f_idx in range(41):
    model.model.load_parameters(os.path.join(load_path, 'model_' + str(f_idx) + '.zip'))
    values = sac_model.sess.run(sac_model.step_ops[6], feed_dict)
    grid_values = np.reshape(values, grid_shape)
    print(obstacle_ypos[np.argmin(grid_values[:, 10])][0])
    ax1.cla()
    surf = ax1.plot_surface(obstacle_xpos, obstacle_ypos, grid_values, cmap=cm.coolwarm)
    ax1.set_xlim(0.9, 1.7)
    ax1.set_ylim(0.3, 1.1)
    ax1.set_xlabel('obstacle xpos')
    ax1.set_ylabel('obstacle ypos')
    ax1.set_zlim(0, 8)
    ax1.set_zlabel('value fn')
    ax1.view_init(elev=60, azim=135)
    ax1.set_title('model_' + str(f_idx))
    # fig.colorbar(surf)
    plt.pause(0.1)
plt.show()
exit()

values = sac_model.sess.run(sac_model.step_ops[6], feed_dict) # value_fn
grid_values = np.reshape(values, grid_shape)
print(batch_obs.shape, batch_actions.shape, values.shape)
# TODO: print idx when value reaches minimum along x=1.3 and y=0.7

# print(grid_values)
# print(obstacle_xpos)
# print(obstacle_ypos)
box_pos = batch_obs[:, 40:43]
obstacle_pos = batch_obs[:, 6:9]
min_value = np.min(values)
max_value = np.max(values)

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(122, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')
ax2 = fig.add_subplot(121)
surf = ax1.plot_surface(obstacle_xpos, obstacle_ypos, grid_values, cmap=cm.coolwarm)
ax1.set_xlim(0.9, 1.7)
ax1.set_ylim(0.3, 1.1)
ax1.set_xlabel('obstacle xpos')
ax1.set_ylabel('obstacle ypos')
ax1.set_zlim(0, 8)
ax1.set_zlabel('value fn')
ax1.view_init(elev=60, azim=135)
fig.colorbar(surf)
ax2.imshow(imgs[0])
plt.show()
exit()
for f_idx in range(box_pos.shape[0]):
    ax2.cla()
    ax3.cla()
    ax1.scatter(box_pos[f_idx, 0], box_pos[f_idx, 1], values[f_idx, 0], c='k')
    # ax1.set_title('target ' + str(batch_obs[0, -3:]))
    ax1.set_xlim(0.9, 1.7)
    ax1.set_ylim(0.3, 1.1)
    ax1.set_xlabel('box xpos')
    ax1.set_ylabel('box ypos')
    ax1.set_zlim(min_value, max_value)
    # ax2.scatter(obstacle_pos[f_idx, 0], obstacle_pos[f_idx, 1], values[f_idx, 0], c='k')
    ax2.plot_surface(obstacle_xpos, obstacle_ypos, grid_values, cmap=cm.coolwarm)
    ax2.set_xlim(0.9, 1.7)
    ax2.set_ylim(0.3, 1.1)
    ax2.set_xlabel('obstacle xpos')
    ax2.set_ylabel('obstacle ypos')
    ax2.set_zlim(min_value, max_value)
    ax3.imshow(imgs[f_idx])
    plt.pause(0.1)
plt.show()