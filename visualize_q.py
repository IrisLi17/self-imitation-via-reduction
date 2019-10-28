import gym, os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from test_her import ENTRY_POINT
from stable_baselines.her import HER


env_name = 'FetchPushWallObstacle-v1'
kwargs = dict(random_box=False)
load_path = 'logs/FetchPushWallObstacle-v1_light_purerandom/her_sac/model_70.zip'
gym.register(env_name, entry_point=ENTRY_POINT[env_name], max_episode_steps=50, kwargs=kwargs)
env = gym.make(env_name)
model = HER.load(load_path, env=env)
env = model.env

if os.path.exists('batch_obs.npy'):
    batch_obs = np.load('batch_obs.npy')
    batch_actions = np.load('batch_actions.npy')
    _batch_obs = []
    _batch_actions = []
    for j in range(0, 16, 3):
        for i in range(25):
            obs = batch_obs[j].copy()
            obs[7] = 0.2 + 0.02 * i
            _batch_obs.append(obs)
            _batch_actions.append(batch_actions[j].copy())
    batch_obs = np.asarray(_batch_obs)
    batch_actions = np.asarray(_batch_actions)
else:
    batch_obs = []
    batch_actions = []
    obs = env.reset()
    while (obs[-3] - env.env.pos_wall[0]) * (obs[3] - env.env.pos_wall[0]) > 0:
        obs = env.reset()
    for t in range(50):
        action, _ = model.predict(obs)
        batch_obs.append(obs)
        batch_actions.append(action)
        obs, reward, done, _ = env.step(action)
    batch_obs = np.asarray(batch_obs)
    batch_actions = np.asarray(batch_actions)
    np.save('batch_obs.npy', batch_obs)
    np.save('batch_actions.npy', batch_actions)

sac_model = model.model
feed_dict = {
            sac_model.observations_ph: batch_obs,
            sac_model.actions_ph: batch_actions,
}
values = sac_model.sess.run(sac_model.step_ops[6], feed_dict) # value_fn
print(batch_obs.shape, batch_actions.shape, values.shape)
print(values)
box_pos = batch_obs[:, 40:43]
obstacle_pos = batch_obs[:, 6:9]
min_value = np.min(values)
max_value = np.max(values)

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
for f_idx in range(box_pos.shape[0]):
    ax1.scatter(box_pos[f_idx, 0], box_pos[f_idx, 1], values[f_idx, 0], c='k')
    ax1.set_title('target ' + str(batch_obs[0, -3:]))
    ax1.set_xlim(0.9, 1.7)
    ax1.set_ylim(0.3, 1.1)
    ax1.set_zlim(min_value, max_value)
    ax2.scatter(obstacle_pos[f_idx, 0], obstacle_pos[f_idx, 1], values[f_idx, 0], c='k')
    ax2.set_xlim(0.9, 1.7)
    ax2.set_ylim(0.3, 1.1)
    ax2.set_zlim(min_value, max_value)
    plt.pause(0.1)
plt.show()