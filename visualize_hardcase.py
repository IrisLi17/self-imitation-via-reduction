import os
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from test_her import ENTRY_POINT
from stable_baselines.her import HER
from stable_baselines.her.utils import KEY_ORDER

'''
This file aims to see the relation between the position of end effector and value fn in hard case.
'''
# Reset the environment to hard configuration.
env_name = 'FetchPushWallObstacle-v1'
kwargs = dict(random_box=False, heavy_obstacle=True)
env = ENTRY_POINT[env_name](**kwargs)

# Pick a load_path.
load_path = 'logs/FetchPushWallObstacle-v1_heavy_purerandom/her_sac'
model = HER.load(os.path.join(load_path, 'model_40.zip'), env=env)

# Reset end effector.
def reset_end_effector(env, xy):
    initial_mocap_pos = env.sim.data.get_mocap_pos('robot0:mocap')
    z = initial_mocap_pos[2]
    initial_mocap_pos[2] = 1.0
    env.sim.data.set_mocap_pos('robot0:mocap', initial_mocap_pos)
    for _ in range(10):
        env.sim.step()
    mocap_pos = np.concatenate([xy, [z]])
    env.sim.data.set_mocap_pos('robot0:mocap', mocap_pos)
    for _ in range(10):
        env.sim.step()
    env._step_callback()

pos_x, pos_y = np.meshgrid(np.linspace(1.05, 1.55, 10), np.linspace(0.55, 0.95, 10))
grid_shape = pos_x.shape
_pos_x = np.reshape(pos_x, (-1, 1))
_pos_y = np.reshape(pos_y, (-1, 1))
pos_xy = np.concatenate((_pos_x, _pos_y), axis=-1)
batch_obs = []
real_pos_x = []
real_pos_y = []
for i in range(pos_xy.shape[0]):
    env.reset()
    # TODO: put box
    object_qpos = env.sim.data.get_joint_qpos('object0:joint')
    stick_qpos = env.sim.data.get_joint_qpos('object1:joint')
    assert object_qpos.shape == (7,)
    object_qpos[:2] = np.asarray([1.47, 0.82])
    stick_qpos[:2] = np.asarray([1.3, 0.75])
    env.sim.data.set_joint_qpos('object0:joint', object_qpos)
    env.sim.data.set_joint_qpos('object1:joint', stick_qpos)
    env.sim.forward()
    reset_end_effector(env, pos_xy[i])
    obs_dict = env._get_obs()
    obs_dict['desired_goal'][:2] = np.asarray([1.2, 0.75])
    # print('gripper', obs_dict['observation'][0:2], 'desired', pos_xy[i], 'difference', np.linalg.norm(obs_dict['observation'][0:2] - pos_xy[i]))
    batch_obs.append(np.concatenate([obs_dict[key] for key in KEY_ORDER]))
    real_pos_x.append(obs_dict['observation'][0])
    real_pos_y.append(obs_dict['observation'][1])
batch_obs = np.asarray(batch_obs)
real_pos_x = np.reshape(np.asarray(real_pos_x), grid_shape)
real_pos_y = np.reshape(np.asarray(real_pos_y), grid_shape)
print(np.sum(np.abs(batch_obs[:, 7] - 0.75)))
# HACK
# batch_obs[:, 7] = 0.55
# batch_obs[:, 6] = 1.4
# batch_obs[:, 13] = batch_obs[:, 7] - batch_obs[:, 1]
# batch_obs[:, 12] = batch_obs[:, 6] - batch_obs[:, 0]

sac_model = model.model
feed_dict = {
            sac_model.observations_ph: batch_obs,
}
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111, projection='3d')
for f_idx in range(90):
    model.model.load_parameters(os.path.join(load_path, 'model_' + str(f_idx) + '.zip'))
    values = sac_model.sess.run(sac_model.step_ops[6], feed_dict)
    grid_values = np.reshape(values, grid_shape)
    print(np.max(values))
    ax1.cla()
    surf = ax1.plot_surface(real_pos_x, real_pos_y, grid_values, cmap=cm.coolwarm)
    ax1.set_xlim(0.9, 1.7)
    ax1.set_ylim(0.4, 1.1)
    ax1.set_xlabel('gripper xpos')
    ax1.set_ylabel('gripper ypos')
    ax1.set_zlim(-4, 4)
    ax1.set_zlabel('value fn')
    ax1.view_init(elev=60, azim=135)
    ax1.set_title('model_' + str(f_idx))
    # fig.colorbar(surf)
    plt.pause(0.1)
plt.show()
