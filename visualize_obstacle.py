import os
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from test_her import ENTRY_POINT
from stable_baselines.her import HER
from stable_baselines.her.utils import KEY_ORDER


def reset_end_effector(env, xy):
    initial_mocap_pos = env.sim.data.get_mocap_pos('robot0:mocap')
    z = initial_mocap_pos[2]
    initial_mocap_pos[2] = 2.0
    env.sim.data.set_mocap_pos('robot0:mocap', initial_mocap_pos)
    for _ in range(10):
        env.sim.step()
    mocap_pos = np.concatenate([xy, [z]])
    env.sim.data.set_mocap_pos('robot0:mocap', mocap_pos)
    for _ in range(10):
        env.sim.step()
    env._step_callback()

def plot_value_obstaclepos(obs, sac_model, load_path):
    # Obstacle pos.
    obstacle_xpos, obstacle_ypos = np.meshgrid(np.linspace(1.0, 1.6, 21), np.linspace(0.4, 1.1, 21))
    grid_shape = obstacle_xpos.shape
    _obstacle_xpos = np.reshape(obstacle_xpos, (-1, 1))
    _obstacle_ypos = np.reshape(obstacle_ypos, (-1, 1))
    batch_obs = np.tile(obs, (_obstacle_xpos.shape[0], 1))
    batch_obs[:, 6] = _obstacle_xpos[:, 0]
    batch_obs[:, 7] = _obstacle_ypos[:, 0]
    batch_obs[:, 12] = batch_obs[:, 6] - batch_obs[:, 0]
    batch_obs[:, 13] = batch_obs[:, 7] - batch_obs[:, 1]

    # sac_model = model.model
    feed_dict = {
                sac_model.observations_ph: batch_obs,
    }
    fig = plt.figure(figsize=(6, 6))
    # ax1 = fig.add_subplot(111, projection='3d')
    ax1 = fig.add_subplot(111)
    for f_idx in range(50, 90):
        sac_model.load_parameters(os.path.join(load_path, 'model_' + str(f_idx) + '.zip'))
        values = sac_model.sess.run(sac_model.step_ops[6], feed_dict)
        grid_values = np.reshape(values, grid_shape)
        print(np.max(values))
        ax1.cla()
        # surf = ax1.plot_surface(obstacle_xpos, obstacle_ypos, grid_values, cmap=cm.coolwarm)
        surf = ax1.contourf(obstacle_xpos, obstacle_ypos, grid_values, cmap=cm.coolwarm)
        ax1.set_xlim(1.0, 1.6)
        ax1.set_ylim(0.4, 1.1)
        ax1.set_xlabel('obstacle xpos')
        ax1.set_ylabel('obstacle ypos')
        # ax1.set_zlim(0, 8)
        # ax1.set_zlabel('value fn')
        # ax1.view_init(elev=60, azim=135)
        ax1.set_title('model_' + str(f_idx))

        if f_idx == 89:
            fig.colorbar(surf, ax=ax1)
        plt.pause(0.1)
    plt.show()

if __name__ == '__main__':
    env_name = 'FetchPushWallObstacle-v1'
    kwargs = dict(random_box=False, heavy_obstacle=True, hide_velocity=True)
    env = ENTRY_POINT[env_name](**kwargs)

    # Pick a load_path.
    load_path = 'logs/FetchPushWallObstacle-v1_heavy_purerandom/her_sac'
    load_path = 'logs/FetchPushWallObstacle-v1_heavy_purerandom_hidev/her_sac'
    model = HER.load(os.path.join(load_path, 'model_40.zip'), env=env)

    # Put box as it is in batch_obs[8]
    env.reset()
    # object_qpos = env.sim.data.get_joint_qpos('object0:joint')
    # object_qpos[:2] = np.array([1.47, 0.82])
    # env.sim.data.set_joint_qpos('object0:joint', object_qpos)
    # env.sim.forward()
    # Set goal.
    env.goal[:2] = np.array([1.2, 0.75])
    # Put end effector where we are interested.
    effector_xy = np.array([1.55, 0.75])
    # effector_xy = np.random.uniform([1.2, 0.6], [1.5, 0.9], size=2)
    print(effector_xy)
    reset_end_effector(env, effector_xy)
    object_qpos = env.sim.data.get_joint_qpos('object0:joint')
    object_qpos[:2] = np.array([1.47, 0.82])
    env.sim.data.set_joint_qpos('object0:joint', object_qpos)
    env.sim.forward()
    img = env.render(mode='rgb_array')
    plt.imshow(img)
    plt.show()

    obs_dict = env._get_obs()
    # Set goal.
    # obs_dict['desired_goal'][:2] = np.asarray([1.2, 0.75])
    # Convert to array.
    obs_array = np.concatenate([obs_dict[key] for key in KEY_ORDER])

    plot_value_obstaclepos(obs_array, model.model, load_path)
