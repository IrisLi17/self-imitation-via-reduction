import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from run_ppo import make_env
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2


def gen_value_with_obstacle(obs, model, object_idx):
    current_object_xy = obs[3 + 3 * object_idx: 5 + 3 * object_idx]
    obstacle_xpos, obstacle_ypos = np.meshgrid(np.linspace(current_object_xy[0] - 0.6, current_object_xy[0] + 0.6, 21),
                                               np.linspace(current_object_xy[1] - 0.6, current_object_xy[1] + 0.6, 21))
    grid_shape = obstacle_xpos.shape
    _obstacle_xpos = np.reshape(obstacle_xpos, (-1, 1))
    _obstacle_ypos = np.reshape(obstacle_ypos, (-1, 1))
    batch_obs = np.tile(obs, (_obstacle_xpos.shape[0], 1))
    batch_obs[:, 3 + 3 * object_idx] = _obstacle_xpos[:, 0]
    batch_obs[:, 4 + 3 * object_idx] = _obstacle_ypos[:, 0]
    batch_obs[:, 3 + 3 * object_idx + 3 * n_object] = batch_obs[:, 3 + 3 * object_idx] - batch_obs[:, 0]
    batch_obs[:, 4 + 3 * object_idx + 3 * n_object] = batch_obs[:, 4 + 3 * object_idx] - batch_obs[:, 1]
    batch_obs[:, -2 * (3 + n_object): -2 * (3 + n_object) + 3] = batch_obs[:, 3 + 3 * object_idx: 6 + 3 * object_idx]
    # Compute value2
    batch_value = model.value(batch_obs)
    grid_value = np.reshape(batch_value, grid_shape)

    # Compute value1
    subgoal_obs = np.tile(obs, (_obstacle_xpos.shape[0], 1))
    # Achieved goal (current obstacle pos)
    subgoal_obs[:, -2 * (3 + n_object): -2 * (3 + n_object) + 3] = subgoal_obs[:, 3 + 3 * object_idx: 6 + 3 * object_idx]
    one_hot = np.zeros(n_object)
    one_hot[object_idx] = 1
    subgoal_obs[:, -2 * (3 + n_object) + 3: -(3 + n_object)] = one_hot
    # Desired goal (sampled perturbed obstacle pos)
    obstacle_xy = np.concatenate([_obstacle_xpos, _obstacle_ypos, subgoal_obs[:, 5 + 3 * object_idx:6 + 3 * object_idx]], axis=-1)
    subgoal_obs[:, -(3 + n_object): -n_object] = obstacle_xy
    subgoal_obs[:, -n_object:] = one_hot
    # Value1 aim to answer if the subgoal is easy to achieve
    value1 = model.value(subgoal_obs)
    grid_value1 = np.reshape(value1, grid_shape)

    # min_value = np.min(np.concatenate([np.expand_dims(value1, 1), np.expand_dims(batch_value,1)], axis=1), axis=1)
    # grid_value_min = np.reshape(min_value, grid_shape)
    normalized_value1 = (value1 - np.min(value1)) / (np.max(value1) - np.min(value1))
    normalized_value2 = (batch_value - np.min(batch_value)) / (np.max(batch_value) - np.min(batch_value))
    value_prod = normalized_value1 * normalized_value2
    grid_value_prod = np.reshape(value_prod, grid_shape)

    return obstacle_xpos, obstacle_ypos, grid_value, grid_value1, grid_value_prod


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python visualize_ppo_value.py [load_path] [n_object]')
    load_path = sys.argv[1]
    n_object = int(sys.argv[2])
    env_name = 'MasspointPushMultiObstacle-v1'
    env_kwargs = dict(random_box=True,
                      random_ratio=0.0,
                      random_pusher=True,
                      max_episode_steps=50*n_object,
                      n_object=n_object,
                      reward_type="sparse",)
    # env_hyperparam = dict(xlim=(1.05, 1.55), ylim=(0.4, 1.1))
    # env_name = 'MasspointPushSingleObstacle-v2'
    # env_kwargs = dict(random_box=True,
    #                   random_ratio=0.0,
    #                   random_pusher=True,
    #                   max_episode_steps=200, )
    # env_hyperparam = dict(xlim=(-1.0, 4.0), ylim=(-1.5, 3.5),
    #                       )
    n_cpu = 1
    env = make_env(env_id=env_name, seed=None, rank=0, log_dir=None, kwargs=env_kwargs)

    model = PPO2.load(load_path)
    plt.rcParams.update({'font.size': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20,
                         'axes.labelsize': 20})
    obs = env.reset()
    # while np.argmax(obs[-2:]) != 0 \
    #         or (obs[0] - obs[6]) * (obs[6] - env.pos_wall0[0]) < 0 \
    #         or (obs[3] - env.pos_wall0[0]) * (obs[6] - env.pos_wall0[0]) < 0:
    #     obs = env.reset()
    # obs = env.get_obs()
    # obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
    for step in range(1):
        img = env.render(mode='rgb_array')
        for obj_idx in range(n_object):
            xs, ys, zs, value1s, value_prods = gen_value_with_obstacle(obs, model, obj_idx)
            best_idx = np.unravel_index(np.argmax(value_prods, axis=None), value_prods.shape)
            print('best idx', best_idx)
            print(xs[best_idx], ys[best_idx])
            print('best value', value_prods[best_idx], 'value1', value1s[best_idx], 'value2', zs[best_idx])
            print(step, 'gripper', obs[:3], 'box', obs[3:6], 'obstacle', obs[6:9], )
            # np.save('xs.npy', xs)
            # np.save('ys.npy', ys)
            # np.save('value1.npy', value1s)
            # np.save('value2.npy', zs)
            # np.save('value_prod.npy', value_prods)
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax[0][0].cla()
            ax[0][1].cla()
            ax[1][0].cla()
            ax[1][1].cla()
            ax[0][0].imshow(img)
            # plt.imsave(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % step), img)
            # exit()
            # ax.cla()
            ax[0][1].contourf(xs, ys, value_prods, 15, cmap=cm.coolwarm)
            ax[1][0].contourf(xs, ys, value1s, 15, cmap=cm.coolwarm)
            ax[1][1].contourf(xs, ys, zs, 15, cmap=cm.coolwarm)
            plt.show()

    exit()
    model_idx = int(os.path.basename(load_path).strip('.zip').split('_')[1])
    os.system(('ffmpeg -r 2 -start_number 0 -i ' + os.path.dirname(load_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
              os.path.join(os.path.dirname(load_path), 'value_obstacle_model_%d.mp4' % model_idx)))
    for step in range(env_kwargs['max_episode_steps']):
        try:
            os.remove(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % step))
        except:
            pass
