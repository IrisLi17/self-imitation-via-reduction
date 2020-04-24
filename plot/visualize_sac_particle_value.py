import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from run_her import make_env, get_env_kwargs
from baselines import HER_HACK
from gym.wrappers import FlattenDictWrapper


def gen_value(obs, model, env_hyperparam, object_idx):
    n_object = env_hyperparam['n_object']
    obstacle_xpos, obstacle_ypos = np.meshgrid(np.linspace(env_hyperparam['xlim'][0], env_hyperparam['xlim'][1], 21),
                                               np.linspace(env_hyperparam['ylim'][0], env_hyperparam['ylim'][1], 21))
    grid_shape = obstacle_xpos.shape
    _obstacle_xpos = np.reshape(obstacle_xpos, (-1, 1))
    _obstacle_ypos = np.reshape(obstacle_ypos, (-1, 1))
    batch_obs = np.tile(obs, (_obstacle_xpos.shape[0], 1))
    batch_obs[:, 3 * (1 + object_idx)] = _obstacle_xpos[:, 0]
    batch_obs[:, 3 * (1 + object_idx) + 1] = _obstacle_ypos[:, 0]
    batch_obs[:, 3 * (1 + n_object + object_idx)] = batch_obs[:, 3 * (1 + object_idx)] - batch_obs[:, 0]
    batch_obs[:, 3 * (1 + n_object + object_idx) + 1] = batch_obs[:, 3 * (1 + object_idx) + 1] - batch_obs[:, 1]
    # Compute value2
    batch_value = model.model.sess.run(model.model.step_ops[6],
                                       {model.model.observations_ph: batch_obs})
    grid_value = np.reshape(batch_value, grid_shape)

    # Compute value1
    one_hot = np.zeros(n_object)
    one_hot[object_idx] = 1
    subgoal_obs = np.tile(obs, (_obstacle_xpos.shape[0], 1))
    # Achieved goal (current obstacle pos)
    subgoal_obs[:, -2 * (3 + n_object): -2 * (3 + n_object) + 3] = subgoal_obs[:, 3 * (1 + object_idx): 3 * (1 + object_idx) + 3]
    subgoal_obs[:, -2 * (3 + n_object) + 3: -(3 + n_object)] = one_hot
    # Desired goal (sampled perturbed obstacle pos)
    obstacle_xy = np.concatenate([_obstacle_xpos, _obstacle_ypos, subgoal_obs[:, 3 * (1 + object_idx) + 2: 3 * (1 + object_idx) + 3]], axis=-1)
    subgoal_obs[:, -(3 + n_object): -n_object] = obstacle_xy
    subgoal_obs[:, -n_object:] = one_hot
    # Value1 aim to answer if the subgoal is easy to achieve
    value1 = model.model.sess.run(model.model.step_ops[6],
                                  {model.model.observations_ph: subgoal_obs})
    grid_value1 = np.reshape(value1, grid_shape)

    # min_value = np.min(np.concatenate([np.expand_dims(value1, 1), np.expand_dims(batch_value,1)], axis=1), axis=1)
    # grid_value_min = np.reshape(min_value, grid_shape)
    normalized_value1 = (value1 - np.min(value1)) / (np.max(value1) - np.min(value1))
    normalized_value2 = (batch_value - np.min(batch_value)) / (np.max(batch_value) - np.min(batch_value))
    value_prod = normalized_value1 * normalized_value2
    grid_value_prod = np.reshape(value_prod, grid_shape)

    return obstacle_xpos, obstacle_ypos, grid_value, grid_value1, grid_value_prod


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python -m plot.visualize_sac_particle_value [load_path]')
    load_path = sys.argv[1]
    env_name = 'MasspointPushDoubleObstacle-v1'
    env_kwargs = get_env_kwargs(env_name, random_ratio=0.0)
    env_hyperparam = dict(xlim=(0.0, 5.0), ylim=(0.0, 5.0), n_object=3)
    # env_name = 'MasspointPushSingleObstacle-v2'
    # env_kwargs = dict(random_box=True,
    #                   random_ratio=0.0,
    #                   random_pusher=True,
    #                   max_episode_steps=200, )
    # env_hyperparam = dict(xlim=(-1.0, 4.0), ylim=(-1.5, 3.5),
    #                       )
    n_cpu = 1
    env = make_env(env_id=env_name, seed=None, rank=0, log_dir=None, kwargs=env_kwargs)
    env = FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])

    model = HER_HACK.load(load_path)

    plt.rcParams.update({'font.size': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20,
                         'axes.labelsize': 20})
    obs = env.reset()
    while not (np.argmax(obs[-env_hyperparam['n_object']:]) == 0):
        obs = env.reset()
    # env.set_goal(np.array([1.2, 0.75, 0.425, 1, 0]))
    # obs = env.get_obs()
    # obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
    img = env.render(mode='rgb_array')
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.cla()
    ax.imshow(img)
    plt.show()
    for idx in range(1, env_hyperparam['n_object']):
        xs, ys, zs, value1s, value_prods = gen_value(obs, model, env_hyperparam, object_idx=idx)
        print('gripper', obs[:3], 'box', obs[3:6], 'obstacle', obs[6:3 * (1 + env_hyperparam['n_object'])],
              'goal', obs[-(3 + env_hyperparam['n_object']):])
        # np.save('xs.npy', xs)
        # np.save('ys.npy', ys)
        # np.save('value1.npy', value1s)
        # np.save('value2.npy', zs)
        # np.save('value_prod.npy', value_prods)
        # plt.imsave(os.path.join(os.path.dirname(load_path), 'obs.png'), img)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.cla()
        surf = ax.contourf((xs - env_hyperparam['xlim'][0]) / (env_hyperparam['xlim'][1] - env_hyperparam['xlim'][0]),
                           (ys - env_hyperparam['ylim'][0]) / (env_hyperparam['ylim'][1] - env_hyperparam['ylim'][0]), value_prods, 15, cmap=cm.coolwarm, vmin=-0.0, vmax=1)
        ax.set_xlabel('x', fontsize=24)
        ax.set_ylabel('y', fontsize=24)
        # ax.plot([(1.25 - 1.05) / 0.5, (1.25 - 1.05) / 0.5], [0, (0.65 - 0.4) / 0.7], 'k', linestyle='--')
        # ax.plot([(1.25 - 1.05) / 0.5, (1.25 - 1.05) / 0.5], [(0.85 - 0.4) / 0.7, (1.1 - 0.4) / 0.7], 'k', linestyle='--')
        ax.axis([0., 1., 0., 1.])
        ax.set_title('value prod')
        cb = plt.colorbar(surf)
        plt.tight_layout()
        # plt.savefig('value_prod.png')

        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        surf = ax.contourf((xs - env_hyperparam['xlim'][0]) / (env_hyperparam['xlim'][1] - env_hyperparam['xlim'][0]),
                           (ys - env_hyperparam['ylim'][0]) / (env_hyperparam['ylim'][1] - env_hyperparam['ylim'][0]), value1s, 15, cmap=cm.coolwarm, vmin=-0.0, vmax=1)
        ax.set_xlabel('x', fontsize=24)
        ax.set_ylabel('y', fontsize=24)
        # ax.plot([(1.25 - 1.05) / 0.5, (1.25 - 1.05) / 0.5], [0, (0.65 - 0.4) / 0.7], 'k', linestyle='--')
        # ax.plot([(1.25 - 1.05) / 0.5, (1.25 - 1.05) / 0.5], [(0.85 - 0.4) / 0.7, (1.1 - 0.4) / 0.7], 'k', linestyle='--')
        ax.axis([0., 1., 0., 1.])
        cb = plt.colorbar(surf)
        plt.tight_layout()
        # plt.savefig('value1.png')

        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        surf = ax.contourf((xs - env_hyperparam['xlim'][0]) / (env_hyperparam['xlim'][1] - env_hyperparam['xlim'][0]),
                           (ys - env_hyperparam['ylim'][0]) / (env_hyperparam['ylim'][1] - env_hyperparam['ylim'][0]), zs, 15, cmap=cm.coolwarm, vmin=-0.0, vmax=1)
        ax.set_xlabel('x', fontsize=24)
        ax.set_ylabel('y', fontsize=24)
        # ax.plot([(1.25 - 1.05) / 0.5, (1.25 - 1.05) / 0.5], [0, (0.65 - 0.4) / 0.7], 'k', linestyle='--')
        # ax.plot([(1.25 - 1.05) / 0.5, (1.25 - 1.05) / 0.5], [(0.85 - 0.4) / 0.7, (1.1 - 0.4) / 0.7], 'k', linestyle='--')
        ax.axis([0., 1., 0., 1.])
        cb = plt.colorbar(surf)
        plt.tight_layout()
        # plt.savefig('value2.png')
        plt.show()
