import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from run_ppo import make_env
from stable_baselines import PPO2


def gen_value_with_obstacle(obs, model, goal_idx, env_hyperparam, axes='xz'):
    obs = obs.copy()
    assert goal_idx == 1 or goal_idx == 2
    obstacle_xpos, obstacle_ypos = np.meshgrid(np.linspace(env_hyperparam[axes[0] + 'lim'][0], env_hyperparam[axes[0] + 'lim'][1], 21),
                                               np.linspace(env_hyperparam[axes[1] + 'lim'][0], env_hyperparam[axes[1] + 'lim'][1], 21))
    id0 = ord(axes[0]) - ord('x')
    id1 = ord(axes[1]) - ord('x')
    grid_shape = obstacle_xpos.shape
    _obstacle_xpos = np.reshape(obstacle_xpos, (-1, 1))
    _obstacle_ypos = np.reshape(obstacle_ypos, (-1, 1))
    batch_obs = np.tile(obs, (_obstacle_xpos.shape[0], 1))
    batch_obs[:, 3 + 3 * goal_idx + id0] = _obstacle_xpos[:, 0]
    batch_obs[:, 3 + 3 * goal_idx + id1] = _obstacle_ypos[:, 0]
    batch_obs[:, 3 + 3 * env_hyperparam['n_object'] + 3 * goal_idx + id0] = batch_obs[:, 3 + 3 * goal_idx + id0] - batch_obs[:, id0]
    batch_obs[:, 3 + 3 * env_hyperparam['n_object'] + 3 * goal_idx + id1] = batch_obs[:, 3 + 3 * goal_idx + id1] - batch_obs[:, id1]
    batch_value = model.value(batch_obs)
    grid_value = np.reshape(batch_value, grid_shape)
    return obstacle_xpos, obstacle_ypos, grid_value

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python visualize_doubleobstacle_value.py [load_path]')
    load_path = sys.argv[1]
    env_name = 'FetchStack-v1'
    env_kwargs = dict(random_box=True,
                      random_ratio=1.0,
                      random_gripper=True,
                      max_episode_steps=100,
                      reward_type='sparse',)
    env = make_env(env_name, seed=None, rank=0, log_dir=None, kwargs=env_kwargs)
    env.reset()
    env_hyperparam = dict(xlim=(1.05, 1.55), ylim=(0.4, 1.1), zlim=(0.425, 0.6), n_object=env.n_object)
    # state = env.sim.get_state()
    # masspoint_jointx_i = env.sim.model.get_joint_qpos_addr('masspoint:slidex')
    # masspoint_jointy_i = env.sim.model.get_joint_qpos_addr('masspoint:slidey')
    # box_jointx_i = env.sim.model.get_joint_qpos_addr('object0:slidex')
    # box_jointy_i = env.sim.model.get_joint_qpos_addr('object0:slidey')
    # obstacle1_jointx_i = env.sim.model.get_joint_qpos_addr('object1:slidex')
    # obstacle1_jointy_i = env.sim.model.get_joint_qpos_addr('object1:slidey')
    # obstacle2_jointx_i = env.sim.model.get_joint_qpos_addr('object2:slidex')
    # obstacle2_jointy_i = env.sim.model.get_joint_qpos_addr('object2:slidey')
    # state.qpos[masspoint_jointx_i] = 2.5
    # state.qpos[masspoint_jointy_i] = 2.5
    # state.qpos[box_jointx_i] = 2.5
    # state.qpos[box_jointy_i] = 2.0
    # state.qpos[obstacle1_jointx_i] = env.pos_wall0[0] - env.size_wall[0] - env.size_obstacle[0]
    # state.qpos[obstacle1_jointy_i] = 2.5
    # state.qpos[obstacle2_jointx_i] = env.pos_wall2[0] - env.size_wall[0] - env.size_obstacle[0]
    # state.qpos[obstacle2_jointy_i] = 2.5
    # env.sim.set_state(state)
    # for _ in range(10):
    #     env.sim.step()
    # env.sim.forward()
    # env.goal[:2] = np.array([3.8, 2.5])
    while env.task_mode != 1:
        env.reset()
    env.goal[2] = 0.425 + 0.05
    one_hot = np.zeros(env.n_object)
    one_hot[0] = 1
    env.goal[3:] = one_hot
    obs = env.get_obs()
    obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
    print(obs)

    model = PPO2.load(load_path)
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    xs1, zs1, value_xz_1 = gen_value_with_obstacle(obs, model, 1, env_hyperparam=env_hyperparam, axes='xz')
    ys1, _, value_yz_1 = gen_value_with_obstacle(obs, model, 1, env_hyperparam=env_hyperparam, axes='yz')
    xs2, zs2, value_xz_2 = gen_value_with_obstacle(obs, model, 2, env_hyperparam=env_hyperparam, axes='xz')
    ys2, _, value_yz_2 = gen_value_with_obstacle(obs, model, 2, env_hyperparam=env_hyperparam, axes='yz')
    ax[0][0].cla()
    ax[0][1].cla()
    ax[0][2].cla()
    ax[0][0].imshow(env.render(mode='rgb_array'))
    surf_xz_1 = ax[0][1].contour(xs1, zs1, value_xz_1, 20, cmap=cm.coolwarm)
    ax[0][1].clabel(surf_xz_1, surf_xz_1.levels, inline=True)
    ax[0][1].scatter(obs[3], obs[5], c='tab:blue')
    surf_yz_1 = ax[0][2].contour(ys1, zs1, value_yz_1, 20, cmap=cm.coolwarm)
    ax[0][2].clabel(surf_yz_1, surf_yz_1.levels, inline=True)
    ax[0][2].scatter(obs[4], obs[5], c='tab:blue')

    surf_xz_2 = ax[1][0].contour(xs2, zs2, value_xz_2, 20, cmap=cm.coolwarm)
    ax[1][0].clabel(surf_xz_2, surf_xz_2.levels, inline=True)
    ax[1][0].scatter(obs[6], obs[8], c='tab:green')
    surf_yz_2 = ax[1][1].contour(ys2, zs2, value_yz_2, 20, cmap=cm.coolwarm)
    ax[1][1].clabel(surf_yz_2, surf_yz_2.levels, inline=True)
    ax[1][1].scatter(obs[7], obs[8], c='tab:green')

    ax[0][1].set_xlim(env_hyperparam['xlim'][0], env_hyperparam['xlim'][1])
    ax[0][1].set_ylim(env_hyperparam['zlim'][0], env_hyperparam['zlim'][1])
    ax[0][1].set_xlabel('obstacle x')
    ax[0][1].set_ylabel('obstacle z')
    ax[0][2].set_xlim(env_hyperparam['ylim'][0], env_hyperparam['ylim'][1])
    ax[0][2].set_ylim(env_hyperparam['zlim'][0], env_hyperparam['zlim'][1])
    ax[0][2].set_xlabel('obstacle y')
    ax[0][2].set_ylabel('obstacle z')
    plt.savefig(os.path.join(os.path.dirname(load_path), 'value.png'))
    plt.show()