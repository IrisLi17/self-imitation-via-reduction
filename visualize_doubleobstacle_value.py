import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from run_ppo import make_env
from stable_baselines import PPO2


def gen_value_with_obstacle(obs, model, goal_idx, env_hyperparam):
    obs = obs.copy()
    assert goal_idx == 1 or goal_idx == 2
    obstacle_xpos, obstacle_ypos = np.meshgrid(np.linspace(env_hyperparam['xlim'][0], env_hyperparam['xlim'][1], 21),
                                               np.linspace(env_hyperparam['ylim'][0], env_hyperparam['ylim'][1], 21))
    grid_shape = obstacle_xpos.shape
    _obstacle_xpos = np.reshape(obstacle_xpos, (-1, 1))
    _obstacle_ypos = np.reshape(obstacle_ypos, (-1, 1))
    batch_obs = np.tile(obs, (_obstacle_xpos.shape[0], 1))
    batch_obs[:, 3 + 3 * goal_idx] = _obstacle_xpos[:, 0]
    batch_obs[:, 4 + 3 * goal_idx] = _obstacle_ypos[:, 0]
    batch_obs[:, 3 + 3 * env_hyperparam['n_object'] + 3 * goal_idx] = batch_obs[:, 3 + 3 * goal_idx] - batch_obs[:, 0]
    batch_obs[:, 4 + 3 * env_hyperparam['n_object'] + 3 * goal_idx] = batch_obs[:, 4 + 3 * goal_idx] - batch_obs[:, 1]
    batch_value = model.value(batch_obs)
    grid_value = np.reshape(batch_value, grid_shape)
    return obstacle_xpos, obstacle_ypos, grid_value

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python visualize_doubleobstacle_value.py [load_path]')
    load_path = sys.argv[1]
    env_name = 'MasspointPushDoubleObstacle-v1'
    env_kwargs = dict(random_box=True,
                      random_ratio=0.0,
                      random_pusher=False,
                      max_episode_steps=150,)
    env = make_env(env_name, seed=None, rank=0, log_dir=None, kwargs=env_kwargs)
    env.reset()
    env_hyperparam = dict(xlim=(0, 5), ylim=(0, 5), n_object=3)
    state = env.sim.get_state()
    masspoint_jointx_i = env.sim.model.get_joint_qpos_addr('masspoint:slidex')
    masspoint_jointy_i = env.sim.model.get_joint_qpos_addr('masspoint:slidey')
    box_jointx_i = env.sim.model.get_joint_qpos_addr('object0:slidex')
    box_jointy_i = env.sim.model.get_joint_qpos_addr('object0:slidey')
    obstacle1_jointx_i = env.sim.model.get_joint_qpos_addr('object1:slidex')
    obstacle1_jointy_i = env.sim.model.get_joint_qpos_addr('object1:slidey')
    obstacle2_jointx_i = env.sim.model.get_joint_qpos_addr('object2:slidex')
    obstacle2_jointy_i = env.sim.model.get_joint_qpos_addr('object2:slidey')
    state.qpos[masspoint_jointx_i] = 2.5
    state.qpos[masspoint_jointy_i] = 2.5
    state.qpos[box_jointx_i] = 2.5
    state.qpos[box_jointy_i] = 2.0
    state.qpos[obstacle1_jointx_i] = env.pos_wall0[0] - env.size_wall[0] - env.size_obstacle[0]
    state.qpos[obstacle1_jointy_i] = 2.5
    state.qpos[obstacle2_jointx_i] = env.pos_wall2[0] - env.size_wall[0] - env.size_obstacle[0]
    state.qpos[obstacle2_jointy_i] = 2.5
    env.sim.set_state(state)
    for _ in range(10):
        env.sim.step()
    # env.sim.forward()
    env.goal[:2] = np.array([3.8, 2.5])
    one_hot = np.zeros(3)
    one_hot[0] = 1
    env.goal[2] = env.sim.data.get_site_xpos('object0')[2]
    env.goal[3:] = one_hot
    obs = env.get_obs()
    obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
    print(obs)

    model = PPO2.load(load_path)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    xs1, ys1, zs1 = gen_value_with_obstacle(obs, model, 1, env_hyperparam=env_hyperparam)
    xs2, ys2, zs2 = gen_value_with_obstacle(obs, model, 2, env_hyperparam=env_hyperparam)
    ax[0].cla()
    ax[1].cla()
    ax[2].cla()
    ax[0].imshow(env.render(mode='rgb_array'))
    surf1 = ax[1].contour(xs1, ys1, zs1, 20, cmap=cm.coolwarm)
    ax[1].clabel(surf1, surf1.levels, inline=True)
    ax[1].scatter(obs[6], obs[7], c='tab:brown')
    surf2 = ax[2].contour(xs2, ys2, zs2, 20, cmap=cm.coolwarm)
    ax[2].clabel(surf2, surf2.levels, inline=True)
    ax[2].scatter(obs[9], obs[10], c='#ff00ff')
    ax[1].set_xlim(env_hyperparam['xlim'][0], env_hyperparam['xlim'][1])
    ax[1].set_ylim(env_hyperparam['ylim'][0], env_hyperparam['ylim'][1])
    ax[1].set_xlabel('obstacle x')
    ax[1].set_ylabel('obstacle y')
    ax[2].set_xlim(env_hyperparam['xlim'][0], env_hyperparam['xlim'][1])
    ax[2].set_ylim(env_hyperparam['ylim'][0], env_hyperparam['ylim'][1])
    ax[2].set_xlabel('obstacle x')
    ax[2].set_ylabel('obstacle y')
    plt.savefig(os.path.join(os.path.dirname(load_path), 'value.png'))
    plt.show()