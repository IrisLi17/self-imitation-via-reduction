import sys, os, pickle, pandas
import numpy as np
from test_ensemble import make_env
from baselines import HER_HACK
from stable_baselines.her.utils import KEY_ORDER
import matplotlib.pyplot as plt
from matplotlib import cm


def get_item(log_file, label):
    data = pandas.read_csv(log_file, index_col=None, comment='#')
    return data[label].values


# Reset end effector.
def reset_end_effector(env, state, xy):
    env.sim.set_state(state)
    env.sim.forward()
    sim_state = env.sim.get_state()
    box_jointx_i = env.sim.model.get_joint_qpos_addr("object0:slidex")
    box_jointy_i = env.sim.model.get_joint_qpos_addr("object0:slidey")
    box_pos_x = sim_state.qpos[box_jointx_i]
    box_pos_y = sim_state.qpos[box_jointy_i]
    stick_qpos = env.sim.data.get_joint_qpos('object1:joint')
    assert stick_qpos.shape == (7,)
    # stick_qpos[:2] = stick_xpos
    # self.sim.data.set_joint_qpos('object1:joint', stick_qpos)
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
    env.unwrapped._step_callback()
    # Make sure box and obstacle remain unchanged.
    sim_state = env.sim.get_state()
    sim_state.qpos[box_jointx_i] = box_pos_x
    sim_state.qpos[box_jointy_i] = box_pos_y
    env.sim.set_state(sim_state)
    env.sim.data.set_joint_qpos('object1:joint', stick_qpos)
    env.sim.forward()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python visualize_value_sanity.py model_path')
    load_path = sys.argv[1]
    states_file = os.path.join(os.path.dirname(load_path), 'states.pkl')
    with open(states_file, 'rb') as f:
        states = pickle.load(f)
    env_kwargs = dict(random_box=True,
                      heavy_obstacle=True,
                      random_ratio=1.0,
                      random_gripper=True,
                      max_episode_steps=100, )
    env = make_env('FetchPushWallObstacle-v4', **env_kwargs)
    model = HER_HACK.load(load_path, env=env)

    fname = os.path.join(os.path.dirname(load_path), 'success_traj.csv')
    goals = []
    for i in range(5):
        goals.append(get_item(fname, 'goal_' + str(i)))
    goals = np.asarray(goals)
    goals = np.swapaxes(goals, 0, 1)
    dones = get_item(fname, 'done')
    end_points = np.where(dones > 0.5)[0]
    parsed_goals = [goals[i] for i in end_points]
    print(parsed_goals)

    env.reset()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # env.unwrapped.sim.set_state(states[0])
    env.unwrapped.goal = parsed_goals[0]
    print(env.unwrapped.goal, env.goal)
    # env.unwrapped.sim.forward()
    # Set end effector
    pos_x, pos_y = np.meshgrid(np.linspace(1.05, 1.55, 10), np.linspace(0.55, 0.95, 10))
    grid_shape = pos_x.shape
    _pos_x = np.reshape(pos_x, (-1, 1))
    _pos_y = np.reshape(pos_y, (-1, 1))
    pos_xy = np.concatenate((_pos_x, _pos_y), axis=-1)
    for step in range(100):
        print('step', step)
        batch_obs = []
        real_pos_x, real_pos_y = [], []
        env.sim.set_state(states[step])
        env.sim.forward()
        gripper_pos = env.sim.data.get_site_xpos('robot0:grip')[:2]
        img = env.render(mode='rgb_array')
        ax[0].cla()
        ax[0].imshow(img)
        ax[1].cla()
        ax[1].scatter(gripper_pos[0], gripper_pos[1], c='tab:orange')
        for i in range(pos_xy.shape[0]):
            reset_end_effector(env, states[step], pos_xy[i])
            obs_dict = env.get_obs()
            assert np.linalg.norm(obs_dict['desired_goal'] - parsed_goals[0]) < 1e-4
            batch_obs.append(np.concatenate([obs_dict[key] for key in KEY_ORDER]))
            real_pos_x.append(obs_dict['observation'][0])
            real_pos_y.append(obs_dict['observation'][1])
        batch_obs = np.asarray(batch_obs)
        real_pos_x = np.reshape(np.asarray(real_pos_x), grid_shape)
        real_pos_y = np.reshape(np.asarray(real_pos_y), grid_shape)
        feed_dict = {model.model.observations_ph: batch_obs}
        values = model.model.sess.run(model.model.step_ops[6], feed_dict)
        grid_values = np.reshape(values, grid_shape)
        surf = ax[1].contour(real_pos_x, real_pos_y, grid_values, 20, cmap=cm.coolwarm)
        ax[1].clabel(surf, surf.levels, inline=True)
        ax[1].set_xlim(1.05, 1.55)
        ax[1].set_ylim(0.55, 0.95)
        ax[1].set_title('step ' + str(step))
        plt.savefig('tempimg%d.png' % step)
        # plt.pause(0.1)
    os.system('ffmpeg -r 2 -start_number 0 -i tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
              os.path.join(os.path.dirname(load_path), 'value_gripperpos.mp4'))
    for i in range(100):
        os.remove('tempimg%d.png' % i)
