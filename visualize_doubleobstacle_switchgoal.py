import sys, os
import numpy as np
import matplotlib.pyplot as plt
from run_ppo import make_env
from stable_baselines import PPO2


if __name__ == '__main__':
    load_path = sys.argv[1]
    env_name = 'MasspointPushDoubleObstacle-v1'
    env_kwargs = dict(random_box=True,
                      random_ratio=0.0,
                      random_pusher=False,
                      max_episode_steps=150, )
    env = make_env(env_name, seed=None, rank=0, log_dir=None, kwargs=env_kwargs)
    model = PPO2.load(load_path)
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
    state.qpos[box_jointx_i] = 3.8
    state.qpos[box_jointy_i] = 2.5
    state.qpos[obstacle1_jointx_i] = env.pos_wall0[0] - env.size_wall[0] - env.size_obstacle[0]
    state.qpos[obstacle1_jointy_i] = 2.5
    state.qpos[obstacle2_jointx_i] = env.pos_wall2[0] - env.size_wall[0] - env.size_obstacle[0]
    state.qpos[obstacle2_jointy_i] = 2.5
    env.sim.set_state(state)
    for _ in range(5):
        env.sim.step()
    env.goal[:2] = np.array([2.5, 2.0])
    one_hot = np.zeros(3)
    one_hot[0] = 1
    env.goal[2] = env.sim.data.get_site_xpos('object0')[2]
    print('object0 height', env.goal[2])
    env.goal[3:] = one_hot
    obs = env.get_obs()
    obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
    print(obs)

    goal_dim = env.goal.shape[0]
    obs_dim = env.observation_space.shape[0] - 2 * goal_dim
    noise_mag = env.size_obstacle[0] * 4
    n_object = env.n_object
    horizon = env_kwargs['max_episode_steps']
    ultimate_goal = obs[-goal_dim:].copy()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    original_obs = obs.copy()
    done = False
    step_so_far = 0
    while not done:
        action, _ = model.predict(original_obs)
        original_obs, _, done, _ = env.step(action)
        ax.cla()
        ax.imshow(env.render(mode='rgb_array'))
        ax.set_title('step %d' % step_so_far)
        plt.savefig(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % step_so_far))
        plt.pause(0.1)
        step_so_far += 1
    model_idx = int(os.path.basename(load_path).strip('.zip').split('_')[1])
    os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(
        load_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
              os.path.join(os.path.dirname(load_path), 'original_policy_model_%d.mp4' % model_idx))
    for i in range(horizon):
        try:
            os.remove(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % i))
        except:
            pass
    exit()

    # Generate obstacle positions to select from.
    n_obstacle = 2
    perturb_obs_samples = []
    subgoal_obs_samples = []
    obstacle_xy_samples = []
    for object_idx in range(1, n_obstacle + 1):
        noise = np.random.uniform(low=-noise_mag, high=noise_mag, size=(100, 2))
        obstacle_xy = np.expand_dims(obs[3 * (object_idx + 1): 3 * (object_idx + 1) + 2], axis=0) + noise
        obstacle_xy_samples.append(obstacle_xy)
        perturb_obs = np.tile(obs, (noise.shape[0], 1))
        perturb_obs[:, 3 * (object_idx + 1): 3 * (object_idx + 1) + 2] = obstacle_xy
        perturb_obs[:, 3 * (object_idx + 1 + n_object): 3 * (object_idx + 1 + n_object) + 2] \
            = perturb_obs[:, 3 * (object_idx + 1): 3 * (object_idx + 1) + 2] - perturb_obs[:, 0:2]
        perturb_obs_samples.append(perturb_obs.copy())

        subgoal_obs = np.tile(obs, (noise.shape[0], 1))
        # Achieved goal (current obstacle pos)
        subgoal_obs[:, obs_dim: obs_dim + 3] = subgoal_obs[:, 3 * (object_idx + 1): 3 * (object_idx + 1) + 3]
        one_hot = np.zeros(n_object)
        one_hot[object_idx] = 1
        subgoal_obs[:, obs_dim + 3: obs_dim + goal_dim] = one_hot
        # Desired goal (sampled perturbed obstacle pos)
        subgoal_obs[:, obs_dim + goal_dim: obs_dim + goal_dim + 2] = obstacle_xy
        subgoal_obs[:, obs_dim + goal_dim + 2] = subgoal_obs[:, 3 + 3 * object_idx + 2]
        subgoal_obs[:, obs_dim + goal_dim + 3: obs_dim + goal_dim * 2] = one_hot
        subgoal_obs_samples.append(subgoal_obs)

    # Value2 aim to answer: if the obstacle is perturbed, will moving the box become easy?
    perturb_obs_samples = np.concatenate(perturb_obs_samples, axis=0)
    value2 = model.value(perturb_obs_samples)
    print(value2.shape)
    # Value1 aim to answer if the subgoal is easy to achieve
    subgoal_obs_samples = np.concatenate(subgoal_obs_samples, axis=0)
    value1 = model.value(subgoal_obs_samples)
    print(value1.shape)

    # Select the best subgoal according to C(V1, V2)
    normalize_value1 = (value1 - np.min(value1)) / (np.max(value1) - np.min(value1))
    normalize_value2 = (value2 - np.min(value2)) / (np.max(value2) - np.min(value2))
    best_idx = np.argmax(normalize_value1 * normalize_value2)
    # obstacle_xy_samples = np.concatenate(obstacle_xy_samples, axis=0)
    # best_subgoal = obstacle_xy_samples[best_idx]
    best_subgoal = subgoal_obs_samples[best_idx][obs_dim + goal_dim: obs_dim + 2 * goal_dim]
    print('best subgoal', best_subgoal,
          'with value1', normalize_value1[best_idx], 'value2', normalize_value2[best_idx])

    env.unwrapped.goal[:] = best_subgoal
    obs = env.get_obs()
    obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']], axis=0)
    step_so_far = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        img = env.render(mode='rgb_array')
        ax.cla()
        ax.imshow(img)
        ax.set_title('step %d' % step_so_far)
        plt.savefig(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % step_so_far))
        plt.pause(0.1)
        step_so_far += 1
        if step_so_far >= horizon:
            break
    if not done:
        print('Fail to achieve subgoal.')
    else:
        # Achieved subgoal, still have time left
        assert step_so_far < horizon
        env.unwrapped.goal[:] = ultimate_goal
        print('ultimate goal', env.goal)
        obs = env.get_obs()
        obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
            assert np.argmax(obs[-goal_dim + 3:]) == 0
            img = env.render(mode='rgb_array')
            ax.cla()
            ax.imshow(img)
            ax.set_title('step %d' % step_so_far)
            plt.savefig(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % step_so_far))
            plt.pause(0.1)
            step_so_far += 1
            if step_so_far >= horizon:
                break
        if step_so_far < horizon:
            print('Success!')
        else:
            print('Fail to achieve ultimate goal.')
    model_idx = int(os.path.basename(load_path).strip('.zip').split('_')[1])
    os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(
        load_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
              os.path.join(os.path.dirname(load_path), 'testtime_switchgoal_model_%d.mp4' % model_idx))
    for i in range(horizon):
        try:
            os.remove(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % i))
        except:
            pass