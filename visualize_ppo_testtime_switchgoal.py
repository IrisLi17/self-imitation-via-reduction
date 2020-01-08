import sys, os
import numpy as np
import matplotlib.pyplot as plt
from run_ppo import make_env
from stable_baselines import PPO2


def same_side(pos0, pos1, sep):
    if (pos0 - sep) * (pos1 - sep) > 0:
        return True
    return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python visualize_ppo_testtime_switchgoal.py [load_path]')
    load_path = sys.argv[1]
    # Prepare environment and model
    env_name = 'FetchPushWallObstacle-v4'
    env_kwargs = dict(random_box=True,
                      heavy_obstacle=True,
                      random_ratio=0.0,
                      random_gripper=True, )
    env_name = 'MasspointPushSingleObstacle-v2'
    env_kwargs = dict(random_box=True,
                      random_ratio=0.0,
                      random_pusher=True,
                      max_episode_steps=200, )
    env = make_env(env_id=env_name, seed=None, rank=0, log_dir=None, kwargs=env_kwargs)
    goal_dim = env.goal.shape[0]
    obs_dim = env.observation_space.shape[0] - 2 * goal_dim
    noise_mag = env.size_obstacle[0] * 5
    n_object = env.n_object
    horizon = env_kwargs['max_episode_steps']
    model = PPO2.load(load_path)
    # We only test the tasks that aim to move box
    obs = env.reset()
    while np.argmax(obs[obs_dim + goal_dim + 3:]) != 0 \
            or (not same_side(obs[3], obs[6], env.pos_wall0[0])) \
            or (not same_side(obs[0], obs[6], env.pos_wall0[0])):
        obs = env.reset()
    ultimate_goal = obs[-goal_dim:].copy()
    # Generate obstacle positions to select from.
    object_idx = 1
    noise = np.random.uniform(low=-noise_mag, high=noise_mag, size=(100, 2))
    obstacle_xy = np.expand_dims(obs[3 * (object_idx + 1): 3 * (object_idx + 1) + 2], axis=0) + noise
    perturb_obs = np.tile(obs, (noise.shape[0], 1))
    perturb_obs[:, 3 * (object_idx + 1): 3 * (object_idx + 1) + 2] = obstacle_xy
    perturb_obs[:, 3 * (object_idx + 1 + n_object): 3 * (object_idx + 1 + n_object) + 2] \
        = perturb_obs[:, 3 * (object_idx + 1): 3 * (object_idx + 1) + 2] - perturb_obs[:, 0:2]
    # Value2 aim to answer: if the obstacle is perturbed, will moving the box become easy?
    value2 = model.value(perturb_obs)
    print(value2.shape)
    subgoal_obs = np.tile(obs, (noise.shape[0], 1))
    # Achieved goal (current obstacle pos)
    subgoal_obs[:, obs_dim: obs_dim + 3] = subgoal_obs[:, 3 * (object_idx + 1): 3 * (object_idx + 1) + 3]
    subgoal_obs[:, obs_dim + 3: obs_dim + goal_dim] = np.array([[0., 1.]])
    # Desired goal (sampled perturbed obstacle pos)
    subgoal_obs[:, obs_dim + goal_dim: obs_dim + goal_dim + 2] = obstacle_xy
    subgoal_obs[:, obs_dim + goal_dim + 3: obs_dim + goal_dim * 2] = np.array([[0., 1.]])
    # Value1 aim to answer if the subgoal is easy to achieve
    value1 = model.value(subgoal_obs)
    print(value1.shape)

    # Select the best subgoal according to C(V1, V2)
    normalize_value1 = (value1 - np.min(value1)) / (np.max(value1) - np.min(value1))
    normalize_value2 = (value2 - np.min(value2)) / (np.max(value2) - np.min(value2))
    best_idx = np.argmax(normalize_value1 * normalize_value2)
    best_subgoal = obstacle_xy[best_idx]
    print('best subgoal', best_subgoal,
          'with value1', normalize_value1[best_idx], 'value2', normalize_value2[best_idx])

    # Modify the env goal to subgoal
    env.goal[0:2] = best_subgoal
    env.goal[2] = obs[3 * (object_idx+ 1) + 2]
    env.goal[3:] = np.array([0., 1.])
    print('subgoal', env.goal)
    # obs = env.unwrapped.get_obs()
    obs[obs_dim: obs_dim + 3] = obs[3 * (object_idx + 1):3 * (object_idx + 1) + 3]
    obs[obs_dim + 3:obs_dim + goal_dim] = np.array([0., 1.])
    obs[obs_dim + goal_dim: obs_dim + goal_dim * 2] = env.goal[:]
    step_so_far = 0
    done = False
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        assert np.argmax(obs[-goal_dim + 3:]) == 1
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
        env.goal[:] = ultimate_goal
        print('ultimate goal', env.goal)
        # obs = env.unwrapped.get_obs()
        obs[obs_dim:obs_dim + 3] = obs[3:6]
        obs[obs_dim + 3:obs_dim +goal_dim] = np.array([1., 0.])
        obs[obs_dim + goal_dim: obs_dim + 2 * goal_dim] = ultimate_goal
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
    os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(load_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
              os.path.join(os.path.dirname(load_path), 'testtime_switchgoal_model_%d.mp4' % model_idx))
    for i in range(horizon):
        try:
            os.remove(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % i))
        except:
            pass
