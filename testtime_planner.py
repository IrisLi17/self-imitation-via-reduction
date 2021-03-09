import sys, os
import numpy as np
import matplotlib.pyplot as plt
from run_ppo import make_env
from stable_baselines import PPO2


def same_side(pos0, pos1, sep):
    if (pos0 - sep) * (pos1 - sep) > 0:
        return True
    return False


def testtime_reduction(env, model, initial_state, ultimate_goal):
    is_success = False
    # obs = env.reset()
    # while np.argmax(obs[obs_dim + goal_dim + 3:]) != 0:
    #     obs = env.reset()
    # ultimate_goal = obs[-goal_dim:].copy()
    ultimate_idx = np.argmax(ultimate_goal[3:])
    env.set_state(initial_state)
    env.set_goal(ultimate_goal)
    obs = env.get_obs()
    obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
    original_value = model.value(np.expand_dims(obs, 0))
    # print(obs)
    # Generate obstacle positions to select from.
    object_idx = 1
    noise = np.random.uniform(low=-noise_mag, high=noise_mag, size=(100, 2))
    obstacle_xy = np.expand_dims(obs[3 * (object_idx + 1): 3 * (object_idx + 1) + 2], axis=0) + noise
    perturb_obs = np.tile(obs, (noise.shape[0] * 2, 1))
    perturb_obs[:noise.shape[0], 3 * (object_idx + 1): 3 * (object_idx + 1) + 2] = obstacle_xy
    perturb_obs[:noise.shape[0], 3 * (object_idx + 1 + n_object): 3 * (object_idx + 1 + n_object) + 2] \
        = perturb_obs[:noise.shape[0], 3 * (object_idx + 1): 3 * (object_idx + 1) + 2] - perturb_obs[:noise.shape[0], 0:2]
    perturb_obs[:noise.shape[0], obs_dim:obs_dim + 3] = perturb_obs[:noise.shape[0], 3 * (ultimate_idx + 1) : 3 * (ultimate_idx + 1) + 3]

    subgoal_obs = np.tile(obs, (noise.shape[0] * 2, 1))
    # Achieved goal (current obstacle pos)
    subgoal_obs[:noise.shape[0], obs_dim: obs_dim + 3] = subgoal_obs[:noise.shape[0], 3 * (object_idx + 1): 3 * (object_idx + 1) + 3]
    subgoal_obs[:noise.shape[0], obs_dim + 3: obs_dim + goal_dim] = np.array([[0., 1.]])
    # Desired goal (sampled perturbed obstacle pos)
    subgoal_obs[:noise.shape[0], obs_dim + goal_dim: obs_dim + goal_dim + 2] = obstacle_xy
    subgoal_obs[:noise.shape[0], obs_dim + goal_dim +2 : obs_dim + goal_dim + 3] = subgoal_obs[:noise.shape[0], 3 * (object_idx + 1) + 2 : 3 * (object_idx + 1) + 3]
    subgoal_obs[:noise.shape[0], obs_dim + goal_dim + 3: obs_dim + goal_dim * 2] = np.array([[0., 1.]])

    # Generate box positions to select from
    object_idx = 0
    noise = np.random.uniform(low=-noise_mag, high=noise_mag, size=(100, 2))
    box_xy = np.expand_dims(obs[3 * (object_idx + 1): 3 * (object_idx + 1) + 2], axis=0) + noise
    perturb_obs[noise.shape[0]:, 3 * (object_idx + 1): 3 * (object_idx + 1) + 2] = box_xy
    perturb_obs[noise.shape[0]:, 3 * (object_idx + 1 + n_object): 3 * (object_idx + 1 + n_object) + 2] \
        = perturb_obs[noise.shape[0]:, 3 * (object_idx + 1): 3 * (object_idx + 1) + 2] - perturb_obs[noise.shape[0]:,
                                                                                         0:2]
    perturb_obs[noise.shape[0]:, obs_dim:obs_dim + 3] = perturb_obs[noise.shape[0]:, 3 * (ultimate_idx + 1): 3 * (ultimate_idx + 1) + 3]

    # Achieved goal (current box pos)
    subgoal_obs[noise.shape[0]:, obs_dim: obs_dim + 3] = subgoal_obs[noise.shape[0]:,
                                                         3 * (object_idx + 1): 3 * (object_idx + 1) + 3]
    subgoal_obs[noise.shape[0]:, obs_dim + 3: obs_dim + goal_dim] = np.array([[1., 0.]])
    # Desired goal (sampled perturbed box pos)
    subgoal_obs[noise.shape[0]:, obs_dim + goal_dim: obs_dim + goal_dim + 2] = box_xy
    subgoal_obs[noise.shape[0]:, obs_dim + goal_dim + 2: obs_dim + goal_dim + 3] = subgoal_obs[noise.shape[0]:,
                                                                                   3 * (object_idx + 1) + 2: 3 * (
                                                                                   object_idx + 1) + 3]
    subgoal_obs[noise.shape[0]:, obs_dim + goal_dim + 3: obs_dim + goal_dim * 2] = np.array([[1., 0.]])

    # Value2 aim to answer: if the obstacle is perturbed, will moving the box become easy?
    value2 = model.value(perturb_obs)
    # print(value2.shape)
    # Value1 aim to answer if the subgoal is easy to achieve
    value1 = model.value(subgoal_obs)
    # print(value1.shape)

    obstacle_xy = np.concatenate([obstacle_xy, box_xy], axis=0)
    # Select the best subgoal according to C(V1, V2)
    normalize_value1 = (value1 - np.min(value1)) / (np.max(value1) - np.min(value1))
    normalize_value2 = (value2 - np.min(value2)) / (np.max(value2) - np.min(value2))
    best_idx = np.argmax(normalize_value1 * normalize_value2)
    # if (value2[best_idx] + value1[best_idx]) / 2 > original_value[0]:
    if value2[best_idx] > original_value[0]:
    # if True:
        best_subgoal = obstacle_xy[best_idx]
        object_idx = int(best_idx < noise.shape[0])
        print('best subgoal', best_subgoal, 'idx', object_idx,
              'with value1', normalize_value1[best_idx], 'value2', normalize_value2[best_idx])

        # Modify the env goal to subgoal
        env.goal[0:2] = best_subgoal
        env.goal[2] = obs[3 * (object_idx + 1) + 2]
        one_hot = np.zeros(2)
        one_hot[object_idx] = 1
        # print(one_hot)
        env.goal[3:] = one_hot
        # print('subgoal', env.goal)
        # obs = env.unwrapped.get_obs()
        obs[obs_dim: obs_dim + 3] = obs[3 * (object_idx + 1):3 * (object_idx + 1) + 3]
        obs[obs_dim + 3:obs_dim + goal_dim] = one_hot
        obs[obs_dim + goal_dim: obs_dim + goal_dim * 2] = env.goal[:]
        # print(obs)
        step_so_far = 0
        done = False
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            assert np.argmax(obs[-goal_dim + 3:]) == object_idx
            # img = env.render(mode='rgb_array')
            # ax.cla()
            # ax.imshow(img)
            # ax.set_title('step %d' % step_so_far)
            # plt.savefig(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % step_so_far))
            # plt.pause(0.1)
            step_so_far += 1
            if step_so_far >= horizon:
                break
        if not done:
            print('Fail to achieve subgoal.')
        elif step_so_far < horizon:
            # Achieved subgoal, still have time left
            assert step_so_far < horizon
            env.goal[:] = ultimate_goal
            print('ultimate goal', env.goal)
            # obs = env.unwrapped.get_obs()
            obs[obs_dim:obs_dim + 3] = obs[3 * (ultimate_idx + 1):3 * (ultimate_idx + 1) + 3]
            one_hot = np.zeros(2)
            one_hot[ultimate_idx] = 1
            obs[obs_dim + 3:obs_dim + goal_dim] = one_hot
            obs[obs_dim + goal_dim: obs_dim + 2 * goal_dim] = ultimate_goal
            # print(obs)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = env.step(action)
                # assert np.argmax(obs[-goal_dim + 3:]) == 0
                # img = env.render(mode='rgb_array')
                # ax.cla()
                # ax.imshow(img)
                # ax.set_title('step %d' % step_so_far)
                # plt.savefig(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % step_so_far))
                # plt.pause(0.1)
                step_so_far += 1
                if step_so_far >= horizon:
                    break
            if step_so_far < horizon:
                is_success = True
                print('Success!')
            else:
                print('Fail to achieve ultimate goal.')
    else:
        step_so_far = 0
        done = False
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            # assert np.argmax(obs[-goal_dim + 3:]) == 1
            step_so_far += 1
            if step_so_far >= horizon:
                break
        if done:
            is_success = True
    return is_success


def testtime_noreduction(env, model, initial_state, ultimate_goal):
    is_success = False
    # obs = env.reset()
    # while np.argmax(obs[obs_dim + goal_dim + 3:]) != 0:
    #     obs = env.reset()
    env.set_state(initial_state)
    env.set_goal(ultimate_goal)
    obs = env.get_obs()
    obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
    # print(obs)
    step_so_far = 0
    done = False
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        # assert np.argmax(obs[-goal_dim + 3:]) == 1
        step_so_far += 1
        if step_so_far >= horizon:
            break
    if done:
        is_success = True
    return is_success


def testtime_reduction_new(env, model, initial_state, ultimate_goal):
    is_success = False
    # obs = env.reset()
    # while np.argmax(obs[obs_dim + goal_dim + 3:]) != 0:
    #     obs = env.reset()
    # ultimate_goal = obs[-goal_dim:].copy()
    ultimate_idx = np.argmax(ultimate_goal[3:])
    env.set_state(initial_state)
    env.set_goal(ultimate_goal)
    obs = env.get_obs()
    obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])

    if ultimate_idx == 1:
        is_success = testtime_noreduction(env, model, initial_state, ultimate_goal)
    if ultimate_idx == 0:
        original_value = model.value(np.expand_dims(obs, 0))
        # print(obs)
        # Generate obstacle positions to select from.
        object_idx = 1
        noise = np.random.uniform(low=-noise_mag, high=noise_mag, size=(100, 2))
        obstacle_xy = np.expand_dims(obs[3 * (object_idx + 1): 3 * (object_idx + 1) + 2], axis=0) + noise
        perturb_obs = np.tile(obs, (noise.shape[0] * 2, 1))
        perturb_obs[:noise.shape[0], 3 * (object_idx + 1): 3 * (object_idx + 1) + 2] = obstacle_xy
        perturb_obs[:noise.shape[0], 3 * (object_idx + 1 + n_object): 3 * (object_idx + 1 + n_object) + 2] \
            = perturb_obs[:noise.shape[0], 3 * (object_idx + 1): 3 * (object_idx + 1) + 2] - perturb_obs[:noise.shape[0], 0:2]
        perturb_obs[:noise.shape[0], obs_dim:obs_dim + 3] = perturb_obs[:noise.shape[0], 3 * (ultimate_idx + 1) : 3 * (ultimate_idx + 1) + 3]

        subgoal_obs = np.tile(obs, (noise.shape[0] * 2, 1))
        # Achieved goal (current obstacle pos)
        subgoal_obs[:noise.shape[0], obs_dim: obs_dim + 3] = subgoal_obs[:noise.shape[0], 3 * (object_idx + 1): 3 * (object_idx + 1) + 3]
        subgoal_obs[:noise.shape[0], obs_dim + 3: obs_dim + goal_dim] = np.array([[0., 1.]])
        # Desired goal (sampled perturbed obstacle pos)
        subgoal_obs[:noise.shape[0], obs_dim + goal_dim: obs_dim + goal_dim + 2] = obstacle_xy
        subgoal_obs[:noise.shape[0], obs_dim + goal_dim +2 : obs_dim + goal_dim + 3] = subgoal_obs[:noise.shape[0], 3 * (object_idx + 1) + 2 : 3 * (object_idx + 1) + 3]
        subgoal_obs[:noise.shape[0], obs_dim + goal_dim + 3: obs_dim + goal_dim * 2] = np.array([[0., 1.]])

        # Generate box positions to select from
        object_idx = 0
        noise = np.random.uniform(low=-noise_mag, high=noise_mag, size=(100, 2))
        box_xy = np.expand_dims(obs[3 * (object_idx + 1): 3 * (object_idx + 1) + 2], axis=0) + noise
        perturb_obs[noise.shape[0]:, 3 * (object_idx + 1): 3 * (object_idx + 1) + 2] = box_xy
        perturb_obs[noise.shape[0]:, 3 * (object_idx + 1 + n_object): 3 * (object_idx + 1 + n_object) + 2] \
            = perturb_obs[noise.shape[0]:, 3 * (object_idx + 1): 3 * (object_idx + 1) + 2] - perturb_obs[noise.shape[0]:,
                                                                                             0:2]
        perturb_obs[noise.shape[0]:, obs_dim:obs_dim + 3] = perturb_obs[noise.shape[0]:, 3 * (ultimate_idx + 1): 3 * (ultimate_idx + 1) + 3]

        # Achieved goal (current box pos)
        subgoal_obs[noise.shape[0]:, obs_dim: obs_dim + 3] = subgoal_obs[noise.shape[0]:,
                                                             3 * (object_idx + 1): 3 * (object_idx + 1) + 3]
        subgoal_obs[noise.shape[0]:, obs_dim + 3: obs_dim + goal_dim] = np.array([[1., 0.]])
        # Desired goal (sampled perturbed box pos)
        subgoal_obs[noise.shape[0]:, obs_dim + goal_dim: obs_dim + goal_dim + 2] = box_xy
        subgoal_obs[noise.shape[0]:, obs_dim + goal_dim + 2: obs_dim + goal_dim + 3] = subgoal_obs[noise.shape[0]:,
                                                                                       3 * (object_idx + 1) + 2: 3 * (
                                                                                       object_idx + 1) + 3]
        subgoal_obs[noise.shape[0]:, obs_dim + goal_dim + 3: obs_dim + goal_dim * 2] = np.array([[1., 0.]])

        # Value2 aim to answer: if the obstacle is perturbed, will moving the box become easy?
        value2 = model.value(perturb_obs)
        # print(value2.shape)
        # Value1 aim to answer if the subgoal is easy to achieve
        value1 = model.value(subgoal_obs)
        # print(value1.shape)

        obstacle_xy = np.concatenate([obstacle_xy, box_xy], axis=0)
        # Select the best subgoal according to C(V1, V2)
        normalize_value1 = (value1 - np.min(value1)) / (np.max(value1) - np.min(value1))
        normalize_value2 = (value2 - np.min(value2)) / (np.max(value2) - np.min(value2))
        best_idx = np.argmax(normalize_value1 * normalize_value2)
        if (value2[best_idx] + value1[best_idx]) / 2 > original_value[0]:
        # if value2[best_idx] > original_value[0]:
        # if True:
            best_subgoal = obstacle_xy[best_idx]
            object_idx = int(best_idx < noise.shape[0])
            # print('best subgoal', best_subgoal, 'idx', object_idx,
            #       'with value1', normalize_value1[best_idx], 'value2', normalize_value2[best_idx])

            # Modify the env goal to subgoal
            env.goal[0:2] = best_subgoal
            env.goal[2] = obs[3 * (object_idx + 1) + 2]
            one_hot = np.zeros(2)
            one_hot[object_idx] = 1
            # print(one_hot)
            env.goal[3:] = one_hot
            # print('subgoal', env.goal)
            # obs = env.unwrapped.get_obs()
            obs[obs_dim: obs_dim + 3] = obs[3 * (object_idx + 1):3 * (object_idx + 1) + 3]
            obs[obs_dim + 3:obs_dim + goal_dim] = one_hot
            obs[obs_dim + goal_dim: obs_dim + goal_dim * 2] = env.goal[:]
            # print(obs)
            step_so_far = 0
            done = False
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = env.step(action)
                assert np.argmax(obs[-goal_dim + 3:]) == object_idx
                # img = env.render(mode='rgb_array')
                # ax.cla()
                # ax.imshow(img)
                # ax.set_title('step %d' % step_so_far)
                # plt.savefig(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % step_so_far))
                # plt.pause(0.1)
                step_so_far += 1
                if step_so_far >= horizon // 2:
                    break
            # if not done:
            #     print('Fail to achieve subgoal.')
            # else:
            # No matter if subtask succeeds or not, rushing towards ultimate goal
            if done:
                print('Achieved subgoal in %d steps' % step_so_far)
            assert step_so_far < horizon
            env.goal[:] = ultimate_goal
            # print('ultimate goal', env.goal)
            # obs = env.unwrapped.get_obs()
            obs[obs_dim:obs_dim + 3] = obs[3 * (ultimate_idx + 1):3 * (ultimate_idx + 1) + 3]
            one_hot = np.zeros(2)
            one_hot[ultimate_idx] = 1
            obs[obs_dim + 3:obs_dim + goal_dim] = one_hot
            obs[obs_dim + goal_dim: obs_dim + 2 * goal_dim] = ultimate_goal
            # print(obs)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = env.step(action)
                # assert np.argmax(obs[-goal_dim + 3:]) == 0
                # img = env.render(mode='rgb_array')
                # ax.cla()
                # ax.imshow(img)
                # ax.set_title('step %d' % step_so_far)
                # plt.savefig(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % step_so_far))
                # plt.pause(0.1)
                step_so_far += 1
                if step_so_far >= horizon:
                    break
            if step_so_far < horizon:
                is_success = True
                print('Success!')
            else:
                print('Fail to achieve ultimate goal', ultimate_goal)
        else:
            step_so_far = 0
            done = False
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = env.step(action)
                # assert np.argmax(obs[-goal_dim + 3:]) == 1
                step_so_far += 1
                if step_so_far >= horizon:
                    break
            if done:
                is_success = True
    return is_success

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python visualize_ppo_testtime_switchgoal.py [load_path]')
    load_path = sys.argv[1]
    # Prepare environment and model
    env_name = 'FetchPushWallObstacle-v4'
    env_kwargs = dict(random_box=True,
                      heavy_obstacle=True,
                      random_ratio=0.0,
                      random_gripper=True,
                    )
    # env_name = 'MasspointPushSingleObstacle-v2'
    # env_kwargs = dict(random_box=True,
    #                   random_ratio=0.0,
    #                   random_pusher=True,
    #                   max_episode_steps=200, )
    env = make_env(env_id=env_name, rank=0, log_dir=None, flatten_dict=True, kwargs=env_kwargs)
    goal_dim = env.goal.shape[0]
    obs_dim = env.observation_space.shape[0] - 2 * goal_dim
    noise_mag = env.size_obstacle[1]
    n_object = env.n_object
    horizon = 100
    model = PPO2.load(load_path)
    # We only test the tasks that aim to move
    ppo_success_count = 0
    success_count = 0
    for i in range(1000):
        obs = env.reset()
        # while np.argmax(obs[obs_dim + goal_dim + 3:]) != 0:
        #     obs = env.reset()
        ultimate_goal = obs[-goal_dim:].copy()
        initial_state = env.get_state()
        is_success = testtime_noreduction(env, model, initial_state, ultimate_goal)
        ppo_success_count += int(is_success)
        is_success = testtime_reduction_new(env, model, initial_state, ultimate_goal)
        success_count += int(is_success)
    print('ppo success', ppo_success_count)
    print('reduction success', success_count)
    exit()
    model_idx = int(os.path.basename(load_path).strip('.zip').split('_')[1])
    os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(load_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
              os.path.join(os.path.dirname(load_path), 'testtime_switchgoal_model_%d.mp4' % model_idx))
    for i in range(horizon):
        try:
            os.remove(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % i))
        except:
            pass
