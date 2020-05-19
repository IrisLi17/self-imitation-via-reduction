import sys, os
import numpy as np
from run_her import make_env, get_env_kwargs
from baselines import HER_HACK
from gym.wrappers import FlattenDictWrapper
from utils.parallel_subproc_vec_env2 import ParallelSubprocVecEnv as ParallelSubprocVecEnv2
import matplotlib.pyplot as plt
from baselines import SAC_augment


def no_reduction(env, model, initial_state, ultimate_goal, horizon):
    env.set_state(initial_state)
    env.set_goal(ultimate_goal)
    obs = env.get_obs()
    obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])

    done = False
    step_so_far = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_so_far += 1
        if step_so_far >= horizon:
            break
    return done, step_so_far


def search_subgoal(obs, model):
    sample_obs_buf = []
    subgoal_obs_buf = []
    noise = np.random.uniform(low=-model.model.noise_mag, high=model.model.noise_mag, size=(100, 2))
    sample_obs = np.tile(obs, (noise.shape[0], 1))
    subgoal_obs = np.tile(obs, (noise.shape[0], 1))
    ultimate_idx = np.argmax(obs[model.model.obs_dim + model.model.goal_dim + 3:])
    for object_idx in range(0, model.model.n_object):
        obstacle_xy = np.expand_dims(obs[3 * (object_idx + 1): 3 * (object_idx + 1) + 2], axis=0) + noise
        # Path2
        sample_obs[:, 3 * (object_idx + 1):3 * (object_idx + 1) + 2] = obstacle_xy
        sample_obs[:, 3 * (object_idx + 1 + model.model.n_object):3 * (object_idx + 1 + model.model.n_object) + 2] \
            = sample_obs[:, 3 * (object_idx + 1):3 * (object_idx + 1) + 2] - sample_obs[:, 0:2]
        # achieved_goal
        sample_obs[:, model.model.obs_dim:model.model.obs_dim + 3] \
            = sample_obs[:, 3 * (ultimate_idx + 1):3 * (ultimate_idx + 1) + 3]
        sample_obs_buf.append(sample_obs.copy())

        # Path1
        # achieved_goal
        subgoal_obs[:, model.model.obs_dim:model.model.obs_dim + 3] = subgoal_obs[:, 3 * (object_idx + 1):3 * (object_idx + 1) + 3]
        one_hot = np.zeros(model.model.n_object)
        one_hot[object_idx] = 1
        subgoal_obs[:, model.model.obs_dim + 3:model.model.obs_dim + model.model.goal_dim] = one_hot
        # desired_goal
        subgoal_obs[:, model.model.obs_dim + model.model.goal_dim:model.model.obs_dim + model.model.goal_dim + 2] = obstacle_xy
        subgoal_obs[:, model.model.obs_dim + model.model.goal_dim + 2:model.model.obs_dim + model.model.goal_dim + 3] \
            = subgoal_obs[:, 3 * (object_idx + 1) + 2:3 * (object_idx + 1) + 3]
        subgoal_obs[:, model.model.obs_dim + model.model.goal_dim + 3:model.model.obs_dim + model.model.goal_dim * 2] = one_hot
        subgoal_obs_buf.append(subgoal_obs)

    sample_obs_buf = np.concatenate(sample_obs_buf, axis=0)
    subgoal_obs_buf = np.concatenate(subgoal_obs_buf)

    feed_dict = {model.model.observations_ph: np.concatenate([sample_obs_buf, subgoal_obs_buf], axis=0)}
    _values = np.squeeze(model.model.sess.run(model.model.step_ops[6], feed_dict), axis=-1)
    value2 = _values[:sample_obs_buf.shape[0]]
    value1 = _values[sample_obs_buf.shape[0]:]
    normalize_value1 = (value1 - np.min(value1)) / (np.max(value1) - np.min(value1))
    normalize_value2 = (value2 - np.min(value2)) / (np.max(value2) - np.min(value2))
    ind = np.argsort(normalize_value1 * normalize_value2)
    good_ind = ind[-1]

    mean_value = (value1[good_ind] + value2[good_ind]) / 2
    subgoal = subgoal_obs_buf[good_ind, model.model.obs_dim + model.model.goal_dim:model.model.obs_dim + model.model.goal_dim * 2]
    return subgoal, mean_value, value1[good_ind], value2[good_ind]


def reduction(env, model, initial_state, ultimate_goal, horizon):
    env.set_state(initial_state)
    env.set_goal(ultimate_goal)
    obs = env.get_obs()
    obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
    obs1 = obs.copy()
    obs1[model.model.obs_dim:model.model.obs_dim + model.model.goal_dim] = np.concatenate([obs1[6:9], np.array([0, 1])])
    obs1[model.model.obs_dim + model.model.goal_dim:] = np.concatenate([obs1[6:9], np.array([0, 1])])
    feed_dict = {model.model.observations_ph: np.expand_dims(obs1, axis=0)}
    original_value1 = np.squeeze(model.model.sess.run(model.model.step_ops[6], feed_dict), axis=-1)
    feed_dict = {model.model.observations_ph: np.expand_dims(obs, axis=0)}
    original_value2 = np.squeeze(model.model.sess.run(model.model.step_ops[6], feed_dict), axis=-1)
    original_value = (original_value1 + original_value2) / 2
    subgoal, mean_value, reduced_value1, reduced_value2 = search_subgoal(obs, model)
    # if np.argmax(ultimate_goal[3:]) != 0 or mean_value < original_value:
    # if reduced_value2 < 0.5:
    #     return no_reduction(env, model, initial_state, ultimate_goal, horizon)
    print('original value', original_value1, original_value2,
          'reduced value', reduced_value1, reduced_value2,
          'goal', ultimate_goal, 'subgoal', subgoal)
    done = False
    step_so_far = 0
    # Run towards subgoal
    env.set_goal(subgoal)
    obs = env.get_obs()
    obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_so_far += 1
        if step_so_far >= horizon // 2:
            break
    if not done:
        pass
        # print('Path1 fails')
        # return False, step_so_far
    done = False
    print('Path1 takes', step_so_far, 'steps')
    # Run towards ultimate goal
    env.set_goal(ultimate_goal)
    obs = env.get_obs()
    obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_so_far += 1
        if step_so_far >= horizon:
            break
    if not done:
        print('Path2 fails')
    return done, step_so_far


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0 python testtime_planner_sac.py FetchPushWallObstacle-v4 logs/FetchPushWallObstacle-v4new_random0.7/her_sac_32workers/obj0.15new_exp0.1_priority/model_19.zip
    # (horizon150) random_ratio=0.0, no reduction success 40, reduction success 51.
    # (horizon150) but random_ratio=1.0, no reduction success 55, reduction success 42.
    # CUDA_VISIBLE_DEVICES=0 python testtime_planner_sac.py FetchPushWallObstacle-v4 logs/FetchPushWallObstacle-v4new_random1.0/her_sac_32workers/0_priority/model_20.zip
    # (horizon100) random_ratio=0.0, no reduction success 38, reduction success 81
    # (horizon100) random_ratio=1.0, no reduction success 69, reduction success 63
    env_id = sys.argv[1]
    model_path = sys.argv[2]
    env_kwargs = get_env_kwargs(env_id, random_ratio=0.0)

    def make_thunk(rank):
        return lambda: make_env(env_id=env_id, seed=0, rank=rank, kwargs=env_kwargs)

    env = ParallelSubprocVecEnv2([make_thunk(i) for i in range(1)])

    aug_env_id = env_id.split('-')[0] + 'Unlimit-' + env_id.split('-')[1]
    aug_env_kwargs = env_kwargs.copy()
    aug_env_kwargs['max_episode_steps'] = None

    aug_env = make_env(aug_env_id, seed=0, rank=0, kwargs=aug_env_kwargs)
    aug_env = FlattenDictWrapper(aug_env, ['observation', 'achieved_goal', 'desired_goal'])

    goal_dim = aug_env.goal.shape[0]
    obs_dim = aug_env.observation_space.shape[0] - 2 * goal_dim
    noise_mag = aug_env.size_obstacle[1]
    n_object = aug_env.n_object
    model = HER_HACK.load(model_path, env=env)
    model.model.env_id = env_id
    model.model.goal_dim = goal_dim
    model.model.obs_dim = obs_dim
    model.model.noise_mag = noise_mag
    model.model.n_object = n_object

    count1 = 0
    count2 = 0
    fail1 = [0, 0]
    fail2 = [0, 0]
    for i in range(1000):
        obs = env.reset()
        ultimate_goal = obs['desired_goal'][0]
        initial_state = env.env_method('get_state')[0]
        success1, _ = no_reduction(aug_env, model, initial_state, ultimate_goal, 1. * env_kwargs['max_episode_steps'])
        if not success1:
            print(i, 'Original fail')
            fail1[np.argmax(ultimate_goal[3:])] += 1
        else:
            print(i, 'Original success')
        count1 += int(success1)
        success2, _ = reduction(aug_env, model, initial_state, ultimate_goal, 1. * env_kwargs['max_episode_steps'])
        if not success2:
            fail2[np.argmax(ultimate_goal[3:])] += 1
        else:
            print(i, 'Reduction success')
        count2 += int(success2)
    print('No reduction success', count1)
    print('Reduction success', count2)
    print('Failure detail', fail1, fail2)
