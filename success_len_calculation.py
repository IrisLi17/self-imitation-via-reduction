import sys, os
import numpy as np
from run_her import make_env, get_env_kwargs
from baselines import HER_HACK
from gym.wrappers import FlattenDictWrapper


if __name__ == '__main__':
    env_id = sys.argv[1]
    model_paths = sys.argv[2:]
    env_kwargs = get_env_kwargs(env_id, random_ratio=0.7)

    aug_env_id = env_id.split('-')[0] + 'Unlimit-' + env_id.split('-')[1]
    aug_env_kwargs = env_kwargs.copy()
    aug_env_kwargs['max_episode_steps'] = None

    aug_env = make_env(aug_env_id, seed=0, rank=0, kwargs=aug_env_kwargs)
    aug_env = FlattenDictWrapper(aug_env, ['observation', 'achieved_goal', 'desired_goal'])

    goal_dim = aug_env.goal.shape[0]
    obs_dim = aug_env.observation_space.shape[0] - 2 * goal_dim
    noise_mag = aug_env.size_obstacle[1]
    n_object = aug_env.n_object
    # model.model.env_id = env_id
    # model.model.goal_dim = goal_dim
    # model.model.obs_dim = obs_dim
    # model.model.noise_mag = noise_mag
    # model.model.n_object = n_object

    test_states, test_goals = [], []
    for i in range(100):
        obs = aug_env.reset()
        goal = obs[-goal_dim:]
        initial_state = aug_env.get_state()
        test_states.append(initial_state)
        test_goals.append(goal)
    for model_path in model_paths:
        model = HER_HACK.load(model_path)
        success_len = []
        for i in range(len(test_states)):
            aug_env.set_state(test_states[i])
            aug_env.set_goal(test_goals[i])
            obs = aug_env.get_obs()
            obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
            done = False
            step_so_far = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = aug_env.step(action)
                step_so_far += 1
                if step_so_far >= env_kwargs['max_episode_steps']:
                    break
            if done:
                success_len.append(step_so_far)
        print(model_path, 'mean success len:', np.mean(success_len), 'over %d trajs' % len(success_len))
