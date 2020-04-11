import sys, os
from run_her import make_env, get_env_kwargs
from baselines import HER_HACK
import numpy as np
from gym.wrappers import FlattenDictWrapper


def eval_model(goal_idx, random_ratio):
    env.unwrapped.random_ratio = random_ratio
    print('Random ratio set to', env.random_ratio)
    success_count = 0
    for _ in range(50):
        obs = env.reset()
        while not (np.argmax(obs[-goal_dim + 3:]) == goal_idx):
            obs = env.reset()
        # print('goal', obs[-goal_dim:], 'has base', env.has_base)
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, info = env.step(action)
            success_count += info['is_success']
    return success_count / 50

if __name__ == '__main__':
    model_path = sys.argv[1]
    env_name = 'MasspointPushDoubleObstacle-v1'
    env_kwargs = get_env_kwargs(env_name, random_ratio=0.0)
    env = make_env(env_name, seed=None, rank=0, kwargs=env_kwargs)
    env = FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
    model = HER_HACK.load(model_path)
    goal_dim = env.goal.shape[0]
    obs_dim = env.observation_space.shape[0] - 2 * goal_dim

    for i in range(env.n_object):
        sr = eval_model(i, 1.0)
        print('goal idx %d easy' % i, sr)
        sr = eval_model(i, 0.0)
        print('goal idx %d hard' % i, sr)
