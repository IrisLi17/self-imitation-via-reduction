import sys, os
from run_ppo import make_env
from baselines import PPO2
import numpy as np


def eval_model(n_obj, is_stack):
    env.unwrapped.random_ratio = 0.0 if is_stack else 1.0
    print('Random ratio set to', env.random_ratio)
    success_count = 0
    for i in range(50):
        obs = env.reset()
        while not (env.unwrapped.current_nobject == n_obj):
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
    env_kwargs = dict(random_box=True,
                      random_ratio=1.0,
                      random_gripper=True,
                      max_episode_steps=100,
                      reward_type="sparse",
                      n_object=2, )
    env = make_env('FetchStack-v1', seed=None, rank=0, kwargs=env_kwargs)
    model = PPO2.load(model_path)
    goal_dim = env.goal.shape[0]
    obs_dim = env.observation_space.shape[0] - 2 * goal_dim
    obj1_pickandplace = eval_model(1, False)
    print('1 obj pick and place', obj1_pickandplace)
    obj1_stack = eval_model(1, True)
    print('1 obj stack', obj1_stack)
    obj2_pickandplace = eval_model(2, False)
    print('2 obj pick and place', obj2_pickandplace)
    obj2_stack = eval_model(2, True)
    print('2 obj stack', obj2_stack)
