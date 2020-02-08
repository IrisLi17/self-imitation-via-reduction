import sys, os
from run_ppo import make_env
from baselines import PPO2
import numpy as np


def eval_model(goal_idx, is_hard):
    env.unwrapped.random_ratio = 0.0 if is_hard else 1.0
    print('Random ratio set to', env.random_ratio)
    success_count = 0
    for i in range(20):
        obs = env.reset()
        while not (np.argmax(obs[-goal_dim + 3:]) == goal_idx):
            obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, info = env.step(action)
            success_count += info['is_success']
    return success_count / 20

if __name__ == '__main__':
    model_path = sys.argv[1]
    env_kwargs = dict(random_box=True,
                      random_ratio=1.0,
                      random_pusher=True,
                      max_episode_steps=150, )
    env = make_env('MasspointPushDoubleObstacle-v1', seed=None, rank=0, kwargs=env_kwargs)
    model = PPO2.load(model_path)
    goal_dim = env.goal.shape[0]
    obs_dim = env.observation_space.shape[0] - 2 * goal_dim
    obj0_easy = eval_model(0, False)
    print('obj0 easy', obj0_easy)
    obj1_easy = eval_model(1, False)
    print('obj1 easy', obj1_easy)
    obj2_easy = eval_model(2, False)
    print('obj2 easy', obj2_easy)
    obj0_hard = eval_model(0, True)
    print('obj0 hard', obj0_hard)
