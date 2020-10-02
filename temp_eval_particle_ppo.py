import sys, os
# from run_her import make_env, get_env_kwargs
from run_ppo import make_env
from baselines import PPO2
import numpy as np
from gym.wrappers import FlattenDictWrapper
import matplotlib.pyplot as plt


def eval_model(goal_idx, random_ratio, n=50, save_image=False):
    env.unwrapped.random_ratio = random_ratio
    print('Random ratio set to', env.random_ratio)
    success_count = 0
    success_stats = [0] * n_object
    total_stats = [0] * n_object
    for _ in range(n):
        obs = env.reset()
        while not (np.argmax(obs[-goal_dim + 3:]) == goal_idx):
            obs = env.reset()
        # print('goal', obs[-goal_dim:], 'has base', env.has_base)
        agent_pos = obs[:3]
        box_pos = obs[-2 * goal_dim: -2 * goal_dim + 3]
        goal_pos = obs[-goal_dim: -goal_dim + 3]
        n_doors = doors_to_move(agent_pos, box_pos, goal_pos, [obs[6 + 3 * i: 9 + 3 * i] for i in range(n_object - 1)])
        done = False
        frame_idx = 0
        while not done:
            if save_image and n_doors == 3:
                img = env.render(mode='rgb_array')
                plt.imsave("tempimg%d.png" % frame_idx, img)
            action, _ = model.predict(obs)
            obs, _, done, info = env.step(action)
            frame_idx += 1
            success_count += info['is_success']
            success_stats[n_doors] += info['is_success']
        total_stats[n_doors] += 1
        if save_image and n_doors == 3 and info['is_success']:
            exit()
        elif save_image and n_doors == 3:
            print('3 doors fail')
            os.system("rm tempimg*.png")
    return success_count / n, ([success_stats[i] / max(total_stats[i], 1e-4) for i in range(n_object)],
                                total_stats)


def doors_to_move(pos_agent, pos_box, pos_goal, pos_obstacles):
    # TODO:
    max_x, min_x = max(pos_agent[0], pos_box[0], pos_goal[0]), min(pos_agent[0], pos_box[0], pos_goal[0])
    max_n = int(max_x / 1.7)
    min_n = int(min_x / 1.7)
    count = 0
    for pos_obstacle in pos_obstacles:
        if abs(pos_obstacle[1] - 2.5) < 1e-3 and min_n < round(pos_obstacle[0] / 1.7) < max_n + 1:
            count += 1
    return count


if __name__ == '__main__':
    model_path = sys.argv[1]
    n_object = int(sys.argv[2])
    # env_name = 'MasspointPushDoubleObstacle-v1'
    env_name = 'MasspointPushMultiObstacle-v1'
    # env_kwargs = get_env_kwargs(env_name, random_ratio=0.0)
    env_kwargs = dict(random_box=True,
                      random_ratio=0.0,
                      random_pusher=True,
                      max_episode_steps=150,
                      reward_type="sparse",)
    env_kwargs['n_object'] = n_object
    # Manual longer horizon
    env_kwargs['max_episode_steps'] = 50 * n_object
    env = make_env(env_name, seed=None, rank=0, kwargs=env_kwargs)
    # env = FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
    model = PPO2.load(model_path)
    goal_dim = env.goal.shape[0]
    obs_dim = env.observation_space.shape[0] - 2 * goal_dim

    for i in range(env.n_object):
        sr, _ = eval_model(i, 1.0, n=200)
        print('goal idx %d easy' % i, sr)
        sr, stats = eval_model(i, 0.0, n=200)
        print('goal idx %d hard' % i, sr)
        if i == 0:
            print(stats)
