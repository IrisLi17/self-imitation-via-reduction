from run_ppo import make_env
from baselines import PPO2
import sys, os, csv
import numpy as np


def eval_model(goal_idx, random_ratio, n=50):
    env.unwrapped.random_ratio = random_ratio
    # print('Random ratio set to', env.random_ratio)
    success_count = 0
    for _ in range(n):
        obs = env.reset()
        while not (np.argmax(obs[-goal_dim + 3:]) == goal_idx):
            obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, info = env.step(action)
            success_count += info['is_success']
    return success_count / n


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python eval_particle_ppo_long.py [path/to/models/folder] [n_object]')
        exit()
    model_dir = sys.argv[1]
    n_object = int(sys.argv[2])
    # env_name = 'MasspointPushDoubleObstacle-v1'
    env_name = 'MasspointPushMultiObstacle-v1'
    # env_kwargs = get_env_kwargs(env_name, random_ratio=0.0)
    env_kwargs = dict(random_box=True,
                      random_ratio=0.0,
                      random_pusher=True,
                      max_episode_steps=50 * n_object * 4,
                      reward_type="sparse",
                      n_object=n_object,)
    env = make_env(env_name, seed=None, rank=0, kwargs=env_kwargs)
    goal_dim = env.goal.shape[0]
    obs_dim = env.observation_space.shape[0] - 2 * goal_dim
    model = PPO2.load(os.path.join(model_dir, 'model_1.zip'))
    for model_idx in range(1, 100):
        if not os.path.exists(os.path.join(model_dir, 'model_%d.zip' % model_idx)):
            continue
        print('evaluating', os.path.join(model_dir, 'model_%d.zip' % model_idx))
        model.load_parameters(os.path.join(model_dir, 'model_%d.zip' % model_idx))
        sr = eval_model(goal_idx=0, random_ratio=0.0, n=100)
        print('success rate', sr)
        # Log to file
        file_path = os.path.join(model_dir, 'eval_long_horizon.csv')
        if not os.path.exists(file_path):
            with open(file_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
                title = ['n_updates', 'mean_eval_reward']
                csvwriter.writerow(title)
        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            data = [model_idx * 10, sr]
            csvwriter.writerow(data)
