from baselines import PPO2_augment
from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from utils.parallel_subproc_vec_env import ParallelSubprocVecEnv
from gym.wrappers import FlattenDictWrapper

from push_wall_obstacle import FetchPushWallObstacleEnv_v4
from masspoint_env import MasspointPushSingleObstacleEnv_v2, MasspointPushDoubleObstacleEnv
from fetch_stack import FetchStackEnv
# from push_wall import FetchPushWallEnv
# from push_box import FetchPushBoxEnv
import gym
from utils.wrapper import DoneOnSuccessWrapper
import numpy as np
import csv, pickle

import os, time, argparse, imageio
import matplotlib.pyplot as plt

ENTRY_POINT = {'FetchPushWallObstacle-v4': FetchPushWallObstacleEnv_v4,
               'FetchPushWallObstacleUnlimit-v4': FetchPushWallObstacleEnv_v4,
               # 'FetchPushWall-v1': FetchPushWallEnv,
               # 'FetchPushBox-v1': FetchPushBoxEnv,
               }
MASS_ENTRY_POINT = {
    'MasspointPushSingleObstacle-v2': MasspointPushSingleObstacleEnv_v2,
    'MasspointPushSingleObstacleUnlimit-v2': MasspointPushSingleObstacleEnv_v2,
    'MasspointPushDoubleObstacle-v1': MasspointPushDoubleObstacleEnv,
    'MasspointPushDoubleObstacleUnlimit-v1': MasspointPushDoubleObstacleEnv,
}

PICK_ENTRY_POINT = {
    'FetchStack-v1': FetchStackEnv,
    'FetchStackUnlimit-v1': FetchStackEnv,
}

def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='FetchPushWallObstacle-v4')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_timesteps', type=float, default=1e8)
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--random_ratio', default=1.0, type=float)
    parser.add_argument('--aug_clip', default=0.1, type=float)
    parser.add_argument('--n_subgoal', default=4, type=int)
    parser.add_argument('--parallel', action="store_true", default=False)
    parser.add_argument('--start_augment', type=float, default=0)
    parser.add_argument('--reuse_times', default=1, type=int)
    parser.add_argument('--play', action="store_true", default=False)
    parser.add_argument('--export_gif', action="store_true", default=False)
    args = parser.parse_args()
    return args


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def make_env(env_id, seed, rank, log_dir=None, allow_early_resets=True, kwargs=None):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The mujoco environment
    """
    if env_id in ENTRY_POINT.keys() or env_id in MASS_ENTRY_POINT.keys() or env_id in PICK_ENTRY_POINT.keys():
        # env = ENTRY_POINT[env_id](**kwargs)
        # print(env)
        # from gym.wrappers.time_limit import TimeLimit
        kwargs = kwargs.copy()
        max_episode_steps = None
        if 'max_episode_steps' in kwargs:
            max_episode_steps = kwargs['max_episode_steps']
            del kwargs['max_episode_steps']
        if env_id in ENTRY_POINT.keys():
            gym.register(env_id, entry_point=ENTRY_POINT[env_id], max_episode_steps=max_episode_steps, kwargs=kwargs)
        elif env_id in MASS_ENTRY_POINT.keys():
            gym.register(env_id, entry_point=MASS_ENTRY_POINT[env_id], max_episode_steps=max_episode_steps, kwargs=kwargs)
        elif env_id in PICK_ENTRY_POINT.keys():
            gym.register(env_id, entry_point=PICK_ENTRY_POINT[env_id], max_episode_steps=max_episode_steps,
                         kwargs=kwargs)
        env = gym.make(env_id)
        # env = TimeLimit(env, max_episode_steps=50)
    else:
        env = gym.make(env_id, reward_type='sparse')
    env = FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
    env = DoneOnSuccessWrapper(env)
    if log_dir is not None:
        env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=allow_early_resets,
                      info_keywords=('is_success',))
    # env.seed(seed + 10000 * rank)
    return env


def eval_model(eval_env, model):
    env = eval_env
    assert abs(env.unwrapped.random_ratio) < 1e-4
    n_episode = 0
    ep_rewards = []
    while n_episode < 20:
        ep_reward = 0.0
        obs = env.reset()
        goal_dim = env.goal.shape[0]
        if goal_dim > 3:
            while (np.argmax(obs[-goal_dim + 3:]) != 0):
                obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        ep_rewards.append(ep_reward)
        n_episode += 1
    return np.mean(ep_rewards)


def log_eval(num_update, mean_eval_reward):
    if not os.path.exists(os.path.join(logger.get_dir(), 'eval.csv')):
        with open(os.path.join(logger.get_dir(), 'eval.csv'), 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            title = ['n_updates', 'mean_eval_reward']
            csvwriter.writerow(title)
    with open(os.path.join(logger.get_dir(), 'eval.csv'), 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
        data = [num_update, mean_eval_reward]
        csvwriter.writerow(data)


def log_traj(aug_obs, aug_done, index, goal_dim=5, n_obstacle=1):
    with open(os.path.join(logger.get_dir(), 'success_traj_%d.csv' % index), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
        title = ['gripper_x', 'gripper_y', 'gripper_z', 'box_x', 'box_y', 'box_z',
                 'obstacle_x', 'obstacle_y', 'obstacle_z']
        for i in range(1, n_obstacle):
            title += ['obstacle' + str(i) + '_x', 'obstacle' + str(i) + '_y', 'obstacle' + str(i) + '_z']
        title += ['goal_' + str(i) for i in range(goal_dim)] + ['done']
        csvwriter.writerow(title)
        for idx in range(aug_obs.shape[0]):
            log_obs = aug_obs[idx]
            data = [log_obs[i] for i in range(6 + 3 * n_obstacle)] + [log_obs[i] for i in range(-goal_dim, 0)] + [aug_done[idx]]
            csvwriter.writerow(data)


def main(env_name, seed, num_timesteps, log_path, load_path, play, export_gif, random_ratio, aug_clip, n_subgoal,
         parallel, start_augment, reuse_times):
    log_dir = log_path if (log_path is not None) else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    configure_logger(log_dir)

    set_global_seeds(seed)

    n_cpu = 32 if not play else 1
    if 'MasspointPushDoubleObstacle' in env_name:
        n_cpu = 64
    if env_name in ['FetchReach-v1', 'FetchPush-v1', 'CartPole-v1']:
        env_kwargs = dict(reward_type='dense')
        # pass
    elif env_name in ENTRY_POINT.keys():
        env_kwargs = dict(random_box=True,
                          heavy_obstacle=True,
                          random_ratio=random_ratio,
                          random_gripper=True,
                          max_episode_steps=100,)
    elif env_name in MASS_ENTRY_POINT.keys():
        env_kwargs = dict(random_box=True,
                          random_ratio=random_ratio,
                          random_pusher=True,
                          max_episode_steps=100,)
        if 'MasspointPushSingleObstacle' in env_name:
            env_kwargs['max_episode_steps']=200
        if 'MasspointPushDoubleObstacle' in env_name:
            env_kwargs['max_episode_steps']=150
    elif env_name in PICK_ENTRY_POINT.keys():
        env_kwargs = dict(random_box=True,
                          random_ratio=random_ratio,
                          random_gripper=True,
                          max_episode_steps=150, )
    else:
        raise NotImplementedError("%s not implemented" % env_name)

    def make_thunk(rank):
        return lambda: make_env(env_id=env_name, seed=seed, rank=rank, log_dir=log_dir, kwargs=env_kwargs)

    # if not parallel:
    env = SubprocVecEnv([make_thunk(i) for i in range(n_cpu)])
    # else:
    #     env = ParallelSubprocVecEnv([make_thunk(i) for i in range(n_cpu)])
    aug_env_name = env_name.split('-')[0] + 'Unlimit-' + env_name.split('-')[1]
    aug_env_kwargs = env_kwargs.copy()
    aug_env_kwargs['max_episode_steps'] = None
    def make_thunk_aug(rank):
        return lambda: make_env(env_id=aug_env_name, seed=seed, rank=rank, kwargs=aug_env_kwargs)
    if not parallel:
        aug_env = make_env(env_id=aug_env_name, seed=seed, rank=0, kwargs=aug_env_kwargs)
    else:
        # aug_env = ParallelSubprocVecEnv([make_thunk_aug(i) for i in range(n_subgoal)])
        aug_env = ParallelSubprocVecEnv([make_thunk_aug(i) for i in range(32)])
    print(aug_env)
    if os.path.exists(os.path.join(logger.get_dir(), 'eval.csv')):
        os.remove(os.path.join(logger.get_dir(), 'eval.csv'))
        print('Remove existing eval.csv')
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['random_ratio'] = 0.0
    eval_env = make_env(env_id=env_name, seed=seed, rank=0, kwargs=eval_env_kwargs)
    print(eval_env)
    print(eval_env.goal.shape[0], eval_env.n_object)
    if not play:
        os.makedirs(log_dir, exist_ok=True)
        policy_kwargs = dict(layers=[256, 256])
        # policy_kwargs = {}
        # TODO: vectorize env
        if 'MasspointPushDoubleObstacle' in env_name:
            n_steps = 8192
        else:
            n_steps = 2048
        model = PPO2_augment('MlpPolicy', env, aug_env=aug_env, verbose=1, n_steps=n_steps, nminibatches=32, lam=0.95,
                             gamma=0.99, noptepochs=10, ent_coef=0.01, aug_clip=aug_clip, learning_rate=3e-4,
                             cliprange=0.2, n_candidate=n_subgoal, parallel=parallel, start_augment=start_augment,
                             policy_kwargs=policy_kwargs, horizon=env_kwargs['max_episode_steps'],
                             reuse_times=reuse_times,
                             )

        def callback(_locals, _globals):
            num_update = _locals["update"]
            mean_eval_reward = eval_model(eval_env, _locals["self"])
            log_eval(num_update, mean_eval_reward)
            aug_obs = _locals["self"].aug_obs
            aug_done = _locals["self"].aug_done
            aug_obs = list(filter(lambda v:v is not None, aug_obs))
            aug_done = list(filter(lambda v:v is not None, aug_done))
            if len(aug_obs):
                aug_obs = np.concatenate(aug_obs, axis=0)
                aug_done = np.concatenate(aug_done, axis=0)
                log_traj(aug_obs, aug_done, num_update, goal_dim=eval_env.goal.shape[0],
                         n_obstacle=eval_env.n_object)
            if num_update % 10 == 0:
                model_path = os.path.join(log_dir, 'model_' + str(num_update // 10))
                model.save(model_path)
                print('model saved to', model_path)
            return True

        # For debug only.
        # model.load_parameters('./logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo/0/model_70.zip')
        # model.load_parameters('./logs/MasspointPushDoubleObstacle-v1/ppo/6/model_71.zip')
        model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed, log_interval=1)
        model.save(os.path.join(log_dir, 'final'))

    else:
        assert load_path is not None
        model = PPO2_augment.load(load_path)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        obs = env.reset()
        while (np.argmax(obs[0][-2:]) != 0):
            obs = env.reset()
        # img = env.render(mode='rgb_array')
        episode_reward = 0.0
        num_episode = 0
        frame_idx = 0
        images = []
        for i in range(500):
            img = env.render(mode='rgb_array')
            images.append(img)
            ax.cla()
            ax.imshow(img)
            ax.set_title('episode ' + str(num_episode) + ', frame ' + str(frame_idx) +
                         ', goal idx ' + str(np.argmax(obs[0][-2:])))
            assert np.argmax(obs[0][-2:]) == 0
            action, _ = model.predict(obs)
            print('action', action)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            frame_idx += 1
            if not export_gif:
                plt.pause(0.1)
            else:
                plt.savefig(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % i))
            if done:
                # obs = env.reset()
                while (np.argmax(obs[0][-2:]) != 0):
                    obs = env.reset()
                print('episode_reward', episode_reward)
                episode_reward = 0.0
                frame_idx = 0
                num_episode += 1
                if num_episode >= 5:
                    break
        # imageio.mimsave(env_name + '.gif', images)
        if export_gif:
            os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(
                load_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
                      os.path.join(os.path.dirname(load_path), env_name + '.mp4'))
            for i in range(500):
                # images.append(plt.imread('tempimg' + str(i) + '.png'))
                try:
                    os.remove(os.path.join(os.path.dirname(load_path), 'tempimg' + str(i) + '.png'))
                except:
                    pass


if __name__ == '__main__':
    args = arg_parse()
    print('arg parsed')
    main(env_name=args.env, seed=args.seed, num_timesteps=int(args.num_timesteps),
         log_path=args.log_path, load_path=args.load_path, play=args.play, export_gif=args.export_gif,
         random_ratio=args.random_ratio, aug_clip=args.aug_clip, n_subgoal=args.n_subgoal,
         parallel=args.parallel, start_augment=int(args.start_augment), reuse_times=args.reuse_times)
