from stable_baselines import PPO2, logger
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from gym.wrappers import FlattenDictWrapper

from push_obstacle import FetchPushObstacleEnv
from push_wall import FetchPushWallEnv
import gym

import os, time, argparse, imageio
import matplotlib.pyplot as plt

ENTRY_POINT = {'FetchPushObstacle-v1': FetchPushObstacleEnv,
               'FetchPushWall-v1': FetchPushWallEnv,
               }

def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='FetchPushObstacle-v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_timesteps', type=float, default=5e6)
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--play', action="store_true", default=False)
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
    if env_id in ENTRY_POINT.keys():
        env = ENTRY_POINT[env_id](**kwargs)
        from gym.wrappers.time_limit import TimeLimit
        env = TimeLimit(env, max_episode_steps=50)
    else:
        env = gym.make(env_id)
    env = FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
    print('logger dir', rank, log_dir)
    env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=allow_early_resets, info_keywords=('is_success',))
    env.seed(seed + 10000 * rank)
    return env

def main(env_name, seed, num_timesteps, log_path, load_path, play):
    log_dir = log_path if (log_path is not None) else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    configure_logger(log_dir) 
    print('main logger dir', logger.get_dir())
    
    set_global_seeds(seed)

    n_cpu = 8 if not play else 1
    if env_name in ['FetchReach-v1', 'FetchPush-v1']:
        pass
    elif env_name in ['FetchPushObstacle-v1', 'FetchPushObstacleMask-v1', 'FetchPushWall-v1']:
        env_kwargs = dict(penaltize_height=True)
    else:
        raise NotImplementedError("%s not implemented" % env_name)
    env = SubprocVecEnv([lambda : make_env(env_name, seed, i, log_dir=log_dir, kwargs=env_kwargs) for i in range(n_cpu)])

    if not play:
        os.makedirs(log_dir, exist_ok=True)

        policy_kwargs = dict(layers=[64, 64, 64])
        # TODO: vectorize env
        model = PPO2('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
        def callback(_locals, _globals):
            num_update = _locals["update"]
            if num_update % 100 == 0:
                model_path = os.path.join(log_dir, 'model_' + str(num_update))
                model.save(model_path)
                print('model saved to', model_path)
            return True
        model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed, log_interval=10)
        model.save(os.path.join(log_dir, 'final'))
    
    else:
        assert load_path is not None
        model = PPO2.load(load_path)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        obs = env.reset()
        img = env.render(mode='rgb_array')
        episode_reward = 0.0
        images = []
        for _ in range(200):
            images.append(img)
            action, _ = model.predict(obs)
            print('action', action)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            ax.cla()
            img = env.render(mode='rgb_array')
            ax.imshow(img)
            plt.pause(0.1)
            if done:
                # obs = env.reset()
                print('episode_reward', episode_reward)
                episode_reward = 0.0
        imageio.mimsave(env_name + '.gif', images)

if __name__ == '__main__':
    args = arg_parse()
    main(env_name=args.env, seed=args.seed, num_timesteps=int(args.num_timesteps), 
         log_path=args.log_path, load_path=args.load_path, play=args.play)