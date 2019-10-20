from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.policies import MlpPolicy
from push_obstacle import FetchPushObstacleEnv
from push_wall import FetchPushWallEnv
from push_box import  FetchPushBoxEnv
import gym
import matplotlib.pyplot as plt
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines import logger
import os, time
import imageio
import argparse
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

ENTRY_POINT = {'FetchPushObstacle-v1': FetchPushObstacleEnv,
               'FetchPushWall-v1': FetchPushWallEnv,
               'FetchPushBox-v1': FetchPushBoxEnv,
               }

def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='FetchReach-v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_timesteps', type=float, default=3e6)
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


def main(env_name, seed, num_timesteps, log_path, load_path, play):
    log_dir = log_path if (log_path is not None) else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(log_dir)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(log_dir, format_strs=[])   
    
    set_global_seeds(seed)

    model_class = SAC  # works also with SAC, DDPG and TD3

    if env_name in ['FetchReach-v1', 'FetchPush-v1']:
        env = gym.make(env_name)
    elif env_name in ENTRY_POINT.keys():
        gym.register(env_name, entry_point=ENTRY_POINT[env_name], max_episode_steps=50, kwargs=dict(penaltize_height=False))
        env = gym.make(env_name)
    else:
        raise NotImplementedError("%s not implemented" % env_name)

    if not play:
        os.makedirs(log_dir, exist_ok=True)

    # Available strategies (cf paper): future, final, episode, random
    goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

    if not play:
        if model_class is DDPG:
            env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=True)
            train_kwargs = dict(action_noise=NormalActionNoise(mean=0.0 * env.action_space.high, sigma=0.05 * env.action_space.high),
                                normalize_observations=True,
                                random_exploration=0.2,
                                nb_rollout_steps=16*50, nb_train_steps=40,
                                buffer_size=int(1e5),
                                actor_lr=1e-3, critic_lr=1e-3,
                                gamma=0.98,
                                batch_size=128,
                                )
            if env_name in ["FetchPush-v1", "FetchPushWall-v1", "FetchPushObstacle-v1"]:
                policy_kwargs = dict(layers=[64, 64, 64])
            else:
                policy_kwargs = {}
            def callback(_locals, _globals):
                steps_per_epoch = train_kwargs["nb_rollout_steps"] * 50
                if rank ==0 and _locals['total_steps'] % (steps_per_epoch * 5) == 0:
                    model_path = os.path.join(log_dir, 'model_' + str(_locals['total_steps'] // (steps_per_epoch)))
                    model.save(model_path)
                    print('model saved to', model_path)
                return True
        elif model_class is SAC:
            # wrap env
            from utils.wrapper import DoneOnSuccessWrapper
            env = DoneOnSuccessWrapper(env)
            env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=True)
            train_kwargs = dict(buffer_size=int(1e6),
                                ent_coef="auto",
                                gamma=0.95,
                                learning_starts=1000,
                                train_freq=1,)
            policy_kwargs = {}
            def callback(_locals, _globals):
                if _locals['step'] % int(1e4) == 0:
                    model_path = os.path.join(log_dir, 'model_' + str(_locals['step']))
                    model.save(model_path)
                return True
        else:
            train_kwargs = {}
            policy_kwargs = {}
            callback = None
        if rank == 0:
            print('train_kwargs', train_kwargs)
            print('policy_kwargs', policy_kwargs)
        # Wrap the model
        model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                    policy_kwargs=policy_kwargs, 
                    verbose=1,
                    **train_kwargs)

        # Train the model
        model.learn(num_timesteps, seed=seed, callback=callback, log_interval=20)

        if rank == 0:
            model.save(os.path.join(log_dir, 'final'))

    # WARNING: you must pass an env
    # or wrap your environment with HERGoalEnvWrapper to use the predict method
    if play and rank == 0:
        assert load_path is not None
        model = HER.load(load_path, env=env)

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
            plt.pause(0.2)
            if done:
                obs = env.reset()
                print('episode_reward', episode_reward)
                episode_reward = 0.0
        imageio.mimsave(env_name + '.gif', images)

if __name__ == '__main__':
    args = arg_parse()
    main(env_name=args.env, seed=args.seed, num_timesteps=int(args.num_timesteps), 
         log_path=args.log_path, load_path=args.load_path, play=args.play)
