from stable_baselines import HER, SAC
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from test_ensemble import make_env
from stable_baselines.common.policies import register_policy
import numpy as np
import matplotlib.pyplot as plt
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

def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='FetchPushWallObstacle-v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--policy', type=str, default='CustomSACPolicy')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_size', type=float, default=1e6)
    parser.add_argument('--num_timesteps', type=float, default=3e6)
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--play', action="store_true", default=False)
    parser.add_argument('--determine_box', action="store_true", default=False)
    parser.add_argument('--heavy_obstacle', action="store_true", default=False)
    parser.add_argument('--random_ratio', type=float, default=1.0)
    parser.add_argument('--random_gripper', action="store_true", default=False)
    parser.add_argument('--reward_offset', type=float, default=1.0)
    parser.add_argument('--hide_velocity', action="store_true", default=False)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(seed, policy, num_timesteps, batch_size, log_path, load_path, play, heavy_obstacle,
         random_gripper, reward_offset, buffer_size, **args):
    log_dir = log_path if (log_path is not None) else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(log_dir)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(log_dir, format_strs=[])

    set_global_seeds(seed)

    model_class = SAC  # works also with SAC, DDPG and TD3

    env_name = args['env']
    env_kwargs = dict(random_box=not args['determine_box'],
                      heavy_obstacle=heavy_obstacle,
                      random_ratio=args['random_ratio'],
                      random_gripper=random_gripper,
                      max_episode_steps=100,)
    env = make_env(env_name, **env_kwargs)

    if not play:
        os.makedirs(log_dir, exist_ok=True)

    # Available strategies (cf paper): future, final, episode, random
    goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE
    her_class = HER

    if not play:
        if model_class is SAC:
            # wrap env
            from utils.wrapper import DoneOnSuccessWrapper
            from stable_baselines.ddpg.noise import NormalActionNoise
            env = DoneOnSuccessWrapper(env, reward_offset=reward_offset)
            env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=True)
            action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=0.1*np.ones(env.action_space.shape))
            train_kwargs = dict(buffer_size=int(buffer_size),
                                batch_size=batch_size,
                                ent_coef="auto",
                                gamma=0.95,
                                learning_starts=1000,
                                train_freq=1,
                                action_noise=action_noise,)
            policy_kwargs = {}

            def callback(_locals, _globals):
                if _locals['step'] % int(1e5) == 0:
                    model_path = os.path.join(log_dir, 'model_' + str(_locals['step'] // int(1e5)))
                    model.save(model_path)
                return True
        else:
            train_kwargs = {}
            policy_kwargs = {}
            callback = None
        if rank == 0:
            print('train_kwargs', train_kwargs)
            print('policy_kwargs', policy_kwargs)

        class CustomSACPolicy(SACPolicy):
            def __init__(self, *args, **kwargs):
                super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                                    layers=[256, 256],
                                                    feature_extraction="mlp")
        register_policy('CustomSACPolicy', CustomSACPolicy)

        # Wrap the model
        model = her_class(policy, env, model_class, n_sampled_goal=4,

                          goal_selection_strategy=goal_selection_strategy,
                          policy_kwargs=policy_kwargs,
                          verbose=1,
                          **train_kwargs)

        # Train the model
        model.learn(int(num_timesteps), seed=seed, callback=callback, log_interval=20)

        if rank == 0:
            model.save(os.path.join(log_dir, 'final'))

    # WARNING: you must pass an env
    # or wrap your environment with HERGoalEnvWrapper to use the predict method
    if play and rank == 0:
        assert load_path is not None
        model = her_class.load(load_path, env=env)
        print(model.get_parameter_list())

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        obs = env.reset()
        print('gripper_pos', obs['observation'][0:3])
        img = env.render(mode='rgb_array')
        episode_reward = 0.0
        images = []
        frame_idx = 0
        episode_idx = 0
        for i in range(6 * env.spec.max_episode_steps):
            images.append(img)
            action, _ = model.predict(obs)

            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            frame_idx += 1
            ax.cla()
            img = env.render(mode='rgb_array')
            ax.imshow(img)
            ax.set_title('episode ' + str(episode_idx) + ', frame ' + str(frame_idx))
            plt.pause(0.05)
            if done:
                obs = env.reset()
                print('gripper_pos', obs['observation'][0:3])
                print('episode_reward', episode_reward)
                episode_reward = 0.0
                frame_idx = 0
                episode_idx += 1
        imageio.mimsave(env_name + '.gif', images)


if __name__ == '__main__':
    args = arg_parse()
    main(**args)
