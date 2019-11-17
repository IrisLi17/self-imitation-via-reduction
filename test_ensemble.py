from stable_baselines import HER, SAC
from baselines import EnsembleSAC
from baselines.sac.ensemble_value import EnsembleMlpPolicy, EnsembleLnMlpPolicy
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from push_wall_obstacle import FetchPushWallObstacleEnv
import gym
import numpy as np
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


ENTRY_POINT = {'FetchPushWallObstacle-v1': FetchPushWallObstacleEnv,
               }

hard_test = True

def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='FetchPushWallObstacle-v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
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
    return args


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(env_name, seed, num_timesteps, batch_size, log_path, load_path, play, determine_box, heavy_obstacle,
         random_ratio, random_gripper, reward_offset, hide_velocity):
    log_dir = log_path if (log_path is not None) else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(log_dir)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(log_dir, format_strs=[])

    set_global_seeds(seed)

    model_class = EnsembleSAC  # works also with SAC, DDPG and TD3

    if env_name in ['FetchReach-v1', 'FetchPush-v1']:
        env = gym.make(env_name)
    elif env_name in ENTRY_POINT.keys():
        kwargs = dict(penaltize_height=False,
                      random_box=True,
                      heavy_obstacle=heavy_obstacle,
                      random_ratio=1.0,
                      random_gripper=random_gripper,)
        gym.register(env_name, entry_point=ENTRY_POINT[env_name], max_episode_steps=50, kwargs=kwargs)
        env = gym.make(env_name)
    else:
        raise NotImplementedError("%s not implemented" % env_name)

    if not play:
        os.makedirs(log_dir, exist_ok=True)

    # Available strategies (cf paper): future, final, episode, random
    goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE

    if not play:
        if model_class is EnsembleSAC:
            # wrap env
            from utils.wrapper import DoneOnSuccessWrapper
            env = DoneOnSuccessWrapper(env, reward_offset=reward_offset)
            env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=True)
            train_kwargs = dict(buffer_size=int(1e6),
                                batch_size=batch_size,
                                ent_coef="auto",
                                gamma=0.95,
                                learning_starts=1000,
                                train_freq=1, )
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
        # Wrap the model
        if load_path is None:
            model = HER(EnsembleMlpPolicy, env, model_class, n_sampled_goal=4,
                        goal_selection_strategy=goal_selection_strategy,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        **train_kwargs)
        else:
            # I want to continue training here.
            model = HER.load(load_path, env=env)

        # Train the model
        model.learn(num_timesteps, seed=seed, callback=callback, log_interval=20)

        if rank == 0:
            model.save(os.path.join(log_dir, 'final'))

    # WARNING: you must pass an env
    # or wrap your environment with HERGoalEnvWrapper to use the predict method
    if play and rank == 0:
        from stable_baselines.her.utils import KEY_ORDER
        assert load_path is not None
        model = HER.load(load_path, env=env)
        value_ensemble = model.model.step_ops[-1]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        if env_name in ['FetchPushWall-v1']:
            obs = env.reset()
            while (obs['achieved_goal'][0] - env.pos_wall[0]) * (obs['desired_goal'][0] - env.pos_wall[0]) > 0 \
                    or (obs['desired_goal'][1] - 0.65) * (obs['desired_goal'][1] - 0.85) < 0 \
                    or (obs['achieved_goal'][1] - 0.75) * (obs['desired_goal'][1] - 0.75) < 0:
                obs = env.reset()
        elif env_name in ['FetchPushWallObstacle-v1']:
            obs = env.reset()
            while not (obs['desired_goal'][0] < env.pos_wall[0] < obs['achieved_goal'][0] or \
                                       obs['desired_goal'][0] > env.pos_wall[0] > obs['achieved_goal'][0]):
                if not hard_test:
                    break
                obs = env.reset()
        else:
            obs = env.reset()
        print('gripper_pos', obs['observation'][0:3])
        img = env.render(mode='rgb_array')
        episode_reward = 0.0
        images = []
        frame_idx = 0
        episode_idx = 0
        for _ in range(250):
            images.append(img)
            action, _ = model.predict(obs)
            values = model.model.sess.run(value_ensemble,
                                          {model.model.observations_ph: np.expand_dims(np.concatenate([obs[key] for key in KEY_ORDER]), axis=0)})
            print(values)
            # print('action', action)
            # print('obstacle euler', obs['observation'][20:23])
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            frame_idx += 1
            ax.cla()
            img = env.render(mode='rgb_array')
            ax.imshow(img)
            ax.set_title('episode ' + str(episode_idx) + ', frame ' + str(frame_idx))
            plt.pause(0.05)
            if done:
                if env_name in ['FetchPushWall-v1']:
                    obs = env.reset()
                    while (obs['achieved_goal'][0] - env.pos_wall[0]) * (obs['desired_goal'][0] - env.pos_wall[0]) > 0 \
                            or (obs['desired_goal'][1] - 0.65) * (obs['desired_goal'][1] - 0.85) < 0 \
                            or (obs['achieved_goal'][1] - 0.75) * (obs['desired_goal'][1] - 0.75) < 0:
                        obs = env.reset()
                elif env_name in ['FetchPushWallObstacle-v1']:
                    obs = env.reset()
                    while not (obs['desired_goal'][0] < env.pos_wall[0] < obs['achieved_goal'][0] or \
                                               obs['desired_goal'][0] > env.pos_wall[0] > obs['achieved_goal'][0]):
                        if not hard_test:
                            break
                        obs = env.reset()
                else:
                    obs = env.reset()
                print('gripper_pos', obs['observation'][0:3])
                print('episode_reward', episode_reward)
                episode_reward = 0.0
                frame_idx = 0
                episode_idx += 1
        imageio.mimsave(env_name + '.gif', images)


if __name__ == '__main__':
    args = arg_parse()
    main(env_name=args.env, seed=args.seed, num_timesteps=int(args.num_timesteps), batch_size=args.batch_size,
         log_path=args.log_path, load_path=args.load_path, play=args.play, determine_box=args.determine_box,
         heavy_obstacle=args.heavy_obstacle, random_ratio=args.random_ratio,
         random_gripper=args.random_gripper, reward_offset=args.reward_offset, hide_velocity=args.hide_velocity)
