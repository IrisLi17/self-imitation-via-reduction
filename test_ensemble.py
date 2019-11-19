from stable_baselines import HER, SAC
from baselines import EnsembleSAC, HER_HACK
from baselines.sac.ensemble_value import EnsembleMlpPolicy, EnsembleLnMlpPolicy, EnsembleFeedForwardPolicy
from stable_baselines.common.policies import register_policy
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from push_wall_obstacle import FetchPushWallObstacleEnv,FetchPushWallObstacleEnv_v4
from stable_baselines.common.vec_env import DummyVecEnv
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
               'FetchPushWallObstacle-v4': FetchPushWallObstacleEnv_v4,
               }

hard_test = True

def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='FetchPushWallObstacle-v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--policy', type=str, default='EnsembleMlpPolicy')
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


def make_env(env_name, **kwargs ):
    if env_name in ['FetchReach-v1', 'FetchPush-v1']:
        env = gym.make(env_name)
    elif env_name in ENTRY_POINT.keys():
        # kwargs = dict(penaltize_height=False,
        #               random_box=not args['determine_box'],
        #               heavy_obstacle=args['heavy_obstacle'],
        #               random_ratio=args['random_ratio'],
        #               random_gripper=args['random_gripper'],)
        max_episode_steps = 100 if env_name == 'FetchPushWallObstacle-v4' else 50
        gym.register(env_name, entry_point=ENTRY_POINT[env_name], max_episode_steps=max_episode_steps, kwargs=kwargs)
        env = gym.make(env_name)
    else:
        raise NotImplementedError("%s not implemented" % env_name)
    return env


def main(seed, num_timesteps, batch_size, log_path, load_path, play, heavy_obstacle,
         random_gripper, reward_offset, buffer_size, **args):
    log_dir = log_path if (log_path is not None) else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(log_dir)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(log_dir, format_strs=[])

    set_global_seeds(seed)

    model_class = EnsembleSAC  # works also with SAC, DDPG and TD3

    env_name = args['env']
    env_kwargs = dict(random_box=not args['determine_box'],
                      heavy_obstacle=heavy_obstacle,
                      random_ratio=args['random_ratio'],
                      random_gripper=random_gripper)
    env = make_env(env_name, **env_kwargs)

    if not play:
        os.makedirs(log_dir, exist_ok=True)

    # Available strategies (cf paper): future, final, episode, random
    goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE
    her_class = HER_HACK if env_name == 'FetchPushWallObstacle-v4' else HER

    if not play:
        if model_class is EnsembleSAC:
            # wrap env
            from utils.wrapper import DoneOnSuccessWrapper
            env = DoneOnSuccessWrapper(env, reward_offset=reward_offset)
            env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=True)
            train_kwargs = dict(buffer_size=int(buffer_size),
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

        class EnsembleCustomSACPolicy(EnsembleFeedForwardPolicy):
            def __init__(self, *args, **kwargs):
                super(EnsembleCustomSACPolicy, self).__init__(*args, **kwargs,
                                                              layers=[256, 256],
                                                              feature_extraction="mlp")
        policy_class = EnsembleCustomSACPolicy if args['policy'] == 'EnsembleCustomSACPolicy' else EnsembleMlpPolicy
        # Wrap the model
        model = her_class(policy_class, env, model_class, n_sampled_goal=4,
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
        from stable_baselines.her.utils import KEY_ORDER
        assert load_path is not None
        model = her_class.load(load_path, env=env)
        print(model.get_parameter_list())
        value_ensemble_op = model.model.step_ops[-1]

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
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
        values_ensemble = []
        for i in range(200):
            # images.append(img)
            action, _ = model.predict(obs)
            values = model.model.sess.run(value_ensemble_op,
                                          {model.model.observations_ph: np.expand_dims(np.concatenate([obs[key] for key in KEY_ORDER]), axis=0)})
            # print(values)
            values_ensemble.append(values)
            # print('action', action)
            # print('obstacle euler', obs['observation'][20:23])
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            frame_idx += 1
            ax[0].cla()
            ax[1].cla()
            img = env.render(mode='rgb_array')
            ax[0].imshow(img)
            ax[0].set_title('episode ' + str(episode_idx) + ', frame ' + str(frame_idx))
            np_values = np.squeeze(np.asarray(values_ensemble), axis=(-2, -1))
            # np_values = np.asarray(values_ensemble)
            # print(np_values.shape)
            # exit()
            mean_values = np.mean(np_values, axis=-1)
            std_values = np.std(np_values, axis=-1)
            # print(mean_values.shape)
            # print(std_values.shape)
            ax[1].set_xlim(0, 50)
            ax[1].set_ylim(-5, 20)
            for j in range(np_values.shape[-1]):
                ax[1].plot(np.arange(len(values_ensemble)), np_values[:, j], 'tab:blue', alpha=0.2)
            # ax[1].plot(np.arange(len(values_ensemble)), mean_values, 'tab:blue')
            # ax[1].fill_between(np.arange(len(values_ensemble)), mean_values - std_values, mean_values + std_values, alpha=0.2)
            plt.savefig('tempimg' + str(i) + '.png')
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
                values_ensemble = []
        for i in range(200):
            images.append(plt.imread('tempimg' + str(i) + '.png'))
            os.remove('tempimg' + str(i) + '.png')
        imageio.mimsave(env_name + '.gif', images)


if __name__ == '__main__':
    args = arg_parse()
    # main(env_name=args.env, seed=args.seed, num_timesteps=int(args.num_timesteps), batch_size=args.batch_size,
    #      log_path=args.log_path, load_path=args.load_path, play=args.play, determine_box=args.determine_box,
    #      heavy_obstacle=args.heavy_obstacle, random_ratio=args.random_ratio,
    #      random_gripper=args.random_gripper, reward_offset=args.reward_offset, hide_velocity=args.hide_velocity,
    #      buffer_size=args.buffer_size)
    main(**args)
