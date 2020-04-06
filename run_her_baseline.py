from stable_baselines import SAC, HER
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.common.policies import register_policy
from run_her import make_env
import matplotlib.pyplot as plt
from stable_baselines.common import set_global_seeds
from stable_baselines import logger
import os, time
import imageio
import argparse
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='FetchPushWallObstacle-v4')
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--policy', type=str, default='MlpPolicy')
    parser.add_argument('--action_noise', type=str, default='none')
    parser.add_argument('--num_timesteps', type=float, default=3e6)
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--play', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--random_ratio', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--reward_type', type=str, default='sparse')
    parser.add_argument('--n_object', type=int, default=2)
    parser.add_argument('--export_gif', action="store_true", default=False)
    args = parser.parse_args()
    return args


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(env_name, seed, num_timesteps, batch_size, log_path, load_path, play,
         export_gif, gamma, random_ratio, action_noise, reward_type, n_object):
    log_dir = log_path if (log_path is not None) else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(log_dir)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(log_dir, format_strs=[])

    set_global_seeds(seed)

    model_class = SAC  # works also with SAC, DDPG and TD3

    # if env_name in ENTRY_POINT.keys():
    #     kwargs = dict(penaltize_height=False, heavy_obstacle=heavy_obstacle, random_gripper=random_gripper)
    #     print(kwargs)
    #     max_episode_steps = 100 if env_name == 'FetchPushWallObstacle-v4' else 50
    #     gym.register(env_name, entry_point=ENTRY_POINT[env_name], max_episode_steps=max_episode_steps, kwargs=kwargs)
    #     env = gym.make(env_name)
    # else:
    #     raise NotImplementedError("%s not implemented" % env_name)
    env_kwargs = dict(random_box=True,
                      random_ratio=random_ratio,
                      random_gripper=True,
                      max_episode_steps=100,
                      reward_type=reward_type,
                      n_object=n_object, )
    env = make_env(env_id=env_name, seed=seed, rank=rank, log_dir=log_dir, kwargs=env_kwargs)

    if not play:
        os.makedirs(log_dir, exist_ok=True)

    # Available strategies (cf paper): future, final, episode, random
    goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE

    if not play:
        if model_class is SAC:
            # wrap env
            # from utils.wrapper import DoneOnSuccessWrapper
            from stable_baselines.ddpg.noise import NormalActionNoise
            # env = DoneOnSuccessWrapper(env)
            # env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=True)
            noise_type = action_noise.split('_')[0]
            if noise_type == 'none':
                parsed_action_noise = None
            elif noise_type == 'normal':
                sigma = float(action_noise.split('_')[1])
                parsed_action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape),
                                                        sigma=sigma * np.ones(env.action_space.shape))
            else:
                raise NotImplementedError
            train_kwargs = dict(buffer_size=int(1e5),
                                ent_coef="auto",
                                gamma=gamma,
                                learning_starts=1000,
                                train_freq=1,
                                batch_size=batch_size,
                                action_noise=parsed_action_noise,
                                )
            if 'FetchStack' in env_name:
                train_kwargs['tau'] = 0.001
                train_kwargs['gamma'] = 0.98
                train_kwargs['batch_size'] = 256
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
        class CustomSACPolicy(SACPolicy):
            def __init__(self, *args, **kwargs):
                super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                                    layers=[256, 256, 256, 256],
                                                    feature_extraction="mlp")
        register_policy('CustomSACPolicy', CustomSACPolicy)
        policy = CustomSACPolicy
        policy_kwargs["layer_norm"] = True

        if rank == 0:
            print('train_kwargs', train_kwargs)
            print('policy_kwargs', policy_kwargs)
        # Wrap the model
        model = HER(policy, env, model_class, n_sampled_goal=4,
                    goal_selection_strategy=goal_selection_strategy,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    **train_kwargs)
        print(model.get_parameter_list())

        # Train the model
        model.learn(num_timesteps, seed=seed, callback=callback, log_interval=100)

        if rank == 0:
            model.save(os.path.join(log_dir, 'final'))

    # WARNING: you must pass an env
    # or wrap your environment with HERGoalEnvWrapper to use the predict method
    if play and rank == 0:
        assert load_path is not None
        model = HER.load(load_path, env=env)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        obs = env.reset()
        # sim_state = env.sim.get_state()
        # print(sim_state)
        # while not (obs['desired_goal'][0] < env.pos_wall[0] < obs['achieved_goal'][0] or
        #             obs['desired_goal'][0] > env.pos_wall[0] > obs['achieved_goal'][0]):
        #     if not hard_test:
        #         break
        #     obs = env.reset()
        print('gripper_pos', obs['observation'][0:3])
        img = env.render(mode='rgb_array')
        episode_reward = 0.0
        images = []
        frame_idx = 0
        episode_idx = 0
        for i in range(env.spec.max_episode_steps * 6):
            # images.append(img)
            action, _ = model.predict(obs)
            # print('action', action)
            # print('obstacle euler', obs['observation'][20:23])
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            frame_idx += 1
            ax.cla()
            img = env.render(mode='rgb_array')
            ax.imshow(img)
            ax.set_title('episode ' + str(episode_idx) + ', frame ' + str(frame_idx) +
                         ', goal idx ' + str(np.argmax(obs['desired_goal'][3:])))
            if export_gif:
                plt.savefig('tempimg' + str(i) + '.png')
            plt.pause(0.02)
            if done:
                obs = env.reset()
                # while not (obs['desired_goal'][0] < env.pos_wall[0] < obs['achieved_goal'][0] or
                #             obs['desired_goal'][0] > env.pos_wall[0] > obs['achieved_goal'][0]):
                #     if not hard_test:
                #         break
                #     obs = env.reset()
                print('gripper_pos', obs['observation'][0:3])
                print('episode_reward', episode_reward)
                episode_reward = 0.0
                frame_idx = 0
                episode_idx += 1
        if export_gif:
            for i in range(env.spec.max_episode_steps * 6):
                images.append(plt.imread('tempimg' + str(i) + '.png'))
                os.remove('tempimg' + str(i) + '.png')
            imageio.mimsave(env_name + '.gif', images)


if __name__ == '__main__':
    args = arg_parse()
    main(env_name=args.env, seed=args.seed, num_timesteps=int(args.num_timesteps),
         log_path=args.log_path, load_path=args.load_path, play=args.play,
         batch_size=args.batch_size, export_gif=args.export_gif,
         gamma=args.gamma, random_ratio=args.random_ratio, action_noise=args.action_noise,
         reward_type=args.reward_type, n_object=args.n_object)