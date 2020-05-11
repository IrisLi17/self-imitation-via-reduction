from baselines import HER_HACK, SAC_augment
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.common.policies import register_policy
from utils.parallel_subproc_vec_env import ParallelSubprocVecEnv
from utils.parallel_subproc_vec_env2 import ParallelSubprocVecEnv as ParallelSubprocVecEnv2
from gym.wrappers import FlattenDictWrapper
import gym
import matplotlib.pyplot as plt
from stable_baselines.common import set_global_seeds
from stable_baselines import logger
from run_her import make_env, get_env_kwargs
import os, time
import imageio
import argparse
import numpy as np
from run_ppo_augment import stack_eval_model, eval_model, log_eval, egonav_eval_model

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


hard_test = False


def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='FetchPushWallObstacle-v4')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--policy', type=str, default='CustomSACPolicy')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
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
    parser.add_argument('--start_augment', type=float, default=0)
    parser.add_argument('--priority', action="store_true", default=False)
    parser.add_argument('--curriculum', action="store_true", default=False)
    parser.add_argument('--imitation_coef', type=float, default=5)
    parser.add_argument('--sequential', action="store_true", default=False)
    parser.add_argument('--export_gif', action="store_true", default=False)
    args = parser.parse_args()
    return args


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(env_name, seed, num_timesteps, batch_size, log_path, load_path, play,
         export_gif, gamma, random_ratio, action_noise, reward_type, n_object, start_augment,
         policy, learning_rate, n_workers, priority, curriculum, imitation_coef, sequential):
    assert n_workers > 1
    log_dir = log_path if (log_path is not None) else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(log_dir)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(log_dir, format_strs=[])

    set_global_seeds(seed)

    model_class = SAC_augment  # works also with SAC, DDPG and TD3

    # env_kwargs = dict(random_box=True,
    #                   random_ratio=random_ratio,
    #                   random_gripper=True,
    #                   # max_episode_steps=50 * n_object if n_object > 3 else 100,
    #                   max_episode_steps=None if sequential else 100,
    #                   reward_type=reward_type,
    #                   n_object=n_object, )
    env_kwargs = get_env_kwargs(env_name, random_ratio=random_ratio, sequential=sequential,
                                reward_type=reward_type, n_object=n_object)

    def make_thunk(rank):
        return lambda: make_env(env_id=env_name, seed=seed, rank=rank, log_dir=log_dir, kwargs=env_kwargs)

    env = ParallelSubprocVecEnv2([make_thunk(i) for i in range(n_workers)])
    # if n_workers > 1:
    #     # env = SubprocVecEnv([make_thunk(i) for i in range(n_workers)])
    # else:
    #     env = make_env(env_id=env_name, seed=seed, rank=rank, log_dir=log_dir, kwargs=env_kwargs)

    def make_thunk_aug(rank):
        return lambda: FlattenDictWrapper(make_env(env_id=aug_env_name, seed=seed, rank=rank, kwargs=aug_env_kwargs),
                                          ['observation', 'achieved_goal', 'desired_goal'])

    aug_env_kwargs = env_kwargs.copy()
    del aug_env_kwargs['max_episode_steps']
    aug_env_name = env_name.split('-')[0] + 'Unlimit-' + env_name.split('-')[1]
    # aug_env = make_env(env_id=aug_env_name, seed=seed, rank=rank, kwargs=aug_env_kwargs)
    aug_env = ParallelSubprocVecEnv([make_thunk_aug(i) for i in range(n_workers)])

    if os.path.exists(os.path.join(logger.get_dir(), 'eval.csv')):
        os.remove(os.path.join(logger.get_dir(), 'eval.csv'))
        print('Remove existing eval.csv')
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['random_ratio'] = 0.0
    eval_env = make_env(env_id=env_name, seed=seed, rank=0, kwargs=eval_env_kwargs)
    eval_env = FlattenDictWrapper(eval_env, ['observation', 'achieved_goal', 'desired_goal'])

    if not play:
        os.makedirs(log_dir, exist_ok=True)

    # Available strategies (cf paper): future, final, episode, random
    goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE

    if not play:
        if model_class is SAC_augment:
            from stable_baselines.ddpg.noise import NormalActionNoise
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
                                priority_buffer=priority,
                                learning_rate=learning_rate,
                                curriculum=curriculum,
                                eval_env=eval_env,
                                aug_env=aug_env,
                                imitation_coef=imitation_coef,
                                sequential=sequential,
                                )
            if n_workers == 1:
                pass
                # del train_kwargs['priority_buffer']
            if 'FetchStack' in env_name:
                train_kwargs['ent_coef'] = "auto"
                train_kwargs['tau'] = 0.001
                train_kwargs['gamma'] = 0.98
                train_kwargs['batch_size'] = 256
                train_kwargs['random_exploration'] = 0.1
            elif 'FetchPushWallObstacle' in env_name:
                train_kwargs['tau'] = 0.001
                train_kwargs['gamma'] = 0.98
                train_kwargs['batch_size'] = 256
                train_kwargs['random_exploration'] = 0.1
            elif 'MasspointPushDoubleObstacle' in env_name:
                train_kwargs['buffer_size'] = int(5e5)
                train_kwargs['ent_coef'] = "auto"
                train_kwargs['gamma'] = 0.99
                train_kwargs['batch_size'] = 256
                train_kwargs['random_exploration'] = 0.2
            elif 'MasspointMaze' in env_name:
                train_kwargs['n_subgoal'] = 1
            policy_kwargs = {}

            def callback(_locals, _globals):
                if _locals['step'] % int(1e3) == 0:
                    if 'FetchStack' in env_name:
                        mean_eval_reward = stack_eval_model(eval_env, _locals["self"],
                                                            init_on_table=(env_name=='FetchStack-v2'))
                    elif 'MasspointPushDoubleObstacle-v2' in env_name:
                        mean_eval_reward = egonav_eval_model(eval_env, _locals["self"], env_kwargs["random_ratio"])
                    else:
                        mean_eval_reward = eval_model(eval_env, _locals["self"])
                    log_eval(_locals['self'].num_timesteps, mean_eval_reward)
                if _locals['step'] % int(2e4) == 0:
                    model_path = os.path.join(log_dir, 'model_' + str(_locals['step'] // int(2e4)))
                    model.save(model_path)
                    print('model saved to', model_path)
                return True
        else:
            train_kwargs = {}
            policy_kwargs = {}
            callback = None
        class CustomSACPolicy(SACPolicy):
            def __init__(self, *args, **kwargs):
                super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                                    layers=[256, 256] if 'MasspointPushDoubleObstacle' in env_name else [256, 256, 256, 256],
                                                    feature_extraction="mlp")
        register_policy('CustomSACPolicy', CustomSACPolicy)
        from utils.sac_attention_policy import AttentionPolicy
        register_policy('AttentionPolicy', AttentionPolicy)
        if policy == "AttentionPolicy":
            assert env_name is not 'MasspointPushDoubleObstacle-v2'
            if 'FetchStack' in env_name:
                policy_kwargs["n_object"] = n_object
                policy_kwargs["feature_extraction"] = "attention_mlp"
            elif 'MasspointPushDoubleObstacle' in env_name:
                policy_kwargs["feature_extraction"] = "attention_mlp_particle"
                policy_kwargs["layers"] = [256, 256, 256, 256]
                policy_kwargs["fix_logstd"] = 0.0
            policy_kwargs["layer_norm"] = True
        elif policy == "CustomSACPolicy":
            policy_kwargs["layer_norm"] = True
        if rank == 0:
            print('train_kwargs', train_kwargs)
            print('policy_kwargs', policy_kwargs)
        # Wrap the model
        model = HER_HACK(policy, env, model_class, n_sampled_goal=4,
                         start_augment_time=start_augment,
                         goal_selection_strategy=goal_selection_strategy,
                         num_workers=n_workers,
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
        model = HER_HACK.load(load_path, env=env)

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
         reward_type=args.reward_type, n_object=args.n_object, start_augment=int(args.start_augment),
         policy=args.policy, n_workers=args.num_workers, priority=args.priority, curriculum=args.curriculum,
         learning_rate=args.learning_rate, imitation_coef=args.imitation_coef, sequential=args.sequential)
