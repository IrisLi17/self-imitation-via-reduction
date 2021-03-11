from baselines import HER2, SAC_SIR
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.common.policies import register_policy
from utils.parallel_subproc_vec_env import ParallelSubprocVecEnv
from gym.wrappers import FlattenDictWrapper
from stable_baselines.common import set_global_seeds
from stable_baselines import logger
from utils.make_env_utils import make_env, get_env_kwargs, get_train_kwargs, get_policy_kwargs
import os, time
import argparse
import numpy as np
from utils.log_utils import eval_model, log_eval, stack_eval_model, egonav_eval_model

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


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


def main(args):
    log_dir = args.log_path if (args.log_path is not None) else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(log_dir)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(log_dir, format_strs=[])

    set_global_seeds(args.seed)

    model_class = SAC_SIR  # works also with SAC, DDPG and TD3

    env_kwargs = get_env_kwargs(args.env, random_ratio=args.random_ratio, sequential=args.sequential,
                                reward_type=args.reward_type, n_object=args.n_object)

    def make_thunk(rank):
        return lambda: make_env(env_id=args.env, rank=rank, log_dir=log_dir, kwargs=env_kwargs)

    env = ParallelSubprocVecEnv([make_thunk(i) for i in range(args.num_workers)], reset_when_done=True)

    def make_thunk_aug(rank):
        return lambda: FlattenDictWrapper(make_env(env_id=aug_env_name, rank=rank, kwargs=aug_env_kwargs),
                                          ['observation', 'achieved_goal', 'desired_goal'])

    aug_env_kwargs = env_kwargs.copy()
    del aug_env_kwargs['max_episode_steps']
    aug_env_name = args.env.split('-')[0] + 'Unlimit-' + args.env.split('-')[1]
    aug_env = ParallelSubprocVecEnv([make_thunk_aug(i) for i in range(args.num_workers)], reset_when_done=False)

    if os.path.exists(os.path.join(logger.get_dir(), 'eval.csv')):
        os.remove(os.path.join(logger.get_dir(), 'eval.csv'))
        print('Remove existing eval.csv')
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['random_ratio'] = 0.0
    eval_env = make_env(env_id=args.env, rank=0, kwargs=eval_env_kwargs)
    eval_env = FlattenDictWrapper(eval_env, ['observation', 'achieved_goal', 'desired_goal'])

    if not args.play:
        os.makedirs(log_dir, exist_ok=True)

    # Available strategies (cf paper): future, final, episode, random
    goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE

    if not args.play:
        from stable_baselines.ddpg.noise import NormalActionNoise
        noise_type = args.action_noise.split('_')[0]
        if noise_type == 'none':
            parsed_action_noise = None
        elif noise_type == 'normal':
            sigma = float(args.action_noise.split('_')[1])
            parsed_action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape),
                                                    sigma=sigma * np.ones(env.action_space.shape))
        else:
            raise NotImplementedError

        train_kwargs = get_train_kwargs("sac_sir", args, parsed_action_noise, eval_env, aug_env)

        def callback(_locals, _globals):
            if _locals['step'] % int(1e3) == 0:
                if 'FetchStack' in args.env:
                    mean_eval_reward = stack_eval_model(eval_env, _locals["self"],
                                                        init_on_table=(args.env=='FetchStack-v2'))
                elif 'MasspointPushDoubleObstacle-v2' in args.env:
                    mean_eval_reward = egonav_eval_model(eval_env, _locals["self"], env_kwargs["random_ratio"], fixed_goal=np.array([4., 4., 0.15, 0., 0., 0., 1.]))
                    mean_eval_reward2 = egonav_eval_model(eval_env, _locals["self"], env_kwargs["random_ratio"],
                                                          goal_idx=0, fixed_goal=np.array([4., 4., 0.15, 1., 0., 0., 0.]))
                    log_eval(_locals['self'].num_timesteps, mean_eval_reward2, file_name="eval_box.csv")
                else:
                    mean_eval_reward = eval_model(eval_env, _locals["self"])
                log_eval(_locals['self'].num_timesteps, mean_eval_reward)
            if _locals['step'] % int(2e4) == 0:
                model_path = os.path.join(log_dir, 'model_' + str(_locals['step'] // int(2e4)))
                model.save(model_path)
                print('model saved to', model_path)
            return True

        class CustomSACPolicy(SACPolicy):
            def __init__(self, *model_args, **model_kwargs):
                super(CustomSACPolicy, self).__init__(*model_args, **model_kwargs,
                                                      layers=[256, 256] if 'MasspointPushDoubleObstacle' in args.env else [256, 256, 256, 256],
                                                      feature_extraction="mlp")
        register_policy('CustomSACPolicy', CustomSACPolicy)
        from utils.sac_attention_policy import AttentionPolicy
        register_policy('AttentionPolicy', AttentionPolicy)
        policy_kwargs = get_policy_kwargs("sac_sir", args)

        if rank == 0:
            print('train_kwargs', train_kwargs)
            print('policy_kwargs', policy_kwargs)
        # Wrap the model
        model = HER2(args.policy, env, model_class, n_sampled_goal=4,
                     start_augment_time=args.start_augment,
                     goal_selection_strategy=goal_selection_strategy,
                     num_workers=args.num_workers,
                     policy_kwargs=policy_kwargs,
                     verbose=1,
                     **train_kwargs)
        print(model.get_parameter_list())

        # Train the model
        model.learn(int(args.num_timesteps), seed=args.seed, callback=callback, log_interval=100 if not ('MasspointMaze-v3' in args.env) else 10)

        if rank == 0:
            model.save(os.path.join(log_dir, 'final'))


if __name__ == '__main__':
    args = arg_parse()
    main(args)
