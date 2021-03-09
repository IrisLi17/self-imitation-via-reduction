from baselines import PPO2_augment
from stable_baselines import logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv

from utils.log_utils import eval_model, log_eval, stack_eval_model
from utils.parallel_subproc_vec_env import ParallelSubprocVecEnv
from stable_baselines.common.policies import register_policy

from utils.make_env_utils import configure_logger, make_env, get_num_workers, get_env_kwargs, get_train_kwargs, \
    get_policy_kwargs

import os, time, argparse


def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='FetchPushWallObstacle-v4')
    parser.add_argument('--policy', type=str, default='MlpPolicy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_timesteps', type=float, default=1e8)
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--random_ratio', default=1.0, type=float)
    parser.add_argument('--aug_clip', default=0.1, type=float)
    parser.add_argument('--aug_adv_weight', default=1.0, type=float)
    parser.add_argument('--n_subgoal', default=4, type=int)
    parser.add_argument('--parallel', action="store_true", default=False)
    parser.add_argument('--self_imitate', action="store_true", default=False)
    parser.add_argument('--sil_clip', default=0.2, type=float)
    parser.add_argument('--start_augment', type=float, default=0)
    parser.add_argument('--reuse_times', default=1, type=int)
    parser.add_argument('--reward_type', default="sparse", type=str)
    parser.add_argument('--n_object', default=2, type=int)
    parser.add_argument('--curriculum', action="store_true", default=False)
    parser.add_argument('--sequential', action="store_true", default=False)
    parser.add_argument('--play', action="store_true", default=False)
    parser.add_argument('--export_gif', action="store_true", default=False)
    parser.add_argument('--log_trace', action="store_true", default=False)
    args = parser.parse_args()
    return args


def main(args):
    log_dir = args.log_path if (args.log_path is not None) else \
        "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    configure_logger(log_dir)

    set_global_seeds(args.seed)

    n_cpu = get_num_workers(args.env) if not args.play else 1

    env_kwargs = get_env_kwargs(args.env, args.random_ratio, args.sequential, args.reward_type,
                                args.n_object, args.curriculum)

    def make_thunk(rank):
        return lambda: make_env(env_id=args.env, rank=rank, log_dir=log_dir, flatten_dict=True, kwargs=env_kwargs)

    env = SubprocVecEnv([make_thunk(i) for i in range(n_cpu)])

    aug_env_name = args.env.split('-')[0] + 'Unlimit-' + args.env.split('-')[1]
    aug_env_kwargs = env_kwargs.copy()
    aug_env_kwargs['max_episode_steps'] = None

    def make_thunk_aug(rank):
        return lambda: make_env(env_id=aug_env_name, rank=rank, flatten_dict=True, kwargs=aug_env_kwargs)

    if not args.parallel:
        aug_env = make_env(env_id=aug_env_name, rank=0, flatten_dict=True, kwargs=aug_env_kwargs)
    else:
        aug_env = ParallelSubprocVecEnv([make_thunk_aug(i) for i in range(min(32, n_cpu))], reset_when_done=False)
    print(aug_env)

    if os.path.exists(os.path.join(logger.get_dir(), 'eval.csv')):
        os.remove(os.path.join(logger.get_dir(), 'eval.csv'))
        print('Remove existing eval.csv')
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['random_ratio'] = 0.0
    if "use_cu" in eval_env_kwargs:
        eval_env_kwargs['use_cu'] = False
    eval_env = make_env(env_id=args.env, rank=0, flatten_dict=True, kwargs=eval_env_kwargs)
    print(eval_env)

    if not args.play:
        os.makedirs(log_dir, exist_ok=True)

        from utils.attention_policy import AttentionPolicy
        register_policy('AttentionPolicy', AttentionPolicy)

        policy_kwargs = get_policy_kwargs("ppo_sir", args)

        train_kwargs = get_train_kwargs("ppo_sir", args, parsed_action_noise=None, eval_env=eval_env, aug_env=aug_env)

        model = PPO2_augment(args.policy, env, verbose=1, nminibatches=32, lam=0.95, gamma=0.99, noptepochs=10,
                             ent_coef=0.01, learning_rate=3e-4, cliprange=0.2, policy_kwargs=policy_kwargs,
                             horizon=env_kwargs['max_episode_steps'], **train_kwargs)

        def callback(_locals, _globals):
            num_update = _locals["update"]
            if 'FetchStack' in args.env:
                mean_eval_reward = stack_eval_model(eval_env, _locals["self"])
            else:
                mean_eval_reward = eval_model(eval_env, _locals["self"])
            log_eval(num_update, mean_eval_reward)
            if num_update % 10 == 0:
                model_path = os.path.join(log_dir, 'model_' + str(num_update // 10))
                model.save(model_path)
                print('model saved to', model_path)
            return True

        model.learn(total_timesteps=int(args.num_timesteps), callback=callback, seed=args.seed, log_interval=1)
        model.save(os.path.join(log_dir, 'final'))


if __name__ == '__main__':
    args = arg_parse()
    print('arg parsed')
    main(args)
