import os

import gym

from fetch_stack import FetchStackEnv
from masspoint_env import MasspointPushDoubleObstacleEnv, MasspointPushDoubleObstacleEnv_v2, MasspointSMazeEnv, \
    MasspointEMazeEasyEnv, MasspointPushMultiObstacleEnv
from push_wall_obstacle import FetchPushWallObstacleEnv_v4

from utils.wrapper import DoneOnSuccessWrapper
from gym.wrappers import FlattenDictWrapper
from stable_baselines import logger
from stable_baselines.bench import Monitor


ENTRY_POINT = {
    'FetchPushWallObstacle-v4': FetchPushWallObstacleEnv_v4,
    'FetchPushWallObstacleUnlimit-v4': FetchPushWallObstacleEnv_v4,
    'MasspointPushDoubleObstacle-v1': MasspointPushDoubleObstacleEnv,
    'MasspointPushDoubleObstacleUnlimit-v1': MasspointPushDoubleObstacleEnv,
    'MasspointPushDoubleObstacle-v2': MasspointPushDoubleObstacleEnv_v2,
    'MasspointPushDoubleObstacleUnlimit-v2': MasspointPushDoubleObstacleEnv_v2,
    'MasspointPushMultiObstacle-v1': MasspointPushMultiObstacleEnv,
    'MasspointPushMultiObstacleUnlimit-v1': MasspointPushMultiObstacleEnv,
    'MasspointMaze-v2': MasspointSMazeEnv,
    'MasspointMazeUnlimit-v2': MasspointSMazeEnv,
    'MasspointMaze-v3': MasspointEMazeEasyEnv,
    'MasspointMazeUnlimit-v3': MasspointEMazeEasyEnv,
    'FetchStack-v1': FetchStackEnv,
    'FetchStackUnlimit-v1': FetchStackEnv,
    }


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def make_env(env_id, rank, log_dir=None, allow_early_resets=True, flatten_dict=False, kwargs=None):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The mujoco environment
    """
    if env_id in ENTRY_POINT.keys():
        kwargs = kwargs.copy()
        max_episode_steps = None
        if 'max_episode_steps' in kwargs:
            max_episode_steps = kwargs['max_episode_steps']
            del kwargs['max_episode_steps']
        gym.register(env_id, entry_point=ENTRY_POINT[env_id], max_episode_steps=max_episode_steps, kwargs=kwargs)
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    if flatten_dict:
        env = FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
    if 'FetchStack' in env_id and ('Unlimit' not in env_id) and max_episode_steps is None:
        from utils.wrapper import FlexibleTimeLimitWrapper
        env = FlexibleTimeLimitWrapper(env, 100)
    if kwargs['reward_type'] != 'sparse':
        env = DoneOnSuccessWrapper(env, 0.0)
    else:
        env = DoneOnSuccessWrapper(env)
    if log_dir is not None:
        env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=allow_early_resets,
                      info_keywords=('is_success',))
    return env


def get_env_kwargs(env_id, random_ratio=None, sequential=None, reward_type=None, n_object=None, curriculum=False):
    if env_id == 'FetchStack-v1' or env_id == 'FetchStack-v2':
        return dict(random_box=True,
                    random_ratio=random_ratio,
                    random_gripper=True,
                    max_episode_steps=None if sequential else 100,
                    reward_type=reward_type,
                    n_object=n_object,)
    elif env_id == 'FetchPushWallObstacle-v4':
        return dict(random_box=True,
                    heavy_obstacle=True,
                    random_ratio=random_ratio,
                    random_gripper=True,
                    reward_type=reward_type,
                    max_episode_steps=100, )
    elif env_id == 'MasspointPushDoubleObstacle-v1' or env_id == 'MasspointPushDoubleObstacle-v2':
        return dict(random_box=True,
                    random_ratio=random_ratio,
                    random_pusher=True,
                    reward_type=reward_type,
                    max_episode_steps=150, )
    elif env_id == 'MasspointPushMultiObstacle-v1':
        return dict(random_box=True,
                    random_ratio=random_ratio,
                    random_pusher=True,
                    reward_type=reward_type,
                    max_episode_steps=150,
                    n_object=4,)
    elif env_id == 'MasspointMaze-v2':
        return dict(random_box=True,
                    random_ratio=random_ratio,
                    random_pusher=True,
                    max_episode_steps=100, )
    elif env_id == "MasspointMaze-v3":
        return dict(random_ratio=random_ratio,
                    random_pusher=False,
                    fix_goal=False,
                    max_episode_steps=100,
                    reward_type=reward_type,
                    use_cu=curriculum)
    else:
        raise NotImplementedError


def get_train_kwargs(algo, args, parsed_action_noise, eval_env, aug_env=None):
    if algo == "sac" or algo == "sac_sir":
        train_kwargs = dict(buffer_size=int(1e5),
                            ent_coef="auto",
                            gamma=args.gamma,
                            learning_starts=1000,
                            train_freq=1,
                            batch_size=args.batch_size,
                            action_noise=parsed_action_noise,
                            priority_buffer=args.priority,
                            learning_rate=args.learning_rate,
                            curriculum=args.curriculum,
                            sequential=args.sequential,
                            eval_env=eval_env,
                            )
        if algo == "sac":
            train_kwargs["sil"] = args.sil
            train_kwargs["sil_coef"] = args.sil_coef
        elif algo == "sac_sir":
            train_kwargs["aug_env"] = aug_env
            train_kwargs["imitation_coef"] = args.imitation_coef
        if 'FetchStack' in args.env:
            train_kwargs['ent_coef'] = "auto"
            train_kwargs['tau'] = 0.001
            train_kwargs['gamma'] = 0.98
            train_kwargs['batch_size'] = 256
            train_kwargs['random_exploration'] = 0.1
        elif 'FetchPushWallObstacle' in args.env:
            train_kwargs['tau'] = 0.001
            train_kwargs['gamma'] = 0.98
            train_kwargs['batch_size'] = 256
            train_kwargs['random_exploration'] = 0.1
        elif 'MasspointPushDoubleObstacle' in args.env:
            train_kwargs['buffer_size'] = int(5e5)
            train_kwargs['ent_coef'] = "auto"
            train_kwargs['gamma'] = 0.99
            train_kwargs['batch_size'] = 256
            train_kwargs['random_exploration'] = 0.2
        elif 'MasspointMaze-v3' in args.env:
            train_kwargs['buffer_size'] = int(1e4)
        elif 'MasspointMaze' in args.env and algo == "sac_sir":
            train_kwargs['n_subgoal'] = 1
    elif algo == "ppo" or algo == "ppo_sir":
        train_kwargs = dict(curriculum=args.curriculum,
                            eval_env=eval_env,)
        if 'MasspointPush' in args.env or 'FetchStack' in args.env:
            n_steps = 8192
        elif 'MasspointMaze' in args.env:
            n_steps = 1024
        else:
            n_steps = 2048
        train_kwargs['n_steps'] = n_steps
        if algo == "ppo":
            train_kwargs['gamma'] = args.gamma
        else:
            train_kwargs['aug_env'] = aug_env
            train_kwargs['aug_clip'] = args.aug_clip
            train_kwargs['n_candidate'] = args.n_subgoal
            train_kwargs['parallel'] = args.parallel
            train_kwargs['start_augment'] = args.start_augment
            train_kwargs['reuse_times'] = args.reuse_times
            train_kwargs['aug_adv_weight'] = args.aug_adv_weight
            train_kwargs['self_imitate'] = args.self_imitate
            train_kwargs['sil_clip'] = args.sil_clip
            train_kwargs['log_trace'] = args.log_trace
            train_kwargs['dim_candidate'] = 3 if 'FetchStack' in args.env else 2
    else:
        raise NotImplementedError

    return train_kwargs


def get_policy_kwargs(algo, args):
    policy_kwargs = {}
    if algo == "sac" or algo == "sac_sir":
        if args.policy == 'AttentionPolicy':
            assert args.env is not 'MasspointPushDoubleObstacle-v2', 'attention policy not supported!'
            if 'FetchStack' in args.env:
                policy_kwargs["n_object"] = args.n_object
                policy_kwargs["feature_extraction"] = "attention_mlp"
            elif 'MasspointPushDoubleObstacle' in args.env:
                policy_kwargs["feature_extraction"] = "attention_mlp_particle"
                policy_kwargs["layers"] = [256, 256, 256, 256]
                policy_kwargs["fix_logstd"] = 0.0
            policy_kwargs["layer_norm"] = True
        elif args.policy == "CustomSACPolicy":
            policy_kwargs["layer_norm"] = True
    elif algo == "ppo" or algo == "ppo_sir":
        policy_kwargs = dict(layers=[256, 256])
        if args.policy == "AttentionPolicy":
            if 'FetchStack' in args.env:
                policy_kwargs["n_object"] = args.n_object
                policy_kwargs["feature_extraction"] = "attention_mlp"
            elif 'MasspointPushDoubleObstacle' in args.env:
                policy_kwargs["feature_extraction"] = "attention_mlp_particle"
                policy_kwargs["n_object"] = 3
            elif 'MasspointPushMultiObstacle' in args.env:
                policy_kwargs["feature_extraction"] = "attention_mlp_particle"
                policy_kwargs["n_object"] = args.n_object
    else:
        raise NotImplementedError
    return policy_kwargs


def get_num_workers(env_name):
    n_cpu = 32
    if 'MasspointPush' in env_name:
        n_cpu = 64
    elif 'MasspointMaze' in env_name:
        if env_name == 'MasspointMaze-v3':
            n_cpu = 1
        else:
            n_cpu = 8
    elif 'FetchStack' in env_name:
        n_cpu = 128
    return n_cpu
