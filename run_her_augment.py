from baselines import HER_HACK, SAC_augment
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.common.policies import register_policy
# from utils.parallel_subproc_vec_env import ParallelSubprocVecEnv
# from utils.parallel_subproc_vec_env2 import ParallelSubprocVecEnv as ParallelSubprocVecEnv2
# from utils.subproc_vec_vae_env import ParallelVAESubprocVecEnv
# from utils.subproc_vec_vae_env2 import ParallelVAESubprocVecEnv as ParallelVAESubprocVecEnv2
from utils.subproc_vec_vae_env_new import ParallelVAESubprocVecEnv
from utils.subproc_vec_vae_env_new2 import ParallelVAESubprocVecEnv as ParallelVAESubprocVecEnv2
from gym.wrappers import FlattenDictWrapper
import gym
import matplotlib.pyplot as plt
from stable_baselines.common import set_global_seeds
from stable_baselines import logger
# from stable_baselines.bench import Monitor
from utils.monitor import Monitor
# from run_her import make_env, get_env_kwargs
from run_her import get_env_kwargs
# from run_ppo_augment import make_env
from push_wall_obstacle import FetchPushWallObstacleEnv_v4
from masspoint_env import MasspointPushSingleObstacleEnv_v2, MasspointPushDoubleObstacleEnv
from masspoint_env import MasspointMazeEnv, MasspointSMazeEnv
from fetch_stack import FetchStackEnv
import os, time
import imageio
import csv,pickle
import argparse
import utils.torch.pytorch_util as ptu
from run_ppo_augment import stack_eval_model, eval_model, log_eval, egonav_eval_model
from utils.wrapper import DoneOnSuccessWrapper,VAEWrappedEnv,LatentWrappedEnv
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


hard_test = False

IMAGE_ENTRY_POINT = {
    'Image84SawyerPushAndReachArenaTrainEnvBig-v0':  'ImagePushAndReach',
    'Image84SawyerPushAndReachArenaTrainEnvBigUnlimit-v0':  'ImagePushAndReach',
    'Image48PointmassUWallTrainEnvBig-v0':'ImageUWall',
    'Image48PointmassUWallTrainEnvBigUnlimit-v0': 'ImageUWall',

}

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
    parser.add_argument('--reward_object',type=int,default=1)
    parser.add_argument('--epsilon',type=float,default=0.06)
    parser.add_argument('--n_object', type=int, default=2)
    parser.add_argument('--start_augment', type=float, default=0)
    parser.add_argument('--priority', action="store_true", default=False)
    parser.add_argument('--curriculum', action="store_true", default=False)
    parser.add_argument('--imitation_coef', type=float, default=5)
    parser.add_argument('--sequential', action="store_true", default=False)
    parser.add_argument('--export_gif', action="store_true", default=False)
    args = parser.parse_args()
    return args
ENTRY_POINT = {'FetchPushWallObstacle-v4': FetchPushWallObstacleEnv_v4,
               'FetchPushWallObstacleUnlimit-v4': FetchPushWallObstacleEnv_v4,
               # 'FetchPushWall-v1': FetchPushWallEnv,
               # 'FetchPushBox-v1': FetchPushBoxEnv,
               }
MASS_ENTRY_POINT = {
    'MasspointPushSingleObstacle-v2': MasspointPushSingleObstacleEnv_v2,
    'MasspointPushSingleObstacleUnlimit-v2': MasspointPushSingleObstacleEnv_v2,
    'MasspointPushDoubleObstacle-v1': MasspointPushDoubleObstacleEnv,
    'MasspointPushDoubleObstacleUnlimit-v1': MasspointPushDoubleObstacleEnv,
    'MasspointMaze-v1': MasspointMazeEnv,
    'MasspointMazeUnlimit-v1': MasspointMazeEnv,
    'MasspointMaze-v2': MasspointSMazeEnv,
    'MasspointMazeUnlimit-v2': MasspointSMazeEnv,
}

PICK_ENTRY_POINT = {
    'FetchStack-v1': FetchStackEnv,
    'FetchStackUnlimit-v1': FetchStackEnv,
}
VAE_LOAD_PATH = {
    'Image84SawyerPushAndReachArenaTrainEnvBig-v0':'/home/yilin/vae_data/pnr//vae.pkl',
    'Image84SawyerPushAndReachArenaTrainEnvBigUnlimit-v0': '/home/yilin/vae_data/pnr/vae.pkl',

    'Image48PointmassUWallTrainEnvBig-v0':'/home/yilin/vae_data/pm//vae.pkl',
    'Image48PointmassUWallTrainEnvBigUnlimit-v0': '/home/yilin/vae_data/pm/vae.pkl',

}
# load the vae_model
# ENV_NAME= 'Image48PointmassUWallTrainEnvBig-v0'
# VAE_FILE = open(VAE_LOAD_PATH[ENV_NAME], 'rb')
# VAE_MODEL = pickle.load(VAE_FILE)
# import utils.torch.pytorch_util as ptu
# ptu.set_device(0)
# ptu.set_gpu_mode(True)
# VAE_MODEL.cuda()
def eval_img_model(eval_env, model,vae_model,regressor=None):
    env = eval_env
    n_episode = 0
    ep_rewards = []
    ep_successes = []
    while n_episode < 20:
        ep_reward = 0.0
        ep_success = 0.0
        obs = env.reset()
        obs_reshape = obs.reshape(-1, vae_model.imlength)
        obs_latent_var, _ = vae_model.encode(ptu.np_to_var(obs_reshape))
        obs_latent = ptu.get_numpy(obs_latent_var)
        obs_latent_reshape = obs_latent.reshape(16 * 3, )
        done = False
        step = 0
        while not done:
            step +=1
            action, _ = model.predict(obs_latent_reshape)
            obs, reward, done, info = env.step(action)
            obs_reshape = obs.reshape(-1, vae_model.imlength)
            obs_latent_var, _ = vae_model.encode(ptu.np_to_var(obs_reshape))
            obs_latent = ptu.get_numpy(obs_latent_var)
            obs_latent_reshape = obs_latent.reshape(16 * 3, )
            obs_latent_input = obs_latent.reshape(3,16)
            state = regressor.predict(obs_latent_input)
            achieved_state = state[1]
            desired_state = state[2]
            if achieved_state.shape[0] == 4:
                puck_dist = achieved_state[:2]-desired_state[:2]
                hand_dist = achieved_state[-2:]-desired_state[-2:]
                dist = np.linalg.norm(puck_dist)+np.linalg.norm(hand_dist)
            else:
                dist = np.linalg.norm(achieved_state-desired_state)
            reward = -1.0*(dist>= 0.06) + 1.0
            info['is_success'] = dist< 0.06
            done = done or info['is_success']
            ep_reward += reward
            ep_success += info['is_success']
        ep_rewards.append(ep_reward)
        ep_successes.append(ep_success)
        n_episode += 1
    return np.mean(ep_successes)

def create_image_84_sawyer_pnr_arena_train_env_big_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_tdm_v4

    wrapped_env = gym.make('SawyerPushAndReachArenaTrainEnvBig-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_pusher_camera_tdm_v4,
        transpose=True,
        normalize=True,
        reward_type='hand_puck_success'
    )
def create_image_84_sawyer_pnr_arena_train_env_big_v5():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_tdm_v4

    wrapped_env = gym.make('SawyerPushAndReachArenaTrainEnvBig-v5')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_pusher_camera_tdm_v4,
        transpose=True,
        normalize=True,
        reward_type='sparse'
    )
def create_image_48_pointmass_uwall_train_env_big_v0():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTrainEnvBig-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )

def make_env(env_id, seed, rank, log_dir=None, allow_early_resets=True, kwargs=None,regressor=None):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The mujoco environment
    """
    if env_id in ENTRY_POINT.keys() or env_id in MASS_ENTRY_POINT.keys() or env_id in PICK_ENTRY_POINT.keys() or env_id in IMAGE_ENTRY_POINT.keys():
        # env = ENTRY_POINT[env_id](**kwargs)
        # print(env)
        # from gym.wrappers.time_limit import TimeLimit
        kwargs = kwargs.copy()
        max_episode_steps = 100
        if env_id in IMAGE_ENTRY_POINT.keys():
            if IMAGE_ENTRY_POINT[env_id] == 'ImagePushAndReach':
                imsize = 84
                gym.register(
                    id=env_id,
                    entry_point=create_image_84_sawyer_pnr_arena_train_env_big_v0,
                    max_episode_steps=max_episode_steps,
                    tags={
                        'git-commit-hash': 'e5c11ac',
                        'author': 'Soroush'
                    },
                )
            elif IMAGE_ENTRY_POINT[env_id] =='ImageUWall':
                imsize = 48
                gym.register(
                    id=env_id,
                    entry_point=create_image_48_pointmass_uwall_train_env_big_v0,
                    max_episode_steps=max_episode_steps,

                    tags={
                        'git-commit-hash': 'e5c11ac',
                        'author': 'Soroush'
                    },
                )
            env = gym.make(env_id)

            # vae_file = open(VAE_LOAD_PATH[env_id], 'rb')
            # vae_model = pickle.load(vae_file)
            # import utils.torch.pytorch_util as ptu
            # # if rank > 3:
            # #     ptu.set_device(1)
            # # else:
            # ptu.set_device(0)
            # ptu.set_gpu_mode(True)
            # env = VAEWrappedEnv(env,vae_model,epsilon=epsilon,use_vae_goals=False,imsize=48,reward_params=dict(type='state_distance'))
            reward_threshold=kwargs['reward_threshold']
            env = LatentWrappedEnv(env,epsilon=reward_threshold,use_vae_goals=False,imsize=imsize,reward_params=dict(type=kwargs['reward_type'],object=kwargs['reward_object']))
            # env.wrapped_env.reward_type='wrapped_env'
            # env.reward_type=kwargs['reward_type']
            # import ipdb;ipdb.set_trace()

        else:
            if 'max_episode_steps' in kwargs:
                max_episode_steps = kwargs['max_episode_steps']
                del kwargs['max_episode_steps']
            if env_id in ENTRY_POINT.keys():
                gym.register(env_id, entry_point=ENTRY_POINT[env_id], max_episode_steps=max_episode_steps, kwargs=kwargs)
            elif env_id in MASS_ENTRY_POINT.keys():
                gym.register(env_id, entry_point=MASS_ENTRY_POINT[env_id], max_episode_steps=max_episode_steps, kwargs=kwargs)
            elif env_id in PICK_ENTRY_POINT.keys():
                gym.register(env_id, entry_point=PICK_ENTRY_POINT[env_id], max_episode_steps=max_episode_steps,
                             kwargs=kwargs)
            env = gym.make(env_id)


        # env = TimeLimit(env, max_episode_steps=50)
    else:
        env = gym.make(env_id, reward_type='sparse')
    # env = FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
    if env_id in PICK_ENTRY_POINT.keys() and kwargs['reward_type'] == 'dense':
        env = DoneOnSuccessWrapper(env, reward_offset=0.0)
    elif kwargs['reward_type'] in ('latent_distance','state_distance'):
        print('reward_type',kwargs['reward_type'])
        env = DoneOnSuccessWrapper(env, reward_offset=0.0)

    elif env_id in IMAGE_ENTRY_POINT.keys():

        env =DoneOnSuccessWrapper(env, reward_offset=0.0)

        # print('env_type',env)
    else:
        env = DoneOnSuccessWrapper(env)
    if log_dir is not None:
        env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=allow_early_resets,
                      info_keywords=('is_success',))


    # env.seed(seed + 10000 * rank)
    print('env_type',env)
    return env

def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(env_name, seed, num_timesteps, batch_size, log_path, load_path, play,
         export_gif, gamma, random_ratio, action_noise, reward_type, reward_object,n_object,epsilon,start_augment,
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
    vae_file = open(VAE_LOAD_PATH[env_name], 'rb')
    vae_model = pickle.load(vae_file)
    vae_model.cuda()
    import utils.torch.pytorch_util as ptu
    ptu.set_device(0)
    ptu.set_gpu_mode(True)
    ## load knn regressor


    # train_latent = np.load('sawyer_dataset_latents.npy')
    # train_state = np.load('sawyer_dataset_states.npy')
    train_latent = np.load('sawyer_dataset_train_latents_all_21.npy')
    train_state = np.load('sawyer_dataset_train_states_all_21.npy')
    print('training_dataset_size',train_latent.shape[0])
    regressor = KNeighborsRegressor()
    regressor.fit(train_latent, train_state)
    print('regressor fit finished!')
    # env_kwargs = dict(random_box=True,
    #                   random_ratio=random_ratio,
    #                   random_gripper=True,
    #                   # max_episode_steps=50 * n_object if n_object > 3 else 100,
    #                   max_episode_steps=None if sequential else 100,
    #                   reward_type=reward_type,
    #                   n_object=n_object, )
    if env_name in IMAGE_ENTRY_POINT.keys():
        env_kwargs = dict(max_episode_steps=100,
                          reward_type=reward_type,reward_object=reward_object,reward_threshold=epsilon)
    else:
        env_kwargs = get_env_kwargs(env_name, random_ratio=random_ratio, sequential=sequential,
                                reward_type=reward_type, n_object=n_object)


    def make_thunk(rank):
        return lambda: make_env(env_id=env_name, seed=seed, rank=rank, log_dir=log_dir, kwargs=env_kwargs)
    if env_name in IMAGE_ENTRY_POINT.keys():
        env = ParallelVAESubprocVecEnv2([make_thunk(i) for i in range(n_workers)],env_id=env_name,log_dir=log_path,regressor=regressor)
    else:
        env = ParallelSubprocVecEnv2([make_thunk(i) for i in range(n_workers)])
    # env_test = env.env_fns
    # print('paralel_env',env_test)
    # if n_workers > 1:
    #     # env = SubprocVecEnv([make_thunk(i) for i in range(n_workers)])
    # else:
    #     env = make_env(env_id=env_name, seed=seed, rank=rank, log_dir=log_dir, kwargs=env_kwargs)
    # print('env',env)
    def make_thunk_aug(rank):
        if env_name in IMAGE_ENTRY_POINT.keys():
            # return lambda: make_env(env_id=aug_env_name, seed=seed, rank=rank, kwargs=aug_env_kwargs),['observation','achieved_goal','desired_goal']
            #
            return lambda: FlattenDictWrapper(make_env(env_id=aug_env_name, seed=seed, rank=rank, kwargs=aug_env_kwargs),['observation','achieved_goal','desired_goal'])
        else:
            # print('using FlattenDictWrapper')
            return lambda: FlattenDictWrapper(make_env(env_id=aug_env_name, seed=seed, rank=rank, kwargs=aug_env_kwargs),
                                          ['observation', 'achieved_goal', 'desired_goal'])

    aug_env_kwargs = env_kwargs.copy()
    del aug_env_kwargs['max_episode_steps']
    aug_env_name = env_name.split('-')[0] + 'Unlimit-' + env_name.split('-')[1]
    # aug_env = make_env(env_id=aug_env_name, seed=seed, rank=rank, kwargs=aug_env_kwargs)
    if env_name in IMAGE_ENTRY_POINT.keys():
        aug_env = ParallelVAESubprocVecEnv([make_thunk_aug(i) for i in range(n_workers)],env_id=env_name,log_dir=log_path,regressor=regressor)

    else:
        aug_env = ParallelSubprocVecEnv([make_thunk_aug(i) for i in range(n_workers)])

    if os.path.exists(os.path.join(logger.get_dir(), 'eval.csv')):
        os.remove(os.path.join(logger.get_dir(), 'eval.csv'))
        print('Remove existing eval.csv')
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['random_ratio'] = 0.0
    eval_env = make_env(env_id=env_name, seed=seed, rank=0, kwargs=eval_env_kwargs)
    if env_name not in IMAGE_ENTRY_POINT.keys():

        eval_env = FlattenDictWrapper(eval_env, ['observation', 'achieved_goal', 'desired_goal'])

    else :
        # changing the dict for eval_env
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
                                tensorboard_log=log_path
                                )
            if n_workers == 1:
                pass
                # del train_kwargs['priority_buffer']
            if env_name in ('FetchStack') :
                train_kwargs['ent_coef'] = "auto"
                train_kwargs['tau'] = 0.001
                train_kwargs['gamma'] = 0.98
                train_kwargs['batch_size'] = 256
                train_kwargs['random_exploration'] = 0.1
            elif env_name in IMAGE_ENTRY_POINT.keys():
                train_kwargs['ent_coef'] = "auto"
                train_kwargs['tau'] = 0.001
                train_kwargs['gamma'] = 0.98
                train_kwargs['batch_size'] = 256
                train_kwargs['random_exploration'] = 0.1
                train_kwargs['n_subgoal']=1
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
                # if _locals['step'] % int(1e3) == 0:
                #     if 'FetchStack' in env_name:
                #         mean_eval_reward = stack_eval_model(eval_env, _locals["self"],
                #                                             init_on_table=(env_name=='FetchStack-v2'))
                #     elif 'MasspointPushDoubleObstacle-v2' in env_name:
                #         mean_eval_reward = egonav_eval_model(eval_env, _locals["self"], env_kwargs["random_ratio"])
                #         mean_eval_reward2 = egonav_eval_model(eval_env, _locals["self"], env_kwargs["random_ratio"],
                #                                               goal_idx=0)
                #         log_eval(_locals['self'].num_timesteps, mean_eval_reward2, file_name="eval_box.csv")
                #     elif env_name in IMAGE_ENTRY_POINT.keys():
                #         mean_eval_reward = eval_img_model(eval_env,_locals["self"],vae_model,regressor=regressor)
                #     else:
                #         mean_eval_reward = eval_model(eval_env, _locals["self"])
                #     log_eval(_locals['self'].num_timesteps, mean_eval_reward)
                if _locals['step'] % int(2e4) == 0:
                    model_save_time0 = time.time()
                    model_path = os.path.join(log_dir, 'model_' + str(_locals['step'] // int(2e4)))
                    model.save(model_path)
                    model_save_time = time.time()-model_save_time0
                    print('model_save_time',model_save_time)
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
        # if rank == 0:
            # print('train_kwargs', train_kwargs)
            # print('policy_kwargs', policy_kwargs)
        # Wrap the model
        model = HER_HACK(policy=policy, env=env, model_class=model_class,env_id=env_name, n_sampled_goal=4,
                         start_augment_time=start_augment,
                         goal_selection_strategy=goal_selection_strategy,
                         num_workers=n_workers,
                         policy_kwargs=policy_kwargs,
                         verbose=1,
                         **train_kwargs)
        # print(model.get_parameter_list())

        # Train the model
        model.learn(num_timesteps, seed=seed, callback=callback, log_interval=100)

        if rank == 0:
            model.save(os.path.join(log_dir, 'final'))

    # WARNING: you must pass an env
    # or wrap your environment with HERGoalEnvWrapper to use the predict method
    if play and rank == 0:
        assert load_path is not None
        model = HER_HACK.load(load_path,env=env,env_id=env_name)
        print('load finished')
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        # fig1,ax1 = plt.subplots(1,1,figsize=(8,8))
        obs = env.env_method('reset')[0]
        obs_img_obs = obs['image_observation']
        obs_img_achieved_goal = obs['image_achieved_goal']
        obs_img_desired_goal = obs['image_desired_goal']
        obs_img = np.stack([obs_img_obs,obs_img_achieved_goal,obs_img_desired_goal])
        obs_latent =ptu.get_numpy(vae_model.encode(ptu.np_to_var(obs_img))[0])
        obs_latent = obs_latent.reshape(-1,vae_model.representation_size*3)
        # sim_state = env.sim.get_state()
        # print(sim_state)
        # while not (obs['desired_goal'][0] < env.pos_wall[0] < obs['achieved_goal'][0] or
        #             obs['desired_goal'][0] > env.pos_wall[0] > obs['achieved_goal'][0]):
        #     if not hard_test:
        #         break
        #     obs = env.reset()
        # print('gripper_pos', obs['observation'][0:3])
        # img = env.render(mode='rgb_array')
        # img = env.env_method('get_image')[0]
        img = env.env_method('get_image_plt')[0]
        episode_reward = 0.0
        images = []
        frame_idx = 0
        episode_idx = 0
        #env.spec.max_episode_steps
        for i in range(100 * 10):
            # images.append(img)

            action, _ = model.predict(obs_latent)
            actions = np.repeat(action,n_workers,axis=0)
            print('action',action)

            # print('action', action)
            # print('obstacle euler', obs['observation'][20:23])
            # obs, reward, done, _ = env.env_method('step',actions)[0]
            obs,rewards,dones,_ = env.step(actions)
            reward = rewards[0]
            done = dones[0]
            state = obs['state_observation'][0]
            goal = obs['state_desired_goal'][0]
            hand_state=state[:2]
            puck_state=state[-2:]
            hand_goal=goal[:2]
            puck_goal=goal[-2:]
            hand_dist = round(np.linalg.norm(hand_state-hand_goal),3)
            puck_dist = round(np.linalg.norm(puck_state-puck_goal),3)
            obs_img_obs = obs['image_observation'][0]
            obs_img_achieved_goal = obs['image_achieved_goal'][0]
            obs_img_desired_goal = obs['image_desired_goal'][0]
            obs_img = np.stack([obs_img_obs, obs_img_achieved_goal, obs_img_desired_goal])
            obs_latent = ptu.get_numpy(vae_model.encode(ptu.np_to_var(obs_img))[0])
            obs_latent = obs_latent.reshape(-1,vae_model.representation_size*3)
            obs_latent_input = obs_latent.reshape(-1,16)

            states = regressor.predict(obs_latent_input)
            state_obs = states[0]
            states_desired_goal = states[2]
            state_diff = round(np.linalg.norm(state_obs - state),3)
            goal_diff = round(np.linalg.norm(states_desired_goal - goal),3)
            print('done',done)
            episode_reward += reward
            frame_idx += 1
            ax.cla()
            # img = env.render(mode='rgb_array')
            img = env.env_method('get_image_plt')[0]
            img_robot = (obs['image_observation'][0].reshape(3,vae_model.imsize,vae_model.imsize).transpose(1,2,0)*255).astype('uint8')
            ax.imshow(img)
            # ax1.imshow(img_robot)
            # ax.set_title('episode ' + str(episode_idx) + ', frame ' + str(frame_idx) +
                         # ', goal idx ' + str(np.argmax(obs['desired_goal'][3:])))
            ax.set_title('ep:'+str(episode_idx)+',fp:'+str(frame_idx)+'rew:'
                         +str(reward)+str(done)+'h_d:'+str(hand_dist)+'p_d:'+str(puck_dist)+'s_d'+str(state_diff)+'g_d'+str(goal_diff))
            # ax1.set_title('episode '+str(episode_idx)+',frame'+str(frame_idx)+' reward'+str(reward)+' done'+str(done))

            if export_gif:
                plt.savefig(os.path.join(os.path.dirname(load_path),'tempimg' + str(i) + '.png'))
                # plt.savefig('temp1img'+str(i) + '.png')
            plt.pause(0.02)
            ##plot another view from the viewer

            ax.cla()
            ax.imshow(img_robot)
            ax.set_title(
                'ep:' + str(episode_idx) + ',fp:' + str(frame_idx) + 'rew:' + str(reward) + str(done) + 'h_d:' + str(
                    hand_dist) + 'p_d:' + str(puck_dist))

            if export_gif:
                plt.savefig(os.path.join(os.path.dirname(load_path),'temp1img' + str(i) + '.png'))
            plt.pause(0.02)

            # plt1.pause(0.02)
            if done:
                obs = env.env_method('reset')[0]
                obs_img_obs = obs['image_observation']
                obs_img_achieved_goal = obs['image_achieved_goal']
                obs_img_desired_goal = obs['image_desired_goal']
                obs_img = np.stack([obs_img_obs, obs_img_achieved_goal, obs_img_desired_goal])
                obs_latent = ptu.get_numpy(vae_model.encode(ptu.np_to_var(obs_img))[0])
                obs_latent = obs_latent.reshape(-1,vae_model.representation_size*3)
                # while not (obs['desired_goal'][0] < env.pos_wall[0] < obs['achieved_goal'][0] or
                #             obs['desired_goal'][0] > env.pos_wall[0] > obs['achieved_goal'][0]):
                #     if not hard_test:
                #         break
                #     obs = env.reset()
                # print('gripper_pos', obs['observation'][0:3])
                print('episode_reward', episode_reward)
                episode_reward = 0.0
                frame_idx = 0
                episode_idx += 1
        if export_gif:
            # #env.spec.max_episode_steps
            # for i in range(100 * 10):
            #     images.append(plt.imread('tempimg' + str(i) + '.png'))
            #     os.remove('tempimg' + str(i) + '.png')
            # imageio.mimsave(env_name + '.gif', images)
            os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(
                load_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
                      os.path.join(os.path.dirname(load_path), env_name + '.mp4'))
            os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(
                load_path) + '/temp1img%d.png -c:v libx264 -pix_fmt yuv420p ' +
                      os.path.join(os.path.dirname(load_path), env_name+'viewer' + '.mp4'))
            for i in range(100*10):
                # images.append(plt.imread('tempimg' + str(i) + '.png'))
                try:
                    os.remove( os.path.join(os.path.dirname(load_path),'tempimg' + str(i) + '.png'))
                    os.remove(os.path.join(os.path.dirname(load_path), 'temp1img' + str(i) + '.png'))

                except:
                    pass


if __name__ == '__main__':
    args = arg_parse()
    main(env_name=args.env, seed=args.seed, num_timesteps=int(args.num_timesteps),
         log_path=args.log_path, load_path=args.load_path, play=args.play,
         batch_size=args.batch_size, export_gif=args.export_gif,
         gamma=args.gamma, random_ratio=args.random_ratio, action_noise=args.action_noise,
         reward_type=args.reward_type, reward_object=args.reward_object,n_object=args.n_object, epsilon=args.epsilon, start_augment=int(args.start_augment),
         policy=args.policy, n_workers=args.num_workers, priority=args.priority, curriculum=args.curriculum,
         learning_rate=args.learning_rate, imitation_coef=args.imitation_coef, sequential=args.sequential)
