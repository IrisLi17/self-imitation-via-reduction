from stable_baselines.common.policies import register_policy
from baselines import PPO2_augment, PPO2_augment_sil
from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from utils.parallel_subproc_vec_env import ParallelSubprocVecEnv
from gym.wrappers import FlattenDictWrapper
from push_wall_obstacle import FetchPushWallObstacleEnv_v4
from masspoint_env import MasspointPushSingleObstacleEnv_v2, MasspointPushDoubleObstacleEnv
from masspoint_env import MasspointMazeEnv, MasspointSMazeEnv
from fetch_stack import FetchStackEnv
# from push_wall import FetchPushWallEnv
# from push_box import FetchPushBoxEnv
import gym
from utils.wrapper import DoneOnSuccessWrapper,VAEWrappedEnv
import pickle
import numpy as np
import csv, pickle
import multiworld
import os, time, argparse, imageio
import matplotlib.pyplot as plt


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
IMAGE_ENTRY_POINT = {
    'Image84SawyerPushAndReachArenaTrainEnvBig-v0':  'ImagePushAndReach',
    'Image84SawyerPushAndReachArenaTrainEnvBigUnlimit-v0':  'ImagePushAndReach',
    'Image48PointmassUWallTrainEnvBig-v0':'ImageUWall',
    'Image48PointmassUWallTrainEnvBigUnlimit-v0': 'ImageUWall',

}
VAE_LOAD_PATH = {
    'Image84SawyerPushAndReachArenaTrainEnvBig-v0':'/home/yilin/leap/data/pnr/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_16_10_33_id000--s85192/vae.pkl',
    'Image84SawyerPushAndReachArenaTrainEnvBigUnlimit-v0': '/home/yilin/leap/data/pnr/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_16_10_33_id000--s85192/vae.pkl',

    'Image48PointmassUWallTrainEnvBig-v0':'/home/yilin/leap/data/pm/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_22_23_14_id000--s4047/vae.pkl',
    'Image48PointmassUWallTrainEnvBigUnlimit-v0': '/home/yilin/leap/data/pm/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_22_23_14_id000--s4047/vae.pkl',

}

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

def make_env(env_id, seed, rank, log_dir=None, allow_early_resets=True, kwargs=None):
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
        max_episode_steps = None
        if env_id in IMAGE_ENTRY_POINT.keys():
            if IMAGE_ENTRY_POINT[env_id] == 'ImagePushAndReach':
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
                gym.register(
                    id=env_id,
                    entry_point=create_image_48_pointmass_uwall_train_env_big_v0,
                    tags={
                        'git-commit-hash': 'e5c11ac',
                        'author': 'Soroush'
                    },
                )
            env = gym.make(env_id)
            # env.wrapped_env.reward_type='sparse'
            vae_file = open(VAE_LOAD_PATH[env_id],'rb')
            import utils.torch.pytorch_util  as ptu
            ptu.set_gpu_mode(True)
            vae_model = pickle.load(vae_file)
            env = VAEWrappedEnv(env,vae_model)
            # env.wrapped_env.reward_type='wrapped_env'
            env.reward_type='latent_sparse'
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
    env = FlattenDictWrapper(env,['latent_observation','latent_achieved_goal','latent_desired_goal'])
    if env_id in PICK_ENTRY_POINT.keys() and kwargs['reward_type'] == 'dense':
        env = DoneOnSuccessWrapper(env, reward_offset=0.0)
    else:
        env = DoneOnSuccessWrapper(env)
    if log_dir is not None:
        env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=allow_early_resets,
                      info_keywords=('is_success',))
    # env.seed(seed + 10000 * rank)
    return env
import matplotlib.pyplot as plt

def main():
    def make_thunk(rank):
        return lambda: make_env(env_id='Image48PointmassUWallTrainEnvBig-v0',seed=None, rank=rank, kwargs={})
    env = make_env(env_id='Image48PointmassUWallTrainEnvBig-v0',seed=None,rank=0,kwargs={})
    # env = SubprocVecEnv([make_thunk(i) for i in range(1)])
    obs = env.reset()
    start_img = env.get_image()
    current_obs = obs[:16]
    goal = obs[-16:]
    # print('reset_obs',obs.shape,obs[0,-16:])
    # current_obs = obs[0,:16]
    # print('current_obs',current_obs)
    # goal = current_obs
    # env.set_goal(goal)
    # env.env_method('set_goal',goal)
    # env.set_attr('desired_goal')
    # desired_goal=env.desired_goal['latent_desired_goal']
    # desired_goal = env.get_attr('desired_goal')[0]['latent_desired_goal']
    # print('desired_goal',desired_goal)
    # n_subgoals = env.env_method('generate_expert_subgoals',6)[0]
    env.use_vae_goals = False
    n_subgoals = env.generate_expert_subgoals(7)

    print(n_subgoals)
    print(env.use_vae_goals)
    print(getattr(env.env.wrapped_env.wrapped_env,'generate_expert_subgoals',None))
    import utils.torch.pytorch_util as ptu
    ptu.set_gpu_mode(True)
    n_subgoals_var = ptu.np_to_var(n_subgoals)
    imgs = env.vae.decode(n_subgoals_var)
    imgs_np = ptu.get_numpy(imgs)
    print('image_max_min',np.max(imgs_np[0]),np.min(imgs_np[0]),imgs_np[0].shape)
    dist = []
    dist.append(0)
    dist.append(np.linalg.norm(current_obs-n_subgoals[0],ord=1))
    for i in range(5):
        dist.append(np.linalg.norm(n_subgoals[i+1]-n_subgoals[i],ord=1))
    dist.append(np.linalg.norm(goal-n_subgoals[5],ord=1))
    fig,axes=plt.subplots(1,8,figsize=(8,8))
    print(axes)
    axes[0].cla()
    axes[0].imshow(start_img)
    axes[0].set_title(str(round(dist[0],2)))
    for i in range(6):
        axes[i+1].cla()
        axes[i+1].imshow((imgs_np[i].reshape(3,48,48).transpose(1,2,0)*255).astype('int'))
        axes[i+1].set_title(str(round(dist[i+1],2))+' '+str(round(np.linalg.norm(goal-n_subgoals[i],ord=1),2)))
    axes[7].cla()
    axes[7].imshow(start_img)

    plt.savefig('8_subgoal.png')




    # action = (env.action_space.sample()*0.2).reshape(-1,2)
    # print(action)
    # obs,reward,done,info = env.step(action)
    # obs=env.env_method('step',action)
    print('new_obs',obs)
    latent_goal_obs = obs[0,-16:]
    print('latent_goal_obs',latent_goal_obs)

if __name__ == '__main__':
    main()