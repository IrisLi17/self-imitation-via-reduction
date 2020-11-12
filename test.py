# import gym
# from masspoint_env import MasspointSMazeEnv
# import cv2
# import imageio
#
# from baselines import PPO2_augment, PPO2_augment_sil
# from stable_baselines import logger
# from stable_baselines.bench import Monitor
# from stable_baselines.common import set_global_seeds
# from stable_baselines.common.vec_env import SubprocVecEnv
# from utils.parallel_subproc_vec_env import ParallelSubprocVecEnv
# from gym.wrappers import FlattenDictWrapper
# from stable_baselines.common.policies import register_policy
#
# from push_wall_obstacle import FetchPushWallObstacleEnv_v4
# from masspoint_env import MasspointPushSingleObstacleEnv_v2, MasspointPushDoubleObstacleEnv
# from masspoint_env import MasspointMazeEnv, MasspointSMazeEnv
# from fetch_stack import FetchStackEnv
# # from push_wall import FetchPushWallEnv
# # from push_box import FetchPushBoxEnv
# import gym
# from utils.wrapper import DoneOnSuccessWrapper,VAEWrappedEnv
# import pickle
# import numpy as np
# import csv, pickle
# import multiworld
# import os, time, argparse, imageio
# import matplotlib.pyplot as plt
#
#
# ENTRY_POINT = {'FetchPushWallObstacle-v4': FetchPushWallObstacleEnv_v4,
#                'FetchPushWallObstacleUnlimit-v4': FetchPushWallObstacleEnv_v4,
#                # 'FetchPushWall-v1': FetchPushWallEnv,
#                # 'FetchPushBox-v1': FetchPushBoxEnv,
#                }
# MASS_ENTRY_POINT = {
#     'MasspointPushSingleObstacle-v2': MasspointPushSingleObstacleEnv_v2,
#     'MasspointPushSingleObstacleUnlimit-v2': MasspointPushSingleObstacleEnv_v2,
#     'MasspointPushDoubleObstacle-v1': MasspointPushDoubleObstacleEnv,
#     'MasspointPushDoubleObstacleUnlimit-v1': MasspointPushDoubleObstacleEnv,
#     'MasspointMaze-v1': MasspointMazeEnv,
#     'MasspointMazeUnlimit-v1': MasspointMazeEnv,
#     'MasspointMaze-v2': MasspointSMazeEnv,
#     'MasspointMazeUnlimit-v2': MasspointSMazeEnv,
# }
#
# PICK_ENTRY_POINT = {
#     'FetchStack-v1': FetchStackEnv,
#     'FetchStackUnlimit-v1': FetchStackEnv,
# }
# IMAGE_ENTRY_POINT = {
#     'Image84SawyerPushAndReachArenaTrainEnvBig-v0':  'ImagePushAndReach',
#     'Image84SawyerPushAndReachArenaTrainEnvBigUnlimit-v0':  'ImagePushAndReach',
#     'Image48PointmassUWallTrainEnvBig-v0':'ImageUWall',
#     'Image48PointmassUWallTrainEnvBigUnlimit-v0': 'ImageUWall',
#
# }
# VAE_LOAD_PATH = {
#     'Image84SawyerPushAndReachArenaTrainEnvBig-v0':'/home/yilin/leap/data/pnr/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_16_10_33_id000--s85192/vae.pkl',
#     'Image84SawyerPushAndReachArenaTrainEnvBigUnlimit-v0': '/home/yilin/leap/data/pnr/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_16_10_33_id000--s85192/vae.pkl',
#
#     'Image48PointmassUWallTrainEnvBig-v0':'/home/yilin/leap/data/pm/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_22_23_14_id000--s4047/vae.pkl',
#     'Image48PointmassUWallTrainEnvBigUnlimit-v0': '/home/yilin/leap/data/pm/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_22_23_14_id000--s4047/vae.pkl',
#
# }
# def configure_logger(log_path, **kwargs):
#     if log_path is not None:
#         logger.configure(log_path)
#     else:
#         logger.configure(**kwargs)
#
# def create_image_84_sawyer_pnr_arena_train_env_big_v0():
#     from multiworld.core.image_env import ImageEnv
#     from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_tdm_v4
#
#     wrapped_env = gym.make('SawyerPushAndReachArenaTrainEnvBig-v0')
#     return ImageEnv(
#         wrapped_env,
#         84,
#         init_camera=sawyer_pusher_camera_tdm_v4,
#         transpose=True,
#         normalize=True,
#         reward_type='sparse'
#     )
# def create_image_48_pointmass_uwall_train_env_big_v0():
#     from multiworld.core.image_env import ImageEnv
#
#     wrapped_env = gym.make('PointmassUWallTrainEnvBig-v0')
#     return ImageEnv(
#         wrapped_env,
#         48,
#         init_camera=None,
#         transpose=True,
#         normalize=True,
#         non_presampled_goal_img_is_garbage=False,
#     )
#
# def make_env(env_id, seed, rank, log_dir=None, allow_early_resets=True, kwargs=None):
#     """
#     Create a wrapped, monitored gym.Env for MuJoCo.
#
#     :param env_id: (str) the environment ID
#     :param seed: (int) the inital seed for RNG
#     :param allow_early_resets: (bool) allows early reset of the environment
#     :return: (Gym Environment) The mujoco environment
#     """
#     if env_id in ENTRY_POINT.keys() or env_id in MASS_ENTRY_POINT.keys() or env_id in PICK_ENTRY_POINT.keys() or env_id in IMAGE_ENTRY_POINT.keys():
#         # env = ENTRY_POINT[env_id](**kwargs)
#         # print(env)
#         # from gym.wrappers.time_limit import TimeLimit
#         kwargs = kwargs.copy()
#         max_episode_steps = None
#         if env_id in IMAGE_ENTRY_POINT.keys():
#             if IMAGE_ENTRY_POINT[env_id] == 'ImagePushAndReach':
#                 gym.register(
#                     id=env_id,
#                     entry_point=create_image_84_sawyer_pnr_arena_train_env_big_v0,
#                     max_episode_steps=max_episode_steps,
#                     tags={
#                         'git-commit-hash': 'e5c11ac',
#                         'author': 'Soroush'
#                     },
#                 )
#             elif IMAGE_ENTRY_POINT[env_id] =='ImageUWall':
#                 gym.register(
#                     id=env_id,
#                     entry_point=create_image_48_pointmass_uwall_train_env_big_v0,
#                     tags={
#                         'git-commit-hash': 'e5c11ac',
#                         'author': 'Soroush'
#                     },
#                 )
#             env = gym.make(env_id)
#             # env.wrapped_env.reward_type='sparse'
#             # vae_file = open(VAE_LOAD_PATH[env_id],'rb')
#             # import utils.torch.pytorch_util as ptu
#             # ptu.set_gpu_mode(True)
#             # vae_model = pickle.load(vae_file)
#             # env = VAEWrappedEnv(env,vae_model,epsilon=1.0,imsize=48,reward_params=dict(type='latent_distance2'))
#             # print(env.observation_space.spaces.keys())
#             # env.wrapped_env.reward_type='wrapped_env'
#             # env.reward_type='latent_sparse'
#         else:
#             if 'max_episode_steps' in kwargs:
#                 max_episode_steps = kwargs['max_episode_steps']
#                 del kwargs['max_episode_steps']
#             if env_id in ENTRY_POINT.keys():
#                 gym.register(env_id, entry_point=ENTRY_POINT[env_id], max_episode_steps=max_episode_steps, kwargs=kwargs)
#             elif env_id in MASS_ENTRY_POINT.keys():
#                 gym.register(env_id, entry_point=MASS_ENTRY_POINT[env_id], max_episode_steps=max_episode_steps, kwargs=kwargs)
#             elif env_id in PICK_ENTRY_POINT.keys():
#                 gym.register(env_id, entry_point=PICK_ENTRY_POINT[env_id], max_episode_steps=max_episode_steps,
#                              kwargs=kwargs)
#             env = gym.make(env_id)
#
#
#         # env = TimeLimit(env, max_episode_steps=50)
#     else:
#         env = gym.make(env_id, reward_type='sparse')
#     # env = FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
#     # env = FlattenDictWrapper(env,['latent_observation','latent_achieved_goal','latent_desired_goal'])
#     # if env_id in PICK_ENTRY_POINT.keys() and kwargs['reward_type'] == 'dense':
#     #     env = DoneOnSuccessWrapper(env, reward_offset=0.0)
#     # else:
#     #     env = DoneOnSuccessWrapper(env)
#     # if log_dir is not None:
#     #     env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=allow_early_resets,
#     #                   info_keywords=('is_success',))
#     # # env.seed(seed + 10000 * rank)
#     return env
#
#
# IM_SIZE=84
# MAX_EPISODE=100
# # gym.register('MasspointMaze-v2',entry_point=MasspointSMazeEnv,max_episode_steps=100)
# #
# # env=gym.make('MasspointMaze-v2')
# env_id = 'Image84SawyerPushAndReachArenaTrainEnvBig-v0'
# env = make_env(env_id=env_id,seed=0,rank=0,kwargs={})
# for i in range(10):
#    obs=env.reset()
#    img = (obs['image_observation'].reshape(3,84,84).transpose(1,2,0)*255).astype('uint8')
#    # img = env.get_image_with_goal(width=400,height=400)
#    copy = img[:, :, 0].copy()
#    img[:, :, 0] = img[:, :, 2]
#    img[:, :, 2] = copy
#    cv2.imwrite('img_env%d.png'%i,img)

# # img = env.sim.render(width=IM_SIZE,height=IM_SIZE,camera_name="fixed")
# episode_reward=0.0
# images=[]
# frame_idx=0
# for i in range(env.spec.max_episode_steps*6):
#    action=env.action_space.sample()
#    obs,reward,done,_=env.step(action)
#    episode_reward+=reward
#    frame_idx+=1
#    img_obs=env.sim.render(width=IM_SIZE,height=IM_SIZE,camera_name="fixed")
#    images.append(img_obs)
#    if done:
#       obs=env.reset()
#       print('episode_reward',episode_reward)
# imageio.mimsave('smaz.gif',images)
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# x = np.arange(40)
# y = np.arange(40)
# z= np.arange(40)
# z_1 = np.arange(40)+1
# fig = plt.figure()
# # ax = fig.gca(projection='3d')
# ax = Axes3D(fig)
# ax.scatter(x,y,z,marker='o',c='tab:orange',label='z')
# ax.scatter(x,y,z_1,marker='^',c='tab:blue',label='z_1')
# fig.legend(loc="lower right", )
# fig.savefig('test.png')
from run_her import VAE_LOAD_PATH,make_env,get_env_kwargs
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
def decode_goal( vae_model,latents):
    latents = ptu.np_to_var(latents)
    latents = latents.view(-1, vae_model.representation_size)
    decoded = vae_model.decode(latents)
    return ptu.get_numpy(decoded)
env_id = 'Image84SawyerPushAndReachArenaTrainEnvBig-v0'
vae_file = open(VAE_LOAD_PATH[env_id], 'rb')
vae_model = pickle.load(vae_file)
vae_model.cuda()
import utils.torch.pytorch_util as ptu
ptu.set_device(0)
ptu.set_gpu_mode(True)
reward_type = 'dense'
reward_threshold=0.08
env_kwargs = get_env_kwargs(env_id, random_ratio=0.0, reward_type=reward_type, reward_object=1,
                            reward_threshold=reward_threshold)

env = make_env(env_id, seed=0, rank=0, kwargs=env_kwargs)
# states = np.load('sawyer_dataset_states.npy')
# latents = np.load('sawyer_dataset_latents.npy')
print('sim data',env.sim.model._joint_id2name)
print('sim data body',env.sim.model._body_id2name)
print('sim data qpos',env.sim.data.qpos)
# train_latent = np.load('sawyer_dataset_latents_total_all_21.npy')
# train_state = np.load('sawyer_dataset_states_total_all_21.npy')
# regressor = KNeighborsRegressor()
# regressor.fit(train_latent, train_state)
# # state = train_state[255]
# state = np.array([-0.21,0.48,0.18,0.61999999])
# latent = train_latent[255]
# print('input state',state)
# img = decode_goal(vae_model,latent)
# goal = dict(image_desired_goal=img[0],state_desired_goal=state)
# env.set_goal(goal)
# base_state = env.get_state()
# print('base_state',base_state)
# print('base_image',img[0])
# current_obs = env.get_obs()
# current_state=current_obs['state_observation']
# current_goal = current_obs['state_desired_goal']
# print('current_state',current_state)
# print('current_goal',current_goal)
# print('current_image',current_obs['image_observation'])
# action = env.action_space.sample()
# obs,reward,info,done = env.step(action)
# print('next_state',obs['state_observation'])
# print('next_image',obs['image_observation'])
# env.set_state(base_state)
# base_revised_state=env.get_state()
# print('base_revised_state',base_revised_state)
# revised_obs = env.get_obs()
# print('revised_state',revised_obs['state_observation'])
# print('revised_image',revised_obs['image_observation'])
# subgoal_latents = []
# img_subgoal_np = np.load('sawyer_dataset_imgs_total_all_21.npy')
# batch_size = 10000
# print('img_size',img_subgoal_np.shape[0])
# img_subgoal_part = img_subgoal_np[16*batch_size:18*batch_size]
# img_subgoal_batch = ptu.np_to_var(img_subgoal_part)
# batch_latent = vae_model.encode(img_subgoal_batch)[0]
# latent_np = ptu.get_numpy(batch_latent)
# np.save('sawyer_dataset_latents_20000_9.npy',latent_np)
# # latents = None
# for i in range(floor(len(img_subgoal_buf)/batch_size)+1):
#     if (i+1)*batch_size<len(img_subgoal_buf):
#         img_subgoal_batch = ptu.np_to_var(img_subgoal_np[:(i+1)*batch_size])
#     else:
#         img_subgoal_batch = ptu.np_to_var(img_subgoal_np[i*batch_size:])
#     batch_latent = vae_model.encode(img_subgoal_batch)[0]
#     latent_np = ptu.get_numpy(batch_latent)
#     # if latents is not None:
#     #     # torch.cat((latents, batch_latent), dim=0)
#     # else:
#         # latents = batch_latent
#
#     subgoal_latents.append(latent_np)
#     print('%d *5000 latents encoded!',i+1)
# subgoal_latents = np.concatenate(subgoal_latents)
# np.save('sawyer_dataset_latents_total_all_21.npy',np.array(subgoal_latents))