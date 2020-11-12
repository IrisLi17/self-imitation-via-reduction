
import pickle
import numpy as np
import sys, os
import numpy as np
from run_her import make_env, get_env_kwargs
from baselines import HER_HACK
from gym.wrappers import FlattenDictWrapper
from utils.parallel_subproc_vec_env2 import ParallelSubprocVecEnv as ParallelSubprocVecEnv2
from utils.subproc_vec_vae_env2 import ParallelVAESubprocVecEnv as SubprocVaeEnv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from baselines import SAC_augment
from sklearn.neighbors import KNeighborsRegressor
import utils.torch.pytorch_util as ptu
import pickle
from sklearn.neighbors import KNeighborsRegressor
from run_her import VAE_LOAD_PATH
from math import floor
import cv2
from mpl_toolkits.mplot3d import Axes3D
import csv
import torch

# env_id = sys.argv[1]
# reward_type = sys.argv[2]
# # model_path = sys.argv[3]
# reward_threshold = float(sys.argv[3])
# # n_trials = int(sys.argv[5])
# n_interval = int(sys.argv[4])
# env_kwargs = get_env_kwargs(env_id, random_ratio=0.0, reward_type=reward_type, reward_object=1,
#                             reward_threshold=reward_threshold)
# log_dir = 'logs/tmp'
# vae_file = open(VAE_LOAD_PATH[env_id], 'rb')
# vae_model = pickle.load(vae_file)
# vae_model.cuda()
# import utils.torch.pytorch_util as ptu
#
# ptu.set_device(0)
# ptu.set_gpu_mode(True)
# # aug_env_id = env_id.split('-')[0] + 'Unlimit-' + env_id.split('-')[1]
# aug_env_id = 'Image84SawyerPushAndReachArenaTrainEnvBig-v0'
# aug_env_kwargs = env_kwargs.copy()
# aug_env_kwargs['max_episode_steps'] = 100
#
# # def make_thunk_aug(rank):
# #     return lambda: FlattenDictWrapper(make_env(env_id=aug_env_id, seed=0, rank=rank, kwargs=aug_env_kwargs),
# #                                       ['observation', 'achieved_goal', 'desired_goal'])
# #
# # aug_env = ParallelSubprocVecEnv([make_thunk_aug(i) for i in range(1)])
# env = make_env(aug_env_id, seed=0, rank=0, kwargs=aug_env_kwargs)
#
# # env = make_env('Image84SawyerPushAndReachArenaTrainEnvBig-v0',  1, 0, 0, None,
# #                kwargs=dict(max_episode_steps=100, reward_type="state_distance"))
# # latents = []  # No matter it is observation or goal
# # states = []
# # select 21*21 point in the state space as the subgoal
# hand_space = env.wrapped_env.wrapped_env.hand_space
#
# pos_x, pos_y = np.meshgrid(np.linspace(hand_space.low[0], hand_space.high[0], n_interval),
#                            np.linspace(hand_space.low[1], hand_space.high[1], n_interval))
# # set the hand_pos to be over the whole space and the puck dist should be out of 0.055
# hand_pos_xy = np.concatenate([pos_x.reshape(-1, 1), pos_y.reshape(-1, 1)], axis=1)
# hand_pos_repeat_1 = np.repeat(hand_pos_xy,n_interval**2,axis=0)
# hand_pos_repeat_2 = np.tile(hand_pos_xy.reshape(2*(n_interval**2)),n_interval**2).reshape(n_interval**4,2)
# index = np.where(np.linalg.norm(hand_pos_repeat_1-hand_pos_repeat_2,axis=1)>0.055)
# hand_puck_pos_list = np.concatenate([hand_pos_repeat_1,hand_pos_repeat_2],axis=1)
# hand_puck_pos_filter = hand_puck_pos_list[index]
# pos_xy = hand_puck_pos_filter
# print('pos_xy',pos_xy)
# np.save('sawyer_dataset_states_total_all_21.npy',np.array(pos_xy))
# img_subgoal_buf = []
# for i in range(pos_xy.shape[0]):
#     if i%1000 == 0:
#         print('set_position %d is finished: ',i)
#     obs = env.set_state_xypos(pos_xy[i])
#     if not isinstance(obs, dict):
#         print('wrong pos setting', pos_xy[i])
#         exit()
#
#     img_subgoal_buf.append(obs['image_observation'])
# batch_size = 5000
# img_subgoal_np = np.stack(img_subgoal_buf)
# np.save('sawyer_dataset_states_total_all_21.npy',np.array(pos_xy))
# np.save('sawyer_dataset_imgs_total_all_21.npy',np.array(img_subgoal_np))
#
# subgoal_latents = []
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
# pos_xy = np.load('sawyer_dataset_states_total_all_21.npy')
# subgoal_latents = []
# for i in range(9):
#     latent_part = np.load('sawyer_dataset_latents_20000_'+str(i+1)+'.npy')
#     subgoal_latents.append(latent_part)
# subgoal_latents = np.concatenate(subgoal_latents)
# np.save('sawyer_dataset_latents_total_all_21.npy',np.array(subgoal_latents))
# pos_xy = np.load('sawyer_dataset_states_total_all.npy')
# subgoal_latents = np.load('sawyer_dataset_latents_total_all_21.npy')
# dataset_size = pos_xy.shape[0]
# test_size = int(dataset_size*0.2)
# print('dataset_size: ',dataset_size,'test_size: ',test_size)
# indices = np.arange(dataset_size)
# np.random.shuffle(indices)
# train_states = pos_xy[indices[test_size:]]
# train_latents = subgoal_latents[indices[test_size:]]
# test_states = pos_xy[indices[:test_size]]
# test_latents = subgoal_latents[indices[:test_size]]
# np.save('sawyer_dataset_train_latents_all_21.npy', np.array(train_latents))
# np.save('sawyer_dataset_train_states_all_21.npy', np.array(train_states))
# np.save('sawyer_dataset_test_latents_all_21.npy', np.array(test_latents))
# np.save('sawyer_dataset_test_states_all_21.npy', np.array(test_states))
train_latents = np.load('sawyer_dataset_train_latents_all_21.npy')
train_states= np.load('sawyer_dataset_train_states_all_21.npy')
test_latents = np.load('sawyer_dataset_train_latents_all.npy')
test_states= np.load('sawyer_dataset_train_states_all.npy')
# fit the knn regressor and make the prediction
regressor = KNeighborsRegressor()
regressor.fit(train_latents, train_states)
predictions = regressor.predict(test_latents)
errors = np.abs(predictions - test_states)
l2_errors = np.linalg.norm(predictions-test_states,axis=1)
l2_error_hand = np.linalg.norm(predictions[:,:2]-test_states[:,:2],axis=1)
l2_error_puck = np.linalg.norm(predictions[:,-2:]-test_states[:,-2:],axis=1)

print('mean error', np.mean(errors), 'along each dim are', [np.mean(errors[:, i]) for i in range(errors.shape[1])])
print('l2_error',np.mean(l2_errors),'along the hand: ',np.mean(l2_error_hand),'along the puck: ',np.mean(l2_error_puck))
# puck_pos_xy = np.repeat(init_state_goal[-2:].reshape(1, 2), n_interval ** 2, axis=0)
# index = np.where(np.linalg.norm(hand_pos_xy - puck_pos_xy, axis=1) > 0.055)
# pos_xy = np.concatenate([hand_pos_xy, puck_pos_xy], axis=1)[index]
# for i in range(2000):
#     env.reset()
#     vae_wrapped_obs = env.get_obs()
#     latent_obs = vae_wrapped_obs['observation']
#     latent_goal = vae_wrapped_obs['desired_goal']
#     latents.append(latent_obs)
#     latents.append(latent_goal)
#     # print('latent_obs shape', latent_obs.shape, latent_goal.shape)  (16)
#     raw_obs = env.unwrapped.wrapped_env.get_obs()
#     state_obs = raw_obs['state_observation']
#     state_goal = raw_obs['state_desired_goal']
#     states.append(state_obs)
#     states.append(state_goal)
#     image_obs = raw_obs['image_observation']
#     image_goal = raw_obs['image_desired_goal']
#     # print('state_obs shape', state_obs.shape, state_goal.shape)  (4)
#     # print('image_obs shape', image_obs.shape, image_goal.shape)
#     # with open('sawyer_dataset.pkl', 'ab') as f:
#     #     pickle.dump(dict(latent_obs=latent_obs, latent_goal=latent_goal,
#     #                      state_obs=state_obs, state_goal=state_goal), f)
#     if i % 100 == 0:
#         print('generated', i, 'samples')
# np.save('sawyer_dataset_latents_all.npy', np.array(latents))
# np.save('sawyer_dataset_states_all.npy', np.array(states))