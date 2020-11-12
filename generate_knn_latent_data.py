from run_ppo_augment import make_env
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
#
# env = make_env('Image84SawyerPushAndReachArenaTrainEnvBig-v0',  1, 0, 0, None,
#                kwargs=dict(max_episode_steps=100, reward_type="state_distance"))
# latents = []  # No matter it is observation or goal
# states = []
# for i in range(2500):
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
# np.save('sawyer_dataset_latents_episode_all.npy', np.array(latents))
# np.save('sawyer_dataset_states_episode_all.npy', np.array(states))
# indices = np.arange(len(latents))
# np.random.shuffle(indices)
# test_size = int(len(latents)*0.2)
# latents = np.array(latents)
# states = np.array(states)
# latents_train = latents[indices[test_size:]]
# latents_test = latents[indices[:test_size]]
# states_train = states[indices[test_size:]]
# states_test = states[indices[:test_size]]
# np.save('sawyer_dataset_latents_episode_train.npy', np.array(latents_train))
# np.save('sawyer_dataset_states_episode_train.npy', np.array(states_train))
# np.save('sawyer_dataset_latents_episode_test.npy', np.array(latents_test))
# np.save('sawyer_dataset_states_episode_test.npy', np.array(states_test))

test_states = np.load('sawyer_dataset_states_episode_test.npy')
test_latents = np.load('sawyer_dataset_latents_episode_test.npy')
# test_states = np.load('sawyer_dataset_train_states_all.npy')
# test_latents = np.load('sawyer_dataset_train_latents_all.npy')


train_states = np.load('sawyer_dataset_states_episode_train.npy')
train_latents = np.load('sawyer_dataset_latents_episode_train.npy')
regressor = KNeighborsRegressor()
regressor.fit(train_latents,train_states)
predict_states=regressor.predict(test_latents)
errors  = np.abs(predict_states-test_states)
l2_errors = np.linalg.norm(predict_states-test_states,axis=1)
l2_hand_errors = np.linalg.norm(predict_states[:,:2]-test_states[:,:2],axis=1)
l2_puck_errors = np.linalg.norm(predict_states[:,-2:]-test_states[:,-2:],axis=1)
print('mean error', np.mean(errors), 'along each dim are', [np.mean(errors[:, i]) for i in range(errors.shape[1])])
print('l2_error',np.mean(l2_errors),'along the hand: ',np.mean(l2_hand_errors),'along the puck: ',np.mean(l2_puck_errors))