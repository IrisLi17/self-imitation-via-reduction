from run_ppo_augment import make_env
import pickle
import numpy as np


env = make_env('Image84SawyerPushAndReachArenaTrainEnvBig-v0',  1, 0, 0, None,
               kwargs=dict(max_episode_steps=100, reward_type="state_distance"))
latents = []  # No matter it is observation or goal
states = []
for i in range(2000):
    env.reset()
    vae_wrapped_obs = env.get_obs()
    latent_obs = vae_wrapped_obs['observation']
    latent_goal = vae_wrapped_obs['desired_goal']
    latents.append(latent_obs)
    latents.append(latent_goal)
    # print('latent_obs shape', latent_obs.shape, latent_goal.shape)  (16)
    raw_obs = env.unwrapped.wrapped_env.get_obs()
    state_obs = raw_obs['state_observation']
    state_goal = raw_obs['state_desired_goal']
    states.append(state_obs)
    states.append(state_goal)
    image_obs = raw_obs['image_observation']
    image_goal = raw_obs['image_desired_goal']
    # print('state_obs shape', state_obs.shape, state_goal.shape)  (4)
    # print('image_obs shape', image_obs.shape, image_goal.shape)
    # with open('sawyer_dataset.pkl', 'ab') as f:
    #     pickle.dump(dict(latent_obs=latent_obs, latent_goal=latent_goal,
    #                      state_obs=state_obs, state_goal=state_goal), f)
    if i % 100 == 0:
        print('generated', i, 'samples')
np.save('sawyer_dataset_latents.npy', np.array(latents))
np.save('sawyer_dataset_states.npy', np.array(states))