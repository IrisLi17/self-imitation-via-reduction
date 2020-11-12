import os.path as osp
import time
import gym
import numpy as np
from masspoint_env import MasspointSMazeEnv

def generate_vae_dataset(variant):
    import cv2

    # env_class = variant.get('env_class', None)
    # env_kwargs = variant.get('env_kwargs',None)
    env_id = variant.get('env_id', None)
    N = variant.get('N', 10000)

    use_images = variant.get('use_images', True)

    imsize = variant.get('imsize', 84)
    show = variant.get('show', False)
    # init_camera = variant.get('init_camera', None)
    oracle_dataset = variant.get('oracle_dataset', False)
    if 'n_random_steps' in variant:
        n_random_steps = variant['n_random_steps']
    else:
        if oracle_dataset:
            n_random_steps = 3
        else:
            n_random_steps = 100
    # vae_dataset_specific_env_kwargs = variant.get('vae_dataset_specific_env_kwargs', None)
    # non_presampled_goal_img_is_garbage = variant.get('non_presampled_goal_img_is_garbage', None)
    from wrapper import ImageEnv
    info = {}
    logdir = variant.get('save_dir','/home/yilin/vae_data/')
    filename = osp.join(logdir, "vae_dataset.npy")

    now = time.time()

    if env_id is not None:
        import gym
        env = gym.make(env_id,reward_type='sparse',random_pusher=True)
    # else:
    #     if vae_dataset_specific_env_kwargs is None:
    #         vae_dataset_specific_env_kwargs = {}
    #     for key, val in env_kwargs.items():
    #         if key not in vae_dataset_specific_env_kwargs:
    #             vae_dataset_specific_env_kwargs[key] = val
    #     env = env_class(**vae_dataset_specific_env_kwargs)
    if not isinstance(env, ImageEnv):
        env = ImageEnv(
            env,
            imsize,
            transpose=True,
            normalize=True,
            camera_name="fixed",
        )
    else:
        imsize = env.imsize
        # env.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage
    env.reset()
    env._render_callback()
    info['env'] = env

    if use_images:
        data_size = env.observation_space.spaces['image_observation'].low.shape
    
        dtype = np.uint8
    else:
        data_size = len(env.observation_space.spaces['state_observation'].low)
        dtype = np.float32

    state_size = len(env.observation_space.spaces['state_observation'].low)

    dataset = {
        'obs': np.tile(np.zeros(data_size, dtype=dtype),[N,1,1,1]),
        'actions': np.zeros((N, len(env.action_space.low)), dtype=np.float32),
        'next_obs':np.tile( np.zeros(data_size, dtype=dtype),[N,1,1,1]),
        'reward': np.zeros((N,1),dtype=dtype),
        'obs_state': np.zeros((N, state_size), dtype=np.float32),
        'next_obs_state': np.zeros((N, state_size), dtype=np.float32),
    }

    for i in range(N):
        if i % (N/50) == 0:
            print(i)
        
        if oracle_dataset:
            if i % 100 == 0:
                env.reset()
            
                goal = env.sample_goal()
           #     print('goal:',goal)
                env.set_goal(goal)
                env._render_callback()
            for _ in range(n_random_steps):
                env.step(env.action_space.sample())
        else:
            env.reset()
            for _ in range(n_random_steps):
                env.step(env.action_space.sample())

        obs = env._get_obs()
        image_observation=obs['image_observation'].reshape(3, imsize, imsize).transpose((1, 2, 0))
        image_observation = image_observation[::, :, ::-1]
        if use_images:
            dataset['obs'][i, :,:,:] = unormalize_image(image_observation)
        else:
            dataset['obs'][i, :] = obs['state_observation']
        dataset['obs_state'][i, :] = obs['state_observation']

        action = env.action_space.sample()
        dataset['actions'][i, :] = action

        obs,reward = env.step(action)[:2]
        dataset['reward'][i,:]=reward
        img = obs['image_observation']
        next_obs=img.reshape(3, imsize, imsize).transpose((1, 2, 0))
        next_obs= next_obs[::, :, ::-1]
        if use_images:
            dataset['next_obs'][i, :,:,:] = unormalize_image(next_obs)
        else:
            dataset['next_obs'][i, :] = obs['state_observation']
        dataset['next_obs_state'][i, :] = obs['state_observation']
        if show or i%2000==0:
            img = img.reshape(3, imsize, imsize).transpose((1, 2, 0))
            img = img[::, :, ::-1]
            if show:
              cv2.imshow('img', img)
              cv2.waitKey(1000)
            else:
              img_name=osp.join(logdir,'img'+str(i)+'.jpg')  
              cv2.imwrite(img_name,unormalize_image(img))
        
    print("keys and shapes:")
    for k in dataset.keys():
        print(k, dataset[k].shape)
    print("done making training data", filename, time.time() - now)
    np.save(filename, dataset)

def unormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)

if __name__ == '__main__':
    MAX_EPISODE = 100
    gym.register('MasspointMaze-v2', entry_point=MasspointSMazeEnv, max_episode_steps=MAX_EPISODE)
    variant={'env_id':"MasspointMaze-v2",
             'N':50000,
             'use_images':True,
             'imsize':84,
             'show':False,
             'oracle_dataset':True,
             'save_dir':'/home/yilin/SIR/dataset_goal_episode/'

    }
    generate_vae_dataset(variant)
