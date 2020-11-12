import multiworld
import gym
import numpy as np
import torch
import utils.torch.pytorch_util as ptu
import csv, pickle
import cv2
from utils.wrapper import VAEWrappedEnv
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

env_id = 'Image48PointmassUWallTrainEnvBig-v0'

env = gym.register(
                    id=env_id,
                    entry_point=create_image_48_pointmass_uwall_train_env_big_v0,
                    tags={
                        'git-commit-hash': 'e5c11ac',
                        'author': 'Soroush'
                    },
                )
VAE_LOAD_PATH = {
    'Image84SawyerPushAndReachArenaTrainEnvBig-v0':'/home/yilin/leap/data/pnr/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_16_10_33_id000--s85192/vae.pkl',
    'Image84SawyerPushAndReachArenaTrainEnvBigUnlimit-v0': '/home/yilin/leap/data/pnr/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_16_10_33_id000--s85192/vae.pkl',

    'Image48PointmassUWallTrainEnvBig-v0':'/home/yilin/leap/data/pm/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_22_23_14_id000--s4047/vae.pkl',
    'Image48PointmassUWallTrainEnvBigUnlimit-v0': '/home/yilin/leap/data/pm/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_22_23_14_id000--s4047/vae.pkl',

}
file = open(VAE_LOAD_PATH[env_id],'rb')
vae_model = pickle.load(file)
ptu.set_gpu_mode(True)
ptu.set_device(0)
vae_model.cuda()
env = gym.make(env_id)
# env = VAEWrappedEnv(env,vae_model)
eval_env = env
env_render = eval_env
import os, time, argparse, imageio
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
obs = env_render.reset()
# while (np.argmax(obs[0][-2:]) != 0):
#     obs = env.reset()
# img = env.render(mode='rgb_array')
# load_path = logger.get_dir()
load_path='/home/yilin/sir_img/'
episode_reward = 0.0
num_episode = 0
frame_idx = 0
images = []
latents = []
latent_dists=[]
latent_order_dists=[]
for i in range(20):
    # print(env.unwrapped)
    # print(env_render.env.env.wrapped_env.wrapped_env)
    # img = env_render.env.env.wrapped_env.wrapped_env.render()
    img = env_render.get_image()
    # print(img.shape)
    # print(img)
    # import ipdb;ipdb.set_trace()
    img_input = (img.transpose()/255.0)
    img_var = ptu.np_to_var(img_input).contiguous()
    img_var.view(-1,6912)
    print(img_var.shape)
    print(vae_model.imlength)
    print(vae_model.added_fc_size)
    latent = vae_model.encode(img_var)[0]
    latent = ptu.get_numpy(latent)
    latents.append(latent)
    if len(latents)==1:
        latent_dist = 0
    else:
        print(latent_dist)
        latent_dist = np.linalg.norm(latents[i]-latents[i-1], ord=1)
    latent_dists.append(latent_dist)
    latent_dist_mean = sum(latent_dists)/len(latent_dists)
    latent_order_dist = np.linalg.norm(latents[i]-latents[0],ord=1)
    latent_order_dists.append(latent_order_dist)
    images.append(img)
    ax.cla()
    ax.imshow(img)
    # ax.set_title('episode ' + str(num_episode) + ', frame ' + str(frame_idx) +
    #              ', goal idx ' + str(np.argmax(obs[0][-2:])))
    ax.set_title('episode'+str(num_episode)+', frame'+str(frame_idx)+', latent_distance'+str(latent_dist)+',mean_distance'+
                 str(latent_dist_mean)+',order_dist'+str(latent_order_dist))

    # obs = img.reshape()
    # assert np.argmax(obs[0][-2:]) == 0
    # action, _ = model.predict(obs)
    action = env_render.action_space.sample()*3
    print('action', action)
    obs, reward, done, _ = env_render.step(action)
    episode_reward += reward
    frame_idx += 1
    export_gif = True
    if not export_gif:
        plt.pause(0.1)
    else:
     plt.savefig(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % i))
     copy = img[:,:,0].copy()
     img[:,:,0]=img[:,:,2]
     img[:,:,2]= copy
     cv2.imwrite(os.path.join(os.path.dirname(load_path),'image%d.png'%i),img)
    if done:
        obs = env_render.reset()
        # while (np.argmax(obs[0][-2:]) != 0):
        #     obs = env_render.reset()
        print('episode_reward', episode_reward)
        episode_reward = 0.0
        frame_idx = 0
        num_episode += 1
        if num_episode >= 5:
            break
# imageio.mimsave(env_name + '.gif', images)

# if export_gif:
#     os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(
#         load_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
#               os.path.join(os.path.dirname(load_path), env_id + '.mp4'))
#     for i in range(500):
#         # images.append(plt.imread('tempimg' + str(i) + '.png'))
#         try:
#             os.remove(os.path.join(os.path.dirname(load_path), 'tempimg' + str(i) + '.png'))
#         except:
#             pass
