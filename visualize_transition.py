import csv
import pandas
import numpy as np
import os
import sys
import pickle
from run_her import VAE_LOAD_PATH
import utils.torch.pytorch_util as ptu
import cv2
def decode_goal( vae_model,latents):
    latents = ptu.np_to_var(latents)
    latents = latents.view(-1, vae_model.representation_size)
    decoded = vae_model.decode(latents)
    return ptu.get_numpy(decoded)
if __name__ == '__main__':
    env_id = sys.argv[1]
    path = sys.argv[2]
    # reward_type = sys.argv[2]
    # model_path = sys.argv[3]
    def read_data(log_file, ):
        data = pandas.read_csv(log_file, index_col=None, comment='#', error_bad_lines=True)
        return data
    def get_item(data,label):
        return data[label].values
    list_value = []
    next_list_value = []
    data = read_data(os.path.join(path,'debug_transition.csv'))
    for i in range(48):
        list_value.append(get_item(data,'latent_'+str(i)))
        next_list_value.append(get_item(data, 'next_obs_latent_' + str(i)))
    latents=np.stack(list_value,axis=1)
    next_latents=np.stack(next_list_value,axis=1)
    ep_len = get_item(data,'episode_len')
    sum_ep_len=0
    start=int(ep_len[0])
    for i in range(1000):
        sum_ep_len +=int(start)
        start=ep_len[sum_ep_len]
    print('start_point',ep_len[sum_ep_len])
    # ep_len_1 =int(ep_len[0])
    # print(ep_len_1)
    # episode_1_latent = latents[:ep_len_1]
    episode_1_latent = latents[sum_ep_len:sum_ep_len+int(start)]

    # print(episode_1_latent)
    # episode_1_next_latent = next_latents[:ep_len_1]
    episode_1_next_latent = next_latents[sum_ep_len:sum_ep_len+int(start)]
    print(episode_1_latent.shape)
    print(episode_1_next_latent.shape)
    vae_file = open(VAE_LOAD_PATH[env_id], 'rb')
    vae_model = pickle.load(vae_file)
    vae_model.cuda()
    ptu.set_device(0)
    ptu.set_gpu_mode(True)
    latents_img = decode_goal(vae_model,episode_1_latent.reshape(-1,16))
    next_latent_img = decode_goal(vae_model,episode_1_next_latent.reshape(-1,16))
    vae_decoded_img = (
            latents_img[2].reshape(3, vae_model.imsize, vae_model.imsize).transpose(1, 2, 0) * 255).astype('uint8')
    cv2.imwrite(os.path.join(os.path.dirname(path), 'goal.png'),
                vae_decoded_img)
    for i in range(int(start)):
        vae_decoded_img= (
                    latents_img[i*3].reshape(3, vae_model.imsize, vae_model.imsize).transpose(1, 2, 0) * 255).astype('uint8')
        cv2.imwrite(os.path.join(os.path.dirname(path), 'obs%d.png' % i),
                    vae_decoded_img)

        vae_decoded_img_next = (
                next_latent_img[i*3].reshape(3, vae_model.imsize, vae_model.imsize).transpose(1, 2, 0) * 255).astype('uint8')
        cv2.imwrite(os.path.join(os.path.dirname(path), 'next_obs%d.png' % i),
                    vae_decoded_img_next)
    # print('regressor fit finished!')