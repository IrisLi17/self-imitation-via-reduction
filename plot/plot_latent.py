import os
import pandas
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import utils.torch.pytorch_util as ptu
import pickle
from sklearn.neighbors import KNeighborsRegressor
from run_her import VAE_LOAD_PATH
if __name__ == '__main__':
    # option = sys.argv[1]
    # log_paths = sys.argv[2:]
    data = pandas.read_csv('latent_data.csv')
    latent = []
    for i in range(4):
        piece = []
        for j in range(8):
            piece.append(np.fromstring(data.values[i*8+j][0],sep=' '))
        latent.append(np.concatenate(piece))
        piece=[]
    latent = np.concatenate(latent)
    print('latent',latent.shape,latent)

    env_id = 'Image84SawyerPushAndReachArenaTrainEnvBig-v0'
    vae_file = open(VAE_LOAD_PATH[env_id], 'rb')
    vae_model = pickle.load(vae_file)
    vae_model.cuda()
    # train_latent = np.load('sawyer_dataset_train_latents_all_21.npy')
    # train_state = np.load('sawyer_dataset_train_states_all_21.npy')
    # print('training_dataset_size', train_latent.shape[0])
    # regressor = KNeighborsRegressor()
    # regressor.fit(train_latent, train_state)
    # print('regressor fit finished!')
    import utils.torch.pytorch_util as ptu
    # data = pandas.read_csv(log_paths+'debug_value.csv')
    # np_str = ''
    # for i in range(data.nrows):
    #     np_str += data_nrows
