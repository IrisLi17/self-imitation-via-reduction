import os
import pandas
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import utils.torch.pytorch_util as ptu
import pickle
import cv2
from sklearn.neighbors import KNeighborsRegressor
from run_her import VAE_LOAD_PATH
def decode_goal( vae_model,latents):
    latents = ptu.np_to_var(latents)
    latents = latents.view(-1, vae_model.representation_size)
    decoded = vae_model.decode(latents)
    return ptu.get_numpy(decoded)
if __name__ == '__main__':
    # option = sys.argv[1]
    # log_paths = sys.argv[2:]
    model_path= sys.argv[1]
    file_name = sys.argv[2]
    data = pandas.read_csv(file_name)
    latent = []
    print(data.shape)
    for i in range(6):
        piece = []
        for j in range(12):
            piece.append(np.fromstring(data.values[i*12+j][0],sep=' '))
        latent.append(np.concatenate(piece))
        piece=[]
    latent = np.concatenate(latent)
    latent = latent.reshape(-1,16)
    print('latent',latent.shape,latent)
    # latent= np.array([ 7.82720186e-03 , 1.09563506e+00 ,-4.95747596e-01 ,-1.17927855e-02,
    #                     -9.17500854e-02,  1.01639891e+00, - 4.37417120e-01, - 4.06024426e-01,
    #                     3.01966257e-03, - 1.47734229e-02 , 7.65015371e-04, - 2.62426324e-02,
    #                     - 1.51530695e+00,  3.89169902e-02 ,- 1.39568467e-03 ,- 7.72636354e-01,
    #                     5.94875142e-02,  1.40232593e-01 ,- 7.13598371e-01, - 4.69293632e-02,
    #                     - 6.10034727e-02,  9.92312014e-01,  7.74421215e-01, - 1.48523498e+00,
    #                     6.88003525e-02, - 6.89898729e-02 , 9.07940138e-03,  9.36824828e-02,
    #                     - 2.00288343e+00,  5.77090979e-01, - 5.94794974e-02, - 3.51175249e-01,
    #                     1.06643960e-02, - 1.43460798e+00 ,- 7.34168112e-01, - 2.08988525e-02,
    #                     6.18906915e-02,  8.91335011e-01, - 1.02260721e+00, - 1.20616481e-02,
    #                     2.10962351e-02, - 2.74828356e-03,  1.87131129e-02,  1.90735906e-02,
    #                     - 9.80954707e-01, - 6.31270647e-01,  1.19928494e-02, - 3.56358051e-01
    #
    #                   ])
    # latent=latent.reshape(-1,16)
    env_id = 'Image84SawyerPushAndReachArenaTrainEnvBig-v0'
    vae_file = open(VAE_LOAD_PATH[env_id], 'rb')
    vae_model = pickle.load(vae_file)
    vae_model.cuda()
    import utils.torch.pytorch_util as ptu

    ptu.set_device(0)
    ptu.set_gpu_mode(True)
    img_plot = decode_goal(vae_model, latent)
    for i in range(6):
        print(os.path.join(os.path.dirname(model_path), 'obs' + str(i) + 'episode_' + str(i) + '.png'))

        obs = (
                img_plot[3*i].reshape(3, vae_model.imsize, vae_model.imsize).transpose(1, 2, 0) * 255).astype(
            'uint8')
        cv2.imwrite(
            os.path.join(os.path.dirname(model_path), 'obs' + str(i) + 'episode_' + str(i) + '.png'),
            obs)
        subgoal = (
                img_plot[3*i+1].reshape(3, vae_model.imsize, vae_model.imsize).transpose(1, 2, 0) * 255).astype(
            'uint8')
        cv2.imwrite(os.path.join(os.path.dirname(model_path),
                                 'subgoal_' + str(i) + 'episode_' + str(i) + '.png'),
                   subgoal)
        final_goal= (
                img_plot[3*i+2].reshape(3, vae_model.imsize, vae_model.imsize).transpose(1, 2, 0) * 255).astype(
            'uint8')
        cv2.imwrite(os.path.join(os.path.dirname(model_path),
                         'final_goal_' + str(i) + 'episode_' + str(i) + '.png'),
            final_goal)

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
