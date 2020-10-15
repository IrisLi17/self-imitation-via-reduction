from sklearn.neighbors import KNeighborsRegressor
import numpy as np

train_latent = np.load('sawyer_dataset_latents.npy')
train_state = np.load('sawyer_dataset_states.npy')
test_latent = np.load('sawyer_testset_latents.npy')
test_state = np.load('sawyer_testset_states.npy')
regressor = KNeighborsRegressor()
regressor.fit(train_latent, train_state)
predictions = regressor.predict(test_latent)
errors = np.abs(predictions - test_state)
print('mean error', np.mean(errors), 'along each dim are', [np.mean(errors[:, i]) for i in range(errors.shape[1])])
