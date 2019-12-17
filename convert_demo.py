import os, pickle
import numpy as np

if __name__=='__main__':
    sanity_data_path = './logs/sanity_data'
    n_episodes = 5
    save_path = os.path.join(sanity_data_path, 'demo.npz')
    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes * 2,))
    episode_starts = []
    episode_starts.append(True)
    for i in range(n_episodes):
        with open(os.path.join(sanity_data_path, 'augment_episode%d.pkl' % i), 'rb') as f:
            augment_buf = pickle.load(f)
            _observation = np.asarray([item[0] for item in augment_buf])
            _observation[:, 40:43] = _observation[:, 3:6]
            _observation[:, 43:45] = np.array([1.0, 0.0])
            _observation[:, 45:50] = _observation[0, 45:50]
            observations += [item[0] for item in augment_buf]
            observations += [_observation[j] for j in range(_observation.shape[0])]
            actions += [item[1] for item in augment_buf]
            actions += [item[1] for item in augment_buf]
            rewards += [item[2] for item in augment_buf]
            rewards += [item[2] for item in augment_buf] # TODO
            episode_returns[2*i] = np.sum([item[2] for item in augment_buf])
            episode_returns[2*i + 1] = np.sum([item[2] for item in augment_buf]) # TODO
            episode_starts += [bool(item[4]) for item in augment_buf]
            episode_starts += [bool(item[4]) for item in augment_buf]
    observations = np.asarray(observations)
    actions = np.asarray(actions)
    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])
    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)
    if save_path is not None:
        np.savez(save_path, **numpy_dict)
