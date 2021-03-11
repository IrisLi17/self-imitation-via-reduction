from stable_baselines.deepq import ReplayBuffer, PrioritizedReplayBuffer
import numpy as np
import random


class DoublePrioritizedReplayWrapper(object):
    def __init__(self, buffer1, buffer2):
        assert isinstance(buffer1, PrioritizedReplayBuffer)
        assert isinstance(buffer2, PrioritizedReplayBuffer)
        self.buffer1 = buffer1
        self.buffer2 = buffer2
        self.min_tree_operation = buffer1._it_min._operation
        self.sum_tree_operation = buffer1._it_sum._operation

    def _sample_proportional(self, batch_size):
        res1, res2 = [], []
        _sum1 = self.buffer1._it_sum.sum()
        _sum2 = self.buffer2._it_sum.sum()
        for i in range(batch_size):
            mass = random.random() * (_sum1 + _sum2)
            if mass < _sum1:
                idx = self.buffer1._it_sum.find_prefixsum_idx(mass)
                res1.append(idx)
            else:
                idx = self.buffer2._it_sum.find_prefixsum_idx(mass - _sum1)
                res2.append(idx)
        return res1, res2

    def sample(self, batch_size, beta=0):
        assert beta > 0

        idxes1, idxes2 = self._sample_proportional(batch_size)

        weights1, weights2 = [],  []
        p_min = self.min_tree_operation(self.buffer1._it_min.min(), self.buffer2._it_min.min()) / self.sum_tree_operation(self.buffer1._it_sum.sum(), self.buffer2._it_sum.sum())
        max_weight = (p_min * (len(self.buffer1._storage) + len(self.buffer2._storage))) ** (-beta)

        for idx in idxes1:
            p_sample = self.buffer1._it_sum[idx] / (self.buffer1._it_sum.sum() + self.buffer2._it_sum.sum())
            weight = (p_sample * (len(self.buffer1._storage) + len(self.buffer2._storage))) ** (-beta)
            weights1.append(weight / max_weight)
        for idx in idxes2:
            p_sample = self.buffer2._it_sum[idx] / (self.buffer1._it_sum.sum() + self.buffer2._it_sum.sum())
            weight = (p_sample * (len(self.buffer1._storage) + len(self.buffer2._storage))) ** (-beta)
            weights2.append(weight / max_weight)

        weights1 = np.array(weights1)
        weights2 = np.array(weights2)
        encoded_sample1 = self.buffer1._encode_sample(idxes1)
        encoded_sample2 = self.buffer2._encode_sample(idxes2)
        return tuple(list(encoded_sample1) + [weights1, idxes1]), tuple(list(encoded_sample2) + [weights2, idxes2])


class MultiWorkerReplayBuffer(ReplayBuffer):
    def __init__(self, size, num_workers=1, gamma=0.99):
        super(MultiWorkerReplayBuffer, self).__init__(size)
        self.num_workers = num_workers
        self.gamma = gamma
        self.local_transitions = [[] for _ in range(self.num_workers)]

    def add(self, obs_t, action, reward, obs_tp1, done):
        assert obs_t.shape[0] == self.num_workers
        for i in range(self.num_workers):
            self.local_transitions[i].append([obs_t[i], action[i], reward[i], obs_tp1[i], done[i]])
            if done[i]:
                for j in range(len(self.local_transitions[i])):
                    super().add(*(self.local_transitions[i][j]))
                self.local_transitions[i] = []


class PrioritizedMultiWorkerReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, size, alpha, num_workers=1, gamma=0.99):
        super(PrioritizedMultiWorkerReplayBuffer, self).__init__(size, alpha)
        self.num_workers = num_workers
        self.gamma = gamma
        self.local_transitions = [[] for _ in range(self.num_workers)]
        self.model = None

    def set_model(self, model):
        self.model = model

    def add(self, obs_t, action, reward, obs_tp1, done):
        assert obs_t.shape[0] == self.num_workers
        for i in range(self.num_workers):
            self.local_transitions[i].append([obs_t[i], action[i], reward[i], obs_tp1[i], done[i]])
            # assert len(self.local_priorities[i]) == len(self.local_transitions[i])
            if done[i]:
                batch_obs, batch_act, batch_reward, batch_next_obs, batch_done = zip(*(self.local_transitions[i]))
                batch_obs, batch_act, batch_reward, batch_next_obs, batch_done = \
                    map(lambda v: np.asarray(v),[batch_obs, batch_act, batch_reward, batch_next_obs, batch_done])
                priorities = compute_priority(self.model, batch_obs, batch_act,
                                              batch_next_obs, batch_reward, batch_done)
                for j in range(len(self.local_transitions[i])):
                    p_idx = self._next_idx  # The add call will change self._next_idx
                    super().add(*(self.local_transitions[i][j]))
                    self.update_priorities([p_idx], [priorities[j]])
                self.local_transitions[i] = []


def discounted_sum(arr, gamma):
    arr = np.asarray(arr)
    return np.sum(arr * np.power(gamma, np.arange(arr.shape[0])))


def compute_priority(sac_model, batch_obs, batch_act, batch_next_obs, batch_reward, batch_done):
    q1, value = sac_model.sess.run([sac_model.step_ops[4], sac_model.value_target], feed_dict={
        sac_model.observations_ph: batch_obs,
        sac_model.actions_ph: batch_act,
        sac_model.next_observations_ph: batch_next_obs,
    })
    priorities = np.reshape(batch_reward, q1.shape) + (
            1 - np.reshape(batch_done, q1.shape)) * sac_model.gamma * value - q1
    priorities = np.squeeze(np.abs(priorities) + 1e-4, axis=-1).tolist()
    return priorities
