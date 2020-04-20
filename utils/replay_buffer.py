from stable_baselines.deepq import ReplayBuffer, PrioritizedReplayBuffer
import numpy as np
import random


class SumRReplayBuffer(ReplayBuffer):
    def __init__(self, size):
        super(SumRReplayBuffer, self).__init__(size)

    def add(self, obs_t, action, reward, obs_tp1, done, sum_r):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        :param sum_r: (float) the discounted sum of reward from this step
        """
        data = (obs_t, action, reward, obs_tp1, done, sum_r)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, sum_rs = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, sum_r = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            sum_rs.append(sum_r)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(sum_rs)


class SumRPrioritizedReplayBuffer(PrioritizedReplayBuffer, SumRReplayBuffer):
    def __init__(self, size, alpha):
        PrioritizedReplayBuffer.__init__(self, size, alpha)

    def add(self, obs_t, action, reward, obs_tp1, done, sum_r):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        :param sum_r: (float) the discounted sum reward from this step
        """
        idx = self._next_idx
        SumRReplayBuffer.add(self, obs_t, action, reward, obs_tp1, done, sum_r)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def sample(self, batch_size, beta=0):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = SumRReplayBuffer._encode_sample(self, idxes)
        return tuple(list(encoded_sample) + [weights, idxes])


class DoublePrioritizedReplayWrapper(object):
    def __init__(self, buffer1, buffer2):
        assert isinstance(buffer1, SumRPrioritizedReplayBuffer)
        assert isinstance(buffer2, SumRPrioritizedReplayBuffer)
        self.buffer1 = buffer1
        self.buffer2 = buffer2
        self.min_tree_operation = buffer1._it_min._operation
        self.sum_tree_operation = buffer1._it_sum._operation

    def _sample_proportional(self, batch_size):
        res1, res2 = [], []
        _sum1 = self.buffer1._it_sum.sum(0, len(self.buffer1._storage) - 1)
        _sum2 = self.buffer2._it_sum.sum(0, len(self.buffer2._storage) - 1)
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


class MultiWorkerReplayBuffer(SumRReplayBuffer):
    def __init__(self, size, num_workers=1, gamma=0.99):
        super(MultiWorkerReplayBuffer, self).__init__(size)
        self.num_workers = num_workers
        self.gamma = gamma
        self.local_transitions = [[] for _ in range(self.num_workers)]

    def add(self, obs_t, action, reward, obs_tp1, done):
        assert obs_t.shape[0] == self.num_workers
        for i in range(self.num_workers):
            self.local_transitions[i].append([obs_t[i], action[i], reward[i], obs_tp1[i], done[i], 0])
            if done[i]:
                for j in range(len(self.local_transitions[i])):
                    # Compute discounted r
                    self.local_transitions[i][j][-1] = discounted_sum(
                        [self.local_transitions[i][k][2] for k in range(j, len(self.local_transitions[i]))], self.gamma)
                    super().add(*(self.local_transitions[i][j]))
                self.local_transitions[i] = []


# TODO: compute priority at first time
class PrioritizedMultiWorkerReplayBuffer(SumRPrioritizedReplayBuffer):
    def __init__(self, size, alpha, num_workers=1, gamma=0.99):
        super(PrioritizedMultiWorkerReplayBuffer, self).__init__(size, alpha)
        self.num_workers = num_workers
        self.gamma = gamma
        self.local_transitions = [[] for _ in range(self.num_workers)]
        self.model = None

    def set_model(self, model):
        self.model = model

    # # TODO: deprecate this method
    # def append_priority(self, p):
    #     for i in range(self.num_workers):
    #         assert len(self.local_transitions[i]) == len(self.local_priorities[i])
    #         self.local_priorities[i].append(p[i])

    def add(self, obs_t, action, reward, obs_tp1, done):
        assert obs_t.shape[0] == self.num_workers
        for i in range(self.num_workers):
            self.local_transitions[i].append([obs_t[i], action[i], reward[i], obs_tp1[i], done[i], 0])
            # assert len(self.local_priorities[i]) == len(self.local_transitions[i])
            if done[i]:
                batch_obs, batch_act, batch_reward, batch_next_obs, batch_done, _ = zip(*(self.local_transitions[i]))
                batch_obs, batch_act, batch_reward, batch_next_obs, batch_done = \
                    map(lambda v: np.asarray(v),[batch_obs, batch_act, batch_reward, batch_next_obs, batch_done])
                priorities = compute_priority(self.model, batch_obs, batch_act,
                                              batch_next_obs, batch_reward, batch_done)
                for j in range(len(self.local_transitions[i])):
                    # Compute discounted r
                    self.local_transitions[i][j][-1] = discounted_sum(
                        [self.local_transitions[i][k][2] for k in range(j, len(self.local_transitions[i]))], self.gamma)
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
