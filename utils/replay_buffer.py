from stable_baselines.deepq import ReplayBuffer, PrioritizedReplayBuffer


class MultiWorkerReplayBuffer(ReplayBuffer):
    def __init__(self, size, num_workers=1):
        super(MultiWorkerReplayBuffer, self).__init__(size)
        self.num_workers = num_workers
        self.local_transitions = [[] for _ in range(self.num_workers)]

    def add(self, obs_t, action, reward, obs_tp1, done):
        assert obs_t.shape[0] == self.num_workers
        for i in range(self.num_workers):
            self.local_transitions[i].append((obs_t[i], action[i], reward[i], obs_tp1[i], done[i]))
            if done[i]:
                for j in range(len(self.local_transitions[i])):
                    super().add(*(self.local_transitions[i][j]))
                self.local_transitions[i] = []


class PrioritizedMultiWorkerReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, size, alpha, num_workers=1):
        super(PrioritizedMultiWorkerReplayBuffer, self).__init__(size, alpha)
        self.num_workers = num_workers
        self.local_transitions = [[] for _ in range(self.num_workers)]
        self.local_priorities = [[] for _ in range(self.num_workers)]

    def append_priority(self, p):
        for i in range(self.num_workers):
            assert len(self.local_transitions[i]) == len(self.local_priorities[i])
            self.local_priorities[i].append(p[i])

    def add(self, obs_t, action, reward, obs_tp1, done):
        assert obs_t.shape[0] == self.num_workers
        for i in range(self.num_workers):
            self.local_transitions[i].append((obs_t[i], action[i], reward[i], obs_tp1[i], done[i]))
            assert len(self.local_priorities[i]) == len(self.local_transitions[i])
            if done[i]:
                for j in range(len(self.local_transitions[i])):
                    p_idx = self._next_idx # The add call will change self._next_idx
                    super().add(*(self.local_transitions[i][j]))
                    self.update_priorities([p_idx], [self.local_priorities[i][j]])
                self.local_transitions[i] = []
                self.local_priorities[i] = []