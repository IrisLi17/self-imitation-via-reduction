import copy
from enum import Enum
from utils.replay_buffer import MultiWorkerReplayBuffer, PrioritizedMultiWorkerReplayBuffer

import numpy as np
import time


class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """
    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2
    # Select a goal that was achieved
    # at some point in the training procedure
    # (and that is present in the replay buffer)
    RANDOM = 3
    # Select N-1 `future` and 1 final. Only works when the episode is successful, otherwise use `future`
    FUTUREANDFINAL = 4


# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    'future': GoalSelectionStrategy.FUTURE,
    'final': GoalSelectionStrategy.FINAL,
    'episode': GoalSelectionStrategy.EPISODE,
    'random': GoalSelectionStrategy.RANDOM,
    'future_and_final': GoalSelectionStrategy.FUTUREANDFINAL,
}


class HindsightExperienceReplayWrapper(object):
    """
    Wrapper around a replay buffer in order to use HER.
    This implementation is inspired by to the one found in https://github.com/NervanaSystems/coach/.
    :param replay_buffer: (ReplayBuffer)
    :param n_sampled_goal: (int) The number of artificial transitions to generate for each actual transition
    :param goal_selection_strategy: (GoalSelectionStrategy) The method that will be used to generate
        the goals for the artificial transitions.
    :param wrapped_env: (HERGoalEnvWrapper) the GoalEnv wrapped using HERGoalEnvWrapper,
        that enables to convert observation to dict, and vice versa
    """

    def __init__(self, replay_buffer, n_sampled_goal, goal_selection_strategy, wrapped_env):
        super(HindsightExperienceReplayWrapper, self).__init__()

        assert isinstance(goal_selection_strategy, GoalSelectionStrategy), "Invalid goal selection strategy," \
                                                                           "please use one of {}".format(
            list(GoalSelectionStrategy))

        assert isinstance(replay_buffer, MultiWorkerReplayBuffer) or isinstance(replay_buffer, PrioritizedMultiWorkerReplayBuffer)

        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy
        self.env = wrapped_env
        # Buffer for storing transitions of the current episode is implemented inside replay_buffer now
        # self.episode_transitions = []
        self.replay_buffer = replay_buffer
        self.temp_container = {'idx': [], 'observation': [], 'action': [], 'next_observation': [],
                               'reward': [], 'done': []}
        self.reward_time = 0.0

    def set_model(self, model):
        self.model = model

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer
        :param obs_t: (np.ndarray) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (np.ndarray) the new observation
        :param done: (bool) is the episode done
        """
        assert self.replay_buffer is not None
        assert obs_t.shape[0] == self.replay_buffer.num_workers
        self._next_idx = self.replay_buffer._next_idx
        for i in range(self.replay_buffer.num_workers):
            self.replay_buffer.local_transitions[i].append((obs_t[i], action[i], reward[i], obs_tp1[i], done[i]))
            if done[i]:
                self._store_episode(i)
                self.replay_buffer.local_transitions[i] = []
        if len(self.temp_container['observation']):
            reward_time0 = time.time()
            # Compute reward in batches
            for _ in range(len(self.temp_container['observation']) // self.replay_buffer.num_workers):
                next_obs = self.temp_container['next_observation'][_ * self.replay_buffer.num_workers: (_ + 1) * self.replay_buffer.num_workers]
                next_obs_dict = self.env.convert_obs_to_dict(np.asarray(next_obs))
                if self.env.goal_dim == 3:
                    assert self.env.reward_type == 'sparse'
                    reward = self.env.compute_reward(next_obs_dict['desired_goal'], next_obs_dict['achieved_goal'], [None] * self.replay_buffer.num_workers, indices=range(len(next_obs)))
                    success = (np.array(reward) > 0.5).tolist()
                else:
                    if self.env.reward_type != 'sparse':
                        reward_and_success = self.env.compute_reward_and_success(next_obs, next_obs_dict['desired_goal'], [None] * self.replay_buffer.num_workers, indices=range(len(next_obs)))
                        reward, success = zip(*reward_and_success)
                        reward = list(reward)
                        success = list(success)
                    else:
                        reward = self.env.compute_reward(next_obs, next_obs_dict['desired_goal'], [None] * self.replay_buffer.num_workers, indices=range(len(next_obs)))
                        success = (np.array(reward) > 0.5).tolist()
                self.temp_container['reward'][_ * self.replay_buffer.num_workers : (_ + 1) * self.replay_buffer.num_workers] = reward.copy()
                self.temp_container['done'][_ * self.replay_buffer.num_workers : (_ + 1) * self.replay_buffer.num_workers] = success.copy()
            # Remainer
            if len(self.temp_container['observation']) % self.replay_buffer.num_workers:
                next_obs = self.temp_container['next_observation'][len(self.temp_container['observation']) // self.replay_buffer.num_workers * self.replay_buffer.num_workers : len(self.temp_container['observation'])]
                next_obs_dict = self.env.convert_obs_to_dict(np.asarray(next_obs))
                if self.env.goal_dim == 3:
                    assert self.env.reward_type == 'sparse'
                    reward = self.env.compute_reward(next_obs_dict['desired_goal'], next_obs_dict['achieved_goal'],
                                                     [None] * self.replay_buffer.num_workers, indices=range(len(next_obs)))
                    success = (np.array(reward) > 0.5).tolist()
                else:
                    if self.env.reward_type != 'sparse':
                        reward_and_success = self.env.compute_reward_and_success(next_obs, next_obs_dict['desired_goal'],
                                                                                 [None] * self.replay_buffer.num_workers, indices=range(len(next_obs)))
                        reward, success = zip(*reward_and_success)
                        reward = list(reward)
                        success = list(success)
                    else:
                        reward = self.env.compute_reward(next_obs, next_obs_dict['desired_goal'],
                                                         [None] * self.replay_buffer.num_workers, indices=range(len(next_obs)))
                        success = (np.array(reward) > 0.5).tolist()
                self.temp_container['reward'][len(self.temp_container['observation']) // self.replay_buffer.num_workers * self.replay_buffer.num_workers : len(self.temp_container['observation'])] = reward.copy()
                self.temp_container['done'][len(self.temp_container['observation']) // self.replay_buffer.num_workers * self.replay_buffer.num_workers : len(self.temp_container['observation'])] = success.copy()

            self.reward_time += time.time() - reward_time0
            # Store into buffer now
            for i in range(len(self.temp_container['observation'])):
                obs = self.temp_container['observation'][i]
                action = self.temp_container['action'][i]
                reward = self.temp_container['reward'][i]
                next_obs = self.temp_container['next_observation'][i]
                done = self.temp_container['done'][i]
                super(type(self.replay_buffer), self.replay_buffer).add(obs, action, reward, next_obs, done)

        if isinstance(self.replay_buffer, PrioritizedMultiWorkerReplayBuffer) and len(self.temp_container['observation']):
            q1, value = self.model.sess.run([self.model.step_ops[4], self.model.value_target], feed_dict={
                self.model.observations_ph: np.asarray(self.temp_container['observation']),
                self.model.actions_ph: np.asarray(self.temp_container['action']),
                self.model.next_observations_ph: np.asarray(self.temp_container['next_observation']),
            })
            priorities = np.reshape(np.asarray(self.temp_container['reward']), q1.shape) \
                         + (1 - np.reshape(np.asarray(self.temp_container['done']), q1.shape)) * self.model.gamma * value - q1
            priorities = np.squeeze(np.abs(priorities) + 1e-4, axis=-1).tolist()
            self.update_priorities(self.temp_container['idx'], priorities)
        for key in self.temp_container.keys():
            self.temp_container[key] = []

    def sample(self, *args, **kwargs):
        return self.replay_buffer.sample(*args, **kwargs)

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.
        :param n_samples: (int)
        :return: (bool)
        """
        return self.replay_buffer.can_sample(n_samples)

    def update_priorities(self, idxes, priorities):
        if isinstance(self.replay_buffer, MultiWorkerReplayBuffer):
            raise NotImplementedError
        return self.replay_buffer.update_priorities(idxes, priorities)

    def __len__(self):
        return len(self.replay_buffer)


    def _sample_achieved_goal(self, episode_transitions, transition_idx):
        """
        Sample an achieved goal according to the sampling strategy.
        :param episode_transitions: ([tuple]) a list of all the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE \
                or self.goal_selection_strategy == GoalSelectionStrategy.FUTUREANDFINAL:
            # Sample a goal that was observed in the same episode after the current step
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # Choose the goal achieved at the end of the episode
            selected_transition = episode_transitions[-1]
        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Random goal achieved during the episode
            selected_idx = np.random.choice(np.arange(len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            # Random goal achieved, from the entire replay buffer
            selected_idx = np.random.choice(np.arange(len(self.replay_buffer)))
            selected_transition = self.replay_buffer.storage[selected_idx]
        else:
            raise ValueError("Invalid goal selection strategy,"
                             "please use one of {}".format(list(GoalSelectionStrategy)))
        return self.env.convert_obs_to_dict(selected_transition[0])['achieved_goal'], selected_idx

    def _sample_achieved_goals(self, episode_transitions, transition_idx):
        """
        Sample a batch of achieved goals according to the sampling strategy.
        :param episode_transitions: ([tuple]) list of the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTUREANDFINAL \
                and abs(np.sum([item[2] for item in episode_transitions]) - 2) < 1e-4:
            # and len(episode_transitions) < self.env.env.spec.max_episode_steps:
            achieved_goals = []
            for i in range(self.n_sampled_goal - 1):
                selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
                selected_transition = episode_transitions[selected_idx]
                achieved_goals.append(self.env.convert_obs_to_dict(selected_transition[0])['achieved_goal'])
            achieved_goals.append(self.env.convert_obs_to_dict(episode_transitions[-1][0])['achieved_goal'])
            return achieved_goals
        return [
            self._sample_achieved_goal(episode_transitions, transition_idx)
            for _ in range(self.n_sampled_goal)
        ]

    def _store_episode(self, i):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        # For each transition in the last episode,
        # create a set of artificial transitions
        for transition_idx, transition in enumerate(self.replay_buffer.local_transitions[i]):

            obs_t, action, reward, obs_tp1, done = transition

            # Add to the replay buffer
            if isinstance(self.replay_buffer, PrioritizedMultiWorkerReplayBuffer) or isinstance(self.replay_buffer, MultiWorkerReplayBuffer):
                self.temp_container['idx'].append(self._next_idx)
                self.temp_container['observation'].append(obs_t)
                self.temp_container['action'].append(action)
                self.temp_container['next_observation'].append(obs_tp1)
                self.temp_container['reward'].append(reward)
                self.temp_container['done'].append(done)
            # Store them later but increment idx in this wrapper.
            self._next_idx  = (self._next_idx + 1) % self.replay_buffer._maxsize
            # We cannot sample a goal from the future in the last step of an episode
            if (transition_idx == len(self.replay_buffer.local_transitions[i]) - 1 and
                    (self.goal_selection_strategy == GoalSelectionStrategy.FUTURE or
                             self.goal_selection_strategy == GoalSelectionStrategy.FUTUREANDFINAL)):
                break

            # Sampled n goals per transition, where n is `n_sampled_goal`
            # this is called k in the paper
            sampled_goals = self._sample_achieved_goals(self.replay_buffer.local_transitions[i], transition_idx)
            # For each sampled goals, store a new transition
            for goal, sampled_idx in sampled_goals:
                # Copy transition to avoid modifying the original one
                obs, action, reward, next_obs, done = copy.deepcopy(transition)

                # Convert concatenated obs to dict, so we can update the goals
                obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obs, next_obs))

                # Update the desired goal in the transition
                if self.env.env.get_attr('spec')[0].id == 'MasspointMaze-v3':
                    obs_dict['desired_goal'][:2] = goal[:2]
                    next_obs_dict['desired_goal'][:2] = goal[:2]
                else:
                    obs_dict['desired_goal'] = goal
                    next_obs_dict['desired_goal'] = goal

                # assert len(goal) in [3, 5, 6]
                if len(goal) > 3:
                    # modify dict, note that desired_goal is already modified, only need to modify achieved goal
                    one_hot = goal[3:]
                    idx = np.argmax(one_hot)
                    obs_dict['achieved_goal'][3:] = one_hot
                    obs_dict['achieved_goal'][0:3] = obs_dict['observation'][3 + 3 * idx: 3 + 3 * (idx + 1)]
                    next_obs_dict['achieved_goal'][3:] = one_hot
                    next_obs_dict['achieved_goal'][0:3] = next_obs_dict['observation'][3 + 3 * idx: 3 + 3 * (idx + 1)]

                reward = 0.0
                done = False

                # Transform back to ndarrays
                obs, next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict))

                # Add artificial transition to the replay buffer
                if isinstance(self.replay_buffer, PrioritizedMultiWorkerReplayBuffer) or isinstance(self.replay_buffer, MultiWorkerReplayBuffer):
                    self.temp_container['idx'].append(self._next_idx)
                    self.temp_container['observation'].append(obs)
                    self.temp_container['action'].append(action)
                    self.temp_container['next_observation'].append(next_obs)
                    self.temp_container['reward'].append(reward)
                    self.temp_container['done'].append(done)
                # Store them later but increment idx in this wrapper.
                self._next_idx = (self._next_idx + 1) % self.replay_buffer._maxsize
