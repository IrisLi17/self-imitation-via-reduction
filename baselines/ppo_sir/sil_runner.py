import time
import sys, os, csv
import multiprocessing
from collections import deque

import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger


class SILRunner(AbstractEnvRunner):
    def __init__(self, *, env, aug_env, model, n_steps, gamma, lam, n_candidate, horizon, dim_candidate=2):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.aug_env = aug_env
        self.lam = lam
        self.gamma = gamma
        self.n_candidate = n_candidate
        # For restart
        # self.ep_transition_buf = [[] for _ in range(self.model.n_envs)]
        self.ep_obs_buf = [[] for _ in range(self.model.n_envs)]
        self.ep_act_buf = [[] for _ in range(self.model.n_envs)]
        self.ep_value_buf = [[] for _ in range(self.model.n_envs)]
        self.ep_neglogpac_buf = [[] for _ in range(self.model.n_envs)]
        self.ep_done_buf = [[] for _ in range(self.model.n_envs)]
        self.ep_reward_buf = [[] for _ in range(self.model.n_envs)]
        self.goal_dim = self.env.get_attr('goal')[0].shape[0]
        self.obs_dim = self.env.observation_space.shape[0] - 2 * self.goal_dim
        self.noise_mag = self.env.get_attr('size_obstacle')[0][1]
        self.n_object = self.env.get_attr('n_object')[0]
        self.dim_candidate = dim_candidate
        self.horizon = horizon
        # self.reuse_times = reuse_times
        print('obs_dim', self.obs_dim, 'goal_dim', self.goal_dim, 'noise_mag', self.noise_mag,
              'n_object', self.n_object, 'horizon', self.horizon)
        # TODO: add buffers
        self.restart_steps = [] # Every element should be scalar
        self.subgoals = [] # Every element should be [*subgoals, ultimate goal]
        self.restart_states = [] # list of (n_candidate) states
        self.transition_storage = [] # every element is list of tuples. length of every element should match restart steps
        self.current_nobject = []
        self.task_mode = []
        # For filter subgoals
        self.mean_value_buf = deque(maxlen=500)
        self.self_aug_ratio = deque(maxlen=500)

    def run(self):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        mb_goals = self.env.get_attr('goal')

        duration = 0.0
        # step_env_duration = 0.0
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)
            for idx, info in enumerate(infos):
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    # if self.goal_dim > 3 and np.argmax(mb_goals[idx][3:]) == 0:
                    ep_infos.append(maybe_ep_info)
                    mb_goals[idx] = self.env.get_attr('goal', indices=idx)[0]
                if self.dones[idx] and (not info['is_success']):
                    rewards[idx] = self.model.value(np.expand_dims(info['terminal_observation'], axis=0))
            mb_rewards.append(rewards)
            for i in range(self.model.n_envs):
                # self.ep_transition_buf[i].append((mb_obs[-1][i], mb_actions[-1][i], mb_values[-1][i],
                #                                   mb_neglogpacs[-1][i], mb_dones[-1][i], mb_rewards[-1][i]))
                self.ep_obs_buf[i].append(mb_obs[-1][i])
                self.ep_act_buf[i].append(mb_actions[-1][i])
                self.ep_value_buf[i].append(mb_values[-1][i])
                self.ep_neglogpac_buf[i].append(mb_neglogpacs[-1][i])
                self.ep_done_buf[i].append(mb_dones[-1][i])
                self.ep_reward_buf[i].append(mb_rewards[-1][i])

            # restart_steps = [[] for _ in range(self.model.n_envs)]
            # subgoals = [[] for _ in range(self.model.n_envs)]
            for idx, done in enumerate(self.dones):
                if done:
                    if infos[idx]['is_success']:
                        # TODO: unpack obs, action etc. compute adv
                        augment_returns = self.compute_adv(self.ep_value_buf[idx], self.ep_done_buf[idx], self.ep_reward_buf[idx])
                        if self.model.aug_obs[-1] is None:
                            self.model.aug_obs[-1] = np.array(self.ep_obs_buf[idx])
                            self.model.aug_act[-1] = np.array(self.ep_act_buf[idx])
                            self.model.aug_neglogp[-1] = np.array(self.ep_neglogpac_buf[idx])
                            self.model.aug_value[-1] = np.array(self.ep_value_buf[idx])
                            self.model.aug_return[-1] = augment_returns
                            self.model.aug_done[-1] = np.array(self.ep_done_buf[idx])
                            self.model.aug_reward[-1] = np.array(self.ep_reward_buf[idx])
                            # self.model.is_selfaug[-1] = np.array(augment_isselfaug_buf)
                            for reuse_idx in range(len(self.model.aug_obs) - 1):
                                # Update previous data with new value and policy parameters
                                if self.model.aug_obs[reuse_idx] is not None:
                                    self.model.aug_neglogp[reuse_idx] = self.model.sess.run(self.model.aug_neglogpac_op,
                                                                                            {self.model.train_aug_model.obs_ph: self.model.aug_obs[reuse_idx],
                                                                                             self.model.aug_action_ph: self.model.aug_act[reuse_idx]})
                                    self.model.aug_value[reuse_idx] = self.model.value(self.model.aug_obs[reuse_idx])
                                    self.model.aug_return[reuse_idx] = self.compute_adv(
                                        self.model.aug_value[reuse_idx], self.model.aug_done[reuse_idx], self.model.aug_reward[reuse_idx])
                        else:
                            self.model.aug_obs[-1] = np.concatenate([self.model.aug_obs[-1], np.array(self.ep_obs_buf[idx])], axis=0)
                            self.model.aug_act[-1] = np.concatenate([self.model.aug_act[-1], np.array(self.ep_act_buf[idx])], axis=0)
                            self.model.aug_neglogp[-1] = np.concatenate(
                                [self.model.aug_neglogp[-1], np.array(self.ep_neglogpac_buf[idx])], axis=0)
                            self.model.aug_value[-1] = np.concatenate(
                                [self.model.aug_value[-1], np.array(self.ep_value_buf[idx])], axis=0)
                            self.model.aug_return[-1] = np.concatenate([self.model.aug_return[-1], augment_returns], axis=0)
                            self.model.aug_done[-1] = np.concatenate(
                                [self.model.aug_done[-1], np.array(self.ep_done_buf[idx])], axis=0)
                            self.model.aug_reward[-1] = np.concatenate(
                                [self.model.aug_reward[-1], np.array(self.ep_reward_buf[idx])], axis=0)
                            # self.model.is_selfaug[-1] = np.concatenate(
                            #     [self.model.is_selfaug[-1], np.array(augment_isselfaug_buf)], axis=0)

                    self.ep_obs_buf[idx] = []
                    self.ep_act_buf[idx] = []
                    self.ep_value_buf[idx] = []
                    self.ep_done_buf[idx] = []
                    self.ep_neglogpac_buf[idx] = []
                    self.ep_reward_buf[idx] = []


            def convert_dict_to_obs(dict_obs):
                assert isinstance(dict_obs, dict)
                return np.concatenate([dict_obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])

            # Try to get rid of every for-loop over n_candidate
            temp_time0 = time.time()
            temp_time1 = time.time()
            duration += (temp_time1 - temp_time0)


        print('augment takes', duration)
        # print('augment stepping env takes', step_env_duration)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward

    def select_subgoal(self, transition_buf, k, dim, env_idx):
        debug = False
        # self.ep_transition_buf, self.model.value
        obs_buf, *_ = zip(*transition_buf)
        obs_buf = np.asarray(obs_buf)
        sample_t = np.random.randint(0, len(transition_buf), 4096)
        sample_obs = obs_buf[sample_t]
        noise = np.random.uniform(low=-self.noise_mag, high=self.noise_mag, size=(len(sample_t), 2))
        # TODO: if there are more than one obstacle
        sample_obs_buf = []
        subgoal_obs_buf = []
        if dim == 2:
            for object_idx in range(1, self.n_object):
                obstacle_xy = sample_obs[:, 3 * (object_idx+1):3*(object_idx+1) + 2] + noise
                sample_obs[:, 3*(object_idx+1):3*(object_idx+1)+2] = obstacle_xy
                sample_obs[:, 3*(object_idx+1+self.n_object):3*(object_idx+1+self.n_object)+2] \
                    = sample_obs[:, 3*(object_idx+1):3*(object_idx+1)+2] - sample_obs[:, 0:2]
                sample_obs_buf.append(sample_obs.copy())

                subgoal_obs = obs_buf[sample_t]
                # if debug:
                #     subgoal_obs = np.tile(subgoal_obs, (2, 1))
                subgoal_obs[:, self.obs_dim:self.obs_dim+3] = subgoal_obs[:, 3*(object_idx+1):3*(object_idx+1)+3]
                one_hot = np.zeros(self.n_object)
                one_hot[object_idx] = 1
                subgoal_obs[:, self.obs_dim+3:self.obs_dim+self.goal_dim] = one_hot
                subgoal_obs[:, self.obs_dim+self.goal_dim:self.obs_dim+self.goal_dim+2] = obstacle_xy
                subgoal_obs[:, self.obs_dim+self.goal_dim+2:self.obs_dim+self.goal_dim+3] = subgoal_obs[:, 3*(object_idx+1)+2:3*(object_idx+1)+3]
                subgoal_obs[:, self.obs_dim+self.goal_dim+3:self.obs_dim+self.goal_dim*2] = one_hot
                subgoal_obs_buf.append(subgoal_obs)
        elif dim == 3:
            for object_idx in range(0, self.n_object):
                if object_idx == np.argmax(sample_obs[0][self.obs_dim + self.goal_dim + 3:]):
                    continue
                if np.linalg.norm(sample_obs[0][3 + object_idx * 3 : 3 + (object_idx + 1) * 3]) < 1e-3:
                    continue
                obstacle_xy = sample_obs[:, 3 * (object_idx + 1):3 * (object_idx + 1) + 2] + noise
                # obstacle_height = np.random.uniform(low=0.425, high=0.425 + 0.15, size=(len(sample_t), 1))
                obstacle_height = max(sample_obs[0][self.obs_dim + self.goal_dim + 2] - 0.05, 0.425) * np.ones((len(sample_t), 1))
                obstacle_xy = np.concatenate([obstacle_xy, obstacle_height], axis=-1)
                sample_obs[:, 3 * (object_idx + 1):3 * (object_idx + 1) + 3] = obstacle_xy
                sample_obs[:, 3 * (object_idx + 1 + self.n_object):3 * (object_idx + 1 + self.n_object) + 3] \
                    = sample_obs[:, 3 * (object_idx + 1):3 * (object_idx + 1) + 3] - sample_obs[:, 0:3]
                sample_obs_buf.append(sample_obs.copy())

                subgoal_obs = obs_buf[sample_t]
                # if debug:
                #     subgoal_obs = np.tile(subgoal_obs, (2, 1))
                subgoal_obs[:, self.obs_dim:self.obs_dim + 3] = subgoal_obs[:,
                                                                3 * (object_idx + 1):3 * (object_idx + 1) + 3]
                one_hot = np.zeros(self.n_object)
                one_hot[object_idx] = 1
                subgoal_obs[:, self.obs_dim + 3:self.obs_dim + self.goal_dim] = one_hot
                subgoal_obs[:, self.obs_dim + self.goal_dim:self.obs_dim + self.goal_dim + 3] = obstacle_xy
                # subgoal_obs[:, self.obs_dim + self.goal_dim + 2:self.obs_dim + self.goal_dim + 3] = subgoal_obs[:, 3 * (
                # object_idx + 1) + 2:3 * (object_idx + 1) + 3]
                subgoal_obs[:, self.obs_dim + self.goal_dim + 3:self.obs_dim + self.goal_dim * 2] = one_hot
                subgoal_obs_buf.append(subgoal_obs)
        if len(sample_obs_buf) == 0:
            return np.array([]), np.array([])
        sample_obs_buf = np.concatenate(sample_obs_buf, axis=0)
        value2 = self.model.value(sample_obs_buf)
        subgoal_obs_buf = np.concatenate(subgoal_obs_buf)
        value1 = self.model.value(subgoal_obs_buf)
        normalize_value1 = (value1 - np.min(value1)) / (np.max(value1) - np.min(value1))
        normalize_value2 = (value2 - np.min(value2)) / (np.max(value2) - np.min(value2))
        # best_idx = np.argmax(normalize_value1 * normalize_value2)
        ind = np.argsort(normalize_value1 * normalize_value2)
        good_ind = ind[-k:]
        # if debug:
        #     print('original value1', 'mean', np.mean(origin_value1), 'std', np.std(origin_value1))
        #     print('original value2', 'mean', np.mean(origin_value2), 'std', np.std(origin_value2))
        #     print(value1[good_ind])
        #     print(value2[good_ind])
        # restart_step = sample_t[best_idx]
        # subgoal = subgoal_obs[best_idx, 45:50]
        mean_values = (value1[good_ind] + value2[good_ind]) / 2
        assert mean_values.shape[0] == k
        for i in range(k):
            self.mean_value_buf.append(mean_values[i])
        filtered_idx = np.where(mean_values >= np.mean(self.mean_value_buf))[0]
        good_ind = good_ind[filtered_idx]
        # self.self_aug_ratio.append(np.sum(good_ind < len(sample_t)) / (len(filtered_idx) + 1e-8))

        restart_step = sample_t[good_ind % len(sample_t)]
        subgoal = subgoal_obs_buf[good_ind, self.obs_dim+self.goal_dim:self.obs_dim+self.goal_dim*2]
        self.self_aug_ratio.append(np.sum(subgoal[:, 3]) / (len(filtered_idx) + 1e-8))

        # print('subgoal', subgoal, 'with value1', normalize_value1[best_idx], 'value2', normalize_value2[best_idx])
        # print('restart step', restart_step)
        return restart_step, subgoal

    def compute_adv(self, values, dones, rewards):
        if not isinstance(values, np.ndarray):
            values = np.asarray(values)
        if not isinstance(dones, np.ndarray):
            dones = np.asarray(dones)
        if not isinstance(rewards, np.ndarray):
            rewards = np.asarray(rewards)
        # discount/bootstrap off value fn
        advs = np.zeros_like(rewards)
        last_gae_lam = 0
        for step in reversed(range(values.shape[0])):
            if step == values.shape[0] - 1:
                # Here we have assumed that the episode ends with done=Fase (not recorded in dones!).
                nextnonterminal = 0.0
                # So nextvalues here will not be used.
                nextvalues = np.zeros(values[0].shape)
            else:
                nextnonterminal = 1.0 - dones[step + 1]
                nextvalues = values[step + 1]
            delta = rewards[step] + self.gamma * nextvalues * nextnonterminal - values[step]
            advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        returns = advs + values
        return returns

def get_schedule_fn(value_schedule):
    """
    Transform (if needed) learning rate and clip range
    to callable.

    :param value_schedule: (callable or float)
    :return: (function)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constfn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def constfn(val):
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (function)
    """

    def func(_):
        return val

    return func


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)
