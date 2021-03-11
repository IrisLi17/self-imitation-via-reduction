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


class SIRRunner(AbstractEnvRunner):
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
        self.ep_state_buf = [[] for _ in range(self.model.n_envs)]
        self.ep_transition_buf = [[] for _ in range(self.model.n_envs)]
        self.ep_current_nobject = [[] for _ in range(self.model.n_envs)]
        self.ep_task_mode = [[] for _ in range(self.model.n_envs)]
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
            internal_states = self.env.env_method('get_state')
            for i in range(self.model.n_envs):
                self.ep_state_buf[i].append(internal_states[i])
            if self.dim_candidate == 3:
                current_nobjects = self.env.get_attr('current_nobject')
                task_modes = self.env.get_attr('task_mode')
                for i in range(self.model.n_envs):
                    self.ep_current_nobject[i].append(current_nobjects[i])
                    self.ep_task_mode[i].append(task_modes[i])
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
                self.ep_transition_buf[i].append((mb_obs[-1][i], mb_actions[-1][i], mb_values[-1][i],
                                                  mb_neglogpacs[-1][i], mb_dones[-1][i], mb_rewards[-1][i]))

            # restart_steps = [[] for _ in range(self.model.n_envs)]
            # subgoals = [[] for _ in range(self.model.n_envs)]
            for idx, done in enumerate(self.dones):
                if self.model.num_timesteps >= self.model.start_augment and done:
                    goal = self.ep_transition_buf[idx][0][0][-self.goal_dim:]
                    if not infos[idx]['is_success']:
                        # Do augmentation
                        # Sample start step and perturbation
                        _restart_steps, _subgoals = self.select_subgoal(self.ep_transition_buf[idx], k=self.n_candidate,
                                                                        dim=self.dim_candidate, env_idx=idx)
                        assert isinstance(_restart_steps, np.ndarray)
                        assert isinstance(_subgoals, np.ndarray)
                        for j in range(_restart_steps.shape[0]):
                            self.restart_steps.append(_restart_steps[j])
                            self.subgoals.append([np.array(_subgoals[j]), goal.copy()])
                            assert len(self.subgoals[-1]) == 2
                            self.restart_states.append(self.ep_state_buf[idx][_restart_steps[j]])
                            self.transition_storage.append(self.ep_transition_buf[idx][:_restart_steps[j]])
                            if self.dim_candidate == 3:
                                self.current_nobject.append(self.ep_current_nobject[idx][0])
                                self.task_mode.append(self.ep_task_mode[idx][0])
                if done:
                    self.ep_state_buf[idx] = []
                    self.ep_transition_buf[idx] = []
                    self.ep_current_nobject[idx] = []
                    self.ep_task_mode[idx] = []


            def convert_dict_to_obs(dict_obs):
                assert isinstance(dict_obs, dict)
                return np.concatenate([dict_obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])

            def switch_subgoal(switch_goal_flag, current_obs):
                for idx, flag in enumerate(switch_goal_flag):
                    if flag:
                        env_subgoals[idx].pop(0)
                        # self.aug_env.set_attr('goal', env_subgoals[idx][0], indices=idx)
                        self.aug_env.env_method('set_goal', [env_subgoals[idx][0]], indices=idx)
                        switch_goal_flag[idx] = False
                        current_obs[idx] = convert_dict_to_obs(self.aug_env.env_method('get_obs', indices=idx)[0])

            # Try to get rid of every for-loop over n_candidate
            temp_time0 = time.time()
            while (len(self.restart_steps) >= self.aug_env.num_envs):
                # TODO Hard work here
                env_restart_steps = self.restart_steps[:self.aug_env.num_envs]
                env_subgoals = self.subgoals[:self.aug_env.num_envs]
                temp_subgoals = [goals[-2] for goals in env_subgoals].copy()
                ultimate_goals = [goals[-1] for goals in env_subgoals]
                env_storage = self.transition_storage[:self.aug_env.num_envs]
                env_increment_storage = [[] for _ in range(self.aug_env.num_envs)]
                self.aug_env.env_method('set_goal', [env_subgoals[idx][0] for idx in range(self.aug_env.num_envs)])
                switch_goal_flag = [False for _ in range(self.aug_env.num_envs)]
                env_end_flag = [False for _ in range(self.aug_env.num_envs)]
                env_end_step = [np.inf for _ in range(self.aug_env.num_envs)]
                env_restart_state = self.restart_states[:self.aug_env.num_envs]
                self.aug_env.env_method('set_state', env_restart_state)
                if self.dim_candidate == 3:
                    self.aug_env.env_method('set_current_nobject', self.current_nobject[:self.aug_env.num_envs])
                    self.aug_env.env_method('set_task_mode', self.task_mode[:self.aug_env.num_envs])
                env_obs = self.aug_env.env_method('get_obs')
                env_obs = [convert_dict_to_obs(d) for d in env_obs]
                increment_step = 0
                while not sum(env_end_flag) == self.aug_env.num_envs:
                    # Switch subgoal according to switch_goal_flag, and update observation
                    switch_subgoal(switch_goal_flag, env_obs)
                    env_action, _, _, _ = self.model.step(np.array(env_obs))
                    relabel_env_obs = self.aug_env.env_method('switch_obs_goal', env_obs, ultimate_goals)
                    clipped_actions = env_action
                    # Clip the actions to avoid out of bound error
                    if isinstance(self.aug_env.action_space, gym.spaces.Box):
                        clipped_actions = np.clip(env_action, self.aug_env.action_space.low, self.aug_env.action_space.high)
                    env_next_obs, _, _, env_info = self.aug_env.step(clipped_actions)
                    self.model.num_aug_steps += (self.aug_env.num_envs - sum(env_end_flag))
                    if self.aug_env.get_attr('reward_type')[0] == 'sparse':
                        temp_info = [None for _ in range(self.aug_env.num_envs)]
                    else:
                        temp_info = [{'previous_obs': env_obs[i]} for i in range(self.aug_env.num_envs)]
                    env_reward = self.aug_env.env_method('compute_reward', env_next_obs, ultimate_goals, temp_info)
                    if self.aug_env.get_attr('reward_type')[0] == 'dense':
                        env_reward_and_success = self.aug_env.env_method('compute_reward_and_success', env_next_obs, ultimate_goals, temp_info)
                    for idx in range(self.aug_env.num_envs):
                        env_increment_storage[idx].append((relabel_env_obs[idx], env_action[idx], False, env_reward[idx]))
                        # if idx == 0:
                        #     print(increment_step, env_obs[idx][:9], env_next_obs[idx][:9], env_reward[idx])
                    env_obs = env_next_obs
                    increment_step += 1
                    # print('increment step', increment_step)

                    for idx, info in enumerate(env_info):
                        # Special case, the agent succeeds the final goal half way
                        if self.aug_env.get_attr('reward_type')[0] == 'sparse' and env_reward[idx] > 0 and env_end_flag[idx] == False:
                            env_end_flag[idx] = True
                            env_end_step[idx] = env_restart_steps[idx] + increment_step
                        elif self.aug_env.get_attr('reward_type')[0] == 'dense' and env_reward_and_success[idx][1] and env_end_flag[idx] == False:
                            env_end_flag[idx] = True
                            env_end_step[idx] = env_restart_steps[idx] + increment_step
                        # Exceed time limit
                        if env_end_flag[idx] == False and env_restart_steps[idx] + increment_step > self.horizon:
                            env_end_flag[idx] = True
                            # But env_end_step is still np.inf
                        if info['is_success']:
                            if len(env_subgoals[idx]) >= 2:
                                switch_goal_flag[idx] = True
                                # if idx == 0:
                                #     print('switch goal')
                            elif env_end_flag[idx] == False:
                                # this is the end
                                env_end_flag[idx] = True
                                env_end_step[idx] = env_restart_steps[idx] + increment_step
                            else:
                                pass
                    if increment_step >= self.horizon - min(env_restart_steps):
                        break

                # print('end step', env_end_step)
                for idx, end_step in enumerate(env_end_step):
                    if end_step <= self.horizon:
                        # print(temp_subgoals[idx])
                        is_self_aug = 1
                        transitions = env_increment_storage[idx][:end_step - env_restart_steps[idx]]
                        augment_obs_buf, augment_act_buf, augment_done_buf, augment_reward_buf = zip(*transitions)
                        augment_value_buf = self.model.value(np.array(augment_obs_buf))
                        augment_neglogp_buf = self.model.sess.run(self.model.aug_neglogpac_op,
                                                                  {self.model.train_aug_model.obs_ph: np.array(augment_obs_buf),
                                                                   self.model.aug_action_ph: np.array(augment_act_buf)})
                        if len(env_storage[idx]):
                            obs_buf, act_buf, value_buf, neglogp_buf, done_buf, reward_buf = zip(*(env_storage[idx]))
                            augment_obs_buf = obs_buf + augment_obs_buf
                            augment_act_buf = act_buf + augment_act_buf
                            augment_value_buf = np.concatenate([np.array(value_buf), augment_value_buf], axis=0)
                            augment_neglogp_buf = np.concatenate([np.array(neglogp_buf), augment_neglogp_buf], axis=0)
                            augment_done_buf = done_buf + augment_done_buf
                            augment_reward_buf = reward_buf + augment_reward_buf
                        if self.aug_env.get_attr('reward_type')[0] == "sparse":
                            assert abs(sum(augment_reward_buf) - 1) < 1e-4
                        if augment_done_buf[0] == 0:
                            augment_done_buf = (True,) + (False,) * (len(augment_done_buf) - 1)
                        augment_isselfaug_buf = (is_self_aug, ) * len(augment_done_buf)
                        augment_returns = self.compute_adv(augment_value_buf, augment_done_buf, augment_reward_buf)
                        assert augment_returns.shape[0] == end_step
                        # if idx == 0:
                        #     print('augment value', augment_value_buf)
                        #     print('augment done', augment_done_buf)
                        #     print('augment_reward', augment_reward_buf)
                        #     print('augment return', augment_returns)
                        if self.model.aug_obs[-1] is None:
                            self.model.aug_obs[-1] = np.array(augment_obs_buf)
                            self.model.aug_act[-1] = np.array(augment_act_buf)
                            self.model.aug_neglogp[-1] = np.array(augment_neglogp_buf)
                            self.model.aug_value[-1] = np.array(augment_value_buf)
                            self.model.aug_return[-1] = augment_returns
                            self.model.aug_done[-1] = np.array(augment_done_buf)
                            self.model.aug_reward[-1] = np.array(augment_reward_buf)
                            self.model.is_selfaug[-1] = np.array(augment_isselfaug_buf)
                            for reuse_idx in range(len(self.model.aug_obs) - 1):
                                # Update previous data with new value and policy parameters
                                if self.model.aug_obs[reuse_idx] is not None:
                                    self.model.aug_neglogp[reuse_idx] = self.model.sess.run(self.model.aug_neglogpac_op,
                                                                                            {self.model.train_aug_model.obs_ph: self.model.aug_obs[reuse_idx],
                                                                                             self.model.aug_action_ph: self.model.aug_act[reuse_idx]})
                                    self.model.aug_value[reuse_idx] = self.model.value(self.model.aug_obs[reuse_idx])
                                    self.model.aug_return[reuse_idx] = self.compute_adv(
                                        self.model.aug_value[reuse_idx], self.model.aug_done[reuse_idx], self.model.aug_reward[reuse_idx])
                                    # print('aug_done', self.model.aug_done[reuse_idx])
                                    # print('aug_reward', self.model.aug_reward[reuse_idx])
                                    # print('aug return', self.model.aug_return[reuse_idx])

                        else:
                            self.model.aug_obs[-1] = np.concatenate([self.model.aug_obs[-1], np.array(augment_obs_buf)], axis=0)
                            self.model.aug_act[-1] = np.concatenate([self.model.aug_act[-1], np.array(augment_act_buf)], axis=0)
                            self.model.aug_neglogp[-1] = np.concatenate(
                                [self.model.aug_neglogp[-1], np.array(augment_neglogp_buf)], axis=0)
                            self.model.aug_value[-1] = np.concatenate(
                                [self.model.aug_value[-1], np.array(augment_value_buf)], axis=0)
                            self.model.aug_return[-1] = np.concatenate([self.model.aug_return[-1], augment_returns], axis=0)
                            self.model.aug_done[-1] = np.concatenate(
                                [self.model.aug_done[-1], np.array(augment_done_buf)], axis=0)
                            self.model.aug_reward[-1] = np.concatenate(
                                [self.model.aug_reward[-1], np.array(augment_reward_buf)], axis=0)
                            self.model.is_selfaug[-1] = np.concatenate(
                                [self.model.is_selfaug[-1], np.array(augment_isselfaug_buf)], axis=0)


                self.restart_steps = self.restart_steps[self.aug_env.num_envs:]
                self.subgoals = self.subgoals[self.aug_env.num_envs:]
                self.restart_states = self.restart_states[self.aug_env.num_envs:]
                self.transition_storage = self.transition_storage[self.aug_env.num_envs:]
                self.current_nobject = self.current_nobject[self.aug_env.num_envs:]
                self.task_mode = self.task_mode[self.aug_env.num_envs:]

            temp_time1 = time.time()
            duration += (temp_time1 - temp_time0)
            # Decrepated code.
            # for env_idx in range(self.model.n_envs):
            #     if len(restart_steps[env_idx]) == 0:
            #         continue
            #     env_restart_steps = restart_steps[env_idx].tolist()
            #     env_subgoals = subgoals[env_idx].tolist()
            #     env_storage = [self.ep_transition_buf[env_idx][:_restart_step] for _restart_step in env_restart_steps]
            #     env_increment_storage = [[] for _ in env_restart_steps]
            #     ultimate_goal = self.ep_transition_buf[env_idx][0][0][-self.goal_dim:]
            #     for i in range(self.n_candidate): # len(env_subgoals) should be equal to n_candidates
            #         env_subgoals[i] = [np.array(env_subgoals[i]), ultimate_goal]
            #     # print('env_subgoals', env_subgoals)
            #     # switch_goal_flag = [True for _ in range(len(env_subgoals))]
            #     self.aug_env.env_method('set_goal', [env_subgoals[idx][0] for idx in range(self.n_candidate)])
            #     switch_goal_flag = [False for _ in range(self.n_candidate)]
            #     env_end_flag = [False for _ in range(self.n_candidate)]
            #     env_end_step = [np.inf for _ in range(self.n_candidate)]
            #     env_restart_state = [self.ep_state_buf[env_idx][step] for step in env_restart_steps]
            #     # temp_time2 = time.time()
            #     self.aug_env.env_method('set_state', env_restart_state)
            #     # step_env_duration += (time.time() - temp_time2)
            #     env_obs = self.aug_env.env_method('get_obs')
            #     env_obs = [convert_dict_to_obs(d) for d in env_obs]
            #     # print('checking goal', self.aug_env.get_attr('goal'))
            #     # print('getting obs', env_obs[0])
            #     # Parallel rollout subtask
            #     increment_step = 0
            #     # print('restart step', env_restart_steps)
            #     while not sum(env_end_flag) == self.n_candidate:
            #         # Switch subgoal according to switch_goal_flag, and update observation
            #         switch_subgoal(switch_goal_flag, env_obs)
            #         # Update env_subgoals.
            #         # switch_subgoal(switch_goal_flag)
            #         # print(env_obs)
            #         env_action, _, _, _ = self.model.step(np.array(env_obs))
            #         relabel_env_obs = self.aug_env.env_method('switch_obs_goal', env_obs, [ultimate_goal for _ in range(self.n_candidate)])
            #         clipped_actions = env_action
            #         # Clip the actions to avoid out of bound error
            #         if isinstance(self.aug_env.action_space, gym.spaces.Box):
            #             clipped_actions = np.clip(env_action, self.aug_env.action_space.low, self.aug_env.action_space.high)
            #         # temp_time2 = time.time()
            #         env_next_obs, _, _, env_info = self.aug_env.step(clipped_actions)
            #         self.model.num_aug_steps += (self.n_candidate - sum(env_end_flag))
            #         # step_env_duration += (time.time() - temp_time2)
            #         # for i, info in enumerate(env_info):
            #         #     if 'terminal_observation' in info.keys():
            #         #         assert 'terminal_state' in info.keys()
            #         #         env_next_obs[i] = info['terminal_observation']
            #         #         self.aug_env.env_method('set_state', [info['terminal_state']], indices=i)
            #         env_reward = self.aug_env.env_method('compute_reward', env_next_obs, [ultimate_goal for _ in range(self.n_candidate)], [None for _ in range(self.n_candidate)])
            #         for idx in range(self.n_candidate):
            #             env_increment_storage[idx].append((relabel_env_obs[idx], env_action[idx], False, env_reward[idx]))
            #             # if idx == 0:
            #             #     print(increment_step, env_obs[idx][:9], env_next_obs[idx][:9], env_reward[idx])
            #         env_obs = env_next_obs
            #         increment_step += 1
            #         # print('increment step', increment_step)
            #
            #         for idx, info in enumerate(env_info):
            #             # Special case, the agent succeeds the final goal half way
            #             if env_reward[idx] > 0 and env_end_flag[idx] == False:
            #                 env_end_flag[idx] = True
            #                 env_end_step[idx] = env_restart_steps[idx] + increment_step
            #             if info['is_success']:
            #                 if len(env_subgoals[idx]) >= 2:
            #                     switch_goal_flag[idx] = True
            #                     # if idx == 0:
            #                     #     print('switch goal')
            #                 elif env_end_flag[idx] == False:
            #                     # this is the end
            #                     env_end_flag[idx] = True
            #                     env_end_step[idx] = env_restart_steps[idx] + increment_step
            #                 else:
            #                     pass
            #         if increment_step >= self.horizon - min(env_restart_steps):
            #             break
            #
            #     # print(env_idx, 'end step', env_end_step)
            #     # DEBUG: log values and indicator of success
            #     if not os.path.exists(os.path.join(logger.get_dir(), 'debug_value.csv')):
            #         with open(os.path.join(logger.get_dir(), 'debug_value.csv'), 'a', newline='') as csvfile:
            #             csvwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            #             title = ['reference_value', 'value1', 'value2', 'is_success']
            #             csvwriter.writerow(title)
            #     with open(os.path.join(logger.get_dir(), 'debug_value.csv'), 'a', newline='') as csvfile:
            #         csvwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            #         for i in range(len(env_end_step)):
            #             data = [self.debug_reference_value[env_idx][i], self.debug_value1[env_idx][i],
            #                     self.debug_value2[env_idx][i], int(env_end_step[i] <= self.horizon)]
            #             csvwriter.writerow(data)
            #     # End DEBUG
            #
            #     for idx, end_step in enumerate(env_end_step):
            #         if end_step <= self.horizon:
            #             transitions = env_increment_storage[idx][:end_step - env_restart_steps[idx]]
            #             augment_obs_buf, augment_act_buf, augment_done_buf, augment_reward_buf = zip(*transitions)
            #             augment_value_buf = self.model.value(np.array(augment_obs_buf))
            #             augment_neglogp_buf = self.model.sess.run(self.model.aug_neglogpac_op,
            #                                                       {self.model.train_aug_model.obs_ph: np.array(augment_obs_buf),
            #                                                        self.model.aug_action_ph: np.array(augment_act_buf)})
            #             if len(env_storage[idx]):
            #                 obs_buf, act_buf, value_buf, neglogp_buf, done_buf, reward_buf = zip(*(env_storage[idx]))
            #                 augment_obs_buf = obs_buf + augment_obs_buf
            #                 augment_act_buf = act_buf + augment_act_buf
            #                 augment_value_buf = np.concatenate([np.array(value_buf), augment_value_buf], axis=0)
            #                 augment_neglogp_buf = np.concatenate([np.array(neglogp_buf), augment_neglogp_buf], axis=0)
            #                 augment_done_buf = done_buf + augment_done_buf
            #                 augment_reward_buf = reward_buf + augment_reward_buf
            #             assert abs(sum(augment_reward_buf) - 1) < 1e-4
            #             if augment_done_buf[0] == 0:
            #                 augment_done_buf = (True,) + (False,) * (len(augment_done_buf) - 1)
            #             augment_returns = self.compute_adv(augment_value_buf, augment_done_buf, augment_reward_buf)
            #             assert augment_returns.shape[0] == end_step
            #             # if idx == 0:
            #             #     print('augment value', augment_value_buf)
            #             #     print('augment done', augment_done_buf)
            #             #     print('augment_reward', augment_reward_buf)
            #             #     print('augment return', augment_returns)
            #             if self.model.aug_obs is None:
            #                 self.model.aug_obs = np.array(augment_obs_buf)
            #                 self.model.aug_act = np.array(augment_act_buf)
            #                 self.model.aug_neglogp = np.array(augment_neglogp_buf)
            #                 self.model.aug_value = np.array(augment_value_buf)
            #                 self.model.aug_return = augment_returns
            #                 self.model.aug_done = np.array(augment_done_buf)
            #             else:
            #                 self.model.aug_obs = np.concatenate([self.model.aug_obs, np.array(augment_obs_buf)], axis=0)
            #                 self.model.aug_act = np.concatenate([self.model.aug_act, np.array(augment_act_buf)], axis=0)
            #                 self.model.aug_neglogp = np.concatenate(
            #                     [self.model.aug_neglogp, np.array(augment_neglogp_buf)], axis=0)
            #                 self.model.aug_value = np.concatenate([self.model.aug_value, np.array(augment_value_buf)],
            #                                                       axis=0)
            #                 self.model.aug_return = np.concatenate([self.model.aug_return, augment_returns], axis=0)
            #                 self.model.aug_done = np.concatenate([self.model.aug_done, np.array(augment_done_buf)],
            #                                                      axis=0)
            #
            # for idx, done in enumerate(self.dones):
            #     if done:
            #         self.ep_state_buf[idx] = []
            #         self.ep_transition_buf[idx] = []
            # temp_time1 = time.time()
            # duration += (temp_time1 - temp_time0)
            # print('end')
            # exit()


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
            if self.env.get_attr('spec')[0].id == 'MasspointMaze-v3':
                obstacle_xy = sample_obs[:, 0:2] + noise
                sample_obs[:, 0:2] = obstacle_xy
                sample_obs[:, self.obs_dim: self.obs_dim + 2] = obstacle_xy
                sample_obs_buf.append(sample_obs.copy())
                subgoal_obs = obs_buf[sample_t]
                subgoal_obs[:, self.obs_dim + self.goal_dim: self.obs_dim + self.goal_dim + 2] = obstacle_xy
                subgoal_obs_buf.append(subgoal_obs)
            else:
                obstacle_xy = sample_obs[:, 0:2] + noise
                sample_obs[:, 0:2] = obstacle_xy
                sample_obs[:, self.obs_dim:self.obs_dim + 2] = obstacle_xy
                sample_obs_buf.append(sample_obs.copy())

                subgoal_obs = obs_buf[sample_t]
                # if debug:
                #     subgoal_obs = np.tile(subgoal_obs, (2, 1))
                subgoal_obs[:, self.obs_dim:self.obs_dim+3] = subgoal_obs[:, 0:3]
                # one_hot = np.zeros(self.n_object)
                # one_hot[object_idx] = 1
                # subgoal_obs[:, self.obs_dim+3:self.obs_dim+self.goal_dim] = one_hot
                subgoal_obs[:, self.obs_dim+self.goal_dim:self.obs_dim+self.goal_dim+2] = obstacle_xy
                subgoal_obs[:, self.obs_dim+self.goal_dim+2:self.obs_dim+self.goal_dim+3] = subgoal_obs[:, 2:3]
                # subgoal_obs[:, self.obs_dim+self.goal_dim+3:self.obs_dim+self.goal_dim*2] = one_hot
                subgoal_obs_buf.append(subgoal_obs)
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
        self.self_aug_ratio.append(1)

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
