import time
import sys
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


class ParallelRunner(AbstractEnvRunner):
    def __init__(self, *, env, aug_env, model, n_steps, gamma, lam, n_candidate):
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

        duration = 0.0
        step_env_duration = 0.0
        for _ in range(self.n_steps):
            internal_states = self.env.env_method('get_state')
            for i in range(self.model.n_envs):
                self.ep_state_buf[i].append(internal_states[i])
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
            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)
            for i in range(self.model.n_envs):
                self.ep_transition_buf[i].append((mb_obs[-1][i], mb_actions[-1][i], mb_values[-1][i],
                                                  mb_neglogpacs[-1][i], mb_dones[-1][i], mb_rewards[-1][i]))
            # _values = self.env.env_method('augment_data', self.ep_transition_buf, self.ep_state_buf)
            # print(_values)
            # exit()
            restart_steps = [[] for _ in range(self.model.n_envs)]
            subgoals = [[] for _ in range(self.model.n_envs)]
            for idx, done in enumerate(self.dones):
                if done:
                    goal = self.ep_transition_buf[idx][0][0][-5:]
                    if np.argmax(goal[3:]) == 0 and (not infos[idx]['is_success']):
                        # Do augmentation
                        # Sample start step and perturbation
                        _restart_steps, _subgoals = self.select_subgoal(self.ep_transition_buf[idx], k=self.n_candidate)
                        assert isinstance(_restart_steps, np.ndarray)
                        assert isinstance(_subgoals, np.ndarray)
                        restart_steps[idx] = _restart_steps
                        subgoals[idx] = _subgoals

            # I cannot avoid calling tensorflow sess, but it is not pickable.
            #  result = self.env.env_method('augment_data', self.ep_transition_buf, self.ep_state_buf,
            #                              restart_steps, subgoals, step_op=self.model.step) # ep_transition_buf[i] will be dispatched to every env
            # print(result)
            # exit()

            # TODO: store the final state of every env for recovering
            def convert_dict_to_obs(dict_obs):
                assert isinstance(dict_obs, dict)
                return np.concatenate([dict_obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
            # Try to get rid of every for-loop over n_candidate
            temp_time0 = time.time()
            for env_idx in range(self.model.n_envs):
                if len(restart_steps[env_idx]) == 0:
                    continue
                env_restart_steps = restart_steps[env_idx].tolist()
                env_subgoals = subgoals[env_idx].tolist()
                env_storage = [self.ep_transition_buf[env_idx][:_restart_step] for _restart_step in env_restart_steps]
                env_increment_storage = [[] for _ in env_restart_steps]
                ultimate_goal = self.ep_transition_buf[env_idx][0][0][-5:]
                for i in range(self.n_candidate): # len(env_subgoals) should be equal to n_candidates
                    env_subgoals[i] = [np.array(env_subgoals[i]), ultimate_goal]
                print('env_subgoals', env_subgoals)
                # switch_goal_flag = [True for _ in range(len(env_subgoals))]
                self.aug_env.env_method('set_goal', [env_subgoals[idx][0] for idx in range(self.n_candidate)])
                switch_goal_flag = [False for _ in range(self.n_candidate)]
                env_end_flag = [False for _ in range(self.n_candidate)]
                env_end_step = [np.inf for _ in range(self.n_candidate)]
                env_restart_state = [self.ep_state_buf[env_idx][step] for step in env_restart_steps]
                temp_time2 = time.time()
                self.aug_env.env_method('set_state', env_restart_state)
                step_env_duration += (time.time() - temp_time2)
                env_obs = self.aug_env.env_method('get_obs')
                env_obs = [convert_dict_to_obs(d) for d in env_obs]
                # print('checking goal', self.aug_env.get_attr('goal'))
                # print('getting obs', env_obs[0])
                # Parallel rollout subtask
                increment_step = 0
                def switch_subgoal(switch_goal_flag, current_obs):
                    for idx, flag in enumerate(switch_goal_flag):
                        if flag:
                            env_subgoals[idx].pop(0)
                            # self.aug_env.set_attr('goal', env_subgoals[idx][0], indices=idx)
                            self.aug_env.env_method('set_goal', [env_subgoals[idx][0]], indices=idx)
                            switch_goal_flag[idx] = False
                            current_obs[idx] = convert_dict_to_obs(self.aug_env.env_method('get_obs', indices=idx)[0])
                # def switch_subgoal(switch_goal_flag):
                #     for idx, flag in enumerate(switch_goal_flag):
                #         if flag:
                #             env_subgoals[idx].pop(0)
                #             switch_goal_flag[idx] = False

                print('restart step', env_restart_steps)
                while not sum(env_end_flag) == self.n_candidate:
                    # Switch subgoal according to switch_goal_flag, and update observation
                    switch_subgoal(switch_goal_flag, env_obs)
                    # Update env_subgoals.
                    # switch_subgoal(switch_goal_flag)
                    # print(env_obs)
                    env_action, _, _, _ = self.model.step(np.array(env_obs))
                    relabel_env_obs = self.aug_env.env_method('switch_obs_goal', env_obs, [ultimate_goal for _ in range(self.n_candidate)])
                    clipped_actions = env_action
                    # Clip the actions to avoid out of bound error
                    if isinstance(self.aug_env.action_space, gym.spaces.Box):
                        clipped_actions = np.clip(env_action, self.aug_env.action_space.low, self.aug_env.action_space.high)
                    temp_time2 = time.time()
                    env_next_obs, _, _, env_info = self.aug_env.step(clipped_actions)
                    step_env_duration += (time.time() - temp_time2)
                    # for i, info in enumerate(env_info):
                    #     if 'terminal_observation' in info.keys():
                    #         assert 'terminal_state' in info.keys()
                    #         env_next_obs[i] = info['terminal_observation']
                    #         self.aug_env.env_method('set_state', [info['terminal_state']], indices=i)
                    env_reward = self.aug_env.env_method('compute_reward', env_next_obs, [ultimate_goal for _ in range(self.n_candidate)], [None for _ in range(self.n_candidate)])
                    for idx in range(self.n_candidate):
                        env_increment_storage[idx].append((relabel_env_obs[idx], env_action[idx], False, env_reward[idx]))
                        # if idx == 0:
                        #     print(increment_step, env_obs[idx][:9], env_next_obs[idx][:9], env_reward[idx])
                    env_obs = env_next_obs
                    increment_step += 1
                    # print('increment step', increment_step)

                    for idx, info in enumerate(env_info):
                        # Special case, the agent succeeds the final goal half way
                        if env_reward[idx] > 0 and env_end_flag[idx] == False:
                            env_end_flag[idx] = True
                            env_end_step[idx] = env_restart_steps[idx] + increment_step
                        if info['is_success']:
                            if len(env_subgoals[idx]) >= 2:
                                switch_goal_flag[idx] = True
                                if idx == 0:
                                    print('switch goal')
                            elif env_end_flag[idx] == False:
                                # this is the end
                                env_end_flag[idx] = True
                                env_end_step[idx] = env_restart_steps[idx] + increment_step
                            else:
                                pass
                    if increment_step >= 100 - min(env_restart_steps):
                        break

                print('end step', env_end_step)
                for idx, end_step in enumerate(env_end_step):
                    if end_step <= 100:
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
                        assert abs(sum(augment_reward_buf) - 1) < 1e-4
                        if augment_done_buf[0] == 0:
                            augment_done_buf = (True,) + (False,) * (len(augment_done_buf) - 1)
                        augment_returns = self.compute_adv(augment_value_buf, augment_done_buf, augment_reward_buf)
                        assert augment_returns.shape[0] == end_step
                        if idx == 0:
                            print('augment value', augment_value_buf)
                            print('augment done', augment_done_buf)
                            print('augment_reward', augment_reward_buf)
                            print('augment return', augment_returns)
                        if self.model.aug_obs is None:
                            self.model.aug_obs = np.array(augment_obs_buf)
                            self.model.aug_act = np.array(augment_act_buf)
                            self.model.aug_neglogp = np.array(augment_neglogp_buf)
                            self.model.aug_value = np.array(augment_value_buf)
                            self.model.aug_return = augment_returns
                            self.model.aug_done = np.array(augment_done_buf)
                        else:
                            self.model.aug_obs = np.concatenate([self.model.aug_obs, np.array(augment_obs_buf)], axis=0)
                            self.model.aug_act = np.concatenate([self.model.aug_act, np.array(augment_act_buf)], axis=0)
                            self.model.aug_neglogp = np.concatenate(
                                [self.model.aug_neglogp, np.array(augment_neglogp_buf)], axis=0)
                            self.model.aug_value = np.concatenate([self.model.aug_value, np.array(augment_value_buf)],
                                                                  axis=0)
                            self.model.aug_return = np.concatenate([self.model.aug_return, augment_returns], axis=0)
                            self.model.aug_done = np.concatenate([self.model.aug_done, np.array(augment_done_buf)],
                                                                 axis=0)

            for idx, done in enumerate(self.dones):
                if done:
                    self.ep_state_buf[idx] = []
                    self.ep_transition_buf[idx] = []
            temp_time1 = time.time()
            duration += (temp_time1 - temp_time0)
            # print('end')
            # exit()




            # for idx, done in enumerate(self.dones):
            #     if done:
            #         # Check if this is failture
            #         goal = self.ep_transition_buf[idx][0][0][-5:]
            #         if np.argmax(goal[3:]) == 0 and (not infos[idx]['is_success']):
            #             # Do augmentation
            #             for k in range(restart_steps.shape[0]):
            #                 restart_step = restart_steps[k]
            #                 subgoal = subgoals[k]
            #                 if restart_step > 0:
            #                     augment_obs_buf, augment_act_buf, augment_value_buf, \
            #                     augment_neglogp_buf, augment_done_buf, augment_reward_buf = \
            #                         zip(*self.ep_transition_buf[idx][:restart_step])
            #                     augment_obs_buf = list(augment_obs_buf)
            #                     augment_act_buf = list(augment_act_buf)
            #                     augment_value_buf = list(augment_value_buf)
            #                     augment_neglogp_buf = list(augment_neglogp_buf)
            #                     augment_done_buf = list(augment_done_buf)
            #                     augment_reward_buf = list(augment_reward_buf)
            #                 else:
            #                     augment_obs_buf, augment_act_buf, augment_value_buf, \
            #                     augment_neglogp_buf, augment_done_buf, augment_reward_buf = [], [], [], [], [], []
            #                 augment_obs1, augment_act1, augment_value1, augment_neglogp1, augment_done1, augment_reward1, next_state = \
            #                     self.rollout_subtask(self.ep_state_buf[idx][restart_step], subgoal, len(augment_obs_buf), goal)
            #                 if augment_obs1 is not None:
            #                     # augment_transition_buf += augment_transition1
            #                     augment_obs_buf += augment_obs1
            #                     augment_act_buf += augment_act1
            #                     augment_value_buf += augment_value1
            #                     augment_neglogp_buf += augment_neglogp1
            #                     augment_done_buf += augment_done1
            #                     augment_reward_buf += augment_reward1
            #                     augment_obs2, augment_act2, augment_value2, augment_neglogp2, augment_done2, augment_reward2, _ = \
            #                         self.rollout_subtask(next_state, goal, len(augment_obs_buf), goal)
            #                     if augment_obs2 is not None:
            #                         print('Success')
            #                         # augment_transition_buf += augment_transition2
            #                         augment_obs_buf += augment_obs2
            #                         augment_act_buf += augment_act2
            #                         augment_value_buf += augment_value2
            #                         augment_neglogp_buf += augment_neglogp2
            #                         augment_done_buf += augment_done2
            #                         augment_reward_buf += augment_reward2
            #
            #                         augment_returns = self.compute_adv(augment_value_buf, augment_done_buf, augment_reward_buf)
            #                         # aug_obs, aug_act = zip(*augment_transition_buf)
            #                         # print(len(augment_obs_buf), len(augment_act_buf), len(augment_neglogp_buf))
            #                         # print(augment_adv_buf)
            #                         # The augment data is directly passed to model
            #                         # self.model.aug_obs += list(aug_obs)
            #                         # self.model.aug_act += list(aug_act)
            #                         for i in range(len(augment_obs_buf)):
            #                             assert np.argmax(augment_obs_buf[i][-2:]) == 0
            #                             assert np.argmax(augment_obs_buf[i][-7:-5]) == 0
            #                         # self.model.aug_obs += augment_obs_buf
            #                         # self.model.aug_act += augment_act_buf
            #                         # self.model.aug_neglogp += augment_neglogp_buf
            #                         # self.model.aug_adv += augment_adv_buf
            #                         if self.model.aug_obs is None:
            #                             self.model.aug_obs = np.array(augment_obs_buf)
            #                             self.model.aug_act = np.array(augment_act_buf)
            #                             self.model.aug_neglogp = np.array(augment_neglogp_buf)
            #                             self.model.aug_value = np.array(augment_value_buf)
            #                             self.model.aug_return = augment_returns
            #                             self.model.aug_done = np.array(augment_done_buf)
            #                         else:
            #                             self.model.aug_obs = np.concatenate([self.model.aug_obs, np.array(augment_obs_buf)], axis=0)
            #                             self.model.aug_act = np.concatenate([self.model.aug_act, np.array(augment_act_buf)], axis=0)
            #                             self.model.aug_neglogp = np.concatenate([self.model.aug_neglogp, np.array(augment_neglogp_buf)],axis=0)
            #                             self.model.aug_value = np.concatenate([self.model.aug_value, np.array(augment_value_buf)], axis=0)
            #                             self.model.aug_return = np.concatenate([self.model.aug_return, augment_returns], axis=0)
            #                             self.model.aug_done = np.concatenate([self.model.aug_done, np.array(augment_done_buf)], axis=0)
            #                     # else:
            #                     #     print('Failed to achieve ultimate goal')
            #                 # else:
            #                 #     print('Failed to achieve subgoal')
            #
            #         # Then update buf
            #         self.ep_state_buf[idx] = []
            #         self.ep_transition_buf[idx] = []
        print('augment takes', duration)
        print('augment stepping env takes', step_env_duration)
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

    def select_subgoal(self, transition_buf, k):
        # self.ep_transition_buf, self.model.value
        assert len(transition_buf) == 100, len(transition_buf)
        obs_buf, *_ = zip(*transition_buf)
        obs_buf = np.asarray(obs_buf)
        sample_t = np.random.randint(0, len(transition_buf), 4096)
        sample_obs = obs_buf[sample_t]
        noise = np.random.uniform(low=-0.15, high=0.15, size=(len(sample_t), 2))
        obstacle_xy = sample_obs[:, 6:8] + noise
        sample_obs[:, 6:8] = obstacle_xy
        sample_obs[:, 12:14] = sample_obs[:, 6:8] - sample_obs[:, 0:2]
        value2 = self.model.value(sample_obs)
        subgoal_obs = obs_buf[sample_t]
        subgoal_obs[:, 40:43] = subgoal_obs[:, 6:9]
        subgoal_obs[:, 43:45] = np.array([[0., 1.]])
        subgoal_obs[:, 45:47] = obstacle_xy
        subgoal_obs[:, 47:48] = subgoal_obs[:, 8:9]
        subgoal_obs[:, 48:50] = np.array([[0., 1.]])
        value1 = self.model.value(subgoal_obs)
        normalize_value1 = (value1 - np.min(value1)) / (np.max(value1) - np.min(value1))
        normalize_value2 = (value2 - np.min(value2)) / (np.max(value2) - np.min(value2))
        # best_idx = np.argmax(normalize_value1 * normalize_value2)
        ind = np.argsort(normalize_value1 * normalize_value2)
        good_ind = ind[-k:]
        # restart_step = sample_t[best_idx]
        # subgoal = subgoal_obs[best_idx, 45:50]
        restart_step = sample_t[good_ind]
        subgoal = subgoal_obs[good_ind, 45:50]
        # print('subgoal', subgoal, 'with value1', normalize_value1[best_idx], 'value2', normalize_value2[best_idx])
        # print('restart step', restart_step)
        return restart_step, subgoal

    def rollout_subtask(self, restart_state, goal, restart_step, ultimate_goal):
        aug_transition = []
        self.aug_env.unwrapped.sim.set_state(restart_state)
        self.aug_env.unwrapped.goal[:] = goal
        dict_obs = self.aug_env.unwrapped.get_obs()
        obs = np.concatenate([dict_obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
        def switch_goal(obs, goal):
            assert len(goal) == 5
            obs[-5:] = goal
            goal_idx = np.argmax(goal[3:])
            obs[-10:-5] = np.concatenate([obs[3 + goal_idx * 3 : 6 + goal_idx * 3], goal[3:5]])
            return obs
        info = {'is_success': False}
        for step_idx in range(restart_step, 100):
            # If I use subgoal obs, value has problem
            # If I use ultimate goal obs, action should be rerunned
            # action, value, _, neglogpac = self.model.step(obs)
            action, _, _, _ = self.model.step(np.expand_dims(obs, axis=0))
            action = np.squeeze(action, axis=0)
            clipped_actions = action
            # Clip the actions to avoid out of bound error
            if isinstance(self.aug_env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(action, self.aug_env.action_space.low, self.aug_env.action_space.high)
            next_obs, _, _, info = self.aug_env.step(clipped_actions)
            reward = self.aug_env.compute_reward(switch_goal(next_obs, ultimate_goal), ultimate_goal, None)
            next_state = self.aug_env.unwrapped.sim.get_state()
            # aug_transition.append((obs, action, value, neglogpac, done, reward))
            aug_transition.append((switch_goal(obs, ultimate_goal), action, False, reward)) # Note that done refers to the output of previous action
            if info['is_success']:
                break
            obs = next_obs
        # print('length of augment transition', len(aug_transition))
        if info['is_success']:
            aug_obs, aug_act, aug_done, aug_reward = zip(*aug_transition)
            aug_obs = list(aug_obs)
            aug_act = list(aug_act)
            aug_done = list(aug_done)
            aug_reward = list(aug_reward)
            aug_neglogpac = self.model.sess.run(self.model.aug_neglogpac_op,
                                                {self.model.train_aug_model.obs_ph: np.array(aug_obs),
                                                 self.model.aug_action_ph: np.array(aug_act)})
            aug_value = self.model.value(np.array(aug_obs))
            # print(aug_neglogpac.shape)
            aug_neglogpac = aug_neglogpac.tolist()
            aug_value = aug_value.tolist()
            # print(np.mean(aug_neglogpac))
            return aug_obs, aug_act, aug_value, aug_neglogpac, aug_done, aug_reward, next_state
        return None, None, None, None, None, None, None

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
