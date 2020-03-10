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
from utils.eval_stack import pp_eval_model


class PPO2_augment_IS(ActorCriticRLModel):
    """
    Proximal Policy Optimization algorithm (GPU version).
    Paper: https://arxiv.org/abs/1707.06347

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param cliprange_vf: (float or callable) Clipping parameter for the value function, it can be a function.
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling.
        To deactivate value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env, aug_env=None, eval_env=None, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4,
                 vf_coef=0.5, aug_clip=0.1, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2,
                 cliprange_vf=None, n_candidate=4, dim_candidate=2, parallel=False, reuse_times=1, start_augment=0,
                 horizon=100, aug_adv_weight=1.0, curriculum=False, self_imitate=False,
                 verbose=0, tensorboard_log=None, _init_setup_model=True,
                 policy_kwargs=None, full_tensorboard_log=False):

        super(PPO2_augment_IS, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                              _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)

        self.aug_env = aug_env
        self.eval_env = eval_env
        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.cliprange_vf = cliprange_vf
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.aug_clip = aug_clip
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.n_candidate = n_candidate
        self.dim_candidate = dim_candidate
        self.parallel = parallel
        self.start_augment = start_augment
        self.curriculum = curriculum
        self.self_imitate = self_imitate
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.graph = None
        self.sess = None
        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.clipfrac = None
        self.params = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.step = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.n_batch = None
        self.summary = None
        self.episode_reward = None

        self.reuse_times = reuse_times
        self.aug_obs = []
        self.aug_act = []
        self.aug_neglogp = []
        self.aug_return = []
        self.aug_value = []
        self.aug_done = []
        self.aug_reward = []
        self.is_selfaug = []
        self.num_aug_steps = 0  # every interaction with simulator should be counted
        self.horizon = horizon
        self.aug_adv_weight = aug_adv_weight

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.n_batch = self.n_envs * self.n_steps

            n_cpu = multiprocessing.cpu_count()
            if sys.platform == 'darwin':
                n_cpu //= 2

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, "\
                        "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step, reuse=False, **self.policy_kwargs)
                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=True, **self.policy_kwargs)
                    # train_aug_model = self.policy(self.sess, self.observation_space, self.action_space,
                    #                               self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                    #                               reuse=True, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")
                    self.is_demo_ph = tf.placeholder(tf.float32, [None], name="is_demo_ph")

                    # self.aug_action_ph = train_aug_model.pdtype.sample_placeholder([None], name="aug_action_ph")
                    # self.aug_advs_ph = tf.placeholder(tf.float32, [None], name="aug_advs_ph")
                    # self.aug_old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="aug_old_neglog_pac_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    # aug_neglogpac = train_aug_model.proba_distribution.neglogp(self.aug_action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                    vpred = train_model.value_flat

                    # Value function clipping: not present in the original PPO
                    if self.cliprange_vf is None:
                        # Default behavior (legacy from OpenAI baselines):
                        # use the same clipping as for the policy
                        self.clip_range_vf_ph = self.clip_range_ph
                        self.cliprange_vf = self.cliprange
                    elif isinstance(self.cliprange_vf, (float, int)) and self.cliprange_vf < 0:
                        # Original PPO implementation: no value function clipping
                        self.clip_range_vf_ph = None
                    else:
                        # Last possible behavior: clipping range
                        # specific to the value function
                        self.clip_range_vf_ph = tf.placeholder(tf.float32, [], name="clip_range_vf_ph")

                    if self.clip_range_vf_ph is None:
                        # No clipping
                        vpred_clipped = train_model.value_flat
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        vpred_clipped = self.old_vpred_ph + \
                            tf.clip_by_value(train_model.value_flat - self.old_vpred_ph,
                                             - self.clip_range_vf_ph, self.clip_range_vf_ph)


                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpred_clipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    if self.self_imitate:
                        ratio = tf.exp(self.old_neglog_pac_ph - tf.minimum(neglogpac, 100))
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                      self.clip_range_ph), tf.float32))
                    self.demo_clipfrac = tf.reduce_mean(self.is_demo_ph * tf.cast(
                        tf.greater(tf.abs(ratio - 1.0), self.clip_range_ph), tf.float32)) / (tf.reduce_mean(self.is_demo_ph) + 1e-5)
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('loss', loss)

                    # print(tf.trainable_variables())
                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                        # print(self.params)
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.summary.histogram(var.name, var)
                    grads = tf.gradients(loss, self.params)
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))
                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = trainer.apply_gradients(grads)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'demo_clipfrac']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    if self.clip_range_vf_ph is not None:
                        tf.summary.scalar('clip_range_vf', tf.reduce_mean(self.clip_range_vf_ph))

                    tf.summary.scalar('old_neglog_action_probabilty', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        tf.summary.histogram('clip_range', self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probabilty', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                self.train_model = train_model
                # self.train_aug_model = train_aug_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                # self.aug_neglogpac_op = aug_neglogpac
                self.neglogpac_op = neglogpac
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                self.summary = tf.summary.merge_all()

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, is_demo, update,
                    writer, states=None, cliprange_vf=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        :param cliprange_vf: (float) Clipping factor for the value function
        """
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        # for i in range(advs.shape[0]):
        #     if is_demo[i]:
        #         advs[i] = np.max([advs[i], self.aug_clip]) * self.aug_adv_weight
        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions,
                  self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values,
                  self.is_demo_ph: is_demo,
                  }
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf

        if states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, demo_clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self.demo_clipfrac, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, demo_clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self.demo_clipfrac, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, demo_clipfrac, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self.demo_clipfrac, self._train], td_map)
            # if aug_obs_slice is not None:
            #     print('demo loss', demo_loss)
                # exit()

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac, demo_clipfrac

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True):
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            if self.env.get_attr('n_object')[0] > 0:
                if self.dim_candidate != 3:
                    from baselines.ppo_augment_is.parallel_runner2 import ParallelRunner2
                    # runner = ParallelRunner(env=self.env, aug_env=self.aug_env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam,
                    #                         n_candidate=self.n_candidate, horizon=self.horizon)
                    runner = ParallelRunner2(env=self.env, aug_env=self.aug_env, model=self, n_steps=self.n_steps,
                                             gamma=self.gamma, lam=self.lam, n_candidate=self.n_candidate,
                                             dim_candidate=self.dim_candidate,
                                             horizon=self.horizon)
                else:
                    raise NotImplementedError
            else:
                # Maze.
                raise NotImplementedError
                from baselines.ppo_augment.parallel_runner2_maze import ParallelRunner2
                runner = ParallelRunner2(env=self.env, aug_env=self.aug_env, model=self, n_steps=self.n_steps,
                                         gamma=self.gamma, lam=self.lam, n_candidate=self.n_candidate,
                                         dim_candidate=self.dim_candidate, horizon=self.horizon)
            self.episode_reward = np.zeros((self.n_envs,))

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()

            n_updates = total_timesteps // self.n_batch
            total_success = 0
            original_success = 0
            _reuse_times = self.reuse_times
            start_decay = n_updates
            pp_sr_buf = deque(maxlen=5)
            for update in range(1, n_updates + 1):
                assert self.n_batch % self.nminibatches == 0
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = self.learning_rate(frac)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)
                if self.curriculum:
                    if 'FetchStack' in self.env.get_attr('spec')[0].id:
                        # Stacking
                        pp_sr = pp_eval_model(self.eval_env, self)
                        pp_sr_buf.append(pp_sr)
                        print('Pick-and-place success rate', np.mean(pp_sr_buf))
                        if start_decay == n_updates and np.mean(pp_sr_buf) > 0.8:
                            start_decay = update
                        _ratio = np.clip(0.7 - 0.8 * (update - start_decay) / 380, 0.3, 0.7) # from 0.7 to 0.3
                    elif 'FetchPushWallObstacle' in self.env.get_attr('spec')[0].id:
                        _ratio = max(1.0 - (update - 1.0) / n_updates, 0.0)
                    else:
                        raise NotImplementedError
                    self.env.env_method('set_random_ratio', _ratio)
                    print('Set random_ratio to', self.env.get_attr('random_ratio')[0])
                aug_success_ratio = (total_success - original_success) / (total_success + 1e-8)
                if aug_success_ratio > 1.0:
                # if aug_success_ratio > 0.25:
                    _reuse_times -= 1
                    _reuse_times = max(1, _reuse_times)
                else:
                    _reuse_times = min(self.reuse_times, _reuse_times + 1)
                # Reuse goalidx=0 only once
                # if len(self.aug_obs) and (self.aug_obs[-1] is not None):
                #     other_aug_idx = np.where(self.is_selfaug[-1] < 0.5)[0]
                #     self.aug_obs[-1] = self.aug_obs[-1][other_aug_idx] if len(other_aug_idx) else None
                #     self.aug_act[-1] = self.aug_act[-1][other_aug_idx] if len(other_aug_idx) else None
                #     self.aug_neglogp[-1] = self.aug_neglogp[-1][other_aug_idx] if len(other_aug_idx) else None
                #     self.aug_return[-1] = self.aug_return[-1][other_aug_idx] if len(other_aug_idx) else None
                #     self.aug_value[-1] = self.aug_value[-1][other_aug_idx] if len(other_aug_idx) else None
                #     self.aug_done[-1] = self.aug_done[-1][other_aug_idx] if len(other_aug_idx) else None
                #     self.aug_reward[-1] = self.aug_reward[-1][other_aug_idx] if len(other_aug_idx) else None
                # Reuse other data
                if _reuse_times > 1 and (not ('MasspointPush' in self.env.get_attr('spec')[0].id) or ('MasspointPush' in self.env.get_attr('spec')[0].id and update > 150)):
                    self.aug_obs = self.aug_obs[-_reuse_times+1:] + [None]
                    self.aug_act = self.aug_act[-_reuse_times+1:] + [None]
                    self.aug_neglogp = self.aug_neglogp[-_reuse_times+1:] + [None]
                    self.aug_return = self.aug_return[-_reuse_times+1:] + [None]
                    self.aug_value = self.aug_value[-_reuse_times+1:] + [None]
                    self.aug_done = self.aug_done[-_reuse_times+1:] + [None]
                    self.aug_reward = self.aug_reward[-_reuse_times+1:] + [None]
                    self.is_selfaug = self.is_selfaug[-_reuse_times+1:] + [None]
                else:
                    self.aug_obs = [None]
                    self.aug_act = [None]
                    self.aug_neglogp = [None]
                    self.aug_return = [None]
                    self.aug_value = [None]
                    self.aug_done = [None]
                    self.aug_reward = [None]
                    self.is_selfaug = [None]
                # true_reward is the reward without discount
                temp_time0 = time.time()
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()
                is_demo = np.zeros(obs.shape[0])
                temp_time1 = time.time()
                print('runner.run() takes', temp_time1 - temp_time0)
                original_episodes = np.sum(masks)
                original_success = np.sum([info['r'] for info in ep_infos])
                total_success = original_success

                # augment_steps = 0 if self.aug_obs is None else self.aug_obs.shape[0]
                augment_steps = sum([item.shape[0] if item is not None else 0 for item in self.aug_obs])
                print([item.shape[0] if item is not None else 0 for item in self.aug_obs])
                # if self.aug_obs is not None:
                if augment_steps > 0:
                    _aug_return = np.concatenate(list(filter(lambda v:v is not None, self.aug_return)), axis=0)
                    _aug_value = np.concatenate(list(filter(lambda v: v is not None, self.aug_value)), axis=0)
                    adv_clip_frac = np.sum((_aug_return - _aug_value) < (np.mean(returns - values) + self.aug_clip * np.std(returns - values))) / _aug_return.shape[0]
                    print('demo adv below average + %f std' % self.aug_clip, adv_clip_frac)
                    obs = np.concatenate([obs, *(list(filter(lambda v:v is not None, self.aug_obs)))], axis=0)
                    returns = np.concatenate([returns, *(list(filter(lambda v:v is not None, self.aug_return)))], axis=0)
                    masks = np.concatenate([masks, *(list(filter(lambda v:v is not None, self.aug_done)))], axis=0)
                    actions = np.concatenate([actions, *(list(filter(lambda v:v is not None, self.aug_act)))], axis=0)
                    values = np.concatenate([values, *(list(filter(lambda v:v is not None, self.aug_value)))], axis=0)
                    neglogpacs = np.concatenate([neglogpacs, *(list(filter(lambda v:v is not None, self.aug_neglogp)))], axis=0)
                    is_demo = np.concatenate([is_demo, np.ones(augment_steps)], axis=0)
                    _aug_reward = np.concatenate(list(filter(lambda v:v is not None, self.aug_reward)), axis=0)
                    total_success += np.sum(_aug_reward)
                    print('augmented data length', obs.shape[0])
                self.num_timesteps += self.n_batch
                ep_info_buf.extend(ep_infos)
                total_episodes = np.sum(masks)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                    inds = np.arange(self.n_batch + augment_steps)
                    # print('length self.aug_obs', len(self.aug_obs), batch_size)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch + augment_steps, batch_size):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_batch + epoch_num *
                                                                            self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            # _recompute_inds = np.where(is_demo[mbinds] > 0.5)[0]
                            # if _recompute_inds.shape[0] > 0:
                            #     neglogpacs[_recompute_inds] = self.sess.run(
                            #         self.aug_neglogpac_op, {self.train_aug_model.obs_ph: obs[_recompute_inds],
                            #                                 self.aug_action_ph: actions[_recompute_inds]})
                            #     if self.self_imitate:
                            #         neglogpacs[_recompute_inds] = np.minimum(neglogpacs[_recompute_inds], 100)
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs, is_demo))
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, writer=writer,
                                                                 update=timestep, cliprange_vf=cliprange_vf_now,
                                                                 ))
                else:  # recurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_envs + epoch_num *
                                                                            self.n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, update=timestep,
                                                                 writer=writer, states=mb_states,
                                                                 cliprange_vf=cliprange_vf_now))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if writer is not None:
                    self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                      true_reward.reshape((self.n_envs, self.n_steps)),
                                                                      masks.reshape((self.n_envs, self.n_steps)),
                                                                      writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("original_timesteps", self.num_timesteps)
                    logger.logkv("total_timesteps", self.num_timesteps + self.num_aug_steps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.logkv("augment_steps", augment_steps)
                    # logger.logkv("original_success", original_success)
                    # logger.logkv("total_success", total_success)
                    logger.logkv("self_aug_ratio", np.mean(runner.self_aug_ratio))
                    logger.dumpkvs()

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

            return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "cliprange_vf": self.cliprange_vf,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


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
