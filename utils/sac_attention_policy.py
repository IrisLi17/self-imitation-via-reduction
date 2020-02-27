import tensorflow as tf
from stable_baselines.sac.sac import SACPolicy
from stable_baselines.sac.policies import mlp, gaussian_entropy, gaussian_likelihood, apply_squashing_func, LOG_STD_MAX, LOG_STD_MIN
from .attention_policy import attention_mlp_extractor


class AttentionPolicy(SACPolicy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param reg_weight: (float) Regularization loss weight for the policy parameters
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers=None,
                 cnn_extractor=None, feature_extraction="cnn", n_object=2, reg_weight=0.0,
                 layer_norm=False, act_fun=tf.nn.relu, **kwargs):
        super(AttentionPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                              reuse=reuse, scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.reuse = reuse
        if layers is None:
            layers = [256, 256]
        self.layers = layers
        self.reg_loss = None
        self.reg_weight = reg_weight
        self.entropy = None
        self.n_object = n_object

        assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

        self.activ_fn = act_fun

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            elif self.feature_extraction == "attention_mlp":
                pi_h = attention_mlp_extractor(tf.layers.flatten(obs), n_object=self.n_object, n_units=128)
            else:
                pi_h = tf.layers.flatten(obs)

            pi_h = mlp(pi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)

            self.act_mu = mu_ = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)
            # Important difference with SAC and other algo such as PPO:
            # the std depends on the state, so we cannot use stable_baselines.common.distribution
            log_std = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)

        # Regularize policy output (not used for now)
        # reg_loss = self.reg_weight * 0.5 * tf.reduce_mean(log_std ** 2)
        # reg_loss += self.reg_weight * 0.5 * tf.reduce_mean(mu ** 2)
        # self.reg_loss = reg_loss

        # OpenAI Variation to cap the standard deviation
        # activation = tf.tanh # for log_std
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        # Original Implementation
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.std = std = tf.exp(log_std)
        # Reparameterization trick
        pi_ = mu_ + tf.random_normal(tf.shape(mu_)) * std
        logp_pi = gaussian_likelihood(pi_, mu_, log_std)
        self.entropy = gaussian_entropy(log_std)
        # MISSING: reg params for log and mu
        # Apply squashing and account for it in the probabilty
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)
        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi

    def make_critics(self, obs=None, action=None, reuse=False, scope="values_fn",
                     create_vf=True, create_qf=True):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            elif self.feature_extraction == "attention_mlp":
                critics_h = attention_mlp_extractor(tf.layers.flatten(obs), n_object=self.n_object, n_units=128)
            else:
                critics_h = tf.layers.flatten(obs)

            if create_vf:
                # Value function
                with tf.variable_scope('vf', reuse=reuse):
                    vf_h = mlp(critics_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    value_fn = tf.layers.dense(vf_h, 1, name="vf")
                self.value_fn = value_fn

            if create_qf:
                # Concatenate preprocessed state and action
                qf_h = tf.concat([critics_h, action], axis=-1)

                # Double Q values to reduce overestimation
                with tf.variable_scope('qf1', reuse=reuse):
                    qf1_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf1 = tf.layers.dense(qf1_h, 1, name="qf1")

                with tf.variable_scope('qf2', reuse=reuse):
                    qf2_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf2 = tf.layers.dense(qf2_h, 1, name="qf2")

                self.qf1 = qf1
                self.qf2 = qf2

        return self.qf1, self.qf2, self.value_fn

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run(self.deterministic_policy, {self.obs_ph: obs})
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run([self.act_mu, self.std], {self.obs_ph: obs})
