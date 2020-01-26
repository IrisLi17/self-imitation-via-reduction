import tensorflow as tf
import numpy as np
import warnings
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import linear


def attention_mlp_extractor(flat_observations, n_object=2, n_units=128):
    # policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    # value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    agent_idx = np.concatenate([np.arange(3), np.arange(3+6*n_object, 3+6*n_object+2),
                                np.arange(3+6*n_object+2+9*n_object, 3+6*n_object+2+9*n_object+6+2*(3+n_object))])
    self_in = tf.gather(flat_observations, agent_idx, axis=1)
    # self_in = np.concatenate([flat_observations[:, :3], flat_observations[:, 3 + 6 * n_object:3 + 6 * n_object + 2],
    #                           flat_observations[:, 3+6*n_object+2+9*n_object:]], axis=1)
    self_out = self_in
    # Maybe nonlinear and more layers
    for i in range(2):
        self_out = tf.nn.relu(linear(self_out, "shared_agent_fc{}".format(i), n_units, init_scale=np.sqrt(2))) # (*, n_units)

    objects_in = []
    for i in range(n_object):
        _object_idx = np.concatenate([np.arange(3+3*i, 3+3*(i+1)), np.arange(3+3*n_object+3*i, 3+3*n_object+3*(i+1)),
                                      np.arange(3+6*n_object+2+3*i, 3+6*n_object+2+3*(i+1)),
                                      np.arange(3+9*n_object+2+3*i, 3+9*n_object+2+3*(i+1)),
                                      np.arange(3+12*n_object+2+3*i, 3+12*n_object+2+3*(i+1))])
        object_in = tf.gather(flat_observations, _object_idx, axis=1)
        assert self_in.shape[1] + n_object * object_in.shape[1] == flat_observations.shape[1], (self_out.shape, object_in.shape)
        # object_in = np.concatenate([flat_observations[:, 3 + 3 * i:3+3*(i+1)], flat_observations[:, 3+3*n_object+3*i:3+3*n_object+3*(i+1)],
        #                                   flat_observations[:, 3+6*n_object+2+3*i:3+6*n_object+2+3*(i+1)],
        #                                   flat_observations[:, 3+9*n_object+2+3*i:3+9*n_object+2+3*(i+1)],
        #                                   flat_observations[:, 3+12*n_object+2+3*i:3+12*n_object+2+3*(i+1)]], axis=1)
        fc1 = tf.nn.relu(linear(object_in, "object{}_fc{}".format(i, 0), n_units, init_scale=np.sqrt(2)))
        fc2 = tf.nn.relu(linear(fc1, "object{}_fc{}".format(i, 1), n_units, init_scale=np.sqrt(2)))
        objects_in.append(fc2)
    objects_in = tf.stack(objects_in, 2) # (*, n_unit, n_object)
    objects_attention = tf.nn.softmax(tf.matmul(tf.expand_dims(self_out, axis=1), objects_in)) # (*, 1, n_object)
    objects_out = tf.squeeze(tf.matmul(objects_attention, tf.transpose(objects_in, [0, 2, 1])), 1) # (*, n_unit)
    objects_out = tf.contrib.layers.layer_norm(objects_out)
    objects_out = tf.nn.relu(objects_out)

    latent = tf.concat([self_out, objects_out], 1) # (*, 2*n_unit)

    # Old code
    # # Iterate through the shared layers and build the shared parts of the network
    # for idx, layer in enumerate(net_arch):
    #     if isinstance(layer, int):  # Check that this is a shared layer
    #         layer_size = layer
    #         latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
    #     else:
    #         assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
    #         if 'pi' in layer:
    #             assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
    #             policy_only_layers = layer['pi']
    #
    #         if 'vf' in layer:
    #             assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
    #             value_only_layers = layer['vf']
    #         break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    # for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
    #     if pi_layer_size is not None:
    #         assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
    #         latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))
    #
    #     if vf_layer_size is not None:
    #         assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
    #         latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

    return latent_policy, latent_value


class AttentionPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, feature_extraction="attention_mlp", **kwargs):
        super(AttentionPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                              scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            assert feature_extraction == 'attention_mlp'
            pi_latent, vf_latent = attention_mlp_extractor(tf.layers.flatten(self.processed_obs), n_object=2)
            # if feature_extraction == "cnn":
            #     pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            # else:
            #     pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
