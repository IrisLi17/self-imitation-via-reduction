import tensorflow as tf
import numpy as np
import warnings
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import linear
from stable_baselines.common.policies import mlp_extractor
from itertools import zip_longest
import math


def attention_mlp_extractor(flat_observations, n_object=2, n_units=128):
    # policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    # value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # agent_idx = np.concatenate([np.arange(3), np.arange(3+6*n_object, 3+6*n_object+2),
    #                             np.arange(3+6*n_object+2+9*n_object, 3+6*n_object+2+9*n_object+7+2*(3+n_object))])
    agent_idx = np.concatenate([np.arange(3), np.arange(3 + 6 * n_object, 3 + 6 * n_object + 2),
                                np.arange(3 + 6 * n_object + 2 + 9 * n_object, int(flat_observations.shape[1]))])
    self_in = tf.gather(flat_observations, agent_idx, axis=1)
    # self_in = np.concatenate([flat_observations[:, :3], flat_observations[:, 3 + 6 * n_object:3 + 6 * n_object + 2],
    #                           flat_observations[:, 3+6*n_object+2+9*n_object:]], axis=1)
    self_out = self_in
    # Maybe nonlinear and more layers
    # for i in range(2):
    #     self_out = tf.nn.relu(linear(self_out, "shared_agent_fc{}".format(i), n_units, init_scale=np.sqrt(2))) # (*, n_units)
    self_out = tf.contrib.layers.fully_connected(
        self_out, num_outputs=n_units, scope='shared_agent_fc0', activation_fn=tf.nn.relu)
    self_out = tf.contrib.layers.fully_connected(
        self_out, num_outputs=n_units // 2, scope='shared_agent_fc1', activation_fn=tf.nn.relu)
    # self_out = tf.contrib.layers.fully_connected(
    #     self_out_latent, num_outputs=n_units // 2, scope="shared_agent_fc2", activation_fn=tf.nn.relu)

    objects_in = []
    for i in range(n_object):
        _object_idx = np.concatenate([np.arange(3+3*i, 3+3*(i+1)), np.arange(3+3*n_object+3*i, 3+3*n_object+3*(i+1)),
                                      np.arange(3+6*n_object+2+3*i, 3+6*n_object+2+3*(i+1)),
                                      np.arange(3+9*n_object+2+3*i, 3+9*n_object+2+3*(i+1)),
                                      np.arange(3+12*n_object+2+3*i, 3+12*n_object+2+3*(i+1))])
        object_in = tf.gather(flat_observations, _object_idx, axis=1)
        assert self_in.shape[1] + n_object * object_in.shape[1] == flat_observations.shape[1], (self_out.shape, object_in.shape)
        with tf.variable_scope("object", reuse=tf.AUTO_REUSE):
            # fc1 = tf.nn.relu(linear(object_in, "fc0", n_units, init_scale=np.sqrt(2)))
            # fc2 = tf.nn.relu(linear(fc1, "fc1", n_units, init_scale=np.sqrt(2)))
            fc1 = tf.contrib.layers.fully_connected(object_in, num_outputs=n_units, scope="fc0", activation_fn=tf.nn.relu)
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=n_units // 2, scope="fc1", activation_fn=tf.nn.relu)
            objects_in.append(fc2)
    objects_in = tf.stack(objects_in, 2) # (*, n_unit, n_object)
    objects_attention = tf.nn.softmax(tf.matmul(tf.expand_dims(self_out, axis=1), objects_in) / math.sqrt(n_units // 2)) # (*, 1, n_object)
    objects_out = tf.squeeze(tf.matmul(objects_attention, tf.transpose(objects_in, [0, 2, 1])), 1) # (*, n_unit // 2)
    objects_out = tf.contrib.layers.layer_norm(objects_out)
    objects_out = tf.nn.relu(objects_out)

    latent = tf.concat([self_out, objects_out], 1) # (*, n_unit)
    # latent = tf.concat([self_out_latent, objects_out], 1)
    return latent


def attention_mlp_extractor2(flat_observations, n_object=2, n_units=128):
    # agent_idx = np.concatenate([np.arange(3), np.arange(3 + 6 * n_object, 3 + 6 * n_object + 2),
    #                             np.arange(3 + 6 * n_object + 2 + 9 * n_object, int(flat_observations.shape[1]))])
    agent_idx = np.concatenate([np.arange(3), np.arange(3 + 6 * n_object, 5 + 6 * n_object),
                                np.arange(5 + 15 * n_object, 12 + 15 * n_object),
                                np.arange(12 + 15 * n_object, 15 + 15 * n_object), # achieved goal pos
                                np.arange(15 + 16 * n_object, 18 + 16 * n_object), # desired goal pos
                                ]) # size 18
    self_in = tf.gather(flat_observations, agent_idx, axis=1)
    self_out = self_in
    self_out = tf.contrib.layers.fully_connected(
        self_out, num_outputs=n_units, scope='shared_agent_fc0', activation_fn=tf.nn.relu)
    self_out = tf.contrib.layers.fully_connected(
        self_out, num_outputs=n_units // 2, scope='shared_agent_fc1', activation_fn=tf.nn.relu)

    objects_in = []
    for i in range(n_object):
        _object_idx = np.concatenate([np.arange(3+3*i, 3+3*(i+1)), np.arange(3+3*n_object+3*i, 3+3*n_object+3*(i+1)),
                                      np.arange(5+6*n_object+3*i, 5+6*n_object+3*(i+1)),
                                      np.arange(5+9*n_object+3*i, 5+9*n_object+3*(i+1)),
                                      np.arange(5+12*n_object+3*i, 5+12*n_object+3*(i+1)),
                                      np.arange(15 + 15 * n_object + i, 15 + 15 * n_object + i + 1), # indicator
                                      ]) # size 16
        object_in = tf.gather(flat_observations, _object_idx, axis=1)
        # object_onehot = tf.tile(tf.expand_dims(tf.one_hot(i, n_object), dim=0), tf.stack([tf.shape(object_in)[0], 1]))
        # object_in = tf.concat([object_in, object_onehot], axis=1)
        with tf.variable_scope("object", reuse=tf.AUTO_REUSE):
            fc1 = tf.contrib.layers.fully_connected(object_in, num_outputs=n_units, scope="fc0", activation_fn=tf.nn.relu)
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=n_units // 2, scope="fc1", activation_fn=tf.nn.relu)
            objects_in.append(fc2)
    objects_in = tf.stack(objects_in, 2) # (*, n_unit, n_object)
    objects_attention = tf.nn.softmax(tf.matmul(tf.expand_dims(self_out, axis=1), objects_in) / math.sqrt(n_units // 2)) # (*, 1, n_object)
    objects_out = tf.squeeze(tf.matmul(objects_attention, tf.transpose(objects_in, [0, 2, 1])), 1) # (*, n_unit)
    objects_out = tf.contrib.layers.layer_norm(objects_out)
    objects_out = tf.nn.relu(objects_out)

    latent = tf.concat([self_out, objects_out], 1) # (*, 2*n_unit)
    return latent


def self_attention_mlp_extractor(flat_observations, n_object=3, n_units=256):
    agent_idx = np.concatenate([np.arange(3), np.arange(3 + 6 * n_object, 3 + 6 * n_object + 2),
                                np.arange(3 + 6 * n_object + 2 + 9 * n_object,
                                          3 + 6 * n_object + 2 + 9 * n_object + 7 + 2 * (3 + n_object))])
    self_in = tf.gather(flat_observations, agent_idx, axis=1)
    self_out = self_in
    # Maybe nonlinear and more layers
    for i in range(2):
        self_out = tf.nn.relu(
            linear(self_out, "shared_agent_fc{}".format(i), n_units, init_scale=np.sqrt(2)))  # (*, n_units)

    objects_in = []
    for i in range(n_object):
        _object_idx = np.concatenate(
            [np.arange(3 + 3 * i, 3 + 3 * (i + 1)), np.arange(3 + 3 * n_object + 3 * i, 3 + 3 * n_object + 3 * (i + 1)),
             np.arange(3 + 6 * n_object + 2 + 3 * i, 3 + 6 * n_object + 2 + 3 * (i + 1)),
             np.arange(3 + 9 * n_object + 2 + 3 * i, 3 + 9 * n_object + 2 + 3 * (i + 1)),
             np.arange(3 + 12 * n_object + 2 + 3 * i, 3 + 12 * n_object + 2 + 3 * (i + 1)),
             np.arange(3 + 15 * n_object + 2 + 7 + 3 + n_object, 3 + 15 * n_object + 2 + 7 + 2 * (3 + n_object))])
        object_in = tf.gather(flat_observations, _object_idx, axis=1)
        assert self_in.shape[1] + n_object * (object_in.shape[1] - (3 + n_object)) == flat_observations.shape[1], (
        self_out.shape, object_in.shape)
        with tf.variable_scope("object", reuse=tf.AUTO_REUSE):
            fc1 = tf.nn.relu(linear(object_in, "fc0", n_units, init_scale=np.sqrt(2)))
            fc2 = tf.nn.relu(linear(fc1, "fc1", n_units, init_scale=np.sqrt(2)))
            objects_in.append(fc2)
    # Do self-attention on objects
    objects_attention_latent = []
    for i in range(len(objects_in)):
        other_objects_in = tf.stack(objects_in[:i] + objects_in[i+1:], 2) # (*, n_unit, n_object-1)
        objects_self_attention = tf.nn.softmax(tf.matmul(tf.expand_dims(objects_in[i], axis=1), other_objects_in)) # (*, 1, n_object-1)
        objects_self_attention_out = tf.squeeze(tf.matmul(objects_self_attention, tf.transpose(other_objects_in, [0, 2, 1])), 1) # (*, n_unit)
        objects_inner_in = tf.concat([objects_in[i], objects_self_attention_out], 1)
        with tf.variable_scope("object_inner", reuse=tf.AUTO_REUSE):
            objects_inner_out = tf.nn.relu(linear(objects_inner_in, "fc0", n_units, init_scale=np.sqrt(2)))
            objects_attention_latent.append(objects_inner_out)
    objects_attention_in = tf.stack(objects_attention_latent, 2)  # (*, n_unit, n_object)
    objects_attention = tf.nn.softmax(tf.matmul(tf.expand_dims(self_out, axis=1), objects_attention_in))  # (*, 1, n_object)
    objects_out = tf.squeeze(tf.matmul(objects_attention, tf.transpose(objects_attention_in, [0, 2, 1])), 1)  # (*, n_unit)
    objects_out = tf.contrib.layers.layer_norm(objects_out)
    objects_out = tf.nn.relu(objects_out)

    latent = tf.concat([self_out, objects_out], 1)  # (*, 2*n_unit)

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent

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
                 act_fun=tf.tanh, feature_extraction="attention_mlp", n_object=2, **kwargs):
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
                layers = [256, 256]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            # assert feature_extraction == 'attention_mlp'
            if feature_extraction == 'attention_mlp':
                latent = attention_mlp_extractor2(tf.layers.flatten(self.processed_obs), n_object=n_object,
                                                 n_units=128)
                pi_latent, vf_latent = mlp_extractor(latent, net_arch, act_fun)
            elif feature_extraction == 'self_attention_mlp':
                pi_latent, vf_latent = self_attention_mlp_extractor(tf.layers.flatten(self.processed_obs), n_object=n_object)
            else:
                raise NotImplementedError
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
