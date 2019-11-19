from stable_baselines.sac.policies import FeedForwardPolicy, nature_cnn, mlp
from stable_baselines.common.policies import register_policy
import tensorflow as tf


class EnsembleFeedForwardPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn", reg_weight=0.0,
                 layer_norm=False, act_fun=tf.nn.relu, **kwargs):
        super(EnsembleFeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env=n_env, n_steps=n_steps, n_batch=n_batch, reuse=reuse, layers=layers,
                 cnn_extractor=cnn_extractor, feature_extraction=feature_extraction, reg_weight=reg_weight,
                 layer_norm=layer_norm, act_fun=act_fun, **kwargs)


    def make_critics(self, obs=None, action=None, n_values=1, reuse=False, scope="values_fn",
                     create_vf=True, create_qf=True):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                critics_h = tf.layers.flatten(obs)

            if create_vf:
                # Value function
                self.value_ensemble = []
                for i in range(n_values):
                    scope_name = 'vf' if i == 0 else 'vf_' + str(i)
                    with tf.variable_scope(scope_name, reuse=reuse):
                        vf_h = mlp(critics_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                        value_fn = tf.layers.dense(vf_h, 1, name="vf")
                    self.value_ensemble.append(value_fn)
                    if i == 0:
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

        return self.qf1, self.qf2, self.value_fn, self.value_ensemble


class EnsembleMlpPolicy(EnsembleFeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(EnsembleMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                feature_extraction="mlp", **_kwargs)


class EnsembleLnMlpPolicy(EnsembleFeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(EnsembleLnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                  feature_extraction='mlp', layer_norm=True, **_kwargs)


register_policy('EnsembleMlpPolicy', EnsembleMlpPolicy)
register_policy('EnsembleLnMlpPolicy', EnsembleLnMlpPolicy)
