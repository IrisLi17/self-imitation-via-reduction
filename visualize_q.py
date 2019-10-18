import gym
import matplotlib.pyplot as plt 
from test_her import ENTRY_POINT
from stable_baselines.her import HER


env_name = 'FetchPushWall-v1'
load_path = None
gym.register(env_name, entry_point=ENTRY_POINT[env_name], max_episode_steps=50)
env = gym.make(env_name)
model = HER.load(load_path, env=env)
# model.model._policy(observation, apply_noise=False, compute_q=True)
# Try to iterate over all possible obs and actions, after I come up with extremely low dimensional while still challenging task
feed_dict = {
    model.model.obs_train: obs,
    model.model.actions: actions,
    model.model.action_train_ph: actions,
}
q = model.model.sess.run(model.model.critic_tf, feed_dict=feed_dict)