from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.bit_flipping_env import BitFlippingEnv
from push_obstacle import FetchPushEnv
import gym
import matplotlib.pyplot as plt
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise
from stable_baselines.bench import Monitor
import os

play = 0
env_name = 'FetchReach-v1'
model_class = DDPG  # works also with SAC, DDPG and TD3

# N_BITS = 16
# env = BitFlippingEnv(N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)
# gym.register('MyFetchPush-v1', entry_point=FetchPushEnv, max_episode_steps=50)
# env = gym.make('MyFetchPush-v1')
# env = gym.make('FetchPush-v1')
if env_name in ['FetchReach-v1', 'FetchPush-v1']:
    env = gym.make(env_name)
elif env_name is 'MyFetchPush-v1':
    gym.register('MyFetchPush-v1', entry_point=FetchPushEnv, max_episode_steps=50)
    env = gym.make('MyFetchPush-v1')

if not play:
    log_dir = os.path.join("./logs", env_name, "her")
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(env, log_dir, allow_early_resets=True)
# Available strategies (cf paper): future, final, episode, random
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

if not play:
    # policy_kwargs = dict(layers=[64, 64, 64])
    if model_class is DDPG:
        train_kwargs = dict(action_noise=NormalActionNoise(mean=0.0 * env.action_space.high, sigma=0.05 * env.action_space.high),
                            normalize_observations=True,
                            random_exploration=0.2,
                            nb_rollout_steps=16*50, nb_train_steps=40,
                            buffer_size=int(1e6),
                            actor_lr=1e-3, critic_lr=1e-3,
                            gamma=0.98,
                            batch_size=128,
                            )
    else:
        train_kwargs = {}
    # Wrap the model
    model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                # policy_kwargs=policy_kwargs, 
                verbose=1,
                **train_kwargs)
    # Train the model
    model.learn(int(2e5))

    model.save(os.path.join("./model", "her_" + env_name))

# WARNING: you must pass an env
# or wrap your environment with HERGoalEnvWrapper to use the predict method
model = HER.load(os.path.join("./model", "her_" + env_name), env=env)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
obs = env.reset()
episode_reward = 0.0
for _ in range(500):
    action, _ = model.predict(obs)
    print('action', action)
    obs, reward, done, _ = env.step(action)
    episode_reward += reward
    ax.cla()
    ax.imshow(env.render(mode='rgb_array'))
    plt.pause(0.2)
    if done:
        obs = env.reset()
        print('episode_reward', episode_reward)
        episode_reward = 0.0