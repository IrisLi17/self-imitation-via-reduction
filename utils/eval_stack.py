import numpy as np


# Deprecated
def pp_eval_model(eval_env, model):
    env = eval_env
    env.unwrapped.random_ratio = 1.0
    temp = env.unwrapped.task_array.copy()
    env.unwrapped.task_array = [(env.n_object, i) for i in range(env.n_object)]
    n_episode = 0
    ep_rewards = []
    ep_successes = []
    while n_episode < 50:
        ep_reward = 0.0
        ep_success = 0.0
        obs = env.reset()
        while env.current_nobject != env.n_object or env.task_mode != 0:
            obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_success += info['is_success']
        ep_rewards.append(ep_reward)
        ep_successes.append(ep_success)
        n_episode += 1
    # return np.mean(ep_rewards)
    env.unwrapped.task_array = temp
    return np.mean(ep_successes)


def eval_model(env, model, max_nobject, random_ratio, init_on_table=False):
    # random_ratio 0: stack only, 1: pick and place only
    temp = env.unwrapped.task_array.copy()
    if init_on_table:
        env.unwrapped.task_array = [(max_nobject, i) for i in range(min(2, max_nobject))]
    else:
        env.unwrapped.task_array = [(max_nobject, i) for i in range(max_nobject)]
    env.unwrapped.random_ratio = random_ratio
    n_episode = 0
    ep_successes = []
    while n_episode < 50:
        ep_reward = 0.0
        ep_success = 0.0
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_success += info['is_success']
        ep_successes.append(ep_success)
        n_episode += 1
    env.unwrapped.task_array = temp
    return np.mean(ep_successes)