import numpy as np


def pp_eval_model(eval_env, model):
    env = eval_env
    env.unwrapped.random_ratio = 1.0
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
    return np.mean(ep_successes)