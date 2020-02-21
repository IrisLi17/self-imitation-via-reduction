import numpy as np


def pp_eval_model(eval_env, model):
    env = eval_env
    env.unwrapped.random_ratio = 1.0
    n_episode = 0
    ep_success = []
    while n_episode < 50:
        obs = env.reset()
        while env.current_nobject != env.n_object or env.task_mode != 0:
            obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
        is_success = float(info['is_success'])
        ep_success.append(is_success)
        n_episode += 1
    return np.mean(ep_success)