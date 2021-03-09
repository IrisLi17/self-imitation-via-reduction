import csv
import os

import numpy as np
from stable_baselines import logger


def eval_model(eval_env, model):
    env = eval_env
    if hasattr(env.unwrapped, 'random_ratio'):
        assert abs(env.unwrapped.random_ratio) < 1e-4
    n_episode = 0
    ep_rewards = []
    ep_successes = []
    while n_episode < 20:
        ep_reward = 0.0
        ep_success = 0.0
        obs = env.reset()
        goal_dim = env.goal.shape[0]
        if goal_dim > 3:
            while (np.argmax(obs[-goal_dim + 3:]) != 0):
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
    return np.mean(ep_successes)


def log_eval(num_update, mean_eval_reward, file_name='eval.csv'):
    if not os.path.exists(os.path.join(logger.get_dir(), file_name)):
        with open(os.path.join(logger.get_dir(), file_name), 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            title = ['n_updates', 'mean_eval_reward']
            csvwriter.writerow(title)
    with open(os.path.join(logger.get_dir(), file_name), 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
        data = [num_update, mean_eval_reward]
        csvwriter.writerow(data)


def stack_eval_model(eval_env, model, init_on_table=False):
    env = eval_env
    env.unwrapped.random_ratio = 0.0
    if init_on_table:
        env.unwrapped.task_array = [(env.n_object, i) for i in range(min(2, env.n_object))]
    else:
        env.unwrapped.task_array = [(env.n_object, i) for i in range(env.n_object)]
    assert abs(env.unwrapped.random_ratio) < 1e-4
    n_episode = 0
    ep_rewards = []
    ep_successes = []
    while n_episode < 20:
        ep_reward = 0.0
        ep_success = 0.0
        obs = env.reset()
        while env.current_nobject != env.n_object or (hasattr(env, 'task_mode') and env.task_mode != 1):
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
    return np.mean(ep_successes)


def egonav_eval_model(eval_env, model, random_ratio=0.0, goal_idx=3, fixed_goal=None):
    env = eval_env
    if hasattr(env.unwrapped, 'random_ratio'):
        env.unwrapped.random_ratio = random_ratio
    n_episode = 0
    ep_rewards = []
    ep_successes = []
    while n_episode < 20:
        ep_reward = 0.0
        ep_success = 0.0
        obs = env.reset()
        goal_dim = env.goal.shape[0]
        if fixed_goal is not None:
            env.unwrapped.goal = fixed_goal.copy()
            obs = env.get_obs()
            obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
        else:
            if goal_dim > 3:
                while np.argmax(obs[-goal_dim + 3:]) != goal_idx:
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
    return np.mean(ep_successes)
