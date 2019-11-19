import sys, os, shutil, imageio
import numpy as np
import matplotlib.pyplot as plt
from test_ensemble import make_env
from stable_baselines import HER
from baselines import HER_HACK
from stable_baselines.her.utils import KEY_ORDER


def reset_goal(env):
    obs = env.reset()
    env.goal[0] = 1.2
    env.goal[1] = obs['observation'][4]
    # env.goal[1] = 0.75
    if len(env.goal) == 5:
        env.goal[3:] = np.array([1., 0])
    obs['desired_goal'] = env.goal
    obs['achieved_goal'][0:3] = obs['observation'][3:6]
    if len(env.goal) == 5:
        obs['achieved_goal'][3:] = np.array([1., 0])
    print(obs)
    return obs


def generate_trajectory(env, obs, model, free=False, greedy=True):
    batch_obs = []
    imgs = []
    for i in range(50):
        if free:
            action, _ = model.predict(obs)
        elif not greedy:
            if i <= 6:
                action = obs['observation'][6:9] - np.asarray([0, 0.2, 0]) - obs['observation'][0:3]
            elif i <= 20:
                action = np.asarray([0.0, 1.0, 0.0])
            elif obs['observation'][0] < obs['achieved_goal'][0] + 0.1 and i <= 27:
                action = obs['observation'][3:6] + np.asarray([0.1, 0.0, 0.1]) - obs['observation'][0:3]
            elif i <= 32:
                action = obs['observation'][3:6] + np.asarray([0.15, 0, -0.05]) - obs['observation'][0:3]
            elif obs['observation'][3] - obs['desired_goal'][0] > 0.05:
                action = np.asarray([-1.0, 0.0, 0.0])
            else:
                action = np.asarray([1., 0, 1.])
        else:
            if obs['observation'][0] < obs['achieved_goal'][0] and i <= 4:
                action = obs['achieved_goal'][:3] + np.array([0.0, 0, 0.1]) - obs['observation'][0:3]
            elif i <= 9:
                action = obs['achieved_goal'][:3] + np.array([0.15, 0, -0.05]) - obs['observation'][0:3]
            else:
                action = np.asarray([-1.0, 0.0, 0.0])
        # print(i, action)
        if not free:
            action /= np.max(np.abs(action))
            action = np.concatenate((action, [0.]))
        batch_obs.append(np.concatenate([obs[key] for key in KEY_ORDER]))
        obs, _, _, _ = env.step(action)
        img = env.render(mode='rgb_array')
        imgs.append(img)
        plt.imshow(img)
        plt.pause(0.1)
    batch_obs = np.asarray(batch_obs)
    batch_imgs = np.asarray(imgs)
    return batch_obs, batch_imgs


# HACK goal
def hack_trajectory(env, obs, model, free=True, greedy=False):
    batch_obs = []
    imgs = []
    flag = True
    box_goal = obs['desired_goal'].copy()
    print('box goal', box_goal)
    for i in range(50):
        if flag and np.argmax(obs['desired_goal'][3:]) == 0:
            # Switch the target to move obstacle
            print(str(i), 'switch target to obstacle')
            env.goal[3:] = np.array([0, 1.])
            env.goal[0:3] = obs['observation'][6:9]
            env.goal[1] -= 0.12
            obs['desired_goal'] = env.goal.copy()
            obs['achieved_goal'][0:3] = obs['observation'][6:9]
            obs['achieved_goal'][3:] = np.array([0, 1.])
            flag = False
        elif np.argmax(obs['desired_goal'][3:]) == 1 and np.linalg.norm(obs['observation'][6:9] - obs['desired_goal'][0:3]) < env.distance_threshold:
            # Switch the target to move box
            print(str(i), 'switch target to box', box_goal)
            env.goal[:] = box_goal
            obs['desired_goal'] = box_goal
            obs['achieved_goal'][0:3] = obs['observation'][3:6]
            obs['achieved_goal'][3:] = np.array([1., 0])
        action, _ = model.predict(obs)
        batch_obs.append(np.concatenate([obs[key] for key in KEY_ORDER]))
        obs, _, _, _ = env.step(action)
        img = env.render(mode='rgb_array')
        imgs.append(img)
        plt.imshow(img)
        plt.title('goal ' + str(obs['desired_goal']))
        plt.pause(0.1)
    batch_obs = np.asarray(batch_obs)
    batch_imgs = np.asarray(imgs)
    return batch_obs, batch_imgs


# python visualize_ensemble.py FetchPushWallObstacle-v4 logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac/custom/model_70.zip 1 0
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python visualize_ensemble.py [env_name] [load_path] [free 0|1] [greedy 0|1]')
        exit()
    env_name = sys.argv[1]
    her_class = HER_HACK if env_name == 'FetchPushWallObstacle-v4' else HER
    load_path = sys.argv[2]
    free = int(sys.argv[3]) == 1
    greedy = int(sys.argv[4]) == 1
    env_kwargs = dict(random_box=False,
                      heavy_obstacle=True,
                      random_ratio=1.0,
                      )
    env = make_env(env_name, **env_kwargs)
    model = her_class.load(load_path, env=env)
    obs = reset_goal(env)
    print(env.goal, obs['achieved_goal'], obs['desired_goal'])
    batch_obs, batch_imgs = generate_trajectory(env, obs, model, free=free, greedy=greedy)
    # HACK goal
    # batch_obs, batch_imgs = hack_trajectory(env, obs, model, free=free, greedy=greedy)
    # imageio.mimsave('free_universe_hackgoal.gif', batch_imgs)
    # exit()
    sac_model = model.model
    value_ensemble_op = sac_model.step_ops[-1]
    feed_dict = {
        sac_model.observations_ph: batch_obs
    }
    values = sac_model.sess.run(value_ensemble_op, feed_dict)
    values = np.squeeze(np.asarray(values), axis=-1)
    print(values.shape)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    os.makedirs('temp', exist_ok=True)
    for i in range(values.shape[1]):
        ax.cla()
        ax2.cla()
        for j in range(values.shape[0]):
            ax.plot(values[j, :i], 'tab:blue', alpha=0.2)
        ax.set_xlim(0, values.shape[1] - 1)
        ax.set_ylim(np.min(values), np.max(values))
        ax.set_xlabel('steps')
        ax.set_ylabel('value fn')
        ax2.imshow(batch_imgs[i])
        plt.savefig(os.path.join('temp', str(i) + '.png'))
        plt.pause(0.1)
    gif_imgs = []
    for i in range(values.shape[1]):
        img = plt.imread(os.path.join('temp', str(i) + '.png'))
        gif_imgs.append(img)
    gif_name = 'greedy_ensemble.gif' if greedy else 'hand_tune_ensemble.gif'
    imageio.mimsave(gif_name, gif_imgs)
    shutil.rmtree('temp')
