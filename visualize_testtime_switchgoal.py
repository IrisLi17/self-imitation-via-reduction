import sys, os, shutil, imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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
        env.goal[2] = obs['observation'][5]
    obs['desired_goal'] = env.goal
    obs['achieved_goal'][0:3] = obs['observation'][3:6]
    if len(env.goal) == 5:
        obs['achieved_goal'][3:] = np.array([1., 0])
    print(obs)
    return obs


def generate_trajectory(env, obs, model, free=False, greedy=True):
    batch_obs = []
    imgs = []
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for i in range(30):
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
        if not free:
            action /= np.max(np.abs(action))
            action = np.concatenate((action, [0.]))
        print(i, action)
        batch_obs.append(np.concatenate([obs[key] for key in KEY_ORDER]))
        obs, _, _, _ = env.step(action)
        img = env.render(mode='rgb_array')
        imgs.append(img)
        ax1.imshow(img)
        plot_value_obstaclepos(obs, model.model, ax2, fig)
        plt.savefig('temp' + str(i) + '.png')
        plt.pause(0.05)
    batch_obs = np.asarray(batch_obs)
    batch_imgs = np.asarray(imgs)
    imgs = []
    for i in range(50):
        try:
            imgs.append(plt.imread('temp' + str(i) + '.png'))
            os.remove('temp' + str(i) + '.png')
        except:
            pass
    # imageio.mimsave('contour_obstacle.gif', imgs, duration=0.5)
    return batch_obs, batch_imgs


# HACK goal
def hack_trajectory(env, obs, model, free=True, greedy=False):
    batch_obs = []
    imgs = []
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    flag = True
    box_goal = obs['desired_goal'].copy()
    print('box goal', box_goal)
    for i in range(100):
        if flag and np.argmax(obs['desired_goal'][3:]) == 0 and obs['observation'][3] < 1.37:
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
        ax1.imshow(img)
        ax1.set_title('goal ' + str(obs['desired_goal']))
        plot_value_obstaclepos(obs, model.model, ax2, fig)
        plt.savefig('temp' + str(i) + '.png')
        plt.pause(0.1)
    batch_obs = np.asarray(batch_obs)
    batch_imgs = np.asarray(imgs)
    imgs = []
    for i in range(100):
        try:
            imgs.append(plt.imread('temp' + str(i) + '.png'))
            os.remove('temp' + str(i) + '.png')
        except:
            pass
    imageio.mimsave('contour_obstacle.gif', imgs, duration=0.5)
    return batch_obs, batch_imgs


# See the value fn at different configuration
def plot_value_obstaclepos(obs, sac_model, ax, fig):
    ax.cla()
    if isinstance(obs, dict):
        obs = np.concatenate([obs[key] for key in KEY_ORDER])
    # Obstacle pos.
    obstacle_xpos, obstacle_ypos = np.meshgrid(np.linspace(1.0, 1.6, 21), np.linspace(0.4, 1.1, 21))
    grid_shape = obstacle_xpos.shape
    _obstacle_xpos = np.reshape(obstacle_xpos, (-1, 1))
    _obstacle_ypos = np.reshape(obstacle_ypos, (-1, 1))
    batch_obs = np.tile(obs, (_obstacle_xpos.shape[0], 1))
    batch_obs[:, 6] = _obstacle_xpos[:, 0]
    batch_obs[:, 7] = _obstacle_ypos[:, 0]
    batch_obs[:, 12] = batch_obs[:, 6] - batch_obs[:, 0]
    batch_obs[:, 13] = batch_obs[:, 7] - batch_obs[:, 1]

    # sac_model = model.model
    feed_dict = {
                sac_model.observations_ph: batch_obs,
    }

    values = sac_model.sess.run(sac_model.step_ops[6], feed_dict)
    grid_values = np.reshape(values, grid_shape)
    # surf = ax1.plot_surface(obstacle_xpos, obstacle_ypos, grid_values, cmap=cm.coolwarm)
    surf = ax.contour(obstacle_xpos, obstacle_ypos, grid_values, 20, cmap=cm.coolwarm)
    ax.clabel(surf, surf.levels, inline=True)
    ax.scatter(obs[6], obs[7])
    ax.set_xlim(1.6, 1.0)
    ax.set_ylim(0.4, 1.1)
    ax.set_xlabel('obstacle xpos')
    ax.set_ylabel('obstacle ypos')
    # ax1.set_zlim(0, 8)
    # ax1.set_zlabel('value fn')
    # ax1.view_init(elev=60, azim=135)

    # plt.pause(0.1)


# python visualize_ensemble.py FetchPushWallObstacle-v4 logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac/custom/model_70.zip 1 0
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python visualize_ensemble.py [env_name] [load_path] [free 0|1] [greedy 0|1]')
        exit()
    np.random.seed(42)
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
    if isinstance(sac_model.step_ops[-1], list):
        value_ensemble_op = sac_model.step_ops[-1]
    else:
        value_ensemble_op = None
    value_op = sac_model.step_ops[6]
    feed_dict = {
        sac_model.observations_ph: batch_obs
    }
    # values = sac_model.sess.run(value_ensemble_op, feed_dict)
    # values = np.squeeze(np.asarray(values), axis=-1)
    values = sac_model.sess.run(value_op, feed_dict)
    values = np.squeeze(values, axis=-1)
    print(values.shape)
    print(values)

    if np.linalg.norm(obs['achieved_goal'][0:3] - obs['desired_goal'][0:3]) < env.distance_threshold:
        exit()
    # Here we assume the policy gets stuck. Perturb the obstacle, select the most promising one and perform subgoal.
    imgs = []
    step_so_far = 30
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ultimate_goal = obs['desired_goal'].copy()
    noise = np.random.uniform(low=-0.10, high=0.10, size=(100, 2))
    noise = np.sign(noise) * 0.05 + noise
    obstacle_pos = np.expand_dims(obs['observation'][6:8], axis=0) + noise
    assert obstacle_pos.shape == (100, 2)
    perturb_obs = np.tile(np.concatenate([obs[key] for key in KEY_ORDER]), (obstacle_pos.shape[0], 1))
    perturb_obs[:, 6:8] = obstacle_pos
    perturb_obs[:, 12:14] = perturb_obs[:, 6:8] - perturb_obs[:, 0:2]
    feed_dict = {sac_model.observations_ph: perturb_obs}
    perturb_values = np.squeeze(sac_model.sess.run(value_op, feed_dict), axis=-1)
    print(perturb_values)

    subgoal_obs = np.tile(np.concatenate([obs[key] for key in KEY_ORDER]), (obstacle_pos.shape[0], 1))
    subgoal_obs[:, 40:43] = subgoal_obs[:, 6:9]
    subgoal_obs[:, 43:45] = np.array([[0, 1.]])
    subgoal_obs[:, 45:47] = obstacle_pos
    subgoal_obs[:, 47:48] = subgoal_obs[:, 42:43]
    subgoal_obs[:, 48:50] = np.array([[0, 1.]])
    feed_dict = {sac_model.observations_ph: subgoal_obs}
    subgoal_values = np.squeeze(sac_model.sess.run(value_op, feed_dict), axis=-1)
    print(subgoal_values)

    best_idx = np.argmax(((perturb_values - np.min(perturb_values)) / (np.max(perturb_values - np.min(perturb_values)))) ** 1
                                          * (subgoal_values - np.min(subgoal_values)) / (np.max(subgoal_values) - np.min(subgoal_values)))
    best_subgoal = obstacle_pos[best_idx]
    print('best subgoal is', best_subgoal, 'corresponding perturb_value', perturb_values[best_idx], 'subgoal_value', subgoal_values[best_idx])
    env.goal[3:] = np.array([0., 1.])
    env.goal[0:3] = obs['observation'][6:9]
    env.goal[0:2] = best_subgoal
    obs['desired_goal'] = env.goal
    obs['achieved_goal'][0:3] = obs['observation'][6:9]
    obs['achieved_goal'][3:] = np.array([0., 1.])
    print('goal is', env.goal)

    while (np.linalg.norm(obs['achieved_goal'][0:3] - obs['desired_goal'][0:3]) > env.distance_threshold):
        action, _ = model.predict(obs)
        print(step_so_far, action)
        obs, _, _, _ = env.step(action)
        step_so_far += 1
        img = env.render(mode='rgb_array')
        imgs.append(img)
        ax.imshow(img)
        plt.pause(0.1)
        if step_so_far >= 100:
            break
    env.goal[:] = ultimate_goal
    obs['desired_goal'] = ultimate_goal
    obs['achieved_goal'][0:3] = obs['observation'][3:6]
    obs['achieved_goal'][3:] = np.array([1., 0.])
    for i in range(70):
        action , _ = model.predict(obs)
        print(step_so_far, action)
        obs, _, _, _ = env.step(action)
        step_so_far += 1
        img = env.render(mode='rgb_array')
        imgs.append(img)
        ax.imshow(img)
        plt.pause(0.1)
        if step_so_far >= 100:
            break
    batch_imgs = np.concatenate([batch_imgs, np.array(imgs)], axis=0)
    imageio.mimsave('testtime_select_subgoal.gif', batch_imgs)
    exit()

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    os.makedirs('temp', exist_ok=True)
    for i in range(values.shape[1]):
        ax.cla()
        ax2.cla()
        for j in range(values.shape[0]):
            ax.plot(values[j, :i], alpha=0.5)
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
