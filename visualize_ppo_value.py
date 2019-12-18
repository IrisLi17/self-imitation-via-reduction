import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from run_ppo import make_env
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2


def gen_value_with_obstacle(obs, model):
    obstacle_xpos, obstacle_ypos = np.meshgrid(np.linspace(1.0, 1.6, 21), np.linspace(0.4, 1.1, 21))
    grid_shape = obstacle_xpos.shape
    _obstacle_xpos = np.reshape(obstacle_xpos, (-1, 1))
    _obstacle_ypos = np.reshape(obstacle_ypos, (-1, 1))
    batch_obs = np.tile(obs, (_obstacle_xpos.shape[0], 1))
    batch_obs[:, 6] = _obstacle_xpos[:, 0]
    batch_obs[:, 7] = _obstacle_ypos[:, 0]
    batch_obs[:, 12] = batch_obs[:, 6] - batch_obs[:, 0]
    batch_obs[:, 13] = batch_obs[:, 7] - batch_obs[:, 1]
    batch_value = model.value(batch_obs)
    grid_value = np.reshape(batch_value, grid_shape)
    return obstacle_xpos, obstacle_ypos, grid_value


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python visualize_ppo_value.py [load_path]')
    load_path = sys.argv[1]
    env_name = 'FetchPushWallObstacle-v4'
    env_kwargs = dict(random_box=True,
                      heavy_obstacle=True,
                      random_ratio=0.0,
                      random_gripper=True, )
    n_cpu = 1
    env = make_env(env_id=env_name, seed=None, rank=0, log_dir=None, kwargs=env_kwargs)
    model = PPO2.load(load_path)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    obs = env.reset()
    while np.argmax(obs[-2:]) != 0:
        obs = env.reset()
    for step in range(100):
        img = env.render(mode='rgb_array')
        xs, ys, zs = gen_value_with_obstacle(obs, model)
        ax[0].cla()
        ax[1].cla()
        ax[0].imshow(img)
        surf = ax[1].contour(xs, ys, zs, 20, cmap=cm.coolwarm)
        ax[1].clabel(surf, surf.levels, inline=True)
        ax[1].scatter(obs[6], obs[7], c='tab:brown')
        ax[1].set_xlim(1.0, 1.6)
        ax[1].set_ylim(0.4, 1.1)
        ax[1].set_xlabel('obstacle x')
        ax[1].set_ylabel('obstacle y')
        ax[1].set_title('step %d' % step)
        plt.savefig(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % step))
        # plt.pause(0.1)


        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            break
    model_idx = int(os.path.basename(load_path).strip('.zip').split('_')[1])
    os.system(('ffmpeg -r 2 -start_number 0 -i ' + os.path.dirname(load_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
              os.path.join(os.path.dirname(load_path), 'value_obstacle_model_%d.mp4' % model_idx)))
    for step in range(100):
        try:
            os.remove(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % step))
        except:
            pass
