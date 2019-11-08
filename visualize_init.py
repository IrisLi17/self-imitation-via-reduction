# import mujoco_py
import matplotlib.pyplot as plt
import numpy as np
from push_wall_obstacle import FetchPushWallObstacleEnv
import gym

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
seed = 42
env = FetchPushWallObstacleEnv(random_box=False, heavy_obstacle=True)
# env = gym.make('FetchSlide-v1')
obs = env.reset()
stick_qpos = env.sim.data.get_joint_qpos('object1:joint')
stick_qpos[:2] = np.asarray([env.pos_wall[0] - env.size_wall[0] - env.size_obstacle[0], env.initial_gripper_xpos[1]])
env.sim.data.set_joint_qpos('object1:joint', stick_qpos)
env.sim.forward()

# img = env.render(mode='rgb_array')
# plt.imshow(img)
# plt.show()
obstacle_xpos = []
for t in range(30):
    u = np.random.uniform(env.action_space.low, env.action_space.high)
    # diff = obs["observation"][6:9] - obs["observation"][:3]
    # diff = diff / np.linalg.norm(diff)
    u[:3] = np.array([-1.0, 0.0, 0.0])
    next_obs, reward, done, info = env.step(u)
    print('t', t, 'u', u,
          # 'object pos', next_obs["observation"][6:9],
          # 'r', reward,
          'obstacle pos', next_obs["observation"][6:9])
    obstacle_xpos.append(next_obs["observation"][6])
    ax.cla()
    ax.imshow(env.render(mode='rgb_array'))
    plt.pause(0.05)
plt.figure()
plt.grid()
plt.plot(obstacle_xpos)
plt.show()