import sys
sys.path.append('/home/yunfei/projects/metaworld')
# from metaworld.envs.mujoco.sawyer_xyz import SawyerBoxCloseSparseEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerBoxOpen6DOFEnv
import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
env = SawyerBoxOpen6DOFEnv()
obs = env.reset()
for i in range(200):
    print('obs', obs)
    img = env.render(mode='rgb_array')
    ax.cla()
    ax.imshow(img)
    plt.pause(0.1)
    if i < 20:
        action = obs[3:6] - obs[0:3]
        action[0] -= 0.08
        action = action / np.linalg.norm(action)
        # action = np.concatenate([action, np.random.uniform(-1.0, 1.0, 1)])
        action = np.concatenate([action, [0]])
        
    elif i < 50:
        action = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        action = np.array([0.0, 0.0, -1.0, 1.0])
    # action = np.concatenate([action, np.random.uniform(-1.0, 1.0, 1)])
    print('action', action)
    obs, rew, done, info = env.step(action)
