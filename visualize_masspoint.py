from masspoint_env import MasspointPushDoubleObstacleEnv, MasspointPushSingleObstacleEnv
import matplotlib.pyplot as plt
import numpy as np


# env = MasspointPushDoubleObstacleEnv(random_ratio=1.0, random_pusher=False)
env = MasspointPushSingleObstacleEnv(random_ratio=0.0, random_pusher=True)
obs = env.reset()
print(obs)
for i in range(100):
    action = obs['observation'][3:5] - obs['observation'][0:2]
    action = action / np.linalg.norm(action)
    obs, _, _, _ = env.step(action)
    print(i, obs['observation'][0:3])
    img = env.render(mode='rgb_array')
    plt.imshow(img)
    plt.pause(0.1)
plt.show()
