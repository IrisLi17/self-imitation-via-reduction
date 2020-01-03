from masspoint_env import MasspointPushDoubleObstacleEnv, MasspointPushSingleObstacleEnv_v2
import matplotlib.pyplot as plt
import numpy as np


# env = MasspointPushDoubleObstacleEnv(random_ratio=1.0, random_pusher=False)
env = MasspointPushSingleObstacleEnv_v2(random_ratio=1.0, random_pusher=True)
obs = env.reset()
print(obs)
for i in range(100):
    action = obs['observation'][3:5] - obs['observation'][0:2]
    action = action / np.linalg.norm(action)
    # action = np.random.uniform(-1, 1, size=2)
    obs, _, _, _ = env.step(action)
    print(i, obs['observation'][0:3], action)
    img = env.render(mode='rgb_array')
    plt.imshow(img)
    plt.pause(0.1)
plt.show()
