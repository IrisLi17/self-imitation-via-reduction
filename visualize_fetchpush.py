from push_wall_obstacle import FetchPushWallObstacleEnv_v4
import matplotlib.pyplot as plt

env = FetchPushWallObstacleEnv_v4(heavy_obstacle=True)
obs = env.reset()
img = env.render(mode='rgb_array')
plt.imshow(img)
plt.show()

