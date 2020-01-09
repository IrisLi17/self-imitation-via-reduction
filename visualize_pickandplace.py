from pick_and_place_box import FetchPickAndPlaceBoxEnv
import matplotlib.pyplot as plt
import numpy as np

env = FetchPickAndPlaceBoxEnv(random_ratio=1.0)
obs = env.reset()
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for _ in range(10):
    ax.cla()
    ax.imshow(env.render(mode='rgb_array'))
    plt.pause(0.1)
    action = np.random.randn(env.action_space.shape[0])
    obs, _, _, _ = env.step(action)
print(obs)
print('gripper', obs['observation'][:3], 'object0', obs['observation'][3:6], 'handle', obs['observation'][6:9])
# img = env.render(mode='rgb_array')
# plt.imshow(img)
# plt.show()

