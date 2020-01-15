from pick_and_place_box import FetchPickAndPlaceBoxEnv
from open_close_box import FetchOpenCloseBoxEnv
import matplotlib.pyplot as plt
import numpy as np

# env = FetchPickAndPlaceBoxEnv(random_ratio=1.0)
env = FetchOpenCloseBoxEnv(random_ratio=1.0)
obs = env.reset()
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
for i in range(100):
    ax.cla()
    ax.imshow(env.render(mode='rgb_array'))
    plt.pause(0.1)
    action = np.random.randn(env.action_space.shape[0])
    # action[:3] = obs['observation'][3:6] - obs['observation'][0:3]
    # action[:3] = action[:3] / np.linalg.norm(action[:3])
    obs, _, _, _ = env.step(action)
    # print(i, obs['observation'][:9], obs['achieved_goal'], obs['desired_goal'])
    print(i, obs['observation'][:6], obs['achieved_goal'], obs['desired_goal'])
print(obs)
# print('gripper', obs['observation'][:3], 'object0', obs['observation'][3:6], 'handle', obs['observation'][6:9])
print('gripper', obs['observation'][:3], 'handle', obs['observation'][3:6])
# img = env.render(mode='rgb_array')
# plt.imshow(img)
plt.show()

