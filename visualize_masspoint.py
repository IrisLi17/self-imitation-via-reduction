from masspoint_env import MasspointPushDoubleObstacleEnv, MasspointPushSingleObstacleEnv_v2
import matplotlib.pyplot as plt
import numpy as np


env = MasspointPushDoubleObstacleEnv(random_ratio=0.0, random_pusher=True)
# env = MasspointPushSingleObstacleEnv_v2(random_ratio=0.0, random_pusher=True)
print(env.sim.model.actuator_biastype)
obs = env.reset()
idx = env.sim.model.jnt_qposadr[env.sim.model.actuator_trnid[0, 0]]
print(env.sim.data.qpos[idx])
# env.reset()
state = env.sim.get_state()
masspoint_jointx_i = env.sim.model.get_joint_qpos_addr('masspoint:slidex')
masspoint_jointy_i = env.sim.model.get_joint_qpos_addr('masspoint:slidey')
box_jointx_i = env.sim.model.get_joint_qpos_addr('object0:slidex')
box_jointy_i = env.sim.model.get_joint_qpos_addr('object0:slidey')
state.qpos[masspoint_jointx_i] = obs['observation'][6]
state.qpos[masspoint_jointy_i] = obs['observation'][7] - 1.0
state.qpos[box_jointx_i] = 0.5
state.qpos[box_jointy_i] = 2.0
env.sim.set_state(state)
env.sim.forward()

print('size obstacle', env.size_obstacle)
print('qpos', env.sim.data.get_joint_qpos('object0:rz'), env.sim.data.get_joint_qpos('object1:rz'))
print('obstacle xmat', env.sim.data.get_site_xpos('object1'))
print(obs)

for i in range(200):
    # action = obs['observation'][6:8] - obs['observation'][0:2]
    # action = action / np.linalg.norm(action)
    # action = np.random.choice([-1, 1], size=2)
    action = np.asarray([0.0, 1.0])
    if i > 20:
        action = -action
    obs, _, _, _ = env.step(action)
    print(i, obs['observation'][0:3], action)
    img = env.render(mode='rgb_array')
    plt.imshow(img)
    plt.pause(0.1)
plt.show()
