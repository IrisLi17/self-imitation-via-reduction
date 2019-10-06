# import mujoco_py
import matplotlib.pyplot as plt
import numpy as np
from push_obstacle import FetchPushEnv 

# def reset_mocap_welds(sim):
#     """Resets the mocap welds that we use for actuation.
#     """
#     if sim.model.nmocap > 0 and sim.model.eq_data is not None:
#         for i in range(sim.model.eq_data.shape[0]):
#             if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
#                 sim.model.eq_data[i, :] = np.array(
#                     [0., 0., 0., 1., 0., 0., 0.])
#     sim.forward()

# path = "assets/fetch/push_obstacle.xml"

# initial_qpos = {
#         'robot0:slide0': 0.405,
#         'robot0:slide1': 0.48,
#         'robot0:slide2': 0.0,
#         'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
#         'object1:joint': [1.3, 0.75, 0.45, 1., 0., 0., 0.],
# }

# model = mujoco_py.load_model_from_path(path)
# sim = mujoco_py.MjSim(model)
# # print(sim.data.get_joint_qpos('object0:joint'))
# for name, value in initial_qpos.items():
#     sim.data.set_joint_qpos(name, value)
# reset_mocap_welds(sim)
# sim.forward()

# gripper_target = np.array([-0.498, 0.005, -0.431 + 0.0]) + sim.data.get_site_xpos('robot0:grip')
# gripper_rotation = np.array([1., 0., 1., 0.])
# sim.data.set_mocap_pos('robot0:mocap', gripper_target)
# sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
# for _ in range(10):
#     sim.step()
# img = sim.render(1200, 1200)
# plt.imshow(img, origin='lower')
# plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
seed = 42
env = FetchPushEnv()
obs = env.reset()
print(obs["observation"].shape)
for t in range(50):
    u = np.random.uniform(env.action_space.low, env.action_space.high)
    next_obs, reward, done, info = env.step(u)
    print('t', t, 'u', u, 'object pos', next_obs["observation"][3:6], 
          'stick pos', next_obs["observation"][25:28], 'r', reward)
    ax.cla()
    ax.imshow(env.render(mode='rgb_array'))
    plt.pause(0.2)
    # plt.show()
# for _ in range(10):
#     env.reset()
#     img = env.render(mode='rgb_array')
#     plt.cla()
#     plt.imshow(img)
#     plt.show()