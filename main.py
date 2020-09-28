# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



import gym
from masspoint_env import MasspointSMazeEnv
import cv2
import imageio
from wrapper import ImageEnv
IM_SIZE=84
MAX_EPISODE=100
gym.register('MasspointMaze-v2',entry_point=MasspointSMazeEnv,max_episode_steps=100)

env=gym.make('MasspointMaze-v2',reward_type='sparse',random_pusher=True,random_ratio=1.0)
env=ImageEnv(env,imsize=IM_SIZE,camera_name="fixed")
obs=env.reset()
img = env.get_image()
episode_reward=0.0
images=[]
frame_idx=0
for i in range(env.spec.max_episode_steps):
   action=env.action_space.sample()
   obs,reward,done,_=env.step(action)
   episode_reward+=reward
   frame_idx+=1
   img_obs=env.get_image()
   images.append(img_obs)
   if done:
      obs=env.reset()
      print('episode_reward',episode_reward)
      episode_reward=0.0
imageio.mimsave('smaz.gif',images)
cv2.imwrite('img1.png',img_obs)
