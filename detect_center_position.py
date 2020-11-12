import argparse
import imutils
import cv2
import os.path as osp
import numpy as np
import multiworld
import gym
import numpy as np
import torch
import utils.torch.pytorch_util as ptu
import csv, pickle
import cv2
# from utils.wrapper import VAEWrappedEnv
def create_image_48_pointmass_uwall_train_env_big_v0():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTrainEnvBig-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )

env_id = 'Image48PointmassUWallTrainEnvBig-v0'

env = gym.register(
                    id=env_id,
                    entry_point=create_image_48_pointmass_uwall_train_env_big_v0,
                    tags={
                        'git-commit-hash': 'e5c11ac',
                        'author': 'Soroush'
                    },)

state_error=[]
pixel_error=[]
use_env =False # use image or use env to generate image
dir = '/home/yilin/sir_img'#path to replace
if not use_env:
    ## state info: 3.43230646 3.9368457(image0--pixel position is large) result: 2.80851064 3.31914894
    ## state info :-0.22103173 -0.74116061(image1--pixel position is small) result: -0.42553191 -0.93617021
    # path to replace
    image = cv2.imread('/home/yilin/local/sir/prob_image.png')
    # cv2.imwrite(image_file, image)
    # get the binary mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # the range of blue
    lower_range = np.array([110, 50, 50])
    upper_range = np.array([130, 255, 255])
    # get the binary mask
    mask = cv2.inRange(hsv, lower_range, upper_range)
    # mask_file = osp.join(dir,'mask%d.png'%i)
    # cv2.imwrite(mask_file,mask)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(cnts)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    image = np.zeros((48, 48, 3))
    areas = np.asarray([cv2.contourArea(c) for c in cnts])
    max_contours = cnts[np.argmax(areas)]
    # compute the center of the contour
    M = cv2.moments(max_contours)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    print('position x,y', cX, cY)
    pos = np.array([cX, cY]) / 47 * 9 - 4.5
    print('image_to_env_pos', pos)
    # draw the contour and center of the shape on the image
    # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 2, (255, 255, 255), -1)
    filename = osp.join(dir,'mark_image%d.png')
    cv2.imwrite(filename,image)
else:
    env = gym.make(env_id)
    env.reset()
    for i in range(100):

        obs=env.get_env_state()
        state = obs['observation']
        image = env.get_image()
        env.step(env.action_space.sample()*2)
        print('state_info',state)
        state_pixel = (state+4.5)/9 * 47
        print('state_pixel_info',state_pixel)
        # switch the channel for cv2 use
        copy = image[:,:,0].copy()
        image[:,:,0]=image[:,:,2]
        image[:,:,2]= copy
        image_file = osp.join(dir,'image_env%d.png'%i)
        cv2.imwrite(image_file,image)
        # get the binary mask
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # the range of blue
        lower_range = np.array([110,50,50])
        upper_range = np.array([130,255,255])
        # get the binary mask
        mask = cv2.inRange(hsv, lower_range, upper_range)
        # mask_file = osp.join(dir,'mask%d.png'%i)
        # cv2.imwrite(mask_file,mask)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        print(cnts)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        image = np.zeros((48, 48, 3))
        areas = np.asarray([cv2.contourArea(c) for c in cnts])
        max_contours = cnts[np.argmax(areas)]
        # compute the center of the contour
        M = cv2.moments(max_contours)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print('position x,y',cX,cY)
        pos = np.array([cX,cY])/47*9-4.5
        print('image_to_env_pos',pos)
        # draw the contour and center of the shape on the image
        # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 2, (255, 255, 255), -1)
        # filename = osp.join(dir,'mark_image%d.png'%i)
        # cv2.imwrite(filename,image)
        print('calculation_error',pos-state)
        state_error.append(np.max(np.abs(pos-state)))
        pixel_error.append(np.max(np.abs(state_pixel-np.array([cX,cY]))))
    print('state_error',max(state_error),' ',min(state_error),' ',sum(state_error)/len(state_error))
    print('pixel_error', max(pixel_error), ' ', min(pixel_error), ' ', sum(pixel_error) / len(pixel_error))


