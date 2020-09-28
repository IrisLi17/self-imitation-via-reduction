import gym
from PIL import Image
import numpy as np
from gym.spaces import Box,Dict
from collections import OrderedDict
class ImageEnv(gym.Wrapper):

    """
    adding image observation and some manipulation of images
    """
    def __init__(self,env,imsize=84,transpose=False,camera_name=None,grayscale=False,normalize=False):
        super(ImageEnv,self).__init__(env)
        self.imsize=imsize
        self.camera_name = camera_name  # None means default camera
        self.transpose = transpose
        self.grayscale = grayscale
        self.normalize = normalize
        new_observation_space=OrderedDict()
        #for key,value in self.observation_space.items():
        #    if key=='observation':
        #        new_observation_space['state_observation']=value
        #    else :
        #        new_observation_space[key]=value
        #new_observation_space['image_observation']=Box(low=0.0,high=255.0,shape=(imsize,imsize,))
        new_observation_space['state_observation']=self.observation_space['observation']
        new_observation_space['image_observation']=Box(low=0,high=255,shape=(3,imsize,imsize,))
        new_observation_space['achieved_goal']=self.observation_space['achieved_goal']
        new_observation_space['desired_goal']=self.observation_space['desired_goal']
        self.observation_space=Dict(new_observation_space)
        if grayscale:
            self.image_length=self.imsize * self.imsize
        else:
            self.image_length=3 * self.imsize * self.imsize

    def step(self, action):
        state_obs,reward,done,info=super().step(action)
        img_observation=self._image_observation()
        full_obs={}
        full_obs['state_observation']=state_obs['observation']
        full_obs['desired_goal']=state_obs['desired_goal']
        full_obs['achieved_goal']=state_obs['achieved_goal']

        full_obs['image_observation']=img_observation
        return full_obs, reward, done, info
    
    def _render_callback(self):
        #visualize the target
        sites_offset=(self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        self.sim.forward()


    def _get_obs(self):
        # positions
        masspoint_pos = self.sim.data.get_site_xpos('masspoint')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # velocity
        masspoint_velp = self.sim.data.get_site_xvelp('masspoint') * dt
        achieved_goal = masspoint_pos.copy()
        obs = np.concatenate([
            masspoint_pos, masspoint_velp,
        ])
        img_obs=self.get_image()
        return {
            'state_observation': obs.copy(),
            'image_observation': img_obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def get_image(self):
        return self._image_observation()

    def sample_goal(self):
        if not hasattr(self, 'size_wall'):
            self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        if not hasattr(self, 'pos_wall0'):
            self.pos_wall0 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        if not hasattr(self, 'pos_wall1'):
            self.pos_wall1 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall1')]
        # g_idx = np.random.randint(self.n_object)
        # one_hot = np.zeros(self.n_object)
        # one_hot[g_idx] = 1
        goal = np.array([2.5, 2.5]) + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)

        def same_side(pos0, pos1, sep):
            if (pos0 - sep) * (pos1 - sep) > 0:
                return True
            return False

        goal = np.concatenate([goal, self.initial_masspoint_xpos[2:3]])
        return goal.copy()


    def _image_observation(self):
        image_obs=self.env.sim.render(width=self.imsize,height=self.imsize,camera_name=self.camera_name)
        if self.grayscale:
            image_obs=Image.fromarray(image_obs).convert('-L')
            image_obs=np.array(image_obs)
        if self.normalize:
            image_obs = image_obs / 255.0
        if self.transpose:
            image_obs = image_obs.transpose()
        return image_obs
