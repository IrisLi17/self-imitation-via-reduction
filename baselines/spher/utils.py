import  stable_baselines.her.utils as U 
import numpy as np

class HERMaskGoalEnvWrapper(U.HERGoalEnvWrapper):
    def __init__(self, env):
        super(HERMaskGoalEnvWrapper, self).__init__(env)
    
    def expand_mask(self, mask):
        return np.outer(mask, np.ones(self.goal_dim // self.env.n_object)).ravel()

    def mask_obs(self, obs, mask):
        dict_obs = self.convert_obs_to_dict(obs)
        dict_obs['achieved_goal'] = dict_obs['achieved_goal'] * self.expand_mask(mask)
        dict_obs['desired_goal'] = dict_obs['desired_goal'] * self.expand_mask(mask)
        return self.convert_dict_to_obs(dict_obs)
        