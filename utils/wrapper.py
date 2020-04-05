import gym


class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """
    def __init__(self, env, reward_offset=1.0):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or info.get('is_success', False)
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


class ScaleRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=1.0):
        super(ScaleRewardWrapper, self).__init__(env)
        self.reward_scale = reward_scale

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward /= self.reward_scale
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward / self.reward_scale


class FlexibleTimeLimitWrapper(gym.Wrapper):
    '''
    ONLY applicable to Stacking environment!
    We can set max_episode_steps = None for gym, (so gym.TimeLimitWrapper is not applied),
    then use this class to avoid potential conflict.
    '''
    def __init__(self, env, time_limit=None):
        super(FlexibleTimeLimitWrapper, self).__init__(env)
        self.time_limit = time_limit
        assert 'FetchStack' in env.spec.id
        assert env.spec.max_episode_steps is None
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        self.time_limit = self.env.unwrapped.current_nobject * 50 if self.env.unwrapped.current_nobject > 2 else 100
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.time_limit:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
