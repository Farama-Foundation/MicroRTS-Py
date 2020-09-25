import numpy as np
import gym
import gym_microrts
from gym.envs.registration import register
from gym_microrts import Config


class NoAvailableActionThenSkipEnv(gym.Wrapper):
    """if no source unit can be selected in microrts,
    automatically execute a NOOP action
    """
    def step(self, action):
        obs, reward, done, info = self.env.step(action, True)
        while self.unit_location_mask.sum()==0:
            obs, reward, done, info = self.env.step(action, True)
            if done:
                break
        return obs, reward, done, info

env = NoAvailableActionThenSkipEnv(gym.make('MicrortsSelfPlayShapedReward-v1').env)
env.action_space.seed(0)
obss = []
try:
    obs = env.env.reset(True)
    obss += [obs]
    env.render()
except Exception as e:
    e.printStackTrace()

print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
obs, r, d, info = env.step([2, 1, 1, 0, 0, 0, 0, 0])
obss += [obs]
env.render()
print(obs)
print(env.opponent_raw_obs)
print(env.action_mask)
print(env.opponent_action_mask)

# env.close()