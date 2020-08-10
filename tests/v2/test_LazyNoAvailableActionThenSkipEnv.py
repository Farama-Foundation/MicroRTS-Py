import numpy as np
import gym
import gym_microrts
from gym.envs.registration import register
from gym_microrts import Config


class LazyNoAvailableActionThenSkipEnv(gym.Wrapper):
    """if no source unit can be selected in microrts,
    automatically execute a NOOP action
    
    """
    def reset(self, **kwargs):
        observation = super(LazyNoAvailableActionThenSkipEnv, self).reset(**kwargs)
        self.first_busy = False
        return observation
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action, True)
        while self.unit_location_mask.sum()==0:
            obs, reward, done, info = self.env.step(action, True)
            if not self.first_busy:
                self.first_busy = True
                break
            if done:
                break
        return obs, reward, done, info

env = LazyNoAvailableActionThenSkipEnv(gym.make('MicrortsTwoWorkersMining-v2').env)
env.action_space.seed(0)
obss = []
try:
    env.reset()
    obs = env.env.reset(True)
    obss += [obs]
    env.render()
except Exception as e:
    e.printStackTrace()

print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
obs, r, d, info = env.step([2, 1, 1, 0, 0, 0, 0, 0])
obss += [obs]
env.render()
print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
obs, r, d, info = env.step([11, 1, 3, 0, 0, 0, 0, 0])
obss += [obs]
env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])
# obss += [obs]
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([12, 4, 0, 0, 0, 1, 3, 0])
# obss += [obs]
# env.render()

# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([3, 1, 3, 0, 0, 0, 0, 0])
# obss += [obs]
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([10, 1, 1, 0, 0, 0, 0, 0])
# obss += [obs]
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# env.step([2, 1, 1, 0, 0, 0, 0, 0, 0], True)
# env.render()
# env.step([3, 1, 2, 0, 0, 0, 0, 0, 0], True)
# env.render()
# env.step([7, 1, 2, 0, 0, 0, 0, 0, 0], True)
# env.render()

# for _ in range(9):
#     assert env.step([11, 5, 0, 0, 0, 0, 0, 15], True)[1] == 0
# # attack correct location
# assert env.step([11, 5, 0, 0, 0, 0, 0, 15], True)[1] > 0
# env.render()

# env.close()