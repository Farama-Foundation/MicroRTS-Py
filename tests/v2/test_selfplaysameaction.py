import numpy as np
import gym
import gym_microrts
from gym.envs.registration import register
from gym_microrts import Config
import json


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
action_reverse_idxs = np.arange((100)-1,-1,-1).reshape(10, 10).flatten()
action_direction_reverse_idxs = np.array([2, 3, 0, 1])
response = env.client.step(np.array([[11,  1,  2,  2,  1,  0,  6, 88]]), 
                           np.array([[action_reverse_idxs[11],  1,  action_direction_reverse_idxs[2],  2,  1,  0,  6, action_reverse_idxs[88]]]), 9)
obs, r, d, info = np.array(response.observation), response.reward[:], response.done[:], json.loads(str(response.info))
obss += [obs]
env.render()
print(obs)
print(env.opponent_raw_obs)
print(env.action_mask)
print(env.opponent_action_mask)

# env.close()