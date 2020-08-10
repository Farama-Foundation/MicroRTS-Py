import numpy as np
import gym
import gym_microrts
from gym.envs.registration import register
from gym_microrts import Config

env = gym.make('MicrortsTwoWorkersMining-v2').env
env.action_space.seed(0)
obss = []
try:
    obs = env.reset(True)
    obss += [obs]
    env.render()
except Exception as e:
    e.printStackTrace()

print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
obs, r, d, info = env.step([2, 1, 1, 0, 0, 0, 0, 0, 0], True)
obss += [obs]
env.render()
print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
obs, r, d, info = env.step([11, 1, 3, 0, 0, 0, 0, 0, 0], True)
obss += [obs]
env.render()
print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
obs, r, d, info = env.step([12, 0, 0, 0, 0, 0, 0, 0, 0], True)
obss += [obs]
env.render()
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