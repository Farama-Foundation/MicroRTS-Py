import numpy as np
import gym
import gym_microrts
from gym.envs.registration import register
from gym_microrts import Config

try:
    env = gym.make('MicrortsMining-v2').env
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(0)
    obs = env.reset()
    env.render()
except Exception as e:
    e.printStackTrace()

print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])


# print("reward is", env.step([2, 1, 3, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# print("reward is", env.step([11, 1, 3, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# print("reward is", env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])


# for i in range(9):
#     env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])
#     env.render()

# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# # # harvest
# print("reward is", env.step([1, 2, 0, 3, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# print("reward is", env.step([10, 2, 0, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# print("reward is", env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])

# # # creating worker
# # print("reward is", env.step([12, 4, 0, 0, 0, 1, 3, 0, 0])[1])
# # env.render()
# # print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])

# for i in range(19):
#     env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])
#     env.render()
#     # print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])


# # # move back
# print("reward is", env.step([1, 1, 1, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# print("reward is", env.step([10, 1, 1, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# print("reward is", env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])

# for i in range(9):
#     env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])
#     env.render()

# # # return
# print("reward is", env.step([2, 3, 0, 0, 2, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# print("reward is", env.step([11, 3, 0, 0, 1, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])

# for i in range(29):
#     env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])
#     env.render()