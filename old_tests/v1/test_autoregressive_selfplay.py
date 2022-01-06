import numpy as np
import gym
import gym_microrts
import torch
from gym.envs.registration import register
from gym_microrts import Config

env = gym.make('MicrortsSelfPlayShapedReward-v1').env
env.action_space.seed(0)
obss = []
try:
    obs = env.reset(True)
    obss += [obs]
    env.render()
except Exception as e:
    e.printStackTrace()

np.testing.assert_equal(
    [np.array(env.get_unit_action_mask(11, player=1))],
    np.array([[1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=np.int32))

np.testing.assert_equal(
    [np.array(env.get_unit_action_mask(11, player=2))],
    np.array([[1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=np.int32))

env.set_opponent_action([11, 1, 2, 0, 0, 0, 0, 0, 0])
print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
obs, r, d, info = env.step([11, 1, 2, 0, 0, 0, 0, 0, 0], True)
obss += [obs]
env.render()

np.testing.assert_equal(
    np.array(env.get_unit_action_mask(21, player=1)),
    np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      dtype=np.int32))

np.testing.assert_equal(
    np.array(env.get_unit_action_mask(21, player=2)),
    np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      dtype=np.int32))

assert (env.get_unit_location_mask(player=1) == env.get_unit_location_mask(player=2)).all()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([12, 0, 0, 0, 0, 0, 0, 0, 0], True)
# obss += [obs]
# env.render()

# np.testing.assert_equal(
#     np.array(env.client.getUnitActionMasks([[10]])),
#     np.array([[1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
#         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
#       dtype=np.int32))

# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([10, 1, 2, 0, 0, 0, 0, 0, 0], True)
# obss += [obs]
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([20, 1, 2, 0, 0, 0, 0, 0, 0], True)
# obss += [obs]
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([30, 1, 2, 0, 0, 0, 0, 0, 0], True)
# obss += [obs]
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([40, 1, 2, 0, 0, 0, 0, 0, 0], True)
# obss += [obs]
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([50, 1, 2, 0, 0, 0, 0, 0, 0], True)
# obss += [obs]
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([60, 1, 2, 0, 0, 0, 0, 0, 0], True)
# obss += [obs]
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([70, 1, 2, 0, 0, 0, 0, 0, 0], True)
# obss += [obs]
# env.render()

# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([80, 1, 1, 0, 0, 0, 0, 0, 0], True)
# obss += [obs]
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([81, 1, 1, 0, 0, 0, 0, 0, 0], True)
# obss += [obs]
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([82, 1, 1, 0, 0, 0, 0, 0, 0], True)
# obss += [obs]
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([83, 1, 1, 0, 0, 0, 0, 0, 0], True)
# obss += [obs]
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([84, 1, 1, 0, 0, 0, 0, 0, 0], True)
# obss += [obs]
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# obs, r, d, info = env.step([85, 1, 1, 0, 0, 0, 0, 0, 0], True)
# obss += [obs]
# env.render()

# np.testing.assert_equal(
#     np.array(env.client.getUnitActionMasks([[86]])),
#     np.array([[1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
#         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
#       dtype=np.int32))
