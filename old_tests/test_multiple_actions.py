import numpy as np
import gym
import gym_microrts
from gym.envs.registration import register
from gym_microrts import Config

gym_id = "MicrortsCombinedReward10x10F5MultiActionsBuildCombatUnits-v0"
env = gym.make(gym_id)
print(gym_id)
env.action_space.seed(0)
try:
    obs = env.reset()
    env.render()
except Exception as e:
    e.printStackTrace()

assert env.step([[11, 1, 0, 0, 0, 0, 0, 0]])[1] == 0
env.render()
assert env.step([[11, 1, 0, 0, 0, 0, 0, 0]])[1] == 0
env.render()
assert env.step([[11, 1, 0, 0, 0, 0, 0, 0]])[1] == 0
env.render()
assert env.step([[1, 2, 0, 3, 0, 0, 0, 0]])[1] >= 0
env.render()
env.close()

