import gym
import gym_microrts
from gym.envs.registration import register
from gym_microrts import Config
import os
import numpy as np

import jpype
from jpype.imports import registerDomain
import jpype.imports
from jpype.types import *

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

if "ParamOpEnv-v0" not in gym.envs.registry.env_specs:
    register(
        "ParamOpEnv-v0",
        entry_point='gym_microrts.envs:ParamOpEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/base4x5.xml",
            # below are dev properties
            microrts_path="~/Documents/work/go/src/github.com/vwxyzjn/microrts",
        )},
        max_episode_steps=1
    )

env = gym.make("ParamOpEnv-v0")
env.action_space.seed(0)
try:
    obs = env.reset()
except Exception as e:
    e.printStackTrace()

# env.client.computeRandomAIWinrate(JArray(JDouble)(softmax(env.action_space.sample())), 0, 3, os.path.expanduser(env.config.microrts_path))

#-------------------------------------------------------------------------------
# DEBUGGING actions
#-------------------------------------------------------------------------------

# mine left
# env.step([4 ,2, 0, 3, 0, 0, 0, 0, 0], True)
# env.render()

# # move right
# # observation, reward, done, info = env.step([1, 0, 1, 1], True)
# env.step([17, 1, 1, 0, 0, 0, 0, 0, 0], True)
# env.render()

# # move left
# env.step([2, 0, 1, 3, 0, 0, 0, 0, 0, 0], True)
# env.render()

# # attack right bottom
# env.step([3, 2, 5, 0, 0, 0, 0, 0, 3, 3], True)
# env.render()

# # make barracks
# env.step([0, 1, 4, 0, 0, 0, 2, 2, 0, 0], True)
# env.render()

# # make light
# env.step([0, 2, 4, 0, 0, 0, 2, 5, 0, 0], True)
# env.render()



# for second worker mine top
# observation, reward, done, info = env.step([0, 1, 2, 0], True)
