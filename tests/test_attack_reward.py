import numpy as np
import gym
import gym_microrts
from gym.envs.registration import register
from gym_microrts import Config

if "GlobalAgentAttackEnv-v0" not in gym.envs.registry.env_specs:
    register(
        "GlobalAgentAttackEnv-v0",
        entry_point='gym_microrts.envs:GlobalAgentAttackEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/4x4/baseTwoWorkers4x4.xml",
            # below are dev properties
            microrts_path="~/Documents/work/go/src/github.com/vwxyzjn/microrts",
        )}
    )

env = gym.make("GlobalAgentAttackEnv-v0")
env.action_space.seed(0)
try:
    obs = env.reset(True)
    env.render()
except Exception as e:
    e.printStackTrace()


env.step([1, 1, 1, 0, 0, 0, 0, 0, 0], True)
env.render()
env.step([2, 1, 1, 0, 0, 0, 0, 0, 0], True)
env.render()
env.step([3, 1, 2, 0, 0, 0, 0, 0, 0], True)
env.render()
env.step([7, 1, 2, 0, 0, 0, 0, 0, 0], True)
env.render()

# attack correct location
assert env.step([11, 5, 0, 0, 0, 0, 0, 15], True)[1] > 0
env.render()

# attack wrong location no reward
assert env.step([11, 5, 0, 0, 0, 0, 0, 3], True)[1] == 0
env.render()

# attack wrong unit no reward
assert env.step([11, 5, 0, 0, 0, 0, 0, 5], True)[1] == 0
env.render()

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
