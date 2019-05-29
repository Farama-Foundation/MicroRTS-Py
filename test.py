import gym
import gym_microrts
import numpy as np
env = gym.make("Microrts-v0")
env.init(4, 4)
observation = env.reset()

#-------------------------------------------------------------------------------
# DEBUGGING actions
#-------------------------------------------------------------------------------

# mine left
# observation, reward, done, info = env.step([1, 0, 2, 3])

# move right
# observation, reward, done, info = env.step([1, 0, 1, 3])

# move left
# observation, reward, done, info = env.step([2, 0, 1, 1])