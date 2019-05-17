import gym
import gym_microrts
import numpy as np
env = gym.make("Microrts-v0")

observation = env.reset()
# for _ in range(1000):
#   # env.render()
#   action = env.action_space.sample() # your agent here (this takes random actions)
#   observation, reward, done, info = env.step(action)

#   if done:
#     observation = env.reset()
# env.close()
#num_classes = 7
#new_obs = np.zeros((4, 16*16, num_classes))
#reshaped_obs = observation.reshape((4,16*16))
#reshaped_obs[2] += 1
#reshaped_obs[3] += 1
#reshaped_obs[reshaped_obs >= num_classes] = num_classes - 1
#for i in range(len(reshaped_obs)):
#    new_obs[i][np.arange(len(reshaped_obs[i])), reshaped_obs[i]] = 1