import gym
import gym_microrts
import time
from gym.wrappers import Monitor

env = gym.make("MicrortsDefeatCoacAIShaped-v3")
# env = Monitor(env, f'videos', force=True)
env.action_space.seed(0)
env.reset()
for i in range(10000):
    action = env.action_space.sample()
    if len((env.unit_location_mask==1).nonzero()[0]) != 0:
        action[0] = (env.unit_location_mask==1).nonzero()[0][0]
    next_obs, reward, done, info = env.step(action)
    if done:
        print("done")
        break
env.close()