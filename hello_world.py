import gym
import gym_microrts
import time
from gym.wrappers import Monitor

env = gym.make("MicrortsWorkerRush-v1")
# env = Monitor(env, f'videos', force=True)
env.action_space.seed(0)
env.reset()
for i in range(10000):
    next_obs, reward, done, info = env.step(env.action_space.sample())
    if done:
        print("done")
        break
env.close()