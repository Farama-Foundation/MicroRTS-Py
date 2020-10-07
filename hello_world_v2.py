import gym
import gym_microrts
import time
from gym.wrappers import TimeLimit, Monitor

try:
    env = gym.make("MicrortsDefeatCoacAIShaped-v2")
    env = Monitor(env, f'videos', force=True)
    env.action_space.seed(0)
    env.reset()
    for i in range(10000):
        # time.sleep(0.2)
        # obs = env.step(env.action_space.sample())
        action = env.action_space.sample()
        if len((env.unit_location_mask==1).nonzero()[0]) != 0:
            action[0] = (env.unit_location_mask==1).nonzero()[0][0]
        next_obs, reward, done, info = env.step(action)
        if done:
            print("done")
            break
    #env.close()
except Exception as e:
    print(e)
    print(e.stacktrace())