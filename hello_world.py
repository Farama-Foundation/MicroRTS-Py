import gym
import gym_microrts
import time
from gym.wrappers import TimeLimit, Monitor

try:
    env = gym.make("MicrortsGlobalAgentRandomEnemy10x10FrameSkip9-v0")
    env = Monitor(env, f'videos')
    env.action_space.seed(0)
    env.reset()
    for i in range(10000):
        time.sleep(0.2)
        obs = env.step(env.action_space.sample())
        next_obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            print("done")
            break
    #env.close()
except Exception as e:
    print(e)
    print(e.stacktrace())