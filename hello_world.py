import gym
import gym_microrts
import time
try:
    env = gym.make("MicrortsGlobalAgentRandomEnemy10x10FrameSkip9-v0")
    env.action_space.seed(0)
    env.reset()
    for i in range(10000):
        env.render()
        time.sleep(0.2)
        obs = env.step(env.action_space.sample())
        if env.step(env.action_space.sample())[2]:
            print("done")
            break
    #env.close()
except Exception as e:
    print(e)
    print(e.stacktrace())