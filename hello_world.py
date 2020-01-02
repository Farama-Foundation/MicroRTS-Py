import gym
import gym_microrts

env = gym.make("MicrortsGlobalAgentsProd-v0")
env.action_space.seed(0)
env.reset()
for i in range(10000):
    env.render()
    obs = env.step(env.action_space.sample())
    if env.step(env.action_space.sample())[2]:
        print("done")
        break
env.close()