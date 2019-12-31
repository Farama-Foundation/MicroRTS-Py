import gym
import gym_microrts

env = gym.make("MicrortsGlobalAgentsProd-v0")
env.reset()
for _ in range(10000):
    env.render()
    env.step(env.action_space.sample())
env.close()