import gym
import gym_microrts

# The prefix `Eval` renders the gmae.
env = gym.make("EvalMicrortsGlobalAgentsProd-v0")
# Alternatively, try headless mode at
# env = gym.make("MicrortsGlobalAgentsProd-v0")
env.reset()
for _ in range(10000):
    env.step(env.action_space.sample())
env.close()