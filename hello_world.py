import gym
import gym_microrts
import time
import numpy as np
from gym.wrappers import Monitor
from gym_microrts import microrts_ai

env = gym.make(
    "MicrortsWorkerRush-v1",
    render_theme=2, # optional customization
    frame_skip=2, # optional customization
    ai2=microrts_ai.workerRushAI, # optional customization
    map_path="maps/10x10/basesTwoWorkers10x10.xml", # optional customization
    reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # optional customization
)

# env = Monitor(env, f'videos', force=True)
env.action_space.seed(0)
env.reset()
for i in range(1000):
    env.render()
    time.sleep(0.03)
    next_obs, reward, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()
env.close()