import gym
import gym_microrts
import time
import numpy as np
from gym.wrappers import Monitor
from gym_microrts import microrts_ai

env = gym.make(
    "MicrortsDefeatCoacAIShaped-v3",
    render_theme=2, # optional customization
    frame_skip=0, # optional customization
    ai2=microrts_ai.coacAI, # optional customization
    map_path="maps/16x16/basesWorkers16x16.xml", # optional customization
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0, 0.0]) # optional customization
)
# env = Monitor(env, f'videos', force=True)
env.action_space.seed(0)
env.reset()
for i in range(10000):
    env.render()
    time.sleep(0.001)
    action = env.action_space.sample()

    # optional: selecting only valid units.
    if len((env.unit_location_mask==1).nonzero()[0]) != 0:
        action[0] = (env.unit_location_mask==1).nonzero()[0][0]

    next_obs, reward, done, info = env.step(action)
    if done:
        env.reset()
env.close()