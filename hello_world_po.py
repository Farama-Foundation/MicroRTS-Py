import gym
import gym_microrts
import time
import numpy as np
from gym.wrappers import Monitor
from gym_microrts import microrts_ai
from gym_microrts.envs.povec_env import POMicroRTSGridModeVecEnv
import gym_microrts
import os, sys

env = POMicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.randomAI for _ in range(1)],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)

env.action_space.seed(0)
env.reset()
for i in range(10000):
    env.render()
    action_mask = np.array(env.vec_client.getMasks(0)).flatten()
    time.sleep(0.001)
    action = [env.action_space.sample() for _ in range(1)]
    
    # optional: selecting only valid units.
    if len(action_mask.nonzero()[0]) != 0:
        action[:][0] = action_mask.nonzero()[0][0]

    next_obs, reward, done, info = env.step([action])
env.close()