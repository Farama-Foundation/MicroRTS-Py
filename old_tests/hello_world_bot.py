import numpy as np
import time
from numpy.random import choice
from stable_baselines3.common.vec_env import VecVideoRecorder

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
try:
    env = MicroRTSBotVecEnv(
        ai1s=[microrts_ai.workerRushAI for _ in range(1)],
        ai2s=[microrts_ai.coacAI for _ in range(1)],
        max_steps=2000,
        render_theme=2,
        map_paths=["maps/16x16/basesWorkers16x16.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    # env = VecVideoRecorder(env, 'videos', record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)


    env.action_space.seed(0)
    env.reset()
    for i in range(10000):
        env.render()
        time.sleep(0.01)
        next_obs, reward, done, info = env.step(
            [[[18, 0, 0, 0, 0, 0, 0, 0],
              [34, 0, 0, 0, 0, 0, 0, 0],
              [49, 0, 3, 0, 0, 0, 0, 0]]])
        if done:
            print(reward)
    env.close()
except Exception as e:
    e.printStackTrace()