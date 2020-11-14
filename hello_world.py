import gym
import gym_microrts
import time
from gym.wrappers import Monitor
from gym_microrts import microrts_ai

env = gym.make(
    "MicrortsWorkerRush-v1",
    render_theme=2,
    frame_skip=9,
    ai2=microrts_ai.passiveAI,
    map_path="maps/10x10/basesTwoWorkers10x10.xml",
    reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
)

# env = Monitor(env, f'videos', force=True)
env.action_space.seed(0)
env.reset()
for i in range(10000):
    next_obs, reward, done, info = env.step(env.action_space.sample())
    if done:
        print("done")
        break
env.close()