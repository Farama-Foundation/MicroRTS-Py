import numpy as np
import gym
import gym_microrts
from gym.envs.registration import register
from gym_microrts import Config

if "GlobalAgentProduceWorkerEnv-v0" not in gym.envs.registry.env_specs:
    register(
        "GlobalAgentProduceWorkerEnv-v0",
        entry_point='gym_microrts.envs:GlobalAgentProduceWorkerEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/4x4/baseTwoWorkers4x4.xml",
            # below are dev properties
            microrts_path="~/Documents/work/go/src/github.com/vwxyzjn/microrts",
        )}
    )

env = gym.make("GlobalAgentProduceWorkerEnv-v0")
env.action_space.seed(0)
try:
    obs = env.reset(True)
    env.render()
except Exception as e:
    e.printStackTrace()


assert env.step([5, 4, 0, 0, 0, 1, 3, 0], True)[1] > 0
env.render()
