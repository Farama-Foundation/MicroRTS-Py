import numpy as np
import gym
import gym_microrts
from gym.envs.registration import register
from gym_microrts import Config
from gym_microrts import microrts_ai

gym_id = "GlobalAgentCombinedRewardEnv"
if gym_id not in gym.envs.registry.env_specs:
    register(
        gym_id+'-v0',
        entry_point=f'gym_microrts.envs:{gym_id}',
        kwargs={'config': Config(
            frame_skip=0,
            ai2=microrts_ai.passiveAI,
            map_path="maps/4x4/base4x4.xml",
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0, 0.0])
        )},
    )
env = gym.make(gym_id+'-v0')
env.client.renderTheme = 2
print(gym_id)
env.action_space.seed(0)
try:
    obs = env.reset(True)
    env.render()
except Exception as e:
    e.printStackTrace()

assert env.step([5, 4, 0, 0, 0, 1, 3, 0], True)[1] > 0
env.render()

# env.close()