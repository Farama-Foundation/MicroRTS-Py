import numpy as np
import gym
import gym_microrts
from gym.envs.registration import register
from gym_microrts import Config

gym_id = "GlobalAgentHRLAttackCloserToEnemyBaseEnv"
if gym_id not in gym.envs.registry.env_specs:
    register(
        gym_id+'-v0',
        entry_point=f'gym_microrts.envs:{gym_id}',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            # below are dev properties
            microrts_path="~/Documents/work/go/src/github.com/vwxyzjn/microrts",
        )}
    )
env = gym.make(gym_id+'-v0')
print(gym_id)
env.action_space.seed(0)
try:
    obs = env.reset()
    env.render()
except Exception as e:
    e.printStackTrace()

print(np.where(env.unit_location_mask == 1))
print(np.where(env.target_unit_location_mask == 1))

env.step([11, 1, 2, 0, 0, 0, 0, 0])
env.render()
env.step([21, 1, 2, 0, 0, 0, 0, 0])
env.render()
env.step([31, 1, 2, 0, 0, 0, 0, 0])
env.render()
env.step([41, 1, 2, 0, 0, 0, 0, 0])
env.render()
env.step([51, 1, 2, 0, 0, 0, 0, 0])
env.render()
env.step([61, 1, 2, 0, 0, 0, 0, 0])
env.render()
env.step([71, 1, 2, 0, 0, 0, 0, 0])
env.render()
env.step([81, 1, 1, 0, 0, 0, 0, 0])
env.render()
env.step([82, 1, 1, 0, 0, 0, 0, 0])
env.render()
env.step([83, 1, 1, 0, 0, 0, 0, 0])
env.render()
env.step([84, 1, 1, 0, 0, 0, 0, 0])
env.render()
env.step([85, 1, 1, 0, 0, 0, 0, 0])
env.render()
env.step([86, 5, 0, 0, 0, 0, 0, 87])
env.render()

expert_replay = np.array(
    [11, 1, 2, 0, 0, 0, 0, 0],
    [21, 1, 2, 0, 0, 0, 0, 0],
    [31, 1, 2, 0, 0, 0, 0, 0],
    [41, 1, 2, 0, 0, 0, 0, 0],
    [51, 1, 2, 0, 0, 0, 0, 0],
    [61, 1, 2, 0, 0, 0, 0, 0],
    [71, 1, 2, 0, 0, 0, 0, 0],
    [81, 1, 1, 0, 0, 0, 0, 0],
    [82, 1, 1, 0, 0, 0, 0, 0],
    [83, 1, 1, 0, 0, 0, 0, 0],
    [84, 1, 1, 0, 0, 0, 0, 0],
    [85, 1, 1, 0, 0, 0, 0, 0],
    [86, 5, 0, 0, 0, 0, 0, 87],
)

# # attack correct location
# assert env.step([11, 5, 0, 0, 0, 0, 0, 15])[1] > 0
# env.render()

# # attack wrong location no reward
# assert env.step([11, 5, 0, 0, 0, 0, 0, 3])[1] == 0
# env.render()

# # attack wrong unit no reward
# assert env.step([11, 5, 0, 0, 0, 0, 0, 5])[1] == 0
# env.render()

# env.close()