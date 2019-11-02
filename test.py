import gym
import gym_microrts
from gym.envs.registration import register
from gym_microrts import Config


if "MicrortsGlobalAgentsDev-v0" not in gym.envs.registry.env_specs:
    register(
        "MicrortsGlobalAgentsDev-v0",
        entry_point='gym_microrts.envs:RandomAgentEnv',
        kwargs={'config': Config(
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/4x4/baseTwoWorkers4x4.xml",
            # below are dev properties
            render=True,
            client_port=9898,
            microrts_repo_path="/home/costa/Documents/work/go/src/github.com/vwxyzjn/microrts"
        )}
    )
    register(
        "MicrortsLocalAgentsDev-v0",
        entry_point='gym_microrts.envs:LocalAgentEnv',
        kwargs={'config': Config(
            frame_skip=10,
            ai1_type="no-penalty-individual",
            ai2_type="passive",
            map_path="maps/4x4/baseTwoWorkers4x4.xml",
            # below are dev properties
            render=True,
            client_port=9898,
            microrts_repo_path="/home/costa/Documents/work/go/src/github.com/vwxyzjn/microrts"
        )}
    )

env = gym.make("MicrortsGlobalAgentsDev-v0")
observation = env.reset()

#-------------------------------------------------------------------------------
# DEBUGGING actions
#-------------------------------------------------------------------------------

# mine left
# observation, reward, done, info = env.step([1, 0, 2, 3])

# move right
# observation, reward, done, info = env.step([1, 0, 1, 3])

# move left
# observation, reward, done, info = env.step([2, 0, 1, 1])

# for second worker mine top
# observation, reward, done, info = env.step([0, 1, 2, 0])
