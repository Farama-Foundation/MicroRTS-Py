import gym
import gym_microrts
from gym.envs.registration import register
from gym_microrts import Config


if "MicrortsLocalAgentsDev-v0" not in gym.envs.registry.env_specs:
    register(
        "MicrortsLocalAgentsDev-v0",
        entry_point='gym_microrts.envs:LocalAgentEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty-individual",
            ai2_type="passive",
            map_path="maps/base4x5.xml",
            # below are dev properties
            microrts_path="~/Documents/work/go/src/github.com/vwxyzjn/microrts",
        )}
    )

env = gym.make("MicrortsLocalAgentsDev-v0")
observation = env.reset(True)

#-------------------------------------------------------------------------------
# DEBUGGING actions
#-------------------------------------------------------------------------------

# mine left
# env.step([2, 0, 3, 0, 0, 0, 0, 0], True)
# env.render()

# # make barracks
# env.step([4, 0, 0, 0, 2, 2, 0, 0], True)
# env.render()


# # move right
# # observation, reward, done, info = env.step([1, 0, 1, 1], True)
# env.step([1, 1, 0, 0, 0, 0, 0, 0], True)
# env.render()

# move down
# env.step([1, 2, 0, 0, 0, 0, 0, 0], True)
# env.render()

# # move left
# env.step([2, 0, 1, 3, 0, 0, 0, 0, 0, 0], True)
# env.render()

# # attack right bottom from the top
# env.step([5, 0, 0, 0, 0, 0, 1, 2], True)
# env.render()

# # attack right bottom from the left
# env.step([5, 0, 0, 0, 0, 0, 2, 0], True)
# env.render()

# # make light
# env.step([0, 2, 4, 0, 0, 0, 2, 5, 0, 0], True)
# env.render()



# for second worker mine top
# observation, reward, done, info = env.step([0, 1, 2, 0], True)
