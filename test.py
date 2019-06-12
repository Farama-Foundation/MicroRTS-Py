import gym
import gym_microrts
env = gym.make("Microrts-v0")
config = gym_microrts.types.Config(
    ai1_type="no-penalty-individual",
    ai2_type="passive",
    map_path="maps/4x4/baseTwoWorkers4x4.xml",
    render=True,
    client_port=9898,
    microrts_path="E:/Go/src/github.com/vwxyzjn/201906051646.microrts",
    microrts_repo_path="E:/Go/src/github.com/vwxyzjn/microrts"
)

env.init(config)
#observation = env.reset()

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
