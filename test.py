import gym
import gym_microrts
env = gym.make("Microrts-v0")
config = gym_microrts.types.Config(
    ai1_type="penalty",
    ai2_type="passive",
    map_path="maps/4x4/base4x4.xml",
    render=False,
    client_port=9898,
    microrts_path="E:/Go/src/github.com/vwxyzjn/201905301257.microrts",
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
