import numpy as np
import gym
import gym_microrts

from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from gym.envs.registration import register
from gym_microrts import Config

try:
    env = MicroRTSVecEnv(
        num_envs=1,
        render_theme=2,
        ai2=microrts_ai.coacAI,
        map_path="maps/16x16/basesWorkers16x16.xml",
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0, 0.0])
    )
    # env = gym.make('MicrortsDefeatCoacAIShaped-v3').env
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env.action_space.seed(0)
    obs = env.reset()
    env.render()
except Exception as e:
    e.printStackTrace()

# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])


print("reward is", env.step([[ 17,   2 ,  0 ,  3 ,  0 ,  1 ,  2, 123]])[1])
env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
print("reward is", env.step([[ 34  , 4 ,  1   ,2  , 1 ,  2  , 3 ,109]])[1])
env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# print("reward is", env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])


# for i in range(9):
#     env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])
#     env.render()

# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# # # harvest
# print("reward is", env.step([1, 2, 0, 3, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# print("reward is", env.step([10, 2, 0, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# print("reward is", env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])

# # # creating worker
# # print("reward is", env.step([12, 4, 0, 0, 0, 1, 3, 0, 0])[1])
# # env.render()
# # print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])

# for i in range(19):
#     env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])
#     env.render()
#     # print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])


# # # move back
# print("reward is", env.step([1, 1, 1, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# print("reward is", env.step([10, 1, 1, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# print("reward is", env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])

# for i in range(9):
#     env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])
#     env.render()

# # # return
# print("reward is", env.step([2, 3, 0, 0, 2, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])
# print("reward is", env.step([11, 3, 0, 0, 1, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.get_unit_location_mask()==1)[0])

# for i in range(29):
#     env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])
#     env.render()