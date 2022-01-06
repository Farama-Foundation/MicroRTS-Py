import numpy as np
import gym
import gym_microrts
from gym.envs.registration import register
from gym_microrts import Config

env = gym.make(
    'MicrortsDefeatWorkerRushEnemyHRL-v2',
    map_path="maps/10x10/basesTwoWorkers10x10.xml").env
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(0)
try:
    obs = env.reset()
    env.render()
except Exception as e:
    e.printStackTrace()

print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])


print("reward is", env.step([2, 1, 3, 0, 0, 0, 0, 0, 0])[3]['rewards'])
env.render()
print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
print("reward is", env.step([11, 1, 3, 0, 0, 0, 0, 0, 0])[3]['rewards'])
env.render()
print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
print("reward is", env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])[3]['rewards'])
env.render()
print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])


for i in range(9):
    env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])
    env.render()

# harvest
print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
print("reward is", env.step([10, 2, 0, 0, 0, 0, 0, 0, 0])[3]['rewards'])
env.render()
print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
print("reward is", env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])[3]['rewards'])
env.render()
print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
print("reward is", env.step([1, 2, 0, 3, 0, 0, 0, 0, 0])[3]['rewards'])
env.render()

# dummy
print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
print("reward is", env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])[3]['rewards'])
env.render()

# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])


# print("reward is", env.step([11, 1, 3, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# print("reward is", env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])[1])
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])


# for i in range(9):
#     env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])
#     env.render()

# # # harvest
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# print("reward is", env.step([10, 2, 0, 0, 0, 0, 0, 0, 0])[3]['rewards'])
# env.render()
# print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
# print("reward is", env.step([12, 0, 0, 0, 0, 0, 0, 0, 0])[3]['rewards'])
# env.render()
# # print("unit_locatiuons are at", np.where(env.unit_location_mask==1)[0])
