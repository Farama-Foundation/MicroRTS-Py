import numpy as np
import gym
import gym_microrts

from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai
from gym.envs.registration import register
from gym_microrts import Config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

try:
    env = MicroRTSGridModeVecEnv(
        num_envs=1,
        render_theme=2,
        ai2s=[microrts_ai.passiveAI],
        map_path="maps/16x16/basesWorkersTestAttack16x16.xml",
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    # env = gym.make('MicrortsDefeatCoacAIShaped-v3').env
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env.action_space.seed(0)
    obs = env.reset()
    env.render()
except Exception as e:
    e.printStackTrace()
# print("reward is", env.step([[[ 17,   2 ,  0 ,  3 ,  0 ,  1 ,  2, 0]]])[1])
# env.render()
# print("reward is", env.step([[[ 34  , 4 ,  1   ,2  , 1 ,  2  , 3 ,0]]])[1])
# env.render()

# print("reward is", env.step([[[ 14*16+14  , 1 ,  0   ,0  , 0 ,  0  , 0 ,0]]])[1])
# env.render()
# for _ in range(100):
#     env.step([[[ 0  , 0 ,  0   ,0  , 0 ,  0  , 0 ,0]]])
#     env.render()

# print("relative target position:", np.where(np.array(env.vec_client.getMasks(0))[0,13,14][1+6+4+4+4+4+7:]==1)[0])
# print("reward is", env.step([[[ 13*16+14  , 5 ,  0   ,0  , 0 ,  0  , 0 ,23]]])[1])
# env.render()


print("reward is", env.step([[[ 14*16+14  , 1 ,  3   ,0  , 0 ,  0  , 0 ,0]]])[1])
env.render()
for _ in range(100):
    env.step([[[ 0  , 0 ,  0   ,0  , 0 ,  0  , 0 ,0]]])
    env.render()

print("mask:", np.array(env.vec_client.getMasks(0))[0,14,13])
print("relative target position:", np.where(np.array(env.vec_client.getMasks(0))[0,14,13][1+6+4+4+4+4+7:]==1)[0])
print("reward is", env.step([[[ 14*16+13  , 5 ,  0   ,0  , 0 ,  0  , 0 ,17]]])[1])
env.render()


# print("reward is", env.step([[[ 13*16+14  , 5 ,  0   ,0  , 0 ,  0  , 0 ,7]]])[1])
# env.render()
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