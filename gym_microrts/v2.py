from .types import Config
import numpy as np

"""
WinLossRewardFunction(), 
ResourceGatherRewardFunction(),  
ProduceWorkerRewardFunction(),
ProduceBuildingRewardFunction(),
AttackRewardFunction(),
ProduceCombatUnitRewardFunction(),
CloserToEnemyBaseRewardFunction(),
reward_weight corresponds to above
"""
def randomAI():
    from ai import RandomBiasedSingleUnitAI
    return RandomBiasedSingleUnitAI()

def passiveAI():
    from ai import PassiveAI
    return PassiveAI()

shaped_reward_envs = True
hrl_envs = True

envs = []
envs += [dict(
    id=f"MicrortsTwoWorkersMiningLegacy-v2",
    entry_point='gym_microrts.envs:GlobalAgentMiningEnv',
    kwargs={'config': Config(
        frame_skip=0,
        ai2=passiveAI,
        map_path="maps/10x10/basesTwoWorkers10x10.xml",
        microrts_path="~/microrts"
    )},
    max_episode_steps=400,
)]

# envs += [dict(
#     id=f"MicrortsTwoWorkersMining-v2",
#     entry_point='gym_microrts.envs:GlobalAgentMiningEnv',
#     kwargs={'config': Config(
#         frame_skip=0,
#         ai2=passiveAI,
#         map_path="maps/10x10/basesTwoWorkers10x10.xml",
#         microrts_path="~/microrts"
#     )},
#     max_episode_steps=2000,
# )]

envs += [dict(
    id=f"MicrortsTwoWorkersMining-v2",
    entry_point='gym_microrts.envs:GlobalAgentMultiActionsCombinedRewardEnv',
    kwargs={'config': Config(
        frame_skip=0,
        ai2=passiveAI,
        map_path="maps/10x10/basesTwoWorkers10x10.xml",
        microrts_path="~/microrts",
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )},
    max_episode_steps=400,
)]

