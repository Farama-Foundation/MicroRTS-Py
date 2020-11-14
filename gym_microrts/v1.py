from .types import Config
import numpy as np
from . import microrts_ai

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
shaped_reward_envs = True
hrl_envs = True

envs = []
envs += [dict(
    id=f"MicrortsMining-v1",
    entry_point='gym_microrts.envs:GlobalAgentCombinedRewardEnv',
    kwargs=dict(
        frame_skip=9,
        ai2=microrts_ai.passiveAI,
        map_path="maps/10x10/basesWorkers10x10.xml",
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ),
    max_episode_steps=200,
)]

envs += [dict(
    id=f"MicrortsProduceWorker-v1",
    entry_point='gym_microrts.envs:GlobalAgentProduceWorkerEnv',
    kwargs=dict(
        frame_skip=9,
        ai2=microrts_ai.passiveAI,
        map_path="maps/10x10/basesWorkers10x10.xml",
        microrts_path="~/microrts"
    ),
    max_episode_steps=200,
)]

envs += [dict(
    id=f"MicrortsAttackSparseReward-v1",
    entry_point='gym_microrts.envs:GlobalAgentCombinedRewardEnv',
    kwargs=dict(
        frame_skip=9,
        ai2=microrts_ai.passiveAI,
        map_path="maps/10x10/basesWorkers10x10.xml",
        reward_weight=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    ),
    max_episode_steps=200,
)]
if shaped_reward_envs:
    envs += [dict(
        id=f"MicrortsAttackShapedReward-v1",
        entry_point='gym_microrts.envs:GlobalAgentCombinedRewardEnv',
        kwargs=dict(
            frame_skip=9,
            ai2=microrts_ai.passiveAI,
            map_path="maps/10x10/basesWorkers10x10.xml",
            reward_weight=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        ),
        max_episode_steps=400,
    )]
if hrl_envs:
    envs += [dict(
        id=f"MicrortsAttackHRL-v1",
        entry_point='gym_microrts.envs:GlobalAgentHRLEnv',
        kwargs=dict(
            frame_skip=9,
            ai2=microrts_ai.passiveAI,
            map_path="maps/10x10/basesWorkers10x10.xml",
            hrl_reward_weights=np.array([
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            ])
        ),
        max_episode_steps=200,
    )]

envs += [dict(
    id=f"MicrortsProduceCombatUnitsSparseReward-v1",
    entry_point='gym_microrts.envs:GlobalAgentProduceCombatUnitEnv',
    kwargs=dict(
        frame_skip=9,
        ai2=microrts_ai.passiveAI,
        map_path="maps/10x10/basesWorkers10x10.xml",
        microrts_path="~/microrts"
    ),
    max_episode_steps=400,
)]
if shaped_reward_envs:
    envs += [dict(
        id=f"MicrortsProduceCombatUnitsShapedReward-v1",
        entry_point='gym_microrts.envs:GlobalAgentCombinedRewardEnv',
        kwargs=dict(
            frame_skip=9,
            ai2=microrts_ai.passiveAI,
            map_path="maps/10x10/basesWorkers10x10.xml",
            reward_weight=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 7.0, 0.0])
        ),
        max_episode_steps=400,
    )]
if hrl_envs:
    envs += [dict(
        id=f"MicrortsProduceCombatUnitHRL-v1",
        entry_point='gym_microrts.envs:GlobalAgentHRLEnv',
        kwargs=dict(
            frame_skip=9,
            ai2=microrts_ai.passiveAI,
            map_path="maps/10x10/basesWorkers10x10.xml",
            hrl_reward_weights=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 7.0, 0.0],
            ])
        ),
        max_episode_steps=400,
    )]


envs += [dict(
    id="MicrortsRandomEnemySparseReward-v1",
    entry_point='gym_microrts.envs:GlobalAgentBinaryEnv',
    kwargs=dict(
        frame_skip=9,
        ai2=microrts_ai.randomAI,
        map_path="maps/10x10/basesWorkers10x10.xml",
        microrts_path="~/microrts"
    ),
    max_episode_steps=600,
)]
if shaped_reward_envs:
    envs += [dict(
        id=f"MicrortsRandomEnemyShapedReward1-v1",
        entry_point='gym_microrts.envs:GlobalAgentCombinedRewardEnv',
        kwargs=dict(
            frame_skip=9,
            ai2=microrts_ai.randomAI,
            map_path="maps/10x10/basesWorkers10x10.xml",
            reward_weight=np.array([0.0, 1.0, 0.0, 1.0, 1.0, 7.0, 0.0])
        ),
        max_episode_steps=600,
    )]
    envs += [dict(
        id=f"MicrortsRandomEnemyShapedReward2-v1",
        entry_point='gym_microrts.envs:GlobalAgentCombinedRewardEnv',
        kwargs=dict(
            frame_skip=9,
            ai2=microrts_ai.randomAI,
            map_path="maps/10x10/basesWorkers10x10.xml",
            reward_weight=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 7.0, 0.0])
        ),
        max_episode_steps=600,
    )]
    envs += [dict(
        id=f"MicrortsRandomEnemyShapedReward3-v1",
        entry_point='gym_microrts.envs:GlobalAgentCombinedRewardEnv',
        kwargs=dict(
            frame_skip=9,
            ai2=microrts_ai.randomAI,
            map_path="maps/10x10/basesWorkers10x10.xml",
            reward_weight=np.array([5.0, 1.0, 1.0, 0.2, 1.0, 7.0, 0.2])
        ),
        max_episode_steps=600,
    )]
if hrl_envs:
    envs += [dict(
        id=f"MicrortsRandomEnemyHRL1-v1",
        entry_point='gym_microrts.envs:GlobalAgentHRLEnv',
        kwargs=dict(
            frame_skip=9,
            ai2=microrts_ai.randomAI,
            map_path="maps/10x10/basesWorkers10x10.xml",
            hrl_reward_weights=np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 1.0, 7.0, 0.0],
            ])
        ),
        max_episode_steps=600,
    )]
    envs += [dict(
        id=f"MicrortsRandomEnemyHRL2-v1",
        entry_point='gym_microrts.envs:GlobalAgentHRLEnv',
        kwargs=dict(
            frame_skip=9,
            ai2=microrts_ai.randomAI,
            map_path="maps/10x10/basesWorkers10x10.xml",
            hrl_reward_weights=np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 7.0, 0.0],
            ])
        ),
        max_episode_steps=600,
    )]
    envs += [dict(
        id=f"MicrortsRandomEnemyHRL3-v1",
        entry_point='gym_microrts.envs:GlobalAgentHRLEnv',
        kwargs=dict(
            frame_skip=9,
            ai2=microrts_ai.randomAI,
            map_path="maps/10x10/basesWorkers10x10.xml",
            hrl_reward_weights=np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [5.0, 1.0, 1.0, 0.2, 1.0, 7.0, 0.2],
            ])
        ),
        max_episode_steps=600,
    )]

envs += [dict(
    id=f"MicrortsWorkerRush-v1",
    entry_point='gym_microrts.envs:GlobalAgentCombinedRewardEnv',
    kwargs=dict(
        frame_skip=0,
        ai2=microrts_ai.workerRushAI,
        map_path="maps/10x10/basesWorkers10x10.xml",
        reward_weight=np.array([0.0, 1.0, 0.0, 1.0, 1.0, 7.0, 0.0])
    ),
    max_episode_steps=600,
)]


envs += [dict(
    id=f"MicrortsSelfPlayShapedReward-v1",
    entry_point='gym_microrts.envs:GlobalAgentCombinedRewardSelfPlayEnv',
    kwargs=dict(
        frame_skip=9,
        map_path="maps/10x10/basesTwoWorkers10x10.xml",
        reward_weight=np.array([5.0, 1.0, 1.0, 0.2, 1.0, 7.0, 0.0])
    ),
    max_episode_steps=600,
)]
