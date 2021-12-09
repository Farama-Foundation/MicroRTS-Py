from copy import deepcopy
from gym.envs.registration import register
import gym
import uuid
from .types import Config
import numpy as np
from .v1 import envs as v1_envs
from .v2 import envs as v2_envs
from .v3 import envs as v3_envs

# enable repeated experiments
# https://github.com/openai/gym/issues/1172
V0NAME = 'Microrts-v0'
if V0NAME not in gym.envs.registry.env_specs:
    register(
        id=V0NAME,
        entry_point='gym_microrts.envs:GlobalAgentEnv',
    )
    
    envs = []
    envs += v1_envs
    envs += v2_envs
    envs += v3_envs

    for env in envs:
        # Regular
        register(
            env['id'],
            entry_point=env['entry_point'],
            kwargs=env['kwargs'],
            max_episode_steps=env['max_episode_steps'])

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# enable repeated experiments
# https://github.com/openai/gym/issues/1172
V0NAME = 'Microrts-v0'
if V0NAME not in gym.envs.registry.env_specs:
    register(
        id=V0NAME,
        entry_point='gym_microrts.envs:GlobalAgentEnv',
    )
    
    envs = []
    envs += v1_envs
    envs += v2_envs
    
    envs += [dict(
        id="MicrortsGlobalAgentsProd-v0",
        entry_point='gym_microrts.envs:GlobalAgentEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/4x4/baseTwoWorkers4x4.xml",
            # below are dev properties
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]

    envs += [dict(
        id="MicrortsLocalAgentsProd-v0",
        entry_point='gym_microrts.envs:LocalAgentEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty-individual",
            ai2_type="passive",
            map_path="maps/4x4/baseTwoWorkers4x4.xml",
            # below are prod properties
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]
    
    # experiments
    envs += [dict(
        id=f"MicrortsGlobalAgentsMaxResources4x4Prod-v0",
        entry_point='gym_microrts.envs:GlobalAgentMiningEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/4x4/baseTwoWorkersMaxResources4x4.xml",
            # below are dev properties
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]

    envs += [dict(
        id=f"MicrortsGlobalAgentRandomEnemy10x10FrameSkip9-v0",
        entry_point='gym_microrts.envs:GlobalAgentRandomEnemyEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/4x4/baseTwoWorkersMaxResources4x4.xml",
            # below are dev properties
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]

    envs += [dict(
        id=f"MicrortsGlobalAgentsMaxResources4x4NoFrameSkipProd-v0",
        entry_point='gym_microrts.envs:GlobalAgentMiningEnv',
        kwargs={'config': Config(
            frame_skip=0,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/4x4/baseTwoWorkersMaxResources4x4.xml",
            # below are dev properties
            microrts_path="~/microrts"
        )},
        max_episode_steps=2000,
    )]

    envs += [dict(
        id=f"MicrortsGlobalAgentsMaxResources6x6Prod-v0",
        entry_point='gym_microrts.envs:GlobalAgentEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/6x6/baseTwoWorkersMaxResources6x6.xml",
            # below are dev properties
            microrts_path="~/microrts"
        )},
        max_episode_steps=300,
    )]

    envs += [dict(
        id=f"MicrortsGlobalAgentsMaxResources8x8Prod-v0",
        entry_point='gym_microrts.envs:GlobalAgentEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/8x8/baseTwoWorkersMaxResources8x8.xml",
            # below are dev properties
            microrts_path="~/microrts"
        )},
        max_episode_steps=400,
    )]

    for i in range(1, 4):
        envs += [dict(
            id=f"MicrortsLocalAgentsMaxResources4x4Window{i}Prod-v0",
            entry_point='gym_microrts.envs:LocalAgentEnv',
            kwargs={'config': Config(
                frame_skip=9,
                ai1_type="no-penalty-individual",
                ai2_type="passive",
                map_path="maps/4x4/baseTwoWorkersMaxResources4x4.xml",
                # below are prod properties
                microrts_path="~/microrts",
                window_size=i
            )},
        max_episode_steps=200,
        )]

        envs += [dict(
            id=f"MicrortsLocalAgentsMaxResources6x6Window{i}Prod-v0",
            entry_point='gym_microrts.envs:LocalAgentEnv',
            kwargs={'config': Config(
                frame_skip=9,
                ai1_type="no-penalty-individual",
                ai2_type="passive",
                map_path="maps/6x6/baseTwoWorkersMaxResources6x6.xml",
                # below are prod properties
                microrts_path="~/microrts",
                window_size=i
            )},
        max_episode_steps=300,
        )]

        envs += [dict(
            id=f"MicrortsLocalAgentsMaxResources8x8Window{i}Prod-v0",
            entry_point='gym_microrts.envs:LocalAgentEnv',
            kwargs={'config': Config(
                frame_skip=9,
                ai1_type="no-penalty-individual",
                ai2_type="passive",
                map_path="maps/8x8/baseTwoWorkersMaxResources8x8.xml",
                # below are prod properties
                microrts_path="~/microrts",
                window_size=i
            )},
        max_episode_steps=400,
        )]

    # Mining tasks
    envs += [dict(
        id=f"MicrortsGlobalAgentsMining4x4Prod-v0",
        entry_point='gym_microrts.envs:GlobalAgentMiningEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/4x4/baseOneWorkerMaxResources4x4.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]

    envs += [dict(
        id=f"MicrortsGlobalAgentsMining8x8Prod-v0",
        entry_point='gym_microrts.envs:GlobalAgentMiningEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/8x8/baseOneWorkerMaxResources8x8.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=400,
    )]

    envs += [dict(
        id=f"MicrortsGlobalAgentMining24x24Prod-v0",
        entry_point='gym_microrts.envs:GlobalAgentMiningEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/24x24/basesWorkers24x24.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=400,
    )]

    envs += [dict(
        id=f"MicrortsGlobalAgentMiningHilbert4x4Prod-v0",
        entry_point='gym_microrts.envs:GlobalAgentMiningHilbertEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/4x4/baseOneWorkerMaxResources4x4.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]

    envs += [dict(
        id=f"MicrortsGlobalAgentMiningHilbert8x8Prod-v0",
        entry_point='gym_microrts.envs:GlobalAgentMiningHilbertEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/8x8/baseOneWorkerMaxResources8x8.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=400,
    )]

    # full game
    envs += [dict(
        id="MicrortsGlobalAgentBinary10x10-v0",
        entry_point='gym_microrts.envs:GlobalAgentBinaryEnv',
        kwargs={'config': Config(
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            # below are dev properties
            microrts_path="~/microrts"
        )},
        max_episode_steps=6000,
    )]

    # full game
    envs += [dict(
        id="MicrortsGlobalAgentHRL10x10-v0",
        entry_point='gym_microrts.envs:GlobalAgentHRLEnv',
        kwargs={'config': Config(
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            # below are dev properties
            microrts_path="~/microrts"
        )},
        max_episode_steps=3000,
    )]

    envs += [dict(
        id="MicrortsGlobalAgentHRLMining10x10-v0",
        entry_point='gym_microrts.envs:GlobalAgentHRLMiningEnv',
        kwargs={'config': Config(
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            # below are dev properties
            microrts_path="~/microrts"
        )},
        max_episode_steps=2000,
    )]

    envs += [dict(
        id=f"MicrortsGlobalAgentHRLMining10x10FrameSkip9-v0",
        entry_point='gym_microrts.envs:GlobalAgentHRLMiningEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]

    envs += [dict(
        id=f"MicrortsGlobalAgentHRLAttackReward10x10FrameSkip9-v0",
        entry_point='gym_microrts.envs:GlobalAgentHRLAttackEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]

    envs += [dict(
        id=f"MicrortsGlobalAgentHRLProduceWorker10x10FrameSkip9-v0",
        entry_point='gym_microrts.envs:GlobalAgentHRLProduceWorkerEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]

    envs += [dict(
        id=f"MicrortsGlobalAgentHRLAttackCloserToEnemyBase10x10FrameSkip9-v0",
        entry_point='gym_microrts.envs:GlobalAgentHRLAttackCloserToEnemyBaseEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]

    envs += [dict(
        id=f"MicrortsGlobalAgentMining10x10-v0",
        entry_point='gym_microrts.envs:GlobalAgentMiningEnv',
        kwargs={'config': Config(
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=2000,
    )]

    envs += [dict(
        id=f"MicrortsGlobalAgentMining10x10FrameSkip9-v0",
        entry_point='gym_microrts.envs:GlobalAgentMiningEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]
    envs += [dict(
        id=f"MicrortsGlobalAgentProduceCombatUnit10x10FrameSkip9-v0",
        entry_point='gym_microrts.envs:GlobalAgentProduceCombatUnitEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=400,
    )]

    
    envs += [dict(
        id=f"MicrortsGlobalAgentMining10x10FrameSkip5-v0",
        entry_point='gym_microrts.envs:GlobalAgentMiningEnv',
        kwargs={'config': Config(
            frame_skip=5,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]

    envs += [dict(
        id=f"ParamOpEnvSingleStep-v0",
        entry_point='gym_microrts.envs:ParamOpEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=1,
    )]

    envs += [dict(
        id=f"ParamOpEnvEpisodeMap-0-v0",
        entry_point='gym_microrts.envs:ParamOpEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        ), "map_index" :0 },
        max_episode_steps=200,
    )]
    envs += [dict(
        id=f"ParamOpEnvEpisodeMap-1-v0",
        entry_point='gym_microrts.envs:ParamOpEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        ), "map_index" :1 },
        max_episode_steps=200,
    )]
    envs += [dict(
        id=f"ParamOpEnvEpisodeMap-2-v0",
        entry_point='gym_microrts.envs:ParamOpEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        ), "map_index" :2 },
        max_episode_steps=200,
    )]
    envs += [dict(
        id=f"ParamOpEnvEpisodeMap-3-v0",
        entry_point='gym_microrts.envs:ParamOpEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        ), "map_index" :3 },
        max_episode_steps=200,
    )]
    envs += [dict(
        id=f"ParamOpEnvEpisodeMap-4-v0",
        entry_point='gym_microrts.envs:ParamOpEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        ), "map_index" :4 },
        max_episode_steps=200,
    )]

    envs += [dict(
        id=f"MicrortsMining4x4F9-v0",
        entry_point='gym_microrts.envs:GlobalAgentMiningEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/4x4/basesWorkers4x4.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]
    envs += [dict(
        id=f"MicrortsMining10x10F9-v0",
        entry_point='gym_microrts.envs:GlobalAgentMiningEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]
    envs += [dict(
        id=f"MicrortsMining16x16F9-v0",
        entry_point='gym_microrts.envs:GlobalAgentMiningEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/16x16/basesWorkers16x16R20.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]
    envs += [dict(
        id=f"MicrortsMining24x24F9-v0",
        entry_point='gym_microrts.envs:GlobalAgentMiningEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/24x24/basesWorkers24x24R20.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]

    envs += [dict(
        id=f"MicrortsAttack10x10F9-v0",
        entry_point='gym_microrts.envs:GlobalAgentAttackEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]
    envs += [dict(
        id=f"MicrortsCloserToEnemyBase10x10F9-v0",
        entry_point='gym_microrts.envs:GlobalAgentCloserToEnemyBaseRewardEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]
    envs += [dict(
        id=f"MicrortsCombinedReward10x10F9-v0",
        entry_point='gym_microrts.envs:GlobalAgentCombinedRewardEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]


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
    def ai2():
        from ai import RandomBiasedSingleUnitAI
        return RandomBiasedSingleUnitAI()
    envs += [dict(
        id=f"MicrortsCombinedReward10x10F9FightRandomBiased-v0",
        entry_point='gym_microrts.envs:GlobalAgentCombinedRewardEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            ai2=ai2,
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts",
            reward_weight=np.array([5.0, 1.0, 1.0, 0.2, 1.0, 3.0, 0.2])
        )},
        max_episode_steps=400,
    )]
    envs += [dict(
        id=f"MicrortsCombinedReward10x10F9FightRandomBiasedW1-v0",
        entry_point='gym_microrts.envs:GlobalAgentCombinedRewardEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            ai2=ai2,
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts",
            reward_weight=np.array([10.0, 1.0, 1.0, 2.0, 1.0, 7.0, 0.2])
        )},
        max_episode_steps=400,
    )]

    # produce combat units
    envs += [dict(
        id=f"MicrortsCombinedReward10x10F9BuildCombatUnits-v0",
        entry_point='gym_microrts.envs:GlobalAgentCombinedRewardEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts",
            reward_weight=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 7.0, 0.0])
        )},
        max_episode_steps=400,
    )]
    envs += [dict(
        id=f"MicrortsGlobalAgentHRLProduceCombatUnit10x10FrameSkip9-v0",
        entry_point='gym_microrts.envs:GlobalAgentHRLProduceCombatUnitEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=200,
    )]
    envs += [dict(
        id=f"MicrortsGlobalAgentHRLProduceCombatUnitPerfect10x10FrameSkip9-v0",
        entry_point='gym_microrts.envs:GlobalAgentHRLProduceCombatUnitPerfectEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts"
        )},
        max_episode_steps=400,
    )]
    envs += [dict(
        id=f"MicrortsCombinedReward10x10F9ProduceCombatUnitsRandomBiased-v0",
        entry_point='gym_microrts.envs:GlobalAgentCombinedRewardEnv',
        kwargs={'config': Config(
            frame_skip=9,
            ai1_type="no-penalty",
            ai2_type="passive",
            ai2=ai2,
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts",
            reward_weight=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 7.0, 0.0])
        )},
        max_episode_steps=2000,
    )]

    # multi-actions
    envs += [dict(
        id=f"MicrortsCombinedReward10x10F5MultiActionsBuildCombatUnits-v0",
        entry_point='gym_microrts.envs:GlobalAgentMultiActionsCombinedRewardEnv',
        kwargs={'config': Config(
            frame_skip=5,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts",
            reward_weight=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 7.0, 0.0])
        )},
        max_episode_steps=400,
    )]

    envs += [dict(
        id=f"MicrortsCombinedReward10x10MultiActionsBuildCombatUnits-v0",
        entry_point='gym_microrts.envs:GlobalAgentMultiActionsCombinedRewardEnv',
        kwargs={'config': Config(
            frame_skip=0,
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/10x10/basesWorkers10x10.xml",
            microrts_path="~/microrts",
            reward_weight=np.array([0.0, 1.0, 0.0, 1.0, 1.0, 7.0, 0.0])
        )},
        max_episode_steps=2000,
    )]
    
    # envs += [dict(
    #     id=f"MicrortsCloserToEnemyBase10x10F9-v0",
    #     entry_point='gym_microrts.envs:GlobalAgentCloserToEnemyBaseRewardEnv',
    #     kwargs={'config': Config(
    #         frame_skip=9,
    #         ai1_type="no-penalty",
    #         ai2_type="passive",
    #         map_path="maps/10x10/basesWorkers10x10.xml",
    #         microrts_path="~/microrts"
    #     )},
    #     max_episode_steps=200,
    # )]

    # Additional variants and registration
    for env in envs:
        # Regular
        register(
            env['id'],
            entry_point=env['entry_point'],
            kwargs=env['kwargs'],
            max_episode_steps=env['max_episode_steps'])
        # Evaluation
        env_p = deepcopy(env)
        env_p['id'] = "Eval" + env_p['id']
        env_p['kwargs']['config'].evaluation_filename = "evals/"+str(uuid.uuid4())+".json"
        register(
            env_p['id'],
            entry_point=env_p['entry_point'],
            kwargs=env_p['kwargs'],
            max_episode_steps=env['max_episode_steps'])