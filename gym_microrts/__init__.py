from copy import deepcopy
from gym.envs.registration import register
import gym
import uuid
from .types import Config

# enable repeated experiments
# https://github.com/openai/gym/issues/1172
V0NAME = 'Microrts-v0'
if V0NAME not in gym.envs.registry.env_specs:
    register(
        id=V0NAME,
        entry_point='gym_microrts.envs:GlobalAgentEnv',
    )
    
    envs = []
    
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
        entry_point='gym_microrts.envs:GlobalAgentEnv',
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

    # Additional variants and registration
    for env in envs:
        # Regular
        register(env['id'], entry_point=env['entry_point'], kwargs=env['kwargs'])
        # Evaluation
        env_p = deepcopy(env)
        env_p['id'] = "Eval" + env_p['id']
        env_p['kwargs']['config'].evaluation_filename = "evals/"+str(uuid.uuid4())+".json"
        register(env_p['id'], entry_point=env_p['entry_point'], kwargs=env_p['kwargs'])