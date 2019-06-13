from gym.envs.registration import register
import gym
from .types import Config

# enable repeated experiments
# https://github.com/openai/gym/issues/1172
V0NAME = 'Microrts-v0'
if V0NAME not in gym.envs.registry.env_specs:
    register(
        id=V0NAME,
        entry_point='gym_microrts.envs:RandomAgentEnv',
    )
    
    register(
        id="MicrortsGlobalAgentsDev-v0",
        entry_point='gym_microrts.envs:RandomAgentEnv',
        kwargs={'config': Config(
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/4x4/baseTwoWorkers4x4.xml",
            # below are dev properties
            render=True,
            client_port=9898,
            microrts_repo_path="E:/Go/src/github.com/vwxyzjn/201906051646.microrts"
        )}
    )
    
    register(
        id="MicrortsGlobalAgentsProd-v0",
        entry_point='gym_microrts.envs:RandomAgentEnv',
        kwargs={'config': Config(
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/4x4/baseTwoWorkers4x4.xml",
            # below are dev properties
            render=False,
            auto_port=True,
            microrts_path="/root/microrts"
        )}
    )
    
    register(
        id="MicrortsLocalAgentsDev-v0",
        entry_point='gym_microrts.envs:LocalAgentEnv',
        kwargs={'config': Config(
            ai1_type="no-penalty-individual",
            ai2_type="passive",
            map_path="maps/4x4/baseTwoWorkers4x4.xml",
            # below are dev properties
            render=True,
            client_port=9898,
            microrts_repo_path="E:/Go/src/github.com/vwxyzjn/201906051646.microrts"
        )}
    )
        
    register(
        id="MicrortsLocalAgentsProd-v0",
        entry_point='gym_microrts.envs:LocalAgentEnv',
        kwargs={'config': Config(
            ai1_type="no-penalty-individual",
            ai2_type="passive",
            map_path="maps/4x4/baseTwoWorkers4x4.xml",
            # below are prod properties
            render=False,
            auto_port=True,
            microrts_path="/root/microrts"
        )}
    )
    
    # experiments
    register(
        id="MicrortsLocalAgentsMaxResourcesProd8x8-v0",
        entry_point='gym_microrts.envs:LocalAgentEnv',
        kwargs={'config': Config(
            ai1_type="no-penalty-individual",
            ai2_type="passive",
            map_path="maps/8x8/baseTwoWorkersMaxResources8x8.xml",
            # below are prod properties
            render=False,
            auto_port=True,
            microrts_path="/root/microrts"
        )}
    )
    
    register(
        id="MicrortsGlobalAgentsMaxResourcesProd8x8-v0",
        entry_point='gym_microrts.envs:LocalAgentEnv',
        kwargs={'config': Config(
            ai1_type="no-penalty",
            ai2_type="passive",
            map_path="maps/8x8/baseTwoWorkersMaxResources8x8.xml",
            # below are dev properties
            render=False,
            auto_port=True,
            microrts_path="/root/microrts"
        )}
    )
