import gym
import socket
import numpy as np
import json
from subprocess import Popen, PIPE
import os
from typing import List, Tuple
from dacite import from_dict
from gym_microrts.types import MicrortsMessage, Config
from gym import error, spaces, utils
import xml.etree.ElementTree as ET
from gym.utils import seeding
from gym_microrts.envs.base_env import BaseSingleAgentEnv
from jpype.types import JArray

class GlobalAgentEnv(BaseSingleAgentEnv):
    """
    observation space is defined as 
    
    
    
    action space is defined as 
    
    [[0]x_coordinate*y_coordinate(x*y), [1]a_t(6), [2]p_move(4), [3]p_harvest(4), 
    [4]p_return(4), [5]p_produce_direction(4), [6]p_produce_unit_type(z), 
    [7]x_coordinate*y_coordinate(x*y)]
    """

    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, SimpleEvaluationRewardFunction
        rfs = JArray(RewardFunctionInterface)([SimpleEvaluationRewardFunction()])
        return JNIClient(rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

    def init_properties(self):
        # [num_planes_hp(5), num_planes_resources(5), num_planes_player(5), 
        # num_planes_unit_type(z), num_planes_unit_action(6)]
        self.num_planes = [5, 5, 3, len(self.utt['unitTypes'])+1, 6]
        self.observation_space = spaces.Box(low=0.0,
            high=1.0,
            shape=(self.config.height, self.config.width,
                   sum(self.num_planes)),
                   dtype=np.int32)
        self.action_space = spaces.MultiDiscrete([
            self.config.height * self.config.width,
            6, 4, 4, 4, 4,
            len(self.utt['unitTypes']),
            self.config.height * self.config.width
        ])

    def _encode_obs(self, obs: List):
        obs = obs.reshape(len(obs), -1).clip(0, np.array([self.num_planes]).T-1)
        obs_planes = np.zeros((self.config.height * self.config.width, 
                               sum(self.num_planes)), dtype=np.int)
        obs_planes[np.arange(len(obs_planes)),obs[0]] = 1

        for i in range(1, len(self.num_planes)):
            obs_planes[np.arange(len(obs_planes)),obs[i]+sum(self.num_planes[:i])] = 1
        return obs_planes.reshape(self.config.height, self.config.width, -1)

    def step(self, action, raw=False):
        obs, reward, done, info = super(GlobalAgentEnv, self).step(action, True)
        # obs[3] - obs[4].clip(max=1) means mask busy units
        # * np.where((obs[2])==2,0, (obs[2]))).flatten() means mask units not owned
        self.unit_location_mask = ((obs[3].clip(max=1) - obs[4].clip(max=1)) * np.where((obs[2])==2,0, (obs[2]))).flatten()
        if not raw:
            obs = self._encode_obs(obs)
        if len(reward)==1:
            return obs, reward[0], done[0], info
        return obs, reward, done, info

    def reset(self, raw=False):
        raw_obs = super(GlobalAgentEnv, self).reset(True)
        self.unit_location_mask = ((raw_obs[3].clip(max=1) - raw_obs[4].clip(max=1)) * np.where((raw_obs[2])==2,0, (raw_obs[2]))).flatten()
        if raw:
            return raw_obs
        return self._encode_obs(raw_obs)

class GlobalAgentBinaryEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction
        rfs = JArray(RewardFunctionInterface)([WinLossRewardFunction()])
        return JNIClient(rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

class GlobalAgentMiningEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, ResourceGatherRewardFunction
        rfs = JArray(RewardFunctionInterface)([ResourceGatherRewardFunction()])
        return JNIClient(rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

class GlobalAgentAttackEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, AttackRewardFunction
        rfs = JArray(RewardFunctionInterface)([AttackRewardFunction()])
        return JNIClient(rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

class GlobalAgentProduceWorkerEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, ProduceWorkerRewardFunction
        rfs = JArray(RewardFunctionInterface)([ProduceWorkerRewardFunction()])
        return JNIClient(rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

class GlobalAgentProduceBuildingEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, ProduceBuildingRewardFunction
        rfs = JArray(RewardFunctionInterface)([ProduceBuildingRewardFunction()])
        return JNIClient(rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

class GlobalAgentProduceCombatUnitEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, ProduceCombatUnitRewardFunction
        rfs = JArray(RewardFunctionInterface)([ProduceCombatUnitRewardFunction()])
        return JNIClient(rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

class GlobalAgentHRLEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction, ResourceGatherRewardFunction, AttackRewardFunction, ProduceWorkerRewardFunction, ProduceBuildingRewardFunction, ProduceCombatUnitRewardFunction
        rfs = JArray(RewardFunctionInterface)([
            WinLossRewardFunction(), 
            ResourceGatherRewardFunction(), 
            AttackRewardFunction(), 
            ProduceWorkerRewardFunction(),
            ProduceBuildingRewardFunction(),
            ProduceCombatUnitRewardFunction(),])
        return JNIClient(rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)
