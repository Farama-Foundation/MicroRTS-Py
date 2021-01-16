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
from gym_microrts import microrts_ai

def calculate_mask(raw_obs, action_space, player=0):
    # obs[3] - obs[4].clip(max=1) means mask busy units
    # * np.where((obs[2])==2,0, (obs[2]))).flatten() means mask units not owned
    unit_location_mask = ((raw_obs[3].clip(max=1) - raw_obs[4].clip(max=1)) * np.where((raw_obs[2])==2,0, (raw_obs[2]))).flatten()
    target_unit_location_mask = ((raw_obs[3].clip(max=1) - raw_obs[4].clip(max=1)) * np.where((raw_obs[2])==1,0, (raw_obs[2]).clip(max=1))).flatten()
    action_mask = np.ones(action_space.nvec.sum())
    action_mask[0:action_space.nvec[0]] = unit_location_mask
    action_mask[-action_space.nvec[-1]:] = target_unit_location_mask
    # print(unit_location_mask.reshape(16,16))
    # print(target_unit_location_mask.reshape(16,16))
    if player == 1:
        return target_unit_location_mask, action_mask
    return unit_location_mask, action_mask


# TODO: fix the temporary relacement of all instance of json.loads(str(response.info)) with {}
# https://github.com/jpype-project/jpype/issues/856

class GlobalAgentEnv(BaseSingleAgentEnv):
    """
    observation space is defined as 
    
    
    
    action space is defined as 
    
    [[0]x_coordinate*y_coordinate(x*y), [1]a_t(6), [2]p_move(4), [3]p_harvest(4), 
    [4]p_return(4), [5]p_produce_direction(4), [6]p_produce_unit_type(z), 
    [7]x_coordinate*y_coordinate(x*y)]
    """
    def __init__(self,
        render_theme=2,
        frame_skip=0, 
        ai2=microrts_ai.passiveAI,
        map_path="maps/10x10/basesTwoWorkers10x10.xml",
        randomize_starting_location=False,
        grid_mode=False,
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        self.randomize_starting_location = randomize_starting_location
        self.grid_mode=grid_mode
        self.reward_weight = reward_weight
        super().__init__(render_theme, frame_skip, ai2, map_path)

    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, SimpleEvaluationRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([SimpleEvaluationRewardFunction()])
        return JNIClient(self.rfs, os.path.expanduser(self.microrts_path), self.map_path)

    def init_properties(self):
        # [num_planes_hp(5), num_planes_resources(5), num_planes_player(5), 
        # num_planes_unit_type(z), num_planes_unit_action(6)]
        self.num_planes = [5, 5, 3, len(self.utt['unitTypes'])+1, 6]
        self.observation_space = spaces.Box(low=0.0,
            high=1.0,
            shape=(self.height, self.width,
                   sum(self.num_planes)),
                   dtype=np.int32)
        self.action_space = spaces.MultiDiscrete([
            self.height * self.width,
            6, 4, 4, 4, 4,
            len(self.utt['unitTypes']),
            self.height * self.width
        ])

    def _encode_obs(self, obs: List):
        obs = obs.reshape(len(obs), -1).clip(0, np.array([self.num_planes]).T-1)
        obs_planes = np.zeros((self.height * self.width, 
                               sum(self.num_planes)), dtype=np.int)
        obs_planes[np.arange(len(obs_planes)),obs[0]] = 1

        for i in range(1, len(self.num_planes)):
            obs_planes[np.arange(len(obs_planes)),obs[i]+sum(self.num_planes[:i])] = 1
        return obs_planes.reshape(self.height, self.width, -1)

    def step(self, action, raw=False, customize=False):
        action = np.array([action])
        response = self.client.step(action, self.player)
        obs, reward, done, info = np.array(response.observation), response.reward[:], response.done[:], {}
        self.unit_location_mask, self.action_mask = calculate_mask(obs, self.action_space, self.player)
        self.unit_location_mask = np.array(self.client.getUnitMasks(self.player)).flatten()
        if not raw:
            obs = self._encode_obs(obs)
        info["dones"] = np.array(done)
        info["rewards"] = np.array(reward)
        info["raw_rewards"] = np.array(reward)
        info["raw_dones"] = np.array(done)
        if customize:
            return obs, reward, done, info
        return obs, reward[0], done[0], info

    def reset(self, raw=False):
        if self.randomize_starting_location:
            self.player = np.random.randint(2)
        else:
            self.player = 0
        response = self.client.reset(self.player)
        obs = np.array(response.observation)
        self.unit_location_mask, self.action_mask = calculate_mask(obs, self.action_space, self.player)
        self.unit_location_mask = np.array(self.client.getUnitMasks(self.player)).flatten()
        if raw:
            return obs
        return self._encode_obs(obs)

    def get_unit_location_mask(self, player=1):
        return np.array(self.client.ai1UnitMasks).flatten()

    def get_unit_location_mask1(self, player=1):
        return np.array(self.client.getUnitMasks(player)).flatten()

    def get_unit_action_mask(self, unit, player=1):
        if self.get_unit_location_mask().sum() == 0:
            return np.zeros(sum(self.action_space.nvec.tolist()[1:]))
        return np.array(self.client.getUnitActionMasks([[unit]]))[0]

class GlobalAgentCombinedRewardEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction, ResourceGatherRewardFunction, AttackRewardFunction, ProduceWorkerRewardFunction, ProduceBuildingRewardFunction, ProduceCombatUnitRewardFunction, CloserToEnemyBaseRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([
            WinLossRewardFunction(), 
            ResourceGatherRewardFunction(),  
            ProduceWorkerRewardFunction(),
            ProduceBuildingRewardFunction(),
            AttackRewardFunction(),
            ProduceCombatUnitRewardFunction(),
            CloserToEnemyBaseRewardFunction(),])
        if self.ai2 is not None:
            return JNIClient(self.rfs, os.path.expanduser(self.microrts_path), self.map_path, self.ai2(self.real_utt), self.real_utt)
        return JNIClient(self.rfs, os.path.expanduser(self.microrts_path), self.map_path)

    def step(self, action, raw=False):
        obs, reward, done, info = super(GlobalAgentCombinedRewardEnv, self).step(action, raw, True)
        reward[-1] = np.clip(reward[-1], -1, 1)
        return obs, (np.array(reward) * self.reward_weight).sum(), done[0], info # win loss as done

class GlobalAgentHRLEnv(GlobalAgentEnv):

    def __init__(self,
        hrl_reward_weights=np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 7.0, 0.0],
        ]), **kwargs):
        self.hrl_reward_weights = hrl_reward_weights
        super().__init__(**kwargs)

    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction, ResourceGatherRewardFunction, AttackRewardFunction, ProduceWorkerRewardFunction, ProduceBuildingRewardFunction, ProduceCombatUnitRewardFunction, CloserToEnemyBaseRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([
            WinLossRewardFunction(), 
            ResourceGatherRewardFunction(),  
            ProduceWorkerRewardFunction(),
            ProduceBuildingRewardFunction(),
            AttackRewardFunction(),
            ProduceCombatUnitRewardFunction(),
            CloserToEnemyBaseRewardFunction(),])
        self.num_reward_function = len(self.hrl_reward_weights)
        if self.ai2 is not None:
            return JNIClient(self.rfs, os.path.expanduser(self.microrts_path), self.map_path, self.ai2(self.real_utt), self.real_utt)
        return JNIClient(self.rfs, os.path.expanduser(self.microrts_path), self.map_path)

    def step(self, action, raw=False):
        obs, reward, done, info = super(GlobalAgentHRLEnv, self).step(action, raw, True)
        info["dones"] = np.array(done)
        info["rewards"] = (np.array(reward) * self.hrl_reward_weights).sum(1)
        return obs, info["rewards"][0], done[0], info

class GlobalAgentMultiActionsCombinedRewardEnv(GlobalAgentEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 150
    }

    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction, ResourceGatherRewardFunction, AttackRewardFunction, ProduceWorkerRewardFunction, ProduceBuildingRewardFunction, ProduceCombatUnitRewardFunction, CloserToEnemyBaseRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([
            WinLossRewardFunction(), 
            ResourceGatherRewardFunction(),  
            ProduceWorkerRewardFunction(),
            ProduceBuildingRewardFunction(),
            AttackRewardFunction(),
            ProduceCombatUnitRewardFunction(),
            CloserToEnemyBaseRewardFunction(),])
        if self.ai2 is not None:
            return JNIClient(self.rfs, os.path.expanduser(self.microrts_path), self.map_path, self.ai2(self.real_utt), self.real_utt)
        return JNIClient(self.rfs, os.path.expanduser(self.microrts_path), self.map_path)

    def step(self, action, raw=False, customize=False):
        if self.grid_mode:
            action = np.array(action)
            response = self.client.gameStep(action, self.player)
            obs, reward, done, info = np.array(response.observation), response.reward[:], response.done[:], {}
            self.unit_location_mask = np.array(self.client.getUnitMasks(self.player)).flatten()
            if not raw:
                obs = self._encode_obs(obs)
            info["dones"] = np.array(done)
            info["rewards"] = np.array(reward)
            info["raw_rewards"] = np.array(reward)
            info["raw_dones"] = np.array(done)
            reward[-1] = np.clip(reward[-1], -1, 1)
            return obs, (np.array(reward) * self.reward_weight).sum(), done[0], info # win loss as done

        action = np.array([action])
        response = self.client.step(action, self.player)
        obs, reward, done, info = np.array(response.observation), response.reward[:], response.done[:], {}
        if not raw:
            obs = self._encode_obs(obs)
        info["dones"] = np.array(done)
        info["rewards"] = np.array(reward)
        info["raw_rewards"] = np.array(reward)
        info["raw_dones"] = np.array(done)
        reward[-1] = np.clip(reward[-1], -1, 1)
        return obs, (np.array(reward) * self.reward_weight).sum(), done[0], info # win loss as done

class GlobalAgentMultiActionsHRLEnv(GlobalAgentMultiActionsCombinedRewardEnv):

    def __init__(self,
        render_theme=2,
        frame_skip=0, 
        ai2=microrts_ai.passiveAI,
        map_path="maps/10x10/basesTwoWorkers10x10.xml",
        hrl_reward_weights=np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 7.0, 0.0],
        ])):
        self.hrl_reward_weights = hrl_reward_weights
        super().__init__(render_theme, frame_skip, ai2, map_path)
    
    def start_client(self):
        self.num_reward_function = len(self.hrl_reward_weights)
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction, ResourceGatherRewardFunction, AttackRewardFunction, ProduceWorkerRewardFunction, ProduceBuildingRewardFunction, ProduceCombatUnitRewardFunction, CloserToEnemyBaseRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([
            WinLossRewardFunction(), 
            ResourceGatherRewardFunction(),  
            ProduceWorkerRewardFunction(),
            ProduceBuildingRewardFunction(),
            AttackRewardFunction(),
            ProduceCombatUnitRewardFunction(),
            CloserToEnemyBaseRewardFunction(),])
        if self.ai2 is not None:
            return JNIClient(self.rfs, os.path.expanduser(self.microrts_path), self.map_path, self.ai2(self.real_utt), self.real_utt)
        return JNIClient(self.rfs, os.path.expanduser(self.microrts_path), self.map_path)

    def step(self, action, raw=False, customize=False):
        obs, reward, done, info = super().step(action, raw, True)
        info["rewards"] = (info["rewards"] * self.hrl_reward_weights).sum(1)
        return obs, info["rewards"][0], done[0], info

    # def reset(self, raw=False):
    #     self.actions = []
    #     # `simulated_rewards` should be subtracted in the end
    #     self.simulated_rewards = 0
    #     return super(GlobalAgentMultiActionsCombinedRewardEnv, self).reset(raw)

