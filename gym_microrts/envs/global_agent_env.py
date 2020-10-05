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
        self.rfs = JArray(RewardFunctionInterface)([SimpleEvaluationRewardFunction()])
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

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

    def step(self, action, raw=False, customize=False):
        obs, reward, done, info = super(GlobalAgentEnv, self).step(action, True)
        # obs[3] - obs[4].clip(max=1) means mask busy units
        # * np.where((obs[2])==2,0, (obs[2]))).flatten() means mask units not owned
        self.unit_location_mask = ((obs[3].clip(max=1) - obs[4].clip(max=1)) * np.where((obs[2])==2,0, (obs[2]))).flatten()
        self.target_unit_location_mask = ((obs[3].clip(max=1) - obs[4].clip(max=1)) * np.where((obs[2])==1,0, (obs[2]).clip(max=1))).flatten()
        self.action_mask = np.ones(self.action_space.nvec.sum())
        self.action_mask[0:self.action_space.nvec[0]] = self.unit_location_mask
        self.action_mask[-self.action_space.nvec[-1]:] = self.target_unit_location_mask

        if not raw:
            obs = self._encode_obs(obs)
        info["dones"] = np.array(done)
        info["rewards"] = np.array(reward).clip(min=-1, max=1)
        info["raw_rewards"] = np.array(reward).clip(min=-1, max=1)
        info["raw_dones"] = np.array(done)
        if customize:
            return obs, reward, done, info
        return obs, reward[0], done[0], info

    def reset(self, raw=False):
        raw_obs = super(GlobalAgentEnv, self).reset(True)
        self.unit_location_mask = ((raw_obs[3].clip(max=1) - raw_obs[4].clip(max=1)) * np.where((raw_obs[2])==2,0, (raw_obs[2]))).flatten()
        self.target_unit_location_mask = ((raw_obs[3].clip(max=1) - raw_obs[4].clip(max=1)) * np.where((raw_obs[2])==1,0, (raw_obs[2]).clip(max=1))).flatten()
        self.action_mask = np.ones(self.action_space.nvec.sum())
        self.action_mask[0:self.action_space.nvec[0]] = self.unit_location_mask
        self.action_mask[-self.action_space.nvec[-1]:] = self.target_unit_location_mask
        if raw:
            return raw_obs
        return self._encode_obs(raw_obs)

    def get_unit_location_mask(self, player=1):
        if player == 1:
            return self.unit_location_mask

    def get_unit_action_mask(self, unit, player=1):
        if player == 1:
            if self.unit_location_mask.sum() == 0:
                return np.zeros(sum(self.action_space.nvec.tolist()[1:]))
            return np.array(self.client.getUnitActionMasks([[unit]]))[0]

class GlobalAgentBinaryEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([WinLossRewardFunction()])
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

class GlobalAgentMiningEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, ResourceGatherRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([ResourceGatherRewardFunction()])
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

class GlobalAgentAttackEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, AttackRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([AttackRewardFunction()])
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

class GlobalAgentProduceWorkerEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, ProduceWorkerRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([ProduceWorkerRewardFunction()])
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

class GlobalAgentProduceBuildingEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, ProduceBuildingRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([ProduceBuildingRewardFunction()])
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

class GlobalAgentProduceCombatUnitEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, ProduceCombatUnitRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([ProduceCombatUnitRewardFunction()])
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

class GlobalAgentCloserToEnemyBaseRewardEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, CloserToEnemyBaseRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([CloserToEnemyBaseRewardFunction()])
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

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
        if self.config.ai2 is not None:
            return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path, self.config.ai2(self.real_utt), self.real_utt)
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

    def step(self, action, raw=False):
        obs, reward, done, info = super(GlobalAgentCombinedRewardEnv, self).step(action, raw, True)
        reward[-1] = np.clip(reward[-1], -1, 1)
        return obs, (np.array(reward).clip(min=-1, max=1) * self.config.reward_weight).sum(), done[0], info # win loss as done

class GlobalAgentHRLEnv(GlobalAgentEnv):
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
        self.num_reward_function = len(self.config.hrl_reward_weights)
        if self.config.ai2 is not None:
            return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path, self.config.ai2(self.real_utt))
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

    def step(self, action, raw=False):
        obs, reward, done, info = super(GlobalAgentHRLEnv, self).step(action, raw, True)
        info["dones"] = np.array(done)
        info["rewards"] = (np.array(reward).clip(min=-1, max=1) * self.config.hrl_reward_weights).sum(1)
        return obs, info["rewards"][0], done[0], info

class GlobalAgentHRLAttackCloserToEnemyBaseEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction, ResourceGatherRewardFunction, AttackRewardFunction, ProduceWorkerRewardFunction, ProduceBuildingRewardFunction, ProduceCombatUnitRewardFunction, CloserToEnemyBaseRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([
            AttackRewardFunction(),
            CloserToEnemyBaseRewardFunction(),])
        self.num_reward_function = len(self.rfs)
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

class GlobalAgentHRLProduceCombatUnitEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction, ResourceGatherRewardFunction, AttackRewardFunction, ProduceWorkerRewardFunction, ProduceBuildingRewardFunction, ProduceCombatUnitRewardFunction, CloserToEnemyBaseRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([
            ProduceCombatUnitRewardFunction(),
            ProduceBuildingRewardFunction(),
            ResourceGatherRewardFunction(),])
        self.num_reward_function = 2
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

class GlobalAgentHRLProduceCombatUnitPerfectEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction, ResourceGatherRewardFunction, AttackRewardFunction, ProduceWorkerRewardFunction, ProduceBuildingRewardFunction, ProduceCombatUnitRewardFunction, CloserToEnemyBaseRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([
            ProduceCombatUnitRewardFunction(),
            ResourceGatherRewardFunction(),
            ProduceBuildingRewardFunction(),])
        self.num_reward_function = 2
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

    def step(self, action, raw=False):
        obs, reward, done, info = super(GlobalAgentHRLProduceCombatUnitPerfectEnv, self).step(action, raw, True)
        info["dones"] = np.array(done)
        info["rewards"] = [reward[0], 
            (np.array(reward).clip(min=-1, max=1)*np.array([7.0,1.0,1.0])).sum()]
        return obs, reward[0], done[0], info

class GlobalAgentRandomEnemyEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNIClient
        from ai import RandomBiasedAI
        from ai.rewardfunction import RewardFunctionInterface, SimpleEvaluationRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([SimpleEvaluationRewardFunction()])
        ai2 = RandomBiasedAI()
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path, ai2)

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
        if self.config.ai2 is not None:
            return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path, self.config.ai2(self.real_utt), self.real_utt)
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

    def step(self, action, raw=False, customize=False):
        # action = np.array(action)
        num_source_units = self.unit_location_mask.sum()

        while num_source_units >= 2:
            source_unit_selected = action[0]
            self.actions += [action]
            self.unit_location_mask[source_unit_selected] = 0
            self.action_mask = np.ones(self.action_space.nvec.sum())
            self.action_mask[0:self.action_space.nvec[0]] = self.unit_location_mask
            self.action_mask[-self.action_space.nvec[-1]:] = self.target_unit_location_mask
            info = {}
            response = self.client.simulateStep(np.array(self.actions), self.config.frame_skip)
            obs, reward, done, info = np.array(response.observation), response.reward[:], response.done[:], json.loads(str(response.info))
            info["dones"] = np.array(done)
            info["rewards"] = np.array(reward)
            info["raw_rewards"] = np.array(reward)
            info["raw_dones"] = np.array(done)
            if not raw:
                obs = self._encode_obs(obs)
            
            new_reward = (np.array(reward) * self.config.reward_weight).sum()
            reward_difference = new_reward - self.simulated_rewards
            return_tuple = (obs, reward_difference, done[0], info)
            self.simulated_rewards += reward_difference
            return return_tuple

        self.actions += [action]
        response = self.client.step(np.array(self.actions), self.config.frame_skip)
        self.actions = []
        obs, reward, done, info = np.array(response.observation), response.reward[:], response.done[:], json.loads(str(response.info))
        # obs[3] - obs[4].clip(max=1) means mask busy units
        # * np.where((obs[2])==2,0, (obs[2]))).flatten() means mask units not owned
        self.unit_location_mask = ((obs[3].clip(max=1) - obs[4].clip(max=1)) * np.where((obs[2])==2,0, (obs[2]))).flatten()
        self.target_unit_location_mask = ((obs[3].clip(max=1) - obs[4].clip(max=1)) * np.where((obs[2])==1,0, (obs[2]).clip(max=1))).flatten()
        self.action_mask = np.ones(self.action_space.nvec.sum())
        self.action_mask[0:self.action_space.nvec[0]] = self.unit_location_mask
        self.action_mask[-self.action_space.nvec[-1]:] = self.target_unit_location_mask
        if not raw:
            obs = self._encode_obs(obs)
        info["dones"] = np.array(done)
        info["rewards"] = np.array(reward)
        info["raw_rewards"] = np.array(reward)
        info["raw_dones"] = np.array(done)
        if customize:
            return obs, reward, done, info
        new_reward = (np.array(reward) * self.config.reward_weight).sum()
        reward_difference = new_reward - self.simulated_rewards
        self.simulated_rewards = 0
        return (obs, reward_difference, done[0], info)

    def reset(self, raw=False):
        self.actions = []
        # `simulated_rewards` should be subtracted in the end
        self.simulated_rewards = 0
        return super(GlobalAgentMultiActionsCombinedRewardEnv, self).reset(raw)

class GlobalAgentMultiActionsHRLEnv(GlobalAgentMultiActionsCombinedRewardEnv):

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
        self.num_reward_function = len(self.config.hrl_reward_weights)
        if self.config.ai2 is not None:
            return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path, self.config.ai2(self.real_utt), self.real_utt)
        return JNIClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path)

    def step(self, action, raw=False, customize=False):
        # action = np.array(action)
        num_source_units = self.unit_location_mask.sum()
        # `simulated_rewards` should be subtracted in the end
        simulated_rewards = np.zeros(self.num_reward_function)
        while num_source_units >= 1:
            source_unit_selected = action[0]
            self.actions += [action]
            self.unit_location_mask[source_unit_selected] = 0
            self.action_mask = np.ones(self.action_space.nvec.sum())
            self.action_mask[0:self.action_space.nvec[0]] = self.unit_location_mask
            self.action_mask[-self.action_space.nvec[-1]:] = self.target_unit_location_mask
            info = {}
            response = self.client.simulateStep(np.array(self.actions), self.config.frame_skip)
            obs, reward, done, info = np.array(response.observation), response.reward[:], response.done[:], json.loads(str(response.info))
            info["dones"] = np.array(done)
            info["rewards"] = (np.array(reward).clip(min=-1, max=1) * self.config.hrl_reward_weights).sum(1)
            info["raw_rewards"] = np.array(reward).clip(min=-1, max=1)
            info["raw_dones"] = np.array(done)
            if not raw:
                obs = self._encode_obs(obs)

            return_tuple = (obs, (info["rewards"] - simulated_rewards)[0], done[0], info)
            simulated_rewards += info["rewards"]
            return return_tuple

        response = self.client.step(np.array(self.actions), self.config.frame_skip)
        self.actions = []
        obs, reward, done, info = np.array(response.observation), response.reward[:], response.done[:], json.loads(str(response.info))
        # obs[3] - obs[4].clip(max=1) means mask busy units
        # * np.where((obs[2])==2,0, (obs[2]))).flatten() means mask units not owned
        self.unit_location_mask = ((obs[3].clip(max=1) - obs[4].clip(max=1)) * np.where((obs[2])==2,0, (obs[2]))).flatten()
        self.target_unit_location_mask = ((obs[3].clip(max=1) - obs[4].clip(max=1)) * np.where((obs[2])==1,0, (obs[2]).clip(max=1))).flatten()
        self.action_mask = np.ones(self.action_space.nvec.sum())
        self.action_mask[0:self.action_space.nvec[0]] = self.unit_location_mask
        self.action_mask[-self.action_space.nvec[-1]:] = self.target_unit_location_mask
        if not raw:
            obs = self._encode_obs(obs)
        info["dones"] = np.array(done)
        info["rewards"] = (np.array(reward).clip(min=-1, max=1) * self.config.hrl_reward_weights).sum(1)
        info["raw_rewards"] = np.array(reward).clip(min=-1, max=1)
        info["raw_dones"] = np.array(done)
        if customize:
            return obs, reward, done, info
        return (obs, (info["rewards"] - simulated_rewards)[0], done[0], info)

class GlobalAgentCombinedRewardSelfPlayEnv(GlobalAgentEnv):
    def start_client(self):
        from ts import JNISelfPlayClient
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction, ResourceGatherRewardFunction, AttackRewardFunction, ProduceWorkerRewardFunction, ProduceBuildingRewardFunction, ProduceCombatUnitRewardFunction, CloserToEnemyBaseRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([
            WinLossRewardFunction(), 
            ResourceGatherRewardFunction(),  
            ProduceWorkerRewardFunction(),
            ProduceBuildingRewardFunction(),
            AttackRewardFunction(),
            ProduceCombatUnitRewardFunction(),
            CloserToEnemyBaseRewardFunction(),])
        self.opponent_action = None
        self.action_reverse_idxs = np.arange((self.config.height * self.config.width)-1,-1,-1).reshape(self.config.height,self.config.width).flatten()
        self.action_direction_reverse_idxs = np.array([2, 3, 0, 1])
        return JNISelfPlayClient(self.rfs, os.path.expanduser(self.config.microrts_path), self.config.map_path, self.real_utt)

    def transform_action(self, obs, action_mask): # the policy
        if self.opponent_action is None:
            return self.action_space.sample() # return a random action
        else:
            action = self.opponent_action
            # now we need to rotate the source and target unit position for 90 degrees
            action[0] = self.action_reverse_idxs[action[0]]
            action[-1] = self.action_reverse_idxs[action[-1]]
            # and rotate the direction of action parameters
            action[2] = self.action_direction_reverse_idxs[action[2]]
            action[3] = self.action_direction_reverse_idxs[action[3]]
            action[4] = self.action_direction_reverse_idxs[action[4]]
            action[5] = self.action_direction_reverse_idxs[action[5]]
            return action

    def get_unit_location_mask(self, player=1):
        if player == 1:
            return self.unit_location_mask
        else:
            return self.opponent_unit_location_mask

    def get_unit_action_mask(self, unit, player=1):
        if player == 1:
            if self.unit_location_mask.sum() == 0:
                return np.zeros(sum(self.action_space.nvec.tolist()[1:]))
            return np.array(self.client.getUnitActionMasks([[unit]]))[0]
        else:
            if self.opponent_unit_location_mask.sum() == 0:
                return np.zeros(sum(self.action_space.nvec.tolist()[1:]))
            # hack replacement for torch.split
            original_mask = np.array(self.client.getUnitActionMasks([[self.action_reverse_idxs[unit]]]))[0]
            logits_idxs = self.action_space.nvec.tolist()[1:]
            unit_action_mask = np.split(original_mask, [sum(logits_idxs[:i+1]) for i in range(len(logits_idxs)-1)])
            # rotate the direction of action parameters
            unit_action_mask[1] = unit_action_mask[1][self.action_direction_reverse_idxs]
            unit_action_mask[2] = unit_action_mask[2][self.action_direction_reverse_idxs]
            unit_action_mask[3] = unit_action_mask[3][self.action_direction_reverse_idxs]
            unit_action_mask[4] = unit_action_mask[4][self.action_direction_reverse_idxs]
            # rotate the target unit position for 90 degrees
            unit_action_mask[6] = unit_action_mask[6][self.action_reverse_idxs]
            return np.concatenate(unit_action_mask)

    def set_opponent_action(self, opponent_action):
        self.opponent_action = opponent_action

    def step(self, action, raw=False, customize=False):
        opponent_action = self.transform_action(self.opponent_obs, self.opponent_action_mask)
        response = self.client.step(np.array([action]), np.array([opponent_action]), self.config.frame_skip)
        obs, reward, done, info = np.array(response.observation), response.reward[:], response.done[:], json.loads(str(response.info))

        # opponent
        self.opponent_raw_obs = np.rot90(obs, 2, (1,2)).copy()
        owner_feature_map = self.opponent_raw_obs[2] - 1
        owner_feature_map[owner_feature_map==0]=2
        owner_feature_map[owner_feature_map==-1]=0
        self.opponent_raw_obs[2] = owner_feature_map
        self.opponent_unit_location_mask, self.opponent_action_mask = calculate_mask(self.opponent_raw_obs, self.action_space)
        self.opponent_obs = self._encode_obs(self.opponent_raw_obs)

        self.unit_location_mask, self.action_mask = calculate_mask(obs, self.action_space)
        if not raw:
            
            obs = self._encode_obs(obs)
        info["dones"] = np.array(done)
        info["rewards"] = np.array(reward).clip(min=-1, max=1)
        info["raw_rewards"] = np.array(reward).clip(min=-1, max=1)
        info["raw_dones"] = np.array(done)
        if customize:
            return obs, reward, done, info
        return obs, (np.array(reward).clip(min=-1, max=1) * self.config.reward_weight).sum(), done[0], info

    def reset(self, raw=False):
        raw_obs = super(GlobalAgentEnv, self).reset(True)
        self.unit_location_mask, self.action_mask = calculate_mask(raw_obs, self.action_space)
        self.opponent_raw_obs = np.rot90(raw_obs, 2, (1,2)).copy()
        owner_feature_map = self.opponent_raw_obs[2] - 1
        owner_feature_map[owner_feature_map==0]=2
        owner_feature_map[owner_feature_map==-1]=0
        self.opponent_raw_obs[2] = owner_feature_map
        self.opponent_unit_location_mask, self.opponent_action_mask = calculate_mask(self.opponent_raw_obs, self.action_space)
        self.opponent_obs = self._encode_obs(self.opponent_raw_obs)
        if raw:
            return raw_obs
        return self._encode_obs(raw_obs)

def calculate_mask(raw_obs, action_space):
    # obs[3] - obs[4].clip(max=1) means mask busy units
    # * np.where((obs[2])==2,0, (obs[2]))).flatten() means mask units not owned
    unit_location_mask = ((raw_obs[3].clip(max=1) - raw_obs[4].clip(max=1)) * np.where((raw_obs[2])==2,0, (raw_obs[2]))).flatten()
    target_unit_location_mask = ((raw_obs[3].clip(max=1) - raw_obs[4].clip(max=1)) * np.where((raw_obs[2])==1,0, (raw_obs[2]).clip(max=1))).flatten()
    action_mask = np.ones(action_space.nvec.sum())
    action_mask[0:action_space.nvec[0]] = unit_location_mask
    action_mask[-action_space.nvec[-1]:] = target_unit_location_mask
    return unit_location_mask, action_mask