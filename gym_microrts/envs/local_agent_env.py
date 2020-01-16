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

class LocalAgentEnv(BaseSingleAgentEnv):

    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import SimpleEvaluationRewardFunction
        rf = SimpleEvaluationRewardFunction()
        return JNIClient(rf, os.path.expanduser(self.config.microrts_path), self.config.map_path, self.config.window_size)
    
    def init_properties(self):
        self.config.height, self.config.width = self.config.window_size*2+1, self.config.window_size*2+1
        self.num_planes = [6, 6, 4, len(self.utt['unitTypes'])+2, 7]
        self.observation_space = spaces.Box(low=0.0,
            high=1.0,
            shape=(self.config.height * self.config.width,
                   sum(self.num_planes)),
                   dtype=np.int32)
        self.action_space = spaces.MultiDiscrete([
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
        return obs_planes
