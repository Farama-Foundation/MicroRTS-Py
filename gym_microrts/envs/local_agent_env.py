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

from gym_microrts.envs.base_env import BaseSingleAgentEnv, get_free_tcp_port

class LocalAgentEnv(BaseSingleAgentEnv):

    def init_properties(self):
        if self.config.auto_port:
            self.config.client_port = get_free_tcp_port()
        self.config.height, self.config.width = self.config.window_size*2+1, self.config.window_size*2+1
        self.num_classes = 8
        self.num_feature_maps = 5
        self.running_first_episode = True
        self.observation_space = spaces.Box(low=-1.0,
            high=1.0,
            shape=(self.num_feature_maps, self.config.height * self.config.width, self.num_classes),
            dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([4, 4])
    
    def _encode_obs(self, observation: List):
        observation = np.array(observation)
        new_obs = np.zeros((self.num_feature_maps, self.config.height * self.config.width, self.num_classes))
        reshaped_obs = observation.reshape((self.num_feature_maps, self.config.height * self.config.width))
        reshaped_obs[reshaped_obs >= self.num_classes] = self.num_classes - 1
        for i in range(len(reshaped_obs)):
            new_obs[i][np.arange(len(reshaped_obs[i])), reshaped_obs[i]] = 1
        return new_obs
