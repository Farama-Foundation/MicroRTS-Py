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
from gym_microrts.envs.global_agent_env import GlobalAgentEnv

class GlobalAgentMiningEnv(GlobalAgentEnv):
    """
    Always return False for the terminal signal of the game
    """

    def step(self, action, raw=False):
        observation, reward, done, info = super(GlobalAgentMiningEnv, self).step(action, raw)
        return observation, reward, False, info
