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

class GlobalAttackBinaryEnv(GlobalAgentEnv):

    def start_client(self):
        from ts import JNIClient
        from ai.rewardfunction import AttackRewardFunction
        rf = AttackRewardFunction()
        return JNIClient(rf, os.path.expanduser(self.config.microrts_path), self.config.map_path)
