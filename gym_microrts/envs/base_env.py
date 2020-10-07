import gym
import socket
import numpy as np
import json
from subprocess import Popen, PIPE
import os
from typing import List, Tuple
from dacite import from_dict
import gym_microrts
from gym_microrts.types import MicrortsMessage, Config
from gym import error, spaces, utils
import xml.etree.ElementTree as ET
from gym.utils import seeding
from PIL import Image
import io
from pathlib import Path

import jpype
from jpype.imports import registerDomain
import jpype.imports
from jpype.types import *

class BaseSingleAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, config=None):
        if config:
            self.init(config)
    
    def init(self, config: Config):
        """
        if `config.microrts_path` is set, then the script will automatically try 
        to launch a microrts client instance. Otherwise you need to set the 
        `config.height` and `config.this script will just wait
        to listen to the microrts client
        """
        self.config = config
        root = ET.parse(os.path.expanduser(os.path.join(self.config.microrts_path, self.config.map_path))).getroot()
        self.config.height, self.config.width = int(root.get("height")), int(root.get("width"))
        self.running_first_episode = True
        self.closed = False

        # Launch the JVM
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            jpype.addClassPath(os.path.join(Path(gym_microrts.__path__[0]).parent, 'microrts', "microrts.jar"))
            jpype.addClassPath(os.path.join(Path(gym_microrts.__path__[0]).parent, 'microrts', "Coac.jar"))
            jpype.startJVM(convertStrings=False)

        from rts.units import UnitTypeTable
        self.real_utt = UnitTypeTable()
        self.client = self.start_client()
        
        # get the unit type table
        self.utt = json.loads(str(self.client.sendUTT()))
        
        # computed properties
        self.init_properties()

    def init_properties(self):
        raise NotImplementedError
        
    def start_client(self):
        raise NotImplementedError

    def step(self, action, raw=False):
        action = np.array([action])
        response = self.client.step(action, self.config.frame_skip)
        if raw:
            return np.array(response.observation), response.reward[:], response.done[:], json.loads(str(response.info))
        return self._encode_obs(np.array(response.observation)), response.reward[:], response.done[:], json.loads(str(response.info))

    def reset(self, raw=False):
        response = self.client.reset()
        if raw:
            return np.array(response.observation)
        return self._encode_obs(np.array(response.observation))

    def render(self, mode='human'):
        if mode=='human':
            self.client.render(False)
        elif mode == 'rgb_array':
            bytes_array = np.array(self.client.render(True))
            image = Image.frombytes("RGB", (640, 640), bytes_array)
            return np.array(image)

    def close(self):
        if jpype._jpype.isStarted():
            self.client.close()
            jpype.shutdownJVM()
    
    def _encode_obs(self, observation: List):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
