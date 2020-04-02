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
from PIL import Image
import io

import jpype
from jpype.imports import registerDomain
import jpype.imports
from jpype.types import *

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class ParamOpEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def __init__(self, map_index=0, num_workers=1, config=None):
        self.map_index = map_index
        self.num_workers = num_workers
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
            jpype.addClassPath(os.path.expanduser(os.path.join(self.config.microrts_path, "microrts.jar")))
            jpype.startJVM(convertStrings=False)

        self.client = self.start_client()
        
        # computed properties
        self.init_properties()

    def init_properties(self):
        dummy_number = 10
        self.dummy_obs = np.ones(dummy_number)
        self.observation_space = spaces.Box(low=0.0,
            high=1.0,
            shape=(dummy_number,),)
        self.action_space = spaces.Box(low=0.0,
            high=1.0,
            shape=(6,))

    def start_client(self):
        from ts import PlayoutPolicyOptimization
        return PlayoutPolicyOptimization()

    def step(self, action, raw=False):
        response = self.client.computeRandomAIWinrate(JArray(JDouble)(action), self.map_index, self.num_workers, os.path.expanduser(self.config.microrts_path))
        return self.dummy_obs, response, False, {}

    def reset(self, raw=False):
        return self.dummy_obs

    def close(self):
        self.client.close()
        jpype.shutdownJVM()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
