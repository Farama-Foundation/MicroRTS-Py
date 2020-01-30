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
            jpype.addClassPath(os.path.expanduser(os.path.join(self.config.microrts_path, "microrts.jar")))
            jpype.startJVM()

        self.client = self.start_client()
        
        # get the unit type table
        self.utt = json.loads(self.client.sendUTT()) 
        
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
            return convert3DJarrayToNumpy(response.observation), response.reward[:], response.done[:], json.loads(response.info)
        return self._encode_obs(convert3DJarrayToNumpy(response.observation)), response.reward[:], response.done[:], json.loads(response.info)

    def reset(self, raw=False):
        response = self.client.reset()
        if raw:
            return convert3DJarrayToNumpy(response.observation)
        return self._encode_obs(convert3DJarrayToNumpy(response.observation))

    def render(self, mode='human'):
        if mode=='human':
            self.client.render(False)
        elif mode == 'rgb_array':
            bytes_array = self.client.render(True)[:]
            image = Image.frombytes("RGB", (640, 640), bytes_array)
            return np.array(image)

    def close(self):
        self.client.close()
        jpype.shutdownJVM()
    
    def _encode_obs(self, observation: List):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

def convert3DJarrayToNumpy(jArray):
    # get shape
    arr_shape = (len(jArray),)
    temp_array = jArray[0]
    while hasattr(temp_array, '__len__'):
        arr_shape += (len(temp_array),)
        temp_array = temp_array[0]
    arr_type = type(temp_array)
    # transfer data
    resultArray = np.empty(arr_shape, dtype=arr_type)
    for ix in range(arr_shape[0]):
        for i,cols in enumerate(jArray[ix][:]):
            resultArray[ix][i,:] = cols[:]
    return resultArray
