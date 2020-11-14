import gym
import numpy as np
import json
from subprocess import Popen, PIPE
import os
import gym_microrts
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
from gym_microrts import microrts_ai

class BaseSingleAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self,
        render_theme=2,
        frame_skip=0, 
        ai2=microrts_ai.passiveAI,
        map_path="maps/10x10/basesTwoWorkers10x10.xml"):

        self.render_theme = render_theme
        self.frame_skip = frame_skip
        self.ai2 = ai2
        self.map_path = map_path

        self.microrts_path = os.path.join(gym_microrts.__path__[0], 'microrts')
        root = ET.parse(os.path.join(self.microrts_path, self.map_path)).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))

        # Launch the JVM
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            jpype.addClassPath(os.path.join(self.microrts_path, "microrts.jar"))
            jpype.addClassPath(os.path.join(self.microrts_path, "Coac.jar"))
            jpype.startJVM(convertStrings=False)

        from rts.units import UnitTypeTable
        self.real_utt = UnitTypeTable()
        self.client = self.start_client()
        self.client.renderTheme = self.render_theme

        # get the unit type table
        self.utt = json.loads(str(self.client.sendUTT()))
        
        # computed properties
        self.init_properties()

    def init_properties(self):
        raise NotImplementedError
        
    def start_client(self):
        raise NotImplementedError

    def step(self, action, raw=False):
        raise NotImplementedError

    def reset(self, raw=False):
        raise NotImplementedError

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
    
    def _encode_obs(self, observation):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
