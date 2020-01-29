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
from gym_microrts.envs.global_agent_env import GlobalAgentMiningEnv
from hilbertcurve.hilbertcurve import HilbertCurve

def build_hilbert_idx(order):
    base_length = 4
    scale = (2**(order-2))
    hilbert_curve = HilbertCurve(p=order, n=2)
    hilber_idx = np.zeros((base_length*scale)**2, dtype=np.int32)
    hilber_curve_path = np.zeros((base_length*scale)**2, dtype=np.int32)
    for i in range(base_length*scale):
        for j in range(base_length*scale):
            dist = hilbert_curve.distance_from_coordinates([i,j])
            hilber_idx[dist] = i*base_length*scale+j
            hilber_curve_path[i*base_length*scale+j] = dist

    hilber_curve_path = hilber_curve_path.reshape((base_length*scale, base_length*scale))
    return hilber_curve_path, hilber_idx

class GlobalAgentMiningHilbertEnv(GlobalAgentMiningEnv):

    def __init__(self, config=None):
        super().__init__(config=config)
        assert self.config.height == self.config.width, ("the map size must be of certain hilbert order")
        self.order = np.log2(self.config.height)
        assert np.abs(self.order - int(self.order)) < 0.00001, ("the map length must equal to 2^x for some integer x")
        self.order = int(self.order)
        self.hilber_curve_path, self.hilber_idx = build_hilbert_idx(self.order)

    def step(self, action, raw=False):
        observation, reward, done, info = super(GlobalAgentMiningEnv, self).step(action, raw)
        return observation[self.hilber_idx], reward, False, info
    def reset(self, raw=False):
        observation = super(GlobalAgentMiningEnv, self).reset(raw)
        return observation[self.hilber_idx]