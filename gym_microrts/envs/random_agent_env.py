import gym
import socket
import numpy as np
import json
import os
from typing import List, Tuple
from dacite import from_dict
from gym_microrts.types import MicrortsMessage, Config
from gym import error, spaces, utils
import xml.etree.ElementTree as ET
from gym.utils import seeding

class RandomAgentEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    conn = None

    def __init__(self):
        pass
        
    def init(self, config: Config):
        self.config = config
        self.dimension_x, self.dimension_y = self.__map_dimension(
            config.microrts_path, config.map_path)
        self.num_classes = 7
        self.num_feature_maps = 5
        self.running_first_episode = True
        self.observation_space = spaces.Box(low=-1.0,
            high=1.0,
            shape=(self.num_feature_maps, self.dimension_x * self.dimension_y, self.num_classes),
            dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.dimension_x, self.dimension_y, 4, 4])
        self.__t = 0
        print("Waiting for connection from the MicroRTS JAVA client")
        s = socket.socket()
        s.bind((config.client_ip, config.client_port))
        s.listen(5)
        self.conn, addr = s.accept()
        print('Got connection from', addr)
        print(self._send_msg("[]"))
        print(self._send_msg("[]"))
    
    def __map_dimension(self, microrts_path: str, map_path: str) -> Tuple[int]:
        whole_map_path = os.path.join(microrts_path, map_path)
        print(whole_map_path)
        root = ET.parse(whole_map_path).getroot()
        return root.get("height"), root.get("width")
        

    def step(self, action):
        action = np.array([action])
        mm = from_dict(data_class=MicrortsMessage, data=json.loads(self._send_msg(str(action.tolist()))))
        if self.__t >= self.config.maximum_t:
            mm.done = True
            self.__t = 0
        self.__t += 1
        return self._encode_obs(mm.observation), mm.reward, mm.done, mm.info

    def reset(self):
        # get the unit table and the computing budget
        if self.running_first_episode:
            self.running_first_episode = False
        else:
            self._send_msg("done")
        mm = from_dict(data_class=MicrortsMessage, data=json.loads(self._send_msg("[]")))
        return self._encode_obs(mm.observation)

    def render(self, mode='human'):
        pass

    def close(self):
        self._send_msg("finished")
        # send a dummy action
        action = np.array([[0,0,0,0]])
        self.conn.send(('%s\n' % str(action.tolist())).encode('utf-8'))

    def _send_msg(self, msg: str):
        self.conn.send(('%s\n' % msg).encode('utf-8'))
        return self.conn.recv(4096).decode('utf-8')
    
    def _encode_obs(self, observation: List):
        observation = np.array(observation)
        new_obs = np.zeros((self.num_feature_maps, self.dimension_x * self.dimension_y, self.num_classes))
        reshaped_obs = observation.reshape((self.num_feature_maps, self.dimension_x * self.dimension_y))
        reshaped_obs[2] += 1
        reshaped_obs[3] += 1
        reshaped_obs[reshaped_obs >= self.num_classes] = self.num_classes - 1
        for i in range(len(reshaped_obs)):
            new_obs[i][np.arange(len(reshaped_obs[i])), reshaped_obs[i]] = 1
        return new_obs
