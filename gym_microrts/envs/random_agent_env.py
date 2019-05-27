import gym
import socket
import numpy as np
import json
from typing import List
from dacite import from_dict
from gym_microrts.types import MicrortsMessage
from gym import error, spaces, utils
from gym.utils import seeding

class RandomAgentEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    conn = None

    def __init__(self, ip_address: str='', port: int=9898, dimension_x: int=16, dimension_y: int=16):
        self.dimension_x = dimension_x
        self.dimension_y = dimension_y
        self.observation_space = spaces.Box(low=-1.0,
            high=1.0,
            shape=(4, self.dimension_x * self.dimension_y, 7),
            dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.dimension_x, self.dimension_y, 4, 4])
        self.t = 0
        self.max_t = 2000
        
        print("Waiting for connection from the MicroRTS JAVA client")
        s = socket.socket()
        s.bind((ip_address, port))
        s.listen(5)
        self.conn, addr = s.accept()
        self.running_first_episode = True
        print('Got connection from', addr)
        print(self._send_msg("[]"))
        print(self._send_msg("[]"))
        
    def set_map_dimension(self, dimension_x: int, dimension_y: int):
        self.dimension_x = dimension_x
        self.dimension_y = dimension_y
        self.observation_space = spaces.Box(low=-1.0,
            high=1.0,
            shape=(4, self.dimension_x * self.dimension_y, 7),
            dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.dimension_x, self.dimension_y, 4, 4])
        

    def step(self, action):
        action = np.array([action])
        mm = from_dict(data_class=MicrortsMessage, data=json.loads(self._send_msg(str(action.tolist()))))
        if self.t >= self.max_t:
            mm.done = True
            self.t = 0
        self.t += 1
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

    def _send_msg(self, msg: str):
        self.conn.send(('%s\n' % msg).encode('utf-8'))
        return self.conn.recv(4096).decode('utf-8')
    
    def _encode_obs(self, observation: List):
        observation = np.array(observation)
        num_classes = 7
        new_obs = np.zeros((4, self.dimension_x * self.dimension_y, num_classes))
        reshaped_obs = observation.reshape((4, self.dimension_x * self.dimension_y))
        reshaped_obs[2] += 1
        reshaped_obs[3] += 1
        reshaped_obs[reshaped_obs >= num_classes] = num_classes - 1
        for i in range(len(reshaped_obs)):
            new_obs[i][np.arange(len(reshaped_obs[i])), reshaped_obs[i]] = 1
        return new_obs
