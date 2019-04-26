import gym
import socket
import numpy as np
import json
from dacite import from_dict
from gym_microrts.types import MicrortsMessage
from gym import error, spaces, utils
from gym.utils import seeding

class RandomAgentEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    conn = None

    def __init__(self, ip_address: str='', port: int=9898):
        print("Waiting for connection from the MicroRTS JAVA client")
        s = socket.socket()
        s.bind((ip_address, port))
        s.listen(5)
        self.conn, addr = s.accept()
        self.running_first_episode = True
        print('Got connection from', addr)
        print(self._send_msg("[]"))
        print(self._send_msg("[]"))

    def step(self, action):
        self._send_msg(str(action))
        for _ in range(8):
            self._send_msg("[]")
        mm = from_dict(data_class=MicrortsMessage, data=json.loads(self._send_msg('[]')))
        return np.array(mm.observation), mm.reward, mm.done, mm.info

    def reset(self):
        # get the unit table and the computing budget
        if self.running_first_episode:
            self.running_first_episode = False
        else:
            print(self._send_msg("done"))
        mm = from_dict(data_class=MicrortsMessage, data=json.loads(self._send_msg("[]")))
        return np.array(mm.observation)

    def render(self, mode='human'):
        pass

    def close(self):
        self._send_msg("finished")

    def _send_msg(self, msg: str):
        self.conn.send(('%s\n' % msg).encode('utf-8'))
        return self.conn.recv(4096).decode('utf-8')
