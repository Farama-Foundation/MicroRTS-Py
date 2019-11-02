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

def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port

class BaseSingleAgentEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    conn = None

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
        
        # computed properties
        self.init_properties()
        
        # start the microrts java client
        if self.config.microrts_path and not self.config.microrts_repo_path:
            self.start_client()

        # start hte microrts server
        self.start_server()

    def init_properties(self):
        raise NotImplementedError

    def start_client(self):
        commands = [
            "java",
            "-cp",
            os.path.expanduser(os.path.join(self.config.microrts_path, "microrts.jar")),
            "tests.sockets.RunClient",
            "--server-port",
            str(self.config.client_port),
            "--map",
            os.path.expanduser(os.path.join(self.config.microrts_path, self.config.map_path)),
            "--window-size",
            str(self.config.window_size)
        ]
        if self.config.render:
            commands += ["--render"]
        if self.config.ai1_type:
            commands += ["--ai1-type", self.config.ai1_type]
        if self.config.ai2_type:
            commands += ["--ai2-type", self.config.ai2_type]
        if self.config.evaluation_filename:
            commands += ["--evaluation-filename", os.path.join(os.getcwd(), self.config.evaluation_filename)]
        print(commands)
        self.process = Popen(
            commands,
            stdout=PIPE,
            stderr=PIPE)

    def start_server(self):
        print("Waiting for connection from the MicroRTS JAVA client")
        s = socket.socket()
        s.bind((self.config.client_ip, self.config.client_port))
        s.listen(5)
        self.conn, addr = s.accept()
        print('Got connection from', addr)
        print(self._send_msg("[]"))
        print(self._send_msg("[]"))

    def print_microrts_outputs(self):
        stdout, stderr = self.process.communicate()
        print("The following is outputed by stdout of microrts client")
        print(stdout.decode("utf-8"))
        print("The following is outputed by stderr of microrts client")
        print(stderr.decode("utf-8"))

    def step(self, action, raw=False):
        action = np.array([action])
        mm = from_dict(data_class=MicrortsMessage, data=json.loads(self._send_msg(str(action.tolist()))))
        if raw:
            return mm.observation, mm.reward, mm.done, mm.info
        return self._encode_obs(mm.observation), mm.reward, mm.done, mm.info

    def reset(self, raw=False):
        # get the unit table and the computing budget
        self._send_msg("done")
        mm = from_dict(data_class=MicrortsMessage, data=json.loads(self._send_msg("[]")))
        if raw:
            return mm.observation, mm.reward, mm.done, mm.info
        return self._encode_obs(mm.observation)

    def render(self, mode='human'):
        pass

    def close(self):
        self._send_msg("finished")
        # send a dummy action
        action = np.array([[0,0,0,0]])
        self.conn.send(('%s\n' % str(action.tolist())).encode('utf-8'))

    def _send_msg(self, msg: str):
        try:
            self.conn.send(('%s\n' % msg).encode('utf-8'))
        except Exception as err:
            print("An error has occured: ", err)
            self.print_microrts_outputs()
        return self.conn.recv(4096).decode('utf-8')
    
    def _encode_obs(self, observation: List):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
