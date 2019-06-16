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

class RandomAgentEnv(gym.Env):
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
        self.compute_properties()
        
        # start the microrts java client
        if self.config.microrts_path and not self.config.microrts_repo_path:
            self.start_client()

        # start hte microrts server
        self.start_server()

    def compute_properties(self):
        if self.config.auto_port:
            self.config.client_port = get_free_tcp_port()
        if self.config.microrts_path:
            root = ET.parse(os.path.join(self.config.microrts_path, self.config.map_path)).getroot()
            self.config.height, self.config.width = int(root.get("height")), int(root.get("width"))
        elif self.config.microrts_repo_path:
            root = ET.parse(os.path.join(self.config.microrts_repo_path, self.config.map_path)).getroot()
            self.config.height, self.config.width = int(root.get("height")), int(root.get("width"))
        else:
            raise Exception("Couldn't read height and width of the map. Set either microrts_repo_path or microrts_path")
        self.num_classes = 7
        self.num_feature_maps = 5
        self.running_first_episode = True
        self.observation_space = spaces.Box(low=-1.0,
            high=1.0,
            shape=(self.num_feature_maps, self.config.height * self.config.width, self.num_classes),
            dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.config.height, self.config.width, 4, 4])
        self.__t = 0

    def start_client(self):
        commands = [
            "java",
            "-cp",
            os.path.join(self.config.microrts_path, "microrts.jar"),
            "tests.sockets.RunClient",
            "--server-port",
            str(self.config.client_port),
            "--map",
            os.path.join(self.config.microrts_path, self.config.map_path),
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
        f = open("/dev/null", "w")
        print(commands)
        self.process = Popen(
            commands,
            stdout=f,
            stderr=f)

    def start_server(self):
        print("Waiting for connection from the MicroRTS JAVA client")
        s = socket.socket()
        s.bind((self.config.client_ip, self.config.client_port))
        s.listen(5)
        self.conn, addr = s.accept()
        print('Got connection from', addr)
        try:
            print(self._send_msg("[]"))
            print(self._send_msg("[]"))
        except Exception as err:
            print("An error has occured: ", err)
            if self.config.microrts_path:
                print("The following is outputed by stdout of microrts client")
                for line in self.process.stdout:
                    print(line.decode("utf-8"))
                print("The following is outputed by stderr of microrts client")
                for line in self.process.stderr:
                    print(line.decode("utf-8"))

    def step(self, action, raw=False):
        action = np.array([action])
        mm = from_dict(data_class=MicrortsMessage, data=json.loads(self._send_msg(str(action.tolist()))))
        if self.__t >= self.config.maximum_t:
            mm.done = True
            self.__t = 0
        self.__t += 1
        if raw:
            return mm.observation, mm.reward, mm.done, mm.info
        return self._encode_obs(mm.observation), mm.reward, mm.done, mm.info

    def reset(self, raw=False):
        # get the unit table and the computing budget
        if self.running_first_episode:
            self.running_first_episode = False
        else:
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
        self.conn.send(('%s\n' % msg).encode('utf-8'))
        return self.conn.recv(4096).decode('utf-8')
    
    def _encode_obs(self, observation: List):
        observation = np.array(observation)
        new_obs = np.zeros((self.num_feature_maps, self.config.height * self.config.width, self.num_classes))
        reshaped_obs = observation.reshape((self.num_feature_maps, self.config.height * self.config.width))
        reshaped_obs[2] += 1
        reshaped_obs[3] += 1
        reshaped_obs[reshaped_obs >= self.num_classes] = self.num_classes - 1
        for i in range(len(reshaped_obs)):
            new_obs[i][np.arange(len(reshaped_obs[i])), reshaped_obs[i]] = 1
        return new_obs
