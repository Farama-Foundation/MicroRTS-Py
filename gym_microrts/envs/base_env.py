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
import struct
import mmap

def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port

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
        with open("/home/costa/Documents/work/go/src/github.com/vwxyzjn/gym-microrts/unix/t", "wb") as f:
            f.seek((1024 * 100) - 1)
            f.write(b"\0")

        with open("/home/costa/Documents/work/go/src/github.com/vwxyzjn/gym-microrts/unix/t", "r+b") as f:
            self.mm = mmap.mmap(f.fileno(), length=0)
        
        self.config = config
        if self.config.auto_port:
            self.config.client_port = get_free_tcp_port()
        if self.config.microrts_path:
            root = ET.parse(os.path.expanduser(os.path.join(self.config.microrts_path, self.config.map_path))).getroot()
            self.config.height, self.config.width = int(root.get("height")), int(root.get("width"))
        elif self.config.microrts_repo_path:
            root = ET.parse(os.path.expanduser(os.path.join(self.config.microrts_repo_path, self.config.map_path))).getroot()
            self.config.height, self.config.width = int(root.get("height")), int(root.get("width"))
        else:
            raise Exception("Couldn't read height and width of the map. Set either microrts_repo_path or microrts_path")
        self.running_first_episode = True
        self.closed = False
        
        # start the microrts java client
        if self.config.microrts_path and not self.config.microrts_repo_path:
            self.start_client()

        # start hte microrts server
        self.start_server()
        
        # computed properties
        self.init_properties()

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
        if self.config.ai1_type:
            commands += ["--ai1-type", self.config.ai1_type]
        if self.config.ai2_type:
            commands += ["--ai2-type", self.config.ai2_type]
        if self.config.evaluation_filename:
            commands += ["--evaluation-filename", os.path.join(os.getcwd(), self.config.evaluation_filename)]
        commands += ["--unix-socket-path", '/home/costa/Documents/work/go/src/github.com/vwxyzjn/gym-microrts/unix/u']
        print(commands)
        self.process = Popen(
            commands,
            stdout=PIPE,
            stderr=PIPE)

    def start_server(self):
        print("unixsocket")
        server_address = '/home/costa/Documents/work/go/src/github.com/vwxyzjn/gym-microrts/unix/u'

        # Make sure the socket does not already exist
        try:
            os.unlink(server_address)
        except OSError:
            if os.path.exists(server_address):
                raise
        
        # Create a UDS socket
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        
        # Bind the socket to the address
        print('starting up on {}'.format(server_address))
        s.bind(server_address)
        
        # Listen for incoming connections
        s.listen(5)
        
        
        print("Waiting for connection from the MicroRTS JAVA client")
        s.listen(5)
        self.conn, addr = s.accept()
        print('Got connection from', addr)
        print(self._send_msg("[]"))
        self.utt = self._send_msg("[]")
        print(self.utt)
        self.utt = json.loads(self.utt[3:])
        
    # def start_server(self):
    #     print("Waiting for connection from the MicroRTS JAVA client")
    #     s = socket.socket()
    #     s.bind((self.config.client_ip, self.config.client_port))
    #     s.listen(5)
    #     self.conn, addr = s.accept()
    #     print('Got connection from', addr)
    #     print(self._send_msg("[]"))
    #     self.utt = self._send_msg("[]")
    #     print(self.utt)
    #     self.utt = json.loads(self.utt[3:])

    def print_microrts_outputs(self):
        stdout, stderr = self.process.communicate()
        print("The following is outputed by stdout of microrts client")
        print(stdout.decode("utf-8"))
        print("The following is outputed by stderr of microrts client")
        print(stderr.decode("utf-8"))

    def step(self, action, raw=False):
        action = np.append(action, [self.config.frame_skip])
        action = np.array([action])
        temp = self._send_msg(str(action.tolist()))
        #print(temp)
        mm = from_dict(data_class=MicrortsMessage, data=json.loads(temp))
        mm.observation = np.array(mm.observation).transpose(0, 2, 1)
        if raw:
            return mm.observation, mm.reward, mm.done, mm.info
        return self._encode_obs(mm.observation), mm.reward, mm.done, mm.info

    def reset(self, raw=False):
        # get the unit table and the computing budget
        #self._send_msg("reset")
        mm = from_dict(data_class=MicrortsMessage, data=json.loads(self._send_msg("reset")))
        mm.observation = np.array(mm.observation).transpose(0, 2, 1)
        if raw:
            return mm.observation, mm.reward, mm.done, mm.info
        return self._encode_obs(mm.observation)

    def render(self, mode='human'):
        # if mode=='human':
        #     self._send_msg("render")
        # elif mode == 'rgb_array':
        self.conn.send(('%s\n' % "render").encode('utf-8'))
        self.recvall(4)
        #print("haha", self.recvall(4))
        self.mm.seek(0)
        length = struct.unpack('>i', self.mm.read(4))[0]
        bytes_array = self.mm.read(length)
        image = Image.open(io.BytesIO(bytes_array))
        return np.array(image)
        
    # def render(self, mode='human'):
    #     if mode=='human':
    #         self._send_msg("render")
    #     elif mode == 'rgb_array':
    #         self.conn.send(('%s\n' % "render").encode('utf-8'))
    #         length = struct.unpack('>i', self.recvall(4))[0]
    #         bytes_array = self.conn.recv(length)
    #         image = Image.open(io.BytesIO(bytes_array))
    #         return np.array(image)

    def close(self):
        if not self.closed:
            print("close")
            self._send_msg("finished")
            # send a dummy action
            action = np.array([[0,0,0,0]])
            self.conn.send(('%s\n' % str(action.tolist())).encode('utf-8'))
            self.closed = True

    def _send_msg(self, msg: str, buffsize: int=4096):
        try:
            self.conn.send(('%s\n' % msg).encode('utf-8'))
        except Exception as err:
            print("An error has occured: ", err)
            self.print_microrts_outputs()
            
        # receive message length
        #length = struct.unpack('>i', self.conn.recv(4))[0]
        return self.conn.recv(buffsize).decode('utf-8')
    
    def _encode_obs(self, observation: List):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def recvall(self, n):
        # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data
