
import os
import json
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

import gym
import gym_microrts
from gym_microrts import microrts_ai

import jpype
from jpype.imports import registerDomain
import jpype.imports
from jpype.types import JArray

class MicroRTSVecEnv:
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 150
    }
    """
    Create a baselines VecEnv environment from a gym3 environment.

    :param env: gym3 environment to adapt
    """

    def __init__(self,
        num_envs=2,
        max_steps=20000,
        render_theme=2,
        frame_skip=0,
        ai2s=[microrts_ai.passiveAI, microrts_ai.passiveAI],
        map_path="maps/10x10/basesTwoWorkers10x10.xml",
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0])):

        assert num_envs == len(ai2s), "for each environment, a microrts ai should be provided"
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.render_theme = render_theme
        self.frame_skip = frame_skip
        self.ai2s = ai2s
        self.map_path = map_path
        self.reward_weight = reward_weight

        # read map
        self.microrts_path = os.path.join(gym_microrts.__path__[0], 'microrts')
        root = ET.parse(os.path.join(self.microrts_path, self.map_path)).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))

        # launch the JVM
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            jars = [
                "microrts.jar", "Coac.jar", "Droplet.jar", "GRojoA3N.jar",
                "Izanagi.jar", "MixedBot.jar", "RojoBot.jar", "TiamatBot.jar", "UMSBot.jar" # "MindSeal.jar"
            ]
            for jar in jars:
                jpype.addClassPath(os.path.join(self.microrts_path, jar))
            jpype.startJVM(convertStrings=False)

        # start microrts client
        from rts.units import UnitTypeTable
        self.real_utt = UnitTypeTable()
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction, ResourceGatherRewardFunction, AttackRewardFunction, ProduceWorkerRewardFunction, ProduceBuildingRewardFunction, ProduceCombatUnitRewardFunction, CloserToEnemyBaseRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([
            WinLossRewardFunction(), 
            ResourceGatherRewardFunction(),  
            ProduceWorkerRewardFunction(),
            ProduceBuildingRewardFunction(),
            AttackRewardFunction(),
            ProduceCombatUnitRewardFunction(),
            # CloserToEnemyBaseRewardFunction(),
        ])
        self.start_client()

        # computed properties
        # [num_planes_hp(5), num_planes_resources(5), num_planes_player(5), 
        # num_planes_unit_type(z), num_planes_unit_action(6)]
        self.num_planes = [5, 5, 3, len(self.utt['unitTypes'])+1, 6]
        self.observation_space = gym.spaces.Box(low=0.0,
            high=1.0,
            shape=(self.height, self.width,
                    sum(self.num_planes)),
                    dtype=np.int32)
        self.action_space = gym.spaces.MultiDiscrete([
            self.height * self.width,
            6, 4, 4, 4, 4,
            len(self.utt['unitTypes']),
            7 * 7
        ])

    def start_client(self):
        from ts import JNIVecClient
        from ai.core import AI
        self.vec_client = JNIVecClient(
            self.num_envs,
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_path,
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt
        )
        # get the unit type table
        self.utt = json.loads(str(self.vec_client.clients[0].sendUTT()))

    def reset(self):
        responses = self.vec_client.reset([0 for _ in range(self.num_envs)])
        raw_obs, reward, done, info = np.array(responses.observation), np.array(responses.reward), np.array(responses.done), {}
        obs = []
        for ro in raw_obs:
            obs += [self._encode_obs(ro)]
        return np.array(obs)

    def _encode_obs(self, obs):
        obs = obs.reshape(len(obs), -1).clip(0, np.array([self.num_planes]).T-1)
        obs_planes = np.zeros((self.height * self.width, 
                               sum(self.num_planes)), dtype=np.int)
        obs_planes[np.arange(len(obs_planes)),obs[0]] = 1

        for i in range(1, len(self.num_planes)):
            obs_planes[np.arange(len(obs_planes)),obs[i]+sum(self.num_planes[:i])] = 1
        return obs_planes.reshape(self.height, self.width, -1)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        responses = self.vec_client.step(self.actions, [0 for _ in range(self.num_envs)])
        raw_obs, reward, done, info = np.array(responses.observation), np.array(responses.reward), np.array(responses.done), [{} for _ in range(self.num_envs)]
        obs = []
        for ro in raw_obs:
            obs += [self._encode_obs(ro)]
        infos = [{"raw_rewards": item} for item in reward]
        return np.array(obs), reward @ self.reward_weight, done[:,0], infos

    def step(self, ac):
        self.step_async(ac)
        return self.step_wait()

    def getattr_depth_check(self, name, already_found):
        """Check if an attribute reference is being hidden in a recursive call to __getattr__

        :param name: (str) name of attribute to check for
        :param already_found: (bool) whether this attribute has already been found in a wrapper
        :return: (str or None) name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return "{0}.{1}".format(type(self).__module__, type(self).__name__)
        else:
            return None

    def render(self, mode="human"):
        if mode == "human":
            self.vec_client.clients[0].render(False)
        elif mode == 'rgb_array':
            bytes_array = np.array(self.vec_client.clients[0].render(True))
            image = Image.frombytes("RGB", (640, 640), bytes_array)
            return np.array(image)

    def close(self):
        pass

class MicroRTSGridModeVecEnv(MicroRTSVecEnv):

    def __init__(self, num_selfplay_envs, num_bot_envs, **kwargs):
        self.num_selfplay_envs = num_selfplay_envs
        self.num_bot_envs = num_bot_envs
        kwargs['num_envs'] = num_bot_envs
        super().__init__(**kwargs)
        self.num_envs = num_selfplay_envs + num_bot_envs

    def start_client(self):
        from ts import JNIGridnetVecClient
        from ai.core import AI
        self.vec_client = JNIGridnetVecClient(
            self.num_selfplay_envs,
            self.num_envs,
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_path,
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt
        )
        self.render_client = self.vec_client.selfPlayClients[0] if len(self.vec_client.selfPlayClients) > 0 else self.vec_client.clients[0]
        # get the unit type table
        self.utt = json.loads(str(self.render_client.sendUTT()))

    def step_wait(self):
        responses = self.vec_client.gameStep(self.actions, [0 for _ in range(self.num_envs)])
        raw_obs, reward, done, info = np.array(responses.observation), np.array(responses.reward), np.array(responses.done), [{} for _ in range(self.num_selfplay_envs+self.num_envs)]
        obs = []
        for ro in raw_obs:
            obs += [self._encode_obs(ro)]
        infos = [{"raw_rewards": item} for item in reward]
        return np.array(obs), reward @ self.reward_weight, done[:,0], infos

    def render(self, mode="human"):
        if mode == "human":
            self.render_client.render(False)
        elif mode == 'rgb_array':
            bytes_array = np.array(self.render_client.render(True))
            image = Image.frombytes("RGB", (640, 640), bytes_array)
            return np.array(image)