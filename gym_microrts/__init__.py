__version__ = "0.1.4"

from copy import deepcopy
from gym.envs.registration import register
import gym
import uuid
from .types import Config
import numpy as np
from .v1 import envs as v1_envs
from .v2 import envs as v2_envs
from .v3 import envs as v3_envs
from .v4 import envs as v4_envs

# enable repeated experiments
# https://github.com/openai/gym/issues/1172
V0NAME = 'Microrts-v0'
if V0NAME not in gym.envs.registry.env_specs:
    register(
        id=V0NAME,
        entry_point='gym_microrts.envs:GlobalAgentEnv',
    )
    
    envs = []
    envs += v1_envs
    envs += v2_envs
    envs += v3_envs
    envs += v4_envs

    for env in envs:
        # Regular
        register(
            env['id'],
            entry_point=env['entry_point'],
            kwargs=env['kwargs'],
            max_episode_steps=env['max_episode_steps'])

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
