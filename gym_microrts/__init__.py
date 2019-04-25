from gym.envs.registration import register
import gym

# enable repeated experiments
# https://github.com/openai/gym/issues/1172 
V0NAME = 'Microrts-v0'
if V0NAME in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[V0NAME]

register(
    id=V0NAME,
    entry_point='gym_microrts.envs:RandomAgentEnv',
)