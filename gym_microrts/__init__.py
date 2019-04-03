from gym.envs.registration import register

register(
    id='Microrts-v0',
    entry_point='gym_microrts.envs:RandomAgentEnv',
)