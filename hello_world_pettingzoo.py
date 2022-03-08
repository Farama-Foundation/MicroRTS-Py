from pdb import set_trace

import numpy as np
from pettingzoo.test import api_test

from gym_microrts import microrts_ai
from gym_microrts.petting_zoo_api import \
    PettingZooMicroRTSGridModeSharedMemVecEnv

TEST_API = False


def main():
    # opponents = [microrts_ai.coacAI for _ in range(1)]
    opponents = []

    env = PettingZooMicroRTSGridModeSharedMemVecEnv(2, 0, ai2s=opponents)

    if TEST_API:
        api_test(env, num_cycles=10, verbose_progress=True)
    else:
        env.reset()

        actions = np.array([env.agent_action_space.sample(),
                           env.agent_action_space.sample()])
        actions = actions.reshape(2, env.width * env.height, env.action_dim)

        for agent in env.agent_iter():
            observation, reward, done, info = env.last()

            if done:
                break

            agent_id = env.agent_name_mapping[agent]
            action = actions[agent_id, :]
            env.step(action)

    # env.reset()

    # done = {agent: False for agent in env.agents}
    # live_agents = set(env.agents[:])
    # has_finished = set()
    # generated_agents = set()
    # accumulated_rewards = defaultdict(int)
    # for agent in env.agent_iter(env.num_agents * num_cycles):
    #     prev_observe, reward, done, info = env.last()
    #     env.step(action)

    #     if isinstance(env.observation_space(agent), gym.spaces.Box):
    #         assert env.observation_space(agent).dtype == prev_observe.dtype
    #     assert env.observation_space(agent).contains(prev_observe), \
    #         ("Out of bounds observation: " + str(prev_observe))


if __name__ == "__main__":
    main()
