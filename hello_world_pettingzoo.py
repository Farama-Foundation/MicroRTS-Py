import numpy as np
from pettingzoo.test import api_test

from gym_microrts import microrts_ai
from gym_microrts.petting_zoo_api import \
    PettingZooMicroRTSGridModeSharedMemVecEnv

TEST_API = False


def main():
    opponents = [microrts_ai.coacAI for _ in range(1)]

    env = PettingZooMicroRTSGridModeSharedMemVecEnv(2, 1, ai2s=opponents)

    if TEST_API:
        api_test(env, num_cycles=10, verbose_progress=True)

    env.reset()
    actions = np.array([
        env.agent_action_space.sample(),
        env.agent_action_space.sample(),
        env.agent_action_space.sample()
    ])
    actions = actions.reshape(3, env.width * env.height, env.action_dim)

    for agent in env.agent_iter():
        observation, reward, done, info = env.last()

        if done:
            break

        agent_id = env.agent_name_mapping[agent]
        action = actions[agent_id, :]
        env.step(action)


if __name__ == "__main__":
    main()
