import numpy as np
from pettingzoo.test import api_test

from gym_microrts.petting_zoo_api import PettingZooMicroRTSGridModeSharedMemVecEnv

TEST_API = False


if __name__ == "__main__":
    # opponents = [microrts_ai.coacAI for _ in range(1)]
    opponents = []

    env = PettingZooMicroRTSGridModeSharedMemVecEnv(2, 0, ai2s=opponents)

    if TEST_API:
        api_test(env, num_cycles=10, verbose_progress=True)
    else:
        env.reset()
        env.render()

        for episode in range(100):
            actions = np.array([env.agent_action_space.sample(), env.agent_action_space.sample()])
            for agent in env.agent_iter():
                env.render()
                observation, reward, done, info = env.last()
                # print(agent, done)
                if done:
                    env.reset()
                    break
                agent_id = env.agent_name_mapping[agent]
                action = actions[agent_id, :]
                env.step(action)

    env.close()
    print("haha")
