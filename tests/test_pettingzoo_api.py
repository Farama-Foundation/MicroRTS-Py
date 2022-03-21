from pettingzoo.test import api_test

from gym_microrts.petting_zoo_api import PettingZooMicroRTSGridModeSharedMemVecEnv


def main():
    opponents = []
    env = PettingZooMicroRTSGridModeSharedMemVecEnv(2, 0, ai2s=opponents)
    api_test(env, num_cycles=10, verbose_progress=True)


if __name__ == "__main__":
    main()
