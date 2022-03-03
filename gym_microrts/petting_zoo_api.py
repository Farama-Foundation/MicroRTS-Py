import functools
import random
from copy import deepcopy
from pdb import set_trace

import gym
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.test import api_test
from pettingzoo.utils import agent_selector

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeSharedMemVecEnv


class PettingZooMicroRTSGridModeSharedMemVecEnv(AECEnv, MicroRTSGridModeSharedMemVecEnv):

    metadata = {'render.modes': ['human'], "name": "micrortsEnv-v0"}

    def __init__(
        self,
        num_selfplay_envs,
        num_bot_envs,
        partial_obs=False,
        max_steps=2000,
        render_theme=2,
        frame_skip=0,
        ai2s=[],
        map_paths=["maps/10x10/basesTwoWorkers10x10.xml"],
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0]),
    ):
        # Initialize Parent
        print("Initializing environment, please wait ...")
        # Initializes AECEnv
        super().__init__()
        # Initializes MicroRTSGridModeSharedMemVecEnv
        super(AECEnv, self).__init__(
            num_selfplay_envs,
            num_bot_envs,
            partial_obs=partial_obs,
            max_steps=max_steps,
            render_theme=render_theme,
            frame_skip=frame_skip,
            ai2s=ai2s,
            map_paths=map_paths,
            reward_weight=reward_weight,
        )
        print("Initialization completed ...")

        self.agent_action_space = deepcopy(self.action_space)
        self.agent_observation_space = deepcopy(self.observation_space)
        del self.action_space
        del self.observation_space

        _players = ["player_" + str(r) for r in range(num_selfplay_envs)]
        _bots = ["bot_" + str(r) for r in range(num_bot_envs)]
        self.possible_agents = _players + _bots

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents)))))

        self._action_spaces = {
            agent: self.agent_action_space for agent in self.possible_agents}
        self._observation_spaces = {
            agent: self.agent_observation_space for agent in self.possible_agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.agent_observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.agent_action_space

    def reset(self):
        _ = MicroRTSGridModeSharedMemVecEnv.reset(self)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        if self._agent_selector.is_last():
            # Step environment
            actions_list = list(self.state.values())
            actions = np.stack(actions_list, axis=0)
            self.step_async(actions)
            obs, reward, done, info = self.step_wait()

            for i, agent in enumerate(self.agents):
                self.rewards[agent] = reward[i]
                self.dones[agent] = bool(done[i].astype)
                self.observations[agent] = obs[i, :]

            self.num_moves += 1
        else:
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    def observe(self, agent):
        '''
        Returns the observation an agent currently can make. `last()` calls this function.
        '''
        agent_id = self.agent_name_mapping[agent]
        obs = self.obs[agent_id, :, :, :]

        return obs


def main():
    opponents = [microrts_ai.coacAI for _ in range(1)]

    env = PettingZooMicroRTSGridModeSharedMemVecEnv(2, 1, ai2s=opponents)
    # set_trace()
    from pettingzoo.test import api_test
    api_test(env, num_cycles=10, verbose_progress=True)

    # actions = np.array([env.agent_action_space.sample(),
    #                     env.agent_action_space.sample(), env.agent_action_space.sample()])
    # actions = actions.reshape(3, env.width * env.height, env.action_dim)

    # for agent in env.agent_iter():
    #     observation, reward, done, info = env.last()
    #     print(done)
    #     if done:
    #         break

    #     agent_id = env.agent_name_mapping[agent]
    #     action = actions[agent_id, :]
    #     env.step(action)

    #     if env.num_moves > 10:
    #         break


if __name__ == "__main__":
    main()
