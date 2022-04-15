from copy import deepcopy

import numpy as np
from gym import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from gym_microrts.envs.vec_env import MicroRTSGridModeSharedMemVecEnv


class PettingZooMicroRTSGridModeSharedMemVecEnv(AECEnv, MicroRTSGridModeSharedMemVecEnv):

    metadata = {"render.modes": ["human"], "name": "micrortsEnv-v0"}

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

        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        self.action_spaces = {agent: self.agent_action_space for agent in self.possible_agents}

        map_size = self.agent_action_space.shape[0] / 7

        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "obs": self.agent_observation_space,
                    "action_masks": spaces.Box(low=0, high=1, shape=(map_size, 78), dtype=np.int32),
                }
            )
            for agent in self.possible_agents
        }

    def render(self, mode="human"):
        super(MicroRTSGridModeSharedMemVecEnv, self).render(mode)

    def close(self):
        super(MicroRTSGridModeSharedMemVecEnv, self).close()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

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
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
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
            mask = self.get_action_mask()

            for i, agent in enumerate(self.agents):
                self.rewards[agent] = reward[i]
                self.dones[agent] = done[i]
                self.observations[agent] = {"obs": obs[i, :], "action_masks": mask[i, :]}

            self.num_moves += 1
        else:
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    def observe(self, agent):
        """
        Returns the observation an agent currently can make. `last()` calls this function.
        """
        agent_id = self.agent_name_mapping[agent]

        obs = self.obs[agent_id, :, :, :]
        mask = self.get_action_mask()[agent_id, :, :]

        return {"obs": obs, "action_masks": mask}

    def get_action_mask(self):
        self.vec_client.getMasks(0)

        return self.action_mask
