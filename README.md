# Gym-μRTS (pronounced "gym-micro-RTS")

[<img src="https://img.shields.io/badge/discord-gym%20microrts-green?label=Discord&logo=discord&logoColor=ffffff&labelColor=7289DA&color=2c2f33">](https://discord.gg/DdJsrdry6F)
[<img src="https://github.com/vwxyzjn/gym-microrts/workflows/build/badge.svg">](
https://github.com/vwxyzjn/gym-microrts/actions)
[<img src="https://badge.fury.io/py/gym-microrts.svg">](
https://pypi.org/project/gym-microrts/)



This repo contains the source code for the gym wrapper of μRTS authored by [Santiago Ontañón](https://github.com/santiontanon/microrts). 



![demo.gif](static/fullgame.gif)

## Technical Paper

Before diving into the code, we highly recommend reading the preprint of our paper: [Gym-μRTS: Toward Affordable Deep Reinforcement Learning Research in Real-time Strategy Games](https://arxiv.org/abs/2105.13807)

## Get Started

```bash
# Make sure you have Java 8.0+ installed
$ pip install gym_microrts --upgrade
```

And run either the `hello_world_v3.py` in this repo or the following file
```python
import numpy as np
import gym
import gym_microrts
import time
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from gym.envs.registration import register
from gym_microrts import Config

try:
    env = MicroRTSVecEnv(
        num_envs=1,
        render_theme=2,
        ai2s=[microrts_ai.coacAI],
        map_path="maps/16x16/basesWorkers16x16.xml",
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    # the above `reward_weight` (in order) means +10 for wining, 
    # +1 for each resource gathered or returned, +1 for each worker produced
    # +0.2 for each building produced, +1 for each attack action issued
    # +4 for each combat units produced
    obs = env.reset()
    env.render()
except Exception as e:
    e.printStackTrace()
env.action_space.seed(0)
env.reset()
for i in range(10000):
    env.render()
    action_mask = np.array(env.vec_client.getUnitLocationMasks()).flatten()
    time.sleep(0.001)
    action = env.action_space.sample()

    # optional: selecting only valid units.
    if len(action_mask.nonzero()[0]) != 0:
        action[0] = action_mask.nonzero()[0][0]

    next_obs, reward, done, info = env.step([action])
    if done:
        env.reset()
env.close()
```


For running a partial observable example, run either the `hello_world_po.py`in this repo or the following file
```import gym
import gym_microrts
import time
import numpy as np
from gym.wrappers import Monitor
from gym_microrts import microrts_ai
from gym_microrts.envs.povec_env import POMicroRTSGridModeVecEnv
import gym_microrts
import os, sys

env = POMicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.randomAI for _ in range(1)],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)

env.action_space.seed(0)
env.reset()
for i in range(10000):
    env.render()
    action_mask = np.array(env.vec_client.getMasks(0)).flatten()
    time.sleep(0.001)
    action = [env.action_space.sample() for _ in range(1)]
    
    # optional: selecting only valid units.
    if len(action_mask.nonzero()[0]) != 0:
        action[:][0] = action_mask.nonzero()[0][0]

    next_obs, reward, done, info = env.step([action])
env.close()
```

To train an agent against the built-in WorkerRushAI, run the following

```bash
python experiments/ppo.py \
    --total-timesteps 100000000 \
    --wandb-project-name gym-microrts \
    --capture-video \
    --seed 1
```


## Environment Specification

Here is a description of Gym-μRTS's observation and action space:

* **Observation Space.** (`Box(0, 1, (h, w, 27), int32)`) Given a map of size `h x w`, the observation is a tensor of shape `(h, w, n_f)`, where `n_f` is a number of feature planes that have binary values. The observation space used in this paper uses 27 feature planes as shown in the following table. A feature plane can be thought of as a concatenation of multiple one-hot encoded features. As an example, if there is a worker with hit points equal to 1, not carrying any resources, owner being Player 1, and currently not executing any actions, then the one-hot encoding features will look like the following:

   `[0,1,0,0,0],  [1,0,0,0,0],  [1,0,0], [0,0,0,0,1,0,0,0],  [1,0,0,0,0,0]`
   

    The 27 values of each feature plane for the position in the map of such worker will thus be:
    
    `[0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0]`

* **Partial Observation Space.** (`Box(0, 1, (h, w, 29), int32)`) Given a map of size `h x w`, the observation is a tensor of shape `(h, w, n_f)`, where `n_f` is a number of feature planes that have binary values. The observation space for partial observability uses 29 feature planes as shown in the following table. A feature plane can be thought of as a concatenation of multiple one-hot encoded features. As an example, if there is a worker with hit points equal to 1, not carrying any resources, owner being Player 1,  currently not executing any actions, and not visible to the opponent, then the one-hot encoding features will look like the following:

   `[0,1,0,0,0],  [1,0,0,0,0],  [1,0,0], [0,0,0,0,1,0,0,0],  [1,0,0,0,0,0], [1,0]`
   

    The 29 values of each feature plane for the position in the map of such worker will thus be:
    
    `[0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0]`

* **Action Space.** (`MultiDiscrete([hw   6   4   4   4   4   7 a_r])`) Given a map of size `h x w` and the maximum attack range `a_r=7`, the action is an 8-dimensional vector of discrete values as specified in the following table. The first component of the action vector represents the unit in the map to issue actions to, the second is the action type, and the rest of components represent the different parameters different action types can take. Depending on which action type is selected, the game engine will use the corresponding parameters to execute the action. As an example, if the RL agent issues a move south action to the worker at $x=3, y=2$ in a 10x10 map, the action will be encoded in the following way:
    
    `[3+2*10,1,2,0,0,0,0,0 ]`

![image](https://user-images.githubusercontent.com/5555347/120344517-a5bf7300-c2c7-11eb-81b6-172813ba8a0b.png)

## Preset Envs:

Gym-μRTS comes with preset environments for common tasks as well as engaging the full game. Feel free to check out the following benchmark:

* [Gym-μRTS V1 Benchmark](https://wandb.ai/vwxyzjn/action-guidance/reports/Gym-microrts-V1-Benchmark--VmlldzozMDQ4MTU)
* [Gym-μRTS V2 Benchmark](https://wandb.ai/vwxyzjn/gym-microrts/reports/Gym-microrts-s-V2-Benchmark--VmlldzoyNTg5NTA)
* [Gym-μRTS V3 Benchmark](https://wandb.ai/vwxyzjn/rts-generalization/reports/Gym-microrts-V3-Environments--VmlldzoyNzQwNzM)


Below are the difference between the versioned environments

|    | use frame skipping | complete invalid action masking            | issuing actions to all units simultaneously | map size |
|----|--------------------|--------------------------------------------|---------------------------------------------|----------|
| v1 | frame skip = 9     | only partial mask on source unit selection | no                                          | 10x10    |
| v2 | no                 | yes                                        | yes                                         | 10x10    |
| v3 | no                 | yes                                        | yes                                         | 16x16    |


## Developer Guide

Clone the repo

```bash
# install gym-microrts
$ git clone --recursive https://github.com/vwxyzjn/gym-microrts.git && \
cd gym-microrts && \
pip install -e .
# build microrts
$ cd gym_microrts/microrts && bash build.sh && cd ..&& cd ..
$ python hello_world.py
```

## Known issues

[ ] Rendering does not exactly work in macos. See https://github.com/jpype-project/jpype/issues/906


## Papers written using Gym-μRTS
* CoG 2021: [Gym-μRTS: Toward Affordable Deep Reinforcement Learning Research in Real-time Strategy Games](https://arxiv.org/abs/2105.13807)
* AAAI RLG 2021: [Generalization in Deep Reinforcement Learning with Real-time Strategy Games](http://aaai-rlg.mlanctot.info/papers/AAAI21-RLG_paper_33.pdf), 
* AIIDE 2020 Strategy Games Workshop: [Action Guidance: Getting the Best of Training Agents with Sparse Rewards and Shaped Rewards](https://arxiv.org/abs/2010.03956), 
* AIIDE 2019 Strategy Games Workshop: [Comparing Observation and Action Representations for Deep Reinforcement Learning in MicroRTS](https://arxiv.org/abs/1910.12134), 


