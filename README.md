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

### Depreciation note

Note that the experiments in the technical paper above are done with [`gym_microrts==0.3.2`](https://github.com/vwxyzjn/gym-microrts/tree/v0.3.2). As we move forward beyond `v0.4.x`, we are planing to deprecate UAS despite its better performance in the paper. This is because UAS has more complex implementation and makes it really difficult to incorporate selfplay or imitation learning in the future.

## Get Started

```bash
# Make sure you have Java 8.0+ installed
$ pip install gym_microrts --upgrade
```

And run either the `hello_world.py` in this repo or the following file
```python
import numpy as np
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from numpy.random import choice
from stable_baselines3.common.vec_env import VecVideoRecorder

env = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(1)],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
env = VecVideoRecorder(env, 'videos', record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)

def sample(logits):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    if sum(logits) == 0: return 0
    return choice(range(len(logits)), p=logits/sum(logits))

env.action_space.seed(0)
env.reset()
nvec = env.action_space.nvec
for i in range(10000):
    env.render()
    actions = []
    action_mask = np.array(env.vec_client.getMasks(0))[0] # (16, 16, 79)
    action_mask = action_mask.reshape(-1, action_mask.shape[-1]) # (256, 79)
    source_unit_mask = action_mask[:,[0]] # (256, 1)
    for source_unit in np.where(source_unit_mask == 1)[0]:
        atpm = action_mask[source_unit,1:] # action_type_parameter_mask (78,)
        actions += [[
            source_unit,
            sample(atpm[0:6]), # action type
            sample(atpm[6:10]), # move parameter
            sample(atpm[10:14]), # harvest parameter
            sample(atpm[14:18]), # return parameter
            sample(atpm[18:22]), # produce_direction parameter
            sample(atpm[22:29]), # produce_unit_type parameter
            sample(atpm[29:sum(env.action_space.nvec[1:])]), # attack_target parameter
        ]]
    next_obs, reward, done, info = env.step([actions])
env.close()
```

To train an agent, run the following

```bash
python experiments/ppo.py \
    --total-timesteps 100000000 \
    --wandb-project-name gym-microrts \
    --capture-video \
    --seed 1
```

For running a partial observable example, run the `hello_world_po.py` in this repo.


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

Required dev environment
```
# install pyenv
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

# install python 3.9.5
pyenv install 3.9.5
pyenv global 3.9.5

# install pipx
python -m install pipx

# install other dev dependencies
pipx install poetry
pipx install black
pipx install autoflake
pipx install black
```


```bash
# install gym-microrts
$ git clone --recursive https://github.com/vwxyzjn/gym-microrts.git && \
cd gym-microrts && \
pyenv install -s $(sed "s/\/envs.*//" .python-version)
pyenv virtualenv $(sed "s/\/envs\// /" .python-version)
poetry install
# build microrts
cd gym_microrts/microrts && bash build.sh > build.log && cd ..&& cd ..
python hello_world.py
```

## Known issues

[ ] Rendering does not exactly work in macos. See https://github.com/jpype-project/jpype/issues/906


## Papers written using Gym-μRTS
* CoG 2021: [Gym-μRTS: Toward Affordable Deep Reinforcement Learning Research in Real-time Strategy Games](https://arxiv.org/abs/2105.13807)
* AAAI RLG 2021: [Generalization in Deep Reinforcement Learning with Real-time Strategy Games](http://aaai-rlg.mlanctot.info/papers/AAAI21-RLG_paper_33.pdf), 
* AIIDE 2020 Strategy Games Workshop: [Action Guidance: Getting the Best of Training Agents with Sparse Rewards and Shaped Rewards](https://arxiv.org/abs/2010.03956), 
* AIIDE 2019 Strategy Games Workshop: [Comparing Observation and Action Representations for Deep Reinforcement Learning in MicroRTS](https://arxiv.org/abs/1910.12134), 


