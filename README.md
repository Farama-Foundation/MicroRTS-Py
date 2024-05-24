<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/MicroRTS-Py/master/micrortspy-text.png" width="500px"/>
</p>

Formerly Gym-μRTS/Gym-MicroRTS

[<img src="https://img.shields.io/badge/discord-gym%20microrts-green?label=Discord&logo=discord&logoColor=ffffff&labelColor=7289DA&color=2c2f33">](https://discord.gg/DdJsrdry6F)
[<img src="https://github.com/vwxyzjn/gym-microrts/workflows/build/badge.svg">](https://github.com/Farama-Foundation/MicroRTS-Py/actions)
[<img src="https://badge.fury.io/py/gym-microrts.svg">](
https://pypi.org/project/gym-microrts/)

This repo contains the source code for the gym wrapper of μRTS authored by [Santiago Ontañón](https://github.com/santiontanon/microrts).

MicroRTS-Py will eventually be updated, maintained, and made compliant with the standards of the Farama Foundation (https://farama.org/project_standards). However, this is currently a lower priority than other projects we're working to maintain. If you'd like to contribute to development, you can join our discord server here- https://discord.gg/jfERDCSw.

![demo.gif](static/fullgame.gif)

## Get Started

Prerequisites:
* Python 3.8+
* [Poetry](https://python-poetry.org)
* Java 8.0+
* FFmpeg (for video recording utilities)

```bash
$ git clone --recursive https://github.com/Farama-Foundation/MicroRTS-Py.git && \
cd MicroRTS-Py
poetry install
# The `poetry install` command above creates a virtual environment for us, in which all the dependencies are installed.
# We can use `poetry shell` to create a new shell in which this environment is activated. Once we are done working with
# MicroRTS, we can leave it again using `exit`.
poetry shell
# By default, the torch wheel is built with CUDA 10.2. If you are using newer NVIDIA GPUs (e.g., 3060 TI), you may need to specifically install CUDA 11.3 wheels by overriding the torch dependency with pip:
# poetry run pip install "torch==1.12.1" --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
python hello_world.py
```

If the `poetry install` command gets stuck on a Linux machine, [it may help to first run](https://github.com/python-poetry/poetry/issues/8623): `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`.

To train an agent, run the following

```bash
cd experiments
python ppo_gridnet.py \
    --total-timesteps 100000000 \
    --capture-video \
    --seed 1
```

[![asciicast](https://asciinema.org/a/586754.svg)](https://asciinema.org/a/586754)

For running a partial observable example, tune the `partial_obs` argument.
```bash
cd experiments
python ppo_gridnet.py \
    --partial-obs \
    --capture-video \
    --seed 1
```

## Technical Paper

Before diving into the code, we highly recommend reading the preprint of our paper: [Gym-μRTS: Toward Affordable Deep Reinforcement Learning Research in Real-time Strategy Games](https://arxiv.org/abs/2105.13807).

### Depreciation notes

1. Note that the experiments in the technical paper above are done with [`gym_microrts==0.3.2`](https://github.com/vwxyzjn/gym-microrts/tree/v0.3.2). As we move forward beyond `v0.4.x`, we are planning to deprecate UAS despite its better performance in the paper. This is because UAS has a more complex implementation and makes it really difficult to incorporate selfplay or imitation learning in the future.
2. [v0.6.1](https://github.com/Farama-Foundation/MicroRTS-Py/releases/tag/v0.6.1) is the last version in which wall/terrain observations were not present in state tensors. As of December 2023, every state observation has an extra channel encoding the presence of walls, and models trained before this will therefore no longer be compatible with code in the `master` branch. Such models should use the code from `v0.6.1` instead.



## Environment Specification

Here is a description of Gym-μRTS's observation and action space:

* **Observation Space.** (`Box(0, 1, (h, w, 29), int32)`) Given a map of size `h x w`, the observation is a tensor of shape `(h, w, n_f)`, where `n_f` is a number of feature planes that have binary values. The observation space used in the original paper used 27 feature planes. Since then, 2 more feature planes (for terrain/walls) have been added, increasing the number of feature planes to 29, as shown below. A feature plane can be thought of as a concatenation of multiple one-hot encoded features. As an example, the unit at a cell could be encoded as follows:

    * the unit has 1 hit point -> `[0,1,0,0,0]`
    * the unit is not carrying any resources, -> `[1,0,0,0,0]`
    * the unit is owned by Player 1 -> `[0,1,0]`
    * the unit is a worker -> `[0,0,0,0,1,0,0,0]`
    * the unit is not executing any actions -> `[1,0,0,0,0,0]`
    * the unit is standing at free terrain cell -> `[1,0]`

    The 29 values of each feature plane for the position in the map of such a worker will thus be:

    `[0,1,0,0,0, 1,0,0,0,0, 0,1,0, 0,0,0,0,1,0,0,0, 1,0,0,0,0,0, 1,0]`
    
* **Partial Observation Space.** (`Box(0, 1, (h, w, 31), int32)`) under the partial observation space, there are two additional binary planes, indicating visibility for the player and their opponent, respectively. If a cell is visible to the player, the second-to-last channel will contain a value of `1`. If the player knows that a cell is visible to the opponent (because the player can observe a nearby enemy unit), the last channel will contain a value of `1`. Using the example above and assuming that the worker unit is not visible to the opponent, then the 31 values of each feature plane for the position in the map of such worker will thus be:

    `[0,1,0,0,0, 1,0,0,0,0, 0,1,0, 0,0,0,0,1,0,0,0, 1,0,0,0,0,0, 1,0, 1,0]`

* **Action Space.** (`MultiDiscrete(concat(h * w * [[6   4   4   4   4   7 a_r]]))`) Given a map of size `h x w` and the maximum attack range `a_r=7`, the action is an (7hw)-dimensional vector of discrete values as specified in the following table. The first 7 component of the action vector represents the actions issued to the unit at `x=0,y=0`, and the second 7 component represents actions issued to the unit at `x=0,y=1`, etc. In these 7 components, the first component is the action type, and the rest of components represent the different parameters different action types can take. Depending on which action type is selected, the game engine will use the corresponding parameters to execute the action. As an example, if the RL agent issues a move south action to the worker at $x=0, y=1$ in a 2x2 map, the action will be encoded in the following way:

    `concat([0,0,0,0,0,0,0], [1,2,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0]]`
    `=[0,0,0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]`

<!-- ![image](https://user-images.githubusercontent.com/5555347/120344517-a5bf7300-c2c7-11eb-81b6-172813ba8a0b.png) -->

Here are tables summarizing observation features and action components, where $a_r=7$ is the maximum attack range, and `-` means not applicable.

| Observation Features        | Planes             | Description                                              |
|-----------------------------|--------------------|----------------------------------------------------------|
| Hit Points                  | 5                  | 0, 1, 2, 3, $\geq 4$                                     |
| Resources                   | 5                  | 0, 1, 2, 3, $\geq 4$                                     |
| Owner                       | 3                  | -,player 1, player 2                                     |
| Unit Types                  | 8                  | -, resource, base, barrack, worker, light, heavy, ranged |
| Current Action              | 6                  | -, move, harvest, return, produce, attack                |
| Terrain                     | 2                  | free, wall                                               |

| Action Components           | Range              | Description                                              |
|-----------------------------|--------------------|----------------------------------------------------------|
| Source Unit                 | $[0,h \times w-1]$ | the location of the unit selected to perform an action   |
| Action Type                 | $[0,5]$            | NOOP, move, harvest, return, produce, attack             |
| Move Parameter              | $[0,3]$            | north, east, south, west                                 |
| Harvest Parameter           | $[0,3]$            | north, east, south, west                                 |
| Return Parameter            | $[0,3]$            | north, east, south, west                                 |
| Produce Direction Parameter | $[0,3]$            | north, east, south, west                                 |
| Produce Type Parameter      | $[0,6]$            | resource, base, barrack, worker, light, heavy, ranged    |
| Relative Attack Position    | $[0,a_r^2 - 1]$    | the relative location of the unit that  will be attacked |

## Evaluation

You can evaluate trained agents against a built-in bot:

```bash
cd experiments
python ppo_gridnet_eval.py \
    --agent-model-path gym-microrts-static-files/agent_sota.pt \
    --ai coacAI
```

Alternatively, you can evaluate the trained RL bots against themselves

```bash
cd experiments
python ppo_gridnet_eval.py \
    --agent-model-path gym-microrts-static-files/agent_sota.pt \
    --agent2-model-path gym-microrts-static-files/agent_sota.pt
```

### Evaluate Trueskill of the agents

This repository already contains a preset Trueskill database in `experiments/league.db`. To evaluate a new AI, try running the following command, which will iteratively find good matches for `agent.pt` until the engine is confident `agent.pt`'s Trueskill (by having the agent's Trueskill sigma below `--highest-sigma 1.4`).

```bash
cd experiments
python league.py --evals gym-microrts-static-files/agent_sota.pt --highest-sigma 1.4 --update-db False
```

To recreate the preset Trueskill database, start a round-robin Trueskill evaluation among built-in AIs by removing the database in `experiments/league.db`.
```bash
cd experiments
rm league.csv league.db
python league.py --evals randomBiasedAI workerRushAI lightRushAI coacAI
```

## Multi-maps support

The training script allows you to train the agents with more than one maps and evaluate with more than one maps. Try executing:

```
cd experiments
python ppo_gridnet.py \
    --train-maps maps/16x16/basesWorkers16x16B.xml maps/16x16/basesWorkers16x16C.xml maps/16x16/basesWorkers16x16D.xml maps/16x16/basesWorkers16x16E.xml maps/16x16/basesWorkers16x16F.xml \
    --eval-maps maps/16x16/basesWorkers16x16B.xml maps/16x16/basesWorkers16x16C.xml maps/16x16/basesWorkers16x16D.xml maps/16x16/basesWorkers16x16E.xml maps/16x16/basesWorkers16x16F.xml
```

where `--train-maps` allows you to specify the training maps and `--eval-maps` the evaluation maps. `--train-maps` and `--eval-maps` do not have to match (so you can evaluate on maps the agent has never trained on before).

## Known issues

[ ] Rendering does not exactly work in macos. See https://github.com/jpype-project/jpype/issues/906

## Papers written using Gym-μRTS

* AIIDE 2022 Strategy Games Workshop: [Transformers as Policies for Variable Action Environments](https://arxiv.org/abs/2301.03679)
* CoG 2021: [Gym-μRTS: Toward Affordable Deep Reinforcement Learning Research in Real-time Strategy Games](https://arxiv.org/abs/2105.13807),
* AAAI RLG 2021: [Generalization in Deep Reinforcement Learning with Real-time Strategy Games](http://aaai-rlg.mlanctot.info/papers/AAAI21-RLG_paper_33.pdf),
* AIIDE 2020 Strategy Games Workshop: [Action Guidance: Getting the Best of Training Agents with Sparse Rewards and Shaped Rewards](https://arxiv.org/abs/2010.03956),
* AIIDE 2019 Strategy Games Workshop: [Comparing Observation and Action Representations for Deep Reinforcement Learning in MicroRTS](https://arxiv.org/abs/1910.12134),

## PettingZoo API

We wrapped our Gym-µRTS simulator into a PettingZoo environment, which is defined in `gym_microrts/pettingzoo_api.py`. An example usage of the Gym-µRTS PettingZoo environment can be found in `hello_world_pettingzoo.py`.


## Cite this project

To cite the Gym-µRTS simulator:

```bibtex
@inproceedings{huang2021gym,
  author    = {Shengyi Huang and
               Santiago Onta{\~{n}}{\'{o}}n and
               Chris Bamford and
               Lukasz Grela},
  title     = {Gym-{\(\mathrm{\mu}\)}RTS: Toward Affordable Full Game Real-time Strategy
               Games Research with Deep Reinforcement Learning},
  booktitle = {2021 {IEEE} Conference on Games (CoG), Copenhagen, Denmark, August
               17-20, 2021},
  pages     = {671--678},
  publisher = {{IEEE}},
  year      = {2021},
  url       = {https://doi.org/10.1109/CoG52621.2021.9619076},
  doi       = {10.1109/CoG52621.2021.9619076},
  timestamp = {Fri, 10 Dec 2021 10:41:01 +0100},
  biburl    = {https://dblp.org/rec/conf/cig/HuangO0G21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

To cite the invalid action masking technique used in our training script:

```bibtex
@inproceedings{huang2020closer,
  author    = {Shengyi Huang and
               Santiago Onta{\~{n}}{\'{o}}n},
  editor    = {Roman Bart{\'{a}}k and
               Fazel Keshtkar and
               Michael Franklin},
  title     = {A Closer Look at Invalid Action Masking in Policy Gradient Algorithms},
  booktitle = {Proceedings of the Thirty-Fifth International Florida Artificial Intelligence
               Research Society Conference, {FLAIRS} 2022, Hutchinson Island, Jensen
               Beach, Florida, USA, May 15-18, 2022},
  year      = {2022},
  url       = {https://doi.org/10.32473/flairs.v35i.130584},
  doi       = {10.32473/flairs.v35i.130584},
  timestamp = {Thu, 09 Jun 2022 16:44:11 +0200},
  biburl    = {https://dblp.org/rec/conf/flairs/HuangO22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
