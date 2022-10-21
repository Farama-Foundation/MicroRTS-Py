<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/MicroRTS-Py/master/micrortspy-text.png" width="500px"/>
</p>

Formerly Gym-μRTS/Gym-MicroRTS

[<img src="https://img.shields.io/badge/discord-gym%20microrts-green?label=Discord&logo=discord&logoColor=ffffff&labelColor=7289DA&color=2c2f33">](https://discord.gg/DdJsrdry6F)
[<img src="https://github.com/vwxyzjn/gym-microrts/workflows/build/badge.svg">](
https://github.com/vwxyzjn/gym-microrts/actions)
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
$ git clone --recursive https://github.com/vwxyzjn/gym-microrts.git && \
cd gym-microrts
poetry install
# build microrts
bash build.sh &> build.log
python hello_world.py
```

To train an agent, run the following

```bash
cd experiments
python ppo_gridnet.py \
    --total-timesteps 100000000 \
    --capture-video \
    --seed 1
```

For running a partial observable example, tune the `partial_obs` argument.
```bash
cd experiments
python ppo_gridnet.py \
    --partial-obs \
    --capture-video \
    --seed 1
```

## Technical Paper

Before diving into the code, we highly recommend reading the preprint of our paper: [Gym-μRTS: Toward Affordable Deep Reinforcement Learning Research in Real-time Strategy Games](https://arxiv.org/abs/2105.13807)

### Depreciation note

Note that the experiments in the technical paper above are done with [`gym_microrts==0.3.2`](https://github.com/vwxyzjn/gym-microrts/tree/v0.3.2). As we move forward beyond `v0.4.x`, we are planing to deprecate UAS despite its better performance in the paper. This is because UAS has more complex implementation and makes it really difficult to incorporate selfplay or imitation learning in the future.



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

* **Action Space.** (`MultiDiscrete(concat(h * w * [[6   4   4   4   4   7 a_r]]))`) Given a map of size `h x w` and the maximum attack range `a_r=7`, the action is an (7hw)-dimensional vector of discrete values as specified in the following table. The first 7 component of the action vector represents the actions issued to the unit at `x=0,y=0`, and the second 7 component represents actions issued to the unit at `x=0,y=1`, etc. In these 7 components, the first component is the action type, and the rest of components represent the different parameters different action types can take. Depending on which action type is selected, the game engine will use the corresponding parameters to execute the action. As an example, if the RL agent issues a move south action to the worker at $x=0, y=1$ in a 2x2 map, the action will be encoded in the following way:

    `concat([0,0,0,0,0,0,0], [1,2,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0]]`
    `=[0,0,0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]`

![image](https://user-images.githubusercontent.com/5555347/120344517-a5bf7300-c2c7-11eb-81b6-172813ba8a0b.png)

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

* CoG 2021: [Gym-μRTS: Toward Affordable Deep Reinforcement Learning Research in Real-time Strategy Games](https://arxiv.org/abs/2105.13807)
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
  pages     = {1--8},
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
