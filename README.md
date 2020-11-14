# Gym-MicroRTS

[<img src="https://img.shields.io/badge/discord-gym%20microrts-green?label=Discord&logo=discord&logoColor=ffffff&labelColor=7289DA&color=2c2f33">](https://discord.gg/5tHykF)
![build](https://github.com/vwxyzjn/gym-microrts/workflows/build/badge.svg)


This repo contains the source code for the gym wrapper of MicroRTS authored by [Santiago Ontañón](https://github.com/santiontanon/microrts). 

# Get Started

```bash
# Make sure you have Java 8.0+ installed
$ pip install gym_microrts --upgrade
```

And run either the `hello_world.py` referenced above or the following file
```python
import gym
import gym_microrts

env = gym.make("MicrortsMining-v1")
env.reset()
for _ in range(10000):
    env.render()
    env.step(env.action_space.sample())
env.close()
```
![demo.gif](demo.gif)


# Developer Guide

Clone the repo

```bash
# install gym-microrts
$ git clone --recursive https://github.com/vwxyzjn/gym-microrts.git && \
cd gym-microrts && \
pip install -e .
# build microrts
$ cd gym_microrts/microrts && bash build.sh && cd ..& cd ..
$ python hello_world.py
```


## Papers written using gym-microrts

* Comparing Observation and Action Representations for Deep Reinforcement Learning in MicroRTS (https://arxiv.org/abs/1910.12134)
    * Logged experiments https://app.wandb.ai/costa-huang/MicrortsRL


