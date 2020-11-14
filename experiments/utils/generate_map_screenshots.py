import argparse
from distutils.util import strtobool
import numpy as np
import gym
import gym_microrts
from gym.wrappers import TimeLimit, Monitor
import copy
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym-id', type=str, default="MicrortsMining-v3",
                        help='the id of the gym environment')
    args = parser.parse_args()

ALL16x16_MAPS = [
    # "maps/16x16/basesWorkers16x16A.xml",
    # "maps/16x16/basesWorkers16x16E.xml",
    # "maps/16x16/basesWorkers16x16I.xml",
    "maps/16x16/basesWorkers16x16noResources.xml",
    "maps/16x16/melee16x16Mixed12.xml",
    "maps/16x16/basesWorkers16x16B.xml",
    "maps/16x16/basesWorkers16x16F.xml",
    "maps/16x16/basesWorkers16x16J.xml",
    "maps/16x16/basesWorkers16x16R20.xml",
    "maps/16x16/melee16x16Mixed8.xml",
    "maps/16x16/basesWorkers16x16C.xml",
    "maps/16x16/basesWorkers16x16G.xml",
    "maps/16x16/basesWorkers16x16K.xml",
    # "maps/16x16/basesWorkers16x16.xml",
    "maps/16x16/TwoBasesBarracks16x16.xml",
    "maps/16x16/basesWorkers16x16D.xml",
    "maps/16x16/basesWorkers16x16H.xml",
    "maps/16x16/basesWorkers16x16L.xml",
    "maps/16x16/EightBasesWorkers16x16.xml",
]

env = gym.make(args.gym_id)
example_config = env.config
raise
# EVALUATION SETTINGS:
# Change maps
for m in ALL16x16_MAPS:
    env = gym.make(args.gym_id)
    c = copy.deepcopy(example_config)
    c.map_path = m
    env.init(config=c)
    env.client.renderTheme = 2
    env.reset()
    img_array = env.render('rgb_array')
    # img = Image.fromarray(img_array[105:607][:,70:571])
    # img = Image.fromarray(img_array)
    img = Image.fromarray(img_array[55:607][:,45:596])
    img.save(f"{m.split('/')[-1]}.png")