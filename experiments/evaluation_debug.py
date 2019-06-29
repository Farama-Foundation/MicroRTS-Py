import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from cleanrl.common import preprocess_obs_space, preprocess_ac_space
import gym_microrts
import argparse
import numpy as np
import time
import random

import json
import os
import wandb
import gym
import gym_microrts
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A2C agent')
    # Common arguments
    parser.add_argument('--seed', type=int, default=5,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=2000,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=10000,
                       help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=bool, default=True,
                       help='whether to set `torch.backends.cudnn.deterministic=True`')
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
        
api = wandb.Api()
run = api.run("costa-huang/MicrortsRL/irnst1vv")

gym_id = run.config["gym_id"]
env = gym.make("Eval" + gym_id)

# TRY NOT TO MODIFY: setup the environment
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
input_shape, preprocess_obs_fn = preprocess_obs_space(env.observation_space)
output_shape, preprocess_ac_fn = preprocess_ac_space(env.action_space)

# TODO: initialize agent here:
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_shape)

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

pg = Policy()
vf = Value()

run.file('pg.pt').download(True)
run.file('vf.pt').download(True)
os.rename("pg.pt", f"models/{run.name}pg.pt")
os.rename("vf.pt", f"models/{run.name}vf.pt")
pg.load_state_dict(torch.load(f"models/{run.name}pg.pt"))
pg.eval()
vf.load_state_dict(torch.load(f"models/{run.name}vf.pt"))
vf.eval()

# TRY NOT TO MODIFY: start the game
global_step = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())
    actions = np.empty((args.episode_length,), dtype=object)
    rewards, dones = np.zeros((2, args.episode_length))
    obs = np.empty((args.episode_length,) + env.observation_space.shape)
    
    # TODO: put other storage logic here
    values = torch.zeros((args.episode_length))
    neglogprobs = torch.zeros((args.episode_length,))
    entropys = torch.zeros((args.episode_length,))
    
    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs[step] = next_obs.copy()
        
        # TODO: put action logic here
        logits = pg.forward([obs[step]])
        values[step] = vf.forward([obs[step]])
        probs, actions[step], neglogprobs[step], entropys[step] = preprocess_ac_fn(logits)
        actions[step] = actions[step][0]
        
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], _ = env.step(actions[step])
        next_obs = np.array(next_obs)
        if dones[step]:
            break
    
    print("episode ends")
        
env.close()
time.sleep(2)
os.remove(f"models/{run.name}pg.pt")
os.remove(f"models/{run.name}vf.pt")
with open(env.config.evaluation_filename) as f:
    data = json.load(f)
