# Reference: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np
import gym
import matplotlib.pyplot as plt
import random

# Hyperparameters
learning_rate = 1e-3
gamma = 0.98
seed = 1
num_episodes = 500
batch_sz = 200
vf_coef=0.25
ent_coef=0.01

# Set up the env
env = gym.make("CartPole-v0")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
env.seed(seed)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, env.action_space.n)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# TODO: initialize agent here:
pg = Policy()
vf = Value()
optimizer = optim.Adam(list(pg.parameters()) + list(vf.parameters()), lr=learning_rate)
loss_fn = nn.MSELoss()

# TRY NOT TO MODIFY: start the game
ep_rews = np.zeros((num_episodes,))
next_obs = env.reset()
for update in range(num_episodes):
    # TRY NOT TO MODIFY: storage helpers for data
    next_obs = env.reset()
    actions = torch.zeros((batch_sz,))
    rewards, dones = np.zeros((2, batch_sz))
    observations = np.empty((batch_sz,) + env.observation_space.shape)
    
    # TODO: put other storage logic here
    values = torch.zeros((batch_sz))
    neglogprobs = torch.zeros((batch_sz,))
    entropys = torch.zeros((batch_sz,))
    
    for step in range(batch_sz):
        observations[step] = next_obs.copy()
        
        # TODO: put action logic here
        logits = pg.forward(observations[step])
        value = vf.forward(observations[step])
        probs = Categorical(logits=logits)
        action = probs.sample()
        neglogprobs[step] = -probs.log_prob(action)
        values[step] = value
        entropys[step] = probs.entropy()
        
        # TRY NOT TO MODIFY: execute the game and log data.
        actions[step] = action
        next_obs, rewards[step], dones[step], _ = env.step(int(actions[step].numpy()))
        if dones[step]:
            break
    
    # TODO: training.
    next_value = vf.forward(next_obs).detach().numpy()
    # use the value function to bootstrap 
    returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
    for t in reversed(range(rewards.shape[0])):
        returns[t] = rewards[t] + gamma * returns[t+1] * (1-dones[t])
    returns = returns[:-1]
    # advantages are returns - baseline, value estimates in our case
    advantages = returns - values.detach().numpy()
    
    vf_loss = loss_fn(torch.Tensor(returns), torch.Tensor(values)) * vf_coef
    pg_loss = torch.Tensor(advantages) * neglogprobs
    loss = (pg_loss - entropys * ent_coef).mean()
    
    optimizer.zero_grad()
    vf_loss.backward()
    loss.backward()
    optimizer.step()

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    ep_rews[update] = rewards.sum()
    if update % 10 == 0:
        print(f"update = {update}, rewards = {ep_rews[update]}")

plt.plot(ep_rews)