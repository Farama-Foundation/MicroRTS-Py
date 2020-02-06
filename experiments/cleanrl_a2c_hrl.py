import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl.common import preprocess_obs_space, preprocess_ac_space
import argparse
import numpy as np
import gym
import gym_microrts
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A2C agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicrortsGlobalAgentHRLAttackCloserToEnemyBase10x10FrameSkip9-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=25000,
                       help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=bool, default=True,
                       help='whether to set `torch.backends.cudnn.deterministic=True`')
    parser.add_argument('--cuda', type=bool, default=True,
                       help='whether to use CUDA whenever possible')
    parser.add_argument('--prod-mode', type=bool, default=False,
                       help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=bool, default=False,
                       help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="gym-microrts",
                       help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    parser.add_argument('--vf-coef', type=float, default=0.25,
                       help="value function's coefficient the loss function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='the maximum norm for the gradient clipping')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help="policy entropy's coefficient the loss function")
    parser.add_argument('--start-e', type=float, default=1.0,
                       help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.01,
                       help="the ending epsilon for exploration")
    parser.add_argument('--start-a', type=float, default=1.0,
                       help="the starting alpha for exploration")
    parser.add_argument('--end-a', type=float, default=0.8,
                       help="the ending alpha for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.8,
                       help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    wandb.save(os.path.abspath(__file__))

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
env = gym.make(args.gym_id)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
input_shape, preprocess_obs_fn = preprocess_obs_space(env.observation_space, device)
output_shape = preprocess_ac_space(env.action_space)
# respect the default timelimit
if int(args.episode_length):
    if not isinstance(env, TimeLimit):
        env = TimeLimit(env, int(args.episode_length))
    else:
        env._max_episode_steps = int(args.episode_length)
else:
    args.episode_length = env._max_episode_steps if isinstance(env, TimeLimit) else 200
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

# ALGO LOGIC: initialize agent here:
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(27, 16, kernel_size=3,),
            nn.MaxPool2d(1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.MaxPool2d(1),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(32*6*6, 128),
            nn.ReLU(),
            nn.Linear(128, output_shape)
        )

    def forward(self, x):
        x = torch.Tensor(np.moveaxis(x, -1, 1)).to(device)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(27, 16, kernel_size=3,),
            nn.MaxPool2d(1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.MaxPool2d(1),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(32*6*6, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = torch.Tensor(np.moveaxis(x, -1, 1)).to(device)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CategoricalMasked(Categorical):

    def __init__(self, probs=None, logits=None, validate_args=None, masks=None):
        self.masks = torch.BoolTensor(masks).to(device)
        logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

num = env.num_reward_function
pgs = [Policy().to(device) for _ in range(num)]
vfs = [Value().to(device) for _ in range(num)]
optimizers = [optim.Adam(list(pgs[i].parameters()) + list(vfs[i].parameters()), 
    lr=args.learning_rate) for i in range(num)]
loss_fn = nn.MSELoss()

# TRY NOT TO MODIFY: start the game
global_step = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())
    actions = np.empty((args.episode_length,), dtype=object)
    rewards, dones = np.zeros((2, num, args.episode_length))
    obs = np.empty((args.episode_length,) + env.observation_space.shape)

    # ALGO LOGIC: put other storage logic here
    values = torch.zeros((num, args.episode_length), device=device)
    neglogprobs = torch.zeros((num, args.episode_length,), device=device)
    entropys = torch.zeros((num, args.episode_length,), device=device)

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)
        alpha = linear_schedule(args.start_a, args.end_a, args.exploration_fraction*args.total_timesteps, global_step)

        # env.render()
        global_step += 1
        obs[step] = next_obs.copy()

        # ALGO LOGIC: put action logic here
        # HRL: select sub logits
        logitss = [pg.forward(obs[step:step+1]) for pg in pgs]
        logits = logitss[0]
        sub_logits_idx = 1
        if global_step % 100 == 0:
            if random.random() < epsilon:
                sub_logits_idx = np.random.randint(1, num)
            else:
                dist = torch.zeros((num)).to(device)
                for i in range(1, num):
                    dist[i] = torch.dist(logits, logitss[i])
                sub_logits_idx = torch.argmin(dist[1:]) + 1
            
        writer.add_scalar("charts/sub_logits_idx", sub_logits_idx, global_step)
        
        for i in range(len(pgs)):
            values[i,step] = vfs[i].forward(obs[step:step+1])

        # ALGO LOGIC: `env.action_space` specific logic
        if isinstance(env.action_space, MultiDiscrete):
            all_logits_categories =  []
            action = []
            all_probs_categories = []
            all_probs_entropies =  []
            all_neglogprob =  []
            for i in range(num):
                all_logits_categories.append(torch.split(logitss[i], env.action_space.nvec.tolist(), dim=1))
                all_probs_categories.append([])
                all_probs_entropies.append(torch.zeros((logitss[i].shape[0]), device=device))
                all_neglogprob.append(torch.zeros((logitss[i].shape[0]), device=device))

            for i in range(num):
                j=0
                all_probs_categories[i].append(CategoricalMasked(logits=all_logits_categories[i][j], masks=env.unit_location_mask))
            for i in range(num):
                for j in range(1, len(all_logits_categories[0])-1):
                    all_probs_categories[i].append(Categorical(logits=all_logits_categories[i][j]))
            for i in range(num):
                j=len(all_logits_categories[0])-1
                all_probs_categories[i].append(CategoricalMasked(logits=all_logits_categories[i][j], masks=env.target_unit_location_mask))

            # action guidence:
            for j in range(0, len(all_logits_categories[0])):
                temp_probs = Categorical(all_probs_categories[0][j].probs + alpha * (all_probs_categories[sub_logits_idx][j].probs - all_probs_categories[0][j].probs))    
                if len(action) != env.action_space.shape:
                    action.append(temp_probs.sample())
            
            for i in range(num):
                for j in range(0, len(all_logits_categories[0])):
                    all_neglogprob[i] -= all_probs_categories[i][j].log_prob(action[j])
                    all_probs_entropies[i] += all_probs_categories[i][j].entropy()
            
            for i in range(len(all_neglogprob)):
                neglogprobs[i,step] = all_neglogprob[i]
                entropys[i,step] = all_probs_entropies[i]
            action = torch.stack(action).transpose(0, 1).tolist()
            actions[step] = action[0]
            
        if args.prod_mode and global_step % 20000 == 0:
            if not os.path.exists(f"models/{experiment_name}"):
                os.makedirs(f"models/{experiment_name}")
            for i in range(num):
                torch.save(pgs[i].state_dict(), f"models/{experiment_name}/pg_{str(env.rfs[i])}.pt")
                torch.save(vfs[i].state_dict(), f"models/{experiment_name}/vf_{str(env.rfs[i])}.pt")
                wandb.save(f"models/{experiment_name}/pg_{str(env.rfs[i])}.pt")
                wandb.save(f"models/{experiment_name}/vf_{str(env.rfs[i])}.pt")

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = env.step(actions[step])
        dones[:,step] = info['dones'] # hack to make TimeLimit work
        rewards[:,step] = info["rewards"]
        next_obs = np.array(next_obs)
        if done:
            break

    # ALGO LOGIC: training.
    # calculate the discounted rewards, or namely, returns
    returns = np.zeros_like(rewards)
    for t in reversed(range(rewards.shape[1]-1)):
        returns[:,t] = rewards[:,t] + args.gamma * returns[:,t+1] * (1-dones[:,t])
    # advantages are returns - baseline, value estimates in our case
    advantages = returns - values.detach().cpu().numpy()

    vf_loss = ((torch.Tensor(returns).to(device) - values)**2).mean(1) * args.vf_coef
    pg_loss = torch.Tensor(advantages).to(device) * neglogprobs
    entropy_loss = (-entropys) * args.ent_coef
    loss = ((pg_loss + entropy_loss).mean(1) + vf_loss).sum()
    
    for i in range(num):
        optimizers[i].zero_grad()
    pg_loss.mean(1).sum().backward(retain_graph=True)
    print(list(pgs[0].fc.children())[0].weight.grad.sum())
    for i in range(num):
        optimizers[i].zero_grad()
    entropy_loss.mean(1).sum().backward(retain_graph=True)
    print(list(pgs[0].fc.children())[0].weight.grad.sum())


    # HRL: update
    for i in range(num):
        optimizers[i].zero_grad()
    loss.backward()
    for i in range(num):
        nn.utils.clip_grad_norm_(list(pgs[i].parameters()) + list(vfs[i].parameters()), args.max_grad_norm)
        optimizers[i].step()

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/epsilon", epsilon, global_step)
    writer.add_scalar("charts/alpha", alpha, global_step)
    for i in range(len(env.rfs)):
        writer.add_scalar(f"charts/episode_reward/{str(env.rfs[i])}", rewards.sum(1)[i], global_step)
        writer.add_scalar(f"losses/value_loss/{str(env.rfs[i])}", vf_loss[i], global_step)
        writer.add_scalar(f"losses/entropy/{str(env.rfs[i])}", entropys.mean(1)[i], global_step)
        writer.add_scalar(f"losses/policy_loss/{str(env.rfs[i])}", pg_loss.mean(1)[i], global_step)
    print(global_step, rewards.sum(1)[0], all_logits_categories[0][1], list(pgs[0].fc.children())[0].weight.grad.sum())
    
    # writer.add_scalar("losses/value_loss", vf_loss.item(), global_step)
    # writer.add_scalar("losses/entropy", entropys[:step].mean().item(), global_step)
    # writer.add_scalar("losses/policy_loss", pg_loss.mean().item(), global_step)
# env.close()
writer.close()
