import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cleanrl.common import preprocess_obs_space, preprocess_ac_space
import argparse
import numpy as np
import gym
import gym_microrts
from gym.wrappers import TimeLimit, Monitor
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
import pandas as pd

# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
class NormalizedEnv(gym.core.Wrapper):
    def __init__(self, env, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        super(NormalizedEnv, self).__init__(env)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=(1,)) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(())
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, news, infos = self.env.step(action)
        infos['real_reward'] = rews
        # print("before", self.ret)
        self.ret = self.ret * self.gamma + rews
        # print("after", self.ret)
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret].copy()))
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret = self.ret * (1-float(news))
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(())
        obs = self.env.reset()
        return self._obfilt(obs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicrortsMining10x10F9-v0",
                       help='the id of the gym environment')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=100000,
                       help='total timesteps of the experiments')
    parser.add_argument('--no-torch-deterministic', action='store_false', dest="torch_deterministic", default=True,
                       help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--no-cuda', action='store_false', dest="cuda", default=True,
                       help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', action='store_true', default=False,
                       help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', action='store_true', default=False,
                       help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                       help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--batch-size', type=int, default=2048,
                       help='the batch size of ppo')
    parser.add_argument('--minibatch-size', type=int, default=256,
                       help='the mini batch size of ppo')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.97,
                       help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help="coefficient of the entropy")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                       help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=10,
                        help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', action='store_true', default=False,
                        help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', action='store_true', default=False,
                        help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.015,
                        help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', action='store_true', default=True,
                        help='Use GAE for advantage computation')
    parser.add_argument('--policy-lr', type=float, default=3e-4,
                        help="the learning rate of the policy optimizer")
    parser.add_argument('--value-lr', type=float, default=3e-4,
                        help="the learning rate of the critic optimizer")
    parser.add_argument('--norm-obs', action='store_true', default=True,
                        help="Toggles observation normalization")
    parser.add_argument('--norm-returns', action='store_true', default=False,
                        help="Toggles returns normalization")
    parser.add_argument('--norm-adv', action='store_true', default=True,
                        help="Toggles advantages normalization")
    parser.add_argument('--obs-clip', type=float, default=10.0,
                        help="Value for reward clipping, as per the paper")
    parser.add_argument('--rew-clip', type=float, default=10.0,
                        help="Value for observation clipping, as per the paper")
    parser.add_argument('--anneal-lr', action='store_true', default=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--weights-init', default="orthogonal", choices=["xavier", 'orthogonal'],
                        help='Selects the scheme to be used for weights initialization'),
    parser.add_argument('--clip-vloss', action="store_true", default=True,
                        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--pol-layer-norm', action='store_true', default=False,
                       help='Enables layer normalization in the policy network')
    parser.add_argument('--invalid-action-penalty', type=float, default=0.00,
                       help='the negative reward penalty for invalid actions')
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

args.features_turned_on = sum([args.kle_stop, args.kle_rollback, args.gae, args.norm_obs, args.norm_returns, args.norm_adv, args.anneal_lr, args.clip_vloss, args.pol_layer_norm])
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
# respect the default timelimit
assert isinstance(env.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"
assert isinstance(env, TimeLimit) or int(args.episode_length), "the gym env does not have a built in TimeLimit, please specify by using --episode-length"
if isinstance(env, TimeLimit):
    if int(args.episode_length):
        env._max_episode_steps = int(args.episode_length)
    args.episode_length = env._max_episode_steps
else:
    env = TimeLimit(env, int(args.episode_length))
env = NormalizedEnv(env.env, ob=args.norm_obs, ret=args.norm_returns, clipob=args.obs_clip, cliprew=args.rew_clip, gamma=args.gamma)
env = TimeLimit(env, int(args.episode_length))
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

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
            nn.Linear(128, env.action_space.nvec.sum())
        )

    def forward(self, x):
        x = torch.Tensor(np.moveaxis(x, -1, 1)).to(device)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_action(self, x, action=None):
        logits = self.forward(x)
        split_logits = torch.split(logits, env.action_space.nvec.tolist(), dim=1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        # entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])

        return action, logprob, [], multi_categoricals

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

def discount_cumsum(x, dones, gamma):
    """
    computing discounted cumulative sums of vectors that resets with dones
    input:
        vector x,  vector dones,
        [x0,       [0,
         x1,        0,
         x2         1,
         x3         0, 
         x4]        0]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2,
         x3 + discount * x4,
         x4]
    """
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1] * (1-dones[t])
    return discount_cumsum

pg = Policy().to(device)
vf = Value().to(device)

# MODIFIED: Separate optimizer and learning rates
pg_optimizer = optim.Adam(list(pg.parameters()), lr=args.policy_lr)
v_optimizer = optim.Adam(list(vf.parameters()), lr=args.value_lr)

# MODIFIED: Initializing learning rate anneal scheduler when need
if args.anneal_lr:
    anneal_fn = lambda f: max(0, 1-f / args.total_timesteps)
    pg_lr_scheduler = optim.lr_scheduler.LambdaLR(pg_optimizer, lr_lambda=anneal_fn)
    vf_lr_scheduler = optim.lr_scheduler.LambdaLR(v_optimizer, lr_lambda=anneal_fn)

loss_fn = nn.MSELoss()

# TRY NOT TO MODIFY: start the game
global_step = 0
while global_step < args.total_timesteps:
    if args.capture_video:
        env.stats_recorder.done=True
    next_obs = np.array(env.reset())

    # ALGO Logic: Storage for epoch data
    obs = np.empty((args.batch_size,) + env.observation_space.shape)

    actions = np.empty((args.batch_size,) + env.action_space.shape)
    logprobs = torch.zeros((env.action_space.nvec.shape[0], args.batch_size,)).to(device)

    rewards = np.zeros((args.batch_size,))
    real_rewards = []
    invalid_action_stats = []

    dones = np.zeros((args.batch_size,))
    values = torch.zeros((args.batch_size,)).to(device)

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.batch_size):
        env.render()
        global_step += 1
        obs[step] = next_obs.copy()

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            values[step] = vf.forward(obs[step:step+1])
            action, logproba, _, probs = pg.get_action(obs[step:step+1])
        
        actions[step] = action[:,0].data.cpu().numpy()
        logprobs[:,[step]] = logproba

        # TRY NOT TO MODIFY: execute the game and log data.
        try:
            next_obs, rewards[step], dones[step], info = env.step(action[:,0].data.cpu().numpy())
            rewards[step] += pd.DataFrame([info['invalid_action_stats']]).sum().sum() * args.invalid_action_penalty
        except Exception as e:
            print(e)
            print(e.stacktrace())
        real_rewards += [info['real_reward']]
        invalid_action_stats += [info['invalid_action_stats']]
        next_obs = np.array(next_obs)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            pg_lr_scheduler.step()
            vf_lr_scheduler.step()

        if dones[step]:
            # Computing the discounted returns:
            writer.add_scalar("charts/episode_reward", np.sum(real_rewards), global_step)
            print(f"global_step={global_step}, episode_reward={np.sum(real_rewards)}")
            real_rewards = []
            for key, idx in zip(info['invalid_action_stats'], range(len(info['invalid_action_stats']))):
                writer.add_scalar(f"stats/{key}", pd.DataFrame(invalid_action_stats).sum(0)[idx], global_step)
            invalid_action_stats = []
            next_obs = np.array(env.reset())

    # bootstrap reward if not done. reached the batch limit
    last_value = 0
    if not dones[step]:
        last_value = vf.forward(next_obs.reshape((1,)+next_obs.shape))[0].detach().cpu().numpy()[0]
    bootstrapped_rewards = np.append(rewards, last_value)

    # calculate the returns and advantages
    if args.gae:
        bootstrapped_values = np.append(values.detach().cpu().numpy(), last_value)
        deltas = bootstrapped_rewards[:-1] + args.gamma * bootstrapped_values[1:] * (1-dones) - bootstrapped_values[:-1]
        advantages = discount_cumsum(deltas, dones, args.gamma * args.gae_lambda)
        advantages = torch.Tensor(advantages).to(device)
        returns = advantages + values
    else:
        returns = discount_cumsum(bootstrapped_rewards, dones, args.gamma)[:-1]
        advantages = returns - values.detach().cpu().numpy()
        advantages = torch.Tensor(advantages).to(device)
        returns = torch.Tensor(returns).to(device)

    # Advantage normalization
    if args.norm_adv:
        EPS = 1e-10
        advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

    # Optimizaing policy network
    entropys = []
    target_pg = Policy().to(device)
    inds = np.arange(args.batch_size,)
    for i_epoch_pi in range(args.update_epochs):
        np.random.shuffle(inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            target_pg.load_state_dict(pg.state_dict())
            
            _, newlogproba, _, _ = pg.get_action(
                obs[minibatch_ind],
                torch.LongTensor(actions[minibatch_ind].astype(np.int)).to(device).T)
            ratio = (newlogproba - logprobs[:,minibatch_ind]).exp()

            # Policy loss as in OpenAI SpinUp
            clip_adv = torch.where(advantages[minibatch_ind] > 0,
                                    (1.+args.clip_coef) * advantages[minibatch_ind],
                                    (1.-args.clip_coef) * advantages[minibatch_ind]).to(device)

            # Entropy computation with resampled actions
            entropy = -(newlogproba.exp() * newlogproba).mean()
            entropys.append(entropy.item())

            policy_loss = -torch.min(ratio * advantages[minibatch_ind], clip_adv) + args.ent_coef * entropy
            policy_loss = policy_loss.mean()
            
            pg_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(pg.parameters(), args.max_grad_norm)
            pg_optimizer.step()

            approx_kl = (logprobs[:,minibatch_ind] - newlogproba).mean()
            # Resample values
            new_values = vf.forward(obs[minibatch_ind]).view(-1)

            # Value loss clipping
            if args.clip_vloss:
                v_loss_unclipped = ((new_values - returns[minibatch_ind]) ** 2)
                v_clipped = values[minibatch_ind] + torch.clamp(new_values - values[minibatch_ind], -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - returns[minibatch_ind])**2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = torch.mean((returns[minibatch_ind]- new_values).pow(2))

            v_optimizer.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(vf.parameters(), args.max_grad_norm)
            v_optimizer.step()

        if args.kle_stop:
            if approx_kl > args.target_kl:
                break
        if args.kle_rollback:
            if (logprobs[:,minibatch_ind] - 
                pg.get_action(
                    obs[minibatch_ind],
                    torch.LongTensor(actions[minibatch_ind].astype(np.int)).to(device).T,
                    )[1]).mean() > args.target_kl:
                pg.load_state_dict(target_pg.state_dict())
                break

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("charts/policy_learning_rate", pg_optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar("charts/value_learning_rate", v_optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
    writer.add_scalar("losses/entropy", np.mean(entropys), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)

env.close()
writer.close()
