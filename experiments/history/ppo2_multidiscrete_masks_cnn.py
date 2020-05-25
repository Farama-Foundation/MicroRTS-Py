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

# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
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
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret]))
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret = float(news)
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
    parser.add_argument('--gym-id', type=str, default="MicrortsGlobalAgentMining10x10FrameSkip9-v0",
                       help='the id of the gym environment')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=500000,
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
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.97,
                       help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.05,
                       help="coefficient of the entropy")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                       help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', action='store_true', default=False,
                        help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', action='store_true', default=False,
                        help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.015,
                        help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', action='store_true', default=True,
                        help='Use GAE for advantage computation')
    parser.add_argument('--policy-lr', type=float, default=7e-4,
                        help="the learning rate of the policy optimizer")
    parser.add_argument('--value-lr', type=float, default=7e-4,
                        help="the learning rate of the critic optimizer")
    parser.add_argument('--norm-obs', action='store_true', default=False,
                        help="Toggles observation normalization")
    parser.add_argument('--norm-returns', action='store_true', default=True,
                        help="Toggles returns normalization")
    parser.add_argument('--norm-adv', action='store_true', default=True,
                        help="Toggles advantages normalization")
    parser.add_argument('--obs-clip', type=float, default=10.0,
                        help="Value for reward clipping, as per the paper")
    parser.add_argument('--rew-clip', type=float, default=10.0,
                        help="Value for observation clipping, as per the paper")
    parser.add_argument('--anneal-lr', action='store_true', default=False,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--weights-init', default="xavier", choices=["xavier", 'orthogonal'],
                        help='Selects the scheme to be used for weights initialization'),
    parser.add_argument('--clip-vloss', action="store_true", default=True,
                        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--pol-layer-norm', action='store_true', default=True,
                       help='Enables layer normalization in the policy network')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

args.features_turned_on = sum([args.kle_stop, args.gae, args.norm_obs, args.norm_returns, args.norm_adv, args.anneal_lr, args.clip_vloss])

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
input_shape, preprocess_obs_fn = preprocess_obs_space(env.observation_space, device)
output_shape = preprocess_ac_space(env.action_space)
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
            nn.Linear(128, output_shape)
        )

        # if args.pol_layer_norm:
        #     self.ln1 = torch.nn.LayerNorm(120)
        #     self.ln2 = torch.nn.LayerNorm(84)
        # if args.weights_init == "orthogonal":
        #     torch.nn.init.orthogonal_(self.fc1.weight)
        #     torch.nn.init.orthogonal_(self.fc2.weight)
        #     torch.nn.init.orthogonal_(self.fc3.weight)
        # elif args.weights_init == "xavier":
        #     torch.nn.init.xavier_uniform_(self.fc1.weight)
        #     torch.nn.init.xavier_uniform_(self.fc2.weight)
        #     torch.nn.init.xavier_uniform_(self.fc3.weight)
        # else:
        #     raise NotImplementedError

    def forward(self, x):
        x = torch.Tensor(np.moveaxis(x, -1, 1)).to(device)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_action(self, x, action=None, invalid_action_masks=None):
        logits = self.forward(x)
        split_logits = torch.split(logits, env.action_space.nvec.tolist(), dim=1)
        
        if invalid_action_masks is not None:
            split_invalid_action_masks = torch.split(invalid_action_masks, env.action_space.nvec.tolist(), dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
        else:
            multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])

        return action, logprob, entropy, multi_categoricals
    
    def get_logprob_entropy(self, x, action=None):
        logits = self.forward(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

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

        # if args.weights_init == "orthogonal":
        #     torch.nn.init.orthogonal_(self.fc1.weight)
        #     torch.nn.init.orthogonal_(self.fc2.weight)
        #     torch.nn.init.orthogonal_(self.fc3.weight)
        # elif args.weights_init == "xavier":
        #     torch.nn.init.xavier_uniform_(self.fc1.weight)
        #     torch.nn.init.xavier_uniform_(self.fc2.weight)
        #     torch.nn.init.orthogonal_(self.fc3.weight)
        # else:
        #     raise NotImplementedError

    def forward(self, x):
        x = torch.Tensor(np.moveaxis(x, -1, 1)).to(device)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
    obs = np.empty((args.episode_length,) + env.observation_space.shape)

    actions = np.empty((args.episode_length,) + env.action_space.shape)
    logprobs = torch.zeros((env.action_space.nvec.shape[0], args.episode_length,)).to(device)

    rewards = np.zeros((args.episode_length,))
    real_rewards = np.zeros((args.episode_length))
    returns = np.zeros((args.episode_length,))

    dones = np.zeros((args.episode_length,))
    values = torch.zeros((args.episode_length,)).to(device)

    episode_lengths = [-1]
    advantages = np.zeros((args.episode_length,))
    deltas = np.zeros((args.episode_length,))
    invalid_action_masks = torch.zeros((args.episode_length, output_shape))
    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs[step] = next_obs.copy()
        env.render()

        # ALGO LOGIC: put action logic here
        invalid_action_mask = torch.ones(output_shape)
        invalid_action_mask[0:env.action_space.nvec[0]] = torch.tensor(env.unit_location_mask)
        invalid_action_mask[-env.action_space.nvec[-1]:] = torch.tensor(env.target_unit_location_mask)
        invalid_action_masks[step] = invalid_action_mask
        with torch.no_grad():
            # raise
            values[step] = vf.forward(obs[step:step+1])
            action, logproba, _, probs = pg.get_action(obs[step:step+1], invalid_action_masks=invalid_action_masks[step:step+1])
        
        actions[step] = action[:,0].data.cpu().numpy()
        logprobs[:,[step]] = logproba

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], info = env.step(action[:,0].data.cpu().numpy())
        real_rewards[step] = info['real_reward']
        next_obs = np.array(next_obs)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            pg_lr_scheduler.step()
            vf_lr_scheduler.step()

        if dones[step]:
            # Computing the discounted returns:
            if args.gae:
                prev_return = 0
                prev_value = 0
                prev_advantage = 0
                for i in reversed(range(episode_lengths[-1], step)):
                    returns[i] = rewards[i] + args.gamma * prev_return * (1 - dones[i])
                    deltas[i] = rewards[i] + args.gamma * prev_value * (1 - dones[i]) - values[i]
                    # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
                    advantages[i] = deltas[i] + args.gamma * args.gae_lambda * prev_advantage * (1 - dones[i])
                    prev_return = returns[i]
                    prev_value = values[i]
                    prev_advantage = advantages[i]
            else:
                returns[step] = rewards[step]
                for t in reversed(range(episode_lengths[-1], step)):
                    returns[t] = rewards[t] + args.gamma * returns[t+1] * (1-dones[t])

            writer.add_scalar("charts/episode_reward", real_rewards[(episode_lengths[-1]+1):step+1].sum(), global_step)
            print(f"global_step={global_step}, episode_reward={real_rewards[(episode_lengths[-1]+1):step+1].sum()}")
            episode_lengths += [step]
            next_obs = np.array(env.reset())

    # bootstrap reward if not done. reached the batch limit
    if not dones[step]:
        returns = np.append(returns, vf.forward(next_obs.reshape(1, -1))[0].detach().cpu().numpy(), axis=-1)
        if args.gae:
            prev_return = 0
            prev_value = 0
            prev_advantage = 0
            for i in reversed(range(episode_lengths[-1], step)):
                returns[i] = rewards[i] + args.gamma * prev_return * (1 - dones[i])
                deltas[i] = rewards[i] + args.gamma * prev_value * (1 - dones[i]) - values[i]
                # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
                advantages[i] = deltas[i] + args.gamma * args.gae_lambda * prev_advantage * (1 - dones[i])
                prev_return = returns[i]
                prev_value = values[i]
                prev_advantage = advantages[i]
            returns = returns[:-1]
        else:
            for t in reversed(range(episode_lengths[-1], step+1)):
                returns[t] = rewards[t] + args.gamma * returns[t+1] * (1-dones[t])
            returns = returns[:-1]

    advantages = torch.Tensor(advantages).to(device) if args.gae else torch.Tensor(returns - values.detach().cpu().numpy()).to(device)

    # Advantage normalization
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / advantages.std()

    # Optimizaing policy network
    # First Tensorize all that is need to be so, clears up the loss computation part
    # logprobs = torch.Tensor(logprobs).to(device) # Called 2 times: during policy update and KL bound checked
    returns = torch.Tensor(returns).to(device) # Called 1 time when updating the values
    approx_kls = []
    entropys = []
    target_pg = Policy().to(device)
    inds = np.arange(args.batch_size,)
    for i_epoch_pi in range(args.update_epochs):
        # raise
        target_pg.load_state_dict(pg.state_dict())
        _, newlogproba, _, _ = pg.get_action(obs, torch.LongTensor(actions.astype(np.int)).to(device).T, invalid_action_masks)
        # raise
        # ratio = (newlogproba - logprobs).exp().mean(0)
        ratio = (newlogproba.sum(0) -  logprobs.sum(0)).exp()

        # Policy loss as in OpenAI SpinUp
        clip_adv = torch.where(advantages > 0,
                                (1.+args.clip_coef) * advantages,
                                (1.-args.clip_coef) * advantages).to(device)

        # Entropy computation with resampled actions
        _, resampled_logprobs, _, _ = pg.get_action(obs, invalid_action_masks=invalid_action_masks)
        entropy = - (resampled_logprobs.exp() * resampled_logprobs).mean(0).mean()
        entropys.append(entropy.item())

        policy_loss = - torch.min(ratio * advantages, clip_adv) + args.ent_coef * entropy
        policy_loss = policy_loss.mean()

        pg_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(pg.parameters(), args.max_grad_norm)
        pg_optimizer.step()

        # KEY TECHNIQUE: This will stop updating the policy once the KL has been breached
        # TODO: Roll back the policy to before at breaches the KL trust region
        approx_kl = (logprobs.sum(0) - newlogproba.sum(0)).mean()
        approx_kls.append(approx_kl.item())
        if args.kle_stop:
            if approx_kl > args.target_kl:
                break
        if args.kle_rollback:
            if (logprobs - pg.get_logproba(obs, actions)).mean() > args.target_kl:
                pg.load_state_dict(target_pg.state_dict())
                break
            
    # raise
    # Optimizing value network
    for i_epoch in range(args.update_epochs):
        # Resample values
        new_values = vf.forward(obs).view(-1)

        # Value loss clipping
        if args.clip_vloss:
            v_loss_unclipped = ((new_values - returns) ** 2)
            v_clipped = values + torch.clamp(new_values - values, -args.clip_coef, args.clip_coef)
            v_loss_clipped = (v_clipped - returns)**2
            v_loss_min = torch.min(v_loss_unclipped, v_loss_clipped)
            v_loss = v_loss_min.mean()
        else:
            v_loss = loss_fn(returns, new_values)

        v_optimizer.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(vf.parameters(), args.max_grad_norm)
        v_optimizer.step()
    # TRY NOT TO MODIFY: record rewards for plotting purposes
    print(np.mean(approx_kls))
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
