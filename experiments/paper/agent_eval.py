# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import glob
import random
import time
from distutils.util import strtobool

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gym.spaces import MultiDiscrete
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv, MicroRTSVecEnv
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO agent")
    # Common arguments
    parser.add_argument(
        "--exp-name", type=str, default="ppo_gridnet_diverse_encode_decode", help="the name of this experiment"
    )
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=100000000, help="total timesteps of the experiments")
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will not be enabled by default",
    )
    parser.add_argument(
        "--prod-mode",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="run the script in production mode and use wandb to log outputs",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)",
    )
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument(
        "--num-bot-envs", type=int, default=0, help="the number of bot game environment; 16 bot envs measn 16 games"
    )
    parser.add_argument(
        "--num-selfplay-envs", type=int, default=16, help="the number of self play envs; 16 self play envs means 8 games"
    )
    parser.add_argument("--num-steps", type=int, default=256, help="the number of steps per game environment")
    parser.add_argument(
        "--num-eval-runs", type=int, default=10, help="the number of bot game environment; 16 bot envs measn 16 games"
    )
    parser.add_argument(
        "--agent-model-path",
        type=str,
        default="trained_models/ppo_gridnet_diverse_encode_decode/agent-1.pt",
        help="the path to the agent's model",
    )
    parser.add_argument("--max-steps", type=int, default=2000, help="the maximum number of game steps in microrts")

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
args.num_envs = args.num_selfplay_envs + args.num_bot_envs


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, "f")
        self.eplens = np.zeros(self.num_envs, "i")
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {"r": ret, "l": eplen, "t": round(time.time() - self.tstart, 6)}
                info["episode"] = epinfo
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos


class MicroRTSStatsRecorder(VecEnvWrapper):
    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                raw_rewards = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                info["microrts_stats"] = dict(zip(raw_names, raw_rewards))
                self.raw_rewards[i] = []
                newinfos[i] = info
        return obs, rews, dones, newinfos


# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text(
    "hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
)
if args.prod_mode:
    import wandb

    run = wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=experiment_name,
        save_code=True,
    )
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# TRY NOT TO MODIFY: seeding
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
all_ais = {
    "randomBiasedAI": microrts_ai.randomBiasedAI,
    "randomAI": microrts_ai.randomAI,
    "passiveAI": microrts_ai.passiveAI,
    "workerRushAI": microrts_ai.workerRushAI,
    "lightRushAI": microrts_ai.lightRushAI,
    "coacAI": microrts_ai.coacAI,
    "naiveMCTSAI": microrts_ai.naiveMCTSAI,
    "mixedBot": microrts_ai.mixedBot,
    "rojo": microrts_ai.rojo,
    "izanagi": microrts_ai.izanagi,
    "tiamat": microrts_ai.tiamat,
    "droplet": microrts_ai.droplet,
    "guidedRojoA3N": microrts_ai.guidedRojoA3N,
}
ai_names, ais = list(all_ais.keys()), list(all_ais.values())
ai_match_stats = dict(zip(ai_names, np.zeros((len(ais), 3))))
args.num_envs = len(ais)
ai_envs = []
gridnet_exps = [
    "ppo_gridnet_diverse_impala",
    "ppo_gridnet_coacai",
    "ppo_gridnet_naive",
    "ppo_gridnet_diverse",
    "ppo_gridnet_diverse_encode_decode",
    "ppo_gridnet_coacai_naive",
    "ppo_gridnet_coacai_partial_mask",
    "ppo_gridnet_coacai_no_mask",
    "ppo_gridnet_selfplay_encode_decode",
    "ppo_gridnet_selfplay_diverse_encode_decode",
]
for i in range(len(ais)):
    if args.exp_name in gridnet_exps:
        envs = MicroRTSGridModeVecEnv(
            num_bot_envs=1,
            num_selfplay_envs=0,
            max_steps=args.max_steps,
            render_theme=2,
            ai2s=[ais[i]],
            map_path="maps/16x16/basesWorkers16x16A.xml",
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        )
        envs = MicroRTSStatsRecorder(envs)
        envs = VecMonitor(envs)
        envs = VecVideoRecorder(
            envs, f"videos/{experiment_name}/{ai_names[i]}", record_video_trigger=lambda x: x % 4000 == 0, video_length=2000
        )
    else:
        envs = MicroRTSVecEnv(
            num_envs=1,
            max_steps=args.max_steps,
            render_theme=2,
            ai2s=[ais[i]],
            map_path="maps/16x16/basesWorkers16x16.xml",
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        )
        envs = MicroRTSStatsRecorder(envs)
        envs = VecMonitor(envs)
        envs = VecVideoRecorder(
            envs, f"videos/{experiment_name}/{ai_names[i]}", record_video_trigger=lambda x: x % 4000 == 0, video_length=2000
        )
    ai_envs += [envs]
assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


if args.exp_name == "ppo_gridnet_diverse_impala":

    class Agent(nn.Module):
        def __init__(self, mapsize=16 * 16):
            super(Agent, self).__init__()
            self.mapsize = mapsize
            h, w, c = envs.observation_space.shape
            shape = (c, h, w)
            conv_seqs = []
            for out_channels in [16, 32, 32]:
                conv_seq = ConvSequence(shape, out_channels)
                shape = conv_seq.get_output_shape()
                conv_seqs.append(conv_seq)
            conv_seqs += [
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
                nn.ReLU(),
            ]
            self.network = nn.Sequential(*conv_seqs)
            self.actor = layer_init(nn.Linear(256, self.mapsize * envs.action_space.nvec[1:].sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2)))  # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            grid_logits = logits.view(-1, envs.action_space.nvec[1:].sum())
            split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)

            if action is None:
                invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
                invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
                split_invalid_action_masks = torch.split(
                    invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
                )
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            else:
                invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
                action = action.view(-1, action.shape[-1]).T
                split_invalid_action_masks = torch.split(
                    invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
                )
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            num_predicted_parameters = len(envs.action_space.nvec) - 1
            logprob = logprob.T.view(-1, 256, num_predicted_parameters)
            entropy = entropy.T.view(-1, 256, num_predicted_parameters)
            action = action.T.view(-1, 256, num_predicted_parameters)
            invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum() + 1)
            return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))


elif args.exp_name == "ppo_diverse_impala":

    class Agent(nn.Module):
        def __init__(self, frames=4):
            super(Agent, self).__init__()
            h, w, c = envs.observation_space.shape
            shape = (c, h, w)
            conv_seqs = []
            for out_channels in [16, 32, 32]:
                conv_seq = ConvSequence(shape, out_channels)
                shape = conv_seq.get_output_shape()
                conv_seqs.append(conv_seq)
            conv_seqs += [
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
                nn.ReLU(),
            ]
            self.network = nn.Sequential(*conv_seqs)
            self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2)))  # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)

            if action is None:
                # 1. select source unit based on source unit mask
                source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(1, -1))
                multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
                action_components = [multi_categoricals[0].sample()]
                # 2. select action type and parameter section based on the
                #    source-unit mask of action type and parameters
                # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                source_unit_action_mask = torch.Tensor(
                    np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1)
                )
                split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
                multi_categoricals = multi_categoricals + [
                    CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)
                ]
                invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
                action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
                action = torch.stack(action_components)
            else:
                split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))


elif args.exp_name == "ppo_coacai":

    class Agent(nn.Module):
        def __init__(self, frames=4):
            super(Agent, self).__init__()
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32 * 6 * 6, 256)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2)))  # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)

            if action is None:
                # 1. select source unit based on source unit mask
                source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(1, -1))
                multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
                action_components = [multi_categoricals[0].sample()]
                # 2. select action type and parameter section based on the
                #    source-unit mask of action type and parameters
                # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                source_unit_action_mask = torch.Tensor(
                    np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1)
                )
                split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
                multi_categoricals = multi_categoricals + [
                    CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)
                ]
                invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
                action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
                action = torch.stack(action_components)
            else:
                split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))


elif args.exp_name == "ppo_coacai_no_mask":

    class Agent(nn.Module):
        def __init__(self, frames=4):
            super(Agent, self).__init__()
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32 * 6 * 6, 256)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2)))  # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)

            if action is None:
                # 1. select source unit based on source unit mask
                source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(1, -1))
                multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
                action_components = [multi_categoricals[0].sample()]
                # 2. select action type and parameter section based on the
                #    source-unit mask of action type and parameters
                # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                source_unit_action_mask = torch.Tensor(
                    np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1)
                )

                # this removes the unit action mask
                source_unit_action_mask[:] = 1
                split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
                multi_categoricals = multi_categoricals + [
                    CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)
                ]
                invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
                action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
                action = torch.stack(action_components)
            else:
                split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))


elif args.exp_name == "ppo_gridnet_naive":

    class Agent(nn.Module):
        def __init__(self, mapsize=16 * 16):
            super(Agent, self).__init__()
            self.mapsize = mapsize
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32 * 6 * 6, 256)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(256, self.mapsize * envs.action_space.nvec[1:].sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2)))  # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            grid_logits = logits.view(-1, envs.action_space.nvec[1:].sum())
            split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)

            if action is None:
                invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
                invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
                split_invalid_action_masks = torch.split(
                    invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
                )
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            else:
                invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
                action = action.view(-1, action.shape[-1]).T
                split_invalid_action_masks = torch.split(
                    invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
                )
                multi_categoricals = [
                    Categorical(logits=logits) for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            num_predicted_parameters = len(envs.action_space.nvec) - 1
            logprob = logprob.T.view(-1, 256, num_predicted_parameters)
            entropy = entropy.T.view(-1, 256, num_predicted_parameters)
            action = action.T.view(-1, 256, num_predicted_parameters)
            invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum() + 1)
            return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))


elif args.exp_name in ["ppo_gridnet_diverse", "ppo_gridnet_coacai", "ppo_gridnet_coacai_naive"]:

    class Agent(nn.Module):
        def __init__(self, mapsize=16 * 16):
            super(Agent, self).__init__()
            self.mapsize = mapsize
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32 * 6 * 6, 256)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(256, self.mapsize * envs.action_space.nvec[1:].sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2)))  # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            grid_logits = logits.view(-1, envs.action_space.nvec[1:].sum())
            split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)

            if action is None:
                invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
                invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
                split_invalid_action_masks = torch.split(
                    invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
                )
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            else:
                invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
                action = action.view(-1, action.shape[-1]).T
                split_invalid_action_masks = torch.split(
                    invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
                )
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            num_predicted_parameters = len(envs.action_space.nvec) - 1
            logprob = logprob.T.view(-1, 256, num_predicted_parameters)
            entropy = entropy.T.view(-1, 256, num_predicted_parameters)
            action = action.T.view(-1, 256, num_predicted_parameters)
            invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum() + 1)
            return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))


elif args.exp_name == "ppo_diverse":

    class Agent(nn.Module):
        def __init__(self, frames=4):
            super(Agent, self).__init__()
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32 * 6 * 6, 256)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2)))  # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)

            if action is None:
                # 1. select source unit based on source unit mask
                source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(1, -1))
                multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
                action_components = [multi_categoricals[0].sample()]
                # 2. select action type and parameter section based on the
                #    source-unit mask of action type and parameters
                # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                source_unit_action_mask = torch.Tensor(
                    np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1)
                )
                split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
                multi_categoricals = multi_categoricals + [
                    CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)
                ]
                invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
                action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
                action = torch.stack(action_components)
            else:
                split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))


elif args.exp_name in [
    "ppo_gridnet_diverse_encode_decode",
    "ppo_gridnet_selfplay_diverse_encode_decode",
    "ppo_gridnet_selfplay_encode_decode",
]:

    class Transpose(nn.Module):
        def __init__(self, permutation):
            super().__init__()
            self.permutation = permutation

        def forward(self, x):
            return x.permute(self.permutation)

    class Encoder(nn.Module):
        def __init__(self, input_channels):
            super().__init__()
            self._encoder = nn.Sequential(
                Transpose((0, 3, 1, 2)),
                layer_init(nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.ReLU(),
                layer_init(nn.Conv2d(128, 256, kernel_size=3, padding=1)),
                nn.MaxPool2d(3, stride=2, padding=1),
            )

        def forward(self, x):
            return self._encoder(x)

    class Decoder(nn.Module):
        def __init__(self, output_channels):
            super().__init__()

            self.deconv = nn.Sequential(
                layer_init(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, output_channels, 3, stride=2, padding=1, output_padding=1)),
                Transpose((0, 2, 3, 1)),
            )

        def forward(self, x):
            return self.deconv(x)

    class Agent(nn.Module):
        def __init__(self, mapsize=16 * 16):
            super(Agent, self).__init__()
            self.mapsize = mapsize
            h, w, c = envs.observation_space.shape

            self.encoder = Encoder(c)

            self.actor = Decoder(78)

            self.critic = nn.Sequential(
                nn.Flatten(),
                layer_init(nn.Linear(256, 128), std=1),
                nn.ReLU(),
                layer_init(nn.Linear(128, 1), std=1),
            )

        def forward(self, x):
            return self.encoder(x)  # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            grid_logits = logits.reshape(-1, envs.action_space.nvec[1:].sum())
            split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)

            if action is None:
                invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
                invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
                split_invalid_action_masks = torch.split(
                    invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
                )
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            else:
                invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
                action = action.view(-1, action.shape[-1]).T
                split_invalid_action_masks = torch.split(
                    invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
                )
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            num_predicted_parameters = len(envs.action_space.nvec) - 1
            logprob = logprob.T.view(-1, 256, num_predicted_parameters)
            entropy = entropy.T.view(-1, 256, num_predicted_parameters)
            action = action.T.view(-1, 256, num_predicted_parameters)
            invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum() + 1)
            return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))


elif args.exp_name == "ppo_coacai_naive":

    class Agent(nn.Module):
        def __init__(self, frames=4):
            super(Agent, self).__init__()
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32 * 6 * 6, 256)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2)))  # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)

            if action is None:
                # 1. select source unit based on source unit mask
                source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(1, -1))
                multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
                action_components = [multi_categoricals[0].sample()]
                # 2. select action type and parameter section based on the
                #    source-unit mask of action type and parameters
                # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                source_unit_action_mask = torch.Tensor(
                    np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1)
                )

                # remove the mask on action parameters, which is a similar setup to pysc2
                source_unit_action_mask[:, 6:] = 1

                split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
                multi_categoricals = multi_categoricals + [
                    CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)
                ]
                invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
                action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
                action = torch.stack(action_components)
            else:
                split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))


elif args.exp_name == "ppo_coacai_partial_mask":

    class Agent(nn.Module):
        def __init__(self, frames=4):
            super(Agent, self).__init__()
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32 * 6 * 6, 256)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2)))  # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)

            if action is None:
                # 1. select source unit based on source unit mask
                source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(1, -1))
                multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
                action_components = [multi_categoricals[0].sample()]
                # 2. select action type and parameter section based on the
                #    source-unit mask of action type and parameters
                # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                source_unit_action_mask = torch.Tensor(
                    np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1)
                )
                split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
                multi_categoricals = multi_categoricals + [
                    CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)
                ]
                invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
                action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
                action = torch.stack(action_components)
            else:
                split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
                multi_categoricals = [
                    Categorical(logits=logits) for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))


elif args.exp_name == "ppo_gridnet_coacai_partial_mask":

    class Agent(nn.Module):
        def __init__(self, mapsize=16 * 16):
            super(Agent, self).__init__()
            self.mapsize = mapsize
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32 * 6 * 6, 256)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(256, self.mapsize * envs.action_space.nvec[1:].sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2)))  # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            grid_logits = logits.view(-1, envs.action_space.nvec[1:].sum())
            split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)

            if action is None:
                invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
                invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])

                # remove the mask on action parameters, which is a similar setup to pysc2
                invalid_action_masks[:, 6:] = 1

                split_invalid_action_masks = torch.split(
                    invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
                )
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            else:
                invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
                action = action.view(-1, action.shape[-1]).T
                split_invalid_action_masks = torch.split(
                    invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
                )
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            num_predicted_parameters = len(envs.action_space.nvec) - 1
            logprob = logprob.T.view(-1, 256, num_predicted_parameters)
            entropy = entropy.T.view(-1, 256, num_predicted_parameters)
            action = action.T.view(-1, 256, num_predicted_parameters)
            invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum() + 1)
            return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))


elif args.exp_name == "ppo_gridnet_coacai_no_mask":

    class Agent(nn.Module):
        def __init__(self, mapsize=16 * 16):
            super(Agent, self).__init__()
            self.mapsize = mapsize
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32 * 6 * 6, 256)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(256, self.mapsize * envs.action_space.nvec[1:].sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2)))  # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            grid_logits = logits.view(-1, envs.action_space.nvec[1:].sum())
            split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)

            if action is None:
                invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
                invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
                real_invalid_action_masks = invalid_action_masks.clone()

                # remove masks
                invalid_action_masks[:] = 1
                split_invalid_action_masks = torch.split(
                    invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
                )
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            else:
                invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
                real_invalid_action_masks = invalid_action_masks.clone()

                # remove masks
                invalid_action_masks[:] = 1
                action = action.view(-1, action.shape[-1]).T
                split_invalid_action_masks = torch.split(
                    invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
                )
                multi_categoricals = [
                    CategoricalMasked(logits=logits, masks=iam)
                    for (logits, iam) in zip(split_logits, split_invalid_action_masks)
                ]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            num_predicted_parameters = len(envs.action_space.nvec) - 1
            logprob = logprob.T.view(-1, 256, num_predicted_parameters)
            entropy = entropy.T.view(-1, 256, num_predicted_parameters)
            action = action.T.view(-1, 256, num_predicted_parameters)
            invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum() + 1)
            real_invalid_action_masks = real_invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum() + 1)
            return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))


else:
    raise Exception("incorrect agent selected")

agent = Agent().to(device)
agent.load_state_dict(torch.load(args.agent_model_path))
agent.eval()

print("Model's state_dict:")
for param_tensor in agent.state_dict():
    print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
total_params = sum([param.nelement() for param in agent.parameters()])
print("Model's total parameters:", total_params)
writer.add_scalar(f"charts/total_parameters", total_params, 0)
# ALGO Logic: Storage for epoch data
mapsize = 16 * 16
action_space_shape = (mapsize, envs.action_space.shape[0] - 1)
invalid_action_shape = (mapsize, envs.action_space.nvec[1:].sum() + 1)

obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + action_space_shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + invalid_action_shape).to(device)
# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()

for envs_idx, envs in enumerate(ai_envs):
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    game_count = 0
    from jpype.types import JArray, JInt

    while True:
        # envs.render()
        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            action, logproba, _, invalid_action_mask = agent.get_action(next_obs, envs=envs)

        # TRY NOT TO MODIFY: execute the game and log data.
        if args.exp_name in gridnet_exps:
            # the real action adds the source units
            real_action = torch.cat(
                [torch.stack([torch.arange(0, mapsize, device=device) for i in range(envs.num_envs)]).unsqueeze(2), action], 2
            )

            # at this point, the `real_action` has shape (num_envs, map_height*map_width, 8)
            # so as to predict an action for each cell in the map; this obviously include a
            # lot of invalid actions at cells for which no source units exist, so the rest of
            # the code removes these invalid actions to speed things up
            real_action = real_action.cpu().numpy()
            valid_actions = real_action[invalid_action_mask[:, :, 0].bool().cpu().numpy()]
            valid_actions_counts = invalid_action_mask[:, :, 0].sum(1).long().cpu().numpy()
            java_valid_actions = []
            valid_action_idx = 0
            for env_idx, valid_action_count in enumerate(valid_actions_counts):
                java_valid_action = []
                for c in range(valid_action_count):
                    java_valid_action += [JArray(JInt)(valid_actions[valid_action_idx])]
                    valid_action_idx += 1
                java_valid_actions += [JArray(JArray(JInt))(java_valid_action)]
            java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_actions)
        else:
            java_valid_actions = action.T.cpu().numpy()

        try:
            next_obs, rs, ds, infos = envs.step(java_valid_actions)
            next_obs = torch.Tensor(next_obs).to(device)
        except Exception as e:
            e.printStackTrace()
            raise

        for idx, info in enumerate(infos):
            if "episode" in info.keys():
                # print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                # writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)

                print("against", ai_names[envs_idx], info["microrts_stats"]["WinLossRewardFunction"])
                if info["microrts_stats"]["WinLossRewardFunction"] == -1.0:
                    ai_match_stats[ai_names[envs_idx]][0] += 1
                elif info["microrts_stats"]["WinLossRewardFunction"] == 0.0:
                    ai_match_stats[ai_names[envs_idx]][1] += 1
                elif info["microrts_stats"]["WinLossRewardFunction"] == 1.0:
                    ai_match_stats[ai_names[envs_idx]][2] += 1
                game_count += 1
                # writer.add_scalar(f"charts/episode_reward/{key}", , global_step)
                # for key in info['microrts_stats']:
                #     writer.add_scalar(f"charts/episode_reward/{key}", info['microrts_stats'][key], global_step)
                # print("=============================================")
                # break
        if game_count >= args.num_eval_runs:
            envs.close()
            for (label, val) in zip(["loss", "tie", "win"], ai_match_stats[ai_names[envs_idx]]):
                writer.add_scalar(f"charts/{ai_names[envs_idx]}/{label}", val, 0)
            if args.prod_mode and args.capture_video:
                video_files = glob.glob(f"videos/{experiment_name}/{ai_names[envs_idx]}/*.mp4")
                for video_file in video_files:
                    print(video_file)
                    wandb.log({f"RL agent against {ai_names[envs_idx]}": wandb.Video(video_file)})
                # labels, values = ["loss", "tie", "win"], ai_match_stats[ai_names[envs_idx]]
                # data = [[label, val] for (label, val) in zip(labels, values)]
                # table = wandb.Table(data=data, columns = ["match result", "number of games"])
                # wandb.log({ai_names[envs_idx]: wandb.plot.bar(table, "match result", "number of games", title=f"RL agent against {ai_names[envs_idx]}")})
            break

n_rows, n_cols = 3, 5
fig = plt.figure(figsize=(5 * 3, 4 * 3))
for i, var_name in enumerate(ai_names):
    ax = fig.add_subplot(n_rows, n_cols, i + 1)
    ax.bar(["loss", "tie", "win"], ai_match_stats[var_name])
    ax.set_title(var_name)
fig.suptitle(args.agent_model_path)
fig.tight_layout()
cumulative_match_results = np.array(list(ai_match_stats.values())).sum(0)
cumulative_match_results_rate = cumulative_match_results / cumulative_match_results.sum()
if args.prod_mode:
    wandb.log({"Match results": wandb.Image(fig)})
    for (label, val) in zip(["loss", "tie", "win"], cumulative_match_results):
        writer.add_scalar(f"charts/cumulative_match_results/{label}", val, 0)
    for (label, val) in zip(["loss rate", "tie rate", "win rate"], cumulative_match_results_rate):
        writer.add_scalar(f"charts/cumulative_match_results/{label}", val, 0)
    # labels, values = ["loss", "tie", "win"], cumulative_match_results
    # data = [[label, val] for (label, val) in zip(labels, values)]
    # table = wandb.Table(data=data, columns = ["cumulative match result", "number of games"])
    # wandb.log({"cumulative": wandb.plot.bar(table, "cumulative match result", "number of games", title="RL agent cumulative results")})


envs.close()
writer.close()
