# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
from gym.spaces import MultiDiscrete
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from torch.utils.tensorboard import SummaryWriter

from ppo_gridnet import Agent, MicroRTSStatsRecorder
import importlib

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=100000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    parser.add_argument('--n-minibatch', type=int, default=4,
        help='the number of mini batch')
    parser.add_argument('--num-selfplay-envs', type=int, default=2,
        help='the number of self play envs; 16 self play envs means 8 games')
    parser.add_argument('--ai', type=str, default="",
        help='the number of steps per game environment')
    parser.add_argument('--num-steps', type=int, default=256,
        help='the number of steps per game environment')
    parser.add_argument("--agent-model-path", type=str, default="agent.pt",
        help="the path to the agent's model")

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()

    # TRY NOT TO MODIFY: setup the environment
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text(
        "hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )
    if args.prod_mode:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            # sync_tensorboard=True,
            config=vars(args),
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.tensorboard.patch(save=False)
        writer = SummaryWriter(f"/tmp/{experiment_name}")
        CHECKPOINT_FREQUENCY = 50

    # TRY NOT TO MODIFY: seeding
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    ais = []
    if args.ai:
        ais = [eval(f"microrts_ai.{args.ai}")]
    # all_ais = {
    #     # "randomBiasedAI": microrts_ai.randomBiasedAI,
    #     # "randomAI": microrts_ai.randomAI,
    #     # "passiveAI": microrts_ai.passiveAI,
    #     # "workerRushAI": microrts_ai.workerRushAI,
    #     # "lightRushAI": microrts_ai.lightRushAI,
    #     # "coacAI": microrts_ai.coacAI,
    #     # "naiveMCTSAI": microrts_ai.naiveMCTSAI,
    #     # "mixedBot": microrts_ai.mixedBot,
    #     # "rojo": microrts_ai.rojo,
    #     # "izanagi": microrts_ai.izanagi,
    #     # "tiamat": microrts_ai.tiamat,
    #     # "droplet": microrts_ai.droplet,
    #     # "guidedRojoA3N": microrts_ai.guidedRojoA3N
    #     # "POLightRush": microrts_ai.POLightRush,
    #     # "POWorkerRush": microrts_ai.POWorkerRush,
    # }
    # ai_names, ais = list(all_ais.keys()), list(all_ais.values())
    # ai_match_stats = dict(zip(ai_names, np.zeros((len(ais), 3))))
    args.num_envs = len(ais) + args.num_selfplay_envs
    envs = MicroRTSGridModeVecEnv(
        num_bot_envs=len(ais),
        num_selfplay_envs=args.num_selfplay_envs,
        partial_obs=args.partial_obs,
        max_steps=5000,
        render_theme=2,
        ai2s=ais,
        map_path="maps/16x16/basesWorkers16x16A.xml",
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )
    envs = MicroRTSStatsRecorder(envs)
    envs = VecMonitor(envs)
    if args.capture_video:
        envs = VecVideoRecorder(
            envs, f"videos/{experiment_name}", record_video_trigger=lambda x: x % 1000000 == 0, video_length=2000
        )
    assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

    agent = Agent(envs).to(device)
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
    # Note how `next_obs` and `next_done` are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = 10000

    ## CRASH AND RESUME LOGIC:
    starting_update = 1
    agent.load_state_dict(torch.load(args.agent_model_path))
    agent.eval()
    from jpype.types import JArray, JInt

    for update in range(starting_update, num_updates + 1):

        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(0, args.num_steps):
            envs.render()
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                action, logproba, _, invalid_action_masks[step], vs = agent.get_action_and_value(
                    next_obs, envs=envs, device=device
                )
                values[step] = vs.flatten()

            actions[step] = action
            logprobs[step] = logproba

            # TRY NOT TO MODIFY: execute the game and log data.
            # the real action adds the source units
            real_action = torch.cat(
                [torch.stack([torch.arange(0, mapsize, device=device) for i in range(envs.num_envs)]).unsqueeze(2), action], 2
            )

            # at this point, the `real_action` has shape (num_envs, map_height*map_width, 8)
            # so as to predict an action for each cell in the map; this obviously include a
            # lot of invalid actions at cells for which no source units exist, so the rest of
            # the code removes these invalid actions to speed things up
            real_action = real_action.cpu().numpy()
            valid_actions = real_action[invalid_action_masks[step][:, :, 0].bool().cpu().numpy()]
            valid_actions_counts = invalid_action_masks[step][:, :, 0].sum(1).long().cpu().numpy()
            java_valid_actions = []
            valid_action_idx = 0
            for env_idx, valid_action_count in enumerate(valid_actions_counts):
                java_valid_action = []
                for c in range(valid_action_count):
                    java_valid_action += [JArray(JInt)(valid_actions[valid_action_idx])]
                    valid_action_idx += 1
                java_valid_actions += [JArray(JArray(JInt))(java_valid_action)]
            java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_actions)

            try:
                next_obs, rs, ds, infos = envs.step(java_valid_actions)
                next_obs = torch.Tensor(next_obs).to(device)
            except Exception as e:
                e.printStackTrace()
                raise
            rewards[step], next_done = torch.Tensor(rs).to(device), torch.Tensor(ds).to(device)

    #         for idx, info in enumerate(infos):
    #             if "episode" in info.keys():
    #                 # print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
    #                 # writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)

    #                 print("against", ai_names[idx], info["microrts_stats"]["WinLossRewardFunction"])
    #                 if info["microrts_stats"]["WinLossRewardFunction"] == -1.0:
    #                     ai_match_stats[ai_names[idx]][0] += 1
    #                 elif info["microrts_stats"]["WinLossRewardFunction"] == 0.0:
    #                     ai_match_stats[ai_names[idx]][1] += 1
    #                 elif info["microrts_stats"]["WinLossRewardFunction"] == 1.0:
    #                     ai_match_stats[ai_names[idx]][2] += 1
    #                 # raise
    #                 # print(info['microrts_stats']['WinLossRewardFunction'])
    #                 assert info["microrts_stats"]["WinLossRewardFunction"] != -2.0
    #                 assert info["microrts_stats"]["WinLossRewardFunction"] != 2.0
    #                 # for key in info['microrts_stats']:
    #                 #     writer.add_scalar(f"charts/episode_reward/{key}", info['microrts_stats'][key], global_step)
    #                 # print("=============================================")
    #                 # break

    # n_rows, n_cols = 3, 5
    # fig = plt.figure(figsize=(5 * 3, 4 * 3))
    # for i, var_name in enumerate(ai_names):
    #     ax = fig.add_subplot(n_rows, n_cols, i + 1)
    #     ax.bar(["loss", "tie", "win"], ai_match_stats[var_name])
    #     ax.set_title(var_name)
    # fig.suptitle(args.agent_model_path)
    # fig.tight_layout()

    envs.close()
    writer.close()
