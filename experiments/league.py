# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import pickle
import pandas as pd
import torch
from gym.spaces import MultiDiscrete
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv, MicroRTSBotVecEnv
from gym_microrts import microrts_ai
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from torch.utils.tensorboard import SummaryWriter
from trueskill import TrueSkill, Rating, rate_1vs1, quality_1vs1
from ppo_gridnet import Agent, MicroRTSStatsRecorder, CategoricalMasked
from jpype.types import JArray, JInt
import itertools

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")

    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    parser.add_argument('--rl-ais', nargs='+', default= ['agent_sota.pt'], #
        help='the ais')
    parser.add_argument('--built-in-ais', nargs='+', default=["randomBiasedAI","workerRushAI","lightRushAI","coacAI"],
        help='the ais')
    parser.add_argument('--num-matches', type=int, default=10,
        help='seed of the experiment')
    # default=["randomBiasedAI","workerRushAI","lightRushAI","coacAI","randomAI","passiveAI","naiveMCTSAI","mixedBot","rojo","izanagi","tiamat","droplet","guidedRojoA3N"]
    args = parser.parse_args()
    # fmt: on
    return args


def create_envs(mode: int, partial_obs: bool, built_in_ais=None, built_in_ais2=None):
    # mode 0: rl-ai vs built-in-ai
    # mode 1: rl-ai vs rl-ai
    # mode 2: built-in-ai vs built-in-ai
    max_steps = 5000
    if mode == 0:
        return MicroRTSGridModeVecEnv(
            num_bot_envs=len(built_in_ais),
            num_selfplay_envs=0,
            partial_obs=partial_obs,
            max_steps=max_steps,
            render_theme=2,
            ai2s=built_in_ais,
            map_path="maps/16x16/basesWorkers16x16A.xml",
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        )
    elif mode == 1:
        return MicroRTSGridModeVecEnv(
            num_selfplay_envs=2,
            partial_obs=partial_obs,
            max_steps=max_steps,
            render_theme=2,
            map_path="maps/16x16/basesWorkers16x16A.xml",
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        )
    else:
        return MicroRTSBotVecEnv(
            ai1s=built_in_ais,
            ai2s=built_in_ais2,
            max_steps=max_steps,
            render_theme=2,
            map_path="maps/16x16/basesWorkers16x16.xml",
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
        )
        

class Match:
    def __init__(self, mode: int, partial_obs: bool, built_in_ais=None, built_in_ais2=None, rl_ai=None, rl_ai2=None):
        # mode 0: rl-ai vs built-in-ai
        # mode 1: rl-ai vs rl-ai
        # mode 2: built-in-ai vs built-in-ai
        self.mode = mode
        self.partial_obs = partial_obs
        self.built_in_ais = built_in_ais
        self.built_in_ais2 = built_in_ais2
        self.rl_ai = rl_ai
        self.rl_ai2 = rl_ai2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_steps = 5000
        if mode == 0:
            self.envs = MicroRTSGridModeVecEnv(
                num_bot_envs=len(built_in_ais),
                num_selfplay_envs=0,
                partial_obs=partial_obs,
                max_steps=max_steps,
                render_theme=2,
                ai2s=built_in_ais,
                map_path="maps/16x16/basesWorkers16x16A.xml",
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
            )
            self.agent = Agent(self.envs).to(self.device)
            self.agent.load_state_dict(torch.load(self.rl_ai))
            self.agent.eval()
        elif mode == 1:
            self.envs = MicroRTSGridModeVecEnv(
                num_selfplay_envs=2,
                partial_obs=partial_obs,
                max_steps=max_steps,
                render_theme=2,
                map_path="maps/16x16/basesWorkers16x16A.xml",
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
            )
            self.agent = Agent(self.envs).to(self.device)
            self.agent.load_state_dict(torch.load(self.rl_ai))
            self.agent.eval()
            self.agent2 = Agent(self.envs).to(self.device)
            self.agent2.load_state_dict(torch.load(self.rl_ai2))
            self.agent2.eval()
        else:
            self.envs = MicroRTSBotVecEnv(
                ai1s=built_in_ais,
                ai2s=built_in_ais2,
                max_steps=max_steps,
                render_theme=2,
                map_path="maps/16x16/basesWorkers16x16.xml",
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
            )
        self.envs = MicroRTSStatsRecorder(self.envs)
        self.envs = VecMonitor(self.envs)

    def run(self, num_matches=7):
        if self.mode == 0:
            return self.run_m0(num_matches)
        else:
            return self.run_m2(num_matches)
        
    def run_m0(self, num_matches):
        results = []
        mapsize = 16 * 16
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        while True:
            # self.envs.render()
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                action, _, _, invalid_action_masks, _ = self.agent.get_action_and_value(
                    next_obs, envs=self.envs, device=self.device
                )
    
            # TRY NOT TO MODIFY: execute the game and log data.
            # the real action adds the source units
            real_action = torch.cat(
                [torch.stack([torch.arange(0, mapsize, device=self.device) for i in range(self.envs.num_envs)]).unsqueeze(2), action], 2
            )
    
            # at this point, the `real_action` has shape (num_envs, map_height*map_width, 8)
            # so as to predict an action for each cell in the map; this obviously include a
            # lot of invalid actions at cells for which no source units exist, so the rest of
            # the code removes these invalid actions to speed things up
            real_action = real_action.cpu().numpy()
            valid_actions = real_action[invalid_action_masks[:, :, 0].bool().cpu().numpy()]
            valid_actions_counts = invalid_action_masks[:, :, 0].sum(1).long().cpu().numpy()
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
                next_obs, rs, ds, infos = self.envs.step(java_valid_actions)
                next_obs = torch.Tensor(next_obs).to(self.device)
            except Exception as e:
                e.printStackTrace()
                raise
    
            for idx, info in enumerate(infos):
                if "episode" in info.keys():
                    # print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                    # writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                    results += [info["microrts_stats"]["WinLossRewardFunction"]]
                    print("against", info["microrts_stats"]["WinLossRewardFunction"])
                    # raise
                    # print(info['microrts_stats']['WinLossRewardFunction'])
                    assert info["microrts_stats"]["WinLossRewardFunction"] != -2.0
                    assert info["microrts_stats"]["WinLossRewardFunction"] != 2.0
                    if len(results) >= num_matches:
                        return results
                    # for key in info['microrts_stats']:
                    #     writer.add_scalar(f"charts/episode_reward/{key}", info['microrts_stats'][key], global_step)
                    # print("=============================================")
                    # break

    def run_m2(self, num_matches):
        results = []
        self.envs.reset()
        while True:
            # self.envs.render()
            # dummy actions
            next_obs, reward, done, infos = self.envs.step(
                [[[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],]]) 
            for idx, info in enumerate(infos):
                if "episode" in info.keys():
                    # print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                    # writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                    results += [info["microrts_stats"]["WinLossRewardFunction"]]
                    print(idx, info["microrts_stats"]["WinLossRewardFunction"])
                    # raise
                    # print(info['microrts_stats']['WinLossRewardFunction'])
                    assert info["microrts_stats"]["WinLossRewardFunction"] != -2.0
                    assert info["microrts_stats"]["WinLossRewardFunction"] != 2.0
                    if len(results) >= num_matches:
                        return results

if __name__ == "__main__":
    args = parse_args()
    # m = Match(0, False, built_in_ais=built_in_ais, rl_ai=rl_ai)
    # m.run()

    # env = TrueSkill(mu=50)
    # m = Match(2, False, built_in_ais=built_in_ais, built_in_ais2=built_in_ais)
    # r = m.run()
    all_ais = args.built_in_ais + args.rl_ais
    ratings = dict(zip(all_ais, [Rating() for _ in range (len(all_ais))]))
    match_historys = dict(zip(all_ais, [{} for _ in range (len(all_ais))]))
    match_ups = list(itertools.combinations(all_ais, 2))
    np.random.shuffle(match_ups)
    for idx in range(2):
        for match_up in match_ups:
            if idx == 0:
                match_up = list(reversed(match_up))
            rl_ais = []
            built_in_ais = []
            for ai in match_up:
                if ai[-3:] == ".pt":
                    rl_ais += [ai]
                else:
                    built_in_ais += [ai]
            
            if len(rl_ais) == 1:
                print("mode0")
                p0 = rl_ais[0]
                p1 = built_in_ais[0]
                m = Match(0, False, rl_ai=p0, built_in_ais=[eval(f"microrts_ai.{p1}")])
            else:
                print("mode2")
                p0 = built_in_ais[0]
                p1 = built_in_ais[1]
                m = Match(2, False, built_in_ais=[eval(f"microrts_ai.{p0}")], built_in_ais2=[eval(f"microrts_ai.{p1}")])
            
            r = m.run(args.num_matches // 2)
            for item in r:
                if item == 1:
                    ratings[p0], ratings[p1] = rate_1vs1(ratings[p0], ratings[p1])
                    if p1 not in match_historys[p0]:
                        match_historys[p0][p1] = [1, 0, 0]
                    else:
                        match_historys[p0][p1][0] += 1
                    if p0 not in match_historys[p1]:
                        match_historys[p1][p0] = [0, 0, 1]
                    else:
                        match_historys[p1][p0][2] += 1
                elif item == 0:
                    ratings[p0], ratings[p1] = rate_1vs1(ratings[p0], ratings[p1], drawn=True)
                    if p1 not in match_historys[p0]:
                        match_historys[p0][p1] = [0, 1, 0]
                    else:
                        match_historys[p0][p1][1] += 1
                    if p0 not in match_historys[p1]:
                        match_historys[p1][p0] = [0, 1, 0]
                    else:
                        match_historys[p1][p0][1] += 1
                else:
                    ratings[p1], ratings[p0] = rate_1vs1(ratings[p1], ratings[p0])
                    if p1 not in match_historys[p0]:
                        match_historys[p0][p1] = [0, 0, 1]
                    else:
                        match_historys[p0][p1][2] += 1
                    if p0 not in match_historys[p1]:
                        match_historys[p1][p0] = [1, 0, 0]
                    else:
                        match_historys[p1][p0][0] += 1
    leaderboard = sorted(ratings, key=lambda item: ratings[item].mu - 3 *ratings[item].sigma, reverse=True)
    leaderboard = [(item, round(ratings[item].mu - 3 *ratings[item].sigma,2), ratings[item])  for item in leaderboard]
    
    trueskills = pd.DataFrame(data = [[item[0], item[1], item[2].mu, item[2].sigma] for item in leaderboard], columns=["ai", "trueskill", "mu", "sigma"])
    
    match_historys_dfs = [[key, pd.DataFrame(match_historys[key], index=["win", "tie", "loss"]).T] for key in match_historys]
    dataset = [trueskills, match_historys_dfs]
    with open('dataset.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    if args.prod_mode:
        import wandb

        experiment_name = f"{args.exp_name}__{int(time.time())}"
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.save('dataset.pickle')
        wandb.log({"trueskills": wandb.Table(dataframe=trueskills)})
        artifact = wandb.Artifact("trueskills", type="dataset")
        artifact.add(wandb.Table(dataframe=trueskills), "trueskills")
        run.log_artifact(artifact)
        for item in match_historys_dfs:
            wandb.log({item[0].rstrip(".pt"): wandb.Table(dataframe=item[1].reset_index(level=0))})
