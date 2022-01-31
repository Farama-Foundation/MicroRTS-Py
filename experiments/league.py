# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import datetime
import itertools
import os
import random
import shutil
import uuid
from distutils.util import strtobool
from enum import Enum

import numpy as np
import pandas as pd
import torch
from peewee import (
    JOIN,
    CharField,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    Model,
    SmallIntegerField,
    SqliteDatabase,
    fn,
)
from stable_baselines3.common.vec_env import VecMonitor
from trueskill import Rating, quality_1vs1, rate_1vs1

from gym_microrts import microrts_ai  # fmt: off

torch.set_num_threads(1)


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
    parser.add_argument('--evals', nargs='+', default=["randomBiasedAI","workerRushAI","lightRushAI", "coacAI"], # [],
        help='the ais')
    parser.add_argument('--num-matches', type=int, default=10,
        help='seed of the experiment')
    parser.add_argument('--update-db', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, the database will be updated')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--highest-sigma', type=float, default=1.4,
        help='the highest sigma of the trueskill evaluation')
    parser.add_argument('--output-path', type=str, default=f"league.temp.csv",
        help='the output path of the leaderboard csv')
    parser.add_argument('--model-type', type=str, default=f"ppo_gridnet_large", choices=["ppo_gridnet_large", "ppo_gridnet"],
        help='the output path of the leaderboard csv')
    # ["randomBiasedAI","workerRushAI","lightRushAI","coacAI"]
    # default=["randomBiasedAI","workerRushAI","lightRushAI","coacAI","randomAI","passiveAI","naiveMCTSAI","mixedBot","rojo","izanagi","tiamat","droplet","guidedRojoA3N"]
    args = parser.parse_args()
    # fmt: on
    return args


args = parse_args()
dbname = "league"
if args.partial_obs:
    dbname = "po_league"
dbpath = f"gym-microrts-static-files/{dbname}.db"
csvpath = f"gym-microrts-static-files/{dbname}.csv"
if not args.update_db:
    if not os.path.exists(f"gym-microrts-static-files/tmp"):
        os.makedirs(f"gym-microrts-static-files/tmp")
    tmp_dbpath = f"gym-microrts-static-files/tmp/{str(uuid.uuid4())}.db"
    shutil.copyfile(dbpath, tmp_dbpath)
    dbpath = tmp_dbpath
db = SqliteDatabase(dbpath)

if args.model_type == "ppo_gridnet_large":
    from ppo_gridnet_large import Agent, MicroRTSStatsRecorder

    from gym_microrts.envs.vec_env import MicroRTSBotVecEnv, MicroRTSGridModeVecEnv
else:
    from ppo_gridnet import Agent, MicroRTSStatsRecorder

    from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
    from gym_microrts.envs.vec_env import (
        MicroRTSGridModeSharedMemVecEnv as MicroRTSGridModeVecEnv,
    )


class BaseModel(Model):
    class Meta:
        database = db


class AI(BaseModel):
    name = CharField(unique=True)
    mu = FloatField()
    sigma = FloatField()
    ai_type = CharField()

    def __str__(self):
        return f"🤖 {self.name} with N({round(self.mu, 3)}, {round(self.sigma, 3)})"


class MatchHistory(BaseModel):
    challenger = ForeignKeyField(AI, backref="challenger_match_histories")
    defender = ForeignKeyField(AI, backref="defender_match_histories")
    win = SmallIntegerField()
    draw = SmallIntegerField()
    loss = SmallIntegerField()
    created_date = DateTimeField(default=datetime.datetime.now)


db.connect()
db.create_tables([AI, MatchHistory])


class Outcome(Enum):
    WIN = 1
    DRAW = 0
    LOSS = -1


class Match:
    def __init__(self, partial_obs: bool, match_up=None):
        # mode 0: rl-ai vs built-in-ai
        # mode 1: rl-ai vs rl-ai
        # mode 2: built-in-ai vs built-in-ai

        built_in_ais = None
        built_in_ais2 = None
        rl_ai = None
        rl_ai2 = None

        # determine mode
        rl_ais = []
        built_in_ais = []
        for ai in match_up:
            if ai[-3:] == ".pt":
                rl_ais += [ai]
            else:
                built_in_ais += [ai]
        if len(rl_ais) == 1:
            mode = 0
            p0 = rl_ais[0]
            p1 = built_in_ais[0]
            rl_ai = p0
            built_in_ais = [eval(f"microrts_ai.{p1}")]
        elif len(rl_ais) == 2:
            mode = 1
            p0 = rl_ais[0]
            p1 = rl_ais[1]
            rl_ai = p0
            rl_ai2 = p1
        else:
            mode = 2
            p0 = built_in_ais[0]
            p1 = built_in_ais[1]
            built_in_ais = [eval(f"microrts_ai.{p0}")]
            built_in_ais2 = [eval(f"microrts_ai.{p1}")]

        self.p0, self.p1 = p0, p1

        self.mode = mode
        self.partial_obs = partial_obs
        self.built_in_ais = built_in_ais
        self.built_in_ais2 = built_in_ais2
        self.rl_ai = rl_ai
        self.rl_ai2 = rl_ai2
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        max_steps = 5000
        if mode == 0:
            self.envs = MicroRTSGridModeVecEnv(
                num_bot_envs=len(built_in_ais),
                num_selfplay_envs=0,
                partial_obs=partial_obs,
                max_steps=max_steps,
                render_theme=2,
                ai2s=built_in_ais,
                map_paths=["maps/16x16/basesWorkers16x16A.xml"],
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
            )
            self.agent = Agent(self.envs).to(self.device)
            self.agent.load_state_dict(torch.load(self.rl_ai, map_location=self.device))
            self.agent.eval()
        elif mode == 1:
            self.envs = MicroRTSGridModeVecEnv(
                num_bot_envs=0,
                num_selfplay_envs=2,
                partial_obs=partial_obs,
                max_steps=max_steps,
                render_theme=2,
                map_paths=["maps/16x16/basesWorkers16x16A.xml"],
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
            )
            self.agent = Agent(self.envs).to(self.device)
            self.agent.load_state_dict(torch.load(self.rl_ai, map_location=self.device))
            self.agent.eval()
            self.agent2 = Agent(self.envs).to(self.device)
            self.agent2.load_state_dict(torch.load(self.rl_ai2, map_location=self.device))
            self.agent2.eval()
        else:
            self.envs = MicroRTSBotVecEnv(
                ai1s=built_in_ais,
                ai2s=built_in_ais2,
                max_steps=max_steps,
                render_theme=2,
                map_paths=["maps/16x16/basesWorkers16x16.xml"],
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
            )
        self.envs = MicroRTSStatsRecorder(self.envs)
        self.envs = VecMonitor(self.envs)

    def run(self, num_matches=7):
        if self.mode == 0:
            return self.run_m0(num_matches)
        elif self.mode == 1:
            return self.run_m1(num_matches)
        else:
            return self.run_m2(num_matches)

    def run_m0(self, num_matches):
        results = []
        16 * 16
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        while True:
            # self.envs.render()
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                mask = torch.tensor(np.array(self.envs.get_action_mask())).to(self.device)
                action, _, _, _, _ = self.agent.get_action_and_value(
                    next_obs, envs=self.envs, invalid_action_masks=mask, device=self.device
                )
            try:
                next_obs, rs, ds, infos = self.envs.step(action.cpu().numpy().reshape(self.envs.num_envs, -1))
                next_obs = torch.Tensor(next_obs).to(self.device)
            except Exception as e:
                e.printStackTrace()
                raise

            for idx, info in enumerate(infos):
                if "episode" in info.keys():
                    results += [info["microrts_stats"]["WinLossRewardFunction"]]
                    if len(results) >= num_matches:
                        return results

    def run_m1(self, num_matches):
        results = []
        16 * 16
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        while True:
            # self.envs.render()
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                mask = torch.tensor(np.array(self.envs.get_action_mask())).to(self.device)

                p1_obs = next_obs[::2]
                p2_obs = next_obs[1::2]
                p1_mask = mask[::2]
                p2_mask = mask[1::2]

                p1_action, _, _, _, _ = self.agent.get_action_and_value(
                    p1_obs, envs=self.envs, invalid_action_masks=p1_mask, device=self.device
                )
                p2_action, _, _, _, _ = self.agent2.get_action_and_value(
                    p2_obs, envs=self.envs, invalid_action_masks=p2_mask, device=self.device
                )
                action = torch.zeros((self.envs.num_envs, p2_action.shape[1], p2_action.shape[2]))
                action[::2] = p1_action
                action[1::2] = p2_action

            try:
                next_obs, rs, ds, infos = self.envs.step(action.cpu().numpy().reshape(self.envs.num_envs, -1))
                next_obs = torch.Tensor(next_obs).to(self.device)
            except Exception as e:
                e.printStackTrace()
                raise

            for idx, info in enumerate(infos):
                if "episode" in info.keys():
                    results += [info["microrts_stats"]["WinLossRewardFunction"]]
                    if len(results) >= num_matches:
                        return results

    def run_m2(self, num_matches):
        results = []
        self.envs.reset()
        while True:
            # self.envs.render()
            # dummy actions
            next_obs, reward, done, infos = self.envs.step(
                [
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ]
            )
            for idx, info in enumerate(infos):
                if "episode" in info.keys():
                    results += [info["microrts_stats"]["WinLossRewardFunction"]]
                    if len(results) >= num_matches:
                        return results


def get_ai_type(ai_name):
    if ai_name[-3:] == ".pt":
        return "rl_ai"
    else:
        return "built_in_ai"


def get_match_history(ai_name):
    query = (
        MatchHistory.select(
            AI.name,
            fn.SUM(MatchHistory.win).alias("wins"),
            fn.SUM(MatchHistory.draw).alias("draws"),
            fn.SUM(MatchHistory.loss).alias("losss"),
        )
        .join(AI, JOIN.LEFT_OUTER, on=MatchHistory.defender)
        .group_by(MatchHistory.defender)
        .where(MatchHistory.challenger == AI.get(name=ai_name))
    )
    return pd.DataFrame(list(query.dicts()))


def get_leaderboard():
    query = AI.select(
        AI.name,
        AI.mu,
        AI.sigma,
        (AI.mu - 3 * AI.sigma).alias("trueskill"),
    ).order_by((AI.mu - 3 * AI.sigma).desc())
    return pd.DataFrame(list(query.dicts()))


def get_leaderboard_existing_ais(existing_ai_names):
    query = (
        AI.select(
            AI.name,
            AI.mu,
            AI.sigma,
            (AI.mu - 3 * AI.sigma).alias("trueskill"),
        )
        .where((AI.name.in_(existing_ai_names)))
        .order_by((AI.mu - 3 * AI.sigma).desc())
    )
    return pd.DataFrame(list(query.dicts()))


if __name__ == "__main__":
    existing_ai_names = [item.name for item in AI.select()]
    all_ai_names = set(existing_ai_names + args.evals)

    for ai_name in all_ai_names:
        ai = AI.get_or_none(name=ai_name)
        if ai is None:
            ai = AI(name=ai_name, mu=25.0, sigma=8.333333333333334, ai_type=get_ai_type(ai_name))
            ai.save()

    # case 1: initialize the league with round robin
    if len(existing_ai_names) == 0:
        match_ups = list(itertools.combinations(all_ai_names, 2))
        np.random.shuffle(match_ups)
        for idx in range(2):  # switch player 1 and 2's starting locations
            for match_up in match_ups:
                if idx == 0:
                    match_up = list(reversed(match_up))

                m = Match(args.partial_obs, match_up)
                challenger = AI.get_or_none(name=m.p0)
                defender = AI.get_or_none(name=m.p1)

                r = m.run(args.num_matches // 2)
                for item in r:
                    drawn = False
                    if item == Outcome.WIN.value:
                        winner = challenger
                        loser = defender
                    elif item == Outcome.DRAW.value:
                        drawn = True
                    else:
                        winner = defender
                        loser = challenger

                    print(f"{winner.name} {'draws' if drawn else 'wins'} {loser.name}")

                    winner_rating, loser_rating = rate_1vs1(
                        Rating(winner.mu, winner.sigma), Rating(loser.mu, loser.sigma), drawn=drawn
                    )

                    winner.mu, winner.sigma = winner_rating.mu, winner_rating.sigma
                    loser.mu, loser.sigma = loser_rating.mu, loser_rating.sigma
                    winner.save()
                    loser.save()

                    MatchHistory(
                        challenger=challenger,
                        defender=defender,
                        win=int(item == 1),
                        draw=int(item == 0),
                        loss=int(item == -1),
                    ).save()
        get_leaderboard().to_csv(csvpath, index=False)

    # case 2: new AIs
    else:
        leaderboard = get_leaderboard_existing_ais(existing_ai_names)
        new_ai_names = [ai_name for ai_name in args.evals if ai_name not in existing_ai_names]
        for new_ai_name in new_ai_names:
            ai = AI.get(name=new_ai_name)

            while ai.sigma > args.highest_sigma:

                match_qualities = []
                for ai2_name in leaderboard["name"]:
                    opponent_ai = AI.get(name=ai2_name)
                    if ai.name == opponent_ai.name:
                        continue
                    match_qualities += [[opponent_ai, quality_1vs1(ai, opponent_ai)]]

                # sort by quality
                match_qualities = sorted(match_qualities, key=lambda x: x[1], reverse=True)
                print("match_qualities[:3]", match_qualities[:3])

                # run a match if the quality of the opponent is high enough
                top_3_ai = [item[0] for item in match_qualities[:3]]
                opponent_ai = random.choice(top_3_ai)
                match_up = (ai.name, opponent_ai.name)
                match_quality = quality_1vs1(ai, opponent_ai)
                print(f"the match up is ({ai}, {opponent_ai}), quality is {round(match_quality, 4)}")
                winner = ai  # dummy setting
                for idx in range(2):  # switch player 1 and 2's starting locations
                    if idx == 0:
                        match_up = list(reversed(match_up))
                    m = Match(args.partial_obs, match_up)
                    challenger = AI.get(name=m.p0)
                    defender = AI.get(name=m.p1)
                    r = m.run(1)
                    for item in r:
                        drawn = False
                        if item == Outcome.WIN.value:
                            winner = challenger
                            loser = defender
                        elif item == Outcome.DRAW.value:
                            drawn = True
                            winner = defender
                            loser = challenger
                        else:
                            winner = defender
                            loser = challenger
                        print(f"{winner.name} {'draws' if drawn else 'wins'} {loser.name}")
                        winner_rating, loser_rating = rate_1vs1(
                            Rating(winner.mu, winner.sigma), Rating(loser.mu, loser.sigma), drawn=drawn
                        )

                        # freeze existing AIs ratings
                        if winner.name == ai.name:
                            ai.mu, ai.sigma = winner_rating.mu, winner_rating.sigma
                            ai.save()
                        else:
                            ai.mu, ai.sigma = loser_rating.mu, loser_rating.sigma
                            ai.save()
                        MatchHistory(
                            challenger=challenger,
                            defender=defender,
                            win=int(item == 1),
                            draw=int(item == 0),
                            loss=int(item == -1),
                        ).save()

        get_leaderboard().to_csv(args.output_path, index=False)

    print("=======================")
    print(get_leaderboard())
    if not args.update_db:
        os.remove(dbpath)
