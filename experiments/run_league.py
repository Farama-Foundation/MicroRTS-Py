import argparse
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
import importlib
import multiprocessing as mp
import numpy as np
from operator import itemgetter
import os.path
import pandas as pd
from pathlib import Path
import pandas as pd
import pickle
import random
import signal
import shutil
import time
from trueskill import Rating, quality_1vs1, rate_1vs1
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import yaml


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def prepare_downstream_args(args, downstream_name: str):
    parser = argparse.ArgumentParser()
    downstream_args = parser.parse_args([])
    downstream_args.__dict__.update(args.config[downstream_name]["args"])
    return downstream_args


def load_entrypoint(args, entrypoint_name: str):
    module, fn = args.config[entrypoint_name]["entrypoint"].split(":")
    return getattr(importlib.import_module(module), fn)


def load_algorithms(args) -> Tuple[Callable, Callable, Callable, Callable]:
    matchmaking_algo = args.config["matchmaking"]["algorithm"]
    # xxx(okachaiev): enum? preset classes?
    if matchmaking_algo == "alphastar":
        league_pick_opponent = alphstar_pick_opponent
    elif matchmaking_algo == "openfive":
        league_pick_opponent = openfive_pick_opponent
    elif matchmaking_algo == "custom":
        league_pick_opponent = load_entrypoint(args, "matchmaking")
    else:
        raise ValueError("Unsupported matchmaking algorithm")

    archive_algo = args.config["archive"]["algorithm"]
    if archive_algo == "alphastar":
        league_requires_archival = alphastar_requires_archival
    elif archive_algo == "openfive":
        league_requires_archival = openfive_requires_archival
    elif archive_algo == "custom":
        league_requires_archival = load_entrypoint(args, "archive")
    else:
        raise ValueError("Unsupported archival algorithm")

    # xxx(okachaiev): add flexiblity here in future
    league_requires_evaluation = league_requires_archival

    bootstrap_algo = args.config["matchmaking"]["bootstrap"]
    if bootstrap_algo == "random":
        league_bootstrap_opponent = random_bootstrap_opponent
    elif bootstrap_algo == "none":
        # xxx(okachaiev): as of now, this option makes no sense because
        # a new opponent is picked up only when previous game is finished
        # thus bootstrapping ensures that matchmaking is invoked
        # i should use normal "pick opponent" when bootstrap is skipped
        league_bootstrap_opponent = noop_bootstrap_opponent
    else:
        raise ValueError("Unsupported bootstrap algorithm")

    return (league_bootstrap_opponent, league_pick_opponent, league_requires_archival, league_requires_evaluation)


def parse_args():
    parser = argparse.ArgumentParser()
    # xxx(okachaiev): find balance for what goes into config file vs. what is defined here
    # i think things like folder, initial_agents, etc should be in the script
    # annd file should only be used as a set of "hyperparams" for the league and it's structure
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--resume", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    args = parser.parse_args()
    assert args.config_file.exists()
    # xxx(okachaiev): I need to find a better way to manage args + configuration file
    with args.config_file.open("r") as f:
        args.config = yaml.safe_load(f)

    args.checkpoints_folder = Path(args.config["league"]["folder"]).joinpath("checkpoints")
    args.checkpoints_folder.mkdir(exist_ok=True)
    args.agents_folder = Path(args.config["league"]["folder"]).joinpath("agents")
    args.agents_folder.mkdir(exist_ok=True)

    args.train_entrypoint = load_entrypoint(args, "train")
    args.evaluate_entrypoint = load_entrypoint(args, "evaluate")

    return args


def run_train_loop(args, match):
    train_args = prepare_downstream_args(args, "train")
    # xxx(okachaiev): I should probably create a copy of args namespace here
    train_args.seed = np.random.randint(0, 1_000_000)
    (wins, losses, draws), info = args.train_entrypoint(train_args, str(match.p1), str(match.p2))
    match_result = MatchResult.for_match(match, (wins, losses, draws), info)

    # xxx(okachaiev): good logger would be nice to have
    print(f"Finished match up {match.p1} vs. {match.p2}: {match_result.payoff()}")

    return match_result


def find_reference_opponents(args):
    # xxx(okachaiev): what about PvE agents?
    # xxx(okachaiev): definitely need to be more flexible here, e.g. SB3 produces .zip files
    reference_opponent_path = list(Path(args.config["evaluate"]["reference_agents_folder"]).glob("*.pt"))
    return [(p.stem, str(p)) for p in reference_opponent_path]


def prepare_eval_reference(args):
    eval_args = prepare_downstream_args(args, "evaluate")

    # if leaderboard is there and not forced, no need to run
    # xxx(okachaiev): ideally this should be "additive", e.g. when min num games
    # is more than was actually played, we can just play more games. or if a
    # new agent was added
    leaderboard_path = Path(args.config["evaluate"]["reference_agents_folder"]).joinpath("leaderboard.csv")
    if leaderboard_path.exists():
        print("Evaluation leaderboard exists, bailing out")
        print(pd.read_csv(leaderboard_path))
        return

    reference_opponents = find_reference_opponents(args)
    assert len(reference_opponents) > 0, "No reference agents found"

    # xxx(okachaiev): should have configuration for start mu/var
    mmr = defaultdict(lambda: (0, Rating()))    

    # xxx(okachaiev): algo should be pluggable, with good presets
    pairs = list(combinations(reference_opponents, 2))
    for _ in range(args.config["evaluate"]["matches_per_opponent"]):
        # xxx(okachaiev): i can use joblib to run in parallel
        # we don't need to run MMR eval immediately after the game finished
        # as we do full round-robin anyways
        np.random.shuffle(pairs)
        for (p1_name, p1_path), (p2_name, p2_path) in pairs:
            (p1_num_games, p1_rating), (p2_num_games, p2_rating) = mmr[p1_name], mmr[p2_name]

            print(f"Playing {p1_name} vs. {p2_name}")

            (wins, losses, draws) = args.evaluate_entrypoint(eval_args, p1_path, p2_path)

            print(f"Evaluation for {p1_name} vs. {p2_name}: {(wins, losses, draws)}")

            num_games = wins + losses + draws

            if wins > 0:
                for _ in range(wins):
                    p1_rating, p2_rating = rate_1vs1(
                        p1_rating, p2_rating, drawn=False
                    )

            if losses > 0:
                for _ in range(losses):
                    p2_rating, p1_rating = rate_1vs1(
                        p2_rating, p1_rating, drawn=False
                    )

            if draws > 0:
                for _ in range(draws):
                    p1_rating, p2_rating = rate_1vs1(
                        p1_rating, p2_rating, drawn=True
                    )

            mmr[p1_name] = (p1_num_games+num_games, p1_rating)
            mmr[p2_name] = (p2_num_games+num_games, p2_rating)

            print(f"New ratings {p1_name}={p1_rating} vs. {p2_name}={p2_rating}")

    rows = [(name, num_games, rating.mu, rating.sigma) for name, (num_games, rating) in mmr.items()]
    rows.sort(key=itemgetter(2), reverse=True)

    df = pd.DataFrame(rows, columns=("name", "num_games", "mu", "sigma"))
    df.to_csv(leaderboard_path)

    print(df)


@dataclass
class EvaluationResult:
    p1: str
    mmr: Rating

# xxx(okachaiev): have to update this to respect flexibility with MMR algos
def run_eval_loop(args, player, shutdown_flag):
    eval_args = prepare_downstream_args(args, "evaluate")

    leaderboard_file = Path(args.config["evaluate"]["reference_agents_folder"]).joinpath("leaderboard.csv")

    assert leaderboard_file.exists(), "MMR evaluation for reference agents should be provided"

    leaderboard = pd.read_csv(leaderboard_file)
    reference_opponents = find_reference_opponents(args)

    assert set(leaderboard.name) == set([name for name, _ in reference_opponents]), "All reference agents have to be evaluated"

    mmr = {
        name: Rating(mu, sigma)
        for (name, mu, sigma)
        in leaderboard[["name", "mu", "sigma"]].to_numpy().tolist()
    }
    current_rating = Rating()

    # one-vs.-all strategy, this one should be pluggable
    for _ in range(args.config["evaluate"]["games_per_opponent"]):
        np.random.shuffle(reference_opponents)
        for (opponent_name, opponent_path) in reference_opponents:
            # the main process got interrupted
            if shutdown_flag.is_set(): return None

             # xxx(okachaiev): API for running batch eval (with multiple opponents)
            (wins, losses, draws) = args.evaluate_entrypoint(eval_args, player.save_path, opponent_path)
            opponent_rating = mmr[opponent_name]

            # xxx(okachaiev): code duplication
            if wins > 0:
                for _ in range(wins):
                    current_rating, opponent_rating = rate_1vs1(
                        current_rating, opponent_rating, drawn=False
                    )

            if losses > 0:
                for _ in range(losses):
                    opponent_rating, current_rating = rate_1vs1(
                        opponent_rating, current_rating, drawn=False
                    )

            if draws > 0:
                for _ in range(draws):
                    current_rating, opponent_rating = rate_1vs1(
                        current_rating, opponent_rating, drawn=True
                    )

    print(f"Trueskill evaluation for {player.save_path} is {current_rating}")

    # xxx(okachaiev): do we need to save it back????
    # it seems like overwrite is only necessary when we add new reference agents,
    # otherwise MMR has to stay the same

    return EvaluationResult(player.save_path, current_rating)


# xxx(okachaiev): should this one be named "Outcome"???
@dataclass
class Payoff:
    wins: int
    draws: int
    losses: int
    
    @classmethod
    def empty(cls) -> "Payoff":
        return cls(0, 0, 0)

    @property
    def num_games(self):
        return self.wins + self.draws + self.losses

    @property
    def winrate(self):
        if self.num_games == 0: return 0.5
        return (self.wins + 0.5*self.draws) / self.num_games

    def __add__(self, other):
        return self.__class__(self.wins + other.wins, self.draws + other.draws, self.losses + other.losses)

    def __iadd__(self, other):
        self.wins += other.wins
        self.draws += other.draws
        self.losses += other.losses
        return self

    def __neg__(self):
        return self.__class__(self.losses, self.draws, self.wins)


# xxx(okachaiev): using Enum here is extremely limited, unfortunately
class PlayerBracket(Enum):
    MAIN_PLAYER = 0
    MAIN_EXPLOITER = 1
    LEAGUE_EXPLOITER = 2
    ARCHIVED = 3

    @classmethod
    def from_str(cls, name: str) -> "PlayerBracket":
        name = name.lower()
        for opt in cls:
            if opt.name.lower() == name:
                return opt
        raise ValueError("Unknown player bracket")


@dataclass
class Player:
    bracket: PlayerBracket
    save_path: str
    # xxx(okachaiev): name seems more human-like than id
    name: str
    parent: Optional["Player"] = None

    @property
    def learner(self):
        return self.bracket != PlayerBracket.ARCHIVED

    # xxx(okachaiev): replace with fancy random generated names
    @staticmethod
    def _generate_name() -> str:
        t = str(int(time.time()))
        r = random.randint(0, 10_000)
        return f"{t}-{r:05d}"

    @staticmethod
    def _prepare_storage(bracket: PlayerBracket, save_dir: Path, player_name: str) -> Path:
        save_path = save_dir.joinpath(bracket.name.lower())
        save_path.mkdir(exist_ok=True, parents=True)
        return save_path.joinpath(f"{player_name}.pt")

    @classmethod
    def from_model(cls, agent_model_path: str, bracket: PlayerBracket, save_dir: Path) -> "Player":
        player_name = Player._generate_name()
        save_path = Player._prepare_storage(bracket, save_dir, player_name)
        shutil.copyfile(agent_model_path, save_path)
        return cls(bracket, str(save_path), player_name)

    def archive(self, save_dir: Path):
        new_player_name = Player._generate_name()
        save_path = Player._prepare_storage(PlayerBracket.ARCHIVED, save_dir, new_player_name)
        shutil.copyfile(self.save_path, save_path)
        return self.__class__(PlayerBracket.ARCHIVED, str(save_path), new_player_name, self)


@dataclass
class Match:
    match_id: str
    p1: str
    p2: str
    seed: int

    @classmethod
    def for_players(cls, p1: Player, p2: Player):
        match_id = f"{int(time.time())}-{random.randint(0, 99):03d}"
        return cls(match_id, p1.save_path, p2.save_path, np.random.randint(0, 1_000_000))


@dataclass
class MatchResult:
    p1: str
    p2: str
    wins: int
    losses: int
    draws: int
    info: Optional[Dict[str, Any]] = None

    @classmethod
    def for_match(
        cls,
        match: Match,
        result: Optional[Tuple[int, int, int]] = None,
        info: Optional[Dict[str, Any]] = None
    ) -> "MatchResult":
        wins, losses, draws = (0, 0, 0) if result is None else result
        return cls(match.p1, match.p2, wins, losses, draws, info)

    def payoff(self) -> Payoff:
        return Payoff(self.wins, self.draws, self.losses)


# xxx(okachaiev): i also need to include decay
# xxx(okachaiev): for checkpointing/display i better track MMR here as well
class PayoffTable:

    def __init__(self):
        # xxx(okachaiev): if i keep max players, i can setup matrix as np array
        # i bet that would work much faster because of nice mem layout
        self.payoffs = defaultdict(Payoff.empty)
        self.info = defaultdict(lambda: deque(maxlen=100))
        self.players = OrderedDict()
        self.player_attrs = defaultdict(dict)

    def add_player(self, new_player: Player) -> None:
        self.players[str(new_player.save_path)] = new_player

    def get_attr(self, player: Union[Player, str], name: str, default_value: Optional[Any] = None) -> Optional[Any]:
        player = player if isinstance(player, str) else str(player.save_path)
        return self.player_attrs[player].get(name, default_value)

    def set_attr(self, player: Union[str, Player], name: str, value: Any) -> None:
        player = player if isinstance(player, str) else str(player.save_path)
        self.player_attrs[player][name] = value

    def update(self, match_result: MatchResult):
        self.payoffs[match_result.p1, match_result.p2] += match_result.payoff()
        self.payoffs[match_result.p2, match_result.p1] += -match_result.payoff()
        self.info[match_result.p1].append(match_result.info)
        # xxx(okachaiev): precompute winrates????

    def winrate(self, p1, p2) -> float:
        pid1 = str(p1.save_path) if isinstance(p1, Player) else p1
        pid2 = str(p2.save_path) if isinstance(p2, Player) else p2
        return self.payoffs[pid1, pid2].winrate

    def filter_players(
        self,
        bracket: Optional[PlayerBracket] = None,
        # xxx(okachaiev): filtering by parent should be definitely done
        # differently. the only one bracket that can have parent is ARCHIVED
        parent: Optional[Union[Player, List[Player]]] = None
    ) -> List[Player]:
        opponents = []
        if parent is not None:
            parent = set([p.save_path for p in parent]) if isinstance(parent, list) else set([parent.save_path])
        for player in self.players.values():
            if bracket is not None and player.bracket != bracket:
                continue
            if parent is not None and (player.parent is None or player.parent.save_path not in parent):
                continue 
            opponents.append(player)
        return opponents

    # xxx(okachaiev): this API could be easily merged with "winrates" call
    def calculate_winrates(self, player: Player, opponents: Iterator[Player]) -> np.ndarray:
        return np.array([self.winrate(player, opponent) for opponent in opponents])

    def to_pandas(self):
        all_players = list(self.players.keys())
        rows = []    
        for player_name in all_players:
            winrates = self.calculate_winrates(player_name, all_players)
            num_games = [self.payoffs[player_name, p2].num_games for p2 in all_players]
            total_games = sum(num_games)
            avg_winrate = 0.5
            if total_games > 0:
                avg_winrate = np.average(winrates, weights=num_games)
            steps = self.get_attr(player_name, "steps_since_last_archive", 0)
            mmr = self.get_attr(player_name, "mmr", None)
            mmr = 0.0 if mmr is None else mmr.mu
            winrates = dict(enumerate(winrates))
            winrates.update(dict(name=player_name, num_games=total_games, mean=avg_winrate, train=steps, mmr=mmr))
            rows.append(winrates)
        static_columns = ["name", "train", "num_games", "mmr", "mean"]
        return pd.DataFrame(rows, columns=static_columns+list(range(len(all_players))))

    def __getstate__(self):
        return dict(
            payoffs=self.__dict__["payoffs"],
            players=self.__dict__["players"],
            player_attrs=self.__dict__["player_attrs"]
        )

    def __setstate__(self, attrs):
        self.__dict__.update(attrs)
        self.info = defaultdict(lambda: deque(maxlen=100))


class SelfplayBranch(Enum):
    ARCHIVED = 0
    VERIFY = 1
    NORMAL = 2


def choice(options: List[Player], p: np.ndarray) -> Optional[Player]:
    if len(options) == 0: return None
    if len(p) == 1: return options[0]
    return np.random.choice(options, p=p/np.linalg.norm(p, ord=1))


def remove_monotonic_suffix(winrates: np.ndarray) -> np.ndarray:
    suffix = np.arange(1, len(winrates))[np.diff(winrates) >= 0]
    return winrates[:suffix.max()+1] if len(suffix) else winrates


def alphstar_pick_opponent(payoff_table: PayoffTable, player: Player) -> Optional[Player]:
    """
    This is the main league workflow. This function is executed after
    each finished match up. It's responsible for finding the next match up
    (opponents) that need to be executed.

    Purposefully keep this as a single function to have visibility on
    how the league works.

    The AlphaStar paper that describes different players in the league:
    https://www.nature.com/articles/s41586-019-1724-z

    Pseudocode for AlphaStar paper: https://github.com/chengyu2/learning_alpha_star
    The most detailed explanation of branching logic could be found here:
    https://github.com/chengyu2/learning_alpha_star/blob/master/multiagent.py

    More on population-based RL: https://arxiv.org/pdf/1807.01281.pdf    
    """
    if player.bracket == PlayerBracket.ARCHIVED:
        # ARCHIVED players do not play as challengers, so this should not happen
        return None
    elif player.bracket == PlayerBracket.MAIN_EXPLOITER:
        potential_opponents = payoff_table.filter_players(bracket=PlayerBracket.MAIN_PLAYER)
        opponent = np.random.choice(potential_opponents)
        if payoff_table.winrate(player, opponent) > 0.1:
            return opponent
        prev_opponents = payoff_table.filter_players(bracket=PlayerBracket.ARCHIVED, parent=opponent)
        winrates = payoff_table.calculate_winrates(player, prev_opponents)
        return choice(prev_opponents, winrates*(1-winrates))
    elif player.bracket == PlayerBracket.LEAGUE_EXPLOITER:
        potential_opponents = payoff_table.filter_players(bracket=PlayerBracket.ARCHIVED)
        winrates = payoff_table.calculate_winrates(player, potential_opponents)
        return choice(potential_opponents, np.minimum(0.5, 1-winrates))
    elif player.bracket == PlayerBracket.MAIN_PLAYER:
        # sample multinomial to decide between playing against archived or main player
        r = np.random.choice([SelfplayBranch.ARCHIVED, SelfplayBranch.VERIFY, SelfplayBranch.NORMAL], p=[0.5,0.15,0.35])
        if r == SelfplayBranch.ARCHIVED:
            # matchup against archived player
            potential_opponents = payoff_table.filter_players(bracket=PlayerBracket.ARCHIVED)
            winrates = payoff_table.calculate_winrates(player, potential_opponents)
            return choice(potential_opponents, (1-winrates)**2)
        else:
            # matchup against another main player
            potential_opponents = payoff_table.filter_players(bracket=PlayerBracket.MAIN_PLAYER)
            opponent = np.random.choice(potential_opponents)
            if r == SelfplayBranch.VERIFY:
                # check out archived exploiters
                exploiters = payoff_table.filter_players(bracket=PlayerBracket.MAIN_EXPLOITER)
                prev_exploiters = payoff_table.filter_players(bracket=PlayerBracket.ARCHIVED, parent=exploiters)
                winrates = payoff_table.calculate_winrates(player, prev_exploiters)
                if len(winrates) and winrates.min() < 0.3:
                    return choice(prev_exploiters, (1-winrates)**2)
                # previous versions of the opponent
                prev_opponents = payoff_table.filter_players(bracket=PlayerBracket.ARCHIVED, parent=opponent)
                winrates = payoff_table.calculate_winrates(player, prev_opponents)
                winrates = remove_monotonic_suffix(winrates)
                if len(winrates) and winrates.min() < 0.7:
                    return choice(prev_opponents, winrates*(1-winrates))
                else:
                    return None
            elif payoff_table.winrate(player, opponent) > 0.3:
                # main player, opponent is not scary
                return opponent
            else:
                # main player, opponent is a bit hard
                potential_opponents = payoff_table.filter_players(bracket=PlayerBracket.ARCHIVED, parent=opponent)
                winrates = payoff_table.calculate_winrates(player, potential_opponents)
                return choice(potential_opponents, winrates*(1-winrates))


# xxx(okachaiev): i might be the case i need to merge "pick_opponent" and "requires_archival"
# into a sigle function. e.g. "on_step". though i don't want to lose clarity by doing so
def alphastar_requires_archival(args, payoff_table: PayoffTable, player: Player) -> bool:
    """
    Archival happens based on the winrate againts different set of opponents.
    """
    if player.bracket == PlayerBracket.ARCHIVED: return False
    steps = payoff_table.get_attr(player, "steps_since_last_archive", 0)
    if steps < args.config["archive"]["args"]["min_steps"]: return False
    if steps > args.config["archive"]["args"]["max_steps"]: return True
    if player.bracket == PlayerBracket.MAIN_PLAYER:
        opponents = payoff_table.filter_players(bracket=PlayerBracket.ARCHIVED)
    elif player.bracket == PlayerBracket.MAIN_EXPLOITER:
        opponents = payoff_table.filter_players(bracket=PlayerBracket.MAIN_PLAYER)
    elif player.bracket == PlayerBracket.LEAGUE_EXPLOITER:
        opponents = payoff_table.filter_players(bracket=PlayerBracket.ARCHIVED)
    winrates = payoff_table.calculate_winrates(player, opponents)
    return len(winrates) and winrates.min() > args.config["archive"]["args"]["winrate_threshold"]



def openfive_pick_opponent(payoff_table: PayoffTable, player: Player) -> Optional[Player]:
    """
    The agent has 80% chance to play against itself, and 20% chance to play against past self.
    """
    if player.bracket == PlayerBracket.ARCHIVED:
        # ARCHIVED players do not play as challengers, so this should not happen
        return None
    selfplay = np.random.rand() > 0.8
    if selfplay: return player
    opponents = payoff_table.filter_players(bracket=PlayerBracket.ARCHIVED, parent=player)
    if len(opponents):
        return choice(opponents, np.ones(len(opponents)))


def openfive_requires_archival(args, payoff_table: PayoffTable, player: Player) -> bool:
    """
    Archive each learning agent right after num_steps of training.
    """
    if player.bracket == PlayerBracket.ARCHIVED: return False
    steps = payoff_table.get_attr(player, "steps_since_last_archive", 0)
    return steps >= args.config["archive"]["args"]["num_steps"]


def random_bootstrap_opponent(args, player: Player, initial_players: List[Player]) -> Optional[List[Player]]:
    if not player.learner: return None
    # xxx(okachaiev): should the player itself being masked?
    return np.random.choice(initial_players, args.config["matchmaking"]["bootstrap_args"]["num_opponents"])


def noop_bootstrap_opponent(args, player: Player, initial_players: List[Player]) -> Optional[List[Player]]:
    return None


def league_checkpoint(args, payoff_table: PayoffTable) -> None:
    existing_files = list(args.checkpoints_folder.glob("*.pcl"))
    # xxx(okachaiev): should prefix be customizable?
    checkpoint_file = f"league-{int(time.time())}-{random.randint(1_000,10_000)}.pcl"
    checkpoint_path = args.checkpoints_folder.joinpath(checkpoint_file)

    print(f"Checkpointing league into {checkpoint_path}")

    with checkpoint_path.open("wb") as f:
        pickle.dump(payoff_table, f, pickle.HIGHEST_PROTOCOL)
    for prev_checkpoint in existing_files:
        prev_checkpoint.unlink()


def league_resume_from_checkpoint(args) -> Optional[PayoffTable]:
    if not args.checkpoints_folder.exists(): return None
    checkpoints = list(args.checkpoints_folder.glob("*.pcl"))
    if not checkpoints: return None

    print(f"Resuming operations from checkpoint {checkpoints[0]}")

    with checkpoints[0].open("rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    args = parse_args()

    (
        league_bootstrap_opponent,
        league_pick_opponent,
        league_requires_archival,
        league_requires_evaluation
    ) = load_algorithms(args)

    # before doing the training, let's make sure that evaluation league is ready
    prepare_eval_reference(args)

    # load from checkpoint (if any)
    payoff_table = league_resume_from_checkpoint(args) if args.resume else None
    if payoff_table is None:
        payoff_table = PayoffTable()

        # load league players, starting from bootstrap
        for agent_model_path in args.config["population"]["initial_agents"]:
            for group in args.config["population"]["structure"]:
                for _ in range(group["num_agents"]):
                    bracket = PlayerBracket.from_str(group["group"])
                    player = Player.from_model(agent_model_path, bracket=bracket, save_dir=args.agents_folder)
                    payoff_table.add_player(player)
                    if group["init_archive"]:
                        payoff_table.add_player(player.archive(save_dir=args.agents_folder))

    ctx = mp.get_context("forkserver")
    executor = ProcessPoolExecutor(max_workers=args.config["train"]["num_workers"], mp_context=ctx, initializer=init_worker)
    eval_executor = ProcessPoolExecutor(max_workers=args.config["evaluate"]["num_workers"], mp_context=ctx, initializer=init_worker)
    manager = mp.Manager()
    shutdown_flag = manager.Event()

    def sigint_handler(signal, frame):
        print("Interrupted, stopping executors")
        # xxx(okachaiev): this won't kill already running training processes
        shutdown_flag.set()
        executor.shutdown(cancel_futures=True)
        eval_executor.shutdown(cancel_futures=True)
        exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    # league loop
    print(f"Staring league with {len(payoff_table.players)} initial players")

    initial_players = list(payoff_table.players.values())
    # xxx(okachaiev): using dicts for mutable values is ... weird
    # would be much better if I can put it into a "League" class or something
    num_scheduled_matches, played_matches = dict(value=0), dict(value=0)
    scheduled_futures = set()

    def process_match_result(future, scheduled_futures):
        if future.cancelled(): return

        match_result = future.result()
        if match_result is None:
            return

        if isinstance(match_result, EvaluationResult):
            payoff_table.set_attr(match_result.p1, "mmr", match_result.mmr)
            league_checkpoint(args, payoff_table)
            return

        played_matches["value"] += 1
        payoff_table.update(match_result)

        # visualize into terminal
        # xxx(okachaiev): use writer to update TensorBoard/W&B
        with pd.option_context("display.float_format", "{:,.4f}".format):
            print(payoff_table.to_pandas())

        winrate = payoff_table.winrate(match_result.p1, match_result.p2)

        print(f"Updated winrate {match_result.p1} vs. {match_result.p2}: {winrate}")

        player = payoff_table.players[match_result.p1]

        # needs evaluation?
        if league_requires_evaluation(args, payoff_table, player):
            print(f"Scheduled evaluation for {player.save_path}...")
            scheduled_futures.add(eval_executor.submit(run_eval_loop, args, player, shutdown_flag))

        # ready to be archived?
        if league_requires_archival(args, payoff_table, player):
            print(f"Archiving {player}...")
            payoff_table.add_player(player.archive(save_dir=args.agents_folder))
            payoff_table.set_attr(player, "steps_since_last_archive", 0)
        else:
            steps = payoff_table.get_attr(player, "steps_since_last_archive", 0)
            steps += args.config["train"]["args"]["total_timesteps"]
            payoff_table.set_attr(player, "steps_since_last_archive", steps)

        league_checkpoint(args, payoff_table)

        # xxx(okachaiev): if we only produce 1 game or None, number of game
        # will decrease overtime :thinking: in this case i need to pick
        # a random pair of agents to match up
        next_opponent = league_pick_opponent(payoff_table, player)
        if next_opponent:
            schedule(Match.for_players(player, next_opponent), scheduled_futures)

    def schedule(match, scheduled_matches):
        if num_scheduled_matches["value"] >= args.config["league"]["max_matches"]: return

        print(f"Queued match {match.p1} vs. {match.p2}")

        future = executor.submit(run_train_loop, args, match)
        scheduled_matches.add(future)
        num_scheduled_matches["value"] += 1

    # xxx(okachaiev): when resumed from checkpoint, do I need to run initial games?
    # i probably need to keep flag that initialization is finished and we should switch
    # to a normal "pick opponent" procedure
    for player in initial_players:
        # xxx(okachaiev): is there a need to have option to get all players vs.all players API?
        opponents = league_bootstrap_opponent(args, player, initial_players)
        if opponents is not None:
            for opponent in opponents:
                schedule(Match.for_players(player, opponent), scheduled_futures)

    while len(scheduled_futures):
        done, scheduled_futures = concurrent.futures.wait(scheduled_futures, return_when=concurrent.futures.FIRST_COMPLETED)
        for match_result in done:
            process_match_result(match_result, scheduled_futures)

    shutdown_flag.set()
    executor.shutdown(wait=True, cancel_futures=False)
    eval_executor.shutdown(wait=True, cancel_futures=False)