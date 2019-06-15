from dataclasses import dataclass
from typing import List, Any, Dict, Optional

ACTION_TYPE_NONE = 0
ACTION_TYPE_MOVE = 1
ACTION_TYPE_HARVEST = 2
ACTION_TYPE_RETURN = 3
ACTION_TYPE_PRODUCE = 4
ACTION_TYPE_ATTACK_LOCATION = 5

ACTION_PARAMETER_DIRECTION_NONE = -1
ACTION_PARAMETER_DIRECTION_UP = 0
ACTION_PARAMETER_DIRECTION_RIGHT = 1
ACTION_PARAMETER_DIRECTION_DOWN = 2
ACTION_PARAMETER_DIRECTION_LEFT = 3

@dataclass
class MicrortsMessage:
    reward: float
    observation: List[List[List[int]]]
    done: bool
    info: Dict

@dataclass
class UnitType:
    id: int
    name: str
    cost: int
    hp: int
    min_damage: int
    max_damage: int
    attack_range: int
    produce_time: int
    move_time: int
    attack_time: int
    harvest_time: int
    return_time: int
    harvest_amount: int
    sight_radius: int
    is_resource: bool
    is_stockpile: bool
    can_harvest: bool
    can_move: bool
    can_attack: bool
    produces: List[str]
    produced_by: List[str]


@dataclass
class GameInfo:
    move_conflict_resolution_strategy: int
    unit_types: List[UnitType]


@dataclass
class Player:
    id: int
    resources: int


@dataclass
class Unit:
    type: str
    id: int
    player: int
    x: int
    y: int
    resources: int
    hitpoints: int


@dataclass
class Pgs:
    width: int
    height: int
    terrain: str
    players: List[Player]
    units: List[Unit]


@dataclass
class GameState:
    time: int
    pgs: Pgs
    actions: List[Any]

@dataclass
class Config:
    ai1_type: str
    ai2_type: str
    map_path: str
    render: bool
    auto_port: Optional[bool] = False
    client_port: Optional[int] = 0
    microrts_path: Optional[str] = ""
    microrts_repo_path: Optional[str] = ""
    maximum_t: Optional[int] = 2000
    client_ip: Optional[str] = "127.0.0.1"
    height: Optional[int] = 0
    width: Optional[int] = 0
    window_size: Optional[int] = 1
