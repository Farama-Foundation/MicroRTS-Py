def randomBiasedAI(utt):
    from ai import RandomBiasedAI
    return RandomBiasedAI()

def randomAI(utt):
    from ai import RandomBiasedSingleUnitAI
    return RandomBiasedSingleUnitAI()

def passiveAI(utt):
    from ai import PassiveAI
    return PassiveAI()

def workerRushAI(utt):
    from ai.abstraction import WorkerRush
    return WorkerRush(utt)

def lightRushAI(utt):
    from ai.abstraction import LightRush
    return LightRush(utt)

def POLightRush(utt):
    from ai.abstraction.partialobservability import POLightRush
    return POLightRush(utt)

def POWorkerRush(utt):
    from ai.abstraction.partialobservability import POWorkerRush
    return POWorkerRush(utt)

def POHeavyRush(utt):
    from ai.abstraction.partialobservability import POHeavyRush
    return POHeavyRush(utt)

def PORangedRush(utt):
    from ai.abstraction.partialobservability import PORangedRush
    return PORangedRush(utt)
# Competition AIs

def coacAI(utt):
    from ai.coac import CoacAI
    return CoacAI(utt)

def naiveMCTSAI(utt):
    from ai.mcts.naivemcts import NaiveMCTS
    return NaiveMCTS(utt)

# https://github.com/AmoyZhp/MixedBotmRTS
def mixedBot(utt):
    from ai.JZ import MixedBot
    return MixedBot(utt)

# https://github.com/jr9Hernandez/RojoBot
def rojo(utt):
    from ai.competition.rojobot import Rojo
    return Rojo(utt)

# https://github.com/rubensolv/IzanagiBot
def izanagi(utt):
    from ai.competition.IzanagiBot import Izanagi
    return Izanagi(utt)

# https://github.com/jr9Hernandez/TiamatBot
def tiamat(utt):
    from ai.competition.tiamat import Tiamat
    return Tiamat(utt)

# https://github.com/zuozhiyang/Droplet/blob/master/GNS/Droplet.java
def droplet(utt):
    from GNS import Droplet
    return Droplet(utt)

# https://github.com/barvazkrav/mayariBot/blob/master/mayari.java
def mayari(utt):
    from mayariBot import mayari
    return mayari(utt)

# # https://github.com/zuozhiyang/MentalSeal
# def mentalSeal(utt):
#     from MentalSeal import MentalSeal
#     return MentalSeal(utt)

# https://github.com/rubensolv/GRojoA3N
def guidedRojoA3N(utt):
    from ai.competition.GRojoA3N import GuidedRojoA3N
    return GuidedRojoA3N(utt)

ALL_AIS = [
    randomBiasedAI,
    randomAI,
    passiveAI,
    workerRushAI,
    lightRushAI,
    coacAI,
    naiveMCTSAI,
]