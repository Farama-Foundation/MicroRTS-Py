from os import path
import pickle
import wandb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='CleanRL Plots')
# Common arguments
parser.add_argument('--wandb-project', type=str, default="vwxyzjn/gym-microrts-paper",
                   help='the name of wandb project (e.g. cleanrl/cleanrl)')
args = parser.parse_args()
api = wandb.Api()
runs = api.runs(args.wandb_project)

for idx, run in enumerate(runs):
    exp_name = run.config['exp_name']
    if not os.path.exists(f"trained_models/{exp_name}"):
        os.makedirs(f"trained_models/{exp_name}")
    if not os.path.exists(f"trained_models/{exp_name}/agent-{run.config['seed']}.pt"):
        trained_model = run.file('agent.pt')
        trained_model.download(f"trained_models/{exp_name}")
        os.rename(f"trained_models/{exp_name}/agent.pt",
                f"trained_models/{exp_name}/agent-{run.config['seed']}.pt")
