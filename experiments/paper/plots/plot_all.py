import pandas as pd
import seaborn as sns
import pickle
import argparse
import wandb
import matplotlib.pyplot as plt
import os
api = wandb.Api()
sns.set_context("talk")


parser = argparse.ArgumentParser(description='CleanRL Plots')
parser.add_argument('--feature-of-interest', type=str, default='charts/episode_reward',
                    help='which feature to be plotted on the y-axis')
parser.add_argument('--smooth-weight', type=float, default=0.99,
                    help='the weight parameter of the exponential moving average')
args = parser.parse_args()


# Change oreilly-class/cifar to <entity/project-name>
keys = ['charts/episode_reward/WinLossRewardFunction', 'charts/episode_reward', 'global_step']
if not os.path.exists("raw_data.pkl"):
    raw_data = []
    runs = api.runs("vwxyzjn/gym-microrts-paper")
    for run in runs: 
        raw_data += [[run.config, run.history(keys=keys), run.summary]]
    with open('raw_data.pkl', 'wb') as handle:
        pickle.dump(raw_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('raw_data.pkl', 'rb') as handle:
        raw_data = pickle.load(handle)
        
config_df = pd.DataFrame([item[0] for item in raw_data])

name2label = {
    "ppo_diverse_impala": "PPO + invalid action masking \n+ diverse opponents + IMPALA-CNN",
    "ppo_diverse": "PPO + invalid action masking \n + diverse opponents",
    "ppo_coacai": "PPO + invalid action masking",
    "ppo_coacai_naive": "PPO + naive invalid action masking",
    "ppo_coacai_partial_mask": "PPO + partial invalid action masking",
    "ppo_coacai_no_mask": "PPO",
    "ppo_gridnet_selfplay_diverse_encode_decode": "PPO + invalid action masking +\nhalf self-play / half bots + encoder-decoder",
    "ppo_gridnet_selfplay_encode_decode": "PPO + invalid action masking \n+ selfplay + encoder-decoder",
    "ppo_gridnet_diverse_encode_decode": "PPO + invalid action masking \n+ diverse opponents + encoder-decoder",
    "ppo_gridnet_diverse_impala": "PPO + invalid action masking \n + diverse opponents + IMPALA-CNN",
    "ppo_gridnet_diverse": "PPO + invalid action masking \n + diverse opponents",
    "ppo_gridnet_coacai": "PPO + invalid action masking",
    "ppo_gridnet_coacai_naive": "PPO + naive invalid action masking",
    "ppo_gridnet_coacai_partial_mask": "PPO + partial invalid action masking",
    "ppo_gridnet_coacai_no_mask": "PPO",
}

# https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar#_=_
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

def plot_data(data, smooth_weight, foi, ax=None):
    valid_idxs = data[foi].notnull()
    x = list(data['global_step'][valid_idxs])
    y = data[foi][valid_idxs]
    y = smooth(list(y), args.smooth_weight)
    sns.lineplot(x=x, y=y, ax=ax)

if not os.path.exists("shaped_vs_sparse_plots"):
    os.makedirs("shaped_vs_sparse_plots")

exp_names = list(set(config_df['exp_name']))
for exp_name in exp_names:
    if exp_name not in name2label:
        continue
    exp_idxs = list(config_df.loc[config_df['exp_name']==exp_name].index)
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for exp_idx in exp_idxs:
        plot_data(raw_data[exp_idx][1], args.smooth_weight, "charts/episode_reward", ax1)
        plot_data(raw_data[exp_idx][1], args.smooth_weight, "charts/episode_reward/WinLossRewardFunction", ax2)
    ax1.set_ylabel("Shaped return")
    ax1.legend(ax1.get_lines(), [
        "Seed 1",
        "Seed 2",
        "Seed 3",
        "Seed 4"
    ])
    ax2.set_ylabel("Sparse return")
    f.suptitle(name2label[raw_data[exp_idx][0]['exp_name']])
    f.tight_layout()
    f.savefig(f"shaped_vs_sparse_plots/{raw_data[exp_idx][0]['exp_name']}.pdf")
