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
        raw_data += [[run.config, run.history(keys=keys)]]
    with open('raw_data.pkl', 'wb') as handle:
        pickle.dump(raw_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('raw_data.pkl', 'rb') as handle:
        raw_data = pickle.load(handle)
        
config_df = pd.DataFrame([item[0] for item in raw_data])

name2label = {
    "ppo_coacai": "PPO + invalid action masking",
    "ppo_gridnet_coacai": "PPO + invalid action masking",
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

def plot_data(data, smooth_weight, foi, ax=None, color=None):
    valid_idxs = data[foi].notnull()
    x = list(data['global_step'][valid_idxs])
    y = data[foi][valid_idxs]
    y = smooth(list(y), args.smooth_weight)
    sns.lineplot(x=x, y=y, ax=ax, color=color)

if not os.path.exists("shaped_vs_sparse_plots"):
    os.makedirs("shaped_vs_sparse_plots")

colors = sns.color_palette(n_colors=2)
exp_names = list(name2label.keys())
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for idx, exp_name in enumerate(exp_names):
    if exp_name not in name2label:
        continue
    exp_idxs = list(config_df.loc[config_df['exp_name']==exp_name].index)
    for exp_idx in exp_idxs:
        plot_data(raw_data[exp_idx][1], args.smooth_weight, "charts/episode_reward", ax1, colors[idx])
        plot_data(raw_data[exp_idx][1], args.smooth_weight, "charts/episode_reward/WinLossRewardFunction", ax2, colors[idx])
    ax1.set_ylabel("Shaped return")
    ax2.set_ylabel("Sparse return")
ax1.legend([list(ax1.get_lines())[0], list(ax1.get_lines())[-1]], [
    "UAS",
    "Gridnet",
], loc='lower right')
# f.suptitle(name2label[raw_data[exp_idx][0]['exp_name']])
f.tight_layout()
f.savefig(f"uas_vs_gridnet.pdf")
