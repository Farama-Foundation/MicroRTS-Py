import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import argparse
from distutils.util import strtobool
# sns.set_context("paper", font_scale=1.4)
# plt.rcParams['figure.figsize']=(5,2)
# plt.style.use('science')
sns.set_context("talk")
rs = np.random.RandomState(8)

parser = argparse.ArgumentParser(description='CleanRL Plots')
parser.add_argument('--feature-of-interest', type=str, default='charts/episode_reward',
                   help='which feature to be plotted on the y-axis')
parser.add_argument('--smooth-weight', type=float, default=0.99,
                    help='the weight parameter of the exponential moving average')
args = parser.parse_args()

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
    interested_columns = [c for c in data.columns if c[-len(foi):] == foi]
    for c in interested_columns:
        x = list(data['global_step'][data[c].notnull()])
        y = data[c][data[c].notnull()]
        y = smooth(list(y), args.smooth_weight)
        sns.lineplot(x=x, y=y, ax=ax)

data = pd.read_csv("shaped.csv")
data2 = pd.read_csv("sparse.csv")

# Set up the matplotlib figure
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
plot_data(data, args.smooth_weight, "charts/episode_reward", ax1)
ax1.set_ylabel("Shaped return")
ax1.legend(ax1.get_lines(), [
    "Seed 1",
    "Seed 2",
    "Seed 3",
    "Seed 4"
])
plot_data(data2, args.smooth_weight, "charts/episode_reward/WinLossRewardFunction", ax2)
ax2.set_ylabel("Sparse return")
# f.suptitle("PPO + invalid action masking + diverse opponents  + impala cnn")
f.tight_layout()
f.savefig("shaped_vs_sparse.pdf")