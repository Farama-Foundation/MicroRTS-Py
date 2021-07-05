import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# sns.set_context("paper", font_scale=1.4)
# plt.rcParams['figure.figsize']=(5,2)
# plt.style.use('science')


name2label = {
    "exp_name: ppo_diverse_impala": "PPO + invalid action masking \n+ diverse bots + IMPALA-CNN",
    "exp_name: ppo_diverse": "PPO + invalid action masking \n + diverse bots",
    "exp_name: ppo_coacai": "PPO + invalid action masking",
    "exp_name: ppo_coacai_naive": "PPO + naive invalid action masking",
    "exp_name: ppo_coacai_partial_mask": "PPO + partial invalid action masking",
    "exp_name: ppo_coacai_no_mask": "PPO",
}

name2label2 = {
    "exp_name: ppo_gridnet_diverse_encode_decode": "PPO + invalid action masking \n+ diverse bots + encoder-decoder",
    "exp_name: ppo_gridnet_diverse_impala": "PPO + invalid action masking \n + diverse bots + IMPALA-CNN",
    "exp_name: ppo_gridnet_diverse": "PPO + invalid action masking \n + diverse bots",
    "exp_name: ppo_gridnet_coacai": "PPO + invalid action masking",
    "exp_name: ppo_gridnet_selfplay_diverse_encode_decode":  "PPO + invalid action masking +\nhalf self-play / half bots + encoder-decoder",
    "exp_name: ppo_gridnet_selfplay_encode_decode":  "PPO + invalid action masking \n+ selfplay + encoder-decoder",
    "exp_name: ppo_gridnet_coacai_naive": "PPO + naive invalid action masking",
    "exp_name: ppo_gridnet_coacai_partial_mask": "PPO + partial invalid action masking",
    "exp_name: ppo_gridnet_coacai_no_mask": "PPO",
}

data = pd.read_csv("uas.csv")
data['Name'] = data['Name'].map(name2label)
data = data[data['Name'].notna()]
data = data.set_index('Name')
data = data.reindex(list(name2label.values()))

data2 = pd.read_csv("gridnet.csv")
data2['Name'] = data2['Name'].map(name2label2)
data2 = data2[data2['Name'].notna()]
data2 = data2.set_index('Name')
data2 = data2.reindex(list(name2label2.values()))

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("talk")
rs = np.random.RandomState(8)

# Set up the matplotlib figure
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.6]})

# sns.barplot(x=x, y=y1, palette="rocket", ax=ax1)
palette = sns.color_palette("magma", n_colors=len(data))
# palette.reverse()

bar1 = sns.barplot(
    data=data,
    y=data.index,
    x='charts/cumulative_match_results/win rate',
    orient='h',
    capsize=.5,
    ax=ax1,
    palette=palette)
ax1.set_title("UAS")
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.set_xlim(right=1.2)
for i, v in enumerate(data['charts/cumulative_match_results/win rate']):
    ax1.text(max(0.05, v +0.05), i+0.2, str(round(v, 2)))


bar2 = sns.barplot(
    data=data2,
    y=data2.index,
    x='charts/cumulative_match_results/win rate',
    orient='h',
    capsize=.5,
    ax=ax2,
    palette=palette)
ax2.set_title("Gridnet")
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.set_xlim(right=1.2)
for i, v in enumerate(data2['charts/cumulative_match_results/win rate']):
    ax2.text(max(0.05, v +0.05), i+0.2, str(round(v, 2)))
f.tight_layout()
f.savefig("ablation_all.pdf")
# ax1.axhline(0, color="k", clip_on=False)
# ax1.set_ylabel("Sequential")