import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_context("talk")
df = pd.DataFrame(
    [["line 1", 20, 30, 100], ["line 2", 10, 40, 90], ["line 3", 10, 35, 120]], columns=["element", "var 1", "var 2", "var 3"]
)
pd.plotting.parallel_coordinates(df, "element")
plt.show()

name2label = {
    "ppo_diverse_impala": "PPO + invalid action masking + diverse opponents+ IMPALA-CNN",
    "ppo_diverse": "PPO + invalid action masking + diverse opponents",
    "ppo_coacai": "PPO + invalid action masking",
    "ppo_coacai_naive": "PPO + naive invalid action masking",
    "ppo_coacai_partial_mask": "PPO + partial invalid action masking",
    "ppo_coacai_no_mask": "PPO",
    "ppo_gridnet_diverse_encode_decode": "PPO + invalid action masking \n+ diverse opponents + encoder-decoder",
    "ppo_gridnet_diverse_impala": "PPO + invalid action masking \n + diverse opponents + IMPALA-CNN",
    "ppo_gridnet_diverse": "PPO + invalid action masking \n + diverse opponents",
    "ppo_gridnet_coacai": "PPO + invalid action masking",
    "ppo_gridnet_coacai_naive": "PPO + naive invalid action masking",
    "ppo_gridnet_coacai_partial_mask": "PPO + partial invalid action masking",
}
data = pd.read_csv("uas_parallel.csv")
data["exp_name"] = data["exp_name"].map(name2label)
del data["Name"]

data["charts/cumulative_match_results/win rate"] *= max(data["charts/total_parameters"])
data = data.rename(
    columns={"charts/cumulative_match_results/win rate": "Cumulative Win Rate", "charts/total_parameters": "Total Parameters"}
)
fig, ax1 = plt.subplots(figsize=(9, 8))
pd.plotting.parallel_coordinates(data, "exp_name", ax=ax1)

ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
ax2 = ax1.twinx()
ax2.set_ylim(0, 1.0)
fig.tight_layout()
fig.savefig("uas_parallel.pdf")

# import numpy as np
# import matplotlib.pyplot as plt

# mean, amp = 40000, 20000
# t = np.arange(50)
# s1 = np.sin(t)*amp + mean #synthetic ts, but closer to my data

# fig, ax1 = plt.subplots()
# ax1.plot(t, s1, 'b-')

# ax1.set_xlabel('time')
# mn, mx = ax1.set_ylim(mean-amp, mean+amp)
# ax1.set_ylabel('km$^3$/year')

# km3yearToSv = 31.6887646e-6

# ax2 = ax1.twinx()
# ax2.set_ylim(mn*km3yearToSv, mx*km3yearToSv)
# ax2.set_ylabel('Sv')
