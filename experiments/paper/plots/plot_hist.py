import wandb
import os
import pandas as pd 
api = wandb.Api()

if not os.path.exists("eval.csv"):
    # Project is specified by <entity/project-name>
    runs = api.runs("vwxyzjn/gym-microrts-paper-eval")
    summary_list = [] 
    config_list = [] 
    name_list = [] 
    for run in runs: 
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict) 
    
        # run.config is the input metrics.
        # We remove special values that start with _.
        config = {k:v for k,v in run.config.items() if not k.startswith('_')}
        config_list.append(config) 
    
        # run.name is the name of the run.
        name_list.append(run.name)       
    
    summary_df = pd.DataFrame.from_records(summary_list) 
    config_df = pd.DataFrame.from_records(config_list) 
    name_df = pd.DataFrame({'name': name_list}) 
    all_df = pd.concat([name_df, config_df,summary_df], axis=1)
    all_df.to_csv("project.csv")
else:
    all_df = pd.read_csv("project.csv")

if not os.path.exists("hists"):
    os.makedirs("hists")

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

# f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

import numpy as np
# from gym_microrts import microrts_ai
import matplotlib.pyplot as plt
all_ais = {
    "randomBiasedAI": "microrts_ai.randomBiasedAI",
    # "randomAI": "microrts_ai.randomAI",
    "passiveAI": "microrts_ai.passiveAI",
    "workerRushAI": "microrts_ai.workerRushAI",
    "lightRushAI": "microrts_ai.lightRushAI",
    "coacAI": "microrts_ai.coacAI",
    "naiveMCTSAI": "microrts_ai.naiveMCTSAI",
    "mixedBot": "microrts_ai.mixedBot",
    "rojo": "microrts_ai.rojo",
    "izanagi": "microrts_ai.izanagi",
    "tiamat": "microrts_ai.tiamat",
    "droplet": "microrts_ai.droplet",
    "guidedRojoA3N": "microrts_ai.guidedRojoA3",
}
ai_names, ais = list(all_ais.keys()) ,list(all_ais.values())
n_rows, n_cols = 3, 4

for idx in range(len(all_df)):
    ai_match_stats = dict(zip(ai_names, np.zeros((len(ais), 3))))
    for ai_name in ai_names:
        ai_match_stats[ai_name] = np.array([
            all_df.iloc[idx][f'charts/{ai_name}/loss'],
            all_df.iloc[idx][f'charts/{ai_name}/tie'],
            all_df.iloc[idx][f'charts/{ai_name}/win'],
        ])
    
    f, axes = plt.subplots(n_rows, n_cols, figsize=(9, 7), sharex=True, sharey=True)
    for i in range(len(ai_names)):
        var_name = ai_names[i]
        # if i>=1: i += 2
        ax=axes.flatten()[i]
        ax.bar(["loss", "tie", "win"], ai_match_stats[var_name])
        ax.set_title(var_name)
    # f.suptitle(name2label[all_df.iloc[idx]['exp_name']])
    f.tight_layout()
    f.savefig(f"hists/{all_df.iloc[idx]['exp_name']}-eval.pdf")
    # break
