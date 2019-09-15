import wandb
import pandas as pd
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
sns.set(font_scale=1.6)
sns.set_style("white")
#sns.set()

def strip_expnames(x):
    s = x
    for key in ["Microrts", "MaxResources", "Prod-v0", "4x4", "6x6", "8x8", "s"]:
        s = s.replace(key, "")
    if s == "GlobalAgent":
        return "Global Representation"
    elif s == "LocalAgentWindow1":
        return "Local Representation with w=1"
    elif s == "LocalAgentWindow2":
        return "Local Representation with w=2"
    return s

def save_plots(runs, filename, title):
    df = pd.DataFrame(columns=['experiment', 'charts/episode_reward', '_step'])
    for run in runs:
        temp_df = run.history(samples=200)[['_step', 'charts/episode_reward']]
        temp_df['experiment'] = run.config['gym_id']
        #temp_df['charts/episode_reward'] = temp_df['charts/episode_reward'].rolling(10).mean()
        df = df.append(temp_df, sort=False, ignore_index=True)
    df['charts/episode_reward']= df['charts/episode_reward'].astype(float)
    
    df['experiment'] = df['experiment'].apply(strip_expnames)
    df = df.rename(columns={'charts/episode_reward': 'episode reward', '_step': 'timestep'})
        
    g = sns.relplot(x="timestep", y="episode reward",
                hue="experiment", style="experiment", estimator=np.mean,
                hue_order=['Global Representation', 'Local Representation with w=1', 'Local Representation with w=2'],
                kind="line", ci=None, data=df, facet_kws={"legend_out":False}, aspect=2)
    handles, labels = g.ax.get_legend_handles_labels()
    g.ax.legend(handles=handles[1:], labels=labels[1:], loc='upper left')
    g.ax.xaxis.set_major_formatter(ticker.EngFormatter())
    g.ax.set_ylabel('')    
    g.ax.set_xlabel('')
    g.fig.suptitle(title, y=1.0, fontsize = 16)
    g.savefig(filename)

    

api = wandb.Api()
save_plots(api.runs(
    "costa-huang/MicrortsRL", 
    {"$or": [
        {"config.gym_id": "MicrortsGlobalAgentsMaxResources4x4Prod-v0"},
        {"config.gym_id": "MicrortsLocalAgentsMaxResources4x4Window1Prod-v0"},
        {"config.gym_id": "MicrortsLocalAgentsMaxResources4x4Window2Prod-v0"},
    ]}), "images/4x4.pdf", "4x4 Map")
    
save_plots(api.runs(
    "costa-huang/MicrortsRL", 
    {"$or": [
        {"config.gym_id": "MicrortsGlobalAgentsMaxResources6x6Prod-v0"},
        {"config.gym_id": "MicrortsLocalAgentsMaxResources6x6Window1Prod-v0"},
        {"config.gym_id": "MicrortsLocalAgentsMaxResources6x6Window2Prod-v0"},
    ]}), "images/6x6.pdf", "6x6 Map")
    
save_plots(api.runs(
    "costa-huang/MicrortsRL", 
    {"$or": [
        {"config.gym_id": "MicrortsGlobalAgentsMaxResources8x8Prod-v0"},
        {"config.gym_id": "MicrortsLocalAgentsMaxResources8x8Window1Prod-v0"},
        {"config.gym_id": "MicrortsLocalAgentsMaxResources8x8Window2Prod-v0"},
    ]}), "images/8x8.pdf", "8x8 Map")
