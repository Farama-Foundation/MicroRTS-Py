import wandb
import pandas as pd
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
#sns.set()

def strip_expnames(x):
    s = x
    for key in ["Microrts", "MaxResources", "Prod-v0", "4x4", "6x6", "8x8", "s"]:
        s = s.replace(key, "")
    return s

def save_plots(runs, filename, title):
    df = pd.DataFrame(columns=['experiment', 'charts/episode_reward', '_step'])
    for run in runs:
        temp_df = run.history(samples=200)[['_step', 'charts/episode_reward']]
        temp_df['experiment'] = run.config['gym_id']
        df = df.append(temp_df, sort=False, ignore_index=True)
    df['charts/episode_reward']= df['charts/episode_reward'].astype(float)
    df['experiment'] = df['experiment'].apply(strip_expnames)
    df = df.rename(columns={'charts/episode_reward': 'episode_reward', '_step': 'timestep'})
        
    g = sns.relplot(x="timestep", y="episode_reward",
                hue="experiment", style="experiment",
                hue_order=['GlobalAgent', 'LocalAgentWindow1', 'LocalAgentWindow2'],
                kind="line", data=df, facet_kws={"legend_out":False}, aspect=2)
    handles, labels = g.ax.get_legend_handles_labels()
    g.ax.legend(handles=handles[1:], labels=labels[1:], loc='upper left')
    g.fig.suptitle(title)
    g.savefig(filename)

    

api = wandb.Api()
save_plots(api.runs(
    "costa-huang/MicrortsRL", 
    {"$or": [
        {"config.gym_id": "MicrortsGlobalAgentsMaxResources4x4Prod-v0"},
        {"config.gym_id": "MicrortsLocalAgentsMaxResources4x4Window1Prod-v0"},
        {"config.gym_id": "MicrortsLocalAgentsMaxResources4x4Window2Prod-v0"},
    ]}), "images/4x4.pdf", "Training Result on the 4x4 Map")
    
save_plots(api.runs(
    "costa-huang/MicrortsRL", 
    {"$or": [
        {"config.gym_id": "MicrortsGlobalAgentsMaxResources6x6Prod-v0"},
        {"config.gym_id": "MicrortsLocalAgentsMaxResources6x6Window1Prod-v0"},
        {"config.gym_id": "MicrortsLocalAgentsMaxResources6x6Window2Prod-v0"},
    ]}), "images/6x6.pdf", "Training Result on the 6x6 Map")
    
save_plots(api.runs(
    "costa-huang/MicrortsRL", 
    {"$or": [
        {"config.gym_id": "MicrortsGlobalAgentsMaxResources8x8Prod-v0"},
        {"config.gym_id": "MicrortsLocalAgentsMaxResources8x8Window1Prod-v0"},
        {"config.gym_id": "MicrortsLocalAgentsMaxResources8x8Window2Prod-v0"},
    ]}), "images/8x8.pdf", "Training Result on the 8x8 Map")
