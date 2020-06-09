import wandb
import numpy as np
import pandas as pd 
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("costa-huang/gym-microrts-mask3")
analysis = True
summary_list = [] 
config_list = [] 
name_list = [] 
for run in runs: 
    # run.summary are the output key/values like accuracy.
    # We call ._json_dict to omit large files 
    summary_json_dict = run.summary._json_dict
    if analysis:
        summary_json_dict = summary_json_dict.copy()
        history = pd.DataFrame(run.scan_history())
        history['rollling_e'] = history['charts/episode_reward'].dropna().rolling(10).mean()
        first_best_reward_idx = (history["rollling_e"] >= 40.0).idxmax()
        if history.iloc[first_best_reward_idx]["rollling_e"] >= 40.0:
            summary_json_dict["first_learned_timestep"] = history.iloc[first_best_reward_idx]["global_step"] / 500000
        else:
            summary_json_dict["first_learned_timestep"] = 1
        summary_json_dict["first_reward_timestep"] = history.iloc[(history['charts/episode_reward'] > 0).idxmax()]["global_step"] / 500000

        # mask removed logic
        if run.config["exp_name"] == "ppo":
            history['evals_rollling_e'] = history['evals/charts/episode_reward'].dropna().rolling(10).mean()
            first_best_reward_idx = (history["evals_rollling_e"] >= 40.0).idxmax()
            if history.iloc[first_best_reward_idx]["evals_rollling_e"] >= 40.0:
                summary_json_dict["evals_first_learned_timestep"] = history.iloc[first_best_reward_idx]["global_step"] / 500000
            else:
                summary_json_dict["evals_first_learned_timestep"] = 1
            summary_json_dict["evals_first_reward_timestep"] = history.iloc[(history['charts/episode_reward'] > 0).idxmax()]["global_step"] / 500000
            
        
        summary_json_dict["charts/episode_reward"] = history["charts/episode_reward"][-10:].mean()
        summary_json_dict["losses/approx_kl"] = history['losses/approx_kl'].astype(np.float64).dropna()[-10:].mean()
        summary_json_dict['stats/num_invalid_action_null'] = history['stats/num_invalid_action_null'].dropna()[-10:].mean()
        summary_json_dict['stats/num_invalid_action_busy_unit'] = history['stats/num_invalid_action_busy_unit'].dropna()[-10:].mean()
        summary_json_dict['stats/num_invalid_action_ownership'] = history['stats/num_invalid_action_ownership'].dropna()[-10:].mean()
        
        if run.config["exp_name"] == "ppo":
            summary_json_dict['evals/charts/episode_reward'] = history['evals/charts/episode_reward'][-10:].mean()
            summary_json_dict['evals/stats/num_invalid_action_null'] = history['evals/stats/num_invalid_action_null'].dropna()[-10:].mean()
            summary_json_dict['evals/stats/num_invalid_action_busy_unit'] = history['evals/stats/num_invalid_action_busy_unit'].dropna()[-10:].mean()
            summary_json_dict['evals/stats/num_invalid_action_ownership'] = history['evals/stats/num_invalid_action_ownership'].dropna()[-10:].mean()
    summary_list.append(summary_json_dict)
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
all_df["losses/approx_kl"] = all_df["losses/approx_kl"].astype(np.float64)

# mask removal
mask_removed = all_df[all_df["exp_name"]=="ppo"].copy()
mask_removed['charts/episode_reward'] = mask_removed['evals/charts/episode_reward']
mask_removed['stats/num_invalid_action_null'] = mask_removed['evals/stats/num_invalid_action_null']
mask_removed['stats/num_invalid_action_busy_unit'] = mask_removed['evals/stats/num_invalid_action_busy_unit']
mask_removed['stats/num_invalid_action_ownership'] = mask_removed['evals/stats/num_invalid_action_ownership']
mask_removed['first_learned_timestep'] = mask_removed['evals_first_learned_timestep']
mask_removed['first_reward_timestep'] = mask_removed['evals_first_reward_timestep']
mask_removed["exp_name"] = "masking removed"
final_all_df = all_df.append(mask_removed, ignore_index=True)


# change names
final_all_df.loc[final_all_df["gym_id"]=="MicrortsMining4x4F9-v0", "gym_id"] = '04x04'
final_all_df.loc[(final_all_df["gym_id"]=="MicrortsMining10x10F9-v0"), "gym_id"] = '10x10'
final_all_df.loc[final_all_df["gym_id"]=="MicrortsMining16x16F9-v0", "gym_id"] = '16x16'
final_all_df.loc[final_all_df["gym_id"]=="MicrortsMining24x24F9-v0", "gym_id"] = '24x24'

final_all_df.loc[final_all_df["exp_name"]=="masking removed", "exp_name"] = 'Masking removed'
final_all_df.loc[(final_all_df["exp_name"]=="ppo"), "exp_name"] = 'Invalid action masking'
final_all_df.loc[final_all_df["exp_name"]=="ppo_no_adj", "exp_name"] = 'Naive invalid action masking'
final_all_df.loc[final_all_df["exp_name"]=="ppo_no_mask", "exp_name"] = 'Invalid action penalty'

results_df = final_all_df.fillna(0).groupby(
    ['exp_name','gym_id',"invalid_action_penalty"]
).mean()[[
    'charts/episode_reward',
    'losses/approx_kl',
    'stats/num_invalid_action_null',
    'stats/num_invalid_action_busy_unit',
    'stats/num_invalid_action_ownership',
    "first_learned_timestep",
    "first_reward_timestep"
]]
final_print_df = results_df.round(2)
final_print_df['losses/approx_kl'] = results_df['losses/approx_kl'].round(5)
# final_print_df['first_learned_timestep'] = results_df['first_learned_timestep'].round(4)
# final_print_df['first_reward_timestep'] = results_df['first_reward_timestep'].round(4)
final_print_df['first_learned_timestep'] = pd.Series(["{0:.2f}%".format(val * 100) for val in results_df['first_learned_timestep'].round(4)], index = results_df.index)
final_print_df['first_reward_timestep'] = pd.Series(["{0:.2f}%".format(val * 100) for val in results_df['first_reward_timestep'].round(4)], index = results_df.index)
print(final_print_df.to_latex())

print(final_print_df.drop(columns=['losses/approx_kl']).to_latex())


# calculate the first time the algorithm solves the environment

# , 'losses/value_loss',
#        'losses/policy_loss', 'charts/episode_reward',
#        , ,
#        ,
#        'charts/episode_reward/ResourceGatherRewardFunction',
#        'evals/charts/episode_reward', 'evals/stats/num_invalid_action_null',
#        'evals/stats/num_invalid_action_busy_unit',
#        'evals/stats/num_invalid_action_ownership'