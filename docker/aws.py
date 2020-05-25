# pip install boto3
import boto3
import re
import time
import os
client = boto3.client('batch')

# get env variable values
wandb_key = os.environ['WANDB_KEY']
assert len(wandb_key) > 0, "set the environment variable `WANDB_KEY` to your WANDB API key, something like `export WANDB_KEY=fdsfdsfdsfads` "

# extract runs from bash scripts
final_run_cmds = []
with open("all.sh") as f:
    strings = f.read()
runs_match = re.findall('(python)(.+)((?:\n.+)+)(seed)',strings)
for run_match in runs_match:
    run_match_str = "".join(run_match).replace("\\\n", "")
    # print(run_match_str)
    for seed in range(1,5):
        final_run_cmds += [run_match_str.replace("$seed", str(seed)).split()]

# use docker directly
# cores = 40
# repo = "vwxyzjn/gym-microrts_shared_memory:latest"
# current_core = 0
# for final_run_cmd in final_run_cmds:
#     print(f'docker run -d --shm-size="500m" --cpuset-cpus="{current_core}" -e WANDB={wandb_key} {repo} ' + " ".join(final_run_cmd))
#     current_core = (current_core + 1) % cores

# submit jobs
# for final_run_cmd in final_run_cmds:
#     job_name = re.findall('(python)(.+)(.py)'," ".join(final_run_cmd))[0][1].strip() + str(int(time.time()))
#     job_name = job_name.replace("/", "_").replace("_param ", "")
#     response = client.submit_job(
#         jobName=job_name,
#         jobQueue='gym-microrts',
#         jobDefinition='gym-microrts',
#         containerOverrides={
#             'vcpus': 1,
#             'memory': 1000,
#             'command': final_run_cmd,
#             'environment': [
#                 {
#                     'name': 'WANDB',
#                     'value': wandb_key
#                 }
#             ]
#         },
#         retryStrategy={
#             'attempts': 1
#         },
#         timeout={
#             'attemptDurationSeconds': 16*60*60 # 16 hours
#         }
#     )
#     if response['ResponseMetadata']['HTTPStatusCode'] != 200:
#         print(response)
#         raise Exception("jobs submit failure")docker run -d --shm-size="500m" --cpuset-cpus="13" -e WANDB=6603a1e99a016ac5002729a06b08e13931d4ee02 vwxyzjn/gym-microrts_shared_memory:latest python mask/ppo_no_mask_10x10.py --exp-name ppo_no_mask --total-timesteps 500000 --wandb-project-name gym-microrts-mask2 --gym-id MicrortsMining10x10F9-v0 --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss --invalid-action-penalty -0.01 --prod-mode --capture-video --seed 2

