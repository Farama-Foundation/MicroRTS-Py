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
    for seed in range(2):
        final_run_cmds += [run_match_str.replace("$seed", str(seed)).split()]

# use docker directly
cores = 40
repo = "vwxyzjn/gym-microrts_shared_memory:latest"
current_core = 0
final_str = ""
for final_run_cmd in final_run_cmds:
    final_str += (f'docker run -d --cpuset-cpus="{current_core}" -e WANDB={wandb_key} {repo} ' + " ".join(final_run_cmds[0]))
    final_str += "\n\n"
    current_core = (current_core + 1) % cores


with open(f"docker.sh", "w+") as f:
    f.write(final_str)