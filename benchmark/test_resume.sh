python ppo_autoregressive.py \
    --wandb-project-name gym-microrts \
    --total-timesteps 100000000 \
    --gym-id MicrortsDefeatWorkerRushEnemyShaped-v2 \
    --prod-mode True \
    --capture-video True


export WANDB_RESUME=must
export WANDB_RUN_ID=2kse3aqy
python ppo_autoregressive.py \
    --wandb-project-name gym-microrts \
    --total-timesteps 100000000 \
    --gym-id MicrortsDefeatWorkerRushEnemyShaped-v2 \
    --prod-mode True \
    --capture-video True

python ppo_full_autoregressive.py \
    --wandb-project-name gym-microrts \
    --total-timesteps 100000000 \
    --gym-id MicrortsMining-v2 \
    --prod-mode True \
    --capture-video True

python ppo_full_autoregressive.py \
    --wandb-project-name gym-microrts \
    --total-timesteps 100000000 \
    --gym-id MicrortsMining-v2 \
    --prod-mode True \
    --capture-video True

python ppo_autoregressive.py \
    --wandb-project-name gym-microrts \
    --total-timesteps 100000000 \
    --gym-id MicrortsMining-v2 \
    --prod-mode True \
    --capture-video True

export WANDB_RESUME=must
export WANDB_RUN_ID=2kse3aqy
python ppo_full_autoregressive.py \
    --wandb-project-name gym-microrts \
    --total-timesteps 100000000 \
    --gym-id MicrortsMining-v2 \
    --prod-mode True \
    --capture-video True
