python generate_exp.py --exp-script scripts/ppo_autoregressive1.sh \
    --algo ppo_autoregressive.py \
    --total-timesteps 10000000 \
    --gym-ids MicrortsMining-v2 MicrortsProduceWorker-v2 MicrortsAttackPassiveEnemySparseReward-v2 MicrortsProduceCombatUnitsSparseReward-v2 \
    --wandb-project-name gym-microrts \
    --other-args "--wandb-entity vwxyzjn --cuda False"

python generate_exp.py --exp-script scripts/ppo_autoregressive2.sh \
    --algo ppo_autoregressive.py \
    --total-timesteps 100000000 \
    --gym-ids MicrortsDefeatRandomEnemyShapedReward-v2 MicrortsDefeatWorkerRushEnemyShaped-v2 MicrortsDefeatLightRushEnemyShaped-v2 MicrortsDefeatWorkerRushEnemyHRL-v2 \
    --wandb-project-name gym-microrts \
    --other-args "--wandb-entity vwxyzjn --cuda False"

python generate_exp.py --exp-script scripts/ppo_autoregressive3.sh \
    --algo ppo_autoregressive.py \
    --total-timesteps 100000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v2 \
    --wandb-project-name gym-microrts \
    --other-args "--wandb-entity vwxyzjn --cuda False"
