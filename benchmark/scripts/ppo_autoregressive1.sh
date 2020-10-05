
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_autoregressive.py \
    --gym-id MicrortsMining-v2 \
    --total-timesteps 10000000 \
    --wandb-project-name gym-microrts \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_autoregressive.py \
    --gym-id MicrortsProduceWorker-v2 \
    --total-timesteps 10000000 \
    --wandb-project-name gym-microrts \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_autoregressive.py \
    --gym-id MicrortsAttackPassiveEnemySparseReward-v2 \
    --total-timesteps 10000000 \
    --wandb-project-name gym-microrts \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_autoregressive.py \
    --gym-id MicrortsProduceCombatUnitsSparseReward-v2 \
    --total-timesteps 10000000 \
    --wandb-project-name gym-microrts \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
