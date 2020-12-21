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
export WANDB_RUN_ID=231phbk0
python ppo_full_autoregressive.py \
    --wandb-project-name gym-microrts \
    --total-timesteps 100000000 \
    --gym-id MicrortsMining-v2 \
    --prod-mode True \
    --capture-video True

for seed in {1..1}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_gridnet.py \
    --gym-id MicrortsDefeatCoacAIShaped-v3 \
    --total-timesteps 100000000 \
    --wandb-project-name rts-generalization \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done


for seed in {1..1}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_larger_model.py \
    --gym-id MicrortsDefeatCoacAIShaped-v3 \
    --total-timesteps 300000000 \
    --wandb-project-name rts-generalization \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..1}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_larger_model.py \
    --gym-id MicrortsDefeatCoacAIShaped-v4 \
    --total-timesteps 300000000 \
    --wandb-project-name rts-generalization \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done


for seed in {1..1}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_gridnet.py \
    --gym-id MicrortsDefeatWorkerRushEnemyShaped-v3 \
    --total-timesteps 100000000 \
    --wandb-project-name rts-generalization \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

