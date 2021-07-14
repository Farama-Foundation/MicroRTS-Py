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


for seed in {1..1}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_fast.py \
    --gym-id MicrortsDefeatCoacAIShaped-v3 \
    --total-timesteps 300000000 \
    --num-envs 64 \
    --num-steps 512 \
    --wandb-project-name coacai \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done


export WANDB_RESUME=must
export WANDB_RUN_ID=3h2cna4i
python ppo_fast_diverse.py --gym-id MicrortsDefeatCoacAIShaped-v3 --total-timesteps 300000000 --wandb-project-name coacai_fast_new --prod-mode --num-envs 16 --num-steps 512 --wandb-entity vwxyzjn --cuda True --capture-video --seed 3

export WANDB_RESUME=must
export WANDB_RUN_ID=eisq0sho
python ppo_fast.py --gym-id MicrortsDefeatCoacAIShaped-v3 --total-timesteps 300000000 --wandb-project-name coacai_fast --prod-mode --num-envs 64 --num-steps 512 --wandb-entity vwxyzjn --cuda True --capture-video --seed 3

export WANDB_RESUME=must
export WANDB_RUN_ID=eisq0sho
python ppo_gridnet.py \
    --exp-name selfplay \
    --total-timesteps 300000000 \
    --num-steps 256 \
    --wandb-project-name coacai_fast_new \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda True \
    --capture-video \

python ppo_gridnet_diverse.py \
    --total-timesteps 300000000 \
    --num-bot-envs 24 \
    --num-selfplay-envs 0\
    --num-steps 256 \
    --wandb-project-name coacai_fast_new \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda True \
    --capture-video \

python ppo_gridnet_diverse.py \
    --total-timesteps 300000000 \
    --num-bot-envs 24 \
    --num-selfplay-envs 0\
    --num-steps 256 \
    --wandb-project-name coacai_fast_new \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda True \
    --capture-video \