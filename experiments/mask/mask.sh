NUM_CORES=1
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_4x4.py \
    --exp-name ppo \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining4x4F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_adj_4x4.py \
    --exp-name ppo_no_adj \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining4x4F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done


for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_4x4.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining4x4F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done


for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_10x10.py \
    --exp-name ppo \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining10x10F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_adj_10x10.py \
    --exp-name ppo_no_adj \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining10x10F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done


for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_10x10.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining10x10F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_16x16.py \
    --exp-name ppo \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining16x16F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_adj_16x16.py \
    --exp-name ppo_no_adj \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining16x16F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done


for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_16x16.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining16x16F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_24x24.py \
    --exp-name ppo \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining24x24F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_adj_24x24.py \
    --exp-name ppo_no_adj \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining24x24F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done


for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_24x24.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining24x24F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

# investigate into invalid action penalty

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_4x4.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining4x4F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss --invalid-action-penalty -0.01  \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_10x10.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining10x10F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss --invalid-action-penalty -0.01  \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_16x16.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining16x16F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss --invalid-action-penalty -0.01  \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_24x24.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining24x24F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss --invalid-action-penalty -0.01  \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_4x4.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining4x4F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss --invalid-action-penalty -0.1  \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_10x10.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining10x10F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss --invalid-action-penalty -0.1  \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_16x16.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining16x16F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss --invalid-action-penalty -0.1  \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_24x24.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining24x24F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss --invalid-action-penalty -0.1  \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_4x4.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining4x4F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss --invalid-action-penalty -1  \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_10x10.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining10x10F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss --invalid-action-penalty -1  \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_16x16.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining16x16F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss --invalid-action-penalty -1  \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done

for seed in {1..4}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_no_mask_24x24.py \
    --exp-name ppo_no_mask \
    --total-timesteps 500000 \
    --wandb-project-name gym-microrts-mask3 \
    --gym-id MicrortsMining24x24F9-v0 \
    --no-cuda --gae --norm-obs --norm-adv --anneal-lr --clip-vloss --invalid-action-penalty -1  \
    --prod-mode \
    --capture-video \
    --seed $seed) >& /dev/null &
done
