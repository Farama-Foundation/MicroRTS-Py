# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
# evaluate the algorithm in parallel
NUM_CORES=2
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python cleanrl_a2c_cnn_mask.py \
    --total-timesteps 2000000 \
    --wandb-project-name gym-microrts \
    --gym-id MicrortsGlobalAgentBinary10x10-v0 \
    --prod-mode True \
    --capture-video True \
    --seed $seed) >& /dev/null &
done

for seed in {1..1}
do
    (sleep 0.3 && nohup xvfb-run -a python cleanrl_a2c_hrl.py \
    --total-timesteps 2000000 \
    --wandb-project-name gym-microrts-hrl \
    --gym-id MicrortsGlobalAgentHRL10x10-v0 \
    --prod-mode True \
    --capture-video True \
    --cuda False \
    --seed $seed) >& /dev/null &
done

for seed in {1..1}
do
    (sleep 0.3 && nohup xvfb-run -a python cleanrl_a2c_hrl_fixed.py \
    --total-timesteps 2000000 \
    --wandb-project-name gym-microrts-hrl \
    --gym-id MicrortsGlobalAgentHRLMining10x10-v0 \
    --prod-mode True \
    --capture-video True \
    --cuda False \
    --seed $seed) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python cleanrl_a2c.py \
    --total-timesteps 2000000 \
    --wandb-project-name gym-microrts \
    --gym-id MicrortsGlobalAgentsMaxResources4x4Prod-v0 \
    --prod-mode True \
    --capture-video True \
    --seed $seed) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python cleanrl_a2c_mask.py \
    --total-timesteps 2000000 \
    --wandb-project-name gym-microrts \
    --gym-id MicrortsGlobalAgentsMaxResources4x4Prod-v0 \
    --prod-mode True \
    --capture-video True \
    --seed $seed) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python cleanrl_a2c.py \
    --total-timesteps 2000000 \
    --wandb-project-name gym-microrts \
    --gym-id MicrortsGlobalAgentsMaxResources4x4NoFrameSkipProd-v0 \
    --prod-mode True \
    --capture-video True \
    --seed $seed) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python cleanrl_a2c_mask.py \
    --total-timesteps 2000000 \
    --wandb-project-name gym-microrts \
    --gym-id MicrortsGlobalAgentsMaxResources4x4NoFrameSkipProd-v0 \
    --prod-mode True \
    --capture-video True \
    --seed $seed) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python cleanrl_a2c.py \
    --total-timesteps 2000000 \
    --wandb-project-name gym-microrts \
    --gym-id MicrortsGlobalAgentsMaxResources6x6Prod-v0 \
    --prod-mode True \
    --capture-video True \
    --seed $seed) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python cleanrl_a2c.py \
    --total-timesteps 2000000 \
    --wandb-project-name gym-microrts \
    --gym-id MicrortsLocalAgentsMaxResources4x4Window1Prod-v0 \
    --prod-mode True \
    --capture-video True \
    --seed $seed) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python cleanrl_a2c.py \
    --total-timesteps 2000000 \
    --wandb-project-name gym-microrts \
    --gym-id MicrortsLocalAgentsMaxResources6x6Window1Prod-v0 \
    --prod-mode True \
    --capture-video True \
    --seed $seed) >& /dev/null &
done
