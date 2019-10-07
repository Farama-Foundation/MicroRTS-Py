# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
# evaluate the algorithm in parallel
NUM_CORES=1
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES
for seed in {1..3}
do
    (sleep 0.3 && nohup python cleanrl_a2c.py \
    --prod-mode True \
    --seed $seed \
    --total-timesteps 2000000 \
    --gym-id MicrortsGlobalAgentsMaxResources8x8Prod-v0) >& /dev/null &
done