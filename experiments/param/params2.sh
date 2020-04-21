for seed in {1..2}
do
    (sleep 0.3 && nohup python ppo2_continuous_action.py \
    --gym-id ParamOpEnvSingleStep-v0 \
    --total-timesteps 2000000 \
    --gae --norm-returns --norm-adv --anneal-lr --kle-stop \
    --wandb-project-name param.op.single \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python ppo_continuous_action.py \
    --gym-id ParamOpEnvSingleStep-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name param.op.single \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python a2c_continuous_action.py \
    --gym-id ParamOpEnvSingleStep-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name param.op.single \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python ddpg2_continuous_action.py \
    --gym-id ParamOpEnvSingleStep-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name param.op.single \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python ddpg_continuous_action.py \
    --gym-id ParamOpEnvSingleStep-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name param.op.single \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python sac_continuous_action.py \
    --gym-id ParamOpEnvSingleStep-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name param.op.single \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done