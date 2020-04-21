for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ppo2_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-0-v0 \
    --total-timesteps 500000 \
    --gae --norm-returns --norm-adv --anneal-lr --kle-stop \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ppo2_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-1-v0 \
    --total-timesteps 500000 \
    --gae --norm-returns --norm-adv --anneal-lr --kle-stop \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ppo2_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-2-v0 \
    --total-timesteps 500000 \
    --gae --norm-returns --norm-adv --anneal-lr --kle-stop \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ppo2_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-3-v0 \
    --total-timesteps 500000 \
    --gae --norm-returns --norm-adv --anneal-lr --kle-stop \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ppo2_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-4-v0 \
    --total-timesteps 500000 \
    --gae --norm-returns --norm-adv --anneal-lr --kle-stop \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done


for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ppo_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-0-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ppo_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-1-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ppo_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-2-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ppo_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-3-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ppo_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-4-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/a2c_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-0-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/a2c_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-1-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/a2c_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-2-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/a2c_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-3-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/a2c_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-4-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done


for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ddpg_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-0-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ddpg_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-1-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ddpg_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-2-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ddpg_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-3-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/ddpg_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-4-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done



for seed in {1..2}
do
    (sleep 0.3 && nohup python param/sac_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-0-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/sac_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-1-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/sac_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-2-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/sac_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-3-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/sac_continuous_action.py \
    --gym-id ParamOpEnvEpisodeMap-4-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done



for seed in {1..2}
do
    (sleep 0.3 && nohup python param/td3_no_noise_annealing.py \
    --gym-id ParamOpEnvEpisodeMap-0-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/td3_no_noise_annealing.py \
    --gym-id ParamOpEnvEpisodeMap-1-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/td3_no_noise_annealing.py \
    --gym-id ParamOpEnvEpisodeMap-2-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/td3_no_noise_annealing.py \
    --gym-id ParamOpEnvEpisodeMap-3-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup python param/td3_no_noise_annealing.py \
    --gym-id ParamOpEnvEpisodeMap-4-v0 \
    --total-timesteps 500000 \
    --wandb-project-name param.op \
    --wandb-entity costa-huang \
    --prod-mode \
    --seed $seed
    ) >& /dev/null &
done

