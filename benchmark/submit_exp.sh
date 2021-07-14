python submit_exp.py --exp-script scripts/ppo_larger_model.sh \
    --algo ppo_larger_model.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 MicrortsDefeatCoacAIShaped-v4 \
    --wandb-project-name coacai \
    --other-args "--wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl \
    --job-definition gym-microrts \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS


python submit_exp.py --exp-script scripts/larger_througput.sh \
    --algo ppo_larger_model.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name coacai \
    --other-args "--num-envs 16 --num-steps 512 --wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl \
    --job-definition gym-microrts \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python submit_exp.py --exp-script scripts/larger_througput1.sh \
    --algo ppo_larger_model.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name coacai \
    --other-args "--num-envs 8 --num-steps 256 --wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl \
    --job-definition gym-microrts \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS



python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_fast.sh \
    --algo ppo_fast.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name coacai_fast_new \
    --other-args "--num-envs 16 --num-steps 512 --wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_fast.sh \
    --algo ppo_fast.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name coacai_fast_new \
    --other-args "--num-envs 32 --num-steps 512 --wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_fast.sh \
    --algo ppo_fast.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name coacai_fast_new \
    --other-args "--num-envs 64 --num-steps 512 --wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

# new experiments

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_resnet.sh \
    --algo ppo_resnet.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name coacai_fast_new \
    --other-args "--num-envs 16 --num-steps 512 --wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_fast_diverse.sh \
    --algo ppo_fast_diverse.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name coacai_fast_new \
    --other-args "--num-envs 16 --num-steps 512 --wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_fast_diverse.sh \
    --algo ppo_fast_diverse.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name coacai_fast_new \
    --other-args "--exp-name a2c --n-minibatch 1 --update-epochs 1 --num-envs 16 --num-steps 512 --wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS


python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_fast_diverse.sh \
    --algo ppo_fast_diverse.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name coacai_fast_new \
    --other-args "--exp-name ppo_fast_diverse_v2 --num-steps 512 --wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_fast_diverse.sh \
    --algo ppo_fast_diverse.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name coacai_fast_new \
    --other-args "--exp-name ppo_fast_diverse_v2 --num-steps 512 --wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_gridnet_fast.sh \
    --algo ppo_gridnet_fast.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name coacai_fast_new \
    --other-args "--num-steps 512 --wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_diverse.sh \
    --algo ppo_diverse.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name gym-microrts-paper \
    --other-args "--wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_gridnet_diverse.sh \
    --algo ppo_gridnet_diverse.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name gym-microrts-paper \
    --other-args "--wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_coacai_no_mask.sh \
    --algo paper/ppo_coacai_no_mask.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name gym-microrts-paper \
    --other-args "--wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_coacai.sh \
    --algo paper/ppo_coacai.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name gym-microrts-paper \
    --other-args "--wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_diverse_impala.sh \
    --algo paper/ppo_diverse_impala.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name gym-microrts-paper \
    --other-args "--wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS


python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_gridnet_diverse_impala.sh \
    --algo paper/ppo_gridnet_diverse_impala.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name gym-microrts-paper \
    --other-args "--wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_gridnet_coacai.sh \
    --algo paper/ppo_gridnet_coacai.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name gym-microrts-paper \
    --other-args "--wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_gridnet_coacai_no_mask.sh \
    --algo paper/ppo_gridnet_coacai_no_mask.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name gym-microrts-paper \
    --other-args "--wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS


python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_gridnet_selfplay_encode_decode.sh \
    --algo paper/ppo_gridnet_diverse_encode_decode.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name gym-microrts-paper \
    --other-args "--num-bot-envs 0 --num-selfplay-envs 24 --wandb-entity vwxyzjn --cuda True --exp-name ppo_gridnet_selfplay_encode_decode" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS


python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_gridnet_selfplay_diverse_encode_decode.sh \
    --algo paper/ppo_gridnet_diverse_encode_decode.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name gym-microrts-paper \
    --other-args "--num-bot-envs 8 --num-selfplay-envs 16 --wandb-entity vwxyzjn --cuda True --exp-name ppo_gridnet_selfplay_diverse_encode_decode" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS


python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_coacai_naive.sh \
    --algo paper/ppo_coacai_naive.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name gym-microrts-paper \
    --other-args "--wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_coacai_partial_mask.sh \
    --algo paper/ppo_coacai_partial_mask.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name gym-microrts-paper \
    --other-args "--wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_gridnet_coacai_naive.sh \
    --algo paper/ppo_gridnet_coacai_naive.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name gym-microrts-paper \
    --other-args "--wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script scripts/ppo_gridnet_coacai_partial_mask.sh \
    --algo paper/ppo_gridnet_coacai_partial_mask.py \
    --total-timesteps 300000000 \
    --gym-ids MicrortsDefeatCoacAIShaped-v3 \
    --wandb-project-name gym-microrts-paper \
    --other-args "--wandb-entity vwxyzjn --cuda True" \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS
