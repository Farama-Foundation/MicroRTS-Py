SUBMIT_AWS=False

python jobs.py --exp-script scripts/ppo_autoregressive1.sh \
    --job-queue cleanrl \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 2 \
    --num-memory 13000 \
    --num-hours 100.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/ppo_autoregressive2.sh \
    --job-queue cleanrl \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 2 \
    --num-memory 13000 \
    --num-hours 100.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/ppo_autoregressive3.sh \
    --job-queue cleanrl \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 2 \
    --num-memory 13000 \
    --num-hours 100.0 \
    --submit-aws $SUBMIT_AWS
