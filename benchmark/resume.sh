python resume.py \
    --wandb-project vwxyzjn/coacai \
    --run-state crashed \
    --job-queue cleanrl \
    --job-definition gym-microrts \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS
    --upload-files-baseurl $UPLOAD_FILES_BASEURL \

python -m cleanrl.utils.resume \
    --wandb-project vwxyzjn/gym-microrts-paper \
    --run-state crashed \
    --job-queue cleanrl_gpu \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 720.0 \
    --submit-aws $SUBMIT_AWS