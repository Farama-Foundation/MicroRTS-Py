# full obs
## training against diverse bots
xvfb-run -a python ppo_gridnet.py \
    --total-timesteps 300000000 \
    --num-bot-envs 24 \
    --num-selfplay-envs 0 \
    --partial-obs False \
    --prod-mode --capture-video

## training using selfplay
xvfb-run -a python ppo_gridnet.py \
    --total-timesteps 300000000 \
    --num-bot-envs 0 \
    --num-selfplay-envs 24 \
    --partial-obs False \
    --prod-mode --capture-video

## evaluating against a particular AI (in this cas)
python ppo_gridnet_eval.py \
    --agent-model-path agent_sota.pt \
    --num-selfplay-envs 0 \
    --ai randomBiasedAI 

## evaluating against selfplay
python ppo_gridnet_eval.py \
    --agent-model-path agent_sota.pt \
    --num-selfplay-envs 2
