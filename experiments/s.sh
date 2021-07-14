xvfb-run -a python ppo_gridnet.py \
    --exp-name selfplay \
    --total-timesteps 300000000 \
    --num-bot-envs 0 \
    --num-selfplay-envs 24 \
    --partial-obs \
    --prod-mode --capture-video

xvfb-run -a python ppo_gridnet.py \
    --total-timesteps 300000000 \
    --num-bot-envs 24 \
    --num-selfplay-envs 0 \
    --partial-obs False \
    --prod-mode --capture-video

# Partial obs, play against POLightRush
python ppo_gridnet_eval.py \
    --agent-model-path agent_po.pt \
    --num-selfplay-envs 0 \
    --ai randomBiasedAI \
    --partial-obs

# Partial obs, play against self
python ppo_gridnet_eval.py \
    --agent-model-path agent_po.pt \
    --num-selfplay-envs 2 \
    --partial-obs

python ppo_gridnet_eval.py \
    --agent-model-path agent.pt \
    --num-selfplay-envs 0 \
    --ai randomBiasedAI