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

# partial obs
## training against diverse bots
xvfb-run -a python ppo_gridnet.py \
    --total-timesteps 300000000 \
    --num-bot-envs 24 \
    --num-selfplay-envs 0 \
    --partial-obs True \
    --prod-mode --capture-video

## training using selfplay
xvfb-run -a python ppo_gridnet.py \
    --total-timesteps 300000000 \
    --num-bot-envs 0 \
    --num-selfplay-envs 24 \
    --partial-obs True \
    --prod-mode --capture-video

## evaluating against a particular AI (in this cas)
python ppo_gridnet_eval.py \
    --agent-model-path agent_po.pt \
    --num-selfplay-envs 0 \
    --partial-obs True \
    --ai randomBiasedAI 

## evaluating against selfplay
python ppo_gridnet_eval.py \
    --agent-model-path agent_po.pt \
    --partial-obs True \
    --num-selfplay-envs 2

WANDB_ENTITY=vwxyzjn WANDB_PROJECT=gym-microrts-league python league.py --prod-mode\
    --built-in-ais randomBiasedAI workerRushAI lightRushAI coacAI randomAI passiveAI naiveMCTSAI mixedBot rojo izanagi tiamat droplet guidedRojoA3N \
    --rl-ais agent_sota.pt
