# Reference: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/

from tensorboardX import SummaryWriter
import gym_microrts
import argparse
import numpy as np
import gym
import time
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A2C agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default="random",
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="Microrts-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=5,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=2000,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=5000,
                       help='total timesteps of the experiments')
    parser.add_argument('--prod-mode', type=bool, default=False,
                       help='run the script in production mode and use wandb to log outputs')
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
env = gym.make(args.gym_id)
config = gym_microrts.types.Config(
    ai1_type="penalty",
    ai2_type="passive",
    map_path="maps/4x4/base4x4.xml",
    render=True,
    client_port=9898,
    microrts_path="E:/Go/src/github.com/vwxyzjn/201906051646.microrts"
)
if args.prod_mode:
    import wandb
    config.render = False
    config.microrts_path = "/root/microrts"
env.init(config)
random.seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# TRY NOT TO MODIFY: start the game
experiment_name = f"{time.strftime('%b%d_%H-%M-%S')}__{args.exp_name}__{args.seed}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    wandb.init(project="MicrortsRL", tensorboard=True, config=vars(args))
    writer = SummaryWriter(f"/tmp/{experiment_name}")
global_step = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())
    rewards, dones = np.zeros((2, args.episode_length))
    obs = np.empty((args.episode_length,) + env.observation_space.shape)
    
    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs[step] = next_obs.copy()
        
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], _ = env.step(
            [np.random.randint(i) for i in env.action_space.nvec])
        next_obs = np.array(next_obs)
        if dones[step]:
            break
    

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
    writer.add_scalar("charts/global_step", global_step, global_step)
env.close()
writer.close()
