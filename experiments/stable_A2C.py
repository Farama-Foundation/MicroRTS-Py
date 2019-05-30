import gym
import gym_microrts
import wandb
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines.common import set_global_seeds

wandb.init(project="MicrortsRL", tensorboard=True)
config = wandb.config
config.seed = 0
config.dimension_x = 4
config.dimension_y = 4
config.total_timesteps = 20000000

set_global_seeds(config.seed)
env = gym.make("Microrts-v0")
env.init(config.dimension_x , config.dimension_y, port=9899)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=config.total_timesteps)
model.save("a2c.model")
wandb.save("a2c.model")

env.close()
