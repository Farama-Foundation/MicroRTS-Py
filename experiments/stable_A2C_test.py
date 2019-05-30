import gym
import gym_microrts
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines.common import set_global_seeds
set_global_seeds(0)
env = gym.make("Microrts-v0")
# env.init(16, 16)
env.init(4, 4)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = A2C(MlpPolicy, env, tensorboard_log="./", verbose=1)
model.learn(total_timesteps=100)

obs = env.reset()
rews = []
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    rews += [rewards[0]]
    
env.close()