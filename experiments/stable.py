import gym
import gym_microrts
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines.common import set_global_seeds
set_global_seeds(0)
env = gym.make("Microrts-v0")
# env.set_map_dimension(16, 16)
env.set_map_dimension(4, 4)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)