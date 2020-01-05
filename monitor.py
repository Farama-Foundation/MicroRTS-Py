# xvfb-run -s "-screen 0 1400x900x24" python monitor.py
import gym
import time
import gym_microrts

from gym.wrappers import Monitor
env = gym.make('MicrortsGlobalAgentsProd-v0')


env = Monitor(env, './video')
env.reset()
env.action_space.seed(0)
start = time.time()
for i in range(1000):
	obs, r, done, info = env.step(env.action_space.sample())
	if done: break
print(time.time() - start)
env.close()
