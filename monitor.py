# xvfb-run -s "-screen 0 1400x900x24" python monitor.py
import gym
import time
import gym_microrts

from gym.wrappers import Monitor
env = gym.make('MicrortsGlobalAgentsProd-v0')

try:
    env = Monitor(env, './video', force=True)
    env.reset()
    env.action_space.seed(0)
    start = time.time()
    for i in range(100):
    	obs, r, done, info = env.step(env.action_space.sample())
    	if done: break
    print(time.time() - start)
    env.close()
except Exception as e:
    print(e)
    print(e.stacktrace())