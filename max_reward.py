import gym
import gym_microrts
env = gym.make("MicrortsGlobalAgentsMaxResources4x4NoFrameSkipProd-v0")
env.action_space.seed(0)
env.reset()
done = False

rewards = []
while not done:
    # mine left
    observation, reward, done, info = env.step([1 ,2, 0, 3, 0, 0, 0, 0, 0])
    env.render()
    rewards += [reward]
    if done:
        break
    # mine top
    observation, reward, done, info = env.step([4 ,2, 0, 0, 0, 0, 0, 0, 0])
    env.render()
    rewards += [reward]
    if done:
        break
    
    for _ in range(8):
        observation, reward, done, info = env.step([0 ,0, 0, 0, 0, 0, 0, 0, 0])
        env.render()
        rewards += [reward]
        if done:
            break
        
    # return bottom
    observation, reward, done, info = env.step([1 ,3, 0, 0, 2, 0, 0, 0, 0])
    env.render()
    rewards += [reward]
    if done:
        break
    # return right
    observation, reward, done, info = env.step([4 ,3, 0, 0, 1, 0, 0, 0, 0])
    env.render()
    rewards += [reward]
    if done:
        break
    
    for _ in range(8):
        observation, reward, done, info = env.step([0 ,0, 0, 0, 0, 0, 0, 0, 0])
        env.render()
        rewards += [reward]
        if done:
            break
        