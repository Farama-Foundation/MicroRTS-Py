import numpy as np
from numpy.random import choice
# if you want to record videos, install stable-baselines3 and use its `VecVideoRecorder`
from stable_baselines3.common.vec_env import VecVideoRecorder

from gym_microrts import microrts_ai
from gym_microrts.envs.new_vec_env import MicroRTSGridModeVecEnv

envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(1)],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
# envs = VecVideoRecorder(envs, 'videos', record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sample(logits):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    if sum(logits) == 0: return 0
    return choice(range(len(logits)), p=logits/sum(logits))

envs.action_space.seed(0)
envs.reset()
print(envs.action_plane_space.nvec)
nvec = envs.action_space.nvec

def sample(logits):
    return np.array(
        [choice(range(len(item)), p=softmax(item)) for item in logits]
    ).reshape(-1, 1)

for i in range(10000):
    envs.render()
    print(i)
    ### TODO: this `sample` function is very very slow.
    action_mask = envs.get_action_mask()
    action_mask = action_mask.reshape(-1, action_mask.shape[-1])
    action_type_mask = action_mask[:,0:6]
    action = np.concatenate((
        sample(action_mask[:,0:6]), # action type
        sample(action_mask[:,6:10]), # move parameter
        sample(action_mask[:,10:14]), # harvest parameter
        sample(action_mask[:,14:18]), # return parameter
        sample(action_mask[:,18:22]), # produce_direction parameter
        sample(action_mask[:,22:29]), # produce_unit_type parameter
        sample(action_mask[:,29:sum(envs.action_space.nvec[1:])]), # attack_target parameter
    ), axis=1)
    action = np.array([envs.action_space.sample()])
    # raise
    # action = action.reshape((1, -1))
    # action = action.reshape((envs.num_envs, envs.width*envs.height, -1))
    # action = np.concatenate((envs.source_unit_idxs, action), 2) # specify source unit
    # action = action[np.where(envs.source_unit_mask==1)] # valid actions
    # raise
    
    # java_actions = []
    # action_counts_per_env = envs.source_unit_mask.sum(1)
    # action_idx = 0
    # for action_count in action_counts_per_env:
    #     java_valid_action = []
    #     for _ in range(action_count):
    #         java_valid_action += [JArray(JInt)(valid_actions[action_idx])]
    #         action_idx += 1
    #     java_actions += [JArray(JArray(JInt))(java_valid_action)]
    # java_actions = JArray(JArray(JArray(JInt)))(java_actions)
    # raise
        
    
    # actions = []
    # envs.vec_client.gameStep(self.actions, [0 for _ in range(envs.num_envs)])
    next_obs, reward, done, info = envs.step(action)
envs.close()
