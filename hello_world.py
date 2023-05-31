import numpy as np

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

# if you want to record videos, install stable-baselines3 and use its `VecVideoRecorder`
# from stable_baselines3.common.vec_env import VecVideoRecorder


envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=2,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(1)],
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
# envs = VecVideoRecorder(envs, 'videos', record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def sample(logits):
    # https://stackoverflow.com/a/40475357/6611317
    p = softmax(logits, axis=1)
    c = p.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    return choices.reshape(-1, 1)


envs.action_space.seed(0)
envs.reset()
nvec = envs.action_space.nvec

for i in range(10000):
    # envs.render()
    action_mask = envs.get_action_mask()
    action_mask = action_mask.reshape(-1, action_mask.shape[-1])
    action_mask[action_mask == 0] = -9e8
    # sample valid actions
    action = np.concatenate(
        (
            sample(action_mask[:, 0:6]),  # action type
            sample(action_mask[:, 6:10]),  # move parameter
            sample(action_mask[:, 10:14]),  # harvest parameter
            sample(action_mask[:, 14:18]),  # return parameter
            sample(action_mask[:, 18:22]),  # produce_direction parameter
            sample(action_mask[:, 22:29]),  # produce_unit_type parameter
            # attack_target parameter
            sample(action_mask[:, 29 : sum(envs.action_space.nvec[1:])]),
        ),
        axis=1,
    )
    # doing the following could result in invalid actions
    # action = np.array([envs.action_space.sample()])
    next_obs, reward, done, info = envs.step(action)
envs.close()
