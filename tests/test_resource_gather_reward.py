import numpy as np


from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv


def test_resource_gather_reward():
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.passiveAI for _ in range(1)],
        map_paths=["maps/4x4/baseTwoWorkers4x4.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    envs.action_space.seed(0)
    try:
        obs = envs.reset()
        envs.render()
    except Exception as e:
        e.printStackTrace()
    num_planes = len(envs.action_plane_space.nvec)


    # mine
    np.array(envs.get_action_mask())
    action = np.zeros(len(envs.action_space.nvec), np.int32)
    action[1*num_planes:(1+1)*num_planes] = [2, 0, 3, 0, 0, 0, 0]
    assert envs.step(action)[1].flatten() > 0
    envs.render()

    # wait for action to finish
    for _ in range(20):
        np.array(envs.get_action_mask())
        action = np.zeros(len(envs.action_space.nvec), np.int32)
        envs.step(action)
        envs.render()

    # return
    np.array(envs.get_action_mask())
    action = np.zeros(len(envs.action_space.nvec), np.int32)
    action[1*num_planes:(1+1)*num_planes] = [3, 0, 0, 2, 0, 0, 0]
    assert envs.step(action)[1].flatten() > 0
    envs.render()
    envs.close()
