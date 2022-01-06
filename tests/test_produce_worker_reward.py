import numpy as np


from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

render = False

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
        if render: envs.render()
    except Exception as e:
        e.printStackTrace()
    num_planes = len(envs.action_plane_space.nvec)


    # produce a worker
    np.array(envs.get_action_mask())
    action = np.zeros(len(envs.action_space.nvec), np.int32)
    action[5*num_planes:(5+1)*num_planes] = [4, 0, 0, 0, 1, 3, 0]
    assert envs.step(action)[1].flatten() > 0
    if render: envs.render()


# assert env.step([5, 4, 0, 0, 0, 1, 3, 0], True)[1] > 0
# env.render()

# env.close()