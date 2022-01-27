import numpy as np

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

render = False


def test_mask():
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.passiveAI for _ in range(1)],
        map_paths=["maps/4x4/baseTwoWorkers4x4.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )
    envs.action_space.seed(0)
    try:
        envs.reset()
        # if render: envs.render()
    except Exception as e:
        e.printStackTrace()
    len(envs.action_plane_space.nvec)

    # fmt: off
    np.testing.assert_array_equal(
        np.array(envs.get_action_mask())[0,1],
        np.array([
            1, 1, 1, 0, 1, 0,
            0, 1, 0, 0,
            0, 0, 0, 1,
            0, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 0,
            # relative attack position below
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ]).astype(np.int32),
    )
    np.testing.assert_array_equal(
        np.array(envs.get_action_mask())[0,4],
        np.array([
            1, 1, 1, 0, 1, 0,
            0, 0, 1, 0,
            1, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 0,
            # relative attack position below
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ]).astype(np.int32),
    )
    np.testing.assert_array_equal(
        np.array(envs.get_action_mask())[0,5],
        np.array([
            1, 0, 0, 0, 1, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 1, 1, 0,
            0, 0, 0, 1, 0, 0, 0,
            # relative attack position below
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ]).astype(np.int32),
    )
    # fmt: on
