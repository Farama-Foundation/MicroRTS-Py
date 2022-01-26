import numpy as np

from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

render = False


def test_observation():
    envs = MicroRTSGridModeVecEnv(
        num_bot_envs=0,
        num_selfplay_envs=2,
        partial_obs=False,
        max_steps=5000,
        render_theme=2,
        ai2s=[],
        map_paths=["maps/16x16/basesWorkers16x16A.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )

    # fmt: off
    next_obs = envs.reset()
    resource = np.array([
        0., 1., 0., 0., 0., # 1 hp
        0., 0., 0., 0., 1., # >= 4 resources
        1., 0., 0.,         # no owner
        0., 1., 0., 0., 0., 0., 0., 0.,  # unit type resource
        1., 0., 0., 0., 0., 0.  # currently not executing actions
    ]).astype(np.int32)
    p1_worker = np.array([
        0., 1., 0., 0., 0., # 1 hp
        1., 0., 0., 0., 0., # 0 resources
        0., 1., 0.,         # player 1 owns it 
        0., 0., 0., 0., 1., 0., 0., 0., # unit type worker
        1., 0., 0., 0., 0., 0. # currently not executing actions
    ]).astype(np.int32)
    p1_base = np.array([
        0., 0., 0., 0., 1.,  # 1 hp
        1., 0., 0., 0., 0.,  # 0 resources
        0., 1., 0.,          # player 1 owns it
        0., 0., 1., 0., 0., 0., 0., 0., # unit type base
        1., 0., 0., 0., 0., 0. # currently not executing actions
    ]).astype(np.int32)
    p2_worker = p1_worker.copy()
    p2_worker[10:13] = np.array([0., 0., 1.,]) # player 2 owns it
    p2_base = p1_base.copy()
    p2_base[10:13] = np.array([0., 0., 1.,]) # player 2 owns it
    empty_cell = np.array([
        1., 0., 0., 0., 0.,  # 0 hp
        1., 0., 0., 0., 0.,  # 0 resources
        1., 0., 0.,          # no owner
        1., 0., 0., 0., 0., 0., 0., 0., # unit type empty cell
        1., 0., 0., 0., 0., 0. # currently not executing actions
    ]).astype(np.int32)
    # fmt: on

    # player 1's perspective
    np.testing.assert_array_equal(next_obs[0][0][0], resource)
    np.testing.assert_array_equal(next_obs[0][1][0], resource)
    np.testing.assert_array_equal(next_obs[0][1][1], p1_worker)
    np.testing.assert_array_equal(next_obs[0][2][2], p1_base)
    np.testing.assert_array_equal(next_obs[0][15][15], resource)
    np.testing.assert_array_equal(next_obs[0][14][15], resource)
    np.testing.assert_array_equal(next_obs[0][14][14], p2_worker)
    np.testing.assert_array_equal(next_obs[0][13][13], p2_base)

    # TODO: fix this BUG
    # player 2's perspective (self play)
    # np.testing.assert_array_equal(next_obs[1][0][0], resource) # BUG: in `MicroRTSGridModeVecEnv` the onwer is correctly set to [0, 1, 0]
    # np.testing.assert_array_equal(next_obs[1][1][0], resource) # BUG: in `MicroRTSGridModeVecEnv` the onwer is correctly set to [0, 1, 0]
    np.testing.assert_array_equal(next_obs[1][1][1], p2_worker)
    np.testing.assert_array_equal(next_obs[1][2][2], p2_base)
    # np.testing.assert_array_equal(next_obs[1][15][15], resource) # BUG: in `MicroRTSGridModeVecEnv` the onwer is correctly set to [0, 1, 0]
    # np.testing.assert_array_equal(next_obs[1][14][15], resource) # BUG: in `MicroRTSGridModeVecEnv` the onwer is correctly set to [0, 1, 0]
    np.testing.assert_array_equal(next_obs[1][14][14], p1_worker)
    np.testing.assert_array_equal(next_obs[1][13][13], p1_base)

    feature_sum = 0
    for item in [resource, resource, p1_worker, p1_base, resource, resource, p2_worker, p2_base]:
        feature_sum += item.sum()
    feature_sum += empty_cell.sum() * (256 - 8)
    assert next_obs.sum() == feature_sum * 2 == 2560.0
