import numpy as np

from gym_microrts.petting_zoo_api import PettingZooMicroRTSGridModeSharedMemVecEnv


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


def policy(observation):
    # Get action mask
    action_mask = observation["action_masks"]
    action_mask = action_mask.reshape(-1, action_mask.shape[-1])
    action_mask[action_mask == 0] = -9e8

    # Sample action from action mask
    action = np.concatenate(
        (
            sample(action_mask[:, 0:6]),  # action type
            sample(action_mask[:, 6:10]),  # move parameter
            sample(action_mask[:, 10:14]),  # harvest parameter
            sample(action_mask[:, 14:18]),  # return parameter
            # produce_direction parameter
            sample(action_mask[:, 18:22]),
            # produce_unit_type parameter
            sample(action_mask[:, 22:29]),
            # attack_target parameter
            sample(action_mask[:, 29:]),
        ),
        axis=1,
    )

    return action


def main():
    opponents = []
    render = False

    env = PettingZooMicroRTSGridModeSharedMemVecEnv(2, 0, ai2s=opponents)

    env.reset()
    if render:
        env.render()

    for episode in range(100):
        # Iterate over all of the agents
        for agent in env.agent_iter():
            if render:
                env.render()

            observation, reward, done, info = env.last()
            action = policy(observation)

            if done:
                env.reset()
                break

            env.step(action)

    env.close()


if __name__ == "__main__":
    main()
