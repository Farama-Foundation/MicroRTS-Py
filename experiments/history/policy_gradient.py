import gym
import gym_microrts
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()

# Hyperparameters
learning_rate = 1e-4
gamma = 0.98
seed = 1
num_episodes = 5000

# Set up the env
env = gym.make("Microrts-v0")
np.random.seed(seed)
tf.random.set_random_seed(seed)
# env.seed(seed)

obs_ph = tf.placeholder(shape=(None,) + (4, 16, 16), dtype=tf.float64)
processed_layer = tf.reshape(obs_ph, (1, 1024))
fc1 = tf.layers.dense(inputs=processed_layer, units=64)
fc2 = tf.layers.dense(inputs=fc1, units=32)
fc3 = tf.layers.dense(inputs=fc2, units=32)
x = tf.layers.dense(inputs=fc3, units=16)
y = tf.layers.dense(inputs=fc3, units=16)
a_type = tf.layers.dense(inputs=fc3, units=4)
p_type = tf.layers.dense(inputs=fc3, units=4)

x_probs = tf.nn.softmax(x)
y_probs = tf.nn.softmax(y)
a_type_probs = tf.nn.softmax(a_type)
p_type_probs = tf.nn.softmax(p_type)

x_dist = tf.distributions.Categorical(probs=x_probs)
y_dist = tf.distributions.Categorical(probs=y_probs)
a_type_dist = tf.distributions.Categorical(probs=a_type_probs)
p_type_dist = tf.distributions.Categorical(probs=p_type_probs)

x_probs_chosen_indices_ph = tf.placeholder(shape=(None), dtype=tf.int32)
y_probs_chosen_indices_ph = tf.placeholder(shape=(None), dtype=tf.int32)
a_type_probs_chosen_indices_ph = tf.placeholder(shape=(None), dtype=tf.int32)
p_type_probs_chosen_indices_ph = tf.placeholder(shape=(None), dtype=tf.int32)

x_probs_chosen = tf.gather_nd(x_probs, x_probs_chosen_indices_ph)
y_probs_chosen = tf.gather_nd(y_probs, y_probs_chosen_indices_ph)
a_type_probs_chosen = tf.gather_nd(a_type_probs, a_type_probs_chosen_indices_ph)
p_type_probs_chosen = tf.gather_nd(p_type_probs, p_type_probs_chosen_indices_ph)

future_rewards_ph = tf.placeholder(shape=(None), dtype=tf.float64)
x_loss = -tf.reduce_mean(tf.log(x_probs_chosen) * future_rewards_ph)
y_loss = -tf.reduce_mean(tf.log(y_probs_chosen) * future_rewards_ph)
a_type_loss = -tf.reduce_mean(tf.log(a_type_probs_chosen) * future_rewards_ph)
p_type_loss = -tf.reduce_mean(tf.log(p_type_probs_chosen) * future_rewards_ph)

loss = x_loss + y_loss + a_type_loss + p_type_loss

# Update paramaters
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

episode_rewards = []
for i_episode in range(num_episodes):
    state = env.reset()
    episode = []
    states = []
    actions_taken = []
    rewards = []
    # One step in the environment
    for t in range(200):

        # Take a step
        action = sess.run(
            tf.squeeze([
                x_dist.sample(), 
                y_dist.sample(), 
                a_type_dist.sample(), 
                p_type_dist.sample()
            ]), feed_dict={
                obs_ph: [state]
        })
        print("test")
        print(action)
        next_state, reward, done, _ = env.step([action])
        

        print("yeah actions taken")

        # Keep track of the transition
        states += [state]
        actions_taken += [action]
        rewards += [reward]

        if done:
            break

        state = next_state

    if i_episode % 10 == 0:
        print(f"i_episode = {i_episode}, rewards = {sum(rewards)}")
        episode_rewards += [sum(rewards)]

    # Go through the episode and make policy updates
    #     # for t, item in enumerate(rewards):
    #     #     # The return after this timestep
    #     #     future_rewards = sum(rewards[t + 1 :])
    #     #     sess.run(
    #     #         train_op,
    #     #         feed_dict={
    #     #             obs_ph: [states[t]],
    #     #             action_probs_chosen_indices_ph: list(enumerate([actions_taken[t]])),
    #     #             future_rewards_ph: future_rewards * gamma ** (t),
    #     #         },
    #     #     )     


plt.plot(episode_rewards)
