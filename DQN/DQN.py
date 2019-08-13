
import numpy as np
import JSONParser as Jp
import NeuralNetworkTF as NnTf


total_num_of_training = 0  # total num of training episodes
current_training = 0
MAX_STEPS = 10000  # max step in episode

GAMMA = 0.99

C_MAX_STEPS = 5000  # Every C steps reset Q'=Q
c_steps = 1


def initialize_replay_memory(game_env, agent, sess):
    game_env.create_minimal_memory(agent, sess)


"""
===================================================================================================================
# Algorithm: Deep Q-learning with Experience Replay (and target NN)

1.  Initialize replay memory D to capacity N
2.1 Initialize action-value function Q with random weights θ
2.2 Initialize target action-value function Q' with weight θ^=θ

3. for episode = 1, M do
4.     Initialise sequence s1 = {x1} and preprocessed sequenced φ1 = φ(s1)
5.     for t = 1, T do
6.         With probability epsilon select a random action a_t
7.         otherwise select a_t = max_a Q∗(φ(s_t), a; θ)
8.         Execute action a_t in emulator and observe reward r_t and image x_t+1
9.         Set s_t+1 = s_t, a_t, x_t+1 and preprocess φ_t+1 = φ(s_t+1)
10.        Store transition (φ_t, a_t, r_t, φ_t+1) in D
11.        Sample random minibatch of transitions (φ_j , a_j , r_j , φ_j+1) from D
12.        Set y_j =
              r_j                                 for terminal φj+1       (if episode terminates at step j+1)
              r_j + γ max_a' Q'(φ_j+1, a' ; θ^)   for non-terminal φj+1   (otherwise)
13.        Perform a gradient descent step on (y_j − Q(φ_j , a_j ; θ))^2 according to equation 3 (according theta)
           (with respect to the network parameter θ)
14.        Every C steps reset Q'=Q
    end for
end for
===================================================================================================================
"""


def run_dqn(sess, game_env, agent, output_directory, fileName, saver, checkpoint_dir):
    global current_training, c_steps, total_num_of_training
    # 1. Initialize replay memory D to capacity N
    initialize_replay_memory(game_env, agent, sess)

    # 2.1 Initialize action-value function Q with random weights
    # 2.2 Initialize target action-value function Q' with weight θ^=θ
    # happen in agent constructor

    # 3. for episode = 1, M do
    while current_training < total_num_of_training:

        # 4. Initialise sequence s1 = {x1} and preprocessed sequenced φ1 = φ(s1)
        state = game_env.env.reset()
        state = agent.nn_img_pre_process.process(sess, state)
        state = np.stack([state] * 4, axis=2)  # shape (84,84,4)

        score_per_episode = 0

        # 5. for t = 1, T do
        t = 0
        for t in range(MAX_STEPS):
            game_env.env.render()
            # 6. With probability epsilon select a random action a_t
            # 7. otherwise select a_t = max_a Q∗(φ(s_t), a; θ)
            action = agent.get_next_action(sess, game_env.env, state)          # 6 + 7

            # 8. Execute action a_t in emulator and observe reward r_t and image x_t+1
            next_state, reward, done, info = game_env.env.step(action)   # 8
            next_state = agent.nn_img_pre_process.process(sess, next_state)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

            score_per_episode += reward

            # 10. Store transition (φ_t, a_t, r_t, φ_t+1) in D
            game_env.store_episode(state, action, reward, next_state, done)

            # 9. Set s_t+1 = s_t, a_t, x_t+1 and preprocess φ_t+1 = φ(s_t+1)
            state = next_state

            # 11. Sample random minibatch of transitions (φ_j , a_j , r_j , φ_j+1) from D
            mini_batch = game_env.get_random_order_memory()
            # unzip batch elements
            states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = map(np.array, zip(*mini_batch))

            # 12. Set y_j =
            #       r_j                                 for terminal φj+1      (if episode terminates at step j+1)
            #       r_j + γ max_a' Q'(φ_j+1, a' ; θ^)   for non-terminal φj+1  (otherwise)

            q_values_next = agent.nn_q_learn.predict(sess, next_states_batch)
            # Returns the indices of the maximum values along an axis.
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = agent.nn_q_target.predict(sess, next_states_batch)

            toggle_dones_batch = []
            for done in dones_batch:
                if done:
                    toggle_dones_batch.append(0.0)
                else:
                    toggle_dones_batch.append(1.0)

            # np.arange(x) = array([0, 1, 2,...,x-1])
            targets_batch = \
                rewards_batch + np.array(toggle_dones_batch) * GAMMA \
                * q_values_next_target[np.arange(game_env.BATCH_SIZE), best_actions]

            # 13. Perform a gradient descent step on (y_j − Q(φ_j ,a_j ; θ))^2 according to equation 3 (according theta)
            states_batch = np.array(states_batch)
            loss = agent.nn_q_learn.update(sess, states_batch, actions_batch, targets_batch)

            # decrease epsilon
            agent.epsilon_decay()

            # 14. Every C steps reset Q'=Q

            if c_steps == C_MAX_STEPS:
                NnTf.reset_q_target(sess, agent.nn_q_learn, agent.nn_q_target)
                c_steps = 0

            c_steps += 1

            print("episode", current_training, "steps", t, "reward", score_per_episode, "loss", loss)

            if done:
                break

        # save reward and one per 100 episodes save model
        if Jp.save_reward(current_training, t, score_per_episode, agent.epsilon, loss, fileName):
            saver.save(sess, checkpoint_dir + "/myModel")

        print("episode", current_training, "steps", t, "reward", score_per_episode)
        current_training += 1

        # option to increase num of episodes
        if current_training == total_num_of_training:
            print("Enter num > 0 for continue training or 0")
            temp = 0
            while True:
                try:
                    temp = int(input())
                    break
                except ValueError:
                    print("That's not an integer!")
                    continue
            total_num_of_training += temp


