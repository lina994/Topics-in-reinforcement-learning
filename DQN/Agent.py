import numpy as np
from NeuralNetworkTF import ProcessImage, NeuralNetwork

EPSILON_UPPER = 1.0
EPSILON_LOWER = 0.1
EPSILON_DECAY = 0.000001


class DqnAgent:
    def __init__(self, pong_game):
        self.epsilon = EPSILON_UPPER
        self.game = pong_game
        self.state_size = pong_game.env.observation_space.shape
        self.action_size = pong_game.env.action_space.n

        # Neural networks
        self.nn_img_pre_process = ProcessImage()
        self.nn_q_learn = NeuralNetwork("q_learn")
        self.nn_q_target = NeuralNetwork("q_target")

    def epsilon_decay(self):
        if self.epsilon > EPSILON_LOWER:
            self.epsilon -= EPSILON_DECAY

    # select random action probability epsilon
    # select next action according policy with probability 1-epsilon
    def get_next_action(self, sess, env, state):
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return self.get_max_action(sess, env, state)

    # select next action according policy
    # state [None, 84, 84, 4]
    def get_max_action(self, sess, env, state):
        # np.expand_dims((x1,x2,...)), 0) = (1,x1,x2,...)
        q_values = self.nn_q_learn.predict(sess, np.expand_dims(state, 0))[0]
        best_action = np.argmax(q_values)  # get action with max q value
        return best_action
