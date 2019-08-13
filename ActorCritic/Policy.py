
import random
import BasisFunction as Bs
import numpy as np


def softmax(h):
    exp = np.exp(h)
    return exp / np.sum(exp)


# ∇_θ log π_θ(s, a) = φ(s, a) − Σ πθ(s, b)φ(s, b)
# a is action
def softmax_derivative(state, a, theta, center_list, phi_matrix_size, action_size):
    phi_arr = [Bs.calc_phi(state, i, center_list, phi_matrix_size) for i in range(action_size)]
    h_arr = [np.dot(i.T, theta) for i in phi_arr]
    m = max(h_arr)
    h_arr = [i - m for i in h_arr]
    pi_arr = softmax(h_arr)
    phi_a = phi_arr[a]
    mean = 0
    for i in range(action_size):
        mean += phi_arr[i] * pi_arr[i]
    return phi_a - mean


def actor_action_according_policy(env, state, theta, center_list, phi_matrix_size):
    action_size = env.action_space.n
    phi_arr = [Bs.calc_phi(state, i, center_list, phi_matrix_size) for i in range(action_size)]
    h_arr = [np.dot(i.T, theta) for i in phi_arr]
    m = max(h_arr)
    h_arr = [i-m for i in h_arr]
    q = softmax(h_arr)
    return np.random.choice(env.action_space.n, p=q)


def get_random_action(env):
    return env.action_space.sample()


# Select action according to the policy(maximum value action).
# action that maximizes the Q value for this state.
# if there are few max action. return random one (like in ass1)
def action_according_policy(env, state, w, center_list, phi_matrix_size):
    q = []
    for action in range(env.action_space.n):
        q.append(Bs.calc_q_per_action(state, action, w, center_list, phi_matrix_size))

    # calculates argmax_a Q(state, a)
    max_value = max(q)  # get maximum of array
    indices = [action for action, value in enumerate(q) if value == max_value]
    return random.choice(indices)  # random choice of element from array


def get_next_action(env, state, w, center_list, phi_matrix_size):  # get action according policy
    if random.uniform(0, 1) < 0.01:
        return env.action_space.sample()  # random action
    return action_according_policy(env, state, w, center_list, phi_matrix_size)

