# -------------------------------------------- Imports ----------------------------------------------
import gym
import numpy as np

import BasisFunction as Bs
import Policy as Pl

# ----------------------------------- Q Actor Critic pseudo code ------------------------------------
# ------------------------ From Lecture 7, page 25 (David Silver RL course) -------------------------
"""
function QAC
1.    Initialise θ
2.    Initialise s          # initial state
3.    Sample a ∼ π_θ        # action
4.    for each step do
5.        Sample reward r = R(s, a)                 # reward
6.        Sample transition s' ∼ P(s, a)            # next state
7.        Sample action a' ∼ π_θ(s', a')            # next action
8.        δ = r + γQ_w (s', a') − Q_w (s, a)        # compute correction (TD error) for action-value at time t
9.        θ = θ + α ∇_θ log π_θ(s, a) Q_w (s, a)    # update the policy parameter (actor)
10.       w ← w + βδφ(s, a)                         # update parameters of Q function (critic)
11,       a ← a'                                    # action
12.       s ← s'                                    # state
      end for
end function
"""
# ------------------------------------ CartPole-v1 observation -------------------------------------
"""
Observation (feature vector): 
        Type: Box(4)
        Num	Observation              Min         Max
        1. Cart Position             -4.8        4.8
        2. Cart Velocity             -Inf        Inf
        3. Pole Angle                -24 deg     24 deg
        4. Pole Velocity At Tip      -Inf        Inf
"""
# -------------------------------------- CartPole-v1 actions ---------------------------------------
"""
Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
"""


# ------------------------------------------ Variables ---------------------------------------------

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

phi_matrix_size = (state_size + 1) * action_size                                    # size for CartPole = 10
center_list = [np.random.uniform(-1, 1, state_size) for i in range(state_size)]     # matrix 4x4 (list of np array)
# w = np.random.uniform(-1, 1, phi_matrix_size)                                     # initializing with a random policy
w = np.zeros(phi_matrix_size)                                       # initializing with a random policy

# theta = np.random.uniform(-1, 1, phi_matrix_size)           # 1.    Initialise θ
theta = np.zeros(phi_matrix_size)

NUM_OF_EPISODES = 1000
# EPSILON = 0.05
GAMMA = 0.95                 # γ, discount factor

ALPHA = 0.07                 # α, learning rate of theta (θ) -  actor
BETA = 0.07                  # β, learning rate of weight (w) - critic

TEST_SIZE = 50


# ----------------------------------- ACTOR CRITIC ALGORITHM  --------------------------------------

def q_actor_critic(state, action, state_t, action_t, reward):
    global env, w, theta

    q_s = Bs.calc_q_per_action(state, action, w, center_list, phi_matrix_size)         # 8.a compute Q_w (s, a)
    q_s_t = Bs.calc_q_per_action(state_t, action_t, w, center_list, phi_matrix_size)   # 8.b compute Q_w (s', a')

    delta = reward + (GAMMA * q_s_t) - q_s                              # 8.  δ = r + γQ_w (s', a') − Q_w (s, a)
    phi_s = Bs.calc_phi(state, action, center_list, phi_matrix_size)    # 9.a compute φ(s, a)
    det = Pl.softmax_derivative(state, action, theta, center_list, phi_matrix_size, action_size) # 9.b  compute ∇_θ log π_θ(s, a)
    theta += ALPHA * det * q_s                                          # 9. θ = θ + α ∇_θ log π_θ(s, a) Q_w (s, a)
    w += BETA * delta * phi_s                                           # 10. w ← w + βδφ(s, a)


# ------------------------------ Learn single game according policy ---------------------------------

def learn_single_episode():
    global env

    state = env.reset()                                                         # 2. Initialise s , Type: Box(4)
    action = Pl.actor_action_according_policy(env, state, w, center_list, phi_matrix_size)    # 3. Sample a ∼ π_θ

    done = False

    while not done:                                                                 # 4. for each step do
        state_t, reward, done, info = env.step(action)                              # 5. Sample reward r = R(a, s)
        #                                                                           # 6. Sample transition s' ∼ P(s, a)
        action_t = Pl.actor_action_according_policy(env, state_t, w, center_list, phi_matrix_size)   # 7.Sample action a'∼π_θ(s', a')

        q_actor_critic(state, action, state_t, action_t, reward)  # step 8-10

        action = action_t       # 11, a ← a'
        state = state_t         # 12. s ← s'
    env.close()


# ------------------------------ Learn multiple games according policy ---------------------------------

def learn_episodes():
    for _ in range(NUM_OF_EPISODES):
        learn_single_episode()


# ------------------------------ Play single game according policy ---------------------------------

def run_single_episode():
    global env
    done = False
    rewards_per_episode = 0
    state = env.reset()

    while not done:
        # env.render()
        action = Pl.action_according_policy(env, state, w, center_list, phi_matrix_size)
        # action = Pl.actor_action_according_policy(env, state, w, center_list, phi_matrix_size)
        state, reward, done, info = env.step(action)
        rewards_per_episode += reward

    return rewards_per_episode


def run_episodes():
    average_reward = 0
    for _ in range(TEST_SIZE):
        average_reward += run_single_episode()
    env.close()
    average_reward /= TEST_SIZE
    return average_reward

