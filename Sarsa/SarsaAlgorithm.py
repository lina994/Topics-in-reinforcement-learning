"""
SARSA implementation of frozen_lake 4x4

pseudo code:
1. Initialize Q(s,a) arbitrarily
2. Repeat (for each episode):
    3. Initialize s
    4. Choose a from s using policy derived from Q (e.g., ε-greedy)
    5. Repeat (for each step of episode):
        6. Take action a, observe r, and new state s’
        7. Choose a’ from s’ using policy derived from Q (e.g., ε-greedy)
        8. Q(s_t, a_t) <- Q(s_t, a_t) + α([r_t+1 + γQ(s_t+1, a_t+1)] − Q(s_t, a_t))
        9. s <- s’
        10. a <- a’
    11. Until s is terminal
12. Improve the policy and repeat the process
"""

import random
import math
import gym


env = gym.make('FrozenLake-v0')
NUM_OF_STATES = env.observation_space.n  # 16 states
NUM_OF_ACTION = env.action_space.n  # 4 action
SIZE = math.floor(math.sqrt(NUM_OF_STATES))
NUM_OF_EPISODES = 10000
MAX_EPISODE_LONG = 100  # For prevent infinity game
EPSILON = 0.05
ALPHA = 0.2
GAMMA = 1

is_display = False
q_value = [[0 for j in range(NUM_OF_ACTION)] for i in range(NUM_OF_STATES)]  # 1. Initialize Q(s,a)
wins_l = 0  # wins during leaning process

NUM_OF_EPISODES_FOR_PLAY = 10000
policy_table = []
wins_t = 0  # wins during test process


# "enumerate" allows us to loop over something and have an automatic counter
# "choice" select a random element from an array
def get_max_value_action(state):
    global q_value
    max_value = max(q_value[state])
    indices = [counter for counter, value in enumerate(q_value[state]) if value == max_value]
    return random.choice(indices)


# randomly generate a number x between 0 and 1
# if x < epsilon  -> next step is random
# if x >= epsilon -> next step with max value
# return: observation, reward, done, info
def get_next_action(state):
    global env, EPSILON, q_value
    if random.uniform(0, 1) < EPSILON:
        return env.action_space.sample()  # random action
    return get_max_value_action(state)  # step with max value


# Q(s_t, a_t) <- Q(s_t, a_t) + α([r_t+1 + γQ(s_t+1, a_t+1)] − Q(s_t, a_t))
def update_q_value_table_sarsa(state, action, next_state, next_action, reward):
    global q_value, ALPHA, GAMMA
    temp = q_value[state][action]
    temp += ALPHA * (reward + GAMMA * q_value[next_state][next_action] - q_value[state][action])
    q_value[state][action] = temp


def learn_single_episode():
    global env, wins_l, is_display, MAX_EPISODE_LONG
    done = False
    env.reset()
    num_of_steps = 0
    if is_display:
        env.render()
    state = 0                                              # 3. Initialize s
    action = get_next_action(state)                        # 4. Choose a from s using policy derived from Q
    while not done and num_of_steps <= MAX_EPISODE_LONG:   # 5. Repeat (for each step of episode):
        num_of_steps += 1
        next_state, reward, done, info = env.step(action)  # 6. Take action a, observe r, and new state s’
        next_action = get_next_action(next_state)          # 7. Choose a’ from s’
        update_q_value_table_sarsa(state, action, next_state, next_action, reward)  # 8. update Q(s_t, a_t)
        state = next_state      # 9. s <- s’
        action = next_action    # 10. a <- a’
        if is_display:
            env.render()
        wins_l += reward


def learn_episodes():
    global NUM_OF_EPISODES, is_display
    for episode in range(NUM_OF_EPISODES):  # 2. Repeat (for each episode):
        if is_display:
            print("episode", episode, ":")
        learn_single_episode()


# *************************************************************
# *********************update  policy**************************
# *************************************************************

def update_policy():
    global policy_table, SIZE
    policy_table = [[get_max_value_action(SIZE*row+column) for column in range(SIZE)] for row in range(SIZE)]
    # policy_table = [get_max_value_action(state) for state in range(NUM_OF_STATES)]


# print result of learning to terminal
def print_result():
    global NUM_OF_EPISODES, wins_l, q_value, policy_table
    print("Q table result is:")
    for state_row in q_value:
        print(state_row)
    print("\nWins during learning process {0:.2f}".format(wins_l / NUM_OF_EPISODES))
    print("\n*************************************************\n")
    print("Policy table is:")
    for state_row in policy_table:
        print(state_row)
    print("\n*************************************************\n")


# *************************************************************
# ***************run according policy**************************
# *************************************************************

def get_policy_action(state):
    global policy_table, SIZE
    row = state // SIZE
    column = state % SIZE
    return policy_table[row][column]


def play_round():
    global env, wins_t, is_display, MAX_EPISODE_LONG
    done = False
    env.reset()
    num_of_steps = 0
    if is_display:
        env.render()
    state = 0
    while not done and num_of_steps <= MAX_EPISODE_LONG:
        num_of_steps += 1
        action = get_policy_action(state)
        state, reward, done, info = env.step(action)
        if is_display:
            env.render()
        wins_t += reward


# print result of playing to terminal
def print_result_of_playing():
    global NUM_OF_EPISODES_FOR_PLAY, wins_t
    # print("wins_t:", wins_t)
    print("\nWins during test {0:.2f}".format(wins_t / NUM_OF_EPISODES_FOR_PLAY))
    print("\n*************************************************\n")


def play_game():
    global NUM_OF_EPISODES_FOR_PLAY, is_display
    for episode in range(NUM_OF_EPISODES_FOR_PLAY):
        if is_display:
            print("episode", episode, ":")
        play_round()
    print_result_of_playing()


# reset global variables
def reset_global():
    global q_value, policy_table, wins_l, wins_t
    q_value = [[0 for j in range(NUM_OF_ACTION)] for i in range(NUM_OF_STATES)]
    wins_l = 0
    policy_table = []
    wins_t = 0

