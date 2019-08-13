
import gym
import numpy as np
import random
import time


class PongGame:
    def __init__(self):
        self.env = gym.make("PongDeterministic-v4")

        self.START_MEM_SIZE = 10000
        self.MAX_MEM_SIZE = 50000
        self.current_mem_size = 0
        self.memory = []  # (state, action, reward, state_t, done)
        self.BATCH_SIZE = 10
        self.TEST_EPISODES = 10
        self.episode_render = False

    # ============================================== create memory ==============================================

    def create_minimal_memory(self, agent, sess):
        self.memory = []  # (s, a, r, s_t)
        while self.current_mem_size < self.START_MEM_SIZE:
            self.add_episode_to_mem(agent, sess)

    def store_episode(self, state, action, reward, state_t, done):
        if self.current_mem_size < self.MAX_MEM_SIZE:
            self.memory.append((state, action, reward, state_t, done))
            self.current_mem_size += 1
        else:
            index = random.randint(0, self.MAX_MEM_SIZE - 1)
            self.memory[index] = (state, action, reward, state_t, done)

    def add_episode_to_mem(self, agent, sess):
        state = self.env.reset()
        state = agent.nn_img_pre_process.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        done = False

        while not done and self.current_mem_size < self.START_MEM_SIZE:
            action = agent.get_next_action(sess, self.env, state)

            state_t, reward, done, info = self.env.step(action)
            state_t = agent.nn_img_pre_process.process(sess, state_t)
            state_t = np.append(state[:, :, 1:], np.expand_dims(state_t, 2), axis=2)

            self.store_episode(state, action, reward, state_t, done)
            state = state_t
            if self.current_mem_size % 10 == 0:
                print("memory size", self.current_mem_size, "/", self.START_MEM_SIZE)

    # ============================================ sample from memory ============================================
    # random.sample(population, k)
    # Return a k length list of unique elements chosen from the population sequence.
    # Used for random sampling without replacement.

    def get_random_order_memory(self):
        return random.sample(self.memory, self.BATCH_SIZE)

    # ============================================ for visual test ============================================

    def run_single_episode(self, sess, agent):
        done = False
        rewards = 0

        state = self.env.reset()
        state = agent.nn_img_pre_process.process(sess, state)
        state = np.stack([state] * 4, axis=2)  # shape (84,84,4)

        while not done:
            self.env.render()
            action = agent.get_next_action(sess, self.env, state)

            s, reward, done, info = self.env.step(action)
            s = agent.nn_img_pre_process.process(sess, s)
            state = np.append(state[:, :, 1:], np.expand_dims(s, 2), axis=2)

            rewards += reward
            time.sleep(0.01)
        self.env.close()






