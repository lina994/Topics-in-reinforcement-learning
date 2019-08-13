
from PongEnv import PongGame
from Agent import DqnAgent
import DQN as Dqn
import VisualRepresent as VR
import JSONParser as JP
import tensorflow as tf
import os


tf.logging.set_verbosity(tf.logging.ERROR)


# ======================================== select action ========================================

def print_select_command_message():
    print("Select action:")
    print("1 - continue train model")
    print("2 - start train new model")
    print("3 - represent result")
    print("4 - run single episode according policy")
    print("5 - for exit")


def get_next_command():
    print_select_command_message()

    while True:
        try:
            action = int(input())
        except ValueError:
            print("That's not an integer!")
            continue

        if action == 1:
            continue_train_model()
        elif action == 2:
            train_new_model()
        elif action == 3:
            represent_result()
        elif action == 4:
            run_single_game()
        elif action == 5:
            break
        else:
            print("input should be integer between 1 to 5")
            continue
        print_select_command_message()


def continue_train_model():
    game_env = PongGame()  # init env
    agent = DqnAgent(game_env)  # init agent

    while True:
        print("enter directory to load:")
        work_dir = input()
        if os.path.isdir("./" + work_dir):
            break

    # input from user
    print("Enter num of iteration:")
    while True:
        try:
            Dqn.total_num_of_training = int(input())
            break
        except ValueError:
            print("That's not an integer!")
            continue

    episode, last_epsilon = JP.load_data(work_dir)
    Dqn.current_training = episode + 1
    Dqn.total_num_of_training += Dqn.current_training
    agent.epsilon = last_epsilon

    file_name = work_dir + "/data_file.txt"
    checkpoint_dir = work_dir + "/checkpoints_directory"

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Load a previous checkpoint if we find one
        latest_checkpoint = tf.train.latest_checkpoint(work_dir + "/checkpoints_directory")
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            sess.run(tf.global_variables_initializer())  # for initialized
            saver.restore(sess, latest_checkpoint)

        Dqn.run_dqn(sess, game_env, agent, work_dir, file_name, saver, checkpoint_dir)


def train_new_model():
    game_env = PongGame()  # init env
    agent = DqnAgent(game_env)  # init agent

    # input from user
    print("Enter num of iteration:")
    while True:
        try:
            Dqn.total_num_of_training = int(input())
            break
        except ValueError:
            print("That's not an integer!")
            continue

    # create new file
    output_directory = JP.create_results_directory()
    file_name = output_directory + "/data_file.txt"

    checkpoint_dir = JP.create_checkpoints_directory(output_directory)
    print(checkpoint_dir)

    saver = tf.train.Saver()

    # call train model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # for initialized
        Dqn.run_dqn(sess, game_env, agent, output_directory, file_name, saver, checkpoint_dir)


def represent_result():
    while True:
        print("enter directory to load:")
        work_dir = input()
        if os.path.isdir("./" + work_dir):
            break
    data = JP.read_data(work_dir)
    VR.visual_interface(data)


def run_single_game():
    game_env = PongGame()  # init env
    agent = DqnAgent(game_env)  # init agent

    while True:
        print("enter directory to load:")
        work_dir = input()
        if os.path.isdir("./" + work_dir):
            break

    episode, last_epsilon = JP.load_data(work_dir)
    Dqn.current_training = episode + 1
    Dqn.total_num_of_training += Dqn.current_training
    agent.epsilon = last_epsilon

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Load a previous checkpoint if we find one
        latest_checkpoint = tf.train.latest_checkpoint(work_dir + "/checkpoints_directory")
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            sess.run(tf.global_variables_initializer())  # for initialized
            saver.restore(sess, latest_checkpoint)
            game_env.run_single_episode(sess, agent)


if __name__ == "__main__":
    get_next_command()



