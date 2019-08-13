import os
from datetime import datetime
import json


information = []


def create_results_directory():
    dir_name = datetime.now().strftime("./%m%d_%H%M")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def create_checkpoints_directory(directory):
    dir_name = directory + "/checkpoints_directory"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def save_data(name, data):
    with open(name, "a+") as write_file:
        data = json.dumps(data)
        data = data + "\n"
        write_file.write(data)


def save_reward(current_training, t, reward, epsilon, loss, write_file):
    global information
    information.append([current_training, t, reward, epsilon, loss])
    if len(information) % 100 == 0:
        calc_average(write_file)
        information = []
        return True
    return False


def calc_average(name):
    # t = 0
    total_score = 0
    total_steps = 0
    total_loss = 0
    index = information[len(information)-1][0]
    last_epsilon = information[len(information)-1][3]

    for t in range(len(information)):
        total_steps += information[t][1]
        total_score += information[t][2]
        total_loss += information[t][4]

    total_steps /= len(information)
    total_score /= len(information)
    total_loss /= len(information)

    data = {
        "episode": index,
        "averageStepsPerEpisode": total_steps,
        "averageScore": total_score,
        "lastEpsilon": last_epsilon,
        "averageLoss": total_loss}

    save_data(name, data)
    print("data saved")


# load from file last episode and last epsilon
def load_data(directory):
    name = directory + "/data_file.txt"
    with open(name, "r") as read_file:
        prev_line = read_file.readline()
        line = prev_line
        if not line:
            print("file is empty, load default values: episode = 0, epsilon = 1")
            return 0, 1.0
        while line:
            prev_line = line
            line = read_file.readline()
        data = json.loads(prev_line[:-1])  # parse JSON
        episode = data["episode"]
        last_epsilon = data["lastEpsilon"]
        print("load values: episode = {}, epsilon = {}", episode, last_epsilon)
        return episode, last_epsilon


# read data_file to array
def read_data(directory):
    name = directory + "/data_file.txt"
    info = []
    with open(name, "r") as read_file:
        line = read_file.readline()
        while line:
            data = json.loads(line[:-1])  # parse JSON
            info.append(
                [data["episode"],
                 data["averageStepsPerEpisode"],
                 data["averageScore"],
                 data["lastEpsilon"],
                 data["averageLoss"]])
            line = read_file.readline()
    return info

