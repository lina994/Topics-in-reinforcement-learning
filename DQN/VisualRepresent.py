import matplotlib.pyplot as plt


def print_select_visual_command_message():
    print("Select action:")
    print("1 - line chart: x: episode, y: reward")
    print("2 - line chart: x: episode, y: episode length")
    print("3 - line chart: x: episode, y: epsilon")
    print("4 - line chart: x: episode, y: loss")
    print("5 - return to main menu")


def visual_interface(data):
    print_select_visual_command_message()

    while True:
        try:
            action = int(input())
        except ValueError:
            print("That's not an integer!")
            continue

        if action == 1:
            line_chart_plot(data, 0, 2, "reward")
        elif action == 2:
            line_chart_plot(data, 0, 1, "episode length")
        elif action == 3:
            line_chart_plot(data, 0, 3, "epsilon")
        elif action == 4:
            line_chart_plot(data, 0, 4, "loss")
        elif action == 5:
            return
        else:
            print("input should be integer between 1 to 5")
            continue
        print_select_visual_command_message()


"""
0: data["episode"],
1: data["averageStepsPerEpisode"],
2: data["averageScore"],
3: data["lastEpsilon"],
4: data["averageLoss"]
"""


def line_chart_plot(data, x_index, y_index, y_label):
    x = []
    y = []
    for d in data:
        x.append(d[x_index])
        y.append(d[y_index])
    plt.plot(x, y)
    plt.ylabel("Score")
    plt.xlabel(y_label)
    plt.grid(linestyle='--')
    plt.show()


