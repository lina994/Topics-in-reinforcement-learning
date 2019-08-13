import SarsaAlgorithm as Sa


def print_to_file(end=False):
    with open('result.txt', 'a') as f:
        if end:
            f.write('*************************************************************************************\n')
        else:
            f.write('NUM_OF_EPISODES:%d, ' % Sa.NUM_OF_EPISODES)
            f.write('EPSILON:%.2f, ' % Sa.EPSILON)
            f.write('ALPHA:%.2f, ' % Sa.ALPHA)
            f.write('GAMMA:%.2f, ' % Sa.GAMMA)
            f.write('wins_l:%.2f, ' % (Sa.wins_l / Sa.NUM_OF_EPISODES))
            f.write('wins_t:%.2f' % (Sa.wins_t / Sa.NUM_OF_EPISODES))
            f.write('\n')


def long_test():
    epsilon_r = [0.01, 0.05, 0.1, 0.2]
    alpha_r = [0.05, 0.1, 0.2, 0.3, 0.4]
    gamma_r = [0.9, 1]
    for g in gamma_r:
        for a in alpha_r:
            for e in epsilon_r:
                Sa.reset_global()
                Sa.EPSILON = e
                Sa.ALPHA = a
                Sa.GAMMA = g
                Sa.learn_episodes()
                Sa.update_policy()
                Sa.print_result()
                Sa.play_game()
                print_to_file()
    print_to_file(True)


def short_test():
    Sa.learn_episodes()
    Sa.update_policy()
    Sa.print_result()
    Sa.play_game()


if __name__ == "__main__":
    short_test()
    # long_test()
    print("\n*****************finish**************************\n")

