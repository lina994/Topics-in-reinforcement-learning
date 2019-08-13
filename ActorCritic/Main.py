import ActorCritic as Ac

best_average_result = 0
best_theta = []
best_w = []

if __name__ == "__main__":

    for i in range(50):
        print("main_iter", i)
        Ac.learn_episodes()
        ar = Ac.run_episodes()

        if ar > best_average_result:
            best_average_result = ar
            best_theta = Ac.theta
            best_w = Ac.w

        if ar > 200:
            print("finish after", i, "/", 50)
            break

    print("\n*****************result**************************\n")
    print("best average score is:", best_average_result)
    print("best w:")
    print(best_w)
    print("best theta:")
    print(best_theta)

    print("\n*****************finish**************************\n")

# w is:
# [19.5730632  -0.70868849  1.26841726 -0.66631781  0.70844383 19.67110215
#  -1.39977871  1.56091431 -0.72016613  1.11453679]
# theta is:
# [-2.31345053 -3.61089236 -0.17183591 -4.63212892  7.38001069  2.31345053
#   3.61089236  0.17183591  4.63212892 -7.38001069]

