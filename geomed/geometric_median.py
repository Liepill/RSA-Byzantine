import numpy as np


def cal_mean(mean_li):
    mean = np.zeros_like(mean_li[0])
    size = len(mean_li)
    for i in range(size):
        mean += mean_li[i]
    mean = mean / size
    return mean


def geometric_median(mean_li):
    max_iter = 1000
    tol = 1e-7
    guess = cal_mean(mean_li)
    iter = 0
    while iter < max_iter:
        dist_li = [np.linalg.norm(item - guess) for _, item in enumerate(mean_li)]
        temp1 = np.zeros_like(mean_li[0])
        temp2 = 0.0
        for elem1, elem2 in zip(mean_li, dist_li):
            if elem2 == 0:
                elem2 = 1.0
            temp1 += elem1 / elem2
            temp2 += 1.0 / elem2
        guess_next = temp1 / temp2
        guess_movement = np.linalg.norm(guess - guess_next)
        guess = guess_next
        if guess_movement <= tol:
            break
        iter += 1
    return guess