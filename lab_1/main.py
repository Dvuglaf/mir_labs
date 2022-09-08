# Лабораторная работа 1. Вариант 25.
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import show

m_1 = np.array(([0], [-2]))
m_2 = np.array(([-1], [1]))
m_3 = np.array(([2], [0]))

n = 2


def generate_normal():
    mean = 0.5
    scale = np.sqrt(1 / 12)

    iterations = 144

    first = np.sum(np.random.uniform(size=iterations))
    second = np.sum(np.random.uniform(size=iterations))

    def __get_value(x): return (x - mean * iterations) / (scale * np.sqrt(iterations))

    return np.array(([__get_value(first)], [__get_value(second)]))


def get_a(b):
    a = np.zeros((2, 2))
    a[0][0] = np.sqrt(b[0][0])
    a[1][0] = b[0][1] / np.sqrt(b[0][0])
    a[1][1] = np.sqrt(b[1][1] - (b[0][1] ** 2) / b[0][0])
    return a


def get_e():
    return generate_normal()


def get_x(a, e, m):
    return np.matmul(a, e) + m


def get_parameters(selection, N):
    mean = np.sum(selection, axis=0) / N

    b = 0
    for i in range(N):
        b += (np.matmul(selection[i, :, :], selection[i, :, :].transpose()) - np.matmul(m_1, m_1.transpose()))
    b /= N

    return mean, b


# Return N vectors
def get_selection(a, m, N=200):
    selection = np.zeros((N, n, 1))

    for i in range(N):
        e = get_e()
        x = get_x(a, e, m)
        selection[i] = x

    return selection


def main():
    b_1 = np.array(([1, 0], [0, 1]))
    b_2 = np.array(([0.7, 0], [0, 0.7]))
    a_1 = get_a(b_1)
    a_2 = get_a(b_2)
    selection_1 = get_selection(a_1, m_1)
    selection_2 = get_selection(a_2, m_2)
    plt.plot(selection_1[:, 0, :], selection_1[:, 1, :], '+', markersize=5, color='black')
    plt.plot(selection_2[:, 0, :], selection_2[:, 1, :], '*', markersize=5, color='black')
    show()

    #parameters = get_parameters(selection, 200)
    #print(f"M: \n{parameters[0]}\nB: \n{parameters[1]}")


main()
