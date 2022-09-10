# Лабораторная работа 1. Вариант 25.
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import show

n = 2  # two components
N = 200  # number of values in selection


# Generating two normally distributed numbers from uniformly distributed numbers
def get_random_normal():
    mean = 0.5
    scale = np.sqrt(1 / 12)

    iterations = 144

    first = np.sum(np.random.uniform(size=iterations))
    second = np.sum(np.random.uniform(size=iterations))

    def __get_value(x): return (x - mean * iterations) / (scale * np.sqrt(iterations))

    return np.array(([__get_value(first)], [__get_value(second)]))


def get_a(b):
    a = np.zeros((n, n))
    a[0][0] = np.sqrt(b[0][0])
    a[1][0] = b[0][1] / np.sqrt(b[0][0])
    a[1][1] = np.sqrt(b[1][1] - (b[0][1] ** 2) / b[0][0])
    return a


def get_e():
    return get_random_normal()


def get_x(a, e, m):
    return np.matmul(a, e) + m


# Return N vectors, element is vector column
def get_selection(b, m):
    selection = np.zeros((N, n, 1))

    a = get_a(b)

    for i in range(N):
        e = get_e()
        x = get_x(a, e, m)
        selection[i] = x

    return selection


# Parameters estimation
def get_parameters(selection, m):
    est_mean = np.sum(selection, axis=0) / N

    est_b = 0
    for vector_col in selection:
        est_b += (np.matmul(vector_col, vector_col.transpose()) - np.matmul(m, m.transpose())) / N

    return est_mean, est_b


def main():
    # Const values
    m_1 = np.array(([0],
                    [-2]))

    m_2 = np.array(([-1],
                    [1]))

    m_3 = np.array(([2],
                    [0]))

    # Manually selected values
    b_1 = np.array(([1, 0],
                    [0, 1]))

    b_2 = np.array(([0.7, 0],
                    [0, 0.7]))

    b_3 = np.array(([0.5, 0],
                    [0, 0.5]))

    # Generating three selections
    selection_1 = get_selection(b_1, m_1)
    selection_2 = get_selection(b_2, m_2)
    selection_3 = get_selection(b_3, m_3)

    # Parameters estimation and printing
    parameters_1 = get_parameters(selection_1, m_1)
    parameters_2 = get_parameters(selection_2, m_2)
    parameters_3 = get_parameters(selection_3, m_3)
    print(f"M_1: \n{parameters_1[0]}\nB_1: \n{parameters_1[1]}\n")
    print(f"M_2: \n{parameters_2[0]}\nB_2: \n{parameters_2[1]}\n")
    print(f"M_3: \n{parameters_3[0]}\nB_3: \n{parameters_3[1]}\n")

    # Show selections
    plt.plot(selection_1[:, 0, :], selection_1[:, 1, :], '+', markersize=5, color='black')
    plt.plot(selection_2[:, 0, :], selection_2[:, 1, :], '*', markersize=5, color='red')
    plt.plot(selection_3[:, 0, :], selection_3[:, 1, :], '.', markersize=5, color='blue')
    show()


main()
# TODO: select b_3 value
