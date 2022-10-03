import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show


def bayes_border(y, p1, p2, b1, b2, m1, m2):
    g = np.linalg.det(b1) * np.linalg.det(b2)
    d = 1 / g * (np.linalg.det(b2) * (m1[0] * b1[1][1] - m1[1] * b1[1][0])
                 - np.linalg.det(b1) * (m2[0] * b2[1][1] - m2[1] * b2[1][0]))
    e = 1 / g * (np.linalg.det(b2) * (m1[1] * b1[0][0] - m1[0] * b1[0][1])
                 - np.linalg.det(b1) * (m2[1] * b2[0][0] - m2[0] * b2[0][1]))
    f = np.log(np.linalg.det(b1) / np.linalg.det(b2)) + 2 * np.log(p1 / p2) \
        - np.matmul(np.matmul(np.transpose(m1), np.linalg.inv(b1)), m1) \
        + np.matmul(np.matmul(np.transpose(m2), np.linalg.inv(b2)), m2)

    if not np.array_equal(b1, b2):
        a = 1 / g * (np.linalg.det(b1) * b2[1][1] - np.linalg.det(b2) * b1[1][1])
        b = 1 / g * (np.linalg.det(b2) * (b1[1][0] + b1[0][1]) - np.linalg.det(b1) * (b2[1][0] + b2[0][1]))
        c = 1 / g * (np.linalg.det(b1) * b2[0][0] - np.linalg.det(b2) * b1[0][0])

        x1 = ((-b * y - d) + np.sqrt((b * y + d) ** 2 - 4 * a * (c * y ** 2 + f + e * y))) / (2 * a)
        x2 = ((-b * y - d) - np.sqrt((b * y + d) ** 2 - 4 * a * (c * y ** 2 + f + e * y))) / (2 * a)

        return x1, x2
    else:
        x = -1 / d * (e * y + f)
        return x


def main():
    N = 200

    m1 = np.array(([0], [-2]))
    m2 = np.array(([-1], [1]))
    m3 = np.array(([2], [0]))

    b = np.array(([0.5, -0.2], [-0.2, 0.5]))
    b1 = np.array(([0.5, 0], [0, 0.5]))
    b2 = np.array(([0.4, 0.1], [0.1, 0.6]))
    b3 = np.array(([0.6, -0.2], [-0.2, 0.6]))

    p = 1 / 2
    p1 = p2 = p3 = 1 / 3

    dataset1 = np.load("C:/Users/meshc/PycharmProjects/mir_labs/task_1_dataset_1.npy")
    dataset2 = np.load("C:/Users/meshc/PycharmProjects/mir_labs/task_1_dataset_2.npy")

    dataset21 = np.load("C:/Users/meshc/PycharmProjects/mir_labs/task_2_dataset_1.npy")
    dataset22 = np.load("C:/Users/meshc/PycharmProjects/mir_labs/task_2_dataset_2.npy")
    dataset23 = np.load("C:/Users/meshc/PycharmProjects/mir_labs/task_2_dataset_3.npy")

    y = np.arange(-4, 4, 0.1)

    x = bayes_border(y, p, p, b, b, m1, m2)
    x = np.reshape(x, y.shape)

    fig = plt.figure()
    plt.plot(dataset1[0, :, :], dataset1[1, :, :], color='red', marker='.')
    plt.plot(dataset2[0, :, :], dataset2[1, :, :], color='green', marker='+')
    plt.xlim(left=-4, right=4)
    plt.plot(x, y, color='black')

    _, x13 = bayes_border(y, p1, p3, b1, b3, m1, m3)
    x13 = np.reshape(x13, y.shape)

    x12, _ = bayes_border(y, p1, p2, b1, b2, m1, m2)
    x12 = np.reshape(x12, y.shape)

    _, x23 = bayes_border(y, p2, p3, b2, b3, m2, m3)
    x23 = np.reshape(x23, y.shape)

    fig = plt.figure()
    plt.plot(dataset21[0, :, :], dataset21[1, :, :], color='red', marker='.')
    plt.plot(dataset22[0, :, :], dataset22[1, :, :], color='green', marker='+')
    plt.plot(dataset23[0, :, :], dataset23[1, :, :], color='blue', marker='x')
    plt.plot(x13, y, color='cyan')
    plt.plot(x12, y, color='magenta')
    plt.plot(x23, y, color='yellow')

    show()


main()
