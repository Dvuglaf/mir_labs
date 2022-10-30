import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show
from lab_2_classifiers.main import bayes_classificator, bayes_discriminant, bayes_border_fixed


def classification_error(dataset, aprior0, aprior1, b0, b1, m0, m1, N, classificator) -> float:
    errors = 0  # показывает число неверно определенных элементов

    for i in range(N):
        errors += classificator(dataset[:, :, i], aprior0, aprior1, b0, b1, m0, m1)

    return errors / N  # ошибка первого рода


def get_risk(aprior0: float, aprior1: float, p0: float, p1: float) -> float:
    return aprior0 * p0 + aprior1 * p1


def get_W_fisher(m0: np.array, m1: np.array, b0: np.array, b1: np.array):
    return np.matmul(np.linalg.inv(0.5 * (b0 + b1)), (m1 - m0))


def get_sigma(W: np.array, b: np.array):
    return np.matmul(np.matmul(W.T, b), W)[0, 0]


def get_w_n(m0: np.array, m1: np.array, b0: np.array, b1: np.array, W: np.array):
    sigma_0_s = get_sigma(W, b0)
    sigma_1_s = get_sigma(W, b1)
    return (- np.matmul(
        (np.matmul((m1 - m0).T, np.linalg.inv(0.5 * (b1 + b0)))),
        (sigma_1_s * m0 + sigma_0_s * m1)) / (sigma_1_s + sigma_0_s))[0, 0]


def fisher_discriminant(x: np.array, m0: np.array, m1: np.array, b0: np.array, b1: np.array):
    W = get_W_fisher(m0, m1, b0, b1)
    w_n = get_w_n(m0, m1, b0, b1, W)

    return np.matmul(W.T, x) + w_n


def fisher_classificator(x: np.array, m0: np.array, m1: np.array, b0: np.array, b1: np.array):
    d = fisher_discriminant(x, m0, m1, b0, b1)
    return 1 if d > 0 else 0


def linear_border(y: np.array, W: np.array):
    a = W[0]
    b = W[1]
    c = W[2]

    if a == 0.:
        return 0
    else:
        return (-b * y - c) / a


def get_z(x: np.array, value):
    return np.append(x, np.array([[value]]), axis=0)


def get_U(dataset0: np.array, dataset1: np.array, k: int) -> np.array:
    z0 = []
    z1 = []

    for i in range(k // 2):
        z0.append(-dataset0[:, :, i])
        z0[i] = get_z(z0[i], -1)

        z1.append(dataset1[:, :, i])
        z1[i] = get_z(z1[i], 1)

    return np.concatenate((z0, z1))


def get_Gamma(U: np.array) -> np.array:
    return np.ones((U.shape[0], 1, 1))


def get_W_mse(U: np.array, Gamma: np.array) -> np.array:
    return np.matmul(
        np.matmul(np.linalg.inv(np.matmul(U[:, :, 0].T, U[:, :, 0])), U[:, :, 0].T),
        Gamma[:, :, 0])


def mse_discriminant(x: np.array, W: np.array) -> np.array:
    z = get_z(x, 1)
    return np.matmul(W.T, z)


def mse_classificator(x: np.array, W: np.array):
    d = mse_discriminant(x, W)
    return 1 if d > 0 else 0


def main():
    m0 = np.array([[0], [-2]])
    m1 = np.array([[-1], [1]])

    b = np.array(([0.5, -0.2], [-0.2, 0.5]))

    b0 = np.array(([0.5, 0], [0, 0.5]))
    b1 = np.array(([0.4, 0.1], [0.1, 0.6]))

    dataset00 = np.load("../lab_1_dataset_generation/task_1_dataset_1.npy")
    dataset01 = np.load("../lab_1_dataset_generation/task_1_dataset_2.npy")
    dataset10 = np.load("../lab_1_dataset_generation/task_2_dataset_1.npy")
    dataset11 = np.load("../lab_1_dataset_generation/task_2_dataset_2.npy")

    y = np.arange(-4, 4, 0.1)
    W_fisher = get_W_fisher(m0, m1, b0, b1)
    w_n = get_w_n(m0, m1, b0, b1, W_fisher)
    W_fisher = np.append(W_fisher, np.array([[w_n]]), axis=0)
    fisher_border_x = linear_border(y, W_fisher)

    plt.figure()
    plt.plot(dataset10[0, :, :], dataset10[1, :, :], color='red', marker='.')
    plt.plot(dataset11[0, :, :], dataset11[1, :, :], color='green', marker='+')
    plt.plot(fisher_border_x, y, color='black')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)

    U = get_U(dataset10, dataset11, 200)
    G = get_Gamma(U)
    W_mse = get_W_mse(U, G)
    mse_border_x = linear_border(y, W_mse)

    plt.figure()
    plt.plot(dataset10[0, :, :], dataset10[1, :, :], color='red', marker='.')
    plt.plot(dataset11[0, :, :], dataset11[1, :, :], color='green', marker='+')
    plt.plot(mse_border_x, y, color='black')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)

    show()


main()


