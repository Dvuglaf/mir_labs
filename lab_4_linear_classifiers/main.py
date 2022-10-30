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


def expand(x: np.array, value):
    return np.append(x, np.array([[value]]), axis=0)


def get_W_fisher(m0: np.array, m1: np.array, b0: np.array, b1: np.array):
    W_fisher = np.matmul(np.linalg.inv(0.5 * (b0 + b1)), (m1 - m0))
    w_n = get_w_n(m0, m1, b0, b1, W_fisher)
    return expand(W_fisher, w_n)


def get_sigma(W: np.array, b: np.array):
    return np.matmul(np.matmul(W.T, b), W)[0, 0]


def get_w_n(m0: np.array, m1: np.array, b0: np.array, b1: np.array, W: np.array):
    sigma_0_s = get_sigma(W, b0)
    sigma_1_s = get_sigma(W, b1)
    return (- np.matmul(
        (np.matmul((m1 - m0).T, np.linalg.inv(0.5 * (b1 + b0)))),
        (sigma_1_s * m0 + sigma_0_s * m1)) / (sigma_1_s + sigma_0_s))[0, 0]


# def fisher_discriminant(x: np.array, m0: np.array, m1: np.array, b0: np.array, b1: np.array):
#     W = get_W_fisher(m0, m1, b0, b1)
#     w_n = get_w_n(m0, m1, b0, b1, W)
#
#     return np.matmul(W.T, x) + w_n
#
#
# def fisher_classificator(x: np.array, m0: np.array, m1: np.array, b0: np.array, b1: np.array):
#     d = fisher_discriminant(x, m0, m1, b0, b1)
#     return 1 if d > 0 else 0


def get_U(dataset0: np.array, dataset1: np.array, k: int) -> np.array:
    U = []

    for i in range(k // 2):
        U.append(-dataset0[:, :, i])
        U[i * 2] = expand(U[i * 2], -1)

        U.append(dataset1[:, :, i])
        U[i * 2 + 1] = expand(U[i * 2 + 1], 1)

    return np.array(U)


def get_Gamma(U: np.array) -> np.array:
    return np.ones((U.shape[0], 1, 1))


def get_W_mse(U: np.array, Gamma: np.array) -> np.array:
    return np.matmul(
        np.matmul(np.linalg.inv(np.matmul(U[:, :, 0].T, U[:, :, 0])), U[:, :, 0].T),
        Gamma[:, :, 0])


# def mse_discriminant(x: np.array, W: np.array) -> np.array:
#     z = get_z(x, 1)
#     return np.matmul(W.T, z)
#
#
# def mse_classificator(x: np.array, W: np.array):
#     d = mse_discriminant(x, W)
#     return 1 if d > 0 else 0


def get_alpha(k: int, beta: float):
    return 1 / (k ** beta)


def r(x: np.array):
    return x[-1][0]


# алгоритм корректирующих приращений
def get_W_aci(W: np.array, U: np.array, k: int, beta: float):
    W_prev = W_next = W

    for i in range(1, k + 1):
        x_i = U[i - 1, :, :]
        alpha_i = get_alpha(i, beta)
        W_next = W_prev + alpha_i * x_i * np.sign(r(x_i) - np.matmul(W_prev.T, x_i))
        W_prev = W_next

    return W_next


def linear_discriminant(z: np.array, W: np.array) -> np.array:
    return np.matmul(W.T, z)


def linear_classificator(z: np.array, W: np.array):
    return 1 if linear_discriminant(z, W) > 0 else 0


def linear_border(y: np.array, W: np.array):
    a = W[0]
    b = W[1]
    c = W[2]

    if a == 0.:
        return 0
    else:
        return (-b * y - c) / a


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
    print(f"Fisher : y = {-W_fisher[0] / W_fisher[1]} * x + {-W_fisher[2] / W_fisher[1]}")
    fisher_border_x = linear_border(y, W_fisher)

    plt.figure()
    plt.plot(dataset10[0, :, :], dataset10[1, :, :], color='red', marker='.')
    plt.plot(dataset11[0, :, :], dataset11[1, :, :], color='green', marker='+')
    plt.plot(fisher_border_x, y, color='black')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)

    k = 400
    U = get_U(dataset10, dataset11, k)
    G = get_Gamma(U)
    W_mse = get_W_mse(U, G)
    print(f"mse : y = {-W_mse[0] / W_mse[1]} * x + {-W_mse[2] / W_mse[1]}")
    mse_border_x = linear_border(y, W_mse)

    plt.figure()
    plt.plot(dataset10[0, :, :], dataset10[1, :, :], color='red', marker='.')
    plt.plot(dataset11[0, :, :], dataset11[1, :, :], color='green', marker='+')
    plt.plot(mse_border_x, y, color='black')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)

    beta = 0.7
    W_aci = get_W_aci(np.full((3, 1), 1), U, k, beta)
    print(f"aci : y = {-W_aci[0] / W_aci[1]} * x + {-W_aci[2] / W_aci[1]}")
    aci_border_x = linear_border(y, W_aci)

    plt.figure()
    plt.plot(dataset10[0, :, :], dataset10[1, :, :], color='red', marker='.')
    plt.plot(dataset11[0, :, :], dataset11[1, :, :], color='green', marker='+')
    plt.plot(aci_border_x, y, color='black')
    # plt.xlim(left=-4, right=4)
    # plt.ylim(bottom=-4, top=4)

    show()


main()


