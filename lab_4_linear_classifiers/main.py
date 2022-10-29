import numpy as np
from lab_2_classifiers.main import bayes_classificator, bayes_discriminant, bayes_border_fixed


def classification_error(dataset, aprior0, aprior1, b0, b1, m0, m1, N, classificator) -> float:
    errors = 0  # показывает число неверно определенных элементов

    for i in range(N):
        errors += classificator(dataset[:, :, i], aprior0, aprior1, b0, b1, m0, m1)

    return errors / N  # ошибка первого рода


def get_risk(aprior0: float, aprior1: float, p0: float, p1: float) -> float:
    return aprior0 * p0 + aprior1 * p1


def fisher_discriminant(x: np.array, m0: np.array, m1: np.array, b0: np.array, b1: np.array):
    W = np.matmul(np.linalg.inv(0.5 * (b0 + b1)), (m1 - m0))

    sigma_0_s = np.matmul(np.matmul(W.T, b0), W)[0, 0]
    sigma_1_s = np.matmul(np.matmul(W.T, b1), W)[0, 0]

    w_n = - np.matmul(
        (np.matmul((m1 - m0).T, np.linalg.inv(0.5 * (b1 + b0)))),
        (sigma_1_s * m0 + sigma_0_s * m1)) / (sigma_1_s + sigma_0_s)

    return np.matmul(W.T, x) + w_n[0, 0]


def fisher_classificator(x: np.array, m0: np.array, m1: np.array, b0: np.array, b1: np.array):
    d = fisher_discriminant(x, m0, m1, b0, b1)
    return 1 if d > 0 else 0


def get_z(dataset0: np.array, dataset1: np.array, k: int) -> tuple:
    z0 = []
    z1 = []
    for i in range(k // 2):
        z0.append(-dataset0[:, :, i])
        z0[i] = np.append(z0[i], np.full((1, 1), -1), axis=0)
        z1.append(dataset1[:, :, i])
        z1[i] = np.append(z1[i], np.full((1, 1), 1), axis=0)

    return np.array(z0), np.array(z1)


def get_U(dataset0: np.array, dataset1: np.array, k: int) -> np.array:
    z0, z1 = get_z(dataset0, dataset1, k)
    return np.concatenate((z0, z1))


def get_Gamma(U: np.array) -> np.array:
    return np.ones((U.shape[0], 1, 1))


def get_W(U: np.array, Gamma: np.array) -> np.array:
    return np.matmul(
        np.matmul(np.linalg.inv(np.matmul(U[:, :, 0], U[:, :, 0].T)), U[:, :, 0]).T,
        Gamma[:, :, 0])


def mse_discriminant(x: np.array, W: np.array) -> np.array:
    z = np.append(x, np.array([[1]]), axis=0)
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

    dataset00 = np.load("task_1_dataset_1.npy")
    dataset01 = np.load("task_1_dataset_2.npy")
    dataset10 = np.load("task_2_dataset_1.npy")
    dataset11 = np.load("task_2_dataset_2.npy")
    U = get_U(dataset00, dataset01, 40)
    G = get_Gamma(U)
    W = get_W(U, G)
    for i in range(200):
        print(fisher_classificator(dataset00[:, :, i], m0, m1, b, b), "\t", mse_classificator(dataset00[:, :, i], W))



main()


