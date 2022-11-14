import numpy as np
import matplotlib.pyplot as plt
import qpsolvers
from skimage.io import show
from qpsolvers import solve_qp
from lab_4_linear_classifiers.main import expand, r, linear_border, plot


def get_P(dataset, N):
    P = np.ndarray(shape=(N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = np.matmul(dataset[j, :-1, :].T * r(dataset[j, :, :]),
                                dataset[i, :-1, :] * r(dataset[i, :, :]))
    return P


def get_A(dataset, N):
    A = np.zeros((N,))
    for j in range(N):
        A[j] = r(dataset[j])
    return A


def get_w_expanded(dataset):

    N = dataset.shape[0]

    P = get_P(dataset, N)

    q = np.full((N, 1), -1)

    G = np.eye(N) * -1

    h = np.zeros((N,))

    A = get_A(dataset, N)

    b = np.zeros(1)

    _lambda = solve_qp(P, q, G, h, A, b, solver='osqp', max_iter=20000)

    w = np.matmul((_lambda * A).T, dataset[:, :-1, 0]).reshape(2, 1)

    w_N = np.mean(A[_lambda > 1.0e-04] - np.matmul(w.T, dataset[_lambda > 1.0e-04, :-1, 0].T))

    return expand(w, w_N)


# TODO: draw support vectors with different colour


def get_train_dataset(dataset1, dataset2):
    N = dataset1.shape[2] + dataset2.shape[2]
    train_dataset = []

    for i in range(N // 2):
        train_dataset.append(dataset1[:, :, i])
        # метка класса для функции r
        train_dataset[i * 2] = expand(train_dataset[i * 2], -1)

        train_dataset.append(dataset2[:, :, i])
        # метка класса для функции r
        train_dataset[i * 2 + 1] = expand(train_dataset[i * 2 + 1], 1)

    return np.array(train_dataset)


def main():
    dataset1 = np.load("../lab_6_svm/dataset_1.npy")
    dataset2 = np.load("../lab_6_svm/dataset_2.npy")
    dataset3 = np.load("../lab_6_svm/dataset_3.npy")
    dataset4 = np.load("../lab_6_svm/dataset_4.npy")

    train_dataset_separated = get_train_dataset(dataset1, dataset2)
    W = get_w_expanded(train_dataset_separated)
    print(W)

    y = np.arange(-4, 4, 0.1)
    x = linear_border(y, W)

    plot("", dataset1, dataset2, [x, x, x], [y, y + 1, y - 1], ['black', 'orange', 'blue'], ['', '', ''])
    show()

main()