import numpy as np
import matplotlib.pyplot as plt
import qpsolvers
from skimage.io import show
from qpsolvers import solve_qp
from lab_4_linear_classifiers.main import expand, r, linear_border, plot
from sklearn.svm import SVC, LinearSVC


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
    q = np.full((N, 1), -1, dtype=np.double)
    G = np.eye(N) * -1
    h = np.zeros((N,))
    A = get_A(dataset, N)
    b = np.zeros(1)
    _lambda = solve_qp(P, q, G, h, A, b, solver='cvxopt')

    w = np.matmul((_lambda * A).T, dataset[:, :-1, 0]).reshape(2, 1)
    w_N = np.mean(A[_lambda > 1e-04] - np.matmul(w.T, dataset[_lambda > 1e-04, :-1, 0].T))

    svc_clf = SVC(C=(np.max(_lambda) + 1e-04), kernel='linear')
    svc_clf.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

    dual_coef_svc_clf = svc_clf.dual_coef_
    support_vectors_svc_clf = svc_clf.support_vectors_

    W_svc_clf = np.matmul(dual_coef_svc_clf, support_vectors_svc_clf).reshape(2, 1)
    w_N_svc_clf = np.mean(np.sign(dual_coef_svc_clf) - np.matmul(W_svc_clf.T, support_vectors_svc_clf.T))

    return expand(w, w_N), expand(W_svc_clf, w_N_svc_clf)


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
    W, W_svc_clf = get_w_expanded(train_dataset_separated)

    print(f"W:\n{W}\nW_svc_clf:\n{W_svc_clf}\n")

    y = np.arange(-4, 4, 0.1)
    x = linear_border(y, W)
    x_svc_clf = linear_border(y, W_svc_clf)

    plot("", dataset1, dataset2, [x, x, x], [y, y + 1, y - 1], ['black', 'orange', 'blue'], ['', '', ''])
    plot(f"", dataset1, dataset2, [x_svc_clf, x_svc_clf, x_svc_clf], [y, y + 1, y - 1], ['black', 'orange', 'blue'], ['', '', ''])
    show()

main()