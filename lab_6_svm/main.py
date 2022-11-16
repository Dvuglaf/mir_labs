import numpy as np
import matplotlib.pyplot as plt
import qpsolvers
from skimage.io import show
from qpsolvers import solve_qp
from lab_4_linear_classifiers.main import expand, r, linear_border, plot, linear_classificator
from sklearn.svm import SVC, LinearSVC


def get_P(dataset, N):
    P = np.ndarray(shape=(N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = np.matmul(dataset[j, :-1, :].T * r(dataset[j, :, :]),
                                dataset[i, :-1, :] * r(dataset[i, :, :]))
    return P


def get_P_kernel(dataset, N, K, params):
    P = np.ndarray(shape=(N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = r(dataset[j, :, :]) * r(dataset[i, :, :]) * K(dataset[j, :-1, :], dataset[i, :-1, :], params)
    return P


def K_polynom(x, y, params):
    d = params[0]
    c = params[1]
    return pow(np.matmul(x.T, y)[0, 0] + c, d)


def K_radial(x, y, params):
    gamma = params[0]
    return np.exp(-gamma * np.sum(np.power((x - y), 2)))


def K_sigmoid(x, y, params):
    gamma = params[0]
    c = params[1]
    return np.tanh(gamma * np.matmul(x.T, y)[0, 0] + c)


def get_discriminant_kernel(support_vectors, _lambda_r, x, K, params):
    sum = 0
    for j in range(support_vectors.shape[0]):
        sum += _lambda_r[j] * K(support_vectors[j].reshape(2, 1), x, params)
    return sum


def get_A(dataset, N):
    A = np.zeros((N,))
    for j in range(N):
        A[j] = r(dataset[j])
    return A


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


def get_distance(W, x):
    return np.abs(np.matmul(W, expand(x, 1))) / np.sqrt(W[0] ** 2 + W[1] ** 2)

"""
кладбон неиспользованных функций

def get_support_vectors_by_classes(support_vectors, W):
    red_support_vectors = []
    green_support_vectors = []

    for el in support_vectors.reshape(support_vectors.shape[0], 2, 1):
        if linear_classificator(expand(el, 1), W) == 0:
            red_support_vectors.append(el)
        else:
            green_support_vectors.append(el)

    red_support_vectors = np.array(red_support_vectors)
    green_support_vectors = np.array(green_support_vectors)

    return red_support_vectors, green_support_vectors


def plot_support_vectors(support_vectors, W):
    # для отображения опорных векторов согласно их принадлежности классам
    red_support_vectors, green_support_vectors = get_support_vectors_by_classes(support_vectors, W)

    plt.scatter(red_support_vectors[:, 0, 0], red_support_vectors[:, 1, 0], color='red')
    plt.scatter(green_support_vectors[:, 0, 0], green_support_vectors[:, 1, 0], color='green')
"""


"""
Задание 2.
Построить линейный классификатор по методу опорных векторов на выборке с линейно разделимыми классами:
а) методом решения квадратичных задач solve_qp;
б) методом sklearn.svm.SVC;
в) методом sklearn.svm.LinearSVC.
"""
def task2(dataset1, dataset2):
    N = dataset1.shape[2] + dataset2.shape[2]

    # подготовка обучающей выборки
    dataset = get_train_dataset(dataset1, dataset2)

    # параметры для решения задачи квадратичного программирования
    P = get_P(dataset, N)
    q = np.full((N, 1), -1, dtype=np.double)
    G = np.eye(N) * -1
    h = np.zeros((N,))
    A = get_A(dataset, N)
    b = np.zeros(1)

    # получаем вектор двойственных коэффициентов
    _lambda = solve_qp(P, q, G, h, A, b, solver='cvxopt')

    # опорные вектора для метода solve_qp
    support_vectors_positions = _lambda > 1e-04
    support_vectors = dataset[support_vectors_positions, :-1, 0]
    support_vectors_classes = dataset[support_vectors_positions, -1, 0]
    red_support_vectors = support_vectors[support_vectors_classes == -1]
    green_support_vectors = support_vectors[support_vectors_classes == 1]

    # находим весовые коэффициенты из выражения через двойственные коэффициенты
    # и пороговое значение через весовые коэффициенты и опорные вектора
    W = np.matmul((_lambda * A)[support_vectors_positions].T, support_vectors).reshape(2, 1)
    w_N = np.mean(support_vectors_classes - np.matmul(W.T, support_vectors.T))
    W = expand(W, w_N)

    # обучение модели SVC (kernel=linear)
    svc_clf = SVC(C=(np.max(_lambda) + 1e-04), kernel='linear')
    svc_clf.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

    # опорные вектора для метода SVC
    support_vectors_svc = svc_clf.support_vectors_
    support_vectors_svc_indices = svc_clf.support_
    support_vectors_svc_classes = dataset[support_vectors_svc_indices, -1, 0]
    red_support_vectors_svc = support_vectors_svc[support_vectors_svc_classes == -1]
    green_support_vectors_svc = support_vectors_svc[support_vectors_svc_classes == 1]

    # весовые коэффициенты и пороговое значение для модели SVC
    W_svc_clf = svc_clf.coef_.T
    w_N_svc_clf = svc_clf.intercept_[0]
    W_svc_clf = expand(W_svc_clf, w_N_svc_clf)

    # обучение модели LinearSVC
    linear_svc_clf = LinearSVC(C=(np.max(_lambda) + 1e-04))
    linear_svc_clf.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

    # весовые коэффициенты и пороговое значение для модели LinearSVC
    W_linear_svc_clf = linear_svc_clf.coef_.T
    w_N_linear_svc_clf = linear_svc_clf.intercept_[0]
    W_linear_svc_clf = expand(W_linear_svc_clf, w_N_linear_svc_clf)

    # выводим весовые коэффициенты, полученные для каждого метода
    print(f"W:\n{W}\n"
          f"W_svc_clf:\n{W_svc_clf}\n"
          f"W_linear_svc_clf:\n{W_linear_svc_clf}\n")

    # строим разделяющую полосу и разделяющую гиперплоскость
    y = np.arange(-4, 4, 0.1)
    x = linear_border(y, W)
    x_svc_clf = linear_border(y, W_svc_clf)
    x_linear_svc_clf = linear_border(y, W_linear_svc_clf)

    plot(f"solve_qp (cvxopt)", dataset1, dataset2, [x, x + 1 / W[0], x - 1 / W[0]], [y, y, y],
         ['black', 'green', 'red'], ['', '', ''])
    plt.scatter(red_support_vectors[:, 0], red_support_vectors[:, 1], color='red')
    plt.scatter(green_support_vectors[:, 0], green_support_vectors[:, 1], color='green')

    plot(f"SVC(kernel=linear)", dataset1, dataset2,
         [x_svc_clf, x_svc_clf + 1 / W_svc_clf[0], x_svc_clf - 1 / W_svc_clf[0]], [y, y, y],
         ['black', 'green', 'red'], ['', '', ''])
    plt.scatter(red_support_vectors_svc[:, 0], red_support_vectors_svc[:, 1], color='red')
    plt.scatter(green_support_vectors_svc[:, 0], green_support_vectors_svc[:, 1], color='green')

    plot(f"LinearSVC", dataset1, dataset2,
         [x_linear_svc_clf, x_linear_svc_clf + 1 / W_linear_svc_clf[0], x_linear_svc_clf - 1 / W_linear_svc_clf[0]],
         [y, y, y],
         ['black', 'green', 'red'], ['', '', ''])


"""
Задание 3.
Построить линейный классификатор по SVM на выборке с линейно неразделимыми классами:
а) методом решения квадратичных задач solve_qp. Указать решения для C=0.1, 1, 10 и подобранным «лучшим коэффициентом»;
б) методом sklearn.svm.SVC.
"""
def task3(dataset3, dataset4):
    N = dataset3.shape[2] + dataset4.shape[2]

    # подготовка обучающей выборки
    dataset = get_train_dataset(dataset3, dataset4)

    # параметры для решения задачи квадратичного программирования
    P = get_P(dataset, N)
    q = np.full((N, 1), -1, dtype=np.double)
    G = np.concatenate((np.eye(N) * -1, np.eye(N)), axis=0)
    A = get_A(dataset, N)
    b = np.zeros(1)

    for C in [0.1, 1, 10, 20]:
        h = np.concatenate((np.zeros((N,)), np.full((N,), C)), axis=0)

        # получаем вектор двойственных коэффициентов
        _lambda = solve_qp(P, q, G, h, A, b, solver='cvxopt')

        # опорные вектора для метода solve_qp
        support_vectors_positions = _lambda > 1e-04
        support_vectors = dataset[support_vectors_positions, :-1, 0]
        support_vectors_classes = dataset[support_vectors_positions, -1, 0]
        red_support_vectors = support_vectors[support_vectors_classes == -1]
        green_support_vectors = support_vectors[support_vectors_classes == 1]

        # находим весовые коэффициенты из выражения через двойственные коэффициенты
        # и пороговое значение через весовые коэффициенты и опорные вектора
        W = np.matmul((_lambda * A)[support_vectors_positions].T, support_vectors).reshape(2, 1)
        w_N = np.mean(support_vectors_classes - np.matmul(W.T, support_vectors.T))
        W = expand(W, w_N)

        # обучение модели SVC
        svc_clf = SVC(C=C, kernel='linear')
        svc_clf.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

        # опорные вектора для метода SVC
        support_vectors_svc = svc_clf.support_vectors_
        support_vectors_svc_indices = svc_clf.support_
        support_vectors_svc_classes = dataset[support_vectors_svc_indices, -1, 0]
        red_support_vectors_svc = support_vectors_svc[support_vectors_svc_classes == -1]
        green_support_vectors_svc = support_vectors_svc[support_vectors_svc_classes == 1]

        # весовые коэффициенты и пороговое значение для модели SVC
        W_svc_clf = svc_clf.coef_.T
        w_N_svc_clf = svc_clf.intercept_[0]
        W_svc_clf = expand(W_svc_clf, w_N_svc_clf)

        print(f"C: {C}\n"
              f"W:\n{W}\n"
              f"W_svc_clf:\n{W_svc_clf}\n"
              f"Num of support vectors (solve_qp): {len(support_vectors)}\n"
              f"Num of support vectors (SVC): {len(support_vectors_svc)}")

        # строим разделяющую полосу и разделяющую гиперплоскость
        y = np.arange(-4, 4, 0.1)
        x = linear_border(y, W)
        x_svc_clf = linear_border(y, W_svc_clf)

        plot(f"solve_qp (cvxopt) C={C}", dataset3, dataset4, [x, x + 1 / W[0], x - 1 / W[0]], [y, y, y],
             ['black', 'green', 'red'], ['', '', ''])
        plt.scatter(red_support_vectors[:, 0], red_support_vectors[:, 1], color='red')
        plt.scatter(green_support_vectors[:, 0], green_support_vectors[:, 1], color='green')

        plot(f"SVC C={C}", dataset3, dataset4,
             [x_svc_clf, x_svc_clf + 1 / W_svc_clf[0], x_svc_clf - 1 / W_svc_clf[0]], [y, y, y],
             ['black', 'green', 'red'], ['', '', ''])
        plt.scatter(red_support_vectors_svc[:, 0], red_support_vectors_svc[:, 1], color='red')
        plt.scatter(green_support_vectors_svc[:, 0], green_support_vectors_svc[:, 1], color='green')


"""
Построить классификатор по SVM, разделяющий линейно неразделимые классы.
а) задачу (14) и метод решения квадратичных задач, исследовать решение для различных значений параметра C=1/10, 1, 10 и 
различных ядер из таблицы;
б) метод sklearn.svm.SVC.
"""
def task4(dataset3, dataset4):
    N = dataset3.shape[2] + dataset4.shape[2]

    # подготовка обучающей выборки
    dataset = get_train_dataset(dataset3, dataset4)

    # параметры для решения задачи квадратичного программирования
    # kernel = 'poly'
    # K = K_polynom
    # params = [3, 1]

    # kernel = 'rbf'
    # K = K_radial
    # params = [1]
    # gauss
    # var = np.var(np.sqrt(np.power(dataset[:, 0, :], 2) + np.power(dataset[:, 0, :], 2)))
    # params = [1 / (2 * var)]

    kernel = 'sigmoid'
    K = K_sigmoid
    params = [1 / 14, -1]

    P = get_P_kernel(dataset, N, K, params)
    q = np.full((N, 1), -1, dtype=np.double)
    G = np.concatenate((np.eye(N) * -1, np.eye(N)), axis=0)
    A = get_A(dataset, N)
    b = np.zeros(1)

    for C in [0.1, 1, 10]:
        h = np.concatenate((np.zeros((N,)), np.full((N,), C)), axis=0)

        # получаем вектор двойственных коэффициентов
        _lambda = solve_qp(P, q, G, h, A, b, solver='cvxopt')

        # опорные вектора для метода solve_qp
        support_vectors_positions = _lambda > 1e-04
        support_vectors = dataset[support_vectors_positions, :-1, 0]
        support_vectors_classes = dataset[support_vectors_positions, -1, 0]
        red_support_vectors = support_vectors[support_vectors_classes == -1]
        green_support_vectors = support_vectors[support_vectors_classes == 1]

        # находим пороговое значение через ядро и опорные вектора
        w_N = []
        for j in range(support_vectors.shape[0]):
            w_N.append(get_discriminant_kernel(support_vectors, (_lambda * A)[support_vectors_positions],
                                             support_vectors[j].reshape(2, 1), K, params))
        w_N = np.mean(support_vectors_classes - np.array(w_N))

        # обучение модели SVC
        # svc_clf = SVC(C=C, kernel=kernel, degree=3, coef0=1) # poly
        # svc_clf = SVC(C=C, kernel=kernel, gamma=1) # radial
        # svc_clf = SVC(C=C, kernel=kernel, gamma=1 / (2 * var)) # radial gauss
        svc_clf = SVC(C=C, kernel=kernel, coef0=-1, gamma=1/14) # sigmoid
        svc_clf.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

        # опорные вектора для метода SVC
        support_vectors_svc = svc_clf.support_vectors_
        support_vectors_svc_indices = svc_clf.support_
        support_vectors_svc_classes = dataset[support_vectors_svc_indices, -1, 0]
        red_support_vectors_svc = support_vectors_svc[support_vectors_svc_classes == -1]
        green_support_vectors_svc = support_vectors_svc[support_vectors_svc_classes == 1]

        # строим разделяющую полосу и разделяющую гиперплоскость
        y = np.linspace(-4, 4, N)
        x = np.linspace(-4, 4, N)
        # создаем координатную сетку
        xx, yy = np.meshgrid(x, y)
        # получаем множество векторов
        xy = np.vstack((xx.ravel(), yy.ravel())).T

        # получаем множество значений решающей функции для нашей сетки (solve_qp)
        discriminant_func_values = []
        for i in range(xy.shape[0]):
            discriminant_func_values.append(get_discriminant_kernel(support_vectors,
                                                                  (_lambda * A)[support_vectors_positions],
                                                                  xy[i].reshape(2, 1), K, params)
                                            + w_N)
        discriminant_func_values = np.array(discriminant_func_values).reshape(xx.shape)

        # получаем множество значений решающей функции для нашей сетки (SVC)
        discriminant_func_values_svc = svc_clf.decision_function(xy).reshape(xx.shape)

        # разделяющая полоса
        plot(f"solve_qp (cvxopt) ({kernel}) C={C}", dataset3, dataset4, [], [], ['black', 'green', 'red'], ['', '', ''])
        plt.contour(xx, yy, discriminant_func_values, levels=[-1, 0, 1], colors=['red', 'black', 'green'])
        plt.scatter(red_support_vectors[:, 0], red_support_vectors[:, 1], color='red')
        plt.scatter(green_support_vectors[:, 0], green_support_vectors[:, 1], color='green')

        plot(f"SVC ({kernel}) C={C}", dataset3, dataset4, [], [], ['black', 'green', 'red'], ['', '', ''])
        plt.contour(xx, yy, discriminant_func_values_svc, levels=[-1, 0, 1], colors=['red', 'black', 'green'])
        plt.scatter(red_support_vectors_svc[:, 0], red_support_vectors_svc[:, 1], color='red')
        plt.scatter(green_support_vectors_svc[:, 0], green_support_vectors_svc[:, 1], color='green')

    ...



def main():
    dataset1 = np.load("../lab_6_svm/dataset_1.npy")
    dataset2 = np.load("../lab_6_svm/dataset_2.npy")
    dataset3 = np.load("../lab_6_svm/dataset_3.npy")
    dataset4 = np.load("../lab_6_svm/dataset_4.npy")

    task2(dataset1, dataset2)
    # task3(dataset3, dataset4)
    # task4(dataset3, dataset4)

    show()

main()