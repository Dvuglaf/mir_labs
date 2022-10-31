import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show
from lab_2_classifiers.main import bayes_classificator, bayes_discriminant, bayes_border_fixed


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


def get_U_mse(dataset0: np.array, dataset1: np.array, k: int) -> np.array:
    U = []

    for i in range(k // 2):
        U.append(-dataset0[:, :, i])
        U[i * 2] = expand(U[i * 2], -1)

        U.append(dataset1[:, :, i])
        U[i * 2 + 1] = expand(U[i * 2 + 1], 1)

    return np.array(U)


def get_Gamma(k: int) -> np.array:
    return np.ones((k, 1, 1))


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


def get_U_robbins_monro(dataset0: np.array, dataset1: np.array, k: int):
    U = []

    for i in range(k // 2):
        U.append(dataset0[:, :, i])
        U[i * 2] = expand(U[i * 2], 1)
        # метка класса для функции r
        U[i * 2] = expand(U[i * 2], -1)

        U.append(dataset1[:, :, i])
        U[i * 2 + 1] = expand(U[i * 2 + 1], 1)
        # метка класса для функции r
        U[i * 2 + 1] = expand(U[i * 2 + 1], 1)

    return np.array(U)


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
        W_next = W_prev + alpha_i * x_i[:-1] * np.sign(r(x_i) - np.matmul(W_prev.T, x_i[:-1]))
        W_prev = W_next

    return W_next


# алгоритм корректирующих приращений с улучшениями
def get_W_aci_boosted(W: np.array, U: np.array, k: int, beta: float):
    ...


def research(dataset00, dataset01, dataset10, dataset11):

    # на что влияет beta (spoiler: ни на что)

    W_next_equal = W_prev_equal = np.ones((3, 1))

    U_length = dataset00.shape[-1] + dataset01.shape[-1]
    U_robbins_monro_equal = get_U_robbins_monro(dataset00, dataset01, U_length)

    eps = np.full((3, 1), 0.005)
    iterations = {}

    for beta in np.arange(0.5, 1, 0.05):
        for i in range(1, U_length + 1):
            W_next_equal = get_W_aci(W_prev_equal, U_robbins_monro_equal[i - 1: i, :, :], 1, beta)
            diff = np.abs(W_next_equal - W_prev_equal)
            # print(f"{beta} : {i} : {diff.T}")

            if (diff < eps).any():
                iterations.update({beta : (i, W_next_equal.T, W_prev_equal.T)})
                break

            W_prev_equal = W_next_equal

    for key in iterations.keys():
        print(f"{key} : {iterations[key]}")


# алгоритм наименьшей СКО
def get_W_min_mse(W: np.array, U: np.array, k: int, beta: float):
    W_prev = W_next = W

    for i in range(1, k + 1):
        x_i = U[i - 1, :, :]
        alpha_i = get_alpha(i, beta)
        W_next = W_prev + alpha_i * x_i[:-1] * (r(x_i) - np.matmul(W_prev.T, x_i[:-1]))
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


def plot(title: str, dataset0: np.array, dataset1: np.array, border_x_arr, border_y_arr, colors, labels):
    plt.figure()
    plt.title(title)

    plt.plot(dataset0[0, :, :], dataset0[1, :, :], color='red', marker='.')
    plt.plot(dataset1[0, :, :], dataset1[1, :, :], color='green', marker='+')

    for i in range(len(border_x_arr)):
        plt.plot(border_x_arr[i], border_y_arr[i], color=colors[i], label=labels[i])

    plt.legend()

    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)


# критерий Фишера
def task1(m0, m1, b0, b1, b, dataset00, dataset01, dataset10, dataset11):
    y = np.arange(-4, 4, 0.1)

    borders_y = [y, y]

    colors = ["black", "blue"]
    labels = ["Фишер", "Байес"]

    # равные корреляционные матрицы
    W_fisher_equal = get_W_fisher(m0, m1, b, b)
    fisher_equal_border_x = linear_border(y, W_fisher_equal)

    bayes_equal_border_x = bayes_border_fixed(y, 0.5, 0.5, b, b, m0, m1)

    borders_x = [fisher_equal_border_x, bayes_equal_border_x]

    plot("Критерий Фишера, равные корреляционные матрицы", dataset00, dataset01, borders_x, borders_y, colors, labels)

    # разные корреляционные матрицы
    W_fisher = get_W_fisher(m0, m1, b0, b1)
    fisher_border_x = linear_border(y, W_fisher)

    bayes_border_x1, _ = bayes_border_fixed(y, 0.5, 0.5, b0, b1, m0, m1)

    borders_x = [fisher_border_x, bayes_border_x1]

    plot("Критерий Фишера, разные корреляционные матрицы", dataset10, dataset11, borders_x, borders_y, colors, labels)

#     TODO: посчитать риск


# критерий минимизации СКО
def task2(m0, m1, b0, b1, b, dataset00, dataset01, dataset10, dataset11, k):
    y = np.arange(-4, 4, 0.1)

    borders_y = [y, y, y]

    colors = ["black", "blue", "pink"]
    labels = ["Фишер", "Байес", "СКО"]

    # равные корреляционные матрицы
    W_fisher_equal = get_W_fisher(m0, m1, b, b)
    fisher_equal_border_x = linear_border(y, W_fisher_equal)

    bayes_equal_border_x = bayes_border_fixed(y, 0.5, 0.5, b, b, m0, m1)

    G = get_Gamma(k)
    U_mse_equal = get_U_mse(dataset00, dataset01, k)
    W_mse_equal = get_W_mse(U_mse_equal, G)

    mse_equal_border_x = linear_border(y, W_mse_equal)

    borders_x = [fisher_equal_border_x, bayes_equal_border_x, mse_equal_border_x]

    plot("Критерий минимизации СКО, равные корреляционные матрицы", dataset00, dataset01, borders_x, borders_y, colors, labels)

    # разные корреляционные матрицы
    W_fisher = get_W_fisher(m0, m1, b0, b1)
    fisher_border_x = linear_border(y, W_fisher)

    bayes_border_x1, _ = bayes_border_fixed(y, 0.5, 0.5, b0, b1, m0, m1)

    U_mse = get_U_mse(dataset10, dataset11, k)
    W_mse = get_W_mse(U_mse, G)

    mse_border_x = linear_border(y, W_mse)

    borders_x = [fisher_border_x, bayes_border_x1, mse_border_x]

    plot("Критерий минимизации СКО, разные корреляционные матрицы", dataset10, dataset11, borders_x, borders_y, colors, labels)

#     TODO: посчитать риск


# процедура Роббинса-Монро
def task3(m0, m1, b0, b1, b, dataset00, dataset01, dataset10, dataset11, k, beta):
    y = np.arange(-4, 4, 0.1)

    borders_y = [y, y, y]

    colors = ["black", "blue", "pink"]
    labels = ["АКП", "Байес", "НСКО"]

    # равные корреляционные матрицы
    U_robbins_monro_equal = get_U_robbins_monro(dataset00, dataset01, k)
    W_aci_equal = get_W_aci(np.full((3, 1), 1), U_robbins_monro_equal, k, beta)
    aci_equal_border_x = linear_border(y, W_aci_equal)

    bayes_equal_border_x = bayes_border_fixed(y, 0.5, 0.5, b, b, m0, m1)

    W_min_mse_equal = get_W_min_mse(np.full((3, 1), 1), U_robbins_monro_equal, k, beta)
    min_mse_equal_border_x = linear_border(y, W_min_mse_equal)

    borders_x = [aci_equal_border_x, bayes_equal_border_x, min_mse_equal_border_x]

    plot("Процедура Роббинса-Монро, равные корреляционные матрицы", dataset00, dataset01, borders_x, borders_y, colors, labels)

    # разные корреляционные матрицы
    U_robbins_monro = get_U_robbins_monro(dataset10, dataset11, k)
    W_aci = get_W_aci(np.full((3, 1), 1), U_robbins_monro, k, beta)
    aci_border_x = linear_border(y, W_aci)

    bayes_border_x1, _ = bayes_border_fixed(y, 0.5, 0.5, b0, b1, m0, m1)

    W_min_mse = get_W_min_mse(np.full((3, 1), 1), U_robbins_monro, k, beta)
    min_mse_border_x = linear_border(y, W_min_mse)

    borders_x = [aci_border_x, bayes_border_x1, min_mse_border_x]

    plot("Процедура Роббинса-Монро, разные корреляционные матрицы", dataset10, dataset11, borders_x, borders_y, colors, labels)

# TODO: посчитать риск, Исследовать
# TODO: зависимость скорости сходимости итерационного процесса и качества
# TODO: классификации от выбора начальных условий и выбора последовательности
# TODO: корректирующих коэффициентов.


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

    # task1(m0, m1, b0, b1, b, dataset00, dataset01, dataset10, dataset11)
    #
    # k = 100
    # task2(m0, m1, b0, b1, b, dataset00, dataset01, dataset10, dataset11, k)
    #
    # k = 200
    # beta = 0.7
    # task3(m0, m1, b0, b1, b, dataset00, dataset01, dataset10, dataset11, k, beta)
    #
    # show()

    research(dataset00, dataset01, dataset10, dataset11)


main()


