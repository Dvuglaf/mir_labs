import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show
from lab_2_classifiers.main import bayes_classificator, bayes_discriminant, bayes_border_fixed
from lab_2_classifiers.main import classification_error as bayes_error


# def get_risk(aprior0: float, aprior1: float, p0: float, p1: float) -> float:
#     return aprior0 * p0 + aprior1 * p1


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
def get_W_aci(W: np.array, U: np.array, size: int, beta: float, k: int):
    W_prev = W_next = W

    j = k

    for i in range(1, size + 1):
        x_i = U[i - 1, :, :]

        alpha_i = get_alpha(j, beta)
        j += 1

        W_next = W_prev + alpha_i * x_i[:-1] * np.sign(r(x_i) - np.matmul(W_prev.T, x_i[:-1]))
        W_prev = W_next

    return W_next


# алгоритм корректирующих приращений с улучшениями
def get_W_aci_boosted(W: np.array, U: np.array, size: int, beta: float, k: int):
    W_prev = W_next = W

    prev_sign = 0.

    j = k
    alpha_j = 1.

    for i in range(1, size + 1):
        x_i = U[i - 1, :, :]

        if np.sign(r(x_i)) != np.sign(np.matmul(W_prev.T, x_i[:-1])):

            sign = np.sign(r(x_i) - np.matmul(W_prev.T, x_i[:-1]))
            if sign != prev_sign:

                alpha_j = get_alpha(j, beta)
                j += 1

                prev_sign = sign

            W_next = W_prev + alpha_j * x_i[:-1] * sign
            W_prev = W_next

    return W_next


# алгоритм наименьшей СКО
def get_W_min_mse(W: np.array, U: np.array, size: int, beta: float, k: int):
    W_prev = W_next = W

    j = k

    for i in range(1, size + 1):
        x_i = U[i - 1, :, :]
        alpha_i = get_alpha(j, beta)
        j += 1
        W_next = W_prev + alpha_i * x_i[:-1] * (r(x_i) - np.matmul(W_prev.T, x_i[:-1]))
        W_prev = W_next

    return W_next


# алгоритм наименьшей СКО с улучшениями
def get_W_min_mse_boosted(W: np.array, U: np.array, size: int, beta: float, k: int):
    W_prev = W_next = W

    prev_sign = 0.

    j = k
    alpha_j = 1.

    for i in range(1, size + 1):
        x_i = U[i - 1, :, :]

        if np.sign(r(x_i)) != np.sign(np.matmul(W_prev.T, x_i[:-1])):

            sign = np.sign(r(x_i) - np.matmul(W_prev.T, x_i[:-1]))
            if sign != prev_sign:
                alpha_j = get_alpha(j, beta)
                j += 1

                prev_sign = sign

            W_next = W_prev + alpha_j * x_i[:-1] * (r(x_i) - np.matmul(W_prev.T, x_i[:-1]))

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


def classification_error(dataset, W, class_id):
    errors = 0  # показывает число неверно определенных элементов
    N = dataset.shape[-1]

    for i in range(N):
        z = expand(dataset[:, :, i], 1)
        if linear_classificator(z, W) != class_id:
            errors += 1

    return errors / N  # ошибка первого рода


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
    print("Задание 1:")

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

    # подсчет вероятностей ошибок классификации
    bayes_error_p0 = bayes_error(dataset00, 0.5, 0.5, b, b, m0, m1, dataset00.shape[-1])
    bayes_error_p1 = bayes_error(dataset01, 0.5, 0.5, b, b, m1, m0, dataset01.shape[-1])

    print(f"Вероятности ошибок Байессовского классификатора при равных корреляционных матрицах:"
          f"\np0={bayes_error_p0}\np1={bayes_error_p1}")

    fisher_error_p0 = classification_error(dataset00, W_fisher_equal, 0)
    fisher_error_p1 = classification_error(dataset01, W_fisher_equal, 1)

    print(f"Вероятности ошибок классификатора (критерий Фишера) при равных корреляционных матрицах:"
          f"\np0={fisher_error_p0}\np1={fisher_error_p1}")


    # разные корреляционные матрицы
    W_fisher = get_W_fisher(m0, m1, b0, b1)
    fisher_border_x = linear_border(y, W_fisher)

    bayes_border_x1, _ = bayes_border_fixed(y, 0.5, 0.5, b0, b1, m0, m1)

    borders_x = [fisher_border_x, bayes_border_x1]

    plot("Критерий Фишера, разные корреляционные матрицы", dataset10, dataset11, borders_x, borders_y, colors, labels)

    # подсчет вероятностей ошибок классификации
    bayes_error_p0 = bayes_error(dataset10, 0.5, 0.5, b0, b1, m0, m1, dataset10.shape[-1])
    bayes_error_p1 = bayes_error(dataset11, 0.5, 0.5, b1, b0, m1, m0, dataset11.shape[-1])

    print(f"Вероятности ошибок Байессовского классификатора при разных корреляционных матрицах:"
          f"\np0={bayes_error_p0}\np1={bayes_error_p1}")

    fisher_error_p0 = classification_error(dataset10, W_fisher, 0)
    fisher_error_p1 = classification_error(dataset11, W_fisher, 1)

    print(f"Вероятности ошибок классификатора (критерий Фишера) при разных корреляционных матрицах:"
          f"\np0={fisher_error_p0}\np1={fisher_error_p1}")


# критерий минимизации СКО
def task2(m0, m1, b0, b1, b, dataset00, dataset01, dataset10, dataset11, k):
    print("Задание 2:")

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

    # подсчет вероятностей ошибок классификации
    bayes_error_p0 = bayes_error(dataset00, 0.5, 0.5, b, b, m0, m1, dataset00.shape[-1])
    bayes_error_p1 = bayes_error(dataset01, 0.5, 0.5, b, b, m1, m0, dataset01.shape[-1])

    print(f"Вероятности ошибок Байессовского классификатора при равных корреляционных матрицах:"
          f"\np0={bayes_error_p0}\np1={bayes_error_p1}")

    fisher_error_p0 = classification_error(dataset00, W_fisher_equal, 0)
    fisher_error_p1 = classification_error(dataset01, W_fisher_equal, 1)

    print(f"Вероятности ошибок классификатора (критерий Фишера) при равных корреляционных матрицах:"
          f"\np0={fisher_error_p0}\np1={fisher_error_p1}")

    mse_error_p0 = classification_error(dataset00, W_mse_equal, 0)
    mse_error_p1 = classification_error(dataset01, W_mse_equal, 1)

    print(f"Вероятности ошибок классификатора (критерий минимума СКО) при равных корреляционных матрицах:"
          f"\np0={mse_error_p0}\np1={mse_error_p1}")

    # разные корреляционные матрицы
    W_fisher = get_W_fisher(m0, m1, b0, b1)
    fisher_border_x = linear_border(y, W_fisher)

    bayes_border_x1, _ = bayes_border_fixed(y, 0.5, 0.5, b0, b1, m0, m1)

    U_mse = get_U_mse(dataset10, dataset11, k)
    W_mse = get_W_mse(U_mse, G)

    mse_border_x = linear_border(y, W_mse)

    borders_x = [fisher_border_x, bayes_border_x1, mse_border_x]

    plot("Критерий минимизации СКО, разные корреляционные матрицы", dataset10, dataset11, borders_x, borders_y, colors, labels)

    # подсчет вероятностей ошибок классификации
    bayes_error_p0 = bayes_error(dataset10, 0.5, 0.5, b0, b1, m0, m1, dataset10.shape[-1])
    bayes_error_p1 = bayes_error(dataset11, 0.5, 0.5, b1, b0, m1, m0, dataset11.shape[-1])

    print(f"Вероятности ошибок Байессовского классификатора при разных корреляционных матрицах:"
          f"\np0={bayes_error_p0}\np1={bayes_error_p1}")

    fisher_error_p0 = classification_error(dataset10, W_fisher, 0)
    fisher_error_p1 = classification_error(dataset11, W_fisher, 1)

    print(f"Вероятности ошибок классификатора (критерий Фишера) при разных корреляционных матрицах:"
          f"\np0={fisher_error_p0}\np1={fisher_error_p1}")

    mse_error_p0 = classification_error(dataset10, W_mse, 0)
    mse_error_p1 = classification_error(dataset11, W_mse, 1)

    print(f"Вероятности ошибок классификатора (критерий минимума СКО) при разных корреляционных матрицах:"
          f"\np0={mse_error_p0}\np1={mse_error_p1}")


# процедура Роббинса-Монро
def task3(m0, m1, b0, b1, b, dataset00, dataset01, dataset10, dataset11, k, beta):
    print("Задание 3:")

    y = np.arange(-4, 4, 0.1)

    borders_y = [y, y, y, y, y]

    colors = ["black", "blue", "yellow", "magenta", "orange"]
    labels = ["Байес", "АКП", "НСКО", "АКП улучшенный", "НСКО улучшенный"]

    # равные корреляционные матрицы
    U_robbins_monro_equal = get_U_robbins_monro(dataset00, dataset01, k)
    W_aci_equal = get_W_aci(np.full((3, 1), 1), U_robbins_monro_equal, k, beta, 1)
    aci_equal_border_x = linear_border(y, W_aci_equal)

    W_aci_equal_boosted = get_W_aci_boosted(np.full((3, 1), 1), U_robbins_monro_equal, k, beta, 1)
    aci_equal_boosted_border_x = linear_border(y, W_aci_equal_boosted)

    bayes_equal_border_x = bayes_border_fixed(y, 0.5, 0.5, b, b, m0, m1)

    W_min_mse_equal = get_W_min_mse(np.full((3, 1), 1), U_robbins_monro_equal, k, beta, 1)
    min_mse_equal_border_x = linear_border(y, W_min_mse_equal)

    W_min_mse_equal_boosted = get_W_min_mse_boosted(np.full((3, 1), 1), U_robbins_monro_equal, k, beta, 1)
    min_mse_equal_boosted_border_x = linear_border(y, W_min_mse_equal_boosted)

    borders_x = [bayes_equal_border_x, aci_equal_border_x, min_mse_equal_border_x, aci_equal_boosted_border_x, min_mse_equal_boosted_border_x]

    plot("Процедура Роббинса-Монро, равные корреляционные матрицы", dataset00, dataset01, borders_x, borders_y, colors, labels)

    # подсчет вероятностей ошибок классификации
    bayes_error_p0 = bayes_error(dataset00, 0.5, 0.5, b, b, m0, m1, dataset00.shape[-1])
    bayes_error_p1 = bayes_error(dataset01, 0.5, 0.5, b, b, m1, m0, dataset01.shape[-1])

    print(f"Вероятности ошибок Байессовского классификатора при равных корреляционных матрицах:"
          f"\np0={bayes_error_p0}\np1={bayes_error_p1}")

    aci_error_p0 = classification_error(dataset00, W_aci_equal, 0)
    aci_error_p1 = classification_error(dataset01, W_aci_equal, 1)

    print(f"Вероятности ошибок классификатора (АКП, обычный) при равных корреляционных матрицах:"
          f"\np0={aci_error_p0}\np1={aci_error_p1}")

    aci_boosted_error_p0 = classification_error(dataset00, W_aci_equal_boosted, 0)
    aci_boosted_error_p1 = classification_error(dataset01, W_aci_equal_boosted, 1)

    print(f"Вероятности ошибок классификатора (АКП, улучшенный) при равных корреляционных матрицах:"
          f"\np0={aci_boosted_error_p0}\np1={aci_boosted_error_p1}")

    min_mse_error_p0 = classification_error(dataset00, W_min_mse_equal, 0)
    min_mse_error_p1 = classification_error(dataset01, W_min_mse_equal, 1)

    print(f"Вероятности ошибок классификатора (НСКО, обычный) при равных корреляционных матрицах:"
          f"\np0={min_mse_error_p0}\np1={min_mse_error_p1}")

    min_mse_boosted_error_p0 = classification_error(dataset00, W_min_mse_equal_boosted, 0)
    min_mse_boosted_error_p1 = classification_error(dataset01, W_min_mse_equal_boosted, 1)

    print(f"Вероятности ошибок классификатора (НСКО, улучшенный) при равных корреляционных матрицах:"
          f"\np0={min_mse_boosted_error_p0}\np1={min_mse_boosted_error_p1}")

    # разные корреляционные матрицы
    U_robbins_monro = get_U_robbins_monro(dataset10, dataset11, k)
    W_aci = get_W_aci(np.full((3, 1), 1), U_robbins_monro, k, beta, 1)
    aci_border_x = linear_border(y, W_aci)

    W_aci_boosted = get_W_aci_boosted(np.full((3, 1), 1), U_robbins_monro, k, beta, 1)
    aci_boosted_border_x = linear_border(y, W_aci_boosted)

    bayes_border_x1, _ = bayes_border_fixed(y, 0.5, 0.5, b0, b1, m0, m1)

    W_min_mse = get_W_min_mse(np.full((3, 1), 1), U_robbins_monro, k, beta, 1)
    min_mse_border_x = linear_border(y, W_min_mse)

    W_min_mse_boosted = get_W_min_mse_boosted(np.full((3, 1), 1), U_robbins_monro, k, beta, 1)
    min_mse_boosted_border_x = linear_border(y, W_min_mse_boosted)

    borders_x = [bayes_border_x1, aci_border_x, min_mse_border_x, aci_boosted_border_x, min_mse_boosted_border_x]

    plot("Процедура Роббинса-Монро, разные корреляционные матрицы", dataset10, dataset11, borders_x, borders_y, colors, labels)

    # подсчет вероятностей ошибок классификации
    bayes_error_p0 = bayes_error(dataset10, 0.5, 0.5, b0, b1, m0, m1, dataset10.shape[-1])
    bayes_error_p1 = bayes_error(dataset11, 0.5, 0.5, b1, b0, m1, m0, dataset11.shape[-1])

    print(f"Вероятности ошибок Байессовского классификатора при разных корреляционных матрицах:"
          f"\np0={bayes_error_p0}\np1={bayes_error_p1}")

    aci_error_p0 = classification_error(dataset00, W_aci, 0)
    aci_error_p1 = classification_error(dataset01, W_aci, 1)

    print(f"Вероятности ошибок классификатора (АКП, обычный) при разных корреляционных матрицах:"
          f"\np0={aci_error_p0}\np1={aci_error_p1}")

    aci_boosted_error_p0 = classification_error(dataset00, W_aci_boosted, 0)
    aci_boosted_error_p1 = classification_error(dataset01, W_aci_boosted, 1)

    print(f"Вероятности ошибок классификатора (АКП, улучшенный) при разных корреляционных матрицах:"
          f"\np0={aci_boosted_error_p0}\np1={aci_boosted_error_p1}")

    min_mse_error_p0 = classification_error(dataset00, W_min_mse, 0)
    min_mse_error_p1 = classification_error(dataset01, W_min_mse, 1)

    print(f"Вероятности ошибок классификатора (НСКО, обычный) при разных корреляционных матрицах:"
          f"\np0={min_mse_error_p0}\np1={min_mse_error_p1}")

    min_mse_boosted_error_p0 = classification_error(dataset00, W_min_mse_boosted, 0)
    min_mse_boosted_error_p1 = classification_error(dataset01, W_min_mse_boosted, 1)

    print(f"Вероятности ошибок классификатора (НСКО, улучшенный) при разных корреляционных матрицах:"
          f"\np0={min_mse_boosted_error_p0}\np1={min_mse_boosted_error_p1}")


def research(dataset00, dataset01, dataset10, dataset11, m0, m1, b0, b1, b, equal, boosted, aci):

    # на что влияет beta

    W_next = W_prev = np.ones((3, 1))

    U_length = dataset00.shape[-1] + dataset01.shape[-1]
    U_robbins_monro = []
    if equal:
        U_robbins_monro = get_U_robbins_monro(dataset00, dataset01, U_length)
    else:
        U_robbins_monro = get_U_robbins_monro(dataset10, dataset11, U_length)

    k = 50

    colors = ["red", "orange", "yellow", "green", "blue", "darkblue", "magenta", "brown", "black"]
    y = np.arange(-4, 4, 0.1)

    K = 10

    for beta in [0.51, 0.6, 0.7, 0.8, 0.9, 1.]:
        title = f"beta={beta}, {'равные корреляционные матрицы' if equal else 'разные корреляционные матрицы'}, " \
                f"{'АКП' if aci else 'НСКО'}, {'улучшенный' if boosted else 'обычный'}"

        borders_x = []
        borders_y = []
        labels = []

        for i in range(k, U_length + 1, k):
            W_next = W_prev = np.ones((3, 1))
            for j in range(K):
                if boosted:
                    if aci:
                        W_next = get_W_aci_boosted(W_prev, U_robbins_monro[0: i, :, :], i, beta, i * j + 1)
                    else:
                        W_next = get_W_min_mse_boosted(W_prev, U_robbins_monro[0: i, :, :], i, beta, i * j + 1)
                else:
                    if aci:
                        W_next = get_W_aci(W_prev, U_robbins_monro[0: i, :, :], i, beta, i * j + 1)
                    else:
                        W_next = get_W_min_mse(W_prev, U_robbins_monro[0: i, :, :], i, beta, i * j + 1)
                W_prev = W_next

                if j == K - 1:
                    borders_x.append(linear_border(y, W_next))
                    borders_y.append(y)
                    labels.append(f"k={i * j + 1}, alpha_k={int(get_alpha(i, beta) * 10000) / 10000}")

        if equal:
            borders_x.append(bayes_border_fixed(y, 0.5, 0.5, b, b, m0, m1))
        else:
            borders_x.append(bayes_border_fixed(y, 0.5, 0.5, b0, b1, m0, m1)[0])
        borders_y.append(y)
        labels.append("bayes")

        # plot(title, dataset00, dataset01, borders_x, borders_y, colors, labels)

    # show()

    # на что влияет начальное состояние

    beta = 1.

    # for value in [-10, -7, -5, -2, -1, 0, 1, 2, 5, 7, 10]:
    for value in [1]:
        W_next = W_prev = np.full((3, 1), value)
        title = f"W_0^T={W_next.T}, {'равные корреляционные матрицы' if equal else 'разные корреляционные матрицы'}, " \
                f"{'АКП' if aci else 'НСКО'}, {'улучшенный' if boosted else 'обычный'}"

        borders_x = []
        borders_y = []
        labels = []

        for i in range(k, U_length + 1, k):
            W_next = W_prev = np.full((3, 1), value)
            for j in range(K):
                if boosted:
                    if aci:
                        W_next = get_W_aci_boosted(W_prev, U_robbins_monro[0: i, :, :], i, beta, i * j + 1)
                    else:
                        W_next = get_W_min_mse_boosted(W_prev, U_robbins_monro[0: i, :, :], i, beta, i * j + 1)
                else:
                    if aci:
                        W_next = get_W_aci(W_prev, U_robbins_monro[0: i, :, :], i, beta, i * j + 1)
                    else:
                        W_next = get_W_min_mse(W_prev, U_robbins_monro[0: i, :, :], i, beta, i * j + 1)
                W_prev = W_next

                if j == K - 1:
                    borders_x.append(linear_border(y, W_next))
                    borders_y.append(y)
                    labels.append(f"k={i * j + 1}")


        if equal:
            borders_x.append(bayes_border_fixed(y, 0.5, 0.5, b, b, m0, m1))
        else:
            borders_x.append(bayes_border_fixed(y, 0.5, 0.5, b0, b1, m0, m1)[0])
        borders_y.append(y)
        labels.append("bayes")

        plot(title, dataset00, dataset01, borders_x, borders_y, colors, labels)

    show()


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

    task1(m0, m1, b0, b1, b, dataset00, dataset01, dataset10, dataset11)

    k = 400
    task2(m0, m1, b0, b1, b, dataset00, dataset01, dataset10, dataset11, k)

    k = 400
    beta = 0.55
    task3(m0, m1, b0, b1, b, dataset00, dataset01, dataset10, dataset11, k, beta)

    show()

    # equal = True
    # boosted = True
    # aci = True
    # research(dataset00, dataset01, dataset10, dataset11, m0, m1, b0, b1, b, equal, boosted, aci)


main()


