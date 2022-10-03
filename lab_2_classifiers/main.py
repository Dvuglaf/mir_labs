import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show
from lab_1_dataset_generation.parameters import get_rho
from scipy.special import erf, erfinv

'''
Минимаксный классификатор: для двух классов с одинаковыми B. Это Байесовский классификатор, для которого вычислены
априорные вероятности, исходя из условия равенства ошибок первого и второго рода.
Для простейшей матрицы штрафов (0 - главная диагональ, 1 - остальное) априорная вероятность = 1/2 дает макс. значение
Байесовского риска (худший случай). Это и нужно для классификатора (минимизируем максимальное значение риска)
'''

'''
Обозначения:
    p -     априорные вероятности 
    c -     матрица штрафов
    d -     решающая функция классификатора
    rho -   мера близости (расстояние Махаланобиса)
    p_i_j - вероятности ошибочной классификации (объект класса i отнесли к классу j)
    m -     матожидание выборки
    b -     корреляционная матрица выборки
'''


def phi_laplace(x):
    return (1 + erf(x / np.sqrt(2))) / 2


def inv_phi_laplace(x):
    return np.sqrt(2) * erfinv(2 * x-1)


# Байесовская решающая граница
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


# Решающая функция Байесовского классификатора
def bayes_discriminant(x, p, m, b):
    return np.log(p) - np.log(np.linalg.det(b)) - get_rho(m, x, b, b)


# Байесовский классификатор (попарное сравнение)
# @return 1 - первый класс, 2 - второй класс
def bayes_classificator(x, p1, p2, b1, b2, m1, m2) -> int:
    d1 = bayes_discriminant(x, p1, m1, b1)
    d2 = bayes_discriminant(x, p2, m2, b2)

    return 1 if d1 > d2 else 2


# Вероятности ошибочной классификации для Байесовского классификатора
def bayes_error(p1, p2, m1, m2, b1, b2, c):
    _lambda = (p1 * (c[0][1] - c[0][0])) / (p2 * (c[1][0] - c[1][1]))
    rho = get_rho(m1, m2, b1, b2)

    p_0_1 = 1 - phi_laplace((np.log(_lambda) + 0.5 * rho) / np.sqrt(rho))  # ошибка первого рода
    p_1_0 = phi_laplace((np.log(_lambda) - 0.5 * rho) / np.sqrt(rho))  # ошибка второго рода

    return p_0_1, p_1_0


# Поиск априорной вероятности в общем случае для минимаксного классификатора (любая матрица штрафов)
def minmax_get_p(c):
    p1 = (c[1][0] - c[1][1]) / (c[0][1] - c[0][0] + c[1][0] - c[1][1])
    return p1, 1 - p1


# Минимаксная решающая граница (частный случай Байесовского)
# @return - координата x
def minmax_border(y, b, c, m1, m2):
    p1, p2 = minmax_get_p(c)
    return bayes_border(y, p1, p2, b, b, m1, m2)


def neyman_pearson_border(y, b, m1, m2, p):
    rho = get_rho(m1, m2, b, b)

    e = 1 / (2 * np.linalg.det(b))
    a = e * ((b[0][1] + b[1][0]) * (m1[1] - m2[1]) - 2 * b[1][1] * (m1[0] - m2[0]))
    c = e * ((b[0][1] + b[1][0]) * (m1[0] - m2[0]) - 2 * b[0][0] * (m1[1] - m2[1]))
    d = e * (b[0][0] * (m1[1] ** 2 - m2[1] ** 2) - (b[0][1] + b[1][0]) * (m1[0] * m1[1] - m2[0] * m2[1]) +
             b[1][1] * (m1[0] ** 2 - m2[0] ** 2)) + 1 / 2 * rho - np.sqrt(rho) * inv_phi_laplace(1 - p)
    return -1 / a * (c * y + d)


# Байесовская решающая граница между двумя классами с равными априорными вероятностями и b1 = b2.
# Вычисление вероятностей ошибочной классификации и суммарной вероятности ошибочной классификации.
def task_1(dataset_1, dataset_2, b, c, m1, m2):
    p = 1 / 2

    y = np.arange(-4, 4, 0.1)

    x = bayes_border(y, p, p, b, b, m1, m2)
    x = np.reshape(x, y.shape)

    plt.figure()
    plt.title("Задание 1 (Байесовский классификатор p1 = p2 и b1 = b2)")
    plt.plot(dataset_1[0, :, :], dataset_1[1, :, :], color='red', marker='.')
    plt.plot(dataset_2[0, :, :], dataset_2[1, :, :], color='green', marker='+')
    plt.plot(x, y, color='black')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)

    errors = bayes_error(p, p, m1, m2, b, b, c)

    print("Задание 1 (Байесовский классификатор p1 = p2 и b1 = b2):")
    print(f"\tВероятности ошибочной классификации:")
    print(f"\t\tp_0_1 = {errors[0]}")
    print(f"\t\tp_1_0 = {errors[1]}")
    print(f"\tСуммарная вероятность ошибочной классификации: {errors[0] + errors[1]}")


def task_2(dataset_1, dataset_2, c, b, m1, m2):
    y = np.arange(-4, 4, 0.1)
    x_minmax = minmax_border(y, b, c, m1, m2)
    x_minmax = np.reshape(x_minmax, y.shape)

    fig = plt.figure()
    sb = fig.add_subplot(1, 2, 1)
    sb.set_title("Задание 2 (Минимаксный классификатор p1 = p2 и b1 = b2)")
    sb.plot(dataset_1[0, :, :], dataset_1[1, :, :], color='red', marker='.')
    sb.plot(dataset_2[0, :, :], dataset_2[1, :, :], color='green', marker='+')
    sb.plot(x_minmax, y, color='black')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)

    x_np = neyman_pearson_border(y, b, m1, m2, 0.05)
    x_np = np.reshape(x_np, y.shape)

    sb = fig.add_subplot(1, 2, 2)
    sb.set_title("Задание 2 (Нейман-Пирс классификатор p0 = 0.05 и b1 = b2)")
    sb.plot(dataset_1[0, :, :], dataset_1[1, :, :], color='red', marker='.')
    sb.plot(dataset_2[0, :, :], dataset_2[1, :, :], color='green', marker='+')
    sb.plot(x_np, y, color='black')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)


# Байесовская решающая граница между тремя классами с равными априорными вероятностями и неравными b.
# Оценка для любых двух классов вероятности ошибочной классификации и относительной погрешности полученных оценок.
# Определить объем выборки, обеспечивающую погрешность не более 5%.
def task_3(dataset_1, dataset_2, dataset_3, c, b1, b2, b3, m1, m2, m3):
    p = 1 / 3

    # Границы получены из исходных графиков, путем нахождения координат окрестности пересечения
    y13 = np.arange(-4, -0.29, 0.1)
    y12 = np.arange(-4, -0.29, 0.1)
    y23 = np.arange(-0.32, 4, 0.1)

    _, x13 = bayes_border(y13, p, p, b1, b3, m1, m3)
    x13 = np.reshape(x13, y13.shape)

    x12, _ = bayes_border(y12, p, p, b1, b2, m1, m2)
    x12 = np.reshape(x12, y12.shape)

    _, x23 = bayes_border(y23, p, p, b2, b3, m2, m3)
    x23 = np.reshape(x23, y23.shape)

    plt.figure()
    plt.title("Задание 3 (Байесовский классификатор p1 = p2 = p3 и неравных b)")
    plt.plot(dataset_1[0, :, :], dataset_1[1, :, :], color='red', marker='.')
    plt.plot(dataset_2[0, :, :], dataset_2[1, :, :], color='green', marker='+')
    plt.plot(dataset_3[0, :, :], dataset_3[1, :, :], color='blue', marker='x')
    plt.plot(x13, y13, color='cyan')
    plt.plot(x12, y12, color='magenta')
    plt.plot(x23, y23, color='yellow')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)

    errors13 = bayes_error(p, p, m1, m3, b1, b3, c)
    errors12 = bayes_error(p, p, m1, m2, b1, b2, c)
    errors23 = bayes_error(p, p, m2, m3, b2, b3, c)

    print("Задание 3 (Байесовский классификатор p1 = p2 = p3 и неравных b):")
    print(f"\tВероятности ошибочной классификации:")
    print(f"\t\tp_0_2 = {errors13[0]}")
    print(f"\t\tp_2_0 = {errors13[1]}")
    print(f"\t\tp_0_1 = {errors12[0]}")
    print(f"\t\tp_1_0 = {errors12[1]}")
    print(f"\t\tp_1_2 = {errors23[0]}")
    print(f"\t\tp_2_1 = {errors23[1]}")


def main():
    m1 = np.array(([0], [-2]))
    m2 = np.array(([-1], [1]))
    m3 = np.array(([2], [0]))

    b = np.array(([0.5, -0.2], [-0.2, 0.5]))  # случай равных корреляционных матриц

    b1 = np.array(([0.5, 0], [0, 0.5]))
    b2 = np.array(([0.4, 0.1], [0.1, 0.6]))
    b3 = np.array(([0.6, -0.2], [-0.2, 0.6]))

    c = np.array(([0, 1], [1, 0]))

    dataset_1_1 = np.load("../lab_1_dataset_generation/dataset_1_1.npy")
    dataset_1_2 = np.load("../lab_1_dataset_generation/dataset_1_2.npy")

    dataset_2_1 = np.load("../lab_1_dataset_generation/dataset_2_1.npy")
    dataset_2_2 = np.load("../lab_1_dataset_generation/dataset_2_2.npy")
    dataset_2_3 = np.load("../lab_1_dataset_generation/dataset_2_3.npy")

    task_1(dataset_1_1, dataset_1_2, b, c, m1, m2)
    task_2(dataset_1_1, dataset_1_2, c, b, m1, m2)
    task_3(dataset_2_1, dataset_2_2, dataset_2_3, c, b1, b2, b3, m1, m2, m3)
    show()


main()

# TODO: задание 3 (оценка погрешности, найти N)
