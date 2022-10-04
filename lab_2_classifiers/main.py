import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show
from lab_1_dataset_generation.parameters import get_rho
from lab_1_dataset_generation.main import generate_dataset
from scipy.special import erf, erfinv
import warnings


warnings.filterwarnings("ignore")  # runtime warning disable

'''
Минимаксный классификатор: для двух классов с одинаковыми B. Это Байесовский классификатор, для которого вычислены
априорные вероятности, исходя из условия равенства ошибок первого и второго рода.
Для простейшей матрицы штрафов (0 - главная диагональ, 1 - остальное) априорная вероятность = 1/2 дает макс. значение
Байесовского риска (худший случай). Это и нужно для классификатора (минимизируем максимальное значение риска).
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


# Байесовская решающая граница для двух классов
def bayes_border(y, p0, p1, b0, b1, m0, m1):
    g = np.linalg.det(b0) * np.linalg.det(b1)
    d = 1 / g * (np.linalg.det(b1) * (m0[0] * b0[1][1] - m0[1] * b0[1][0])
                 - np.linalg.det(b0) * (m1[0] * b1[1][1] - m1[1] * b1[1][0]))
    e = 1 / g * (np.linalg.det(b1) * (m0[1] * b0[0][0] - m0[0] * b0[0][1])
                 - np.linalg.det(b0) * (m1[1] * b1[0][0] - m1[0] * b1[0][1]))
    f = np.log(np.linalg.det(b0) / np.linalg.det(b1)) + 2 * np.log(p0 / p1) \
        - np.matmul(np.matmul(np.transpose(m0), np.linalg.inv(b0)), m0) \
        + np.matmul(np.matmul(np.transpose(m1), np.linalg.inv(b1)), m1)

    if not np.array_equal(b0, b1):
        a = 1 / g * (np.linalg.det(b0) * b1[1][1] - np.linalg.det(b1) * b0[1][1])
        b = 1 / g * (np.linalg.det(b1) * (b0[1][0] + b0[0][1]) - np.linalg.det(b0) * (b1[1][0] + b1[0][1]))
        c = 1 / g * (np.linalg.det(b0) * b1[0][0] - np.linalg.det(b1) * b0[0][0])

        x0 = ((-b * y - d) + np.sqrt((b * y + d) ** 2 - 4 * a * (c * y ** 2 + f + e * y))) / (2 * a)
        x1 = ((-b * y - d) - np.sqrt((b * y + d) ** 2 - 4 * a * (c * y ** 2 + f + e * y))) / (2 * a)

        return x0, x1
    else:
        x = -1 / d * (e * y + f)
        return x


# Решающая функция Байесовского классификатора
def bayes_discriminant(x, p, m, b) -> float:
    return np.log(p) - np.log(np.linalg.det(b)) - get_rho(m, x, b, b)  # расстояние Махаланобиса


# Байесовский классификатор (попарное сравнение)
# @return 0 - первый класс, 1 - второй класс
def bayes_classificator(x, p0, p1, b0, b1, m0, m1) -> int:
    d0 = bayes_discriminant(x, p0, m0, b0)
    d1 = bayes_discriminant(x, p1, m1, b1)

    return 0 if d0 > d1 else 1


# Вероятности ошибочной классификации для Байесовского классификатора
def bayes_error(p0, p1, m0, m1, b0, b1, c) -> (float, float):
    _lambda = (p0 * (c[0][1] - c[0][0])) / (p1 * (c[1][0] - c[1][1]))
    rho = get_rho(m0, m1, b0, b1)

    p_0_1 = 1 - phi_laplace((np.log(_lambda) + 0.5 * rho) / np.sqrt(rho))  # ошибка первого рода
    p_1_0 = phi_laplace((np.log(_lambda) - 0.5 * rho) / np.sqrt(rho))  # ошибка второго рода

    return p_0_1, p_1_0


# Поиск априорной вероятности в общем случае для минимаксного классификатора (любая матрица штрафов)
def minmax_get_p(c) -> (float, float):
    p0 = (c[1][0] - c[1][1]) / (c[0][1] - c[0][0] + c[1][0] - c[1][1])
    return p0, 1 - p0


# Минимаксная решающая граница (частный случай Байесовского)
# @return - координата x
def minmax_border(y, b, c, m0, m1):
    p0, p1 = minmax_get_p(c)
    return bayes_border(y, p0, p1, b, b, m0, m1)


# Решающая граница Неймана-Пирсона
def neyman_pearson_border(y, b, m0, m1, p):
    rho = get_rho(m0, m1, b, b)

    e = 1 / (2 * np.linalg.det(b))
    a = e * ((b[0][1] + b[1][0]) * (m0[1] - m1[1]) - 2 * b[1][1] * (m0[0] - m1[0]))
    c = e * ((b[0][1] + b[1][0]) * (m0[0] - m1[0]) - 2 * b[0][0] * (m0[1] - m1[1]))
    d = e * (b[0][0] * (m0[1] ** 2 - m1[1] ** 2) - (b[0][1] + b[1][0]) * (m0[0] * m0[1] - m1[0] * m1[1]) +
             b[1][1] * (m0[0] ** 2 - m1[0] ** 2)) + 1 / 2 * rho - np.sqrt(rho) * inv_phi_laplace(1 - p)

    return -1 / a * (c * y + d)


# Подсчет ошибки первого рода (p_0)
def classification_error(dataset, p0, p1, b0, b1, m0, m1, N) -> float:
    errors = 0  # показывает число неверно определенных элементов

    for i in range(N):
        errors += bayes_classificator(dataset[:, :, i], p0, p1, b0, b1, m0, m1)

    return errors / N  # ошибка первого рода


# Подсчет относительной погрешности
# @param p - ошибка первого рода
def get_e(p, N) -> float:
    try:
        return np.sqrt((1 - p) / (N * p))
    except ZeroDivisionError:
        return 1.


# Определение объема выборки для относительной погрешности < max_e
def get_N(p0, p1, b0, b1, m0, m1, max_e=0.05) -> int:
    N = 1000  # начальное значение
    step = 1000

    while True:
        dataset = generate_dataset(b0, m0, N)
        p = classification_error(dataset, p0, p1, b0, b1, m0, m1, N)
        e = get_e(p, N)
        if e < max_e:
            return N
        N += step


# Байесовская решающая граница между двумя классами с равными априорными вероятностями и b0 = b1.
# Вычисление вероятностей ошибочной классификации и суммарной вероятности ошибочной классификации.
def task_1(dataset_1, dataset_2, b, c, m0, m1):
    p = 1 / 2

    y = np.arange(-4, 4, 0.1)

    x = bayes_border(y, p, p, b, b, m0, m1)
    x = np.reshape(x, y.shape)

    plt.figure()
    plt.title("Задание 1 (Байесовский классификатор p0 = p1 и b0 = b1)")
    plt.plot(dataset_1[0, :, :], dataset_1[1, :, :], color='red', marker='.')
    plt.plot(dataset_2[0, :, :], dataset_2[1, :, :], color='green', marker='+')
    plt.plot(x, y, color='black')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)

    errors = bayes_error(p, p, m0, m1, b, b, c)

    print("Задание 1 (Байесовский классификатор p0 = p1 и b0 = b1):")
    print(f"\tВероятности ошибочной классификации:")
    print(f"\t\tp_0_1 = {errors[0]}")
    print(f"\t\tp_1_0 = {errors[1]}")
    print(f"\tСуммарная вероятность ошибочной классификации: {errors[0] + errors[1]}\n")


# Минимаксный и классификатор Неймана-Пирсона для p_0 = 0.05 (2 класса с b0 = b1).
# Изобразить решающие границы.
def task_2(dataset_1, dataset_2, c, b, m0, m1):
    y = np.arange(-4, 4, 0.1)
    x_minmax = minmax_border(y, b, c, m0, m1)
    x_minmax = np.reshape(x_minmax, y.shape)

    fig = plt.figure()
    sb = fig.add_subplot(1, 2, 1)
    sb.set_title("Задание 2 (Минимаксный классификатор p0 = p1 и b0 = b1)")
    sb.plot(dataset_1[0, :, :], dataset_1[1, :, :], color='red', marker='.')
    sb.plot(dataset_2[0, :, :], dataset_2[1, :, :], color='green', marker='+')
    sb.plot(x_minmax, y, color='black')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)

    x_np = neyman_pearson_border(y, b, m0, m1, 0.05)
    x_np = np.reshape(x_np, y.shape)

    sb = fig.add_subplot(1, 2, 2)
    sb.set_title("Задание 2 (Нейман-Пирс классификатор p0 = 0.05 и b0 = b1)")
    sb.plot(dataset_1[0, :, :], dataset_1[1, :, :], color='red', marker='.')
    sb.plot(dataset_2[0, :, :], dataset_2[1, :, :], color='green', marker='+')
    sb.plot(x_np, y, color='black')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)


# Байесовская решающая граница между тремя классами с равными априорными вероятностями и неравными b.
# Оценка для любых двух классов вероятности ошибочной классификации и относительной погрешности полученных оценок.
# Определить объем выборки, обеспечивающую погрешность не более 5%.
def task_3(dataset_1, dataset_2, dataset_3, b0, b1, b2, m0, m1, m2, N):
    p = 1 / 3

    # Границы получены из исходных графиков, путем нахождения координат окрестности пересечения
    y13 = np.arange(-4, -0.29, 0.1)
    y12 = np.arange(-4, -0.29, 0.1)
    y23 = np.arange(-0.32, 4, 0.1)

    _, x02 = bayes_border(y13, p, p, b0, b2, m0, m2)
    x02 = np.reshape(x02, y13.shape)

    x01, _ = bayes_border(y12, p, p, b0, b1, m0, m1)
    x01 = np.reshape(x01, y12.shape)

    _, x12 = bayes_border(y23, p, p, b1, b2, m1, m2)
    x12 = np.reshape(x12, y23.shape)

    plt.figure()
    plt.title("Задание 3 (Байесовский классификатор p0 = p1 = p2 и неравных b)")
    plt.plot(dataset_1[0, :, :], dataset_1[1, :, :], color='red', marker='.')
    plt.plot(dataset_2[0, :, :], dataset_2[1, :, :], color='green', marker='+')
    plt.plot(dataset_3[0, :, :], dataset_3[1, :, :], color='blue', marker='x')
    plt.plot(x02, y13, color='cyan')
    plt.plot(x01, y12, color='magenta')
    plt.plot(x12, y23, color='yellow')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)

    # Относительные погрешности оценки вероятности
    print("Задание 3 (Байесовский классификатор p0 = p1 = p2 и неравных b):")
    p02 = classification_error(dataset_1, p, p, b0, b2, m0, m2, N)
    e_error02 = get_e(p02, N)
    print(f"\tОценка вероятности ошибочной классификации и погрешность для классов 0 и 2:")
    print(f"\t\tp_0_2 = {p02}")
    print(f"\t\te_0_2 = {e_error02}")
    if e_error02 > 0.05:
        print("\tНеобходимо увеличить объем выборки для уменьшения относительной погрешности")
        new_N = 40000
        new_dataset = generate_dataset(b0, m0, new_N)
        p02 = classification_error(new_dataset, p, p, b0, b2, m0, m2, new_N)
        e_error02 = get_e(p02, new_N)
        print(f"\tПри N = {new_N}:")
        print(f"\t\tp_0_2 = {p02}")
        print(f"\t\te_0_2 = {e_error02}")


def main():
    N = 200

    m0 = np.array(([0], [-2]))
    m1 = np.array(([-1], [1]))
    m2 = np.array(([2], [0]))

    b = np.array(([0.5, -0.2], [-0.2, 0.5]))  # случай равных корреляционных матриц
    b0 = np.array(([0.5, 0], [0, 0.5]))
    b1 = np.array(([0.4, 0.1], [0.1, 0.6]))
    b2 = np.array(([0.6, -0.2], [-0.2, 0.6]))

    c = np.array(([0, 1], [1, 0]))  # простейшая матрица штрафов

    dataset_1_1 = np.load("../lab_1_dataset_generation/dataset_1_1.npy")
    dataset_1_2 = np.load("../lab_1_dataset_generation/dataset_1_2.npy")

    dataset_2_1 = np.load("../lab_1_dataset_generation/dataset_2_1.npy")
    dataset_2_2 = np.load("../lab_1_dataset_generation/dataset_2_2.npy")
    dataset_2_3 = np.load("../lab_1_dataset_generation/dataset_2_3.npy")

    task_1(dataset_1_1, dataset_1_2, b, c, m0, m1)
    task_2(dataset_1_1, dataset_1_2, c, b, m0, m1)
    task_3(dataset_2_1, dataset_2_2, dataset_2_3, b0, b1, b2, m0, m1, m2, N)
    show()


main()
