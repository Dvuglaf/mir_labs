import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show, imshow
from scipy.special import erf
import warnings


warnings.filterwarnings("ignore")


def phi_laplace(x) -> float:
    return (1 + erf(x / np.sqrt(2))) / 2


# Генерируется случайный вектор N со значениями [0., 1.].
# Если значение вектора меньше вероятности p, то соответствующее значение представителя класса инвертируется.
def generate_dataset(symbol: np.array, N: int, p: float) -> np.array:
    dataset = []

    for i in range(N):
        random_matrix = np.random.uniform(size=symbol.shape)
        inverted = np.where(random_matrix <= p, 1 - symbol, symbol)
        dataset.append(inverted.reshape(symbol.shape[0] * symbol.shape[1]))

    return np.array(dataset)


# Подсчет распределения вероятностей.
# Считается количество единиц по всем элементам выборки на соответствующих местах и делится на объем выборки.
def get_probability(dataset: np.array) -> np.array:
    ones_count = np.sum(dataset, axis=0)
    return ones_count / dataset.shape[0]


def get_w(prob_0: np.array, prob_1: np.array) -> np.array:
    return np.log(prob_0 * (1 - prob_1) / ((1 - prob_0) * prob_1))


# Отношение правдоподобия.
def get_lambda(x: np.array, prob_0: np.array, prob_1: np.array) -> np.array:
    return np.sum(x * get_w(prob_0, prob_1))


# Пороговое значение для отношения правдоподобия (ln лямбды маленькой).
def get_threshold(
    aprior_0: float, aprior_1: float, prob_0: float, prob_1: float
) -> np.array:
    return np.log(aprior_1 / aprior_0) + np.sum(np.log((1 - prob_1) / (1 - prob_0)))


# Байесовский классификатор по отношению правдоподобия.
def bayes_classificator(
    x: np.array, aprior_0: float, aprior_1: float, prob_0: float, prob_1: float
) -> int:
    _Lambda = get_lambda(x, prob_0, prob_1)
    _lambda = get_threshold(aprior_0, aprior_1, prob_0, prob_1)

    return 0 if _Lambda >= _lambda else 1


# Аналитические вероятности ошибочной классификации (p_0 и p_1).
# @return список вероятностей при выполнении ЦПТ и не выполнении ЦПТ (верхние границы).
def get_errors(
    aprior_0: float, aprior_1: float, prob_0: float, prob_1: float
) -> list:
    w_1_0 = np.log(prob_1 * (1 - prob_0) / ((1 - prob_1) * prob_0))

    m_0 = np.sum(w_1_0 * prob_0)
    m_1 = np.sum(w_1_0 * prob_1)

    sigma_0 = np.sqrt(np.sum((w_1_0 ** 2) * prob_0 * (1 - prob_0)))
    sigma_1 = np.sqrt(np.sum((w_1_0 ** 2) * prob_1 * (1 - prob_1)))

    _lambda = np.log(aprior_0 / aprior_1) + np.sum(np.log((1 - prob_0) / (1 - prob_1)))

    errors = [1 - phi_laplace((_lambda - m_0) / sigma_0),
              phi_laplace((_lambda - m_1) / sigma_1)]
    upper_bounds = [sigma_0 ** 2 / ((m_0 - _lambda) ** 2),
                    sigma_1 ** 2 / ((m_1 - _lambda) ** 2)]

    return [errors, upper_bounds]


# Экспериментальные ошибки классификации (где 1, там ошибочная классификация).
# @return ошибки классификации, индексы неверно классифицированных объектов.
def classification_error(
    dataset: np.array, aprior_0: float, aprior_1: float,
    prob_0: float, prob_1: float, N: int
) -> tuple[float, np.array]:
    num_errors = 0
    indices_wrong_classified = []

    for i in range(N):
        result = bayes_classificator(dataset[i], aprior_0, aprior_1, prob_0, prob_1)
        num_errors += result
        if result == 1:
            indices_wrong_classified.append(i)

    return num_errors / N, np.array(indices_wrong_classified)


# Получение верно распознанных объектов.
def get_right_classified(dataset: np.array, ind_wrong_classified: np.array) -> np.array:
    for i in range(dataset.shape[0]):
        if i not in ind_wrong_classified:
            return dataset[i].reshape((9, 9))


if __name__ == "__main__":
    # Представители класса
    symbol_u = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0]])

    symbol_z = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0]])
    N = 200  # объем выборки
    aprior_u = aprior_z = 0.5  # априорные вероятности
    p = 0.3  # вероятность инвертировать бит

    dataset_u = generate_dataset(symbol_u, N, p)
    dataset_z = generate_dataset(symbol_z, N, p)

    analytic_prob_u = np.array([[0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3],
                                [0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3],
                                [0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3],
                                [0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3],
                                [0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3],
                                [0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3],
                                [0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3],
                                [0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3],
                                [0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3]])
    analytic_prob_z = np.array([[0.3, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.3],
                                [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3],
                                [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3],
                                [0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3],
                                [0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.3],
                                [0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3],
                                [0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                                [0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                                [0.3, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.3]])

    prob_u = get_probability(dataset_u)  # закон распределения P(x = 1/u)
    prob_z = get_probability(dataset_z)  # закон распределения P(x = 1/z)

    errors = get_errors(aprior_u, aprior_z, prob_u, prob_z)

    exp_error_u, wrong_classified_u = classification_error(dataset_u, aprior_u, aprior_z, prob_u, prob_z, N)
    exp_error_z, wrong_classified_z = classification_error(dataset_z, aprior_z, aprior_u, prob_z, prob_u, N)

    print(f"Аналитические ошибки классификации: {errors[0][0]}, {errors[0][1]}")
    print(f"Экспериментальные ошибки классификации: {exp_error_u}, {exp_error_z}")
    print(f"Число неверно классифицированных объектов класса U: {len(wrong_classified_u)}")
    print(f"Число неверно классифицированных объектов класса Z: {len(wrong_classified_z)}")

    fig = plt.figure()
    sp = fig.add_subplot(2, 2, 1)
    sp.set_title("Класс U")
    imshow(1 - symbol_u, cmap='gray')
    sp = fig.add_subplot(2, 2, 2)
    sp.set_title("Класс Z")
    imshow(1 - symbol_z, cmap='gray')
    sp = fig.add_subplot(2, 2, 3)
    sp.set_title("Верно классифицированный в класс U")
    imshow(1 - get_right_classified(dataset_u, wrong_classified_u), cmap='gray')
    sp = fig.add_subplot(2, 2, 4)
    sp.set_title("Верно классифицированный в класс Z")
    imshow(1 - get_right_classified(dataset_z, wrong_classified_z), cmap='gray')

    """
    plt.figure()
    plt.title("Распределение вероятностей (X=1/OMEGA)")
    plt.plot(range(0, 81), prob_u, color='pink', label='(X=1/U)')
    plt.plot(range(0, 81), prob_z, color='blue', label='(X=1/Z)')
    plt.legend()
    """

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.title("Распределение вероятностей (X=1/U)")
    imshow(prob_u.reshape((9, 9)), cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.title("Распределение вероятностей (X=1/Z)")
    imshow(prob_z.reshape((9, 9)), cmap='gray')

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.title("Неверно классифицированный объект класса U")
    if wrong_classified_u.size != 0:
        imshow(1 - dataset_u[wrong_classified_u[0]].reshape((9, 9)), cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.title("Неверно классифицированный объект класса Z")
    if wrong_classified_z.size != 0:
        imshow(1 - dataset_u[wrong_classified_z[0]].reshape((9, 9)), cmap='gray')

    w_u_z = get_w(prob_u, prob_z)
    w_z_u = get_w(prob_z, prob_u)

    fig = plt.figure()
    sp = fig.add_subplot(1, 2, 1)
    sp.set_title("Компоненты вектора w_u_z")
    imshow(w_u_z.reshape((9, 9)), cmap='gray')
    sp = fig.add_subplot(1, 2, 2)
    sp.set_title("Компоненты вектора w_z_u")
    imshow(w_z_u.reshape((9, 9)), cmap='gray')

    w_1_0 = np.log(prob_z * (1 - prob_u) / ((1 - prob_z) * prob_u))

    m_0 = np.sum(w_1_0 * prob_u)
    m_1 = np.sum(w_1_0 * prob_z)

    sigma_0 = np.sqrt(np.sum((w_1_0 ** 2) * prob_u * (1 - prob_u)))
    sigma_1 = np.sqrt(np.sum((w_1_0 ** 2) * prob_z * (1 - prob_z)))

    _Lambda_u = np.random.normal(m_0, sigma_0 ** 2, 5000)
    _Lambda_z = np.random.normal(m_1, sigma_1 ** 2, 5000)

    hist_u, edges_u = np.histogram(_Lambda_u, bins=15)
    hist_z, edges_z = np.histogram(_Lambda_z, bins=15)

    threshold = np.ones((6, )) * get_threshold(aprior_u, aprior_z, prob_u, prob_z)

    plt.figure()
    plt.title("")
    plt.plot(edges_u[:-1], hist_u / 5000, color='pink', linestyle='-', label='Lambda_u')
    plt.plot(edges_z[:-1], hist_z / 5000, color='blue', linestyle='-', label='Lambda_z')
    plt.plot(threshold, np.arange(0, 0.3, 0.05), color='red', linestyle='-')
    plt.legend()

    show()

