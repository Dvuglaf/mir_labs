import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show, imshow
from scipy.special import erf
import warnings


warnings.filterwarnings("ignore")


def phi_laplace(x) -> float:
    return (1 + erf(x / np.sqrt(2))) / 2


def generate_dataset(symbol: np.array, N: int, p: float) -> np.array:
    dataset = []

    for i in range(N):
        random_matrix = np.random.uniform(size=symbol.shape)
        inverted = np.where(random_matrix <= p, 1 - symbol, symbol)
        dataset.append(inverted.reshape(symbol.shape[0] * symbol.shape[1]))

    return np.array(dataset)


def get_probability(dataset: np.array) -> float:
    ones_count = np.sum(dataset, axis=0)
    return ones_count / dataset.shape[0]


def get_w(prob_1: np.array, prob_2: np.array) -> np.array:
    return np.log(prob_1 * (1 - prob_2) / ((1 - prob_1) * prob_2))


def get_lambda(x: np.array, prob_1: np.array, prob_2: np.array) -> np.array:
    return np.sum(x * get_w(prob_1, prob_2))


def get_threshold(
    aprior_1: float, aprior_2: float, prob_1: float, prob_2: float
) -> np.array:
    return np.log(aprior_2 / aprior_1) + \
        np.sum(np.log((1 - prob_2) / (1 - prob_1)))


def bayes_classificator(
    x: np.array, aprior_1: float, aprior_2: float, prob_1: float, prob_2: float
) -> int:
    _Lambda = get_lambda(x, prob_1, prob_2)
    _lambda = get_threshold(aprior_1, aprior_2, prob_1, prob_2)

    return 0 if _Lambda >= _lambda else 1


def get_errors(
    aprior_1: float, aprior_2: float, prob_1: float, prob_2: float
) -> list:
    base = np.log(prob_2 * (1 - prob_1) / ((1 - prob_2) * prob_1))

    m_1 = np.sum(base * prob_1)
    m_2 = np.sum(base * prob_2)

    sigma_1 = np.sqrt(np.sum((base ** 2) * prob_1 * (1 - prob_1)))
    sigma_2 = np.sqrt(np.sum((base ** 2) * prob_2 * (1 - prob_2)))

    _lambda = np.log(aprior_1 / aprior_2)

    errors = [1 - phi_laplace((_lambda - m_1) / sigma_1),
              phi_laplace((_lambda - m_2) / sigma_2)]
    upper_bounds = [sigma_1 ** 2 / (((m_1) - _lambda) ** 2),
                    sigma_2 ** 2 / (((m_2) - _lambda) ** 2)]

    return errors, upper_bounds


# Подсчет ошибки
def classification_error(
    dataset: np.array, aprior_1: float, aprior_2: float,
    prob_1: float, prob_2: float, N: int
):
    errors = 0  # показывает число неверно определенных элементов
    indices_wrong_classified = []

    for i in range(N):
        result = bayes_classificator(dataset[i], aprior_1, aprior_2, prob_1, prob_2)
        errors += result
        if result == 1:
            indices_wrong_classified.append(i)

    return errors / N, np.array(indices_wrong_classified)


# Подсчет относительной погрешности
def get_e(p: float, N: int) -> float:
    try:
        return np.sqrt((1 - p) / (N * p))
    except ZeroDivisionError:
        return 1.


def get_right_classified(dataset: np.array, ind_wrong_classified: np.array):
    for i in range(dataset.shape[0]):
        if i not in ind_wrong_classified:
            return dataset[i].reshape((9, 9))


def main():
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

    N = 200

    dataset_u = generate_dataset(symbol_u, N, 0.3)
    dataset_z = generate_dataset(symbol_z, N, 0.3)

    aprior_u = aprior_z = 0.5

    prob_u = get_probability(dataset_u)
    prob_z = get_probability(dataset_z)

    errors = get_errors(aprior_u, aprior_z, prob_u, prob_z)

    exp_error_u, wrong_classified_u = classification_error(dataset_u, aprior_u, aprior_z, prob_u, prob_z, N)
    exp_error_z, wrong_classified_z = classification_error(dataset_z, aprior_z, aprior_u, prob_z, prob_u, N)

    print(f"Аналитические ошибки классификации: {errors[0][0]}, {errors[0][1]}")
    print(f"Экспериментальные ошибки классификации: {exp_error_u}, {exp_error_z}")

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

    plt.figure()
    plt.title("Распределение вероятностей (X=1/OMEGA)")
    plt.plot(range(0, 81), prob_u, color='pink', label='(X=1/U)')
    plt.plot(range(0, 81), prob_z, color='blue', label='(X=1/Z)')
    plt.legend()

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.title("Распределение вероятностей (X=1/U)")
    imshow(prob_u.reshape((9, 9)), cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.title("Распределение вероятностей (X=1/Z)")
    imshow(prob_z.reshape((9, 9)), cmap='gray')

    if wrong_classified_u.size != 0:
        plt.figure()
        plt.title("Неверно классифицированный объект класса U")
        imshow(1 - dataset_u[wrong_classified_u[0]].reshape((9, 9)), cmap='gray')

    if wrong_classified_z.size != 0:
        plt.figure()
        plt.title("Неверно классифицированный объект класса Z")
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

    show()


main()
