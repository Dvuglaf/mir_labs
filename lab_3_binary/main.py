import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show, imshow
from scipy.special import erf


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


def get_lambda(x: np.array, prob_1: np.array, prob_2: np.array) -> np.array:
    w = np.log(prob_1 * (1 - prob_2) /
               ((1 - prob_1) * prob_2))

    return np.sum(x * w)


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
) -> float:
    errors = 0  # показывает число неверно определенных элементов

    for x in dataset:
        errors += bayes_classificator(x, aprior_1, aprior_2, prob_1, prob_2)

    return errors / N  # ошибка первого рода


# Подсчет относительной погрешности
def get_e(p: float, N: int) -> float:
    try:
        return np.sqrt((1 - p) / (N * p))
    except ZeroDivisionError:
        return 1.


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

    experimental_errors = [classification_error(dataset_u, aprior_u, aprior_z, prob_u, prob_z, N),
                           classification_error(dataset_z, aprior_z, aprior_u, prob_z, prob_u, N)]

    print(f"{errors[0][0]}, {errors[0][1]}")
    print(f"{experimental_errors[0]}, {experimental_errors[1]}")

    plt.figure()
    plt.plot(range(0, 81), prob_u, color='pink', label='probability U')
    plt.plot(range(0, 81), prob_z, color='blue', label='probability Z')
    plt.legend()

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.title("Распределение вероятностей (X = 1/U)")
    imshow(prob_u.reshape((9, 9)), cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.title("Распределение вероятностей (X = 1/Z)")
    imshow(prob_z.reshape((9, 9)), cmap='gray')
    show()

main()
