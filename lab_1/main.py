import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show

def generate_normal(mu, sigma, size):  # из равномерного в нормальное
    n = 12
    mu = 1 / 2
    sigma = np.sqrt(1 / 12)
    sum1 = np.sum(np.random.uniform(size=n))
    sum2 = np.sum(np.random.uniform(size=n))
    val1 = (sum1 - n * mu) / (sigma * np.sqrt(n))
    val2 = (sum2 - n * mu) / (sigma * np.sqrt(n))
    return np.array(([val1], [val2]))


def generate_a(b):
    a = np.zeros((2, 2))
    a[0, 0] = np.sqrt(b[0, 0])
    a[1, 0] = b[0, 1] / a[0, 0]
    a[1, 1] = np.sqrt(b[1, 1] - b[0, 1] ** 2 / b[0, 0])
    return a


def generate_x(a, e, m):
    return np.matmul(a, e) + m


def generate_dataset(a, m, size, N):
    dataset = np.ndarray((2, 1, N))
    for i in range(N):
        mu, sigma, size = 0, 1, (2, 1)
        e = generate_normal(mu, sigma, size)
        x = generate_x(a, e, m)
        dataset[:, :, i] = x
    return dataset


def get_parameters(dataset, N):
    m = np.sum(dataset, axis=2) / N
    b = 0
    for i in range(N):
        b += np.matmul((dataset[:,:,i] - m), np.transpose(dataset[:,:,i] - m))
    b /= N
    return m, b


def get_rho(m1, m2, b1, b2):
    rho = 0
    if np.array_equal(b1, b2):  # расстояние Махаланобиса
        rho = np.matmul(np.matmul(np.transpose(np.subtract(m2, m1)), np.linalg.inv(b1)), np.subtract(m2, m1))
    else:  # расстояние Бхатачария
        rho = (1 / 4) * np.matmul(np.matmul(np.transpose(np.subtract(m2, m1)), np.linalg.inv((b1 + b2) / 2)), np.subtract(m2, m1))
        + (1 / 2) * np.log(np.linalg.det((b1 + b2) / 2) / np.sqrt(np.linalg.det(b1) * np.linalg.det(b2)))
    return rho


def main():
    N = 200
    size = (2, 1)
    m1 = [[0], [-2]]
    m2 = [[-1], [1]]
    m3 = [[2], [0]]

    # два вектора с равными корреляционными матрицами
    b = np.array(([0.5, -0.2], [-0.2, 0.5]))
    a = generate_a(b)

    dataset1 = generate_dataset(a, m1, size, N)
    dataset2 = generate_dataset(a, m2, size, N)

    np.save("task_1_dataset_1", dataset1)
    np.save("task_1_dataset_2", dataset2)

    m1_exp, b1_exp = get_parameters(dataset1, N)
    m2_exp, b2_exp = get_parameters(dataset2, N)
    print(f'Матожидание 1й выборки:\n{m1_exp}\nКорреляционная матрица 1й выборки:\n{b1_exp}')
    print(f'Матожидание 2й выборки:\n{m2_exp}\nКорреляционная матрица 2й выборки:\n{b2_exp}')
    rho = get_rho(m1, m2, b, b)
    print(f'Мера близости распределений (расстояние Махаланобиса):\n{rho}')

    fig = plt.figure()
    plt.plot(dataset1[0, :, :], dataset1[1, :, :], color='red', marker = '.')
    plt.plot(dataset2[0, :, :], dataset2[1, :, :], color='green', marker = '+')
    show()

    print('\n')

    # три вектора с разными корреляционными матрицами
    b1 = np.array(([0.5, 0], [0, 0.5]))
    b2 = np.array(([0.4, 0.1], [0.1, 0.6]))
    b3 = np.array(([0.6, -0.2], [-0.2, 0.6]))

    a1 = generate_a(b1)
    a2 = generate_a(b2)
    a3 = generate_a(b3)

    dataset1 = generate_dataset(a1, m1, size, N)
    dataset2 = generate_dataset(a2, m2, size, N)
    dataset3 = generate_dataset(a3, m3, size, N)

    np.save("task_2_dataset_1", dataset1)
    np.save("task_2_dataset_2", dataset2)
    np.save("task_2_dataset_3", dataset3)

    m1_exp, b1_exp = get_parameters(dataset1, N)
    m2_exp, b2_exp = get_parameters(dataset2, N)
    m3_exp, b3_exp = get_parameters(dataset3, N)
    print(f'Матожидание 1й выборки:\n{m1_exp}\nКорреляционная матрица 1й выборки:\n{b1_exp}')
    print(f'Матожидание 2й выборки:\n{m2_exp}\nКорреляционная матрица 2й выборки:\n{b2_exp}')
    print(f'Матожидание 3й выборки:\n{m3_exp}\nКорреляционная матрица 3й выборки:\n{b3_exp}')
    rho12 = get_rho(m1, m2, b1, b2)
    print(f'Мера близости распределений 1 и 2 (расстояние Бхатачария):\n{rho12}')
    rho13 = get_rho(m1, m3, b1, b3)
    print(f'Мера близости распределений 1 и 3 (расстояние Бхатачария):\n{rho13}')
    rho23 = get_rho(m2, m3, b2, b3)
    print(f'Мера близости распределений 2 и 3 (расстояние Бхатачария):\n{rho23}')

    fig = plt.figure()
    plt.plot(dataset1[0, :, :], dataset1[1, :, :], color='red', marker='.')
    plt.plot(dataset2[0, :, :], dataset2[1, :, :], color='green', marker='+')
    plt.plot(dataset3[0, :, :], dataset3[1, :, :], color='blue', marker='x')
    show()

main()