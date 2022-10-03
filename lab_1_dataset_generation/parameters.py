import numpy as np


# Оценка параметров выборки
def evaluate_parameters(dataset, N):
    m = np.sum(dataset, axis=2) / N
    b = 0
    for i in range(N):
        b += np.matmul((dataset[:, :, i] - m), np.transpose(dataset[:, :, i] - m))
    b /= (N - 1)
    return m, b


# Мера близости распределений
def get_rho(m1, m2, b1, b2) -> float:
    rho = 0
    if np.array_equal(b1, b2):  # расстояние Махаланобиса
        rho = np.matmul(np.matmul(np.transpose(np.subtract(m2, m1)), np.linalg.inv(b1)), np.subtract(m2, m1))
    else:  # расстояние Бхатачария
        rho = (1 / 4) * np.matmul(np.matmul(np.transpose(np.subtract(m2, m1)), np.linalg.inv((b1 + b2) / 2)),
                                  np.subtract(m2, m1))
        + (1 / 2) * np.log(np.linalg.det((b1 + b2) / 2) / np.sqrt(np.linalg.det(b1) * np.linalg.det(b2)))
    return rho[0][0]  # to number
