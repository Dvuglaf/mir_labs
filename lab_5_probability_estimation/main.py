import numpy as np
from lab_1_dataset_generation.main import generate_dataset, evaluate_parameters


def Kernel_Density_Estimation_Normal(x, dataset):
    N = dataset.shape[-1]
    m, B = evaluate_parameters(dataset, N)
    k = 1 / 4
    h = np.float_power(N, -k / 2)
    B_parzen = (1 + h ** 2) * B
    result = 0
    for i in range(N):
        result += 1 / (2 * np.pi) * (h ** -2) / np.sqrt(np.linalg.det(B_parzen)) * np.exp(
            -0.5 * (h ** -2) * np.matmul(np.matmul((x - dataset[:, :, i]).T, np.linalg.inv(B_parzen)),
                                         (x - dataset[:, :, i])))
    return result[0] / N


def classificator_parzen(x, datasets):
    N_values = []
    for dataset in datasets:
        N_values.append(dataset.shape[-1])

    P_values = []
    for dataset in datasets:
        P_values.append(dataset.shape[-1] / np.sum(N_values))
    P_values = np.array(P_values)

    density_values = []
    for dataset in datasets:
        density_values.append(Kernel_Density_Estimation_Normal(x, dataset))
    density_values = np.array(density_values)[:, 0]

    result = P_values * density_values

    return np.argmax(result)


def classificator_knn(x, datasets, K):
    distances = []
    class_label = 0

    for dataset in datasets:
        for j in range(dataset.shape[-1]):
            distances.append(np.array([np.sum((x - dataset[:, :, j]) * (x - dataset[:, :, j])), class_label]))

        class_label += 1

    distances = np.array(distances)

    nearest_neighbors = distances[distances[:, 0].argsort()][:K, :]

    # количество соседей из каждого класса
    count_neighbors = []
    for i in range(len(datasets)):
        count_neighbors.append(len(nearest_neighbors[nearest_neighbors[:, 1] == i]))

    return np.argmax(count_neighbors)


def main():
    m0 = np.array(([0], [-2]))
    m1 = np.array(([-1], [1]))
    m2 = np.array(([2], [0]))

    b = np.array(([0.5, -0.2], [-0.2, 0.5]))  # случай равных корреляционных матриц

    b0 = np.array(([0.5, 0], [0, 0.5]))
    b1 = np.array(([0.4, 0.1], [0.1, 0.6]))
    b2 = np.array(([0.6, -0.2], [-0.2, 0.6]))

    train_datasets = [generate_dataset(b, m0, 50), generate_dataset(b, m1, 50)]
    test_datasets = [generate_dataset(b, m0, 100), generate_dataset(b, m1, 100)]

    # обобщить для двух классификаторов, сделать вывод

    k = 5

    test_results = []
    for test_dataset in test_datasets:
        test_result = []

        for j in range(test_dataset.shape[-1]):
            test_result.append(classificator_knn(test_dataset[:, :, j], train_datasets, k))

        test_results.append(np.array(test_result))

    p0 = np.sum([test_results[0] != 0]) / len(test_results[0])
    p1 = np.sum([test_results[1] != 1]) / len(test_results[1])

    N0 = test_datasets[0].shape[-1]
    N1 = test_datasets[1].shape[-1]
    P0 = N0 / (N0 + N1)
    P1 = N1 / (N0 + N1)

    risk_estimation = P0 * p0 + P1 * p1

    print(f"Вероятность ошибки первого рода p0: {p0}")
    print(f"Вероятность ошибки второго рода p1: {p1}")
    print(f"Эмпирический риск: {risk_estimation}")

    train_datasets = [generate_dataset(b0, m0, 50), generate_dataset(b1, m1, 50), generate_dataset(b2, m2, 50)]
    test_datasets = [generate_dataset(b0, m0, 100), generate_dataset(b1, m1, 100), generate_dataset(b2, m2, 100)]

    test_results = []
    for test_dataset in test_datasets:
        test_result = []

        for j in range(test_dataset.shape[-1]):
            test_result.append(classificator_knn(test_dataset[:, :, j], train_datasets, k))

        test_results.append(np.array(test_result))

    p01 = np.sum([test_results[0] == 1]) / len(test_results[0])
    p10 = np.sum([test_results[1] == 0]) / len(test_results[1])

    p02 = np.sum([test_results[0] == 2]) / len(test_results[0])
    p20 = np.sum([test_results[2] == 0]) / len(test_results[2])

    p12 = np.sum([test_results[1] == 2]) / len(test_results[1])
    p21 = np.sum([test_results[2] == 1]) / len(test_results[2])

    N0 = test_datasets[0].shape[-1]
    N1 = test_datasets[1].shape[-1]
    N2 = test_datasets[2].shape[-1]

    P0 = N0 / (N0 + N1 + N2)
    P1 = N1 / (N0 + N1 + N2)
    P2 = N2 / (N0 + N1 + N2)

    risk_estimation = P0 * (p01 + p02) + P1 * (p10 + p12) + P2 * (p20 + p21)

    print(f"Вероятность ошибки p01: {p01}")
    print(f"Вероятность ошибки p10: {p10}")
    print(f"Вероятность ошибки p02: {p02}")
    print(f"Вероятность ошибки p20: {p20}")
    print(f"Вероятность ошибки p12: {p12}")
    print(f"Вероятность ошибки p21: {p21}")
    print(f"Эмпирический риск: {risk_estimation}")









main()