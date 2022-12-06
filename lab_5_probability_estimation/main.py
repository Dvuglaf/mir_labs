import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show
from lab_1_dataset_generation.main import generate_dataset, evaluate_parameters
from lab_2_classifiers.main import classification_error, bayes_border_fixed


def Kernel_Density_Estimation_Normal(x, dataset):
    N = dataset.shape[-1]
    m, B = evaluate_parameters(dataset, N)
    k = 1 / 4
    h = np.float_power(N, -k / 2)
    # B_parzen = (1 + h ** 2) * B
    result = 0
    for i in range(N):
        result += 1 / (2 * np.pi) * (h ** -2) / np.sqrt(np.linalg.det(B)) * np.exp(
            -0.5 * (h ** -2) * np.matmul(np.matmul((x - dataset[:, :, i]).T, np.linalg.inv(B)),
                                         (x - dataset[:, :, i])))
    return result[0] / N


def classificator_parzen(x, datasets, args=0):
    P_values = get_aprior_probs(datasets)

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
    for class_label in range(len(datasets)):
        count_neighbors.append(len(nearest_neighbors[nearest_neighbors[:, 1] == class_label]))

    return np.argmax(count_neighbors)


def get_p_errors(test_results):
    p_errors = []
    for i in range(len(test_results)):
        p_errors_ij = []

        for j in range(len(test_results)):
            if i != j:
                p_errors_ij.append(np.sum([test_results[i] == j]) / len(test_results[i]))
            else:
                p_errors_ij.append(0)

        p_errors.append(np.array(p_errors_ij))

    return np.array(p_errors)


def get_aprior_probs(datasets):
    N_values = []
    for dataset in datasets:
        N_values.append(dataset.shape[-1])

    P_values = []
    for dataset in datasets:
        P_values.append(dataset.shape[-1] / np.sum(N_values))

    return np.array(P_values)


def get_risk_estimation(aprior_probs, p_errors):
    return np.sum(aprior_probs * np.sum(p_errors, axis=1))


def classify_data(train_datasets, test_datasets, classificator, param=0):
    test_results = []
    for test_dataset in test_datasets:
        test_result = []

        for j in range(test_dataset.shape[-1]):
            test_result.append(classificator(test_dataset[:, :, j], train_datasets, param))

        test_results.append(np.array(test_result))

    return np.array(test_results)


def get_classification_errors(test_results, train_datasets):
    p_errors = get_p_errors(test_results)
    aprior_probs = get_aprior_probs(train_datasets)
    risk_estimation = get_risk_estimation(aprior_probs, p_errors)

    for i in range(len(p_errors)):
        for j in range(len(p_errors)):
            if i != j:
                print(f"Вероятность ошибочной классификации p{i}{j}: {p_errors[i, j]}")

    print(f"Эмпирический риск: {risk_estimation}")


def plot_test_data(test_datasets, test_results, title, colors):
    plt.figure()
    plt.suptitle(title)

    for i in range(len(test_datasets)):
        plt.scatter(test_datasets[i][0, :, :], test_datasets[i][1, :, :], color=colors[i])
        plt.scatter(test_datasets[i][0, :, test_results[i] != i], test_datasets[i][1, :, test_results[i] != i],
                 linewidth=1.5, facecolors='none', edgecolors='k')


if __name__ == "__main__":
    m0 = np.array(([0], [-2]))
    m1 = np.array(([-1], [1]))
    m2 = np.array(([2], [0]))

    b = np.array(([0.5, -0.2], [-0.2, 0.5]))  # случай равных корреляционных матриц

    b0 = np.array(([0.5, 0], [0, 0.5]))
    b1 = np.array(([0.4, 0.1], [0.1, 0.6]))
    b2 = np.array(([0.6, -0.2], [-0.2, 0.6]))

    train_datasets = [generate_dataset(b, m0, 50), generate_dataset(b, m1, 50)]
    test_datasets = [generate_dataset(b, m0, 100), generate_dataset(b, m1, 100)]

    print("\nклассификатор Байеса, равные корреляционные матрицы, 2 класса")
    p01 = classification_error(test_datasets[0], 0.5, 0.5, b, b, m0, m1, test_datasets[0].shape[-1])
    p10 = classification_error(test_datasets[1], 0.5, 0.5, b, b, m1, m0, test_datasets[1].shape[-1])
    print(f"Вероятность ошибочной классификации p01: {p01}")
    print(f"Вероятность ошибочной классификации p10: {p10}")
    print(f"Эмпирический риск: {0.5 * p01 + 0.5 * p10}")

    y = np.arange(-4, 4, 0.1)

    x = bayes_border_fixed(y, 0.5, 0.5, b, b, m0, m1)
    x = np.reshape(x, y.shape)

    title = "\nметод Парзена, равные корреляционные матрицы, 2 класса"
    print(title)
    test_results = classify_data(train_datasets, test_datasets, classificator_parzen)
    get_classification_errors(test_results, train_datasets)
    plot_test_data(test_datasets, test_results, title, ['red', 'green'])
    plt.plot(x, y, color='black')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)
    plt.figure()
    plt.scatter(train_datasets[0][0, :, :], train_datasets[0][1, :, :], color='red')
    plt.scatter(train_datasets[1][0, :, :], train_datasets[1][1, :, :], color='green')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)

    for k in [1, 3, 5]:
        title = f"\nметод K ближайщих соседей, k = {k}, равные корреляционные матрицы, 2 класса"
        print(title)
        test_results = classify_data(train_datasets, test_datasets, classificator_knn, k)
        get_classification_errors(test_results, train_datasets)
        plot_test_data(test_datasets, test_results, title, ['red', 'green'])
        plt.plot(x, y, color='black')
        plt.xlim(left=-4, right=4)
        plt.ylim(bottom=-4, top=4)

    train_datasets = [generate_dataset(b0, m0, 50), generate_dataset(b1, m1, 50), generate_dataset(b2, m2, 50)]
    test_datasets = [generate_dataset(b0, m0, 100), generate_dataset(b1, m1, 100), generate_dataset(b2, m2, 100)]

    print("\nклассификатор Байеса, разные корреляционные матрицы, 3 класса")
    p01 = classification_error(test_datasets[0], 1 / 3, 1 / 3, b0, b1, m0, m1, test_datasets[0].shape[-1])
    p02 = classification_error(test_datasets[0], 1 / 3, 1 / 3, b0, b2, m0, m2, test_datasets[0].shape[-1])
    p10 = classification_error(test_datasets[1], 1 / 3, 1 / 3, b1, b0, m1, m0, test_datasets[1].shape[-1])
    p12 = classification_error(test_datasets[1], 1 / 3, 1 / 3, b1, b2, m1, m2, test_datasets[1].shape[-1])
    p20 = classification_error(test_datasets[2], 1 / 3, 1 / 3, b2, b0, m2, m0, test_datasets[2].shape[-1])
    p21 = classification_error(test_datasets[2], 1 / 3, 1 / 3, b2, b1, m2, m1, test_datasets[2].shape[-1])
    print(f"Вероятность ошибочной классификации p01: {p01}")
    print(f"Вероятность ошибочной классификации p02: {p02}")
    print(f"Вероятность ошибочной классификации p10: {p10}")
    print(f"Вероятность ошибочной классификации p12: {p12}")
    print(f"Вероятность ошибочной классификации p20: {p20}")
    print(f"Вероятность ошибочной классификации p21: {p21}")
    print(f"Эмпирический риск: {1 / 3 * (p01 + p02) + 1 / 3 * (p10 + p12) + 1 / 3 * (p20 + p21)}")

    title = "\nметод Парзена, разные корреляционные матрицы, 3 класса"
    print(title)
    test_results = classify_data(train_datasets, test_datasets, classificator_parzen)
    get_classification_errors(test_results, train_datasets)
    # plot_test_data(test_datasets, test_results, title, ['red', 'green', 'blue'])


    for k in [1, 3, 5]:
        title = f"\nметод K ближайщих соседей, k = {k}, разные корреляционные матрицы, 3 класса"
        print(title)
        test_results = classify_data(train_datasets, test_datasets, classificator_knn, k)
        get_classification_errors(test_results, train_datasets)
        plot_test_data(test_datasets, test_results, title, ['red', 'green', 'blue'])

    show()

