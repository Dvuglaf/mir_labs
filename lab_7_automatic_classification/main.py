from lab_1_dataset_generation.main import generate_dataset
import numpy as np
import matplotlib.pyplot as plt


def maxmin_dist_clasters(dataset):
    N = dataset.shape[-1]
    claster_centres = []

    d_typical_values = []
    d_maxmin_values = []

    # Шаг №1
    mean_x = np.mean(dataset, axis=2)
    distances = np.zeros(N)
    for i in range(N):
        distances[i] = np.linalg.norm(dataset[:, :, i] - mean_x)
    # центр первого кластера
    claster_centres.append(dataset[:, :, np.argmax(distances)])

    # Шаг №2
    distances = np.zeros(N)
    for i in range(N):
        distances[i] = np.linalg.norm(dataset[:, :, i] - claster_centres[0])
    # центр второго кластера
    claster_centres.append(dataset[:, :, np.argmax(distances)])

    # Шаг L > 2
    finished = False
    x_clasters = []
    while not finished:
        # расчет d_typical
        num_of_clasters = len(claster_centres)
        distances = np.zeros((num_of_clasters, num_of_clasters))
        for i in range(num_of_clasters):
            for j in range(i + 1, num_of_clasters):
                distances[i, j] = np.linalg.norm(claster_centres[i] - claster_centres[j])
        d_typical = 1 / 2 * 2 / (num_of_clasters * (num_of_clasters - 1)) * np.sum(distances)

        d_typical_values.append(d_typical)

        # вычисляем все расстояния между векторами и центрами кластеров
        distances = np.zeros((num_of_clasters, N))
        for i in range(num_of_clasters):
            for j in range(N):
                distances[i, j] = np.linalg.norm(dataset[:, :, j] - claster_centres[i])

        # для каждого вектора находим тот центр кластера l, расстояние до которого минимально
        x_clasters = np.argmin(distances, axis=0)

        # вывод
        colors = ['red', 'olive', 'gold', 'green', 'blue', 'purple', 'brown', 'neon']
        fig = plt.figure()
        plt.suptitle(f"Кластеризация максиминным алгоритмом, число кластеров: {num_of_clasters}")

        sp = fig.add_subplot(121)
        sp.set_title("Исходные вектора")
        for i in range(dataset.shape[-1]):
            sp.scatter(dataset[0, :, i], dataset[1, :, i], color=colors[i % 5], marker='.')

        sp = fig.add_subplot(122)
        sp.set_title("Результат кластеризации")
        sp.scatter(np.array(claster_centres)[:, 0, 0], np.array(claster_centres)[:, 1, 0],
                   linewidth=1, facecolors='none', edgecolors='k')
        for i in range(len(claster_centres)):
            sp.scatter(dataset[0, :, x_clasters == i], dataset[1, :, x_clasters == i], color=colors[i],
                       marker='.')

        # претендент на новый центр кластера
        new_claster_centre = dataset[:, :, np.argmax(distances[x_clasters, range(N)])]

        d_maxmin_values.append(np.max(distances[x_clasters, range(N)]))

        # вычисление d_min и сравнение с d_typical
        sub_arr = np.array(claster_centres) - new_claster_centre
        distances = np.zeros(len(sub_arr))
        for i in range(len(sub_arr)):
            distances[i] = np.linalg.norm(sub_arr[i])
        d_min = np.min(distances)
        if d_min > d_typical:
            claster_centres.append(new_claster_centre)
        else:
            finished = True

    # графики
    fig = plt.figure()
    plt.suptitle("Зависимости максиминного и типичного расстояний от числа кластеров")
    plt.plot(range(2, len(claster_centres) + 1, 1), d_maxmin_values, color='orange', label='maxmin')
    plt.plot(range(2, len(claster_centres) + 1, 1), d_typical_values, color='blue', label='typical')
    plt.xticks(range(2, len(claster_centres) + 1, 1))
    plt.legend()

    return np.array(claster_centres), x_clasters


def main():
    N = 50

    M_0 = np.array([[0], [-2]])
    M_1 = np.array([[-1], [1]])
    M_2 = np.array([[2], [0]])
    M_3 = np.array([[-1], [0]])
    M_4 = np.array([[-1], [-1]])

    B_0 = np.array([[0.05, 0.01], [0.01, 0.05]])
    B_1 = np.array([[0.04, 0.005], [0.005, 0.02]])
    B_2 = np.array([[0.2, -0.1], [-0.1, 0.2]])
    B_3 = np.array([[0.06, 0.003], [0.003, 0.01]])
    B_4 = np.array([[0.05, 0.007], [0.007, 0.025]])

    dataset_0 = generate_dataset(B_0, M_0, N)
    dataset_1 = generate_dataset(B_1, M_1, N)
    dataset_2 = generate_dataset(B_2, M_2, N)
    dataset_3 = generate_dataset(B_3, M_3, N)
    dataset_4 = generate_dataset(B_4, M_4, N)

    dataset = np.zeros((2, 1, N * 5))
    for i in range(N):
        dataset[:, :, 5 * i] = dataset_0[:, :, i]
        dataset[:, :, 5 * i + 1] = dataset_1[:, :, i]
        dataset[:, :, 5 * i + 2] = dataset_2[:, :, i]
        dataset[:, :, 5 * i + 3] = dataset_3[:, :, i]
        dataset[:, :, 5 * i + 4] = dataset_4[:, :, i]

    claster_centres, dataset_clasters = maxmin_dist_clasters(dataset)
    print(len(claster_centres))

    plt.show()


main()