from lab_1_dataset_generation.main import generate_dataset
import numpy as np
import matplotlib.pyplot as plt




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

    plt.figure()
    plt.scatter(dataset_0[0, :, :], dataset_0[1, :, :], color='red')
    plt.scatter(dataset_1[0, :, :], dataset_1[1, :, :], color='orange')
    plt.scatter(dataset_2[0, :, :], dataset_2[1, :, :], color='yellow')
    plt.scatter(dataset_3[0, :, :], dataset_3[1, :, :], color='green')
    plt.scatter(dataset_4[0, :, :], dataset_4[1, :, :], color='blue')

    dataset = np.zeros((2, 1, N * 5))
    for i in range(N):
        dataset[:, :, 5 * i] = dataset_0[:, :, i]
        dataset[:, :, 5 * i + 1] = dataset_1[:, :, i]
        dataset[:, :, 5 * i + 2] = dataset_2[:, :, i]
        dataset[:, :, 5 * i + 3] = dataset_3[:, :, i]
        dataset[:, :, 5 * i + 4] = dataset_4[:, :, i]

    plt.figure()
    plt.scatter(dataset[0, :, :], dataset[1, :, :], color='black')
    plt.show()



main()