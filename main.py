import numpy as np
import random
from keras.datasets import mnist
import matplotlib.pyplot as plt


def kmeans(data, k, max_iterations=100):
    n_samples, n_features = data.shape

    # Inicialización de los centroides usando el algoritmo KMeans++
    centroids = [data[random.randint(0, n_samples - 1)]]
    for i in range(1, k):
        distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in data])
        probabilities = distances / np.sum(distances)
        cum_probabilities = np.cumsum(probabilities)
        r = random.random()
        for j, cp in enumerate(cum_probabilities):
            if r < cp:
                centroids.append(data[j])
                break

    # Asignación de cada muestra al centroide más cercano
    labels = np.zeros(n_samples)
    distances_to_centroids = np.zeros((n_samples, k))
    for iteration in range(max_iterations):
        for i in range(n_samples):
            distances_to_centroids[i] = [np.linalg.norm(data[i] - c) ** 2 for c in centroids]
        new_labels = np.argmin(distances_to_centroids, axis=1)
        if np.all(labels == new_labels):
            break
        labels = new_labels

        # Actualización de los centroides
        for i in range(k):
            centroids[i] = np.mean(data[labels == i], axis=0)

    return labels, centroids


def infer(true_labels, cluster_labels, k=10):
    cluster_coincidences = dict()
    for i in range(k):
        cluster_coincidences[i] = dict()
        for j in range(k):
            cluster_coincidences[i][j] = 0
    for cluster in cl

    for i in range(k):
        print(f'Cluster: {i}, Coincidencias: {cluster_coincidences[i]}')





def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = x_train.reshape(len(x_train), -1)
    Y = y_train
    X = X.astype(float) / 255.
    data = X.reshape(len(X), -1)
    data = data[:1000]
    cluster_labels, centroids = kmeans(data, 10)
    infer(Y, cluster_labels)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
