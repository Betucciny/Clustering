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
        for j in range(len(true_labels)):
            cluster_coincidences[i][j] = 0
    for cluster, true in zip(cluster_labels, true_labels):
        cluster_coincidences[cluster][true] += 1

    mejor_coincidencia = []
    porcentaje_suma = 0
    for i in range(k):
        mejor_coincidencia.append(max(cluster_coincidences[i].items(), key=lambda x: x[1])[0])
        mejor_cantidad = cluster_coincidences[i][mejor_coincidencia[i]]
        porcentaje = mejor_cantidad / sum(cluster_coincidences[i].values())
        porcentaje_suma += porcentaje
        # print(f'Porcentaje de aciertos para el cluster {i}: {porcentaje:.2%}')
    porcentaje_promedio = porcentaje_suma / k
    print(f'Porcentaje promedio de aciertos: {porcentaje_promedio:.2%}')
    return mejor_coincidencia


def plot_centroids(centroids, labels, x, y):
    fig, axes = plt.subplots(x, y, figsize=(20, 20))
    for i, pa in enumerate(zip(axes.flat, labels)):
        ax, label = pa
        ax.imshow(centroids[i].reshape(28, 28), cmap='binary')
        ax.set_title(f"Label {label}", fontsize=16)
        ax.set(xticks=[], yticks=[])
    plt.show()
    plt.clf()


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = x_train.reshape(len(x_train), -1)
    Y = y_train
    X = X.astype(float) / 255.
    data = X.reshape(len(X), -1)
    # data = data[:1000]
    clusters = [10, 30, 50, 70, 90]
    for cluster in clusters:
        cluster_labels, centroids = kmeans(data, cluster)
        mejor_coincidencia = infer(Y, cluster_labels, cluster)
        plot_centroids(centroids, mejor_coincidencia, cluster//10, 10)


if __name__ == '__main__':
    main()


