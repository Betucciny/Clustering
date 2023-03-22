import numpy as np
from keras.datasets import mnist


# Lectura de centroides
def read_centroid(clusters):
    centroids = []
    labels = []
    for cluster in clusters:
        centroids.append(np.loadtxt(f'centroids_{cluster}.txt'))
        labels.append(np.loadtxt(f'labels_{cluster}.txt'))
    return centroids, labels


# Calculo de la cantidad de coincidencias entre los clusters y las etiquetas verdaderas
def coincidence(centroids, labels, X, Y):
    count_good = 0
    for vector, label in zip(X,Y):
        best_centroid = None
        best_distance = np.inf
        for i, centroid in enumerate(centroids):
            distance = np.linalg.norm(vector - centroid)
            if distance < best_distance:
                best_centroid = i
                best_distance = distance
        if labels[best_centroid] == label:
            count_good += 1
    return count_good / len(X)


# Función principal
def main():
    # Cantidad de clusters
    clusters = [10, 30, 50, 70, 90]
    # Lectura de centroides
    centroids, labels = read_centroid(clusters)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = x_test.reshape(len(x_test), -1)
    Y = y_test
    X = X.astype(float) / 255.
    for i, cluster in enumerate(clusters):
        print(f'Porcentaje de aciertos para {cluster} clusters: {coincidence(centroids[i], labels[i], X, Y):.2%}')


if __name__ == '__main__':
    main()