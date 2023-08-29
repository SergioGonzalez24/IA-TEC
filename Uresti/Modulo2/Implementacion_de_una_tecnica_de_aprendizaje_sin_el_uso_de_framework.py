# -------------------------------------------------------------------------------------------------
# |                                                                                               |
# | Título: Implementación de una técnica de aprendizaje sin el uso de framework                  |
# | Implementación de una técnica de aprendizaje sin el uso de framework                          |
# | Autor: Sergio Gonzalez - A01745446                                                            |
# | Fecha: 28/08/2023                                                                             |
# |                                                                                               |
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# |                                                                                               |
# |Instrucciones para el desarrollo del programa:                                                 |
# |                                                                                               |
# | Programa uno de los algoritmos de ML sin usar ninguna biblioteca o framework de aprendizaje   |
# | máquina, ni de estadística avanzada.                                                          |
# | Lo que se busca es que se implemente manualmente el algoritmo, no que importes un algoritmo   |
# | ya implementado.                                                                              |
# | Las predicciones las puedes correr en consola o las puedes implementar con una interfaz       |
# | gráfica como matplotlib o seaborn.                                                            |
# |                                                                                               |
# -------------------------------------------------------------------------------------------------
# |                                                                                               |
# | Instrucciones para la ejecución del programa:                                                 |
# |                                                                                               |
# | Prueba la implementación con un set de datos y realiza algunas predicciones.                  |
# |                                                                                               |
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# |                                                                                               |
# | Descripción del Programa:                                                                     |
# |                                                                                               |
# -------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv("HousingData.csv")

class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def fit(self, X, max_iterations=100):
        y = np.zeros(X.shape[0], dtype=int)  # Initialize labels
        
        # Manual normalization
        X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        
        self.centroids = np.random.uniform(0, 1, (self.k, X_normalized.shape[1]))

        for _ in range(max_iterations):
            prev_centroids = np.copy(self.centroids)
            y = []

            for data_point in X_normalized:
                distances = KMeansClustering.distance(data_point, self.centroids)
                cluster_number = np.argmin(distances)
                y.append(cluster_number)

            y = np.array(y)

            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X_normalized[indices], axis=0)[0])

            self.centroids = np.array(cluster_centers)

            if np.max(np.abs(prev_centroids - self.centroids)) < 0.0001:
                break
        
        return y

# List of column names for features
data_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Extract relevant columns from the DataFrame
X = df[data_columns].values

# K-Means clustering
kmeans = KMeansClustering(k=3)
labels = kmeans.fit(X)

# Print cluster assignments for each data point
for i, label in enumerate(labels):
    print(f"Data point {i}: Cluster {label}")

# Print final cluster centroids
print("Final Cluster Centroids:")
print(kmeans.centroids)

# Visualization
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c="red", marker="x")
plt.show()
