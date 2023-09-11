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
# | Descripción del Programa:                                                                   |
# |                                                                                               |
# -------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def distancia(punto_datos, centroides):
        return np.sqrt(np.sum((centroides - punto_datos) ** 2, axis=1))

    def ajustar(self, X, max_iteraciones=100):
        y = np.zeros(X.shape[0], dtype=int)  # Inicializar etiquetas
        
        # Normalización manual
        X_normalizado = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        
        self.centroids = np.random.uniform(0, 1, (self.k, X_normalizado.shape[1]))

        for _ in range(max_iteraciones):
            centroides_previos = np.copy(self.centroids)
            y = []

            for punto_datos in X_normalizado:
                distancias = KMeansClustering.distancia(punto_datos, self.centroids)
                numero_cluster = np.argmin(distancias)
                y.append(numero_cluster)

            y = np.array(y)

            indices_clusters = []

            for i in range(self.k):
                indices_clusters.append(np.argwhere(y == i))

            centroides_clusters = []

            for i, indices in enumerate(indices_clusters):
                if len(indices) == 0:
                    centroides_clusters.append(self.centroids[i])
                else:
                    centroides_clusters.append(np.mean(X_normalizado[indices], axis=0)[0])

            self.centroids = np.array(centroides_clusters)

            if np.max(np.abs(centroides_previos - self.centroids)) < 0.0001:
                break
        
        return y

df = pd.read_csv('/Users/sergiogonzalez/Documents/GitHub/IA-TEC/Uresti/Modulo2/spotify-2023.csv', encoding='ISO-8859-1')

# Lista de nombres de columna para características
columnas_datos = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 
                   'instrumentalness_%', 'liveness_%', 'speechiness_%']

# Extraer columnas relevantes del DataFrame
X = df[columnas_datos].values

# Transformar columnas en valores numéricos
for col in columnas_datos:
    unique_values = np.unique(X[:, columnas_datos.index(col)])
    mapping = {val: i for i, val in enumerate(unique_values)}
    X[:, columnas_datos.index(col)] = [mapping[val] for val in X[:, columnas_datos.index(col)]]

# Agrupamiento K-Means
kmeans = KMeansClustering(k=5)
etiquetas = kmeans.ajustar(X)

# Imprimir asignaciones de clusters para cada punto de datos
for i, etiqueta in enumerate(etiquetas):
    print(f"Punto de datos {i}: Cluster {etiqueta}")

# Imprimir centroides finales de los clusters
print("Centroides Finales de los Clusters:")
print(kmeans.centroids)

# Visualización
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=etiquetas, cmap='viridis', alpha=0.4, s=20)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c="red", marker="x", s=100)
plt.title('Agrupamiento K-Means')
plt.colorbar(label='Cluster')
plt.show()
