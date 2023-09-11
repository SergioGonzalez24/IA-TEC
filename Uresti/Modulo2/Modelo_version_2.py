# -------------------------------------------------------------------------------------------------
# |                                                                                               |
# | Título: Implementación de K-Means Clustering sin el uso de framework                          |
# | Implementación de K-Means Clustering sin el uso de framework                                  |
# | Autor: Sergio Gonzalez                                                                        |
# | Fecha: 11/09/2023                                                                             |
# |                                                                                               |
# | Notas: Documentación generada con la ayuda de la herramienta Mintlify Doc Writer.             |
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# |                                                                                               |
# | Instrucciones para el desarrollo del programa:                                                |
# |                                                                                               |
# | Programa la implementación del algoritmo K-Means desde cero, sin utilizar bibliotecas o       |
# | frameworks de aprendizaje automático ni de estadística avanzada.                              |
# | Las predicciones se pueden realizar en consola o utilizando una interfaz gráfica como         |
# | matplotlib o seaborn.                                                                         |
# |                                                                                               |
# -------------------------------------------------------------------------------------------------
# |                                                                                               |
# | Instrucciones para la ejecución del programa:                                                 |
# |                                                                                               |
# | Prueba la implementación con un conjunto de datos y realiza evaluaciones de rendimiento.      |
# |                                                                                               |
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# |                                                                                               |
# | Descripción del Programa:                                                                     |
# |                                                                                               |
# | Este programa implementa el algoritmo K-Means para realizar agrupamientos de datos.           |
# | Los datos se cargan desde un archivo CSV y se preprocesan antes de aplicar K-Means.           |
# | Luego, se divide el conjunto de datos en entrenamiento, validación y prueba, se entrena el    |
# | modelo K-Means y se evalúa su rendimiento en los conjuntos de validación y prueba.            |
# |                                                                                               |
# -------------------------------------------------------------------------------------------------


# El código está importando las bibliotecas necesarias para el programa.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def distancia(p1, p2):
    """
    La función calcula la distancia entre dos puntos en un espacio multidimensional.
    
    :param p1: El parámetro p1 representa las coordenadas del primer punto en un espacio bidimensional
    :param p2: El parámetro `p2` representa las coordenadas de un punto en un espacio multidimensional
    :return: la distancia euclidiana entre dos puntos, p1 y p2.
    """
    return np.sqrt(np.sum((p1 - p2) ** 2))

# La clase `KMeansClustering` implementa el algoritmo de agrupación K-means para agrupar puntos de
# datos en k grupos.
class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    def ajustar(self, X, max_iteraciones=100):
        self.centroids = random.sample(list(X), self.k)

        for _ in range(max_iteraciones):
            clusters = [[] for _ in range(self.k)]

            for punto in X:
                distancias = [distancia(punto, centroid) for centroid in self.centroids]
                cluster_asignado = np.argmin(distancias)
                clusters[cluster_asignado].append(punto)

            nuevos_centroids = [np.mean(cluster, axis=0) if cluster else centroid for cluster, centroid in zip(clusters, self.centroids)]

            if np.allclose(self.centroids, nuevos_centroids):
                break

            self.centroids = nuevos_centroids

        etiquetas = [np.argmin([distancia(punto, centroid) for centroid in self.centroids]) for punto in X]

        return etiquetas

# Carga de datos y preprocesamiento
df = pd.read_csv('spotify-2023.csv', encoding='ISO-8859-1')

# El código `df['streams'] = pd.to_numeric(df['streams'], errores='coerce')` está convirtiendo la
# columna 'streams' en el DataFrame `df` a valores numéricos. La función `pd.to_numeric()` se utiliza
# para convertir los valores de la columna al tipo de datos numéricos. El parámetro `errors='coerce'`
# se utiliza para reemplazar cualquier valor no numérico con NaN (no es un número).
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
df = df.dropna(subset=['streams'])

# La línea `columnas_relevantes = ['released_year', 'released_month', 'released_day', 'streams',
# 'bpm', 'danceability_%', 'valence_%']` está creando una lista llamada `columnas_relevantes` que
# contiene los nombres de las columnas en el DataFrame `df` que se consideran relevantes para el
# algoritmo de agrupamiento de K-Means. Estas columnas son 'año_de_lanzamiento', 'mes_de_lanzamiento',
# 'día_de_lanzamiento', 'streams', 'bpm', 'danceability_%' y 'valence_%'.
columnas_relevantes = ['released_year', 'released_month', 'released_day', 'streams', 'bpm', 'danceability_%', 'valence_%']

# El código `X = df[columnas_relevantes].values` es seleccionar las columnas especificadas en la lista
# `columnas_relevantes` del DataFrame `df` y asignar los valores resultantes a la variable `X`.
X = df[columnas_relevantes].values
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# División del conjunto de datos en entrenamiento, validación y prueba
random.seed(42)  # Semilla aleatoria para reproducibilidad
n = len(X)
n_train = int(0.6 * n)
n_val = int(0.2 * n)
n_test = n - n_train - n_val
# El código `indices = list(range(n))` está creando una lista de índices de 0 a n-1.

indices = list(range(n))
random.shuffle(indices)

# El código divide el conjunto de datos "X" en conjuntos de entrenamiento, validación y prueba.
X_train = X[indices[:n_train]]
X_val = X[indices[n_train:n_train + n_val]]
X_test = X[indices[n_train + n_val:]]

# Entrenamiento del modelo K-Means
k = 2
modelo = KMeansClustering(k)
etiquetas_entrenamiento = modelo.ajustar(X_train)

# Asignar etiquetas al conjunto de validación
etiquetas_validacion = []

# Este código asigna etiquetas al conjunto de validación según el modelo de agrupación de K-Means.
for punto in X_val:
    distancias = [distancia(punto, centroid) for centroid in modelo.centroids]
    cluster_asignado = np.argmin(distancias)
    etiquetas_validacion.append(cluster_asignado)

# Cálculo de la inercia en el conjunto de validación
def inercia(clusters, centroids):
    return sum(np.sum((p - centroids[i]) ** 2) for i, cluster in enumerate(clusters) for p in cluster)

clusters_val = [[] for _ in range(k)]

# El código asigna cada punto en el conjunto de validación (`X_val`) al centroide del grupo más
# cercano según la distancia euclidiana. Calcula las distancias entre el punto y cada centroide usando
# la función `distancia()`, y luego asigna el punto al grupo con la distancia mínima. El punto se
# agrega al grupo correspondiente en la lista `clusters_val`. Este proceso se repite para cada punto
# del conjunto de validación.
for punto in X_val:
    distancias = [distancia(punto, centroid) for centroid in modelo.centroids]
    cluster_asignado = np.argmin(distancias)
    clusters_val[cluster_asignado].append(punto)

# La línea `inercia_val = inercia(clusters_val, modelo.centroids)` está calculando la inercia de los
# clusters en el conjunto de validación. La inercia es una medida de cuán internamente coherentes son
# los grupos. Se calcula como la suma de las distancias al cuadrado entre cada punto y su centroide
# asignado dentro de cada grupo. La función `inercia()` toma los grupos y centroides como entrada y
# calcula la inercia iterando sobre cada grupo y sumando las distancias al cuadrado. El valor de
# inercia resultante se almacena en la variable `inercia_val`.
inercia_val = inercia(clusters_val, modelo.centroids)

print(f'Inercia en conjunto de validación: {inercia_val}')

# Visualización de clusters en conjunto de validación
plt.scatter(X_val[:, 0], X_val[:, 1], c=etiquetas_validacion, cmap='viridis', alpha=0.4, s=20)
plt.scatter(np.array(modelo.centroids)[:, 0], np.array(modelo.centroids)[:, 1], c="red", marker="x", s=100)
plt.title('Agrupamiento K-Means en conjunto de validación')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.show()
