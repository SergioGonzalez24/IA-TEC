# -------------------------------------------------------------------------------------------------
# |                                                                                               |
# | Título: Implementación de un modelo de clasificación utilizando Scikit-Learn                  |
# | Implementación de un modelo de clasificación utilizando Scikit-Learn                          |
# | Fecha: 11/09/2023                                                                             |
# |                                                                                               |
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# |                                                                                               |
# | Instrucciones para el desarrollo del programa:                                                |
# |                                                                                               |
# | Este programa implementa un modelo de clasificación utilizando Scikit-Learn, una biblioteca   |
# | de aprendizaje automático de Python. El modelo se entrena y evalúa en un conjunto de datos    |
# | que se divide en conjuntos de entrenamiento, validación y prueba. Se utilizan métricas de     |
# | evaluación como precisión, recall y F1-score, y se muestra una matriz de confusión como       |
# | representación gráfica.                                                                       |
# |                                                                                               |
# | Asegúrate de tener Scikit-Learn, Pandas y Matplotlib instalados en tu entorno de Python       |
# | antes de ejecutar este programa.                                                              |
# |                                                                                               |
# -------------------------------------------------------------------------------------------------
# |                                                                                               |
# | Instrucciones para la ejecución del programa:                                                 |
# |                                                                                               |
# | 1. Reemplaza "hotel_bookings_completo.csv" con la ruta de tu propio conjunto de datos en      |
# |    formato CSV.                                                                               |
# | 2. Ajusta los hiperparámetros del modelo RandomForestClassifier según sea necesario.          |
# | 3. Ejecuta el programa y observa las métricas de evaluación y la matriz de confusión en la    |
# |    consola y la representación gráfica.                                                       |
# |                                                                                               |
# -------------------------------------------------------------------------------------------------



# El código importa las bibliotecas y módulos necesarios para realizar tareas de aprendizaje
# automático.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# La línea `data = pd.read_csv("hotel_bookings_completo.csv")` lee un archivo CSV llamado
# "hotel_bookings_completo.csv" y almacena su contenido en un DataFrame de pandas llamado `data`.
data = pd.read_csv("Módulo2-Uso-de-framework/hotel_bookings_completo.csv")

# El código realiza codificación one-hot en variables categóricas en el DataFrame `data` usando la
# función `pd.get_dummies()`. Esto convierte variables categóricas en columnas binarias, donde cada
# columna representa una categoría única y contiene un 1 si la categoría está presente y un 0 si no lo
# está.
data = pd.get_dummies(data, columns=["hotel", "is_canceled", "arrival_date_month", "assigned_room_type", "deposit_type", "customer_type"])

# El código divide el conjunto de datos en dos partes: "X" e "y".
X = data.drop("children", axis=1)
y = data["children"]

# El código divide el conjunto de datos en tres partes: conjunto de entrenamiento, conjunto de
# validación y conjunto de prueba.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# El código crea una instancia del modelo RandomForestClassifier con hiperparámetros especificados. El
# parámetro `n_estimators` determina el número de árboles de decisión en el bosque aleatorio, `max_
# Depth` establece la profundidad máxima de cada árbol y `random_state` garantiza la reproducibilidad
# de los resultados. Luego, el modelo se entrena con los datos de entrenamiento (`X_train` e
# `y_train`) utilizando el método `fit()`.
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# `y_val_pred = model.predict(X_val)` predice la variable objetivo (`y`) para el conjunto de
# validación (`X_val`) utilizando el modelo entrenado (`model`). Los valores predichos se almacenan en
# la variable `y_val_pred`.
y_val_pred = model.predict(X_val)

# El código calcula varias métricas de evaluación para el conjunto de validación.
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, average="weighted")  
recall = recall_score(y_val, y_val_pred, average="weighted")        
f1 = f1_score(y_val, y_val_pred, average="weighted")

print("Métricas de evaluación en el conjunto de validación:")
print(f"Accuracy: {accuracy}")
print(f"Precisión: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# `y_test_pred = model.predict(X_test)` predice la variable objetivo (`y`) para el conjunto de prueba
# (`X_test`) usando el modelo entrenado (`model`). Los valores predichos se almacenan en la variable
# `y_test_pred`.
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nEvaluación en el conjunto de prueba:")
print(f"Accuracy en el conjunto de prueba: {test_accuracy}")

# La línea `conf_matrix = confusion_matrix(y_val, y_val_pred)` calcula la matriz de confusión para el
# conjunto de validación. La matriz de confusión es una tabla que muestra los valores verdadero
# positivo, verdadero negativo, falso positivo y falso negativo para un modelo de clasificación. Ayuda
# a evaluar el rendimiento del modelo mostrando qué tan bien predice cada clase.
conf_matrix = confusion_matrix(y_val, y_val_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.title("Matriz de Confusión")
plt.show()
