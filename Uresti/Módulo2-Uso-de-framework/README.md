# Modelo de Clasificación con Scikit-Learn

Este repositorio contiene un código Python que implementa un modelo de clasificación utilizando la biblioteca Scikit-Learn de Python. El modelo se entrena y evalúa en un conjunto de datos y se utilizan diversas métricas de evaluación para evaluar su rendimiento. Además, se representa gráficamente la matriz de confusión.

## Contenido de la Carpeta

- `Modelo.py`: Este es el archivo principal que contiene el código fuente del modelo de clasificación. Se implementa un RandomForestClassifier de Scikit-Learn y se utilizan métricas como precisión, recall y F1-score para evaluar el modelo. Además, se genera una matriz de confusión como representación gráfica.
- `hotel_bookings_completo.csv`: Este archivo CSV es el conjunto de datos utilizado para entrenar y evaluar el modelo. Contiene características relacionadas con reservas de hoteles y la variable objetivo "children" que se utiliza en la clasificación.

## Cómo Utilizar el Código

1. Asegúrate de tener instaladas las siguientes bibliotecas de Python: Scikit-Learn, Pandas, Matplotlib y Seaborn. Puedes instalarlas utilizando pip:

   ```
   pip install scikit-learn pandas matplotlib seaborn
   ```
2. Reemplaza "hotel_bookings_completo.csv" con la ruta de tu propio conjunto de datos en formato CSV si deseas utilizar tus propios datos.
3. Ajusta los hiperparámetros del modelo RandomForestClassifier en el archivo `Modelo.py` según tus necesidades específicas.
4. Ejecuta el programa utilizando un entorno de Python. Puedes hacerlo mediante la línea de comandos:

```
python3 Modelo_clasificacion.py
```

5. Observa las métricas de evaluación en la salida, incluyendo precisión, recall, F1-score y la matriz de confusión representada gráficamente.

## Reglas Cumplidas por el Modelo

A continuación se muestra una tabla que enumera las reglas que este modelo cumple, de acuerdo a los requisitos especificados:

| Regla                                         | Cumplimiento |
| --------------------------------------------- | ------------ |
| Demuestra generalización                     | Sí          |
| Presenta múltiples métricas de evaluación  | Sí          |
| Divide el Dataset en train, validation y test | Sí          |
| Se ejecuta sin IDE o notebook                 | Sí          |
| Utiliza el framework Scikit-Learn             | Sí          |

Este modelo de clasificación cumple con los requisitos establecidos y se puede utilizar como punto de partida para tareas de clasificación en conjunto con Scikit-Learn.
