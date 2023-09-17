import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos en un DataFrame de pandas
df = pd.read_csv('AUTO_MODIF.csv')  # Reemplaza 'tu_archivo.csv' con el nombre de tu archivo de datos

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df[['horsepower', 'curb-weight', 'engine-size', 'city-mpg']]  # Variables independientes
y = df['price']  # Variable dependiente (precio)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Puedes ajustar los hiperparámetros según sea necesario
rf_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = rf_model.predict(X_test)

# Calcular las métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R^2): {r2}')