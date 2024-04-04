import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np

# Cargar el conjunto de datos
data = pd.read_csv('online_shoppers_intention.csv')

# Codificar variables categóricas
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop('Revenue', axis=1)
y = data['Revenue']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos de entrenamiento y prueba
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir el modelo de red neuronal
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo con los datos escalados
model.fit(X_train_scaled, y_train, epochs=50, batch_size=10)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int) # Convertir predicciones a etiquetas binarias

# Convertir y_pred a una lista
y_pred_list = np.ravel(y_pred).tolist() # Aplanar y_pred

# Crear un DataFrame con las predicciones
df_predicciones = pd.DataFrame({
    'Cliente': range(1, len(y_pred_list) + 1), # Asumiendo que los clientes están numerados desde 1
    'Predicción': y_pred_list,
})

# Convertir las predicciones a enteros
df_predicciones['Predicción'] = df_predicciones['Predicción'].astype(int)

# Definir una función para convertir las predicciones a 'Sí' o 'No'
def convertir_prediccion(pred):
    return 'Sí' if pred == 1 else 'No'

# Aplicar la función a la columna 'Predicción' para generar la columna 'Efectuará compra'
df_predicciones['Efectuará compra'] = df_predicciones['Predicción'].apply(convertir_prediccion)

# Guardar el DataFrame de predicciones en un archivo CSV
df_predicciones.to_csv('predicciones.csv', index=False, encoding='utf-8-sig')



# Crear un modelo de regresión logística para RFE
lr = LogisticRegression()

# Aplicar RFE para seleccionar las características más relevantes
rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X_train_scaled, y_train)

# Crear un DataFrame con las características relevantes
traducciones = {
    'ProductRelated': 'Relacionado con el producto',
    'ExitRates': 'Tasas de salida',
    'PageValues': 'Valores de la página',
    'SpecialDay': 'Día especial',
    'Month': 'Mes'
}

df_caracteristicas = pd.DataFrame({
    'Característica': X.columns[rfe.support_],
    'Traducción': [traducciones[caracteristica] for caracteristica in X.columns[rfe.support_]]
})

# Guardar el DataFrame de características relevantes en un archivo CSV
df_caracteristicas.to_csv('caracteristicas_relevantes.csv', index=False, encoding='utf-8-sig')
