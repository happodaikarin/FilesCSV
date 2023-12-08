# Importaciones necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import numpy as np

# Carga de datos
file_path = 'cybersecurity_attacks.csv'  # Asegúrate de tener el archivo correcto en tu entorno de Colab
data = pd.read_csv(file_path)

# Limpieza de datos: descartando columnas con muchos valores faltantes
columns_to_drop = ['Malware Indicators', 'Alerts/Warnings', 'Proxy Information', 'Firewall Logs', 'IDS/IPS Alerts']
data_cleaned = data.drop(columns=columns_to_drop)

# Preprocesamiento de datos: Codificación de características categóricas y escalado de características numéricas
categorical_features = data_cleaned.select_dtypes(include=['object']).columns.tolist()
numerical_features = data_cleaned.select_dtypes(exclude=['object']).columns.tolist()

# Removiendo columnas con muchos valores únicos
categorical_features.remove('Source IP Address')
categorical_features.remove('Destination IP Address')
categorical_features.remove('Payload Data')
categorical_features.remove('User Information')
categorical_features.remove('Device Information')
categorical_features.remove('Geo-location Data')

# Codificación one-hot
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
data_encoded = encoder.fit_transform(data_cleaned[categorical_features])
data_encoded_df = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Escalado de características numéricas
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cleaned[numerical_features])
data_scaled_df = pd.DataFrame(data_scaled, columns=numerical_features)

# Combinando datos codificados y escalados
processed_data = pd.concat([data_encoded_df, data_scaled_df], axis=1)

# Preparando la variable objetivo
target = data_cleaned['Attack Type']
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(processed_data, target_encoded, test_size=0.2, random_state=42)

# Construcción de la red neuronal
num_classes = len(label_encoder.classes_)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Evaluación del modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Predicciones y reporte de clasificación
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
