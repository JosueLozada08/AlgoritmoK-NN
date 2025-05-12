
"""
Script para aplicar clasificación K-NN al dataset de diabetes.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Cargar el dataset
df = pd.read_csv('diabetes.csv')

print("Primeras filas del dataset:")
print(df.head())
print("\nResumen del dataset:")
print(df.info())

# 2. Separar variables predictoras (X) y variable objetivo (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Escalar características (muy importante para K-NN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. Crear y entrenar el modelo K-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 6. Hacer predicciones
y_pred = knn.predict(X_test)

# 7. Evaluación del modelo
print("\n--- Evaluación del modelo K-NN ---")
print(f"Precisión (accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
