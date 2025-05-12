
# 🧠 Clasificación K-NN – Dataset de Diabetes

## 🎯 Objetivo del Modelo

Este proyecto aplica el algoritmo de clasificación **K-Nearest Neighbors (K-NN)** para predecir si una persona tiene diabetes (`Outcome = 1`) o no (`Outcome = 0`) a partir de datos clínicos.

---

## ⚙️ Variables Utilizadas

Las variables predictoras son:

- `Pregnancies`: Número de embarazos
- `Glucose`: Nivel de glucosa
- `BloodPressure`: Presión arterial
- `SkinThickness`: Grosor del pliegue cutáneo
- `Insulin`: Nivel de insulina
- `BMI`: Índice de masa corporal
- `DiabetesPedigreeFunction`: Historial genético
- `Age`: Edad

Variable objetivo:
- `Outcome`: 0 = no tiene diabetes, 1 = tiene diabetes

---

## 🔧 Preparación del Modelo

- Escalado de variables con `StandardScaler`
- División 80% entrenamiento / 20% prueba
- K-NN con `n_neighbors = 5`

---

## 📈 Resultados del Modelo

- **Accuracy**: `0.6883`

### 🧮 Matriz de Confusión

|               | Predicho 0 | Predicho 1 |
|---------------|------------|------------|
| Real 0 (no)   | 79         | 20         |
| Real 1 (sí)   | 28         | 27         |

---

### 📋 Reporte de Clasificación

| Clase | Precision | Recall | F1-score | Soporte |
|-------|-----------|--------|----------|---------|
| 0     | 0.74      | 0.80   | 0.77     | 99      |
| 1     | 0.57      | 0.49   | 0.53     | 55      |
| Macro avg | 0.66  | 0.64   | 0.65     | 154     |
| Weighted avg | 0.68 | 0.69 | 0.68     | 154     |

---

## 📌 Interpretación

- El modelo clasifica mejor a los pacientes **sin diabetes**.
- Tiene menor capacidad para identificar correctamente a los pacientes con diabetes (recall = 0.49).
- F1-score para la clase positiva (`1`) fue 0.53.

---

## 🚀 Posibles Mejoras

- Probar otros valores de `k`
- Evaluar con validación cruzada
- Usar algoritmos más complejos (Random Forest, SVM)
- Aplicar balanceo de clases (SMOTE)

---

## 🧾 Librerías utilizadas

```python
pandas
scikit-learn
```

---

## 📁 Archivo de datos

El archivo `diabetes.csv` es de uso libre y proviene de [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
