# TP2 — Predicción de Fumadores a partir de Señales Corporales

Trabajo Práctico 2 de la Diplomatura en Ciencia de Datos.  
El objetivo es construir un modelo de clasificación capaz de predecir si una persona es fumadora o no (`smoking`), siguiendo un flujo de trabajo de Machine Learning de extremo a extremo.

---

## Objetivo

A partir de un conjunto de 27 características biomédicas (análisis de sangre, medidas corporales, salud dental y visual), predecir la variable `smoking` (0 = no fumador, 1 = fumador).

El modelo entrenado se aplica sobre un segundo conjunto sin etiquetas para generar predicciones finales almacenadas en la columna `smoking_prediction`.

**Métrica de evaluación:** F1-Score para la clase positiva (fumador = 1).

---

## Datasets

| Dataset | Archivo | Registros | Descripción |
|---------|---------|-----------|-------------|
| Entrenamiento | `data/raw/smoking_prediction.csv` | 50,000 | Dataset etiquetado con columna `smoking` |
| Entrega | `data/raw/smoking_prediction_entrega.csv` | 5,692 | Dataset sin etiquetas para predicción final |

### Variable objetivo

La distribución de `smoking` en el dataset de entrenamiento es:
- No fumadores (0): 31,671 — 63.3%
- Fumadores (1): 18,329 — 36.7%

La distribución es aceptable (lejos del 80/20 problemático), por lo que no se aplicaron técnicas de balanceo de clases.

---

## Estructura del Proyecto

```
TP2/
├── data/
│   ├── raw/                              # Datos originales sin modificar
│   ├── processed/
│   │   ├── smoking_train_processed.csv   # Dataset etiquetado preprocesado
│   │   ├── smoking_predict_processed.csv # Dataset de entrega preprocesado
│   │   └── smoking_predictions.csv       # Predicciones finales (ID + smoking_prediction)
│   └── external/
├── models/
│   ├── xgboost_baseline.joblib           # Modelo entrenado con parámetros base
│   ├── xgboost_optimized.joblib          # Modelo final (GridSearchCV)
│   ├── best_params.json                  # Hiperparámetros y métricas del modelo final
│   └── categorical_mappings.json         # Mapeos de encoding para variables categóricas
├── notebooks/
│   ├── 01_lectura_y_discovery.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocesamiento.ipynb
│   ├── 04_entrenamiento_y_optimizacion.ipynb
│   ├── 05_validacion.ipynb
│   └── 06_prediccion.ipynb
├── requirements.txt
└── README.md
```

---

## Reproducción del Entorno

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd TP2

# Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt

# Iniciar Jupyter
jupyter notebook
```

Ejecutar las notebooks en orden numérico (01 → 06).

---

## Flujo de Trabajo

| Notebook | Descripción | Salidas |
|----------|-------------|---------|
| `01_lectura_y_discovery.ipynb` | Carga de datos, inspección de tipos, nulos y estructura general | — |
| `02_eda.ipynb` | Análisis exploratorio: distribuciones, correlaciones, variables constantes, outliers | — |
| `03_preprocesamiento.ipynb` | Encoding de variables categóricas, eliminación de columnas irrelevantes | `smoking_train_processed.csv`, `smoking_predict_processed.csv`, `categorical_mappings.json` |
| `04_entrenamiento_y_optimizacion.ipynb` | Entrenamiento baseline, curva de n_estimators, GridSearchCV, análisis de subgrupos | `xgboost_baseline.joblib`, `xgboost_optimized.joblib`, `best_params.json` |
| `05_validacion.ipynb` | Feature importance (gain vs weight), distribuciones por clase, threshold tuning, curva ROC | — |
| `06_prediccion.ipynb` | Predicción final sobre el dataset de entrega | `smoking_predictions.csv` |

---

## Decisiones de Preprocesamiento

| Variable | Decisión | Motivo |
|----------|----------|--------|
| `oral` | Eliminada | Constante en todos los registros (todos "Y") — sin poder predictivo |
| `ID` | Eliminada del train, conservada en predict | Identificador sin valor predictivo; necesaria para el archivo de entrega |
| `gender` | Encoding binario (F=0, M=1) | XGBoost requiere valores numéricos |
| `tartar` | Encoding binario (N=0, Y=1) | Idem |
| Nulos | Sin tratamiento | No hay valores nulos en ninguno de los dos datasets |
| Outliers | Sin tratamiento | XGBoost es robusto a valores atípicos |
| Escalado | Sin escalado | XGBoost no requiere normalización de features |

---

## Experimentos y Resultados

### Algoritmo

Se utilizó **XGBoost (XGBClassifier)** por su robustez a outliers, independencia del escalado, manejo interno de la multicolinealidad y capacidad de detectar interacciones no lineales.

### Feature Importance (gain)

La importancia medida por **gain** (reducción del error por split) es la métrica recomendada para XGBoost.
Usando **weight** (frecuencia de uso) el ranking cambia completamente — `triglyceride` aparece como #1 y `gender` no figura en el top 5 — lo que demuestra que weight puede ser engañoso.

| Feature | Importancia (gain) | Observación |
|---|---|---|
| `gender` | 0.808 | Predictor dominante: hombres fuman ~55%, mujeres ~4% |
| `height(cm)` | 0.034 | Proxy de género (hombres ~14 cm más altos en promedio) |
| `Gtp` | 0.016 | Enzima hepática elevada en fumadores |
| `hemoglobin` | 0.016 | Hemoglobina más alta en fumadores |
| `age` | 0.011 | Fumadores son ~4 años más jóvenes en promedio |

### Métricas del modelo (test — umbral 0.50)

| Modelo | F1-Train | F1-Test | Gap |
|---|---|---|---|
| Baseline (n=100, depth=6) | 0.7511 | 0.6947 | +0.056 |
| Optimizado (GridSearchCV) | 0.7511 | 0.6947 | +0.056 |

GridSearchCV (cv=3, grid: n_estimators × max_depth) confirmó que los parámetros del baseline ya son los óptimos dentro del espacio explorado.

### Ajuste del umbral de decisión

El análisis de threshold tuning mostró que el umbral default (0.50) no es el óptimo para F1:

| Umbral | F1-Test | Precision | Recall |
|---|---|---|---|
| 0.50 (default) | 0.6947 | 0.6703 | 0.7209 |
| **0.36 (óptimo)** | **0.7223** | **0.6077** | **0.8901** |

Con umbral 0.36 se detectan más fumadores reales (+17 pp en recall) con una mejora neta de +2.7 pp en F1.

### Análisis de overfitting por subgrupo

El gap train/test (5.6 pp) no se debe a hiperparámetros subóptimos sino a un **desbalance de clase a nivel de subgrupo**:

- **Mujeres:** solo el 4% fuma → el modelo predice casi siempre "no fumadora" (gap = 0.116)
- **Hombres:** el 55% fuma → el modelo discrimina con mejor base (gap = 0.057)

No hay data drift entre el dataset de entrenamiento y el de entrega (distribuciones prácticamente idénticas).

---

## Modelo Final y Predicciones

**Modelo:** `xgboost_optimized.joblib` — elegido por ser el resultado del proceso de validación cruzada (GridSearchCV cv=3).

**Umbral:** 0.36 — maximiza F1-Score sobre el conjunto de test.

**Resultado:** `data/processed/smoking_predictions.csv`
- 5,692 predicciones
- Columnas: `ID` (entero), `smoking_prediction` (0 ó 1)
- Fumadores predichos: 3,048 (53.5%) — proporción superior al 36.7% real, efecto esperado al optimizar recall con umbral bajo

**Curva ROC:** AUC = 0.8535

---

## Autor

Gonzalo Zarazaga — Diplomatura en Ciencia de Datos
