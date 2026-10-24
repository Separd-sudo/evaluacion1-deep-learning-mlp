# Evaluación Parcial N.º 1 — Fundamentos de Deep Learning (DLY0100)

## Red Neuronal Multicapa (MLP) para Clasificación: Caso de Estudio Bank Marketing

Este repositorio contiene la solución completa, operacional y documentada de la Evaluación Parcial N.º 1 de la asignatura **Fundamentos de Deep Learning (DLY0100)**. Se implementa y optimiza una arquitectura de **Perceptrón Multicapa (MLP)** construida con **TensorFlow/Keras** para predecir si un cliente bancario contratará un depósito a plazo fijo (`y = "yes"` o `y = "no"`).

---

## 1. Objetivo del Proyecto

Construir un clasificador basado en redes neuronales profundas que:
* **Evite fugas de información (*Data Leakage*):** Excluyendo variables posteriores al contacto (como `duration`) y realizando el escalamiento numérico exclusivamente con datos de entrenamiento.
* **Maneje el desbalance severo de clases:** Empleando ponderación de clases (`class_weight`), métricas orientadas al negocio (*Precision*, *Recall*, *F1-Score*) y optimización del umbral de decisión.
* **Evalúe empíricamente hiperparámetros y arquitecturas:** Comparando tasas de aprendizaje, tamaños de lote, número de épocas y profundidad de capas bajo el principio *ceteris paribus*.
* **Mitigue el sobreajuste:** Evaluando la regularización conjunta (*Batch Normalization*, *Dropout*, penalización *L2*) mediante *Binary Crossentropy* comparable y en modo inferencia.

---

## 2. Dataset Utilizado

* **Nombre:** *Bank Marketing Data Set* (UCI Machine Learning Repository).
* **Dimensiones:** 45.211 registros y 17 columnas (7 numéricas, 9 categóricas y la variable objetivo `y`).
* **Desbalance de clase:** 88.30% clientes `"no"` (39.922) vs. 11.70% clientes `"yes"` (5.289).
* **Partición Estratificada:**
  * **Entrenamiento (70%):** 31.647 registros (11.70% `"yes"`).
  * **Validación (15%):** 6.782 registros (11.71% `"yes"`).
  * **Prueba (15%):** 6.782 registros (11.69% `"yes"`).

---

## 3. Dependencias Necesarias

El proyecto utiliza Python 3.10+ y las siguientes librerías científicas y de aprendizaje profundo:

* `tensorflow >= 2.16.0`
* `keras >= 3.0.0`
* `pandas >= 2.0.0`
* `numpy >= 1.24.0`
* `scikit-learn >= 1.3.0`
* `matplotlib >= 3.7.0`
* `jupyter` / `ipykernel`

---

## 4. Cómo Abrir y Ejecutar el Notebook

### Opción A: Google Colab (Recomendada para ejecución rápida en la nube)
1. Abre el notebook directamente en Google Colab haciendo clic en el badge superior del archivo o a través del enlace del repositorio.
2. La celda de la **Opción A** descarga y descomprime automáticamente el dataset oficial desde el repositorio UCI.
3. Ejecuta las celdas secuencialmente desde el menú: **Entorno de ejecución → Ejecutar todas**.

### Opción B: Entorno Local (Jupyter Notebook / VS Code)
1. Clona el repositorio:
   ```bash
   git clone https://github.com/Separd-sudo/evaluacion1-deep-learning-mlp.git
   cd evaluacion1-deep-learning-mlp
   ```
2. Crea y activa un entorno virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # En Linux/macOS
   # .venv\Scripts\activate     # En Windows
   ```
3. Instala las dependencias requeridas:
   ```bash
   pip install pandas numpy scikit-learn matplotlib tensorflow jupyter ipykernel
   ```
4. Asegúrate de tener el archivo `bank_marketing.csv` en la raíz del proyecto (o utiliza la descarga automática integrada en el notebook).
5. Inicia Jupyter Notebook o JupyterLab:
   ```bash
   jupyter notebook Evaluacion_MLP_Dataset_BankMarketing.ipynb
   ```
6. Ejecuta todas las celdas en orden de arriba a abajo.

---

## 5. Estructura de Fases y Resultados Obtenidos

El notebook está organizado en 7 fases metodológicas rigurosas:

1. **Fase 1 y 2 (Preprocesamiento y División):** Exclusión de `duration`, transformación de `pdays` en variable indicadora `contactado_antes`, *One-Hot Encoding* de variables categóricas, escalado estandarizado sin fuga y partición estratificada 70/15/15.
2. **Fase 3 (Modelo Base MLP):** Arquitectura base (64 y 32 neuronas ReLU, salida Sigmoide, optimizador Adam `lr=0.001`, pérdida *Binary Crossentropy* y `class_weight`).
3. **Fase 4 (Experimentos Controlados de Hiperparámetros):**
   * *Tasa de aprendizaje:* `lr = 0.001` logra el mejor compromiso de convergencia y estabilidad (`F1 = 0.4216`, `Val Loss = 0.5015`).
   * *Tamaño de lote:* `batch_size = 256` estabiliza el descenso de gradiente (`F1 = 0.4364`, `Val Loss = 0.4820`).
   * *Número de épocas:* `12` épocas alcanza el óptimo (`F1 = 0.4328`, `Val Loss = 0.4793`), evitando la divergencia observada al extender a 25 épocas.
   * *Profundidad MLP:* La red de 2 capas ocultas supera a la de 1 capa y previene el sobreajuste de la red de 3 capas (`Val Loss = 0.4755` vs. `0.5062`).
4. **Fase 5 (Activación y Pérdida):** Comparación sistemática de ReLU, Tanh, LeakyReLU, Binary Crossentropy, MSE y Focal Loss. `ReLU + BCE` y `LeakyReLU + BCE` ofrecen las fronteras más sólidas y consistentes.
5. **Fase 6 (Regularización y Optimización):** Incorporación de *Batch Normalization*, *Dropout* (0.3 y 0.2) y regularización *L2*. Medición con *Binary Crossentropy* no ponderada en modo inferencia, reduciendo la brecha entre validación y entrenamiento de `0.0249` a `0.0055`.
6. **Fase 7 (Evaluación Final en Test y Umbral de Decisión):**
   * **Baseline mayoritario:** Accuracy engañoso de `88.31%` con Recall de `0.00%` (cero clientes captados).
   * **Modo Cobertura (`u = 0.50`):** Recall de `64.44%` en datos de prueba no vistos (detecta 2 de cada 3 ventas potenciales).
   * **Modo Eficiencia (`u = 0.68`, optimizado en validación):** Precision de `48.17%`, F1-Score de `0.4817` y Accuracy de `87.88%`, cuadruplicando la tasa de conversión base del banco (~11.69%) y optimizando los recursos del call center.

---

## 6. Autores y Licencia

* **Asignatura:** Fundamentos de Deep Learning (DLY0100).
* **Evaluación:** Evaluación Parcial N.º 1.
* **Licencia del Dataset:** Creative Commons Attribution 4.0 International (CC BY 4.0).
