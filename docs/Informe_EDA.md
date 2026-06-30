# Análisis Exploratorio de Datos y Modelo Supervisado

Archivo: `eda_y_modelo_supervisado.py`


## 1. Introducción

El presente informe documenta el análisis exploratorio de datos (EDA) y la implementación de un modelo de aprendizaje supervisado sobre el dataset obtenido a partir del scraping de Books to Scrape. El conjunto de datos contiene 999 registros y 11 columnas, incluyendo variables numéricas, categóricas y booleanas.

El objetivo es explorar la distribución y relaciones entre las variables disponibles, formular una hipótesis de predicción y evaluar el desempeño de dos algoritmos de aprendizaje supervisado.


## 2. Desarrollo de la Actividad

### 2.1 Hipótesis

Se planteó la siguiente hipótesis de trabajo:

> El precio de un libro puede predecirse a partir de su calificación, género, número de reseñas y cantidad de unidades disponibles en stock.

Esta hipótesis se evalúa mediante técnicas de aprendizaje supervisado, dado que el conjunto de datos cuenta con una variable objetivo claramente definida (`price_gbp`), lo cual descarta el uso de algoritmos no supervisados como clustering.

### 2.2 Herramientas y Tecnologías Utilizadas

- **pandas** – carga del dataset y manipulación del DataFrame.
- **matplotlib** – generación de gráficos base.
- **seaborn** – visualizaciones estadísticas.
- **numpy** – cálculo de métricas de evaluación.
- **scikit-learn** – implementación de los modelos supervisados y métricas de desempeño.

### 2.3 Exploración de Datos

Se realizó una exploración inicial mediante estadísticas descriptivas (`head()`, `info()`, `describe()`) y verificación de valores nulos, confirmando que el dataset no contiene registros incompletos.

Las visualizaciones generadas incluyen:

- **Distribución de precios** — histograma con curva de densidad para observar la dispersión general de `price_gbp`.
- **Distribución de calificaciones** — conteo de libros por cada nivel de rating (1 a 5 estrellas).
- **Cantidad de libros por género** — distribución de los 50 géneros presentes en el catálogo.
- **Precio promedio por género** — comparación del precio medio entre géneros.
- **Matriz de correlación** — relación lineal entre `price_gbp`, `rating`, `stock_count` e `in_stock`.
- **Distribución de precio por calificación** — boxplot que compara la dispersión de precios dentro de cada nivel de rating.
- **Precio promedio por segmento** — comparación entre los segmentos `budget`, `mid-range`, `premium` y `luxury`.
- **Detección de outliers** — boxplot general sobre `price_gbp`.

### 2.4 Variables Excluidas del Modelo

Durante el análisis se identificaron dos atributos sin valor predictivo:

- **`n_reviews`** — todos los registros presentan un valor constante de 0, por lo que fue excluida tanto de la matriz de correlación como del modelo supervisado.
- **`in_stock`** — todos los libros del catálogo se encuentran en stock, generando una variable sin variación. Se mantuvo en la matriz de correlación como referencia, aunque su celda puede aparecer sin valor numérico al no existir desviación estándar calculable.

Ambas variables se conservan en el CSV y en el código por consistencia — si el catálogo cambiara en el futuro (libros agotados, reseñas reales), podrían volverse relevantes sin necesidad de modificar el scraper.

### 2.5 Selección y Justificación del Modelo

Se evaluaron dos algoritmos de aprendizaje supervisado para la tarea de regresión:

- **Regresión Lineal** — modelo base, adecuado cuando se espera una relación lineal entre las variables predictoras y la variable objetivo.
- **Random Forest Regressor** — modelo de ensamble basado en árboles de decisión, capaz de capturar relaciones no lineales y manejar variables categóricas codificadas sin asumir linealidad.

Como variables predictoras se utilizaron `rating`, `stock_count`, `in_stock` y `genre` (codificada mediante one-hot encoding). El conjunto de datos se dividió en 80% para entrenamiento y 20% para prueba con `random_state=42` para garantizar reproducibilidad.

La comparación entre ambos modelos permite determinar si la relación entre las variables disponibles y el precio responde mejor a un enfoque lineal o no lineal.

### 2.6 Resultados

Las métricas obtenidas para ambos modelos sobre el conjunto de prueba fueron las siguientes:

#### Regresión Lineal

- MAE: 12.28
- RMSE: 14.34
- R²: -0.029

#### Random Forest Regressor

- MAE: 13.50
- RMSE: 16.39
- R²: -0.343

Ambos modelos presentan un coeficiente de determinación (R²) negativo, lo cual indica que ninguno logra explicar la variabilidad del precio mejor de lo que lo haría simplemente predecir el promedio general de `price_gbp` para todos los registros.

Adicionalmente, Regresión Lineal obtuvo un desempeño ligeramente superior a Random Forest, resultado inusual que sugiere sobreajuste del segundo modelo dado el bajo número de variables con verdadero poder predictivo.

Este resultado constituye un hallazgo relevante sobre la naturaleza del conjunto de datos. Books to Scrape es un sitio diseñado específicamente para practicar técnicas de web scraping, por lo que sus precios son generados de forma sintética y no responden a un modelo de negocio real. En consecuencia, no existe una relación genuina entre atributos como la calificación, el género o las unidades en stock y el precio asignado a cada libro.

La gráfica de importancia de variables del modelo Random Forest refuerza esta conclusión: ninguna variable individual concentra una proporción dominante de importancia, lo cual es consistente con la ausencia de una relación estructural fuerte entre los predictores y la variable objetivo.

```
Métrica     Interpretación
MAE         Error promedio absoluto en GBP
RMSE        Error promedio penalizado por valores extremos
R²          Qué tan bien el modelo explica la variación del precio (1 es perfecto)
```


## 3. Conclusión

El desarrollo de este componente del proyecto permitió aplicar correctamente el flujo de trabajo de aprendizaje supervisado: formulación de hipótesis, selección de variables, codificación de atributos categóricos, entrenamiento de dos modelos distintos y evaluación comparativa mediante métricas estándar.

Si bien los resultados numéricos no alcanzan un desempeño predictivo satisfactorio, el proceso demuestra la capacidad de identificar, documentar e interpretar correctamente las limitaciones inherentes a un conjunto de datos sintético. El R² bajo no es un error del análisis — es un hallazgo válido que evidencia que el precio en Books to Scrape se asigna de forma independiente a cualquier atributo del libro.

Como trabajo futuro, sería recomendable replicar este mismo flujo sobre un conjunto de datos con relaciones de precio reales, donde variables como género, calificación o disponibilidad sí podrían tener una influencia significativa y permitir un modelo con mayor capacidad predictiva.
