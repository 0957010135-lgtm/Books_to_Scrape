import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Carga de datos ────────────────────────────────────────────────────────────
df = pd.read_csv("books.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())  # verificar valores nulos


# ── Distribución de precios ───────────────────────────────────────────────────
plt.figure(figsize=(6, 4))
sns.histplot(df["price_gbp"], kde=True)
plt.title("Distribución del Precio de Libros")
plt.xlabel("Precio (GBP)")
plt.ylabel("Frecuencia")
plt.show()

# ── Distribución de calificaciones ───────────────────────────────────────────
plt.figure(figsize=(6, 4))
sns.countplot(x="rating", data=df)
plt.title("Distribución de Calificaciones")
plt.xlabel("Calificación (estrellas)")
plt.ylabel("Cantidad de libros")
plt.show()

# ── Distribución de libros por género ────────────────────────────────────────
plt.figure(figsize=(10, 8))
genre_counts = df["genre"].value_counts()
sns.barplot(x=genre_counts.values, y=genre_counts.index)
plt.title("Cantidad de Libros por Género")
plt.xlabel("Cantidad")
plt.ylabel("Género")
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# ── Precio promedio por género ────────────────────────────────────────────────
plt.figure(figsize=(10, 8))
genre_price = df.groupby("genre")["price_gbp"].mean().sort_values(ascending=False)
sns.barplot(x=genre_price.values, y=genre_price.index)
plt.title("Precio Promedio por Género")
plt.xlabel("Precio Promedio (GBP)")
plt.ylabel("Género")
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# ── Matriz de correlación ─────────────────────────────────────────────────────
# price_usd se excluye porque es transformación directa de price_gbp (correlación exacta de 1.0)
# n_reviews se excluye porque es constante en el dataset (siempre 0)
# in_stock aparecerá en blanco si todos los libros están en stock (valor constante = sin variación)
df_corr = df[["price_gbp", "rating", "stock_count", "in_stock"]].copy()
df_corr["in_stock"] = df_corr["in_stock"].astype(int)
plt.figure(figsize=(7, 5))
sns.heatmap(df_corr.corr(), annot=True, cmap="Blues", vmin=-1, vmax=1, annot_kws={"size": 9})
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title("Matriz de Correlación")
plt.show()

# ── Distribución de precio por calificación ───────────────────────────────────
# boxplot por rating: muestra la distribución de precios dentro de cada calificación
plt.figure(figsize=(6, 4))
sns.boxplot(x="rating", y="price_gbp", data=df)
plt.title("Distribución de Precio por Calificación")
plt.xlabel("Calificación (estrellas)")
plt.ylabel("Precio (GBP)")
plt.show()

# ── Precio promedio por segmento ──────────────────────────────────────────────
# si luxury no aparece es porque no hay libros con precio >£60 en el dataset
plt.figure(figsize=(6, 4))
sns.barplot(x="price_tier", y="price_gbp", data=df, order=["budget", "mid-range", "premium", "luxury"])
plt.title("Precio Promedio por Segmento")
plt.xlabel("Segmento")
plt.ylabel("Precio Promedio (GBP)")
plt.show()

# ── Detección de valores atípicos en precio ───────────────────────────────────
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["price_gbp"])
plt.title("Detección de Outliers en Precio")
plt.show()

'''
Análisis                        Hallazgo esperado
Distribución de precios         Precios distribuidos de forma relativamente uniforme
Distribución de calificaciones  Todas las calificaciones representadas en proporciones similares
Libros por género               Algunos géneros con mayor cantidad de títulos disponibles
Precio por género               Variación de precio entre géneros — variable con potencial predictivo
Correlación                     Baja correlación entre rating y price_gbp
Boxplot por calificación        Sin diferencia significativa de precio entre calificaciones
Precio por segmento             Cada segmento refleja su rango de precio definido
Outliers                        Posibles valores atípicos en el extremo superior del precio
'''


# ════════════════════════════════════════════════════════════════════════════════
# MODELO DE PREDICCIÓN — REGRESIÓN LINEAL Y RANDOM FOREST
# ════════════════════════════════════════════════════════════════════════════════

# codificación de género mediante one-hot encoding para uso en modelos
# n_reviews se excluye del modelo por ser constante (siempre 0 en el dataset)
df_model = pd.get_dummies(df[["rating", "stock_count", "in_stock", "genre"]], columns=["genre"])
df_model["in_stock"] = df_model["in_stock"].astype(int)

X = df_model
y = df["price_gbp"]

# división en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Regresión Lineal ──────────────────────────────────────────────────────────
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

mae_lr  = mean_absolute_error(y_test, y_pred_lr)
mse_lr  = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr   = r2_score(y_test, y_pred_lr)

print("\n── Regresión Lineal ──────────────────────────")
print("MAE:", mae_lr)
print("MSE:", mse_lr)
print("RMSE:", rmse_lr)
print("R2:", r2_lr)

# ── Random Forest ─────────────────────────────────────────────────────────────
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

mae_rf  = mean_absolute_error(y_test, y_pred_rf)
mse_rf  = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf   = r2_score(y_test, y_pred_rf)

print("\n── Random Forest ─────────────────────────────")
print("MAE:", mae_rf)
print("MSE:", mse_rf)
print("RMSE:", rmse_rf)
print("R2:", r2_rf)

'''
Métrica     Interpretación
MAE         Error promedio absoluto en GBP
RMSE        Error promedio penalizado por valores extremos
R2          Qué tan bien el modelo explica la variación del precio (1 es perfecto)

Se comparan ambos modelos para evaluar cuál captura mejor la relación
entre las variables predictoras y el precio.
Random Forest tolera mejor variables mixtas y relaciones no lineales.
'''

# ── Gráfico real vs predicho — Regresión Lineal ───────────────────────────────
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_lr)
plt.title("Precio Real vs Predicho — Regresión Lineal")
plt.xlabel("Real (GBP)")
plt.ylabel("Predicho (GBP)")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red")
plt.show()

# ── Gráfico real vs predicho — Random Forest ─────────────────────────────────
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_rf)
plt.title("Precio Real vs Predicho — Random Forest")
plt.xlabel("Real (GBP)")
plt.ylabel("Predicho (GBP)")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red")
plt.show()

# ── Importancia de variables — Random Forest ──────────────────────────────────
# muestra qué variables tienen mayor influencia en la predicción del precio
feature_importances = pd.Series(model_rf.feature_importances_, index=X.columns)
top_features = feature_importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Importancia de Variables — Random Forest")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

'''
Books to Scrape es un sitio sintético de práctica — los precios no siguen un patrón real,
lo que explica el R2 bajo en ambos modelos independientemente del algoritmo utilizado.
Si Random Forest supera a Regresión Lineal, confirma que la relación no es lineal.
La gráfica de importancia de variables revela qué atributos
(género, stock_count, calificación) tienen mayor influencia relativa sobre el precio.
'''