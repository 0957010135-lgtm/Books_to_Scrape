# Web Scraper: Books to Scrape

Archivo: `web_scraping_y_preprocesamiento.py`


## 1. Introducción

En el campo de la Minería de Datos y el Descubrimiento de Conocimiento en Bases de Datos (KDD), la recolección automatizada de datos web constituye una etapa fundamental del proceso. El presente informe documenta el desarrollo de un web scraper orientado a la extracción, limpieza y transformación de datos del sitio Books to Scrape (https://books.toscrape.com), un portal de prueba diseñado específicamente para la práctica del web scraping.

El proceso implementado sigue las fases clásicas del modelo KDD:

- **Scraping:** extracción automatizada de registros de libros desde todas las páginas del catálogo y sus páginas de detalle.
- **Selección:** identificación de los atributos relevantes para el análisis.
- **Limpieza:** corrección de inconsistencias y estandarización de valores.
- **Transformación:** derivación de nuevas variables y codificación de campos para facilitar el análisis.
- **Exportación:** volcado del conjunto de datos final a formato CSV mediante la librería Pandas.

El objetivo principal es obtener un dataset estructurado y limpio que sirva de base para etapas posteriores de análisis exploratorio y modelado supervisado.


## 2. Desarrollo de la Actividad

### 2.1 Herramientas y Tecnologías Utilizadas

El scraper fue implementado en Python 3, apoyándose en las siguientes librerías:

- **requests** – envío de peticiones HTTP para obtener el contenido HTML de cada página.
- **BeautifulSoup (bs4)** – análisis y navegación del árbol DOM para extraer los datos de cada libro.
- **pandas** – construcción del DataFrame, aplicación de transformaciones y exportación a CSV.
- **re** – limpieza del campo de precio y extracción de unidades en stock mediante expresiones regulares.
- **time** – pausa configurable entre peticiones (crawl delay) para un scraping responsable.

### 2.2 Fase 1 – Scraping

La función `scrape_all_books()` recorre iterativamente todas las páginas del catálogo siguiendo el botón "next" de paginación. Por cada página, `get_soup()` realiza la petición HTTP y devuelve el objeto BeautifulSoup. Para cada elemento `<article class="product_pod">` se extraen los campos en bruto mediante `scrape_book()`:

- **title** – título completo desde el atributo `[title]` del enlace.
- **price_raw** – precio con símbolo de moneda (p. ej. `£51.77`).
- **rating_word** – calificación en texto (`"One"` a `"Five"`) desde la clase CSS del elemento `<p>`.
- **availability** – texto de disponibilidad (`"In stock"`).
- **url** – URL absoluta de la página de detalle del libro.

Adicionalmente, por cada libro se accede a su página de detalle mediante `scrape_detail()` para extraer variables con mayor potencial analítico:

- **genre** – género del libro, obtenido desde el breadcrumb de navegación.
- **n_reviews** – número de reseñas, extraído desde la tabla de información del producto.
- **stock_count** – cantidad exacta de unidades en stock, derivada del texto de disponibilidad mediante expresión regular.

```python
for article in articles:
    book = scrape_book(article)
    detail = scrape_detail(book["url"])
    book.update(detail)
    books.append(book)
```

### 2.3 Fase 2 – Selección

Se conservan únicamente los ocho atributos relevantes para el análisis, descartando cualquier metadato interno generado durante el scraping:

```python
df = df[["title", "price_raw", "rating_word", "availability",
         "genre", "n_reviews", "stock_count", "url"]].copy()
```

### 2.4 Fase 3 – Limpieza

Las siguientes operaciones se aplican sobre el DataFrame:

1. Eliminación de filas con valores nulos en `title` o `price_raw` mediante `dropna()`.
2. Eliminación de títulos duplicados con `drop_duplicates()`, conservando la primera ocurrencia.
3. Limpieza del precio: eliminación del símbolo `£` y conversión a `float` mediante expresión regular `[^\d.]`.
4. Normalización de espacios en `availability` y `genre`.

```python
df["price_raw"] = df["price_raw"].apply(lambda x: float(re.sub(r"[^\d.]", "", x)))
df["availability"] = df["availability"].apply(lambda x: " ".join(x.split()))
df["genre"] = df["genre"].str.strip()
```

### 2.5 Fase 4 – Transformación

La etapa de transformación enriquece el dataset con las siguientes operaciones:

1. **Renombrado:** `price_raw` → `price_gbp` para reflejar la unidad de moneda correcta.
2. **Codificación de rating:** mapeo del texto a entero (`"Three"` → `3`) mediante `RATING_MAP`.
3. **Segmentación de precio** en cuatro niveles usando `pd.cut()`:

#### Niveles de price_tier

- `budget` — hasta £20
- `mid-range` — £20 a £40
- `premium` — £40 a £60
- `luxury` — más de £60

4. **Variable booleana** `in_stock` derivada del texto de disponibilidad.
5. **Conversión de precio a USD:** `price_usd = price_gbp × 1.27`, redondeado a 2 decimales.

```python
df["price_tier"] = pd.cut(
    df["price_gbp"],
    bins=[0, 20, 40, 60, float("inf")],
    labels=["budget", "mid-range", "premium", "luxury"]
)
df["in_stock"] = df["availability"].str.lower().str.contains("in stock")
df["price_usd"] = (df["price_gbp"] * 1.27).round(2)
```

### 2.6 Fase 5 – Exportación

El DataFrame final es exportado a `books.csv` con codificación UTF-8 e índice suprimido. El archivo contiene las siguientes columnas definitivas:

- `title`, `rating`, `price_gbp`, `price_usd`, `price_tier`, `in_stock`, `stock_count`, `n_reviews`, `genre`, `availability`, `url`

```python
df.to_csv("books.csv", index=False, encoding="utf-8")
```


## 3. Conclusión

El desarrollo del pipeline KDD sobre el portal Books to Scrape permitió consolidar de forma práctica las etapas teóricas del proceso de Descubrimiento de Conocimiento en Bases de Datos. A través de la implementación en Python se construyó un flujo de trabajo reproducible y modular que transita desde la recolección automatizada de datos hasta la obtención de un dataset analítico limpio y enriquecido.

La extensión del scraper para acceder a las páginas de detalle de cada libro — capturando género, número de reseñas y cantidad exacta de unidades en stock — incrementó el tiempo de ejecución pero amplió significativamente el conjunto de variables disponibles para etapas de análisis posteriores.

Las fases de limpieza y transformación pusieron en evidencia la importancia de preparar adecuadamente los datos antes de cualquier análisis: la conversión de tipos, la eliminación de duplicados y la derivación de variables como `price_tier`, `in_stock` y `price_usd` incrementaron el poder descriptivo del dataset resultante.

Finalmente, el uso de `pandas` como eje central del pipeline facilitó tanto las operaciones de transformación como la exportación al formato CSV, ampliamente compatible con herramientas de análisis y visualización de datos.
