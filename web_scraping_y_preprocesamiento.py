import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_URL  = "https://books.toscrape.com/catalogue/"
START_URL = "https://books.toscrape.com/catalogue/page-1.html"
DELAY_SEC = 0.5      # pausa entre peticiones para un scraping responsable
MAX_PAGES = None     # asignar un entero (p. ej. 5) para limitar páginas

RATING_MAP = {
    "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5
}


# ════════════════════════════════════════════════════════════════════════════════
# SCRAPING
# ════════════════════════════════════════════════════════════════════════════════

def get_soup(url: str) -> BeautifulSoup:
    """Realiza una petición HTTP a la URL indicada y retorna un objeto BeautifulSoup."""
    headers  = {"User-Agent": "Mozilla/5.0 (BooksScraper/1.0)"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def scrape_book(article) -> dict:
    """Extrae los campos en bruto de un elemento <article> correspondiente a un libro."""
    title   = article.h3.a["title"]
    price   = article.find("p", class_="price_color").text.strip()
    rating  = article.p["class"][1]          # p. ej. "Three"
    avail   = article.find("p", class_="instock availability").text.strip()
    rel_url = article.h3.a["href"]
    url     = BASE_URL + rel_url.replace("../", "")

    return {
        "title":        title,
        "price_raw":    price,
        "rating_word":  rating,
        "availability": avail,
        "url":          url,
    }


def scrape_detail(url: str) -> dict:
    """
    Accede a la página de detalle de un libro y extrae variables adicionales:
    género, número de reseñas y cantidad exacta de unidades en stock.
    """
    soup = get_soup(url)

    # género desde el breadcrumb (tercer elemento: Home > Categoría > Libro)
    breadcrumb = soup.select("ul.breadcrumb li")
    genre = breadcrumb[-2].text.strip() if len(breadcrumb) >= 2 else "Unknown"

    # tabla de información del producto
    table = {}
    for row in soup.select("table.table tr"):
        key   = row.th.text.strip()
        value = row.td.text.strip()
        table[key] = value

    # número de reseñas
    n_reviews = int(table.get("Number of reviews", 0))

    # unidades exactas en stock desde el texto de disponibilidad
    avail_text  = table.get("Availability", "")
    stock_match = re.search(r"\d+", avail_text)
    stock_count = int(stock_match.group()) if stock_match else 0

    return {
        "genre":       genre,
        "n_reviews":   n_reviews,
        "stock_count": stock_count,
    }


def scrape_all_books(start_url: str, max_pages=None) -> list[dict]:
    """
    Recorre todas las páginas del catálogo, recopila los campos en bruto
    de cada libro y accede a su página de detalle para extraer variables adicionales.
    """
    books = []
    url   = start_url
    page  = 1

    while url:
        print(f"  Extrayendo página {page} …")
        soup     = get_soup(url)
        articles = soup.find_all("article", class_="product_pod")

        for article in articles:
            book = scrape_book(article)
            time.sleep(DELAY_SEC)
            detail = scrape_detail(book["url"])
            book.update(detail)
            books.append(book)

        # seguir el botón "next" de paginación
        next_btn = soup.find("li", class_="next")
        if next_btn and (max_pages is None or page < max_pages):
            url  = BASE_URL + next_btn.a["href"]
            page += 1
            time.sleep(DELAY_SEC)
        else:
            url = None

    print(f"  Total de libros extraídos: {len(books)}")
    return books


# ── Ejecución del scraping ────────────────────────────────────────────────────
print("\n[1/4] Extrayendo datos …")
raw_records = scrape_all_books(START_URL, max_pages=MAX_PAGES)
df = pd.DataFrame(raw_records)
print(f"      Dimensiones en bruto: {df.shape}")


# ════════════════════════════════════════════════════════════════════════════════
# KDD — SELECCIÓN · LIMPIEZA · TRANSFORMACIÓN
# ════════════════════════════════════════════════════════════════════════════════

# ── Selección ─────────────────────────────────────────────────────────────────
print("[2/4] Seleccionando columnas …")

# conservar únicamente los atributos relevantes para el análisis
df = df[["title", "price_raw", "rating_word", "availability", "genre", "n_reviews", "stock_count", "url"]].copy()


# ── Limpieza ──────────────────────────────────────────────────────────────────
print("[3/4] Limpiando datos …")

# eliminar filas con campos críticos ausentes
df.dropna(subset=["title", "price_raw"], inplace=True)

# eliminar títulos duplicados, conservando la primera ocurrencia
df.drop_duplicates(subset=["title"], keep="first", inplace=True)

# limpiar precio: quitar símbolo de moneda y convertir a float
df["price_raw"] = df["price_raw"].astype(str).str.strip()
df["price_raw"] = df["price_raw"].apply(lambda x: float(re.sub(r"[^\d.]", "", x)))

# normalizar espacios en el campo de disponibilidad
df["availability"] = df["availability"].apply(lambda x: " ".join(x.split()))

# normalizar género: strip de espacios
df["genre"] = df["genre"].str.strip()

print(f"      Dimensiones tras limpieza: {df.shape}")


# ── Transformación ────────────────────────────────────────────────────────────
print("[4/4] Transformando datos …")

# renombrar precio para reflejar la unidad de moneda correcta
df.rename(columns={"price_raw": "price_gbp"}, inplace=True)

# codificar calificación en texto a entero (1–5)
df["rating"] = df["rating_word"].map(RATING_MAP)
df.drop(columns=["rating_word"], inplace=True)

# segmento de precio según rangos definidos
df["price_tier"] = pd.cut(
    df["price_gbp"],
    bins=[0, 20, 40, 60, float("inf")],
    labels=["budget", "mid-range", "premium", "luxury"],
    right=True,
)

# variable booleana de disponibilidad en stock
df["in_stock"] = df["availability"].str.lower().str.contains("in stock")

# precio convertido a USD (aproximado)
df["price_usd"] = (df["price_gbp"] * 1.27).round(2)

# orden final de columnas
df = df[[
    "title", "rating", "price_gbp", "price_usd", "price_tier",
    "in_stock", "stock_count", "n_reviews", "genre", "availability", "url"
]]


# ── Exportación ───────────────────────────────────────────────────────────────
df.to_csv("books.csv", index=False, encoding="utf-8")
print(f"\n  {len(df)} registros guardados → books.csv")