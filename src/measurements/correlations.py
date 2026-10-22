import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config.rutas import RUTA_DATA_PROCESSED


# Filtrado de columnas numéricas
def obtener_numericas(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        [
            col
            for col, dtype in df.schema.items()
            if dtype.is_numeric() and col != "datetime"
        ]
    )


# Matriz de correlación
def matriz_correlaciones(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas().corr(method="pearson")


# Graficación del mapa de calor
def graficar_heatmap(
    matriz_corr: pd.DataFrame, titulo: str = "Matriz de Correlación de Pearson"
):
    """
    Genera un mapa de calor (Heatmap) con Seaborn a partir de la matriz de correlación.
    """
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matriz_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        square=True,
    )
    plt.title(titulo, fontsize=13, pad=15)
    plt.tight_layout()
    plt.show()


# Pares de variables cuya correlación absoluta supera el umbral
def pares_redundantes(matriz_corr: pd.DataFrame, umbral: float = 0.85) -> pl.DataFrame:
    columnas = list(matriz_corr.columns)
    pares = [
        (columnas[i], columnas[j], float(matriz_corr.iloc[i, j]))
        for i in range(len(columnas))
        for j in range(i + 1, len(columnas))
        if abs(matriz_corr.iloc[i, j]) >= umbral
    ]

    return pl.DataFrame(
        pares, schema=["variable_1", "variable_2", "pearson"], orient="row"
    )


# Descarte de variables redundantes
def descartar_redundantes(
    df: pl.DataFrame, columnas: tuple[str, ...] = ("global_intensity",)
) -> pl.DataFrame:
    return df.drop(list(columnas))


# Pipeline completo de generación de mapa de calor de dataset normalizado
def graficar_cuantitativas_de_normalizado():
    lectura_normalizada = pl.read_csv(
        RUTA_DATA_PROCESSED / "normalized_consumption.csv"
    )
    graficar_heatmap(matriz_correlaciones(obtener_numericas(lectura_normalizada)))
