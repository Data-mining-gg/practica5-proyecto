import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Filtrado de columnas numéricas
def obtener_numericas(df:pl.DataFrame) -> pl.DataFrame:
    return df.select([
        col
        for col, dtype
        in df.schema.items() 
        if
            dtype.is_numeric() and col != "datetime"
    ])

# Matriz de correlación
def calcular_matriz_pearson(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas().corr(method="pearson")

# Graficación del mapa de calor
def graficar_heatmap(matriz_corr: pd.DataFrame, titulo: str = "Matriz de Correlación de Pearson"):
    """
    Genera un mapa de calor (Heatmap) con Seaborn a partir de la matriz de correlación.
    """
    plt.figure(figsize=(9,7))
    sns.heatmap(
        matriz_corr, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        linewidths=0.8,
        square=True
    )
    plt.title(titulo, fontsize=13, pad=15)
    plt.tight_layout()
    plt.show()