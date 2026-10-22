import polars as pl
from config.rutas import RUTA_DATA_PROCESSED
from src.measurements.correlations import descartar_redundantes
from src.reduction.pca import (
    ajustar_pca,
    seleccionar_componentes,
    incorporar_componentes,
)


def pipeline_consolidacion(
    df: pl.DataFrame, prop_entrenamiento: float = 0.8, umbral_varianza: float = 0.8
) -> pl.DataFrame:
    df_reducido = descartar_redundantes(df)

    pca = ajustar_pca(df_reducido, prop_entrenamiento)
    n_componentes = seleccionar_componentes(pca, umbral_varianza)

    return incorporar_componentes(df_reducido, pca, n_componentes)


def escribir(df):
    df.write_csv(str(RUTA_DATA_PROCESSED / "prepared_consumption.csv"))


def consolidacion_de_normalizado(
    prop_entrenamiento: float = 0.8, umbral_varianza: float = 0.8
):
    try:
        lectura_normalizada = pl.read_csv(
            RUTA_DATA_PROCESSED / "normalized_consumption.csv", try_parse_dates=True
        )
        escribir(
            pipeline_consolidacion(
                lectura_normalizada, prop_entrenamiento, umbral_varianza
            )
        )
    except FileNotFoundError:
        print("Error: No se encontró el dataset de valores normalizados.")
