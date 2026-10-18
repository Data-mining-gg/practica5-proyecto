import polars as pl
from config.rutas import RUTA_DATA_RAW, RUTA_DATA_PROCESSED


def pipeline_normalizacion(df:pl.DataFrame) -> pl.DataFrame:
    numericas = [
        col
        for col, dtype
        in df.schema.items()
        if
            dtype.is_numeric() and col != "datetime"
    ]
    
    # Cálculo Z-score
    df_estandarizado = df.with_columns([
        (
            (pl.col(col) - pl.col(col).mean())
            / pl.col(col).std()
        )
            .alias(col)
        for col
        in numericas
    ])
    
    return df_estandarizado

# Escritura de dataset en apartado de datos procesados
def escribir(df):
    df.write_csv(str(RUTA_DATA_PROCESSED / "normalized_consumption.csv"))

# Pipeline de extracción, normalización y escritura de dataset preprocesado:
try:
    lectura_preprocesada = pl.read_csv(RUTA_DATA_PROCESSED / "preprocessed_consumption.csv")
    escribir(
        pipeline_normalizacion(lectura_preprocesada)
    )
except FileNotFoundError:
    print("Error: No se encontró el dataset de valores preprocesados.")
