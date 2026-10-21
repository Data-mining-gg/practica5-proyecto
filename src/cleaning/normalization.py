import polars as pl
from config.rutas import RUTA_DATA_PROCESSED


def pipeline_normalizacion(df:pl.DataFrame, prop_entrenamiento:float=0.8) -> pl.DataFrame:
    df = df.sort("datetime")

    # Corte a proporción de columnas de entrenamiento
    corte_entrenamiento = int(df.height * prop_entrenamiento)
    df_entrenamiento = df.slice(0, corte_entrenamiento)
    
    # Columnas numéricas
    numericas = [
        col
        for col, dtype
        in df.schema.items()
        if
            dtype.is_numeric() and col != "datetime"
    ]
    
    # Cálculo Z-score
    expresiones = []
    for col in numericas:
        media_t = df_entrenamiento[col].mean()
        std_t = df_entrenamiento[col].std()
        
        expresiones.append(
            (
                (pl.col(col) - media_t)
                /
                std_t
            ).alias(col)
        )

    return df.with_columns(expresiones)
    

# Escritura de dataset en apartado de datos procesados
def escribir(df):
    df.write_csv(str(RUTA_DATA_PROCESSED / "normalized_consumption.csv"))

# Pipeline de extracción, normalización y escritura de dataset preprocesado:
def normalizacion_de_preprocesado(prop_entrenamiento:float=0.8):
    try:
        lectura_preprocesada = pl.read_csv(RUTA_DATA_PROCESSED / "preprocessed_consumption.csv")
        escribir(
            pipeline_normalizacion(lectura_preprocesada, prop_entrenamiento)
        )
    except FileNotFoundError:
        print("Error: No se encontró el dataset de valores preprocesados.")
