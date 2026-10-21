import polars as pl
from config.rutas import RUTA_DATA_RAW, RUTA_DATA_PROCESSED
from pathlib import Path

def lectura_dataset_raw() -> pl.DataFrame:
    return pl.read_csv(
        RUTA_DATA_RAW / "household_power_consumption.txt",
        separator=";",
        null_values="?",
        truncate_ragged_lines=True
    )

# 1. Normalización del encabezado
def normalizar_encabezado(df: pl.DataFrame) -> pl.DataFrame:
    minusculas = {
        columna: columna.strip().lower()
        for columna
        in df.columns
    }

    df_encabezado = df.rename(minusculas)
            
    return df_encabezado

# 2. Casteo de tipos: date,time a datetime, y numéricos a Float64
def castear_tipos(df: pl.DataFrame) -> pl.DataFrame:
    # Formato date,time -> formato datetime
    df_datetime = df.with_columns(
        pl.concat_str([pl.col("date"), pl.col("time")], separator=" ")
            .str.to_datetime(format="%d/%m/%Y %H:%M:%S")
            .alias("datetime")
    )
    df_datetime = df_datetime.select(
        ["datetime"] + [
            col
            for col
            in df_datetime.columns
            if col not in ["date", "time", "datetime"]
        ]
    )
    
    # Variables numéricas a formato Float64
    columnas_medicion = [
        col
        for col
        in df_datetime.columns
        if col != "datetime"
    ]
    df_cast = df_datetime.with_columns([
        pl.col(col).cast(pl.Float64)
        for col
        in columnas_medicion
    ])
    
    return df_cast

# 3. Deduplicación
def deduplicar(df: pl.DataFrame) -> pl.DataFrame:
    return df.unique(subset=["datetime"], keep="first").sort("datetime")

# 4. Tratamiento de valores faltantes
def tratar_valores_faltantes(df: pl.DataFrame) -> pl.DataFrame:
    columnas_medicion = [
        col
        for col
        in df.columns
        if col != "datetime"
    ]
    
    # Aplicar interpolación lineal + ffill / bfill para extremos
    df_imputado = df.with_columns([
        pl.col(col)
        .interpolate()   # Estimación lineal continua
        .fill_null(strategy="forward")   # Para vacíos al final del dataset
        .fill_null(strategy="backward")  # Para vacíos al inicio del dataset
        .alias(col)
        for col in columnas_medicion
    ])
    
    return df_imputado

# Pipeline de preprocesamiento completo
def pipeline_preprocesamiento(df:pl.DataFrame) -> pl.DataFrame:    
    return tratar_valores_faltantes(
        deduplicar(
            castear_tipos(
                normalizar_encabezado(df)
            )
        )
    )

# Conteo de duplicados a eliminar
def contar_duplicados(df: pl.DataFrame) -> dict:
    total_filas = df.height
    
    # Duplicados según la estampa de tiempo
    filas_unicas_clave = df.unique(subset=["datetime"]).height
    duplicados = total_filas - filas_unicas_clave

    return {
        "total_filas": total_filas,
        "duplicados": duplicados
    }

# Pipeline de preprocesamiento completo + conteo de duplicados
def pipeline_preprocesamiento_y_conteo_de_duplicados(df:pl.DataFrame) -> pl.DataFrame:    
    df_cast = castear_tipos(
        normalizar_encabezado(df)
    )

    print (pl.DataFrame(
        contar_duplicados(df_cast)
    ))
    
    return tratar_valores_faltantes(
        deduplicar(
            df_cast
        )
    )


# Escritura de dataset en apartado de datos procesados
def escribir(df):
    df.write_csv(str(RUTA_DATA_PROCESSED / "preprocessed_consumption.csv"))

# Pipeline completo de extracción, preprocesamiento y escritura de dataset crudo:
def preprocesamiento_de_crudo():
    try:
        escribir(
            pipeline_preprocesamiento(
                lectura_dataset_raw()
            )
        )
    except FileNotFoundError:
        print("Error: Dataset no disponible.")

# preprocesamiento_de_crudo + conteo de duplicados:
def preprocesamiento_y_conteo_de_duplicados_de_crudo():
    try:
        escribir(
            pipeline_preprocesamiento_y_conteo_de_duplicados(
                lectura_dataset_raw()
            )
        )
    except FileNotFoundError:
        print("Error: Dataset no disponible.")