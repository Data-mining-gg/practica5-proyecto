import polars as pl
from config.rutas import RUTA_DATA_RAW, RUTA_DATA_PROCESSED
from pathlib import Path

def lectura_dataset_raw(df:pl.DataFrame) -> pl.DataFrame:
    return pl.read_csv(
        ruta_archivo,
        separator=";",
        null_values="?",
        truncate_ragged_lines=True
    )

# 1. Normalización de celdas con texto
def normalizar_texto(df: pl.DataFrame) -> pl.DataFrame:
    minusculas = {
        columna: columna.strip().lower()
        for columna
        in df.columns
    }

    df_normalizado = df.rename(minusculas)
    
    # Eliminación de espacios en blanco iniciales/finales
    columnas_texto = [
        columna
        for columna, dtype
        in df_normalizado.schema.items()
        if dtype == pl.String
        ]
    if columnas_texto:
        df_normalizado = df_normalizado.with_columns([
            pl.col(columna).str.strip_chars()
            for columna 
            in columnas_texto
        ])
        
    return df_normalizado

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
                normalizar_texto(df)
            )
        )
    )

# Escritura de dataset en apartado de datos procesados
def escribir(df):
    df.write_csv(str(RUTA_DATA_PROCESSED / "preprocessed_consumption.csv"))

# Pipeline completo de extracción, preprocesamiento y escritura de dataset crudo:
try:
    data_crudo = pl.read_csv(
        RUTA_DATA_RAW / "household_power_consumption.txt",
        separator=";",
        null_values="?"
    )
    escribir(
        pipeline_preprocesamiento(data_crudo)
    )
except FileNotFoundError:
    print("Error: Dataset no disponible.")