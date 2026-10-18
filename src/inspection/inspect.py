import polars as pl
from config.rutas import RUTA_DATA_RAW, RUTA_DATA_PROCESSED

# Configuración GLOBAL de visualización (Aplica a todas las funciones y prints)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_width_chars(200)

# Dimensiones de dataset
def obtener_dimensiones(df: pl.DataFrame) -> tuple[int, int]:
    return df.shape

# Intervalo de fechas
def intervalo_fechas(df: pl.DataFrame) -> tuple:
    fechas = df["Date"].str.to_date("%d/%m/%Y")
    return fechas.min(), fechas.max()

# Tipos de datos
def tipos(df: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({
        "Columna": df.columns,
        "Tipo_Dato": [str(dtype) for dtype in df.dtypes]
    })

# Valores nulos por columna
def nulos_por_col(df: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({
        "Columna": df.columns,
        "Nulos": [df[col].null_count() for col in df.columns]
    })

# Cantidad de filas con al menos un nulo
def cols_nulas(df: pl.DataFrame) -> int:
    return df.height - df.drop_nulls().height

# Estadísticas descriptivas básicas
def estadisticas(df: pl.DataFrame, redondeo=6) -> pl.DataFrame:
    redondeables = [col for col in df.columns if df[col].dtype.is_numeric()]
    
    if redondeables:
        descripcion = df.select(redondeables).describe()
        cols_a_redondear = [c for c in descripcion.columns if c!="statistic"]
        descripcion = descripcion.with_columns([
            pl.col(c).cast(pl.Float64, strict=False).round(redondeo)
            for c in cols_a_redondear
        ])
    
    return descripcion.transpose(
        include_header=True, 
        header_name="variable", 
        column_names=descripcion["statistic"]
    ).filter(
        pl.col("variable") != "statistic"
    )
