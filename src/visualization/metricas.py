import polars as pl
from src.visualization.carga import ZONAS

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

MESES = [
    "Ene", "Feb", "Mar", "Abr", "May", "Jun",
    "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
]  # fmt: skip

VARIABLES = {
    "global_active_power": "Potencia activa (kW)",
    "global_reactive_power": "Potencia reactiva (kW)",
    "voltage": "Voltaje (V)",
    "global_intensity": "Intensidad (A)",
    "sub_metering_1": "Cocina (Wh)",
    "sub_metering_2": "Lavandería (Wh)",
    "sub_metering_3": "Calentador y aire (Wh)",
}

COLUMNAS_PERIODO = {
    "datetime": "Periodo",
    "potencia_promedio_kw": "Potencia promedio (kW)",
    "potencia_maxima_kw": "Potencia máxima (kW)",
    "reactiva_promedio_kw": "Potencia reactiva promedio (kW)",
    "voltaje_promedio_v": "Voltaje promedio (V)",
    "intensidad_promedio_a": "Intensidad promedio (A)",
    "energia_kwh": "Energía total (kWh)",
    **{zona: f"{nombre} (kWh)" for zona, nombre in ZONAS.items()},
}

ESTADISTICOS = {
    "variable": "Variable",
    "count": "Registros",
    "null_count": "Nulos",
    "mean": "Media",
    "std": "Desviación estándar",
    "min": "Mínimo",
    "25%": "Q1 (25%)",
    "50%": "Mediana",
    "75%": "Q3 (75%)",
    "max": "Máximo",
}


def kpis(df: pl.DataFrame) -> dict:
    fila_pico = df.row(df["global_active_power"].arg_max(), named=True)
    dias = df["datetime"].dt.date().n_unique()
    energia_total = df["energia_total"].sum()

    return {
        "potencia_promedio": df["global_active_power"].mean(),
        "potencia_pico": fila_pico["global_active_power"],
        "fecha_pico": fila_pico["datetime"],
        "energia_total": energia_total,
        "energia_diaria": energia_total / dias,
        "voltaje_promedio": df["voltage"].mean(),
        "hora_pico": int(por_hora(df).sort("potencia", descending=True)["hora"][0]),
        "dias": dias,
    }


def por_hora(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(pl.col("datetime").dt.hour().alias("hora"))
        .agg(pl.col("global_active_power").mean().alias("potencia"))
        .sort("hora")
    )


def por_dia_y_hora(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(
            pl.col("datetime").dt.weekday().alias("dia"),
            pl.col("datetime").dt.hour().alias("hora"),
        )
        .agg(pl.col("global_active_power").mean().alias("potencia"))
        .pivot(on="hora", index="dia", values="potencia", sort_columns=True)
        .sort("dia")
    )


def energia_por_zona(df: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "zona": list(ZONAS.values()),
            "energia_kwh": [df[zona].sum() for zona in ZONAS],
        }
    )


# Cuartiles por mes para el diagrama de caja, sin enviar millones de puntos al navegador
def cuartiles_por_mes(df: pl.DataFrame, variable: str) -> pl.DataFrame:
    return (
        df.group_by(pl.col("datetime").dt.month().alias("mes"))
        .agg(
            pl.col(variable).min().alias("minimo"),
            pl.col(variable).quantile(0.25).alias("q1"),
            pl.col(variable).median().alias("mediana"),
            pl.col(variable).quantile(0.75).alias("q3"),
            pl.col(variable).quantile(0.99).alias("p99"),
        )
        .sort("mes")
    )


def estadisticas(df: pl.DataFrame) -> pl.DataFrame:
    descripcion = df.select(list(VARIABLES)).describe()
    return (
        descripcion.transpose(
            include_header=True,
            header_name="variable",
            column_names=descripcion["statistic"],
        )
        .filter(pl.col("variable") != "statistic")
        .with_columns(
            pl.col("variable").replace(VARIABLES),
            pl.exclude("variable").cast(pl.Float64),
        )
        .with_columns(pl.col("count", "null_count").cast(pl.Int64))
        .rename(ESTADISTICOS)
    )
