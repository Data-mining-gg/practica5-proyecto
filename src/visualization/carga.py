import datetime as dt
import polars as pl
from config.rutas import RUTA_DATA_PROCESSED

AGREGACIONES = {"Hora": "1h", "Día": "1d", "Semana": "1w", "Mes": "1mo"}

ZONAS = {
    "cocina": "Cocina",
    "lavanderia": "Lavandería",
    "calentador_aire": "Calentador y aire",
    "otros": "Resto de la casa",
}


def cargar_limpio() -> pl.DataFrame:
    df = pl.read_csv(
        RUTA_DATA_PROCESSED / "preprocessed_consumption.csv", try_parse_dates=True
    )
    return agregar_energia(df)


def cargar_preparado() -> pl.DataFrame:
    return pl.read_csv(
        RUTA_DATA_PROCESSED / "prepared_consumption.csv", try_parse_dates=True
    )


# Energía por minuto en kWh: la potencia activa está en kW y los sub-medidores en Wh
def agregar_energia(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("global_active_power") / 60).alias("energia_total"),
        (pl.col("sub_metering_1") / 1000).alias("cocina"),
        (pl.col("sub_metering_2") / 1000).alias("lavanderia"),
        (pl.col("sub_metering_3") / 1000).alias("calentador_aire"),
    ).with_columns(
        (
            pl.col("energia_total")
            - pl.col("cocina")
            - pl.col("lavanderia")
            - pl.col("calentador_aire")
        )
        .clip(lower_bound=0)
        .alias("otros")
    )


def filtrar_fechas(df: pl.DataFrame, inicio: dt.date, fin: dt.date) -> pl.DataFrame:
    return df.filter(pl.col("datetime").dt.date().is_between(inicio, fin))


def agregar_periodo(df: pl.DataFrame, agregacion: str) -> pl.DataFrame:
    return (
        df.sort("datetime")
        .group_by_dynamic("datetime", every=AGREGACIONES[agregacion])
        .agg(
            pl.col("global_active_power").mean().alias("potencia_promedio_kw"),
            pl.col("global_active_power").max().alias("potencia_maxima_kw"),
            pl.col("global_reactive_power").mean().alias("reactiva_promedio_kw"),
            pl.col("voltage").mean().alias("voltaje_promedio_v"),
            pl.col("global_intensity").mean().alias("intensidad_promedio_a"),
            pl.col("energia_total").sum().alias("energia_kwh"),
            *[pl.col(zona).sum() for zona in ZONAS],
        )
    )
