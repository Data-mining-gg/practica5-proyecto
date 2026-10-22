import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from src.visualization.carga import ZONAS
from src.visualization.metricas import DIAS, MESES

# Paleta de https://coolors.co/palettes/trending
PALETA = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
ESCALA = [[0, "#264653"], [0.5, "#e9c46a"], [1, "#e76f51"]]
ESCALA_DIVERGENTE = [[0, "#264653"], [0.5, "#f1f1f1"], [1, "#e76f51"]]


def _estilo(fig: go.Figure, titulo: str) -> go.Figure:
    return fig.update_layout(
        title=titulo,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )


def linea_tiempo(df: pl.DataFrame, columnas: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    for (columna, nombre), color in zip(columnas.items(), PALETA):
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df[columna],
                name=nombre,
                mode="lines",
                line=dict(color=color),
            )
        )
    return _estilo(fig, "Consumo a lo largo del tiempo")


def area_energia(df: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    for (zona, nombre), color in zip(ZONAS.items(), PALETA[1:]):
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df[zona],
                name=nombre,
                stackgroup="energia",
                line=dict(color=color, width=0.5),
            )
        )
    fig.update_yaxes(title="kWh")
    return _estilo(fig, "Energía por zona de la casa")


def dona_energia(df: pl.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=df["zona"],
            values=df["energia_kwh"],
            hole=0.5,
            marker=dict(colors=PALETA[1:]),
            sort=False,
        )
    )
    return _estilo(fig, "Reparto de la energía")


def barras_hora(df: pl.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=df["hora"], y=df["potencia"], marker_color=PALETA[1]))
    fig.update_xaxes(title="Hora del día", dtick=1)
    fig.update_yaxes(title="kW promedio")
    return _estilo(fig, "Potencia promedio por hora del día")


def heatmap_semana(df: pl.DataFrame) -> go.Figure:
    horas = [col for col in df.columns if col != "dia"]
    fig = go.Figure(
        go.Heatmap(
            z=df.select(horas).to_numpy(),
            x=[int(h) for h in horas],
            y=[DIAS[d - 1] for d in df["dia"]],
            colorscale=ESCALA,
            colorbar=dict(title="kW"),
        )
    )
    fig.update_xaxes(title="Hora del día", dtick=1)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    return _estilo(fig, "Potencia promedio por día de la semana y hora")


# Histograma calculado con NumPy para no mandar millones de puntos al navegador
def histograma(serie: pl.Series, nombre: str, bins: int = 60) -> go.Figure:
    conteos, bordes = np.histogram(serie.to_numpy(), bins=bins)
    fig = go.Figure(
        go.Bar(
            x=(bordes[:-1] + bordes[1:]) / 2,
            y=conteos,
            width=np.diff(bordes),
            marker_color=PALETA[2],
        )
    )
    fig.update_xaxes(title=nombre)
    fig.update_yaxes(title="Registros")
    fig.update_layout(hovermode="closest")
    return _estilo(fig, f"Distribución de {nombre}")


def caja_mensual(df: pl.DataFrame, nombre: str) -> go.Figure:
    fig = go.Figure(
        go.Box(
            x=[MESES[m - 1] for m in df["mes"]],
            lowerfence=df["minimo"],
            q1=df["q1"],
            median=df["mediana"],
            q3=df["q3"],
            upperfence=df["p99"],
            marker_color=PALETA[0],
            fillcolor=PALETA[1],
            name=nombre,
        )
    )
    fig.update_yaxes(title=nombre)
    fig.update_layout(hovermode="closest", showlegend=False)
    return _estilo(fig, f"{nombre} por mes (bigote superior: percentil 99)")


def heatmap_correlacion(matriz: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        matriz.round(2),
        text_auto=True,
        zmin=-1,
        zmax=1,
        color_continuous_scale=ESCALA_DIVERGENTE,
        aspect="auto",
    )
    fig.update_layout(hovermode="closest")
    return _estilo(fig, "Correlación de Pearson")


def varianza_pca(df: pl.DataFrame, umbral: float = 0.8) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["componente"],
            y=df["varianza_explicada"],
            name="Varianza explicada",
            marker_color=PALETA[1],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["componente"],
            y=df["varianza_acumulada"],
            name="Varianza acumulada",
            mode="lines+markers",
            line=dict(color=PALETA[4]),
        )
    )
    fig.add_hline(y=umbral, line_dash="dash", line_color="gray")
    fig.update_yaxes(tickformat=".0%")
    return _estilo(fig, "Varianza explicada por componente")


def dispersion_pca(df: pl.DataFrame) -> go.Figure:
    fig = px.scatter(
        df.to_pandas(),
        x="pc1",
        y="pc2",
        color="hora",
        color_continuous_scale=ESCALA,
        opacity=0.5,
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(hovermode="closest")
    return _estilo(fig, "Proyección PC1 vs. PC2 (color: hora del día)")
