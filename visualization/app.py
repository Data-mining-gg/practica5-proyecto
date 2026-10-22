import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl
import streamlit as st

import src.visualization.carga as carga
import src.visualization.metricas as metricas
import src.visualization.graficas as graficas
import src.measurements.correlations as correlate
import src.reduction.pca as pca_mod

st.set_page_config(
    page_title="Consumo eléctrico doméstico", page_icon="⚡", layout="wide"
)


@st.cache_resource(show_spinner="Cargando datos...")
def datos_limpios() -> pl.DataFrame:
    return carga.cargar_limpio()


@st.cache_resource(show_spinner="Cargando datos...")
def datos_preparados() -> pl.DataFrame:
    return carga.cargar_preparado()


@st.cache_resource(show_spinner="Ajustando PCA...")
def modelo_pca():
    preparado = datos_preparados()
    return pca_mod.ajustar_pca(
        preparado.drop([col for col in preparado.columns if col.startswith("pc")])
    )


try:
    limpio = datos_limpios()
    preparado = datos_preparados()
except FileNotFoundError:
    st.error(
        "No se encontraron los datasets procesados. Ejecuta primero "
        "`notebooks/Genera_normalizado.ipynb` y `notebooks/PCA.ipynb`."
    )
    st.stop()


# Filtros
fecha_min = limpio["datetime"].min().date()
fecha_max = limpio["datetime"].max().date()

with st.sidebar:
    st.header("Filtros")
    rango = st.date_input(
        "Rango de fechas",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max,
    )
    agregacion = st.radio(
        "Nivel de agregación", list(carga.AGREGACIONES), index=1, horizontal=True
    )
    variable = st.selectbox(
        "Variable para distribuciones",
        list(metricas.VARIABLES),
        format_func=metricas.VARIABLES.get,
    )

if len(rango) != 2:
    st.info("Selecciona la fecha de inicio y la de fin.")
    st.stop()

inicio, fin = rango
df = carga.filtrar_fechas(limpio, inicio, fin)
if df.is_empty():
    st.warning("No hay registros en el rango seleccionado.")
    st.stop()

df_periodo = carga.agregar_periodo(df, agregacion)


# Encabezado y KPIs
st.title("⚡ Consumo eléctrico de un hogar")
st.caption(
    f"Mediciones minuto a minuto de una casa en Sceaux, Francia · "
    f"{inicio:%d/%m/%Y} – {fin:%d/%m/%Y} · {df.height:,} registros"
)

k = metricas.kpis(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Potencia promedio", f"{k['potencia_promedio']:.2f} kW")
c2.metric(
    "Pico de potencia",
    f"{k['potencia_pico']:.2f} kW",
    help=f"Registrado el {k['fecha_pico']:%d/%m/%Y a las %H:%M}",
)
c3.metric(
    "Energía total", f"{k['energia_total']:,.0f} kWh", help=f"En {k['dias']} días"
)
c4.metric("Energía diaria promedio", f"{k['energia_diaria']:.1f} kWh")
c5.metric("Hora de mayor consumo", f"{k['hora_pico']}:00 h")


# Pestañas
tiempo, patrones, distribuciones, relaciones, tablas = st.tabs(
    ["Tendencia", "Patrones", "Distribuciones", "Relaciones y PCA", "Tablas"]
)

with tiempo:
    st.plotly_chart(
        graficas.linea_tiempo(
            df_periodo,
            {
                "potencia_promedio_kw": "Potencia promedio (kW)",
                "potencia_maxima_kw": "Potencia máxima (kW)",
            },
        ),
        width="stretch",
    )
    izquierda, derecha = st.columns([2, 1])
    izquierda.plotly_chart(graficas.area_energia(df_periodo), width="stretch")
    derecha.plotly_chart(
        graficas.dona_energia(metricas.energia_por_zona(df)), width="stretch"
    )

with patrones:
    st.plotly_chart(
        graficas.heatmap_semana(metricas.por_dia_y_hora(df)), width="stretch"
    )
    st.plotly_chart(graficas.barras_hora(metricas.por_hora(df)), width="stretch")

with distribuciones:
    nombre = metricas.VARIABLES[variable]
    izquierda, derecha = st.columns(2)
    izquierda.plotly_chart(graficas.histograma(df[variable], nombre), width="stretch")
    derecha.plotly_chart(
        graficas.caja_mensual(metricas.cuartiles_por_mes(df, variable), nombre),
        width="stretch",
    )

with relaciones:
    st.plotly_chart(
        graficas.heatmap_correlacion(
            correlate.matriz_correlaciones(
                df.select(list(metricas.VARIABLES)).rename(metricas.VARIABLES)
            )
        ),
        width="stretch",
    )

    pca = modelo_pca()
    varianza = pca_mod.tabla_varianza(pca)
    n_componentes = pca_mod.seleccionar_componentes(pca)

    izquierda, derecha = st.columns(2)
    izquierda.plotly_chart(graficas.varianza_pca(varianza), width="stretch")
    izquierda.caption(
        f"Se retienen {n_componentes} componentes "
        f"({varianza['varianza_acumulada'][n_componentes - 1]:.1%} de la varianza)."
    )

    muestra = (
        carga.filtrar_fechas(preparado, inicio, fin)
        .sample(n=min(10_000, df.height), seed=42)
        .with_columns(pl.col("datetime").dt.hour().alias("hora"))
    )
    derecha.plotly_chart(graficas.dispersion_pca(muestra), width="stretch")

    variables_pca = pca_mod.columnas_numericas(
        preparado.drop([col for col in preparado.columns if col.startswith("pc")])
    )
    st.subheader("Loadings de los componentes retenidos")
    # Sin unidades: el PCA trabaja con las variables estandarizadas
    nombres_pca = {
        col: nombre.split(" (")[0] for col, nombre in metricas.VARIABLES.items()
    }
    st.dataframe(
        pca_mod.tabla_loadings(pca, variables_pca, n_componentes)
        .with_columns(pl.col("variable").replace(nombres_pca))
        .rename({"variable": "Variable"}),
        hide_index=True,
        width="stretch",
    )

with tablas:
    st.subheader(f"Resumen por {agregacion.lower()}")
    resumen = df_periodo.rename(metricas.COLUMNAS_PERIODO)
    st.dataframe(resumen, hide_index=True, width="stretch")
    st.download_button(
        "Descargar CSV",
        resumen.write_csv(),
        file_name=f"consumo_por_{agregacion.lower()}.csv",
        mime="text/csv",
    )

    st.subheader("Estadísticas descriptivas")
    st.dataframe(metricas.estadisticas(df), hide_index=True, width="stretch")

    st.subheader("Registros minuto a minuto")
    st.caption("Primeros 1,000 registros del rango seleccionado.")
    st.dataframe(
        df.select("datetime", *metricas.VARIABLES)
        .head(1000)
        .rename({"datetime": "Fecha y hora", **metricas.VARIABLES}),
        hide_index=True,
        width="stretch",
    )
