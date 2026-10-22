# Descripción del proyecto

En este proyecto preparamos un dataset de consumo eléctrico doméstico para entrenar modelos de aprendizaje automático en prácticas posteriores. Para ello limpiamos los datos, normalizamos las variables numéricas, reducimos la dimensionalidad con el coeficiente de correlación de Pearson y con PCA, y mostramos los resultados en un dashboard interactivo hecho en Streamlit.

## Preprocesamiento del dataset

### Casteo de tipos

Los principales cambios realizados con respecto a las deficiencias del formato original son la normalización del encabezado a nombres en minúsculas, la conversión al tipo nativo `'null`' de valores faltantes marcados como '?', y la conversión de las columnas de fecha y hora por separado a una sola columna con datos del tipo `datetime`.

### Deduplicado de filas

Ya que los registros del dataset consisten en mediciones eléctricas en intervalos de 1 minuto, desde el punto de vista físico del sensor, no es posible que existan dos lecturas simultáneas válidas para la misma vivienda en el mismo minuto, por lo que el criterio de deduplicado se basa registros con misma fecha y hora.

Realizando el conteo de registros duplicados, tenemos que el dataset no tenía duplicados en primer lugar.

### Tratamiento de valores ausentes

Dado que el valor del minuto actual depende de lo que ocurrió en el minuto anterior, se empleó función nativa de Polars, _interpolate_, para realizar una estimación para los puntos ausentes intermedios con respecto a la curva lineal formada por los valores entre los que se encuentra y, con los valores faltantes de los extremos, se repite el valor más cercano.

### Normalización del dataset

Ya que se busca que la media y desviación de la normalización es exclusivamente con respecto a una proporción de entrenamiento que, en este caso, se decidió definir por defecto de 80%, entonces se genera una línea de corte de la que tomamos solo dicha porción de valores de "entrenamiento" para calcular las medidas de media y desviación estándar.

Podemos observar que la normalización funciona dado que, para todas las variables cuantitativas del nuevo dataset, la media se aproxima a 0 y la desviación estándar a 1.

## Mapa de calor

Ya que todas las columnas almacenan valores de tipo numérico, con excepción de la columna `datetime`, entonces simplemente se excluye a esta última de las mediciones.

Dados los valores generados en el mapa, la principal redundancia a remover es la de la relación entre `global_intensity` y `global_active_power`, ya que la intersección muestra un coeficiente de correlación de 0.9989 (el mapa lo redondea a 1.00), por lo que ambas variables contienen información prácticamente idéntica, pudiendo generar redundancia de cálculo en Machine Learning. Esto tiene sentido físico: la potencia activa es aproximadamente el producto del voltaje por la intensidad, y como el voltaje varía muy poco, la intensidad es casi proporcional a la potencia.

Se descarta `global_intensity` y se conserva `global_active_power`, ya que esta última es la variable objetivo natural del dataset (consumo del hogar) y guarda relación directa con los sub-medidores.

En cuanto al resto de variables, dado que todas tienen entre sí niveles de correlación por debajo del umbral estándar (por convención) de 0.85, entonces éstas reflejan por sí mismas una parte importante del consumo global sin redundancias.

## Análisis de Componentes Principales (PCA)

### Estandarización

El PCA se aplica sobre el dataset normalizado con z-score y sin `global_intensity`, es decir, 6 variables cuantitativas. Se verificó que, en la porción de entrenamiento (primer 80%), todas tienen media aproximadamente 0 y desviación estándar 1. Por la misma razón que en la normalización, el PCA se ajusta solo con la porción de entrenamiento y después se proyecta todo el dataset, para no filtrar información del futuro.

### Varianza explicada y número de componentes

| Componente | Valor propio | Varianza explicada | Varianza acumulada |
| ---------- | ------------ | ------------------ | ------------------ |
| PC1        | 2.311        | 38.5%              | 38.5%              |
| PC2        | 0.986        | 16.4%              | 54.9%              |
| PC3        | 0.934        | 15.6%              | 70.5%              |
| PC4        | 0.834        | 13.9%              | 84.4%              |
| PC5        | 0.771        | 12.9%              | 97.3%              |
| PC6        | 0.164        | 2.7%               | 100%               |

Se retienen **4 componentes**, con el criterio de alcanzar al menos 80% de varianza acumulada (84.4%). El criterio de Kaiser (valor propio mayor a 1) retendría solo PC1, lo cual perdería más del 60% de la información; además PC2, PC3 y PC4 tienen valores propios cercanos a 1, por lo que en el scree plot no se observa un "codo" claro después de PC1. PC6 aporta apenas 2.7%, lo que indica que todavía queda algo de redundancia lineal entre las variables.

### Interpretación de loadings

- **PC1 (consumo general):** todas las variables de consumo pesan de forma positiva (`global_active_power` 0.61, `sub_metering_3` 0.43, `sub_metering_1` 0.36, `sub_metering_2` 0.32) y `voltage` de forma negativa (-0.38), ya que el voltaje cae cuando la demanda de la casa sube. Valores altos de PC1 representan momentos de alto consumo.
- **PC2 (lavandería/reactiva vs. calentador/aire):** contrasta `global_reactive_power` (0.68) y `sub_metering_2` (0.51) contra `sub_metering_3` (-0.50). Los motores de la lavadora y secadora generan potencia reactiva, mientras que el calentador de agua y el aire acondicionado son cargas principalmente resistivas.
- **PC3 (cocina vs. lavandería):** contrasta `sub_metering_1` (0.73) contra `sub_metering_2` (-0.58).
- **PC4 (reactiva y calentador vs. cocina y lavandería):** `global_reactive_power` (0.63) y `sub_metering_3` (0.44) contra `sub_metering_1` (-0.46) y `sub_metering_2` (-0.44).

### Proyección en 2D

En la proyección PC1 vs. PC2 (muestra de 20,000 registros coloreada por hora del día) se observa una nube densa en valores bajos de PC1, que corresponde principalmente a la madrugada (consumo base), y una dispersión mucho mayor hacia valores altos de PC1 en horas de la mañana y la noche, que corresponden a los picos de consumo. Las líneas diagonales en el borde inferior se deben a que los sub-medidores registran valores enteros (Wh), lo que produce niveles discretos.

## Dataset preparado

El dataset final se guarda en `data/data-processed/prepared_consumption.csv` y contiene `datetime`, las 6 variables normalizadas (sin `global_intensity`) y los componentes `pc1` a `pc4`. El archivo original en `data/data-raw/` no se modifica.

# Descripción del dataset y su fuente

Usamos el dataset [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) del UCI Machine Learning Repository (Hebrail y Berard, 2006). Contiene mediciones del consumo eléctrico de una casa en Sceaux, Francia, tomadas cada minuto entre diciembre de 2006 y noviembre de 2010.

- **Tamaño:** 2,075,259 registros y 9 columnas.
- **Valores faltantes:** 25,979 registros (cerca del 1.25%) vienen marcados con `?`.
- **Columnas:**
  - `Date` y `Time`: fecha y hora de la medición.
  - `Global_active_power`: potencia activa total de la casa (kW).
  - `Global_reactive_power`: potencia reactiva total de la casa (kW).
  - `Voltage`: voltaje (V).
  - `Global_intensity`: intensidad de corriente (A).
  - `Sub_metering_1`: energía de la cocina (Wh).
  - `Sub_metering_2`: energía de la lavandería (Wh).
  - `Sub_metering_3`: energía del calentador de agua y el aire acondicionado (Wh).

# Descripción de organización de carpetas

```
proyecto/
|-- config/
|   |-- rutas.py                 # rutas a las carpetas del proyecto
|-- data/
|   |-- data-raw/                # dataset original, sin modificar
|   |-- data-processed/          # datasets generados por el proyecto
|-- notebooks/                   # ejecución y revisión de cada paso
|-- src/
|   |-- inspection/              # inspección inicial del dataset
|   |-- cleaning/                # preprocesamiento, normalización y dataset final
|   |-- measurements/            # correlación de Pearson y mapa de calor
|   |-- reduction/               # PCA
|   |-- visualization/           # carga, métricas y gráficas del dashboard
|-- visualization/
|   |-- app.py                   # código principal del dashboard
|-- README.md
|-- requirements.txt
```

Los notebooks y el dashboard solo llaman a las funciones de `src/`, que es donde está la lógica.

# Instrucciones de instalación

Necesitas Python 3.13 o superior. Desde la carpeta `proyecto/`:

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Instalación del dataset

El dataset original pesa unos 127 MB, por lo que no lo subimos al repositorio. Para obtenerlo:

1. Descarga el ZIP desde la [página del dataset en UCI](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).
2. Descomprímelo y coloca `household_power_consumption.txt` en `data/data-raw/`.
3. Ejecuta `notebooks/Genera_normalizado.ipynb` para generar los datasets preprocesado y normalizado.
4. Ejecuta `notebooks/PCA.ipynb` para generar el dataset final, `data/data-processed/prepared_consumption.csv`.

# Ejecución del dashboard

Con el entorno activado y el dataset final ya generado, desde la carpeta `proyecto/`:

```bash
streamlit run visualization/app.py
```

El dashboard se abre en el navegador, en `http://localhost:8501`.

## Contenido del dashboard

El dashboard lee los datasets de `data/data-processed/`: el limpio, para mostrar el consumo en sus unidades reales, y el preparado, para la sección de PCA.

- **Filtros (barra lateral):** rango de fechas, nivel de agregación (hora, día, semana o mes) y variable a analizar en las distribuciones.
- **KPIs:** potencia promedio, pico de potencia, energía total, energía diaria promedio y hora de mayor consumo.
- **Tendencia:** consumo a lo largo del tiempo, energía por zona de la casa y reparto de la energía.
- **Patrones:** consumo promedio por día de la semana y hora, y por hora del día.
- **Distribuciones:** histograma y diagrama de caja por mes de la variable elegida.
- **Relaciones y PCA:** matriz de correlación, varianza explicada, proyección en 2D y loadings.
- **Tablas:** resumen por periodo (descargable en CSV), estadísticas descriptivas y registros minuto a minuto.
