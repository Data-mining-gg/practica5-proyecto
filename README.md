# Descripción del proyecto



## Preprocesamiento del dataset

### Casteo de tipos

Los principales cambios realizados con respecto a las deficiencias del formato original son la normalización del encabezado a nombres en minúsculas, la conversión al tipo nativo ```'null```' de valores faltantes marcados como '?', y la conversión de las columnas de fecha y hora por separado a una sola columna con datos del tipo ```datetime```.

### Deduplicado de filas

Ya que los registros del dataset consisten en mediciones eléctricas en intervalos de 1 minuto, desde el punto de vista físico del sensor, no es posible que existan dos lecturas simultáneas válidas para la misma vivienda en el mismo minuto, por lo que el criterio de deduplicado se basa registros con misma fecha y hora.

Realizando el conteo de registros duplicados, tenemos que el dataset no tenía duplicados en primer lugar.

### Tratamiento de valores ausentes

Dado que el valor del minuto actual depende de lo que ocurrió en el minuto anterior, se empleó función nativa de Polars, _interpolate_, para realizar una estimación para los puntos ausentes intermedios con respecto a la curva lineal formada por los valores entre los que se encuentra y, con los valores faltantes de los extremos, se repite el valor más cercano.

### Normalización del dataset

Ya que se busca que la media y desviación de la normalización es exclusivamente con respecto a una proporción de entrenamiento que, en este caso, se decidió definir por defecto de 80%, entonces se genera una línea de corte de la que tomamos solo dicha porción de valores de "entrenamiento" para calcular las medidas de media y desviación estándar.

Podemos observar que la normalización funciona dado que, para todas las variables cuantitativas del nuevo dataset, la media se aproxima a 0 y la desviación estándar a 1.

## Mapa de calor

Ya que todas las columnas almacenan valores de tipo numérico, con excepción de la columna ```datetime```, entonces simplemente se excluye a esta última de las mediciones.

Dados los valores generados en el mapa, la principal redundancia a remover es la de la relación entre ```global_intensity``` y ```global_active_power```, ya que la intersección muestra un coeficiente de correlación de 1, por lo que ambas variables contienen información estrictamente semejante, pudiendo generar redundancia de cálculo en Machine Learning.

En cuanto al resto de variables, dado que todas tienen entre sí niveles de correlación por debajo del umbral estándar (por convención) de 0.85, entonces éstas reflejan por sí mismas una parte importante del consumo global sin redundancias.


# Descripción del dataset y su fuente



# Descripción de organización de carpetas



# Instrucciones de instalación


## Instalación del dataset



# Ejecución del dashboard