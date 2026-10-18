# Descripción del proyecto

## Normalización del dataset
### Tratamiento de valores ausentes
Dado que el valor del minuto actual depende de lo que ocurrió en el minuto anterior, se empleó función nativa de Polars, _intercalate_, para realizar una estimación para los puntos ausentes intermedios con respecto a la curva lineal formada por los valores entre los que se encuentra y, con los valores faltantes de los extremos, se repite el valor más cercano.

# Descripción del dataset y su fuente


# Descripción de organización de carpetas


# Instrucciones de instalación

## Instalación del dataset


# Ejecución del dashboard