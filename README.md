# Descripción del proyecto



## Preprocesamiento del dataset


### Tratamiento de valores ausentes

Dado que el valor del minuto actual depende de lo que ocurrió en el minuto anterior, se empleó función nativa de Polars, _intercalate_, para realizar una estimación para los puntos ausentes intermedios con respecto a la curva lineal formada por los valores entre los que se encuentra y, con los valores faltantes de los extremos, se repite el valor más cercano.

## Mapa de calor

Dados los valores generados en el mapa, la principal redundancia a remover es la de la relación entre ```global_intensity``` y ```global_active_power```, ya que la intersección muestra un coeficiente de correlación de 1, por lo que ambas variables contienen información estrictamente semejante, pudiendo generar redundancia de cálculo en Machine Learning.

En cuanto al resto de variables, dado que todas tienen entre sí niveles de correlación por debajo del umbral estándar de 0.85, entonces éstas reflejan por sí mismas una parte importante del consumo global sin redundancias.


# Descripción del dataset y su fuente



# Descripción de organización de carpetas



# Instrucciones de instalación


## Instalación del dataset



# Ejecución del dashboard