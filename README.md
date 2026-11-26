# ¿Cuánto dura un hit? Benchmark temporal con PyTorch

Este proyecto continúa el análisis estadístico de duración de los hits de spotify durante los años para el ramo de Estadística Computacional. La demo permite revisar los resultados y explorar las predicciones canción por canción.

## Ejecución

Usa Python 3.10 e instala la versión de PyTorch para CPU. El entrenamiento y el informe se generan localmente.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.9.1+cpu' -r requirements.txt
.\.venv\Scripts\python.exe -m spotify_duration.benchmark
Start-Process .\artifacts\report.html
```

La primera ejecución usa KaggleHub para descargar la versión 4 del dataset. Si Kaggle solicita aceptar las condiciones o quieres usar una copia local, descarga y extrae `songs_normalize.csv` y pasa su ruta con `--csv`. [Dataset Top Hits Spotify from 2000–2019](https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019)

```powershell
.\.venv\Scripts\python.exe -m spotify_duration.benchmark --csv 'C:\ruta\songs_normalize.csv'
```

Puedes cambiar la carpeta con `--output-dir` y repetir con otra semilla mediante `--seed`. La semilla predeterminada es 42.

| Archivo en `artifacts/` | Contenido |
| --- | --- |
| `report.html` | Demo interactiva que funciona en el navegador sin servidor |
| `mae_by_year.png` | Error absoluto medio por año y número de canciones en cada grupo |
| `metrics.json` | Métricas de validación y prueba, tamaños de partición, huella SHA-256 del CSV, versiones e historial de entrenamiento |
| `predictions.csv` | Duración real y predicciones de cada canción de prueba, en segundos |

## Diseño del experimento

| Partición | Años | Uso |
| --- | --- | --- |
| Entrenamiento | 1998–2009 | Ajustar el preprocesamiento, la mediana, el Random Forest y los pesos de la red |
| Validación | 2010–2014 | Elegir la época de la red mediante su pérdida por cuantiles |
| Prueba | 2015–2019 | Medir desempeño en años posteriores |

La limpieza conserva canciones con duración positiva, género informado y popularidad de al menos 8,9. Las canciones se identifican por artista y título normalizados para evitar repeticiones entre particiones. Si hay varias filas para el mismo identificador, se conserva la del año más antiguo. Los modelos que usan características reciben las mismas nueve variables numéricas de audio. El año define las particiones, la popularidad se usa para el filtro y ni el artista ni el título entran como predictores. La imputación por mediana y la estandarización se ajustan únicamente con el entrenamiento.

- **Mediana histórica.** Predice la mediana del entrenamiento para cada canción. Su rango fijo es el comprendido entre los percentiles 10 y 90 históricos.
- **Random Forest.** Estima un valor de duración a partir de las características de audio. Usa la configuración de referencia del experimento original.
- **MLP en PyTorch.** Usa dos capas ocultas y aprende los cuantiles 10, 50 y 90 con la pérdida *pinball*. Predice tres cuantiles ordenados y recupera los pesos de la época con menor pérdida de validación. El percentil 50 se usa como predicción puntual y los otros dos como límites.

Los resultados describen esta muestra histórica de hits. El identificador por artista y título es una aproximación a la identidad de la canción, y un mismo artista puede aparecer en distintos períodos.

## Resultados con el dataset original

Con la semilla 42 y la versión 4 de Kaggle quedaron 1.728 canciones después de la limpieza, de las cuales 412 entraron en la prueba de 2015–2019. Un MAE menor indica una estimación puntual más precisa en esta muestra.

| Modelo | MAE de prueba | Cobertura | Ancho medio |
| --- | ---: | ---: | ---: |
| Mediana histórica | 34,69 s | 65,0% | 96,7 s |
| Random Forest | 41,56 s | — | — |
| MLP por cuantiles | 33,55 s | 66,3% | 88,6 s |

La MLP tuvo el menor MAE global por 1,14 segundos frente a la mediana histórica. En 2019 la mediana obtuvo 46,25 s de error frente a 47,31 s de la red. La cobertura de la MLP pasó de 81,6% en validación a 66,3% en prueba y llegó a 41,25% en 2019. Esa diferencia muestra por qué conviene revisar el error y la cobertura por año antes de confiar en los intervalos.
