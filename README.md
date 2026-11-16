# VIALAB · Predicción de siniestros viales en Neiva

App de Streamlit del proyecto VIALAB (Especialización en Inteligencia Artificial, Pontificia Universidad Javeriana). Apoya la preparación de los operativos preventivos de tránsito en Neiva: para cada comuna y franja de 4 horas estima la probabilidad de que ocurra al menos un siniestro con víctimas, recomienda una acción y reparte los recursos del turno.

App publicada: https://vialab-app.onrender.com/

Se usa antes de cada turno; no trabaja en tiempo real. La herramienta propone y el funcionario decide.

## Pestañas

| Pestaña | Qué hace |
|---|---|
| Ocurrencia | Probabilidad, nivel de riesgo y acción recomendada para una comuna, una franja y una fecha |
| Mapa de riesgo | Mapa de las comunas, barras por franja y grilla de todas las comunas y franjas de una fecha |
| Plan del turno | Ordena las comunas de la franja, reparte agentes y unidades y permite descargar el plan |
| Severidad | Análisis de qué hace mortal un siniestro con víctimas. Usa datos que solo se conocen después del siniestro, así que sirve para explicar, no para planear |
| Cómo funciona | Qué hace la IA y qué no, métricas de 2025 y 2026 y límites |

## Archivos

| Archivo | Contenido |
|---|---|
| `app_vialab.py` | La app |
| `modelo_ocurrencia.pkl` | Regresión logística de ocurrencia (C = 10). Variables: comuna, franja, día de la semana, mes, festivo, combinación de franja y día, e historial de siniestros con víctimas de 90 y 365 días |
| `modelo_mortalidad.pkl` | Regresión logística con pesos por clase (C = 1) para la pestaña Severidad, con las variables del siniestro y un umbral de alerta de 0,44 |
| `datos_app.json` | Historial congelado por comuna y franja, festivos, umbrales y matriz de 6 acciones aprobados por el equipo, coeficientes de la mortalidad según el contexto, métricas de los modelos y límites de las comunas. Si trae puntos críticos, la app los muestra en el mapa y en el plan |
| `requirements.txt` | Dependencias. `scikit-learn` va fijo en 1.6.1 porque los modelos se guardaron con esa versión |
| `.python-version` | Versión de Python para Render (3.13) |

Los dos `.pkl` son diccionarios con el preprocesador (`pre`), el modelo (`mod`), las columnas (`cols`), las categorías (`cats`) y el umbral (`umbral`). Pendiente: anotar aquí el cuaderno y la fecha con que se entrenaron.

## Ejecutar en local

Con Python 3.13:

```bash
pip install -r requirements.txt
```

```bash
streamlit run app_vialab.py
```

## Despliegue en Render

Render vuelve a publicar la app cuando cambia la rama `main`, si el servicio tiene activo el despliegue automático. La versión de Python sale de `.python-version`; si el servicio tiene definida la variable `PYTHON_VERSION`, esa variable manda. El archivo `runtime` no lo lee Render.

## Actualizar los datos

`datos_app.json` lo genera el equipo de datos de VIALAB con los scripts `medir_modelos_app_render.py` y `build_datos_app_render.py` del proyecto. No están en este repositorio porque necesitan la base completa. Hay que regenerarlo y subirlo en estos casos:

- Cuando llegue un corte nuevo del Observatorio, para actualizar el historial.
- Cuando cambie alguno de los modelos, para volver a medir las métricas de la pestaña «Cómo funciona».

Si se reemplaza un `.pkl`, hay que guardarlo con scikit-learn 1.6.1 o cambiar esa versión en `requirements.txt`.

## Datos y límites

- Fuente: registros de siniestros del Observatorio de la Secretaría de Movilidad de Neiva, de enero de 2020 al 20 de septiembre de 2026. Este repositorio solo tiene conteos agregados, sin datos personales.
- Solo se usan los siniestros con víctimas. Puede haber subregistro.
- Los umbrales de riesgo son los percentiles 70 y 90 de la probabilidad de ocurrencia en 2024, aprobados por el equipo el 19 de octubre de 2026.
- En septiembre, noviembre y diciembre la alerta de mortalidad alta aparece en más de la mitad de las comunas y franjas. El equipo está ajustando ese umbral.
- Límites de las comunas: © colaboradores de OpenStreetMap (ODbL).
