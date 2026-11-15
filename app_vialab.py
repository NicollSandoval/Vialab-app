"""
VIALAB — App de predicción de siniestros viales
Carga los modelos generados por el notebook (modelo_ocurrencia.pkl y modelo_mortalidad.pkl)
y los insumos de datos_app.json (historial congelado, festivos, umbrales y acciones aprobados,
mortalidad según el contexto, puntos críticos, métricas y comunas).

Ejecutar:
    streamlit run app_vialab.py
"""

import html
import json
import math
import unicodedata
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Configuración ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VIALAB · Predicción de Siniestros",
    page_icon="🚦",
    layout="centered",
)

DIR = Path(__file__).parent  # carpeta donde está este archivo

# ── Carga de modelos y datos ───────────────────────────────────────────────────
@st.cache_resource
def cargar(nombre):
    ruta = DIR / nombre
    if not ruta.exists():
        return None
    return joblib.load(ruta)

@st.cache_resource
def cargar_datos():
    ruta = DIR / "datos_app.json"
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))

art_occ = cargar("modelo_ocurrencia.pkl")
art_sev = cargar("modelo_mortalidad.pkl")
DATOS = cargar_datos()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🚦 VIALAB · Predicción de Siniestros")
st.caption("VIALAB · Predicción de siniestros ciudad de Neiva")

if art_occ is None or art_sev is None or DATOS is None:
    st.error(
        "No se encontraron los archivos de modelos o de datos.\n\n"
        "Copia `modelo_ocurrencia.pkl`, `modelo_mortalidad.pkl` y `datos_app.json` "
        "en la misma carpeta que este archivo."
    )
    st.stop()

# ── Parámetros aprobados por el equipo ─────────────────────────────────────────
UMB_ALTA = DATOS["umbrales"]["ocurrencia_alta"]     # percentil 90 de 2024
UMB_MEDIA = DATOS["umbrales"]["ocurrencia_media"]   # percentil 70 de 2024
UMB_MORT = DATOS["umbrales"].get("mortalidad_alta") # percentil 75 de 2024 (mortalidad según el contexto)
ACCIONES = DATOS.get("acciones", {})                # matriz de 6 acciones: ocurrencia × mortalidad
CTX = DATOS.get("mortalidad_contexto")              # coeficientes de la mortalidad según el contexto
PUNTOS = DATOS.get("puntos_criticos", [])           # sitios con 4 o más siniestros con víctimas a menos de 150 m
METRICAS = DATOS.get("metricas")
PREV = DATOS["prevalencia_ocurrencia"]              # tasa base por comuna y franja
CORTE = date.fromisoformat(DATOS["historial_hasta"])
FESTIVOS = set(DATOS["festivos"])
COMUNAS = {c["id"]: c for c in DATOS["comunas"]}
DIAS_SEMANA = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
FRANJAS = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
MESES_NOMBRE = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                "septiembre", "octubre", "noviembre", "diciembre"]
HOY = datetime.now(timezone(timedelta(hours=-5))).date()   # fecha de Colombia (el servidor trabaja en UTC)
HAY_PLAN = CTX is not None and UMB_MORT is not None and len(ACCIONES) == 6

CATS = art_occ["cats"]
COMUNAS_MODELO = CATS.get("comuna", sorted(COMUNAS))
FRANJAS_MODELO = CATS.get("franja_horaria", FRANJAS)

COLORES = {"Alta": "#e74c3c", "Media": "#f39c12", "Baja": "#2ecc71"}
COLOR_ST = {"Alta": "red", "Media": "orange", "Baja": "green"}
ICONO = {"Alta": "🔴", "Media": "🟡", "Baja": "🟢"}
ADJETIVO = {"Alta": "alto", "Media": "medio", "Baja": "bajo"}

# ── Utilidades ─────────────────────────────────────────────────────────────────
def predecir(art, fila: dict) -> float:
    df = pd.DataFrame([fila])[art["cols"]]
    X  = art["pre"].transform(df)
    return float(art["mod"].predict_proba(X)[0, 1])

def predecir_lote(art, filas: list) -> np.ndarray:
    """Predice varias filas en una sola llamada (más rápido que una por una)."""
    df = pd.DataFrame(filas)[art["cols"]]
    return art["mod"].predict_proba(art["pre"].transform(df))[:, 1]

def pct(x: float, decimales: int = 1) -> str:
    """Porcentaje con coma decimal: 0.0437 → «4,4 %»."""
    return f"{x * 100:.{decimales}f}".replace(".", ",") + " %"

def numero(x: float, decimales: int = 1) -> str:
    return f"{x:.{decimales}f}".replace(".", ",")

def miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")

def cantidad(n: int, uno: str, varios: str) -> str:
    return f"{n} {uno if n == 1 else varios}"

def sin_tildes(texto: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", texto)
                   if unicodedata.category(ch) != "Mn").lower()

def nombre_dia(fecha: date, categorias) -> str:
    """Nombre del día tal como lo escribió el entrenamiento (con o sin tilde)."""
    buscado = DIAS_SEMANA[fecha.weekday()]
    for c in categorias:
        if sin_tildes(c) == buscado:
            return c
    return buscado

def es_festivo(fecha: date) -> bool:
    return fecha.isoformat() in FESTIVOS

def historial(comuna: str, franja: str):
    """Siniestros con víctimas: 90 días en la comuna y franja, 90 y 365 días en la comuna."""
    return DATOS["historial"][comuna][franja]

def nivel_ocurrencia(p: float) -> str:
    if p >= UMB_ALTA:
        return "Alta"
    if p >= UMB_MEDIA:
        return "Media"
    return "Baja"

def nombre_comuna(cid: str) -> str:
    c = COMUNAS.get(str(cid))
    return f"Comuna {cid} ({c['nombre']})" if c else f"Comuna {cid}"

def fila_ocurrencia(comuna, franja, dia, mes, festivo, h):
    return {
        "hist_victimas_90d_comuna_franja": h[0],
        "hist_victimas_90d_comuna":        h[1],
        "hist_victimas_365d_comuna":       h[2],
        "festivo":                         int(festivo),
        "comuna":                          comuna,
        "franja_horaria":                  franja,
        "dia_semana":                      dia,
        "mes":                             mes,
        "franja_dia":                      franja + "_" + dia,
    }

# ── Mortalidad según el contexto, acción recomendada y reparto ─────────────────
def _coef(tabla: dict, clave: str) -> float:
    """Coeficiente de una categoría (los días se comparan sin tildes)."""
    if clave in tabla:
        return tabla[clave]
    for k, v in tabla.items():
        if sin_tildes(k) == sin_tildes(clave):
            return v
    return 0.0

def mortalidad_contexto(comuna, franja, dia, mes, festivo) -> float:
    """Probabilidad de que un siniestro con víctimas sea mortal, según la comuna, la franja, el día,
    el mes y si es festivo. Es una regresión logística sin pesos; se conoce antes del turno."""
    z = (CTX["intercepto"] + CTX["festivo"] * int(festivo) + _coef(CTX["comuna"], comuna)
         + _coef(CTX["franja"], franja) + _coef(CTX["dia"], dia) + _coef(CTX["mes"], mes))
    return 1 / (1 + math.exp(-z))

def nivel_mortalidad(pm: float) -> str:
    return "Alta" if pm >= UMB_MORT else "Normal"

def accion(nivel: str, nivel_m: str) -> str:
    return ACCIONES.get(f"{nivel}|{nivel_m}", "")

def texto_mortalidad(pm: float, nivel_m: str) -> str:
    return (f"Mortalidad {'alta' if nivel_m == 'Alta' else 'normal'} para esta comuna, franja, día y mes: "
            f"si ocurre un siniestro con víctimas, la probabilidad de que sea mortal es del {pct(pm, 0)} "
            f"(se considera alta desde el {pct(UMB_MORT, 0)}).")

def turno(fecha: date, franja: str, festivo: bool) -> list:
    """Todas las comunas de una franja, de mayor a menor probabilidad de ocurrencia."""
    dia = nombre_dia(fecha, CATS.get("dia_semana", DIAS_SEMANA))
    mes = f"{fecha.month:02d}"
    comunas = [c for c in COMUNAS_MODELO if c in DATOS["historial"]]
    probs = predecir_lote(art_occ, [fila_ocurrencia(c, franja, dia, mes, festivo, historial(c, franja))
                                    for c in comunas])
    filas = []
    for c, p in zip(comunas, probs):
        nivel = nivel_ocurrencia(p)
        pm = mortalidad_contexto(c, franja, dia, mes, festivo)
        nivel_m = nivel_mortalidad(pm)
        filas.append({"comuna": c, "p": float(p), "nivel": nivel, "pm": pm, "nivel_m": nivel_m,
                      "accion": accion(nivel, nivel_m)})
    return sorted(filas, key=lambda r: -r["p"])

def repartir(total: int, pesos: list) -> list:
    """Reparte un número entero en proporción a los pesos (método del resto mayor)."""
    suma = sum(pesos) or 1
    crudo = [w / suma * total for w in pesos]
    base = [int(v) for v in crudo]
    resto = total - sum(base)
    orden = sorted(range(len(pesos)), key=lambda i: crudo[i] - base[i], reverse=True)
    for i in orden[:resto]:
        base[i] += 1
    return base

def puntos_de(comuna: str, franja: str, n: int = 3) -> list:
    """Puntos críticos de una comuna, primero los que más siniestros tienen en la franja."""
    i = FRANJAS.index(franja) if franja in FRANJAS else 0
    candidatos = [p for p in PUNTOS if p["comuna"] == comuna]
    return sorted(candidatos, key=lambda p: (-p["franjas"][i], -p["n"]))[:n]

def enlace_ver(p: dict) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={p['lat']},{p['lon']}"

def enlace_ruta(p: dict) -> str:
    return f"https://www.google.com/maps/dir/?api=1&destination={p['lat']},{p['lon']}&travelmode=driving"

def aviso_corte(fecha: date):
    dias = (fecha - CORTE).days
    if dias > 45:
        st.warning(
            f"El historial reciente llega hasta el {CORTE:%d/%m/%Y} (último corte del Observatorio). "
            f"La fecha elegida está {dias} días después: la predicción usa ese historial y puede estar "
            "desactualizada hasta que se cargue un corte nuevo."
        )
    elif dias < 0:
        st.info(
            f"La fecha es anterior al corte del {CORTE:%d/%m/%Y}: la predicción usa el historial de ese "
            "corte, no el que había en esa fecha."
        )

REGLA_NIVELES = (
    f"Alta: {pct(UMB_ALTA)} o más (el 10 % de celdas de mayor riesgo en 2024) · "
    f"Media: {pct(UMB_MEDIA)} a {pct(UMB_ALTA)} · Baja: menos de {pct(UMB_MEDIA)}. "
    "Umbrales aprobados por el equipo el 19-oct-2026."
)

st.caption(f"Historial de siniestros con víctimas al {CORTE:%d/%m/%Y}. "
           "Las predicciones son por franja de 4 horas para preparar cada turno; no son en tiempo real.")

tab_occ, tab_mapa, tab_plan, tab_sev, tab_info = st.tabs(
    ["🗺️ Ocurrencia", "🔥 Mapa de riesgo", "📋 Plan del turno", "🚑 Severidad", "ℹ️ Cómo funciona"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# OCURRENCIA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_occ:
    st.subheader("¿Habrá al menos un siniestro con víctimas?")
    st.caption("Por comuna y franja de 4 horas")

    cats = CATS

    col1, col2 = st.columns(2)
    with col1:
        fecha = st.date_input("Fecha del turno", value=HOY, format="DD/MM/YYYY", key="o_fecha")
        comuna = st.selectbox("Comuna", COMUNAS_MODELO, format_func=nombre_comuna, key="o_comuna")
        franja = st.selectbox("Franja horaria", FRANJAS_MODELO, key="o_franja")
    dia = nombre_dia(fecha, cats.get("dia_semana", DIAS_SEMANA))
    mes = f"{fecha.month:02d}"
    with col2:
        festivo = st.checkbox("Día festivo", value=es_festivo(fecha), key=f"o_fest_{fecha}")
        st.markdown(f"**Día:** {dia}")
        st.markdown(f"**Historial al {CORTE:%d/%m/%Y}** (siniestros con víctimas)")
        h_real = historial(comuna, franja)
        if st.checkbox("Probar otro historial (escenario)", key="o_ajustar"):
            clave = f"{comuna}_{franja}"
            h = [
                st.number_input("90 días — comuna y franja", 0, value=int(h_real[0]), key=f"o_h0_{clave}"),
                st.number_input("90 días — comuna",          0, value=int(h_real[1]), key=f"o_h1_{clave}"),
                st.number_input("365 días — comuna",         0, value=int(h_real[2]), key=f"o_h2_{clave}"),
            ]
        else:
            h = h_real
            st.markdown(f"90 días en la comuna y franja: **{h[0]}**  \n"
                        f"90 días en la comuna: **{h[1]}**  \n"
                        f"365 días en la comuna: **{h[2]}**")

    aviso_corte(fecha)

    if st.button("Predecir ocurrencia", type="primary", key="btn_ocurrencia"):
        prob = predecir(art_occ, fila_ocurrencia(comuna, franja, dia, mes, festivo, h))
        nivel = nivel_ocurrencia(prob)
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Probabilidad", pct(prob))
        with c2:
            st.markdown(f"### :{COLOR_ST[nivel]}[{ICONO[nivel]} Riesgo {ADJETIVO[nivel]}]")
            st.caption(f"{numero(prob / PREV)} veces el promedio de la ciudad ({pct(PREV)} por comuna y franja)")
        if HAY_PLAN:
            pm = mortalidad_contexto(comuna, franja, dia, mes, festivo)
            nivel_m = nivel_mortalidad(pm)
            st.markdown(f"**Acción recomendada:** {accion(nivel, nivel_m)}")
            st.caption(texto_mortalidad(pm, nivel_m))
        st.caption(REGLA_NIVELES)
        st.caption(f"Modelo: {art_occ['modelo']}")

# ═══════════════════════════════════════════════════════════════════════════════
# SEVERIDAD (ANÁLISIS)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sev:
    st.subheader("¿Qué hace mortal un siniestro con víctimas?")
    st.caption(
        "Análisis de severidad. Usa datos que solo se conocen después del siniestro (clase de evento, "
        "actores, vehículos), así que sirve para explicar y orientar la pedagogía, no para planear operativos."
    )

    cats2 = art_sev["cats"]
    umbral = art_sev.get("umbral", 0.5)

    col1, col2 = st.columns(2)
    with col1:
        franja2 = st.selectbox("Franja horaria ", cats2.get("franja_horaria", FRANJAS))
        dias2 = cats2.get("nombre_dia", ["lunes","martes","miércoles","jueves","viernes","sábado","domingo"])
        dia2 = st.selectbox("Día de la semana ", dias2)
        meses2 = cats2.get("mes", [f"{m:02d}" for m in range(1, 13)])
        mes2 = st.selectbox("Mes ", meses2)
        comuna2 = st.selectbox("Comuna ", cats2.get("comuna_norm", [str(i).zfill(2) for i in range(1, 11)]),
                               format_func=lambda c: nombre_comuna(c) if c in COMUNAS else c)
        tipo_via = st.selectbox("Tipo de vía", cats2.get("tipo_via", ["VIA_URBANA", "VIA_RURAL"]))
        clase_evento = st.selectbox("Clase de evento", cats2.get("clase_evento", ["ATROPELLO","CHOQUE_VEHICULO","CHOQUE_OBJETO_FIJO","CAIDA_OCUPANTE","INCENDIO"]))

    with col2:
        n_vehiculos = st.number_input("Número de vehículos", 1, 10, value=2)
        edad_min = st.number_input("Edad mínima conductor", 14, 100, value=30)
        edad_max = st.number_input("Edad máxima conductor", 14, 100, value=40)
        festivo2 = st.checkbox("Día festivo ", key="fest2")
        st.markdown("**Actores involucrados**")
        col_a, col_b = st.columns(2)
        with col_a:
            moto  = st.checkbox("Motocicleta")
            peat  = st.checkbox("Peatón")
            bici  = st.checkbox("Bicicleta")
            carga = st.checkbox("Vehículo de carga")
        with col_b:
            tc    = st.checkbox("Transporte colectivo")
            sp    = st.checkbox("Servicio público")
            cond  = st.checkbox("Conductora mujer")

    if st.button("Evaluar severidad", type="primary", key="btn_severidad"):
        fila2 = {
            "n_vehiculos":              n_vehiculos,
            "edad_min_conductor":       float(edad_min),
            "edad_max_conductor":       float(edad_max),
            "festivo":                  int(festivo2),
            "hay_motocicleta":          int(moto),
            "hay_peaton":               int(peat),
            "hay_bicicleta":            int(bici),
            "hay_vehiculo_carga":       int(carga),
            "hay_transporte_colectivo": int(tc),
            "hay_servicio_publico":     int(sp),
            "hay_conductora":           int(cond),
            "franja_horaria":           franja2,
            "nombre_dia":               dia2,
            "mes":                      mes2,
            "comuna_norm":              comuna2,
            "tipo_via":                 tipo_via,
            "clase_evento":             clase_evento,
        }
        puntaje = predecir(art_sev, fila2)
        st.divider()
        if puntaje >= umbral:
            st.markdown("### :red[🔴 Alerta de mortalidad alta]")
        else:
            st.markdown("### :green[🟢 Sin alerta de mortalidad alta]")
        st.caption(
            "El modelo usa pesos por clase para no dejar pasar casos mortales: su salida es un puntaje de "
            "alerta, no una probabilidad (en promedio es unas 2,8 veces la frecuencia real). Con este umbral "
            "detectó el 82 % de los siniestros mortales de 2025 y el 73 % de los de 2026."
        )
        st.caption(f"Puntaje {numero(puntaje, 2)} frente a un umbral de {numero(umbral, 2)} · Modelo: {art_sev['modelo']}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAPA DE RIESGO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mapa:
    st.subheader("Mapa de riesgo de ocurrencia por comuna")
    st.caption("Predice para todas las comunas y franjas de la fecha elegida, con el historial real de cada comuna")

    try:
        import folium
        from streamlit_folium import st_folium
        import plotly.graph_objects as go
        libs_ok = True
    except ImportError:
        libs_ok = False

    if not libs_ok:
        st.warning("Instala: `pip install folium streamlit-folium plotly`")
    else:
        comunas_lista = [c for c in COMUNAS_MODELO if c in DATOS["historial"]]
        franjas_lista = FRANJAS_MODELO

        col1, col2 = st.columns(2)
        with col1:
            fecha_m = st.date_input("Fecha del turno", value=HOY, format="DD/MM/YYYY", key="m_fecha")
            franja_m = st.selectbox("Franja horaria (para el mapa)", franjas_lista, key="m_franja")
        dia_m = nombre_dia(fecha_m, CATS.get("dia_semana", DIAS_SEMANA))
        mes_m = f"{fecha_m.month:02d}"
        with col2:
            festivo_m = st.checkbox("Día festivo", value=es_festivo(fecha_m), key=f"m_fest_{fecha_m}")
            st.markdown(f"**Día:** {dia_m}")
            ver_puntos = bool(PUNTOS) and st.checkbox("Mostrar puntos críticos", value=True, key="m_puntos")

        aviso_corte(fecha_m)

        if st.button("Generar visualización", type="primary", key="btn_mapa"):

            # ── Calcular matriz completa: comunas × franjas ────────────────
            filas_m = [fila_ocurrencia(com, fr, dia_m, mes_m, festivo_m, historial(com, fr))
                       for com in comunas_lista for fr in franjas_lista]
            probs_m = predecir_lote(art_occ, filas_m).reshape(len(comunas_lista), len(franjas_lista))
            df_matriz = pd.DataFrame(probs_m, index=comunas_lista, columns=franjas_lista)

            # ── 1. GRÁFICO DE BARRAS — riesgo promedio por franja ─────────
            st.divider()
            st.markdown("#### Riesgo promedio por franja horaria")
            avg_franja = df_matriz.mean(axis=0)
            fig_bar = go.Figure(go.Bar(
                x=avg_franja.index.tolist(),
                y=avg_franja.values.tolist(),
                marker_color=[COLORES[nivel_ocurrencia(v)] for v in avg_franja.values],
                text=[pct(v) for v in avg_franja.values],
                textposition="outside",
            ))
            for valor, nombre in [(UMB_MEDIA, "media"), (UMB_ALTA, "alta")]:
                fig_bar.add_hline(y=valor, line_dash="dot", line_color="gray",
                                  annotation_text=f"umbral {nombre}", annotation_position="top left")
            fig_bar.update_layout(
                height=300,
                margin=dict(t=20, b=20, l=20, r=20),
                yaxis=dict(tickformat=".0%", range=[0, max(avg_franja.max(), UMB_ALTA) * 1.4]),
                xaxis=dict(type="category", title="Franja horaria"),
                yaxis_title="Probabilidad",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, width="stretch")

            # ── 2. MAPA — franja seleccionada ─────────────────────────────
            st.markdown(f"#### Mapa — franja {franja_m}")
            probs_franja = df_matriz[franja_m]
            m = folium.Map(location=[2.935, -75.282], zoom_start=13, tiles="OpenStreetMap")
            puntos = []
            hay_mort_alta = False
            for com, p in probs_franja.items():
                info = COMUNAS.get(str(com))
                if info is None:
                    continue
                nivel = nivel_ocurrencia(p)
                texto = f"{nombre_comuna(com)}: {pct(p)} · riesgo {ADJETIVO[nivel]}"
                detalle = f"<b>{nombre_comuna(com)}</b><br>Probabilidad: {pct(p)}<br>Riesgo {ADJETIVO[nivel]}"
                borde = {"color": "#34495e", "weight": 1}
                if HAY_PLAN:
                    nivel_m = nivel_mortalidad(mortalidad_contexto(com, franja_m, dia_m, mes_m, festivo_m))
                    if nivel_m == "Alta":
                        hay_mort_alta = True
                        texto += " · mortalidad alta"
                        detalle += "<br>Mortalidad alta"
                        borde = {"color": "#7b241c", "weight": 2.5, "dash_array": "6 4"}
                    detalle += f"<br><i>{html.escape(accion(nivel, nivel_m))}</i>"
                for anillo in info["anillos"]:
                    puntos += anillo
                    folium.Polygon(
                        locations=anillo,
                        fill=True,
                        fill_color=COLORES[nivel],
                        fill_opacity=0.55,
                        tooltip=texto,
                        popup=folium.Popup(detalle, max_width=260),
                        **borde,
                    ).add_to(m)
                folium.Marker(
                    location=info["centro"],
                    icon=folium.DivIcon(html=f"<div style='font-size:12px;font-weight:700;color:#222'>{com}</div>"),
                ).add_to(m)

            # Puntos críticos: los tres con más siniestros en la franja, por comuna
            if ver_puntos:
                i_fr = FRANJAS.index(franja_m) if franja_m in FRANJAS else 0
                for com in probs_franja.index:
                    for pt in puntos_de(com, franja_m):
                        etiqueta = html.escape(pt["etiqueta"])
                        folium.CircleMarker(
                            location=[pt["lat"], pt["lon"]],
                            radius=6,
                            color="#1f2d3d",
                            weight=2,
                            fill=True,
                            fill_color="#ffffff",
                            fill_opacity=0.95,
                            tooltip=f"{etiqueta}: {pt['franjas'][i_fr]} en esta franja",
                            popup=folium.Popup(
                                f"<b>{etiqueta}</b><br>{pt['franjas'][i_fr]} siniestros con víctimas en esta franja"
                                f"<br>{pt['n']} en total · {pt['mortales']} mortales"
                                f"<br><a href='{enlace_ver(pt)}' target='_blank' rel='noopener'>Ver en Google Maps</a>"
                                f" · <a href='{enlace_ruta(pt)}' target='_blank' rel='noopener'>Cómo llegar</a>",
                                max_width=260,
                            ),
                        ).add_to(m)

            if puntos:
                lats = [q[0] for q in puntos]; lons = [q[1] for q in puntos]
                m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

            # Leyenda manual
            extra = ""
            if hay_mort_alta:
                extra += "<br><span style='color:#7b241c'>┅</span> Borde punteado: mortalidad alta"
            if ver_puntos:
                extra += "<br>○ Punto crítico"
            legend_html = f"""
            <div style="position:fixed;bottom:30px;left:30px;background:white;
                        padding:10px 14px;border-radius:8px;border:1px solid #ccc;
                        font-size:13px;z-index:1000;line-height:1.8;color:#222">
              <b>Nivel de riesgo</b><br>
              <span style="color:{COLORES['Baja']}">●</span> Bajo (&lt;{pct(UMB_MEDIA)})<br>
              <span style="color:{COLORES['Media']}">●</span> Medio ({pct(UMB_MEDIA)} a {pct(UMB_ALTA)})<br>
              <span style="color:{COLORES['Alta']}">●</span> Alto (≥{pct(UMB_ALTA)}){extra}
            </div>"""
            m.get_root().html.add_child(folium.Element(legend_html))
            st_folium(m, width=700, height=460, returned_objects=[])
            st.caption(f"Límites de las comunas: {DATOS['atribucion_mapa']}. "
                       "Toca una comuna para ver la acción recomendada." if HAY_PLAN
                       else f"Límites de las comunas: {DATOS['atribucion_mapa']}")
            if ver_puntos:
                st.caption(f"Puntos críticos: {DATOS.get('puntos_regla', '')}. Se muestran los tres con más "
                           "siniestros en la franja, por comuna. La ubicación es aproximada.")

            # ── 3. GRILLA DE CALOR — comunas × franjas ────────────────────
            st.divider()
            st.markdown("#### Grilla de riesgo — todas las comunas y franjas")

            codigos = {"Baja": 0, "Media": 1, "Alta": 2}
            z_vals  = [[codigos[nivel_ocurrencia(v)] for v in row] for row in df_matriz.values]
            texts   = [[pct(v) for v in row] for row in df_matriz.values]
            y_labels = [nombre_comuna(c) for c in df_matriz.index]

            fig_heat = go.Figure(go.Heatmap(
                z=z_vals,
                x=franjas_lista,
                y=y_labels,
                text=texts,
                texttemplate="%{text}",
                textfont=dict(size=12),
                colorscale=[
                    [0.0,   COLORES["Baja"]],  [0.333, COLORES["Baja"]],
                    [0.333, COLORES["Media"]], [0.667, COLORES["Media"]],
                    [0.667, COLORES["Alta"]],  [1.0,   COLORES["Alta"]],
                ],
                zmin=0,
                zmax=2,
                showscale=False,
                hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
            ))
            fig_heat.update_layout(
                height=max(300, len(comunas_lista) * 38),
                margin=dict(t=10, b=10, l=150, r=20),
                xaxis=dict(type="category", title="Franja horaria"),
                yaxis=dict(title="", autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_heat, width="stretch")
            st.caption(REGLA_NIVELES)

# ═══════════════════════════════════════════════════════════════════════════════
# PLAN DEL TURNO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_plan:
    st.subheader("Plan del turno")
    st.caption(
        "Ordena las comunas de una franja por su riesgo, recomienda la acción de cada una y reparte los "
        "recursos disponibles. Es una propuesta: el funcionario la revisa, la ajusta y decide."
    )

    if not HAY_PLAN:
        st.info("Para armar el plan hace falta un `datos_app.json` con las acciones y la mortalidad según el contexto.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fecha_p = st.date_input("Fecha del turno", value=HOY, format="DD/MM/YYYY", key="p_fecha")
            franja_p = st.selectbox("Franja horaria", FRANJAS_MODELO, index=min(4, len(FRANJAS_MODELO) - 1), key="p_franja")
            festivo_p = st.checkbox("Día festivo", value=es_festivo(fecha_p), key=f"p_fest_{fecha_p}")
        with col2:
            agentes = int(st.number_input("Agentes disponibles", 0, 60, value=6, key="p_agentes"))
            unidades = int(st.number_input("Unidades móviles", 0, 20, value=2, key="p_unidades"))
            zonas = int(st.number_input("Zonas a priorizar", 1, 5, value=3, key="p_zonas"))
        dia_p = nombre_dia(fecha_p, CATS.get("dia_semana", DIAS_SEMANA))

        aviso_corte(fecha_p)

        filas_turno = turno(fecha_p, franja_p, festivo_p)
        prioridades = filas_turno[:zonas]
        rep_agentes = repartir(agentes, [r["p"] for r in prioridades])
        rep_unidades = repartir(unidades, [r["p"] for r in prioridades])
        i_fr = FRANJAS.index(franja_p) if franja_p in FRANJAS else 0

        st.divider()
        st.markdown(f"#### Prioridades: {dia_p} {fecha_p:%d/%m/%Y}, franja {franja_p}")

        lineas = ["VIALAB · Plan de operativo",
                  f"{dia_p} {fecha_p:%d/%m/%Y} · franja {franja_p} · {cantidad(agentes, 'agente', 'agentes')} · "
                  f"{cantidad(unidades, 'unidad móvil', 'unidades móviles')}", ""]
        for i, (r, n_ag, n_un) in enumerate(zip(prioridades, rep_agentes, rep_unidades), 1):
            nivel = r["nivel"]
            with st.container(border=True):
                etiquetas = f":{COLOR_ST[nivel]}[{ICONO[nivel]} Ocurrencia {nivel.lower()} · {pct(r['p'])}]"
                if r["nivel_m"] == "Alta":
                    etiquetas += " · :red[Mortalidad alta]"
                st.markdown(f"**{i}. {nombre_comuna(r['comuna'])}**  \n{etiquetas}")
                c1, c2, c3 = st.columns([1, 1, 3])
                c1.metric("Agentes", n_ag)
                c2.metric("Unidades", n_un)
                c3.markdown(f"**Acción recomendada**  \n{r['accion']}")
                if nivel == "Baja":
                    st.caption("Ocurrencia baja: considera reasignar estos recursos a una zona de mayor riesgo.")
                pts = puntos_de(r["comuna"], franja_p)
                if pts:
                    with st.expander(f"Puntos críticos de la comuna ({len(pts)})"):
                        for pt in pts:
                            st.markdown(
                                f"- **{pt['etiqueta']}**: {pt['franjas'][i_fr]} en esta franja · {pt['n']} siniestros "
                                f"con víctimas en total · {pt['mortales']} mortales · "
                                f"[Ver en Google Maps]({enlace_ver(pt)}) · [Cómo llegar]({enlace_ruta(pt)})"
                            )
            lineas.append(f"{i}. {nombre_comuna(r['comuna'])}: ocurrencia {nivel.lower()} ({pct(r['p'])})"
                          f"{', mortalidad alta' if r['nivel_m'] == 'Alta' else ''}. "
                          f"{cantidad(n_ag, 'agente', 'agentes')} y {cantidad(n_un, 'unidad', 'unidades')}.")
            lineas.append(f"   Acción: {r['accion']}")
            for pt in pts:
                lineas.append(f"   Punto crítico: {pt['etiqueta']} ({enlace_ver(pt)})")
        lineas += ["", f"Historial de siniestros con víctimas al {CORTE:%d/%m/%Y}.",
                   "Propuesta generada con VIALAB. La decisión es del funcionario."]

        st.download_button("Descargar el plan (.txt)", "\n".join(lineas),
                           file_name=f"plan_vialab_{fecha_p.isoformat()}_{franja_p}.txt", mime="text/plain",
                           key="p_descargar")

        st.markdown("#### Todas las comunas en esta franja")
        st.dataframe(
            pd.DataFrame([{
                "Comuna": nombre_comuna(r["comuna"]),
                "Probabilidad": pct(r["p"]),
                "Ocurrencia": r["nivel"],
                "Mortalidad": r["nivel_m"],
                "Acción recomendada": r["accion"],
            } for r in filas_turno]),
            hide_index=True,
        )
        st.caption("Los agentes y las unidades se reparten entre las zonas priorizadas en proporción a su "
                   "probabilidad de ocurrencia. " + REGLA_NIVELES)
        meses_alerta = [MESES_NOMBRE[int(x) - 1] for x in CTX.get("meses_alerta_frecuente", [])]
        if meses_alerta:
            st.caption(
                "La mortalidad se considera alta cuando la probabilidad de que un siniestro con víctimas sea mortal "
                f"supera el {pct(UMB_MORT, 0)}. En {', '.join(meses_alerta[:-1])} y {meses_alerta[-1]} esa alerta "
                "aparece en más de la mitad de las comunas y franjas; el equipo está ajustando ese umbral."
                if len(meses_alerta) > 1 else
                "La mortalidad se considera alta cuando la probabilidad de que un siniestro con víctimas sea mortal "
                f"supera el {pct(UMB_MORT, 0)}. En {meses_alerta[0]} esa alerta aparece en más de la mitad de las "
                "comunas y franjas; el equipo está ajustando ese umbral."
            )

# ═══════════════════════════════════════════════════════════════════════════════
# CÓMO FUNCIONA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.subheader("Cómo funciona VIALAB")

    st.markdown(
        "#### Para qué sirve\n"
        "VIALAB apoya la preparación de los operativos preventivos de tránsito en Neiva. Para cada comuna y "
        "franja de 4 horas estima la probabilidad de que ocurra al menos un siniestro con víctimas. Con eso "
        "ordena las zonas, recomienda una acción y reparte los recursos del turno.\n\n"
        "Se usa antes de cada turno; no trabaja en tiempo real. La herramienta propone y el funcionario decide."
    )

    st.markdown(
        "#### Qué hace la IA y qué no\n"
        "- **Modelos de IA.** Una regresión logística estima la probabilidad de ocurrencia con la comuna, la "
        "franja, el día de la semana, el mes, si es festivo y el historial reciente de siniestros con víctimas "
        "(90 y 365 días). Otra estima qué tan probable es que un siniestro con víctimas sea mortal en ese mismo "
        "contexto.\n"
        "- **Conteos históricos.** El historial que se muestra"
        + (" y los puntos críticos son conteos" if PUNTOS else " es un conteo")
        + " de los registros, no predicciones.\n"
        "- **Reglas del equipo.** Los niveles de riesgo y la acción recomendada salen de umbrales y de una "
        "matriz de seis acciones que aprobó el equipo VIALAB."
    )

    if METRICAS:
        m25, m26 = METRICAS["ocurrencia"]["prueba_2025"], METRICAS["ocurrencia"]["fuera_2026"]
        filas_met = [
            ("Combinaciones de comuna, franja y día evaluadas", miles(m25["n"]), miles(m26["n"])),
            ("Con al menos un siniestro con víctimas", pct(m25["prevalencia"]), pct(m26["prevalencia"])),
            ("Aciertos si solo se cubre el 10 % de mayor riesgo", pct(m25["recall_10"], 0), pct(m26["recall_10"], 0)),
            ("El mismo indicador ordenando solo por el historial de 90 días",
             pct(m25["recall_10_historial"], 0), pct(m26["recall_10_historial"], 0)),
            ("Combinaciones que quedan en riesgo alto o medio", pct(m25["pct_celdas_alerta"], 0), pct(m26["pct_celdas_alerta"], 0)),
            ("Casos con víctimas que caen en riesgo alto o medio", pct(m25["pct_victimas_en_alerta"], 0), pct(m26["pct_victimas_en_alerta"], 0)),
            ("Casos con siniestro mortal que caen en riesgo alto o medio", pct(m25["pct_mortales_en_alerta"], 0), pct(m26["pct_mortales_en_alerta"], 0)),
            ("ROC-AUC (0,5 es azar y 1 es perfecto)", numero(m25["roc_auc"], 2), numero(m26["roc_auc"], 2)),
        ]
        tabla_md = "| Indicador | 2025 | 2026 (ene a sep) |\n|---|---|---|\n" + "\n".join(
            f"| {a} | {b} | {c} |" for a, b, c in filas_met)
        st.markdown(
            "#### Qué tan bien funciona\n"
            "El modelo de ocurrencia se evaluó con los datos de 2025 y de 2026 (enero a septiembre).\n\n" + tabla_md
        )
        st.markdown(
            f"**Cómo leerlo.** Un siniestro con víctimas en una comuna y franja concretas es poco frecuente: pasa en "
            f"menos del 2 % de los casos. Por eso las probabilidades son bajas y la herramienta sirve para ordenar "
            f"prioridades, no para afirmar que habrá un siniestro. Al azar, cubrir el 10 % de las combinaciones "
            f"encontraría el 10 % de los casos; el modelo encontró el {pct(m25['recall_10'], 0)} en 2025 y el "
            f"{pct(m26['recall_10'], 0)} en 2026."
        )
        s25, s26 = METRICAS["severidad"]["prueba_2025"], METRICAS["severidad"]["fuera_2026"]
        alerta_min, alerta_max = sorted([s25["pct_alerta"], s26["pct_alerta"]])
        st.markdown(
            f"**Análisis de severidad.** El modelo de la pestaña Severidad detectó el {pct(s25['sensibilidad'], 0)} de "
            f"los siniestros mortales de 2025 y el {pct(s26['sensibilidad'], 0)} de los de 2026. Para lograrlo marca "
            f"entre el {pct(alerta_min, 0)} y el {pct(alerta_max, 0)} de los siniestros con víctimas, y alrededor de "
            f"3 de cada 10 alertas corresponden a un siniestro mortal."
        )

    meses_info = [MESES_NOMBRE[int(x) - 1] for x in (CTX or {}).get("meses_alerta_frecuente", [])]
    limites = [
        "**Fuente.** Los datos vienen de los informes de tránsito que consolida el Observatorio de la Secretaría "
        "de Movilidad de Neiva. Puede haber subregistro: no todos los siniestros se reportan, y los fallecidos "
        "que reportan las clínicas llegan sin los detalles del informe.",
        "**Solo siniestros con víctimas.** Desde 2022 los choques simples ya no exigen informe, así que no son "
        "comparables en el tiempo y no se usan.",
        f"**Historial al {CORTE:%d/%m/%Y}.** Las predicciones usan el historial de ese corte hasta que se cargue "
        "uno nuevo.",
        "**Por comuna, no por cruce.** La predicción es para toda la comuna."
        + (" Los puntos críticos orientan dónde ubicarse, pero su posición es aproximada." if PUNTOS else ""),
        "**Sin datos de tráfico ni de operativos.** El modelo no conoce los aforos, el clima ni los controles que "
        "ya se hicieron.",
    ]
    if meses_info:
        nombres = (", ".join(meses_info[:-1]) + " y " + meses_info[-1]) if len(meses_info) > 1 else meses_info[0]
        limites.append(f"**Alerta de mortalidad.** En {nombres} aparece en más de la mitad de las comunas y "
                       "franjas, por el efecto del mes. El equipo está ajustando ese umbral.")
    st.markdown("#### Límites que hay que tener presentes\n" + "\n".join(f"- {x}" for x in limites))

    st.caption(
        f"Modelo de ocurrencia: {art_occ['modelo']}. Modelo de severidad: {art_sev['modelo']}. "
        f"Datos generados el {date.fromisoformat(DATOS['generado']):%d/%m/%Y}. "
        "Proyecto VIALAB, Especialización en Inteligencia Artificial, Pontificia Universidad Javeriana."
    )
