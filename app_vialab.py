"""
VIALAB — App de predicción de siniestros viales
Carga los modelos generados por el notebook (modelo_ocurrencia.pkl y modelo_mortalidad.pkl)
y los insumos de datos_app.json (historial congelado, festivos, umbrales aprobados y comunas).

Ejecutar:
    streamlit run app_vialab.py
"""

import json
import unicodedata
import warnings
from datetime import date
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
PREV = DATOS["prevalencia_ocurrencia"]              # tasa base por comuna y franja
CORTE = date.fromisoformat(DATOS["historial_hasta"])
FESTIVOS = set(DATOS["festivos"])
COMUNAS = {c["id"]: c for c in DATOS["comunas"]}
DIAS_SEMANA = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
HOY = date.today()

COLORES = {"Alta": "#e74c3c", "Media": "#f39c12", "Baja": "#2ecc71"}
COLOR_ST = {"Alta": "red", "Media": "orange", "Baja": "green"}
ICONO = {"Alta": "🔴", "Media": "🟡", "Baja": "🟢"}
ADJETIVO = {"Alta": "alto", "Media": "medio", "Baja": "bajo"}

# ── Utilidades ─────────────────────────────────────────────────────────────────
def predecir(art, fila: dict) -> float:
    df = pd.DataFrame([fila])[art["cols"]]
    X  = art["pre"].transform(df)
    return float(art["mod"].predict_proba(X)[0, 1])

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
    f"Alta: {UMB_ALTA:.1%} o más (el 10 % de celdas de mayor riesgo en 2024) · "
    f"Media: {UMB_MEDIA:.1%} a {UMB_ALTA:.1%} · Baja: menos de {UMB_MEDIA:.1%}. "
    "Umbrales aprobados por el equipo el 19-oct-2026."
)

st.caption(f"Historial de siniestros con víctimas al {CORTE:%d/%m/%Y}. "
           "Las predicciones son por franja de 4 horas para preparar cada turno; no son en tiempo real.")

tab_occ, tab_sev, tab_mapa = st.tabs(["🗺️ Ocurrencia", "🚑 Severidad", "🔥 Mapa de riesgo"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OCURRENCIA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_occ:
    st.subheader("¿Habrá al menos un siniestro con víctimas?")
    st.caption("Por comuna y franja de 4 horas")

    cats = art_occ["cats"]

    col1, col2 = st.columns(2)
    with col1:
        fecha = st.date_input("Fecha del turno", value=HOY, format="DD/MM/YYYY", key="o_fecha")
        comuna = st.selectbox("Comuna", cats.get("comuna", sorted(COMUNAS)), format_func=nombre_comuna)
        franja = st.selectbox("Franja horaria", cats.get("franja_horaria", ["00-04","04-08","08-12","12-16","16-20","20-24"]))
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

    if st.button("Predecir ocurrencia", type="primary"):
        prob = predecir(art_occ, fila_ocurrencia(comuna, franja, dia, mes, festivo, h))
        nivel = nivel_ocurrencia(prob)
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Probabilidad", f"{prob:.1%}")
        with c2:
            st.markdown(f"### :{COLOR_ST[nivel]}[{ICONO[nivel]} Riesgo {ADJETIVO[nivel]}]")
            st.caption(f"{prob / PREV:.1f} veces el promedio de la ciudad ({PREV:.1%} por comuna y franja)")
        st.caption(REGLA_NIVELES)
        st.caption(f"Modelo: {art_occ['modelo']}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEVERIDAD (ANÁLISIS)
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
        franja2 = st.selectbox("Franja horaria ", cats2.get("franja_horaria", ["00-04","04-08","08-12","12-16","16-20","20-24"]))
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

    if st.button("Evaluar severidad", type="primary"):
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
        st.caption(f"Puntaje {puntaje:.2f} frente a un umbral de {umbral:.2f} · Modelo: {art_sev['modelo']}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MAPA DE RIESGO
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
        cats_m = art_occ["cats"]
        comunas_lista = cats_m.get("comuna", sorted(COMUNAS))
        franjas_lista = cats_m.get("franja_horaria", ["00-04","04-08","08-12","12-16","16-20","20-24"])

        col1, col2 = st.columns(2)
        with col1:
            fecha_m = st.date_input("Fecha del turno", value=HOY, format="DD/MM/YYYY", key="m_fecha")
            franja_m = st.selectbox("Franja horaria (para el mapa)", franjas_lista, key="m_franja")
        dia_m = nombre_dia(fecha_m, cats_m.get("dia_semana", DIAS_SEMANA))
        mes_m = f"{fecha_m.month:02d}"
        with col2:
            festivo_m = st.checkbox("Día festivo", value=es_festivo(fecha_m), key=f"m_fest_{fecha_m}")
            st.markdown(f"**Día:** {dia_m}")

        aviso_corte(fecha_m)

        if st.button("Generar visualización", type="primary"):

            # ── Calcular matriz completa: comunas × franjas ────────────────
            matriz = {}
            for fr in franjas_lista:
                matriz[fr] = [predecir(art_occ, fila_ocurrencia(com, fr, dia_m, mes_m, festivo_m, historial(com, fr)))
                              for com in comunas_lista]
            df_matriz = pd.DataFrame(matriz, index=comunas_lista)

            # ── 1. GRÁFICO DE BARRAS — riesgo promedio por franja ─────────
            st.divider()
            st.markdown("#### Riesgo promedio por franja horaria")
            avg_franja = df_matriz.mean(axis=0)
            fig_bar = go.Figure(go.Bar(
                x=avg_franja.index.tolist(),
                y=avg_franja.values.tolist(),
                marker_color=[COLORES[nivel_ocurrencia(v)] for v in avg_franja.values],
                text=[f"{v:.1%}" for v in avg_franja.values],
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
            for com, p in probs_franja.items():
                info = COMUNAS.get(str(com))
                if info is None:
                    continue
                nivel = nivel_ocurrencia(p)
                texto = f"{nombre_comuna(com)}: {p:.1%} · riesgo {ADJETIVO[nivel]}"
                for anillo in info["anillos"]:
                    puntos += anillo
                    folium.Polygon(
                        locations=anillo,
                        color="#34495e",
                        weight=1,
                        fill=True,
                        fill_color=COLORES[nivel],
                        fill_opacity=0.55,
                        tooltip=texto,
                        popup=folium.Popup(f"<b>{nombre_comuna(com)}</b><br>Probabilidad: {p:.1%}<br>Riesgo {ADJETIVO[nivel]}",
                                           max_width=200),
                    ).add_to(m)
                folium.Marker(
                    location=info["centro"],
                    icon=folium.DivIcon(html=f"<div style='font-size:12px;font-weight:700;color:#222'>{com}</div>"),
                ).add_to(m)
            if puntos:
                lats = [q[0] for q in puntos]; lons = [q[1] for q in puntos]
                m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

            # Leyenda manual
            legend_html = f"""
            <div style="position:fixed;bottom:30px;left:30px;background:white;
                        padding:10px 14px;border-radius:8px;border:1px solid #ccc;
                        font-size:13px;z-index:1000;line-height:1.8;color:#222">
              <b>Nivel de riesgo</b><br>
              <span style="color:{COLORES['Baja']}">●</span> Bajo (&lt;{UMB_MEDIA:.1%})<br>
              <span style="color:{COLORES['Media']}">●</span> Medio ({UMB_MEDIA:.1%}–{UMB_ALTA:.1%})<br>
              <span style="color:{COLORES['Alta']}">●</span> Alto (≥{UMB_ALTA:.1%})
            </div>"""
            m.get_root().html.add_child(folium.Element(legend_html))
            st_folium(m, width=700, height=460, returned_objects=[])
            st.caption(f"Límites de las comunas: {DATOS['atribucion_mapa']}")

            # ── 3. GRILLA DE CALOR — comunas × franjas ────────────────────
            st.divider()
            st.markdown("#### Grilla de riesgo — todas las comunas y franjas")

            codigos = {"Baja": 0, "Media": 1, "Alta": 2}
            z_vals  = [[codigos[nivel_ocurrencia(v)] for v in row] for row in df_matriz.values]
            texts   = [[f"{v:.1%}" for v in row] for row in df_matriz.values]
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
