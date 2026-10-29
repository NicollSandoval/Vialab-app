"""
VIALAB — App de predicción de siniestros viales
Carga los modelos generados por el notebook (modelo_ocurrencia.pkl y modelo_mortalidad.pkl).

Ejecutar:
    streamlit run app_vialab.py
"""

import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Configuración ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VIALAB · Predicción de Siniestros",
    page_icon="🚦",
    layout="centered",
)

DIR = Path(__file__).parent  # carpeta donde está este archivo

# ── Carga de modelos ───────────────────────────────────────────────────────────
@st.cache_resource
def cargar(nombre):
    ruta = DIR / nombre
    if not ruta.exists():
        return None
    return joblib.load(ruta)

art_occ = cargar("modelo_ocurrencia.pkl")
art_sev = cargar("modelo_mortalidad.pkl")

# ── Utilidades ─────────────────────────────────────────────────────────────────
def predecir(art, fila: dict) -> float:
    df = pd.DataFrame([fila])[art["cols"]]
    X  = art["pre"].transform(df)
    return float(art["mod"].predict_proba(X)[0, 1])

def semaforo(p: float, umbral: float | None = None):
    if umbral is not None:
        clase = "🔴 MORTAL" if p >= umbral else "🟢 No mortal"
        color = "red" if p >= umbral else "green"
    else:
        if p < 0.10:
            clase, color = "🟢 Riesgo bajo", "green"
        elif p < 0.25:
            clase, color = "🟡 Riesgo medio", "orange"
        else:
            clase, color = "🔴 Riesgo alto", "red"
    return clase, color

def color_riesgo(p: float) -> str:
    if p < 0.10:
        return "#2ecc71"   # verde
    elif p < 0.25:
        return "#f39c12"   # naranja
    else:
        return "#e74c3c"   # rojo

MESES = {
    "01 Enero": "01", "02 Febrero": "02", "03 Marzo": "03",
    "04 Abril": "04", "05 Mayo": "05", "06 Junio": "06",
    "07 Julio": "07", "08 Agosto": "08", "09 Septiembre": "09",
    "10 Octubre": "10", "11 Noviembre": "11", "12 Diciembre": "12",
}

# Coordenadas aproximadas del centro de cada comuna de Neiva
COORDS_NEIVA = {
    "1":  (2.960, -75.285), "01": (2.960, -75.285),
    "2":  (2.952, -75.270), "02": (2.952, -75.270),
    "3":  (2.942, -75.258), "03": (2.942, -75.258),
    "4":  (2.932, -75.272), "04": (2.932, -75.272),
    "5":  (2.927, -75.284), "05": (2.927, -75.284),
    "6":  (2.918, -75.292), "06": (2.918, -75.292),
    "7":  (2.908, -75.298), "07": (2.908, -75.298),
    "8":  (2.944, -75.297), "08": (2.944, -75.297),
    "9":  (2.955, -75.302), "09": (2.955, -75.302),
    "10": (2.912, -75.268),
}

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🚦 VIALAB · Predicción de Siniestros")
st.caption("Modelos entrenados con datos de Neiva · Sprint 5")

if art_occ is None or art_sev is None:
    st.error(
        "No se encontraron los archivos de modelos.\n\n"
        "Ejecuta la celda **'Guardar modelos para Streamlit'** del notebook "
        "y copia `modelo_ocurrencia.pkl` y `modelo_mortalidad.pkl` "
        "en la misma carpeta que este archivo."
    )
    st.stop()

tab_occ, tab_sev, tab_mapa = st.tabs(["🗺️ Ocurrencia", "💀 Mortalidad", "🔥 Mapa de riesgo"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OCURRENCIA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_occ:
    st.subheader("¿Habrá al menos un siniestro con víctimas?")
    st.caption("Por comuna y franja de 4 horas")

    cats = art_occ["cats"]

    col1, col2 = st.columns(2)
    with col1:
        comuna = st.selectbox("Comuna", cats.get("comuna", [str(i).zfill(2) for i in range(1, 11)]))
        franja = st.selectbox("Franja horaria", cats.get("franja_horaria", ["00-04","04-08","08-12","12-16","16-20","20-24"]))
        mes_label = st.selectbox("Mes", list(MESES.keys()))
        mes = MESES[mes_label]
    with col2:
        dias = cats.get("dia_semana", ["lunes","martes","miércoles","jueves","viernes","sábado","domingo"])
        dia = st.selectbox("Día de la semana", dias)
        festivo = st.checkbox("Día festivo")
        st.markdown("**Historial de siniestros (del panel)**")
        h90f = st.number_input("Víctimas 90 d — franja + comuna", 0.0, step=0.5, value=1.0)
        h90c = st.number_input("Víctimas 90 d — comuna", 0.0, step=1.0, value=5.0)
        h365 = st.number_input("Víctimas 365 d — comuna", 0.0, step=1.0, value=20.0)

    if st.button("Predecir ocurrencia", type="primary"):
        fila = {
            "hist_victimas_90d_comuna_franja": h90f,
            "hist_victimas_90d_comuna":        h90c,
            "hist_victimas_365d_comuna":       h365,
            "festivo":                         int(festivo),
            "comuna":                          comuna,
            "franja_horaria":                  franja,
            "dia_semana":                      dia,
            "mes":                             mes,
            "franja_dia":                      franja + "_" + dia,
        }
        prob = predecir(art_occ, fila)
        clase, color = semaforo(prob)
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Probabilidad", f"{prob:.1%}")
        with c2:
            st.markdown(f"### :{color}[{clase}]")
        st.progress(min(prob * 4, 1.0))
        st.caption(f"Prevalencia base: ~2 % · Modelo: {art_occ['modelo']}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MORTALIDAD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sev:
    st.subheader("¿El siniestro con víctimas será mortal?")
    st.caption("Dado que ya ocurrió un siniestro con víctimas")

    cats2 = art_sev["cats"]
    umbral = art_sev.get("umbral", 0.5)

    col1, col2 = st.columns(2)
    with col1:
        franja2 = st.selectbox("Franja horaria ", cats2.get("franja_horaria", ["00-04","04-08","08-12","12-16","16-20","20-24"]))
        dias2 = cats2.get("nombre_dia", ["lunes","martes","miércoles","jueves","viernes","sábado","domingo"])
        dia2 = st.selectbox("Día de la semana ", dias2)
        mes_label2 = st.selectbox("Mes ", list(MESES.keys()))
        mes2 = MESES[mes_label2]
        comuna2 = st.selectbox("Comuna ", cats2.get("comuna_norm", [str(i).zfill(2) for i in range(1, 11)]))
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

    if st.button("Predecir mortalidad", type="primary"):
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
        prob2 = predecir(art_sev, fila2)
        clase2, color2 = semaforo(prob2, umbral)
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Probabilidad", f"{prob2:.1%}")
            st.caption(f"Umbral: {umbral:.2f}")
        with c2:
            st.markdown(f"### :{color2}[{clase2}]")
        st.progress(prob2)
        st.caption(f"Prevalencia base: ~15 % · Modelo: {art_sev['modelo']}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MAPA DE RIESGO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mapa:
    st.subheader("Mapa de riesgo de ocurrencia por comuna")
    st.caption("Predice para todas las comunas y franjas con las condiciones seleccionadas")

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
        comunas_lista = cats_m.get("comuna", [str(i).zfill(2) for i in range(1, 11)])
        franjas_lista = cats_m.get("franja_horaria", ["00-04","04-08","08-12","12-16","16-20","20-24"])

        col1, col2 = st.columns(2)
        with col1:
            franja_m  = st.selectbox("Franja horaria (para el mapa)", franjas_lista, key="m_franja")
            mes_lbl_m = st.selectbox("Mes", list(MESES.keys()), key="m_mes")
            mes_m     = MESES[mes_lbl_m]
        with col2:
            dias_m    = cats_m.get("dia_semana", ["lunes","martes","miércoles","jueves","viernes","sábado","domingo"])
            dia_m     = st.selectbox("Día de la semana", dias_m, key="m_dia")
            festivo_m = st.checkbox("Día festivo", key="m_fest")

        with st.expander("Historial típico (aplica igual a todas las comunas)"):
            h90f_m = st.number_input("Víctimas 90 d — franja + comuna", 0.0, step=0.5, value=1.0, key="m_h90f")
            h90c_m = st.number_input("Víctimas 90 d — comuna",          0.0, step=1.0, value=5.0, key="m_h90c")
            h365_m = st.number_input("Víctimas 365 d — comuna",         0.0, step=1.0, value=20.0, key="m_h365")

        if st.button("Generar visualización", type="primary"):

            # ── Calcular matriz completa: comunas × franjas ────────────────
            matriz = {}
            for fr in franjas_lista:
                col_probs = []
                for com in comunas_lista:
                    fila_m = {
                        "hist_victimas_90d_comuna_franja": h90f_m,
                        "hist_victimas_90d_comuna":        h90c_m,
                        "hist_victimas_365d_comuna":       h365_m,
                        "festivo":                         int(festivo_m),
                        "comuna":                          com,
                        "franja_horaria":                  fr,
                        "dia_semana":                      dia_m,
                        "mes":                             mes_m,
                        "franja_dia":                      fr + "_" + dia_m,
                    }
                    col_probs.append(predecir(art_occ, fila_m))
                matriz[fr] = col_probs

            df_matriz = pd.DataFrame(matriz, index=comunas_lista)

            # ── 1. GRÁFICO DE BARRAS — riesgo promedio por franja ─────────
            st.divider()
            st.markdown("#### Riesgo promedio por franja horaria")
            avg_franja = df_matriz.mean(axis=0)
            fig_bar = go.Figure(go.Bar(
                x=avg_franja.index.tolist(),
                y=avg_franja.values.tolist(),
                marker_color=[color_riesgo(v) for v in avg_franja.values],
                text=[f"{v:.1%}" for v in avg_franja.values],
                textposition="outside",
            ))
            fig_bar.update_layout(
                height=280,
                margin=dict(t=20, b=20, l=20, r=20),
                yaxis=dict(tickformat=".0%", range=[0, max(avg_franja.max() * 1.4, 0.05)]),
                xaxis_title="Franja horaria",
                yaxis_title="Probabilidad",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── 2. MAPA — franja seleccionada ─────────────────────────────
            st.markdown(f"#### Mapa — franja {franja_m}")
            probs_franja = df_matriz[franja_m]
            m = folium.Map(location=[2.935, -75.282], zoom_start=13, tiles="CartoDB positron")

            for com, p in probs_franja.items():
                coords = COORDS_NEIVA.get(str(com).lstrip("0") or "0",
                          COORDS_NEIVA.get(str(com), (2.935, -75.282)))
                lat, lon = coords
                col_hex  = color_riesgo(p)
                clase_txt, _ = semaforo(p)
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=max(14, min(32, 14 + p * 60)),
                    color=col_hex,
                    fill=True,
                    fill_color=col_hex,
                    fill_opacity=0.8,
                    popup=folium.Popup(
                        f"<b>Comuna {com}</b><br>Probabilidad: {p:.1%}<br>{clase_txt}",
                        max_width=180,
                    ),
                    tooltip=f"C{com}: {p:.1%}",
                ).add_to(m)

            # Leyenda manual
            legend_html = """
            <div style="position:fixed;bottom:30px;left:30px;background:white;
                        padding:10px 14px;border-radius:8px;border:1px solid #ccc;
                        font-size:13px;z-index:1000;line-height:1.8">
              <b>Nivel de riesgo</b><br>
              <span style="color:#2ecc71">●</span> Bajo (&lt;10%)<br>
              <span style="color:#f39c12">●</span> Medio (10–25%)<br>
              <span style="color:#e74c3c">●</span> Alto (&gt;25%)
            </div>"""
            m.get_root().html.add_child(folium.Element(legend_html))
            st_folium(m, width=700, height=460)

            # ── 3. GRILLA DE CALOR — comunas × franjas ────────────────────
            st.divider()
            st.markdown("#### Grilla de riesgo — todas las comunas y franjas")

            z_vals  = df_matriz.values.tolist()
            texts   = [[f"{v:.1%}" for v in row] for row in df_matriz.values]
            y_labels = [f"Comuna {c}" for c in df_matriz.index]

            fig_heat = go.Figure(go.Heatmap(
                z=z_vals,
                x=franjas_lista,
                y=y_labels,
                text=texts,
                texttemplate="%{text}",
                textfont=dict(size=12),
                colorscale=[
                    [0.0,  "#2ecc71"],
                    [0.2,  "#f1c40f"],
                    [0.5,  "#f39c12"],
                    [1.0,  "#e74c3c"],
                ],
                zmin=0,
                zmax=0.5,
                showscale=True,
                colorbar=dict(
                    title="Prob.",
                    tickformat=".0%",
                    thickness=14,
                ),
            ))
            fig_heat.update_layout(
                height=max(300, len(comunas_lista) * 38),
                margin=dict(t=10, b=10, l=90, r=20),
                xaxis_title="Franja horaria",
                yaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

