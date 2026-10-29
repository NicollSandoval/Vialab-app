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
    st.caption("Predice para todas las comunas con las condiciones seleccionadas")

    try:
        import folium
        from streamlit_folium import st_folium
        folium_ok = True
    except ImportError:
        folium_ok = False

    if not folium_ok:
        st.warning("Instala las librerías del mapa: `pip install folium streamlit-folium`")
    else:
        cats_m = art_occ["cats"]
        comunas_lista = cats_m.get("comuna", [str(i).zfill(2) for i in range(1, 11)])

        col1, col2 = st.columns(2)
        with col1:
            franja_m  = st.selectbox("Franja horaria", cats_m.get("franja_horaria", ["00-04","04-08","08-12","12-16","16-20","20-24"]), key="m_franja")
            mes_lbl_m = st.selectbox("Mes", list(MESES.keys()), key="m_mes")
            mes_m     = MESES[mes_lbl_m]
        with col2:
            dias_m   = cats_m.get("dia_semana", ["lunes","martes","miércoles","jueves","viernes","sábado","domingo"])
            dia_m    = st.selectbox("Día de la semana", dias_m, key="m_dia")
            festivo_m = st.checkbox("Día festivo", key="m_fest")

        with st.expander("Historial típico (aplica igual a todas las comunas)"):
            h90f_m = st.number_input("Víctimas 90 d — franja + comuna", 0.0, step=0.5, value=1.0, key="m_h90f")
            h90c_m = st.number_input("Víctimas 90 d — comuna",          0.0, step=1.0, value=5.0, key="m_h90c")
            h365_m = st.number_input("Víctimas 365 d — comuna",         0.0, step=1.0, value=20.0, key="m_h365")

        if st.button("Generar mapa", type="primary"):
            # Predecir para cada comuna
            resultados = []
            for com in comunas_lista:
                fila_m = {
                    "hist_victimas_90d_comuna_franja": h90f_m,
                    "hist_victimas_90d_comuna":        h90c_m,
                    "hist_victimas_365d_comuna":       h365_m,
                    "festivo":                         int(festivo_m),
                    "comuna":                          com,
                    "franja_horaria":                  franja_m,
                    "dia_semana":                      dia_m,
                    "mes":                             mes_m,
                    "franja_dia":                      franja_m + "_" + dia_m,
                }
                prob_m = predecir(art_occ, fila_m)
                coords = COORDS_NEIVA.get(str(com).lstrip("0") or "0",
                          COORDS_NEIVA.get(str(com), (2.935, -75.282)))
                resultados.append({"comuna": com, "prob": prob_m, "coords": coords})

            # Construir mapa
            m = folium.Map(location=[2.935, -75.282], zoom_start=13, tiles="CartoDB positron")

            for r in resultados:
                lat, lon = r["coords"]
                p = r["prob"]
                col_hex = color_riesgo(p)
                clase_txt, _ = semaforo(p)
                radio = 300 + int(p * 800)

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=max(12, min(30, 12 + p * 50)),
                    color=col_hex,
                    fill=True,
                    fill_color=col_hex,
                    fill_opacity=0.75,
                    popup=folium.Popup(
                        f"<b>Comuna {r['comuna']}</b><br>"
                        f"Probabilidad: {p:.1%}<br>"
                        f"{clase_txt}",
                        max_width=180,
                    ),
                    tooltip=f"Comuna {r['comuna']}: {p:.1%}",
                ).add_to(m)

            st_folium(m, width=700, height=480)

            # Tabla resumen
            st.divider()
            df_res = pd.DataFrame([
                {"Comuna": r["comuna"], "Probabilidad": f"{r['prob']:.1%}", "Riesgo": semaforo(r["prob"])[0]}
                for r in sorted(resultados, key=lambda x: -x["prob"])
            ])
            st.dataframe(df_res, hide_index=True, use_container_width=True)
