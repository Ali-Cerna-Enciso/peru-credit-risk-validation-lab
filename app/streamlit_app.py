"""Monitoreo de validación — demo local Streamlit (costo $0).

Lee SOLO gold/*.parquet + metricas (nada se recalcula en la app).
Filtros: segmento/vintage sobre la demo sintética; periodo sobre panel real.
Alertas: Gini OOT <0.35, PSI>=0.25, grado en rojo, mora comparable.

Uso local:
  .venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
Demo free: Streamlit Community Cloud apuntando a este archivo (ver deploy/).
"""

from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
GOLD = REPO / "data" / "gold"
SYN = REPO / "data" / "synthetic"


@st.cache_data
def carga():
    import json
    met = json.loads((GOLD / "validacion_metricas.json").read_text(encoding="utf-8"))
    return {
        "met": met,
        "panel": pd.read_parquet(GOLD / "macro_panel.parquet"),
        "bt": pd.read_parquet(GOLD / "validacion_backtesting.parquet"),
        "psi": pd.read_parquet(GOLD / "validacion_psi.parquet"),
        "est": pd.read_parquet(GOLD / "validacion_estabilidad.parquet"),
        "esc": pd.read_parquet(GOLD / "estres_escenarios.parquet"),
        "syn": pd.read_parquet(SYN / "synthetic_scored.parquet"),
    }


d = carga()
met = d["met"]
st.caption(f"build {met['fecha_build']} · E[PD] sintético {met['e_pd']:.4f} "
           f"(ilustrativo, no anclado) · lo sintético va etiquetado")

alertas = []
if met["gini_oot"] < 0.35:
    alertas.append(f"Gini OOT {met['gini_oot']:.3f} < 0.35: reconstruir.")
for _, r in d["psi"].iterrows():
    if r["estado"] == "reconstruir":
        alertas.append(f"PSI {r['variable']}={r['psi']:.3f}: reconstruir.")
rojos = d["bt"][d["bt"]["semaforo"] == "rojo"]
if len(rojos):
    alertas.append(f"Grados en rojo: {', '.join(rojos['grado'].astype(str))}.")
if alertas:
    for a in alertas:
        st.error("ALERTA " + a)
else:
    st.success("Sin alertas: Gini, PSI y backtesting dentro de umbrales.")

c1, c2, c3 = st.columns(3)
c1.metric("Gini OOT (sintético)", f"{met['gini_oot']:.3f}", f"umbral 0.40")
c2.metric("KS OOT (sintético)", f"{met['ks_oot']:.3f}", f"banda {met['ks_banda_max']}")
c3.metric("Mora sistema real", f"{d['panel'].dropna(subset=['mora_sistema']).iloc[-1]['mora_sistema']:.2f}%")

seg = st.multiselect("Segmento (demo sintética)", sorted(d["syn"]["segmento"].unique()),
                     default=sorted(d["syn"]["segmento"].unique()))
vin = st.slider("Vintage desde", "2018-01", "2026-06", "2024-01")
sub = d["syn"][d["syn"]["segmento"].isin(seg) & (d["syn"]["vintage"] >= vin)]
st.write(f"DR observado (sintético, filtro): {sub['default_observado'].mean():.4f} · n={len(sub)}")
st.bar_chart(sub.groupby("vintage")["default_observado"].mean())

per = st.slider("Panel real desde", "2010-01", "2026-08", "2018-01")
st.line_chart(d["panel"][d["panel"]["periodo"] >= per].set_index("periodo")
              [["mora_sistema", "mora_banbif", "mora_mibanco"]])
st.subheader("Backtesting por grado (sintético)")
st.dataframe(d["bt"])
st.subheader("PSI train vs OOT (sintético)")
st.dataframe(d["psi"])
st.subheader("Escenarios de estrés (real)")
st.dataframe(d["esc"])
