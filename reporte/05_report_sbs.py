"""05_report_sbs.py — informe de validación (md + xlsx, 7 secciones).

Toma cifras únicamente de gold/*.parquet ya validados (este script redacta,
no calcula). Genera informe_validacion.md + anexos.xlsx (1 sheet por tabla
gold + metricas/satelite + diccionario).

Uso:
  .venv/Scripts/python.exe reporte/05_report_sbs.py [--gold data/gold] [--out informe/]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TABLAS = ["macro_panel", "validacion_calibracion", "validacion_backtesting",
          "validacion_psi", "validacion_estabilidad", "validacion_migracion",
          "estres_escenarios", "estres_tornado"]
NECESARIOS = [f"{t}.parquet" for t in TABLAS] + ["validacion_metricas.json",
                                                 "estres_satelite.json", "linaje.json"]
DICCIONARIO = [
    ("macro_panel", "periodo", "mes AAAA-MM, panel mensual 2010-.."),
    ("macro_panel", "mora_sistema", "% cartera atrasada neta/colocaciones, empresas bancarias (BCRP)"),
    ("macro_panel", "tasa_referencia", "% política monetaria BCRP"),
    ("validacion_calibracion", "pd_media/dr", "PD media predicha vs tasa default observada por decil OOT"),
    ("validacion_backtesting", "z/p/semaforo", "binomial por grado H0:DR=PD; verde p≥0.05"),
    ("validacion_psi", "psi/estado", "deriva train-vs-OOT; <0.10 ok, <0.25 monitorear"),
    ("validacion_estabilidad", "ventana/tipo/gini/ks/alerta", "walk-forward W1-W3 + ventana"),
    ("validacion_migracion", "grados t/t+1", "simulacion ilustrativa: ruido sobre el mismo corte, no transicion t a t+1 de cartera"),
    ("estres_escenarios", "mora_estres/delta_mora_pp", "base/adverso/severo, satélite OLS"),
    ("estres_tornado", "delta_mora_pp", "sensibilidad univariada +1 por variable"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Informe auto SBS (md+xlsx)")
    ap.add_argument("--gold", default=str(REPO / "data" / "gold"))
    ap.add_argument("--out", default=str(REPO / "informe"))
    args = ap.parse_args()
    gold, out = Path(args.gold), Path(args.out)
    faltan = [n for n in NECESARIOS if not (gold / n).exists()]
    if faltan:
        raise SystemExit(f"Faltan insumos gold (corre 01/02/03 primero): {faltan}")
    out.mkdir(parents=True, exist_ok=True)

    met = json.loads((gold / "validacion_metricas.json").read_text(encoding="utf-8"))
    sat = json.loads((gold / "estres_satelite.json").read_text(encoding="utf-8"))
    lin = json.loads((gold / "linaje.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(gold / "macro_panel.parquet").set_index("periodo").sort_index()
    bt = pd.read_parquet(gold / "validacion_backtesting.parquet")
    psi = pd.read_parquet(gold / "validacion_psi.parquet")
    esc = pd.read_parquet(gold / "estres_escenarios.parquet")
    est = pd.read_parquet(gold / "validacion_estabilidad.parquet")
    ult = panel.dropna(subset=["mora_sistema"]).iloc[-1]

    sem = bt.set_index("grado")["semaforo"].to_dict()
    peor_psi = psi.loc[psi["psi"].idxmax()]
    adv = esc.set_index("escenario").loc["adverso"]
    sev = esc.set_index("escenario").loc["severo"]
    co = sat["coeficientes"]
    plan = met["plan_accion_rojo"] or {}
    ntbl = len([t for t in lin["tablas"] if t.endswith(".parquet")])
    woot = est[est["tipo"] == "oot"]
    wf_rango = f"{woot['gini'].min():.2f}–{woot['gini'].max():.2f}"
    rojos = bt[bt["semaforo"] == "rojo"]
    psi_bureau = float(psi.set_index("variable").loc["bureau", "psi"])
    psi_vs_beta = ""
    if len(rojos):
        r0 = rojos.sort_values("p").iloc[0]
        psi_vs_beta = (f" Lectura PSI vs beta: bureau PSI {psi_bureau:.3f} (ok) "
            f"mientras el grado {r0['grado']} cae en rojo (z={r0['z']:.2f}, "
            f"p={r0['p']:.4f}). El PSI mide cambio de distribucion, no cambio de "
            f"pendiente: el backtesting por grado no se sustituye con PSI.")

    md = f"""# Informe de validación — riesgo de crédito (build {met["fecha_build"]})

> Cartera: agregados reales BCRPData (origen SBS) + demo individual sintética declarada.
> Linaje: código sha `{lin["codigo_sha16"]}`, {ntbl} tablas gold con hash input.

## 1. Portafolio y mora (REAL BCRPData)

Mora sistema empresas bancarias {ult.name}: **{ult["mora_sistema"]:.2f}%**
(BanBif {ult["mora_banbif"]:.2f}%, MiBanco {ult["mora_mibanco"]:.2f}%).
Tasa de referencia {panel.iloc[-1]["tasa_referencia"]:.2f}%, TC S/ {panel.iloc[-1]["tc_venta_prom"]:.2f}.
Niveles de mora comparables desde 2018-01 (quiebre metodológico BCRP previo, ver FUENTES).

## 2. Discriminación (SINTÉTICO seed {met["seed"]}, n={met["n"]}, E[PD]={met["e_pd"]:.4f} ilustrativo NO anclado)

Gini train {met["gini_train"]:.3f} / OOT **{met["gini_oot"]:.3f}** (umbral 0.40;
deriva diseñada 2025+: bureau pierde poder en OOT, por eso train>OOT).
Challenger gradient boosting OOT {met["gini_challenger_oot"]:.3f}: titular sigue
siendo logística por explicabilidad ante auditoría. KS OOT **{met["ks_oot"]:.3f}**
(ventiles), banda del máximo {met["ks_banda_max"]}.

## 3. Calibración y backtesting (SINTÉTICO)

Brier OOT {met["brier_oot"]:.4f} (consumo {met["brier_segmento"]["consumo"]:.4f},
hipotecario {met["brier_segmento"]["hipotecario"]:.4f},
microempresa {met["brier_segmento"]["microempresa"]:.4f}).
Hosmer-Lemeshow {met["hl_stat"]:.1f}, p={met["hl_p"]:.3f} (rechaza ajuste perfecto:
coherente con deriva OOT). Backtesting binomial por grado (H0: DR=PD):
{", ".join(f"{g}={s}" for g, s in sem.items())}.
Plan acción grado {plan.get("grado")} (p={plan.get("p_valor")}): {plan.get("causa")}.
Acción: {plan.get("accion")}. Dueño: {plan.get("dueno")}. Límite: {plan.get("fecha_limite")}.

## 4. Estabilidad (SINTÉTICO + ventana real)

Walk-forward W1–W3 (train/valid/OOT rodantes) en anexos: Gini OOT {wf_rango}
(amarilla si <0.40 con causa macro escrita). Peor PSI train-vs-OOT:
**{peor_psi["variable"]}={peor_psi["psi"]:.4f} ({peor_psi["estado"]})** — deriva
visible pero contenida (beta-break del bureau diseñado 2025+).{psi_vs_beta}

## 5. Estrés macro (REAL, satélite OLS {sat["ventana"]}, n={sat["n"]}, R²={sat["r2"]})

b_tasa={co["b_tasa_pp"]["b"]:.3f} (p={co["b_tasa_pp"]["p"]:.3f}),
b_ipc={co["b_ipc_pp"]["b"]:.3f} (p={co["b_ipc_pp"]["p"]:.3f}, no significativo:
correlación parcial, no causal),
b_tc={co["b_tc_var12m"]["b"]:.4f} (p={co["b_tc_var12m"]["p"]:.3f}).
Adverso (tasa+200pb, IPC+1.5pp, TC+10%): mora {adv["mora_base"]:.2f}% → **{adv["mora_estres"]:.2f}%**.
Severo (+300pb, +2.5pp, +20%): **{sev["mora_estres"]:.2f}%** (Δ{sev["delta_mora_pp"]:.2f}pp, LGD 45%).

## 6. Calidad, gobierno y límites

Bronze inmutable con hash sha por insumo; silver validado con Pandera (5 reglas,
toda rota = alerta); gold con linaje (hash input + sha código por tabla).
Mora por producto (consumo/hipotecario/microempresa) fuera de alcance: el
estrés usa mora sistema y el LGD es supuesto (45%). Sintético solo en
`data/synthetic/`, todo gráfico suyo titulado "sintético".

## 7. Riesgos y qué rompería esto

- **Sin mora por producto** no hay consumo/hipotecario/microempresa ni
  provisiones reales: el estrés usa mora sistema y el LGD es supuesto (45%).
- **Lo sintético solo prueba método individual** (Gini/KS/backtesting): sus
  niveles (E[PD]~12%) son ilustrativos y su deriva es diseñada; no describen
  ninguna cartera peruana real.
- **El satélite es lineal y corto** (n={sat["n"]}, R²={sat["r2"]}, IPC no
  significativo): un quiebre estructural o no-linealidad fuerte lo deja fuera
  de rango; el sigma de residuos ({sat["sigma_residuos"]:.3f}pp) es el piso de
  su error.
"""
    (out / "informe_validacion.md").write_text(md, encoding="utf-8")

    with pd.ExcelWriter(out / "anexos.xlsx", engine="openpyxl") as w:
        for name in TABLAS:
            pd.read_parquet(gold / f"{name}.parquet").to_excel(w, sheet_name=name[:31], index=False)
        pd.DataFrame([met]).to_excel(w, sheet_name="metricas", index=False)
        pd.DataFrame([sat["coeficientes"]]).to_excel(w, sheet_name="satelite", index=False)
        pd.DataFrame(DICCIONARIO, columns=["tabla", "columna", "descripcion"]).to_excel(
            w, sheet_name="diccionario", index=False)
    print(f"OK: {out.relative_to(REPO)}/informe_validacion.md + anexos.xlsx "
          f"({len(TABLAS)} tablas + diccionario, 0 inventadas).")


if __name__ == "__main__":
    main()
