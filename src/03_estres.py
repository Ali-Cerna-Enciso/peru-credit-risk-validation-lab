"""03_estres.py — satélite OLS con diagnóstico + 3 escenarios (datos REALES).

Satélite (2018-01 en adelante, ventana comparable; quiebre 2018 declarado):
  mora_sistema = b0 + b1*tasa_ref + b2*ipc_var12m + b3*tc_var12m + e
Diagnóstico: R², p-valores (t aproximada), sigma residuos. Los betas son
correlaciones parciales, no efectos causales (ver §7 del informe).
Escenarios nombrados con shocks explícitos: base / adverso / severo.
Δpérdida = Δmora*LGD (LGD=45% supuesto declarado, EAD unitario).
Tornado univariado: shock +1 por variable top (tasa pp, ipc pp, tc %).

Salidas: data/gold/estres_{satelite.json,escenarios,tornado}.parquet (+ linaje)
         informe/figures/estres_tornado.png + mora_macro_real.png (reales)

Uso:
  .venv/Scripts/python.exe src/03_estres.py
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
bld = importlib.import_module("01_build_medallion")
GOLD = REPO / "data" / "gold"
FIG = REPO / "informe" / "figures"
LGD = 0.45  # supuesto declarado (informe y entrevista)
ESCENARIOS = [("base", 0.0, 0.0, 0.0),
              ("adverso", 2.0, 1.5, 10.0),
              ("severo", 3.0, 2.5, 20.0)]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(GOLD / "macro_panel.parquet").set_index("periodo").sort_index()
    p["tc_var12m"] = p["tc_venta_prom"].pct_change(12) * 100
    est = p.loc["2018-01":].dropna(subset=["mora_sistema", "tasa_referencia",
                                           "ipc_var12m", "tc_var12m"]).copy()
    cols = ["tasa_referencia", "ipc_var12m", "tc_var12m"]
    X = np.column_stack([np.ones(len(est)), est[cols].to_numpy()])
    y = est["mora_sistema"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(est) - X.shape[1]
    sigma = float(np.sqrt((resid ** 2).sum() / dof))
    cov = sigma ** 2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    pv = 2 * (1 - norm.cdf(np.abs(beta / se)))
    r2 = 1 - ((resid) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    nombres = ["b0", "b_tasa_pp", "b_ipc_pp", "b_tc_var12m"]
    sat = {"fecha_build": dt.date.today().isoformat(), "ventana": "2018-01.."
           + str(est.index[-1]), "n": len(est), "r2": round(float(r2), 4),
           "quiebre_2018": "mora neta BCRP negativa pre-2018 (cambio metodológico); "
                           "ventana arranca 2018-01",
           "coeficientes": {n: {"b": round(float(b), 4), "se": round(float(s), 4),
                                "p": round(float(v), 4)}
                            for n, b, s, v in zip(nombres, beta, se, pv)},
           "sigma_residuos": round(sigma, 4), "lgd_supuesto": LGD,
           "nota": "betas = correlación parcial, no causalidad"}
    (GOLD / "estres_satelite.json").write_text(json.dumps(sat, indent=1), encoding="utf-8")
    print(f"satélite n={len(est)} R²={r2:.3f} sigma={sigma:.3f} | "
          + " ".join(f"{n}={b:.3f}(p={v:.3f})" for n, b, v in zip(nombres[1:], beta[1:], pv[1:])))

    base = p.iloc[-1]
    mora0 = float(p["mora_sistema"].dropna().iloc[-1])
    per0 = str(p["mora_sistema"].dropna().index[-1])
    rows = []
    for nombre, dt_, di, dtc in ESCENARIOS:
        dmora = beta[1] * dt_ + beta[2] * di + beta[3] * dtc
        rows.append({"escenario": nombre, "periodo_mora_base": per0,
                     "shock_tasa_pp": dt_, "shock_ipc_pp": di, "shock_tc_pct": dtc,
                     "mora_base": round(mora0, 3), "mora_estres": round(mora0 + dmora, 3),
                     "delta_mora_pp": round(float(dmora), 3),
                     "delta_perdida_unit": round(float(dmora) / 100 * LGD, 5),
                     "tasa_base": float(base["tasa_referencia"]),
                     "ipc_var12m_base": round(float(base["ipc_var12m"]), 2),
                     "tc_base": round(float(base["tc_venta_prom"]), 4)})
    esc = pd.DataFrame(rows)
    esc.to_parquet(GOLD / "estres_escenarios.parquet", index=False)
    print(esc[["escenario", "mora_estres", "delta_mora_pp"]].to_string(index=False))

    tor = pd.DataFrame([
        {"variable": "tasa_referencia", "shock": "+1pp", "delta_mora_pp": round(float(beta[1]), 4)},
        {"variable": "ipc_var12m", "shock": "+1pp", "delta_mora_pp": round(float(beta[2]), 4)},
        {"variable": "tc_var12m", "shock": "+1%", "delta_mora_pp": round(float(beta[3]), 4)},
    ]).sort_values("delta_mora_pp")
    tor.to_parquet(GOLD / "estres_tornado.parquet", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(tor["variable"], tor["delta_mora_pp"],
            color=["g" if v < 0 else "r" for v in tor["delta_mora_pp"]])
    ax.set_title(f"Tornado univariado Δmora (pp) — REAL BCRP, base {per0}")
    ax.set_xlabel("Δ mora (pp) por +1 de shock")
    fig.tight_layout(); fig.savefig(FIG / "estres_tornado.png", dpi=110)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    pp = p.loc["2018-01":].reset_index()
    ax1.plot(pp["periodo"].iloc[::12], pp["mora_sistema"].iloc[::12], "o-",
             label="mora sistema % (BCRP)")
    ax1.plot(pp["periodo"].iloc[::12], pp["mora_banbif"].iloc[::12], "s-",
             label="mora BanBif %")
    ax1.plot(pp["periodo"].iloc[::12], pp["mora_mibanco"].iloc[::12], "D-",
             label="mora MiBanco %")
    ax2 = ax1.twinx()
    ax2.plot(pp["periodo"].iloc[::12], pp["tasa_referencia"].iloc[::12], "^-",
             color="gray", alpha=0.7, label="tasa referencia %")
    ax1.set_title("Mora bancaria vs tasa de referencia — REAL BCRPData 2018–2026")
    ax1.tick_params(axis="x", rotation=45)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "mora_macro_real.png", dpi=110)
    bld.extiende_linaje({t: {"filas": len(pd.read_parquet(GOLD / t)),
                             "inputs": {"macro_panel.parquet": "ver macro_panel.parquet"}}
                         for t in ["estres_escenarios.parquet", "estres_tornado.parquet"]})
    print("OK: 3 escenarios + tornado univariado + 2 figuras reales.")


if __name__ == "__main__":
    main()
