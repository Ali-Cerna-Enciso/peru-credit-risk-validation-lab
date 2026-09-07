"""02_score_validate.py — scoring + validacion. UNICO modulo que toca synthetic/.

Microdata sintetica (seed fija): E[PD] ilustrativo NO anclado a mora real.
Desde 2025 el generador aplica un beta-break (el bureau pierde pendiente fuera
de muestra): el diseno permite observar como responden Gini OOT, PSI por
variable y backtesting por grado ante un cambio de pendiente.

Modelo titular: regresion logistica. Challenger: gradient boosting (documentado,
no titular: explicabilidad ante auditoria).
Walk-forward 3 ventanas (train/valid/OOT rodantes) en validacion_estabilidad.
HL (chi2-8) + Brier global/segmento. Plan de accion para grado en rojo.

Salidas: data/synthetic/synthetic_scored.parquet (score, pd_predicha,
           default_observado, segmento)
         data/gold/validacion_{metricas.json,calibracion,backtesting,psi,
           estabilidad,migracion}.parquet (+ linaje extendido)
         informe/figures/valid_*.png (titulados "sintético")

Uso:
  .venv/Scripts/python.exe src/02_score_validate.py [--n 60000] [--seed 42]
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
bld = importlib.import_module("01_build_medallion")  # reutiliza extiende_linaje
GOLD = REPO / "data" / "gold"
SYN = REPO / "data" / "synthetic"
FIG = REPO / "informe" / "figures"
SEED = 42
B0 = -2.6  # E[PD]~12% ilustrativo, sin anclar a mora real
GRADOS = ["AAA", "AA", "A", "BBB", "BB", "B", "C"]
CORTES_PD = [0.02, 0.04, 0.07, 0.11, 0.16, 0.25]  # pueblan los 7 grados con ese nivel
VENTANAS_WF = [("W1", "2021-12", "2022-01", "2022-12", "2023-01", "2023-12"),
               ("W2", "2022-12", "2023-01", "2023-12", "2024-01", "2024-12"),
               ("W3", "2023-12", "2024-01", "2024-12", "2025-01", "2026-06")]


def genera(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vintages = pd.period_range("2018-01", "2026-06", freq="M").astype(str)
    macro = pd.read_parquet(GOLD / "macro_panel.parquet").set_index("periodo")
    tasa = macro["tasa_referencia"].reindex(vintages).ffill().bfill().to_numpy()
    tc = macro["tc_venta_prom"].reindex(vintages).ffill().bfill().to_numpy()
    tasa_z = (tasa - tasa.mean()) / tasa.std()
    tc_z = (tc - tc.mean()) / tc.std()

    idx = rng.integers(0, len(vintages), n)
    oot = vintages[idx] >= "2025-01"  # desde 2025: shock fuera de muestra
    seg = rng.choice(["consumo", "microempresa", "hipotecario"], n, p=[0.5, 0.3, 0.2])
    bureau = rng.normal(620 + (seg == "hipotecario") * 60 - (seg == "microempresa") * 40
                        - oot * 20, 80, n)
    cuota_ing = np.clip(rng.beta(2.2, 5, n) + (seg == "microempresa") * 0.08
                        + oot * 0.02, 0.01, 0.95)
    antig = np.clip(rng.exponential(36, n), 1, 240)
    b_z = (bureau - 620) / 80
    c_z = (cuota_ing - cuota_ing.mean()) / cuota_ing.std()
    a_z = (np.log(antig) - np.log(antig).mean()) / np.log(antig).std()
    bcoef = -0.70 + oot * 0.12  # el bureau pesa menos fuera de muestra
    base = (B0 + bcoef * b_z + 0.45 * c_z - 0.25 * a_z
            + 0.30 * tasa_z[idx] + 0.18 * tc_z[idx]
            + (seg == "microempresa") * 0.45 + (seg == "consumo") * 0.15
            + oot * 0.15 + oot * rng.normal(0, 0.25, n))
    pd_true = 1 / (1 + np.exp(-base))
    print(f"generado: E[PD]={pd_true.mean():.4f} (ilustrativo, sin anclar a mora real)")
    return pd.DataFrame({
        "vintage": vintages[idx], "segmento": seg, "bureau": bureau.round(0),
        "cuota_ingreso": cuota_ing.round(4), "antiguedad_m": antig.round(0),
        "default_observado": (rng.random(n) < pd_true).astype(int)})

def gini(y: np.ndarray, p: np.ndarray) -> float:
    return float(2 * roc_auc_score(y, p) - 1)


def ks_tabla(y: np.ndarray, score: np.ndarray, k: int = 20) -> tuple[float, str]:
    q = pd.qcut(score, k, labels=False, duplicates="drop")
    t = pd.DataFrame({"banda": q, "y": y, "score": score})
    g = t.groupby("banda").agg(n=("y", "size"), bads=("y", "sum"),
                               score_min=("score", "min"), score_max=("score", "max"))
    g["goods"] = g["n"] - g["bads"]
    g = g.sort_values("score_min")
    g["F_bad"] = g["bads"].cumsum() / g["bads"].sum()
    g["F_good"] = g["goods"].cumsum() / g["goods"].sum()
    g["dif"] = (g["F_bad"] - g["F_good"]).abs()
    i = g["dif"].idxmax()
    return float(g.loc[i, "dif"]), f"{g.loc[i, 'score_min']:.0f}-{g.loc[i, 'score_max']:.0f}"


def psi(esp: pd.Series, act: pd.Series) -> float:
    qs = np.unique(np.quantile(esp.to_numpy(dtype=float), np.linspace(0, 1, 11)))
    if len(qs) < 3:
        return 0.0
    e = pd.cut(esp, qs, include_lowest=True).value_counts(normalize=True).sort_index()
    o = pd.cut(act, qs, include_lowest=True).value_counts(normalize=True).sort_index()
    e = e.reindex(e.index.union(o.index)).fillna(0.0005)
    o = o.reindex(e.index).fillna(0.0005)
    return float(((o - e) * np.log(o / e)).sum())


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    d = pd.DataFrame({"y": y, "p": p})
    d["g"] = pd.qcut(d["p"], 10, labels=False, duplicates="drop")
    g = d.groupby("g").agg(o=("y", "sum"), n=("y", "size"), e=("p", "sum"))
    g = g[g["e"] * (1 - g["e"] / g["n"]) > 0]
    hl = float((((g["o"] - g["e"]) ** 2) / (g["e"] * (1 - g["e"] / g["n"]))).sum())
    return hl, float(1 - chi2.cdf(hl, max(len(g) - 2, 1)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Scoring + validacion (sintético aislado)")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    SYN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df = genera(args.n, args.seed)
    feats = ["bureau", "cuota_ingreso", "antiguedad_m"]
    X = df[feats].copy()
    X["antiguedad_m"] = np.log(X["antiguedad_m"])
    X = pd.get_dummies(pd.concat([X, df[["segmento"]]], axis=1), columns=["segmento"])
    y = df["default_observado"].to_numpy()
    oot_mask = (df["vintage"] >= "2025-01").to_numpy()
    Xtr, Xte, ytr, yte = X[~oot_mask], X[oot_mask], y[~oot_mask], y[oot_mask]

    log = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    ch = HistGradientBoostingClassifier(random_state=args.seed).fit(Xtr, ytr)
    df["pd_predicha"] = log.predict_proba(X)[:, 1]
    p_oot = df.loc[oot_mask, "pd_predicha"].to_numpy()
    g_tr = gini(ytr, log.predict_proba(Xtr)[:, 1])
    g_oot = gini(yte, p_oot)
    g_ch = gini(yte, ch.predict_proba(Xte)[:, 1])
    print(f"Gini train={g_tr:.3f} OOT={g_oot:.3f} | challenger GB OOT={g_ch:.3f}")
    if g_oot < 0.35:
        print("ALERTA: Gini OOT < 0.35.")
    df["score"] = (850 - 550 * pd.Series(df["pd_predicha"]).rank(pct=True)).round(0)
    df.to_parquet(SYN / "synthetic_scored.parquet", index=False)

    ks, banda = ks_tabla(yte, df.loc[oot_mask, "score"].to_numpy())
    print(f"KS OOT={ks:.3f} banda max={banda}")
    brier = float(brier_score_loss(yte, p_oot))
    brier_seg = {s: float(brier_score_loss(g["default_observado"], g["pd_predicha"]))
                 for s, g in df[oot_mask].groupby("segmento")}
    hl, hl_p = hosmer_lemeshow(yte, p_oot)
    print(f"Brier OOT={brier:.4f} seg={ {k: round(v, 4) for k, v in brier_seg.items()} } "
          f"HL={hl:.1f} p={hl_p:.3f}")

    d = df[oot_mask].copy()
    d["decil"] = pd.qcut(d["pd_predicha"], 10, labels=False, duplicates="drop")
    cal = d.groupby("decil").agg(n=("default_observado", "size"),
                                 pd_media=("pd_predicha", "mean"),
                                 dr=("default_observado", "mean")).reset_index()
    cal.to_parquet(GOLD / "validacion_calibracion.parquet", index=False)

    d["grado"] = pd.cut(d["pd_predicha"], [-np.inf] + CORTES_PD + [np.inf], labels=GRADOS)
    bt = d.groupby("grado", observed=True).agg(
        n=("default_observado", "size"), pd=("pd_predicha", "mean"),
        observados=("default_observado", "sum")).reset_index()
    bt["esperados"] = bt["n"] * bt["pd"]
    bt["z"] = (bt["observados"] / bt["n"] - bt["pd"]) / np.sqrt(bt["pd"] * (1 - bt["pd"]) / bt["n"])
    bt["p"] = 2 * (1 - norm.cdf(bt["z"].abs()))
    bt["semaforo"] = pd.cut(bt["p"], [-np.inf, 0.01, 0.05, np.inf],
                            labels=["rojo", "amarillo", "verde"])
    bt.to_parquet(GOLD / "validacion_backtesting.parquet", index=False)
    print(bt[["grado", "n", "pd", "z", "p", "semaforo"]].to_string(index=False))

    plan = None
    rojos = bt[bt["semaforo"] == "rojo"].sort_values("p")
    if len(rojos):
        r = rojos.iloc[0]
        plan = {"grado": str(r["grado"]), "p_valor": round(float(r["p"]), 4),
                "causa": "drift OOT: el bureau pierde pendiente fuera de muestra (beta-break diseñado 2025+, no ruido)",
                "accion": "reestimar coeficientes con ventana 2025+ (refit completo: ajuste de intercepto solo no corrige un cambio de pendiente del bureau)",
                "dueno": "responsable de validacion",
                "fecha_limite": str(dt.date.today() + dt.timedelta(days=90))}

    tr, te = df[~oot_mask], df[oot_mask]
    filas_psi = []
    for col in ["score", "bureau", "cuota_ingreso"]:
        v = psi(tr[col], te[col])
        filas_psi.append({"variable": col, "psi": round(v, 4),
                          "estado": "ok" if v < 0.10 else ("monitorear" if v < 0.25 else "reconstruir")})
    e = tr["segmento"].value_counts(normalize=True)
    o = te["segmento"].value_counts(normalize=True)
    ix = e.index.union(o.index)
    v = float((((o.reindex(ix).fillna(0.0005) - e.reindex(ix).fillna(0.0005))
                * np.log(o.reindex(ix).fillna(0.0005) / e.reindex(ix).fillna(0.0005)))).sum())
    filas_psi.append({"variable": "segmento", "psi": round(v, 4),
                      "estado": "ok" if v < 0.10 else ("monitorear" if v < 0.25 else "reconstruir")})
    psi_df = pd.DataFrame(filas_psi)
    psi_df.to_parquet(GOLD / "validacion_psi.parquet", index=False)
    peor = psi_df.loc[psi_df["psi"].idxmax()]
    print(f"peor PSI: {peor['variable']}={peor['psi']:.4f} ({peor['estado']})")

    wf = []
    for nombre, ftr, iva, fva, ioo, foo in VENTANAS_WF:
        mtr = (df["vintage"] <= ftr).to_numpy()
        mva = ((df["vintage"] >= iva) & (df["vintage"] <= fva)).to_numpy()
        moo = ((df["vintage"] >= ioo) & (df["vintage"] <= foo)).to_numpy()
        m = LogisticRegression(max_iter=2000).fit(X[mtr], y[mtr])
        for tipo, mm in [("valid", mva), ("oot", moo)]:
            pw = m.predict_proba(X[mm])[:, 1]
            gw = gini(y[mm], pw)
            kw, _ = ks_tabla(y[mm], (850 - 550 * pd.Series(pw).rank(pct=True)).to_numpy())
            alerta = "amarilla: Gini<0.40 (ventana con shock macro OOT)" if gw < 0.40 else "ok"
            wf.append({"ventana": nombre, "tipo": tipo, "n": int(mm.sum()),
                       "gini": round(gw, 4), "ks": round(kw, 4),
                       "dr": round(float(y[mm].mean()), 4), "alerta": alerta})
            print(f"{nombre}/{tipo}: n={mm.sum()} Gini={gw:.3f} KS={kw:.3f} {alerta}")
    est = pd.DataFrame(wf)
    est.to_parquet(GOLD / "validacion_estabilidad.parquet", index=False)

    d0 = df[oot_mask].copy()
    g_t = pd.cut(d0["pd_predicha"], [-np.inf] + CORTES_PD + [np.inf], labels=GRADOS)
    rng = np.random.default_rng(args.seed + 1)
    pd_t1 = np.clip(d0["pd_predicha"].to_numpy() * np.exp(rng.normal(0, 0.25, len(d0))),
                    1e-6, 0.999)
    g_t1 = pd.cut(pd_t1, [-np.inf] + CORTES_PD + [np.inf], labels=GRADOS)
    mig = pd.crosstab(g_t, g_t1, normalize="index").round(4).reset_index()
    mig.to_parquet(GOLD / "validacion_migracion.parquet", index=False)

    met = {"fecha_build": dt.date.today().isoformat(), "n": len(df), "seed": args.seed,
           "e_pd": round(float(df["default_observado"].mean()), 4),
           "e_pd_ilustrativo": True, "ancla_mora_real": None,
           "gini_train": round(g_tr, 4), "gini_oot": round(g_oot, 4),
           "gini_challenger_oot": round(g_ch, 4),
           "ks_oot": round(ks, 4), "ks_banda_max": banda, "brier_oot": round(brier, 4),
           "brier_segmento": {k: round(v, 4) for k, v in brier_seg.items()},
           "hl_stat": round(hl, 2), "hl_p": round(hl_p, 4),
           "plan_accion_rojo": plan,
           "modelo_titular": "logistica", "challenger": "HistGradientBoosting"}
    (GOLD / "validacion_metricas.json").write_text(json.dumps(met, indent=1), encoding="utf-8")
    bld.extiende_linaje({t: {"filas": len(pd.read_parquet(GOLD / t)),
                             "inputs": {"synthetic_scored.parquet": bld.sha(SYN / "synthetic_scored.parquet"),
                                        "macro_panel.parquet": "ver macro_panel.parquet"}}
                         for t in ["validacion_calibracion.parquet", "validacion_backtesting.parquet",
                                   "validacion_psi.parquet", "validacion_estabilidad.parquet",
                                   "validacion_migracion.parquet"]})

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cal["pd_media"], cal["dr"], "o-", label="observado vs predicho")
    ax.plot([0, cal["pd_media"].max()], [0, cal["pd_media"].max()], "k--", label="ideal")
    ax.set_title("Predicho vs observado por decil — SINTÉTICO (demo, HL p="
                 f"{hl_p:.3f})")
    ax.set_xlabel("PD media predicha"); ax.set_ylabel("tasa default observada")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "valid_calibracion.png", dpi=110)

    fig, ax = plt.subplots(figsize=(7, 4))
    woot = est[est.tipo == "oot"]
    ax.bar(woot["ventana"], woot["gini"], color=["y" if a.startswith("amarilla") else "g"
                                                 for a in woot["alerta"]])
    ax.axhline(0.40, color="r", ls="--", label="umbral 0.40")
    ax.set_title("Walk-forward Gini OOT por ventana — SINTÉTICO")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "valid_estabilidad.png", dpi=110)
    print(f"OK: n={len(df)} seed={args.seed} | gold/validacion_* + synthetic_scored.parquet")


if __name__ == "__main__":
    main()
