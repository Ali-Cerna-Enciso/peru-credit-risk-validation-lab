"""Tests validacion: metricas, sintético aislado y umbrales banca PE."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
GOLD = REPO / "data" / "gold"


@pytest.fixture(scope="module")
def run():
    r = subprocess.run([sys.executable, str(REPO / "src" / "02_score_validate.py")],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, f"02 fallo:\n{r.stdout}\n{r.stderr}"
    return json.loads((GOLD / "validacion_metricas.json").read_text(encoding="utf-8"))


def test_gini_oot_umbral(run):
    assert run["gini_oot"] >= 0.40, "bajo umbral banca retail PE"


def test_gini_metricas_coherentes(run):
    assert 0 < run["gini_oot"] < 1 and 0 < run["gini_train"] < 1
    assert 0 < run["gini_challenger_oot"] < 1


def test_ks_positivo_con_banda(run):
    assert run["ks_oot"] > 0.2 and "-" in run["ks_banda_max"]


def test_challenger_documentado(run):
    assert run["modelo_titular"] == "logistica" and "gini_challenger_oot" in run


def test_sintetico_columnas_y_n(run):
    df = pd.read_parquet(REPO / "data" / "synthetic" / "synthetic_scored.parquet")
    assert {"score", "pd_predicha", "default_observado", "segmento",
            "vintage", "bureau"} <= set(df.columns)
    assert len(df) == run["n"] and df["pd_predicha"].between(0, 1).all()

def test_e_pd_no_anclado(run):
    assert run["e_pd_ilustrativo"] is True and run["ancla_mora_real"] is None
    p = pd.read_parquet(GOLD / "macro_panel.parquet").dropna(subset=["mora_sistema"])
    assert abs(run["e_pd"] - p.iloc[-1]["mora_sistema"] / 100) > 0.05



def test_backtesting_semaforo(run):
    bt = pd.read_parquet(GOLD / "validacion_backtesting.parquet")
    assert set(bt["semaforo"].dropna()) <= {"verde", "amarillo", "rojo"}
    assert bt["z"].notna().all() and (bt["n"] > 0).all()


def test_calibracion_deciles(run):
    cal = pd.read_parquet(GOLD / "validacion_calibracion.parquet")
    assert cal["pd_media"].is_monotonic_increasing
    assert ((cal["dr"] - cal["pd_media"]).abs() < 0.05).all()


def test_psi_estados(run):
    psi = pd.read_parquet(GOLD / "validacion_psi.parquet")
    assert set(psi["estado"]) <= {"ok", "monitorear", "reconstruir"}
    assert len(psi) == 4 and psi["psi"].ge(0).all()


def test_estabilidad_y_migracion(run):
    est = pd.read_parquet(GOLD / "validacion_estabilidad.parquet")
    assert len(est) > 5 and est["gini"].notna().all()
    mig = pd.read_parquet(GOLD / "validacion_migracion.parquet")
    num = mig.select_dtypes("number")
    assert ((num.sum(axis=1) - 1).abs() < 0.02).all()


def test_pd_en_rango(run):
    df = pd.read_parquet(REPO / "data" / "synthetic" / "synthetic_scored.parquet")
    assert df["pd_predicha"].between(0, 1).all()


def test_walk_forward_3_ventanas(run):
    est = pd.read_parquet(GOLD / "validacion_estabilidad.parquet")
    assert set(est["ventana"]) == {"W1", "W2", "W3"}
    assert set(est["tipo"]) == {"valid", "oot"} and len(est) == 6
    assert est["gini"].between(0, 1).all() and est["ks"].between(0, 1).all()


def test_wf_alerta_regla(run):
    est = pd.read_parquet(GOLD / "validacion_estabilidad.parquet")
    mal = est[est["gini"] < 0.40]
    assert ((est["alerta"] == "ok") | (est["alerta"].str.startswith("amarilla"))).all()
    assert (mal["alerta"].str.startswith("amarilla")).all()


def test_plan_accion_si_hay_rojo(run):
    bt = pd.read_parquet(GOLD / "validacion_backtesting.parquet")
    assert set(bt["semaforo"].dropna()) <= {"verde", "amarillo", "rojo"}
    if (bt["semaforo"] == "rojo").any():
        plan = run["plan_accion_rojo"]
        assert plan and {"grado", "p_valor", "causa", "accion", "dueno",
                         "fecha_limite"} <= set(plan)


def test_hl_y_brier_segmento(run):
    assert run["hl_stat"] > 0 and 0 <= run["hl_p"] <= 1
    assert set(run["brier_segmento"]) == {"consumo", "hipotecario", "microempresa"}
    assert all(0 < v < 0.5 for v in run["brier_segmento"].values())
