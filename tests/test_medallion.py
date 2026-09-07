"""Tests medallion: bronze inmutable, silver validado, gold con linaje."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


@pytest.fixture(scope="module")
def build():
    r = subprocess.run([PY, str(REPO / "src" / "01_build_medallion.py"), "--strict"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, f"pipeline fallo:\n{r.stdout}\n{r.stderr}"
    return r


def test_bronze_manifest(build):
    m = json.loads((REPO / "data" / "bronze" / "manifest.json").read_text(encoding="utf-8"))
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", m["fecha_build"])
    assert len(m["archivos"]) >= 9  # 9 series BCRP del build


def test_bronze_parquet_por_insumo(build):
    m = json.loads((REPO / "data" / "bronze" / "manifest.json").read_text(encoding="utf-8"))
    for name in m["archivos"]:
        assert (REPO / "data" / "bronze" / (Path(name).stem + ".parquet")).exists()


def test_bronze_hash_coincide_raw(build):
    import hashlib
    m = json.loads((REPO / "data" / "bronze" / "manifest.json").read_text(encoding="utf-8"))
    for name, meta in m["archivos"].items():
        h = hashlib.sha256((REPO / "data" / "raw" / name).read_bytes()).hexdigest()[:16]
        assert h == meta["sha256_16"], f"raw modificado tras bronze: {name}"


def test_silver_columnas(build):
    df = pd.read_parquet(REPO / "data" / "silver" / "macro_mensual.parquet")
    assert list(df.columns) == ["periodo", "serie_codigo", "valor", "mora_comparable"]


def test_silver_periodo_formato(build):
    df = pd.read_parquet(REPO / "data" / "silver" / "macro_mensual.parquet")
    assert df["periodo"].str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all()


def test_silver_sin_nulos_clave(build):
    df = pd.read_parquet(REPO / "data" / "silver" / "macro_mensual.parquet")
    assert df[["periodo", "serie_codigo", "valor"]].notna().all().all()


def test_silver_unicidad(build):
    df = pd.read_parquet(REPO / "data" / "silver" / "macro_mensual.parquet")
    assert not df.duplicated(["periodo", "serie_codigo"]).any()


def test_silver_series_conocidas(build):
    df = pd.read_parquet(REPO / "data" / "silver" / "macro_mensual.parquet")
    assert set(df["serie_codigo"]) <= {"PD12912AM", "PD04722MM", "PN01206PM", "PN01728AM",
                                       "PN38705PM", "PN07746EM", "PN07728EM", "PN07736EM",
                                       "PN07737EM"}


def test_mora_flag_precorte(build):
    df = pd.read_parquet(REPO / "data" / "silver" / "macro_mensual.parquet")
    vieja = df[(df.serie_codigo == "PN07746EM") & (df.periodo < "2018-01")]
    assert len(vieja) > 0 and (~vieja["mora_comparable"]).all()


def test_mora_flag_reciente(build):
    df = pd.read_parquet(REPO / "data" / "silver" / "macro_mensual.parquet")
    rec = df[(df.serie_codigo == "PN07746EM") & (df.periodo == "2026-06")]
    assert len(rec) == 1 and bool(rec.iloc[0]["mora_comparable"])


def test_gold_panel_ordenado(build):
    p = pd.read_parquet(REPO / "data" / "gold" / "macro_panel.parquet")
    assert p["periodo"].is_monotonic_increasing
    assert "ipc_var12m" in p.columns


def test_gold_mora_reciente_sin_huecos(build):
    p = pd.read_parquet(REPO / "data" / "gold" / "macro_panel.parquet")
    ventana = p[(p.periodo >= "2018-01") & (p.periodo <= "2026-06")]
    assert ventana["mora_sistema"].notna().all()


def test_gold_ipc_var12m_consistente(build):
    p = pd.read_parquet(REPO / "data" / "gold" / "macro_panel.parquet").set_index("periodo")
    calc = (p.loc["2026-08", "ipc_idx"] / p.loc["2025-08", "ipc_idx"] - 1) * 100
    assert abs(calc - p.loc["2026-08", "ipc_var12m"]) < 1e-9


def test_gold_valor_real_pineado(build):
    p = pd.read_parquet(REPO / "data" / "gold" / "macro_panel.parquet").set_index("periodo")
    assert abs(p.loc["2026-08", "exp_inflacion_12m"] - 3.05) < 1e-9
    assert abs(p.loc["2026-08", "tasa_referencia"] - 4.25) < 1e-9


def test_linaje_completo(build):
    lin = json.loads((REPO / "data" / "gold" / "linaje.json").read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{16}", lin["codigo_sha16"])
    assert lin["tablas"]["macro_panel.parquet"]["filas"] > 150
    assert len(lin["tablas"]["macro_panel.parquet"]["inputs"]) >= 9


def test_filas_silver_igual_raw(build):
    import csv
    total = sum(1 for n in (REPO / "data" / "raw").glob("bcrp_*.csv")
                for _ in open(n, encoding="utf-8")) - 9  # menos 9 cabeceras
    df = pd.read_parquet(REPO / "data" / "silver" / "macro_mensual.parquet")
    assert len(df) == total
