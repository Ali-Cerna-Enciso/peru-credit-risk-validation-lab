"""Tests robustez a drift de inputs: Pandera bloquea + alerta (copias temporales)."""
import importlib

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors
m = importlib.import_module("src.01_build_medallion")


def test_bloquea_nulos():
    df = pd.DataFrame({"periodo": ["2026-06"], "serie_codigo": ["PN07746EM"],
                       "valor": [None], "mora_comparable": [True]})
    with pytest.raises(SchemaErrors):
        m.schema_silver_macro.validate(df, lazy=True)

def test_bloquea_periodo_duplicado():
    base = {"periodo": "2026-06", "serie_codigo": "PN07746EM",
            "valor": 2.85, "mora_comparable": True}
    with pytest.raises((SchemaError, SchemaErrors)):
        m.schema_silver_macro.validate(pd.DataFrame([base, base]), lazy=True)

@pytest.fixture(scope="module")
def cadena():
    import json as _j
    import subprocess as _sp
    import sys as _s
    from pathlib import Path as _P
    repo = _P(__file__).resolve().parents[1]
    for script in ["src/02_score_validate.py", "src/03_estres.py",
                   "reporte/05_report_sbs.py"]:
        r = _sp.run([_s.executable, str(repo / script)],
                    capture_output=True, text=True, cwd=repo)
        assert r.returncode == 0, f"{script} fallo:\n{r.stdout}\n{r.stderr}"
    return _j.loads((repo / "data" / "gold" / "estres_satelite.json").read_text(encoding="utf-8"))


def test_satelite_diagnostico(cadena):
    assert cadena["n"] >= 100 and 0 < cadena["r2"] < 1
    assert "2018" in cadena["quiebre_2018"] and "no causal" in cadena["nota"]
    for v in cadena["coeficientes"].values():
        assert {"b", "se", "p"} <= set(v) and 0 <= v["p"] <= 1


def test_escenarios_nombrados(cadena):
    from pathlib import Path
    esc = pd.read_parquet(Path(__file__).resolve().parents[1]
                          / "data" / "gold" / "estres_escenarios.parquet")
    assert list(esc["escenario"]) == ["base", "adverso", "severo"]
    adv = esc.set_index("escenario").loc["adverso"]
    assert (adv["shock_tasa_pp"], adv["shock_ipc_pp"], adv["shock_tc_pct"]) == (2.0, 1.5, 10.0)
    assert esc.set_index("escenario").loc["severo", "mora_estres"] > adv["mora_estres"] > 2.85


def test_tornado_univariado(cadena):
    from pathlib import Path
    tor = pd.read_parquet(Path(__file__).resolve().parents[1]
                          / "data" / "gold" / "estres_tornado.parquet")
    assert set(tor["variable"]) == {"tasa_referencia", "ipc_var12m", "tc_var12m"}


def test_linaje_extendido(cadena):
    import json
    from pathlib import Path
    lin = json.loads((Path(__file__).resolve().parents[1] / "data" / "gold"
                      / "linaje.json").read_text(encoding="utf-8"))
    tablas = lin["tablas"]
    assert {"macro_panel.parquet", "validacion_backtesting.parquet",
            "validacion_estabilidad.parquet", "estres_escenarios.parquet",
            "estres_tornado.parquet"} <= set(tablas)
    assert all(t.get("filas", 0) > 0 for t in tablas.values())


def test_informe_s7(cadena):
    from pathlib import Path
    md = (Path(__file__).resolve().parents[1] / "informe"
          / "informe_validacion.md").read_text(encoding="utf-8")
    assert "## 7. Riesgos y qué rompería esto" in md
    assert "mora por producto" in md and "lineal" in md


def test_anexos_diccionario(cadena):
    from pathlib import Path
    from openpyxl import load_workbook
    wb = load_workbook(Path(__file__).resolve().parents[1] / "informe" / "anexos.xlsx",
                       read_only=True)
    assert {"diccionario", "metricas", "satelite", "macro_panel",
            "estres_escenarios"} <= set(wb.sheetnames)


def test_lock_pineado(cadena):
    from pathlib import Path
    lines = (Path(__file__).resolve().parents[1] / "requirements-lock.txt").read_text().splitlines()
    assert len(lines) > 20 and not any(">=" in ln for ln in lines)
    assert any(ln.startswith("pandas==") for ln in lines)
