"""01_build_medallion.py — bronze → silver → gold sobre agregados REALES.

Entrada: data/raw/bcrp_*.csv (agregados reales BCRPData).
Salida:  data/bronze/*.parquet + manifest.json (inmutable, con hash input)
         data/silver/macro_mensual.parquet (validado Pandera)
         data/gold/macro_panel.parquet + linaje.json

Uso:
  .venv/Scripts/python.exe src/01_build_medallion.py [--raw DIR] [--strict]
  --strict: cualquier regla rota aborta (para CI). Sin flag: alerta y sigue.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
BRONZE = REPO / "data" / "bronze"
SILVER = REPO / "data" / "silver"
GOLD = REPO / "data" / "gold"

MESES = {"Ene": "01", "Feb": "02", "Mar": "03", "Abr": "04", "May": "05",
         "Jun": "06", "Jul": "07", "Ago": "08", "Sep": "09", "Oct": "10",
         "Nov": "11", "Dic": "12"}

# Mora "neta" BCRP negativa pre-2018 = quiebre metodologico (ver FUENTES.md).
MORA_CORTE = {"PN07746EM": "2018-01", "PN07737EM": "2018-01", "PN07736EM": "2013-12"}

PANEL_COLS = {"PD12912AM": "exp_inflacion_12m", "PD04722MM": "tasa_referencia",
              "PN01206PM": "tc_venta_prom", "PN01728AM": "pbi_var_interanual",
              "PN38705PM": "ipc_idx", "PN07746EM": "mora_sistema",
              "PN07728EM": "coloc_crec_mensual", "PN07736EM": "mora_banbif",
              "PN07737EM": "mora_mibanco"}

schema_silver_macro = DataFrameSchema({
    "periodo": Column(str, Check.str_matches(r"^\d{4}-(0[1-9]|1[0-2])$"), nullable=False),  # 1
    "serie_codigo": Column(str, Check.isin(list(PANEL_COLS)), nullable=False),              # 2
    "valor": Column(float, nullable=False),                                                 # 3
    "mora_comparable": Column(bool, nullable=False),                                        # 4
}, unique=["periodo", "serie_codigo"])  # 5 (unicidad periodo×serie)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def normaliza_periodo(s: str) -> str:
    s = s.strip().rstrip(".")
    if "-" in s and len(s) == 7 and s[:4].isdigit():  # ya YYYY-MM
        return s
    mon, anio = s.split(".")
    return f"{anio}-{MESES[mon]}"


def bronze(raw: Path = RAW) -> dict:
    BRONZE.mkdir(parents=True, exist_ok=True)
    manifest = {"fecha_build": dt.date.today().isoformat(), "archivos": {}}
    insumos = sorted(raw.glob("bcrp_*.csv"))
    if not insumos:
        sys.exit("ERROR: data/raw sin insumos (corre 00_download.py primero).")
    for p in insumos:
        df = pd.read_csv(p)
        out = BRONZE / (p.stem + ".parquet")
        df.to_parquet(out, index=False)
        manifest["archivos"][p.name] = {"sha256_16": sha(p), "filas": len(df),
                                        "bronze": out.name}
    print(f"bronze: {len(insumos)} archivo(s) inmutables + manifest.")
    return manifest


def silver(manifest: dict, strict: bool) -> pd.DataFrame:
    SILVER.mkdir(parents=True, exist_ok=True)
    largos = []
    for name in manifest["archivos"]:
        if not name.startswith("bcrp_"):
            continue
        codigo = name.removeprefix("bcrp_").removesuffix(".csv")
        df = pd.read_parquet(BRONZE / (name.removesuffix(".csv") + ".parquet"))
        df.columns = ["periodo", "valor"]
        df["serie_codigo"] = codigo
        largos.append(df)
    macro = pd.concat(largos, ignore_index=True)
    macro["periodo"] = macro["periodo"].map(normaliza_periodo)
    macro["valor"] = macro["valor"].astype(float)
    macro["mora_comparable"] = macro.apply(
        lambda r: r["periodo"] >= MORA_CORTE.get(r["serie_codigo"], "0000-00"), axis=1)
    macro = macro[["periodo", "serie_codigo", "valor", "mora_comparable"]].sort_values(["serie_codigo", "periodo"]).reset_index(drop=True)
    try:
        schema_silver_macro.validate(macro, lazy=True)
    except pa.errors.SchemaErrors as e:
        print(f"ALERTA silver macro: {len(e.failure_cases)} regla(s) rota(s).", file=sys.stderr)
        if strict:
            raise SystemExit(1) from e
    macro.to_parquet(SILVER / "macro_mensual.parquet", index=False)
    print(f"silver: macro {len(macro)} filas (validado).")
    return macro


def gold(macro: pd.DataFrame, manifest: dict) -> None:
    GOLD.mkdir(parents=True, exist_ok=True)
    panel = macro.pivot_table(index="periodo", columns="serie_codigo",
                              values="valor", aggfunc="first").rename(columns=PANEL_COLS)
    panel = panel.reindex(sorted(panel.index)).reset_index()
    panel["ipc_var12m"] = panel["ipc_idx"].pct_change(12) * 100
    panel.to_parquet(GOLD / "macro_panel.parquet", index=False)
    codigo_version = sha(Path(__file__))
    linaje = {"fecha_build": dt.date.today().isoformat(), "codigo_sha16": codigo_version,
              "tablas": {"macro_panel.parquet":
                         {"filas": len(panel), "inputs": manifest["archivos"]}}}
    (GOLD / "linaje.json").write_text(json.dumps(linaje, indent=1, default=str), encoding="utf-8")
    print(f"gold: macro_panel {panel.shape} + linaje.json (sha codigo {codigo_version}).")


def extiende_linaje(tablas: dict) -> None:
    """Añade tablas al linaje con hash input + sha de este codigo + fecha."""
    p = GOLD / "linaje.json"
    lin = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"tablas": {}}
    lin["fecha_build"] = dt.date.today().isoformat()
    lin["codigo_sha16"] = sha(Path(__file__))
    lin["tablas"].update(tablas)
    p.write_text(json.dumps(lin, indent=1, default=str), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Medallion bronze→silver→gold")
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    man = bronze(Path(args.raw))
    macro = silver(man, args.strict)
    gold(macro, man)


if __name__ == "__main__":
    main()
