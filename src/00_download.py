"""00_download.py — descarga agregados públicos frescos.

BCRPData vía API (series macro: inflación, tasa referencia, tipo de cambio,
PBI). Los CSV quedan en data/raw/bcrp_*.csv y cada descarga se registra
en fuentes_build.csv.

Nada se hardcodea: cada valor trae fecha_build + código de serie + URL.
La sesión ignora proxies heredados del entorno (trust_env=False).

Uso:
  .venv/Scripts/python.exe src/00_download.py --all
  .venv/Scripts/python.exe src/00_download.py --serie PD12912AM --desde 2020-1 --hasta 2026-8
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
PROVENANCE = RAW / "fuentes_build.csv"
HEADER = ["fecha_build", "fuente", "serie_codigo", "serie_nombre",
          "periodo", "valor", "url", "archivo_local"]

# Códigos documentados en docs/FUENTES.md. Cualquier otro código BCRPData
# válido puede pasarse por --serie sin tocar el código.
SERIES = {
    "PD12912AM": "Expectativa de inflación 12 meses",
    "PD04722MM": "Tasa de referencia de la política monetaria",
    "PN01206PM": "TC interbancario venta, promedio del periodo (S/ por US$)",
    "PN01728AM": "PBI, variación porcentual interanual",
    "PN38705PM": "IPC Lima, índice Dic.2021=100",
    "PN07746EM": "Mora sistema: cartera atrasada neta / colocaciones netas %, empresas bancarias",
    "PN07728EM": "Colocaciones sistema: tasa mensual de crecimiento %, empresas bancarias",
    "PN07736EM": "Mora BanBif: cartera atrasada neta / colocaciones netas %",
    "PN07737EM": "Mora MiBanco: cartera atrasada neta / colocaciones netas %",
}

def bcrp_url(codigo: str, desde: str, hasta: str) -> str:
    return (f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/"
            f"{codigo}/json/{desde}/{hasta}")


def sesion_sin_proxy() -> requests.Session:
    s = requests.Session()
    s.trust_env = False  # ignora http_proxy/https_proxy heredados (proxy caído)
    return s


def fetch_bcrp(codigo: str, desde: str, hasta: str) -> tuple[list[dict], str]:
    url = bcrp_url(codigo, desde, hasta)
    r = sesion_sin_proxy().get(url, timeout=60)
    r.raise_for_status()
    payload = r.json()
    nombre = SERIES.get(codigo, codigo)
    filas: list[dict] = []
    # Formato esperado BCRPData: {"periods": [{"name": "2026-08", "values": ["3.05"]}]}.
    # Si el formato cambia, se guarda el crudo y se avisa en vez de inventar parseo.
    try:
        for p in payload["periods"]:
            periodo = p.get("name", "")
            valores = p.get("values", [])
            valor = valores[0] if valores else ""
            filas.append({"periodo": periodo, "valor": valor})
    except (KeyError, TypeError, IndexError) as e:
        crudo = RAW / f"bcrp_{codigo}_crudo.json"
        crudo.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        print(f"AVISO: formato BCRPData cambió ({e}); crudo en {crudo}. "
              f"Registra el parseo a mano.", file=sys.stderr)
    if filas:
        destino = RAW / f"bcrp_{codigo}.csv"
        with destino.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["periodo", "valor"])
            w.writeheader()
            w.writerows(filas)
    return filas, url


def registra(rows: list[list[str]]) -> None:
    existe = PROVENANCE.exists()
    with PROVENANCE.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not existe or PROVENANCE.stat().st_size == 0:
            w.writerow(HEADER)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Descarga BCRPData fresco")
    ap.add_argument("--serie", default="PD12912AM",
                    help="código de serie BCRPData (default: PD12912AM)")
    ap.add_argument("--desde", default="2020-1")
    ap.add_argument("--hasta", default="2026-8")
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--all", action="store_true",
                    help="descarga las 9 series de SERIES (ignora --serie)")
    args = ap.parse_args()

    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    hoy = dt.date.today().isoformat()
    codigos = list(SERIES) if args.all else [args.serie]
    for codigo in codigos:
        try:
            filas, url = fetch_bcrp(codigo, args.desde, args.hasta)
            nombre = SERIES.get(codigo, codigo)
            registra([[hoy, "BCRPData", codigo, nombre,
                       r["periodo"], r["valor"], url, f"bcrp_{codigo}.csv"]
                      for r in filas])
            print(f"BCRPData {codigo}: {len(filas)} periodos → "
                  f"data/raw/bcrp_{codigo}.csv")
        except requests.RequestException as e:
            print(f"ERROR BCRPData {codigo} ({e}). Revisa red/proxy y reintenta.",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
