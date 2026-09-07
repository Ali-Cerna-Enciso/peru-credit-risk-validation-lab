"""ETL medallion local.

bronze (raw inmutable) → silver (Pandera: tipos, rangos, referencial) →
gold (tablas modelables + trazabilidad versión código + hash input).
Motor: pandas + Parquet. Implementado en `src/01_build_medallion.py`.
Sin cloud en esta etapa (ver deploy/README.md).
"""
