# data/ — versionados: código, `raw/bcrp_*.csv` + `fuentes_build.csv` (series públicas), READMEs y `.gitkeep`. Gitignored: bronze/silver/gold/synthetic parquet (regenerables) y `sbs_*.csv` manuales. Nada interno entra aquí.

- `raw/`: agregados reales SBS (mora por entidad/producto, cartera, provisiones) + BCRPData vía API
  (inflación, tasa referencia, tipo de cambio, PBI) + `fuentes_build.csv`
  (fecha + código serie BCRPData/SBS + valor pineado por build). Nada hardcodeado de un chat.
- `bronze/`, `silver/`, `gold/`: salidas del pipeline medallion sobre esos agregados (.parquet).
- `synthetic/`: SOLO demo Gini/KS individual (microdata préstamo-a-préstamo que la SBS no publica
 por secreto bancario). Generador propio con seed fija y nivel ilustrativo NO anclado a la mora
 real (`ancla_mora_real: null`). Declarado como sintético en informe y entrevista.
 Ningún otro módulo lo usa.
- Regla institucional: cero datos INEI, empleadores o clientes. Solo público o sintético declarado.
