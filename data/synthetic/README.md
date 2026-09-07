# data/synthetic/ — SOLO demo Gini/KS individual (sintético declarado)

La SBS no publica microdata préstamo-a-préstamo por secreto bancario, así que
la discriminación individual (Gini, KS por score) no puede calcularse sobre
datos peruanos reales. Este es el ÚNICO módulo que usa sintético.

Reglas:

- Generador propio en `src/02_score_validate.py` con seed fija (42, n=60k).
  Nivel ilustrativo E[PD]~12% explícitamente NO anclado a la mora real
  (`ancla_mora_real: null` en `validacion_metricas.json`).
- La deriva desde 2025+ (el bureau pierde pendiente fuera de muestra) es un
  beta-break diseñado para probar qué métrica lo detecta, no un hallazgo de
  cartera. Las métricas se reportan tal cual salen y el grado en rojo se
  conserva con su plan de acción.
- Ningún otro módulo lo usa: ETL medallion, backtesting, PSI y estrés corren
  sobre sus propios insumos (el backtesting y el PSI también son sintéticos;
  lo real es el panel agregado y el satélite de estrés).
- Todo gráfico o tabla que salga de aquí lleva etiqueta "sintético" en título,
  informe y entrevista. No afirmar conocimiento de ninguna cartera real.

Salidas típicas (gitignored, regenerables): `synthetic_scored.parquet` con
columnas `score, pd_predicha, default_observado, segmento, vintage, bureau`.
Lo escribe `02` en cada corrida; el linaje hashea ese archivo recién escrito.
