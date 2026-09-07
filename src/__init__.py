"""peru-credit-risk-validation-lab.

Etapas (ver docs/METODOLOGIA.md):
  00 descarga fresca SBS/BCRP (data/raw + fuentes_build.csv)
  01 medallion bronze->silver->gold (Pandera + linaje)
  02 scoring PD y validacion (Gini/KS/PSI/backtesting; sintetico aislado)
  03 estres macro (satelite OLS + escenarios)
  05 informe auto (md + xlsx) + app Streamlit local
"""
