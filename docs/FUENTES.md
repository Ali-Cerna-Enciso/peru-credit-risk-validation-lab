# FUENTES — de qué aviso sale cada módulo (trazabilidad demanda → repo)

Cruce propio 04-sep-2026 sobre 156 avisos de Lima (ago-sep). Formato:
módulo → avisos que lo piden.

## src/ (scoring + validación)

- Caja Los Andes, Analista Validación Modelos Riesgos, pub. 03-sep, score 9.0:
  Gini, KS, estabilidad temporal/poblacional, backtesting, sensibilidad,
  SQL+Python/R+Power BI, Spark deseable, postgrado/especialización deseable.
- BanBif, Analista Control Riesgo Crédito, pub. 04-sep, score 9.5: Bachiller
  Economía explícito, 1–2 años, SQL/Power BI/R/Python, apetito de riesgo,
  provisiones, estrés, reportes SBS/ASBANC, automatización ETL/RPA.
- Interbank, Model Risk Validator (ATS largo): ciclo de vida, documentación,
  gobierno de modelos, sinergia con Data Analytics/Riesgos.
- BCP, Especialista Estrategia Transformación Riesgos (descartada por
  senioridad, norte de carrera): scoring, calibraciones, GenAI/agenticAI.

## src/etl_medallion/ (Medallion + calidad)

- Experis/Moventi banca-telco (Azure+Databricks+Medallion+ADF, 2 años):
  pipelines, ETL/ELT, modelamiento, calidad/integridad.
- Pandero, Ing. Datos Calidad (reserva 2–4 años): SQL avanzado, ETL/
  DataQuality, gobierno de datos, financiera.
- Protiviti (Bumeran): ETL Python + SQL Server + SSIS.

## Tableros (fuera de este repo)

- Niubiz Performance Negocio 9.0 pedía Power BI + rolling forecast: el estrés
  y el informe por segmento cubren el fondo, pero no se presenta BI aquí.

## reporte/ + deploy/

- NTT Conversational AI (Kore.ai, APIs), Laureate AI Engineer
  (LangChain/HF/RAG/GCP): informe auto + asistente sobre warehouse propio,
  cifras solo desde CSV validado.
- QuantumBlack DS McKinsey (norte, inglés alto): nivel de acabado objetivo.

## Series externas (se descargan frescas al construir, no se hardcodean)

- BCRPData vía API (`src/00_download.py`, build 2026-09-04, 9 series en
  `data/raw/bcrp_*.csv`, provenance por periodo en `fuentes_build.csv`):
  PD12912AM expectativa inflación 12m (2010–ago.2026, 3.05%) ·
  PD04722MM tasa de referencia (ago.2026 4.25%) ·
  PN01206PM TC interbancario venta promedio periodo (ago.2026 S/ 3.37) ·
  PN01728AM PBI var% interanual (2010–jun.2026) ·
  PN38705PM IPC Lima Dic.2021=100 ·
  PN07746EM mora sistema empresas bancarias (jun.2026 2.85%) ·
  PN07728EM colocaciones sistema var% mensual ·
  PN07736EM mora BanBif (jun.2026 3.12%) · PN07737EM mora MiBanco (3.89%).
- Quiebre metodológico BCRP: mora "cartera atrasada neta" sale negativa
  2010–2017 y salta a +3.1 en ene.2018 (sistema y MiBanco; BanBif solo hasta
  nov.2013). Niveles comparables desde ene.2018; antes no usar en niveles.
  Raw inmutable, el corte se aplica en silver.
- SBS por producto (mora consumo/hipotecario/microempresa, provisiones):
  fuera de alcance — el portal no tiene API pública y este repo no es una
  guía de descarga. El estrés usa mora sistema.
