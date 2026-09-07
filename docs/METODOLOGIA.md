# METODOLOGÍA — validación de modelos de riesgo (réplica día a día banca)

Fuente de requisitos: Caja Los Andes pub. 03-sep-2026 (Gini, KS, estabilidad
temporal/poblacional, backtesting, sensibilidad) + BanBif (métricas apetito de
riesgo, provisiones, estrés, reportes SBS/ASBANC) + Interbank MRV
(documentación, ciclo de vida, gobierno de modelos).

## 1. Scoring PD

- Modelo base: regresión logística sobre application score
  (`logit(PD) = β₀ + Σβᵢxᵢ`). Benchmark: gradient boosting solo como
  challenger documentado, no como titular (explicabilidad ante auditoría).
- Segmentación previa: hipotecario / consumo / microempresa-rural
  (población objetivo Los Andes: rural). Un modelo por segmento o un modelo
  con dummies + test de estabilidad por segmento.

## 2. Discriminación

- **Gini = 2·AUC − 1.** Umbral operativo banca retail PE: Gini out-of-time
  ≥ 0.40 (documentar si cae bajo 0.35 → alerta).
- **KS = máx_bandas |F_bads(b) − F_goods(b)|**, F = acumulada por banda de
  score ordenada. Reportar banda del máximo (punto de corte candidato).

## 3. Calibración y backtesting

- Binomial por banda/grado: H₀: PD_real = PD_predicha.
  z = (DR − PD) / √(PD(1−PD)/n); semáforo verde/amarillo/rojo por p-valor.
- Traffic-light regulatorio (estilo SBS/Basilea): comparar defaults
  observados vs esperados por grado con zonas verde/amarilla/roja.
- Curva calibración predicho vs observado por decil + Brier score.

## 4. Estabilidad

- **PSI = Σ (%actual − %esperada)·ln(%actual/%esperada)** por variable y por
  score. Regla: <0.10 ok, 0.10–0.25 monitorear, >0.25 reconstruir.
- Estabilidad temporal (ventanas móviles) y poblacional (train vs OOT vs
  producción).
- Simulación ilustrativa de migración de grados (ruido sobre el mismo corte,
  no transición t → t+1 de cartera).

## 5. Sensibilidad y estrés

- Escenarios macro calibrados a BCRPData/SBS jalados frescos al construir
  (tasa, inflación, tipo de cambio, mora hipotecaria): shock ±x pb →
  ΔPD portafolio, Δprovisiones, Δcapital. Tabla + gráfico tornado.
- Sensibilidad univariada por variable top (PD por decil de ingreso,
  ratio cuota-ingreso, etc.).

## 6. Calidad y gobierno de datos (etapa medallion)

- Bronze (raw inmutable) → silver (limpio, Pandera: tipos, rangos,
  no-nulos, referencial) → gold (tablas modelables + features).
- Trazabilidad: cada tabla gold registra versión de código + hash de input.
- Toda regla rota = alerta, no silencio. Dashboard de calidad.

## 7. Salidas

- `informe/figures/`: mora, calibración, walk-forward, tornado (se regeneran con 02/03).
- `reporte/05_report_sbs.py`: informe auto (md + xlsx) con las 7 secciones.
- `docs/LIMITACIONES.md`: qué NO es esto (leer antes de citar el repo).
