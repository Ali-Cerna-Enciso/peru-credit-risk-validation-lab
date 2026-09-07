# Informe de validación — riesgo de crédito (build 2026-09-07)

> Cartera: agregados reales BCRPData (origen SBS) + demo individual sintética declarada.
> Linaje: código sha `449ffc6ea3837ee9`, 8 tablas gold con hash input.

## 1. Portafolio y mora (REAL BCRPData)

Mora sistema empresas bancarias 2026-06: **2.85%**
(BanBif 3.12%, MiBanco 3.89%).
Tasa de referencia 4.25%, TC S/ 3.37.
Niveles de mora comparables desde 2018-01 (quiebre metodológico BCRP previo, ver FUENTES).

## 2. Discriminación (SINTÉTICO seed 42, n=60000, E[PD]=0.1274 ilustrativo NO anclado)

Gini train 0.505 / OOT **0.470** (umbral 0.40;
deriva diseñada 2025+: bureau pierde poder en OOT, por eso train>OOT).
Challenger gradient boosting OOT 0.462: titular sigue
siendo logística por explicabilidad ante auditoría. KS OOT **0.347**
(ventiles), banda del máximo 484-509.

## 3. Calibración y backtesting (SINTÉTICO)

Brier OOT 0.1149 (consumo 0.1008,
hipotecario 0.0727,
microempresa 0.1647).
Hosmer-Lemeshow 15.6, p=0.048 (rechaza ajuste perfecto:
coherente con deriva OOT). Backtesting binomial por grado (H0: DR=PD):
AAA=verde, AA=rojo, A=verde, BBB=verde, BB=verde, B=verde, C=verde.
Plan acción grado AA (p=0.0003): drift OOT: el bureau pierde pendiente fuera de muestra (beta-break diseñado 2025+, no ruido).
Acción: reestimar coeficientes con ventana 2025+ (refit completo: ajuste de intercepto solo no corrige un cambio de pendiente del bureau). Dueño: responsable de validacion. Límite: 2026-12-06.

## 4. Estabilidad (SINTÉTICO + ventana real)

Walk-forward W1–W3 (train/valid/OOT rodantes) en anexos: Gini OOT 0.47–0.53
(amarilla si <0.40 con causa macro escrita). Peor PSI train-vs-OOT:
**bureau=0.0493 (ok)** — deriva
visible pero contenida (beta-break del bureau diseñado 2025+). Lectura PSI vs beta: bureau PSI 0.049 (ok) mientras el grado AA cae en rojo (z=3.60, p=0.0003). El PSI mide cambio de distribucion, no cambio de pendiente: el backtesting por grado no se sustituye con PSI.

## 5. Estrés macro (REAL, satélite OLS 2018-01..2026-06, n=102, R²=0.635)

b_tasa=0.233 (p=0.000),
b_ipc=-0.024 (p=0.131, no significativo:
correlación parcial, no causal),
b_tc=0.0656 (p=0.000).
Adverso (tasa+200pb, IPC+1.5pp, TC+10%): mora 2.85% → **3.94%**.
Severo (+300pb, +2.5pp, +20%): **4.80%** (Δ1.95pp, LGD 45%).

## 6. Calidad, gobierno y límites

Bronze inmutable con hash sha por insumo; silver validado con Pandera (5 reglas,
toda rota = alerta); gold con linaje (hash input + sha código por tabla).
Mora por producto (consumo/hipotecario/microempresa) fuera de alcance: el
estrés usa mora sistema y el LGD es supuesto (45%). Sintético solo en
`data/synthetic/`, todo gráfico suyo titulado "sintético".

## 7. Riesgos y qué rompería esto

- **Sin mora por producto** no hay consumo/hipotecario/microempresa ni
  provisiones reales: el estrés usa mora sistema y el LGD es supuesto (45%).
- **Lo sintético solo prueba método individual** (Gini/KS/backtesting): sus
  niveles (E[PD]~12%) son ilustrativos y su deriva es diseñada; no describen
  ninguna cartera peruana real.
- **El satélite es lineal y corto** (n=102, R²=0.635, IPC no
  significativo): un quiebre estructural o no-linealidad fuerte lo deja fuera
  de rango; el sigma de residuos (0.292pp) es el piso de
  su error.
