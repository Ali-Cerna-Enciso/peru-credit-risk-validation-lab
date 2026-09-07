# LIMITACIONES — leer antes de citar este repo en un CV o entrevista

1. **Agregados peruanos reales + sintético aislado.** Mora/cartera/provisiones SBS y macro
   BCRPData son reales y pineados por build (fecha + serie + valor). Solo la demo de
   porque la SBS no publica préstamo-a-préstamo por secreto bancario; vive en
   `data/synthetic/` con nivel ilustrativo NO anclado a la mora real
   (`ancla_mora_real: null`). Sirve para demostrar método
   (Gini/KS/PSI/backtesting individual), no para afirmar conocimiento de ninguna cartera.
   Declarado en README, informe y entrevista.
2. **No homologado SBS.** Réplica de práctica estándar (traffic-light,
   binomial por grados, PSI), no certificación regulatoria.
3. **Números macro pineados a fecha de construcción.** Cada build registra
   fecha + serie + valor (BCRPData/SBS). Nada viene hardcodeado de un chat.
4. **Cero datos internos.** Ni INEI, ni empleadores, ni clientes. Si un input
   no es público o sintético, no entra.
5. **Alcance junior explícito.** Logística + challenger documentado; sin
   deep learning de crédito, sin MLOps productivo, sin decisiones sobre
   personas reales.
