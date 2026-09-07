# deploy/ — estrategia de publicación con costo $0

Un solo cloud (Azure, el más pedido en los avisos revisados). El pipeline corre
en local con pandas + Parquet, sin servicios encendidos.

1. CI: `.github/workflows/ci.yml` corre pytest en cada push (Python 3.11),
   sin secretos ni tarjeta.
2. Demo viva: Streamlit Community con muestra agregada (ver `app/`).

Decisiones de costo: sin ADF (cobra por ejecución), sin recursos con tarjeta
personal, sin dual-cloud. Nada de infraestructura como código contra una
suscripción real.
