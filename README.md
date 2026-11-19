# PSet #3 — Experimentacion y seleccion del modelo (Churn)

**Proyecto:** prediccion de abandono de clientes de telecomunicaciones.
**Origen:** `PSET2_CHURN.OBT.CUSTOMER_CHURN_OBT` en Snowflake, construido en PSet #2.
**Estado de este paquete:** arranque del entorno y notebook de EDA; las etapas de candidatos, tuning y registro se desarrollaran despues. No presenta resultados inventados.

## Requisitos
- Docker Desktop operativo y acceso a la misma cuenta de Snowflake del PSet #2.
- Clave privada cifrada configurada en el archivo local `.env.snowflake` del PSet #2.
- El rol `PSET2_SPARK_ROLE` debe tener `SELECT` sobre la OBT (ver `docs/grants_pset3.sql`).

## 1. Preparacion local (PowerShell)
1. Extraer el contenido de esta carpeta `pset_3` en una carpeta nueva. No reemplazar el proyecto `pset_2`.
2. `Copy-Item .env.example .env`
3. Editar `.env` y completar `SNOWFLAKE_ACCOUNT` con el identificador utilizado en PSet #2 (sin dominio), ademas de un token local fuerte para Jupyter.
4. Copiar **solo de forma local** el archivo `pset_2/.env.snowflake` dentro de `pset_3/.env.snowflake`. Git lo ignora. **Nunca subirlo al repositorio.**
5. Ejecutar en Snowflake `docs/grants_pset3.sql` si al conectar se obtiene un error de permiso `SELECT`.

## 2. Arranque
```powershell
docker compose up -d --build
docker compose ps
```
- JupyterLab: `http://localhost:8888/lab?token=<tu JUPYTER_TOKEN>`
- MLflow: `http://localhost:5000`
- Backend de MLflow: SQLite persistente en volumen de Docker; artefactos en almacenamiento local persistente separado dentro del mismo volumen.

## 3. Primer notebook
Abre `notebooks/01_eda.ipynb` y ejecuta **Restart Kernel and Run All**.
Genera gráficos, tablas y `docs/eda_resumen.json` a partir de los datos reales de la OBT.

## 4. Próximas etapas
`02_features.ipynb` → `03_baseline_candidatos.ipynb` → `04_tuning.ipynb` → `05_seleccion.ipynb`.
Preparacion de sklearn en `src/pipeline.py` (ampliar al crear Pipeline y ColumnTransformer).

## Seguridad
Archivos `.env`, `.env.snowflake`, `.pem` y snapshots locales no se suben a GitHub. No imprimir secretos en notebooks. MLflow y Jupyter estan publicados solo en `127.0.0.1` de la computadora local.

## Limitaciones
La OBT es un corte transversal, no un historial temporal. `INGESTED_AT` es fecha tecnica y **no** corresponde al periodo de abandono. Ningun beneficio comercial puede estimarse como causal sin informacion de campañas e intervenciones.
