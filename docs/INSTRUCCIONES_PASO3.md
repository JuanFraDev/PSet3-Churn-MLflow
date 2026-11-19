# PSet #3 — Paso 3: split, líneas base y candidatos

## Instalar

1. Copia el contenido de este paquete sobre la carpeta `pset_3/` que ya funciona.
2. Se agregan `notebooks/03_baseline_candidatos.ipynb`, `src/baselines.py`, `src/experimentos.py` y estas instrucciones.
3. **No reemplaza** `src/pipeline.py`, `src/datos.py`, `docker-compose.yml`, `.env` o las credenciales.
4. No se requiere reconstruir Docker si `02_features.ipynb` ya corrió.

## Ejecutar

- En JupyterLab abre `notebooks/03_baseline_candidatos.ipynb`.
- Ejecuta **Kernel > Restart Kernel and Run All Cells**. El entrenamiento de MLP y SVC puede tardar.
- Abre `http://localhost:5000`, experimento `PSet3_Churn_Telco`.
- Captura tabla de comparación, ejecuciones MLflow y confirma que aparezcan 7 runs.
- Comprueba `docs/resultados_candidatos.csv`, `docs/split_info.json` y `docs/split_train_test.csv`.

## Consideraciones metodológicas

- La validación es estratificada; no hay fechas de abandono individuales para una CV temporal.
- Solo se utiliza entrenamiento y CV: test reservado para notebook 05.
- AP es el resumen no interpolado de curva PR, distinto de AUC trapezoidal; el PSet se refiere a PR-AUC.
- `ReglaNegocio`: contrato mensual y antigüedad **menor que 12 meses**, como en PSet #1.
- Costos ficticios en el notebook (USD 5/contacto, USD 120/abandono, 20% eficacia): sirven solamente como simulación preliminar, no ROI real.
- Si no hay repositorio Git, la etiqueta de commit será `SIN_COMMIT_LOCAL`; haz un commit para las siguientes corridas.
- `CUSTOMER_ID` e `INGESTED_AT` no se usan como predictores, y las transformaciones se ajustan dentro de CV.
- OBT es transversal y no permite garantizar disponibilidad histórica previa al abandono; se documenta como limitación.
- El requisito de experimento LLM por prompting se desarrolla aparte; **no** cuenta como uno de estos 5 candidatos tradicionales hasta que se evalúe formalmente.
