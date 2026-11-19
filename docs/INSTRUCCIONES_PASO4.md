# PSet3 paso 4 — Tuning

1. Copia la carpeta `pset_3` del ZIP sobre tu proyecto existente. Se agregan `notebooks/04_tuning.ipynb`, `src/tuning.py` y este archivo. **No reemplaza** notebooks anteriores.
2. Verifica `docker compose ps`: MLflow y Notebook deben estar `Up`.
3. En `http://localhost:8888/lab`, abre `notebooks/04_tuning.ipynb` y usa **Restart Kernel and Run All**.
4. Cada una de las tres celdas de entrenamiento ejecuta **30 configuraciones × 5 folds**. Se crean 30 corridas hijas y una padre por familia; además se guardan los pipelines entrenados.
5. No ejecutar el notebook 05 ni consultar el conjunto de test hasta cerrar la selección de CV.
6. Verifica en MLflow el experimento `PSet3_Churn_Telco`; se esperan 90 corridas hijas de tuning + 3 padres, además de las 7 corridas del notebook 03.
7. Evidencia: `docs/resultados_tuning.csv`, `docs/tuning_configuraciones_*.csv`, `docs/tuning_comparacion.png`, `docs/corridas.csv`.
8. Si la ejecución falla en una familia, **no borres los logs**. Comparte la traza de error y repite solo esa celda tras corregirlo.

Las diferencias pequeñas respecto a la desviación entre folds se reportan como inconclusas. El dataset no incluye resultados reales de campañas ni históricos por cliente.
