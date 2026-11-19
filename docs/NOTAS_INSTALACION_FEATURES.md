# Complemento de features — PSet 3

Copiar `notebooks/02_features.ipynb` y reemplazar `src/pipeline.py` en tu `pset_3` existente.
No eliminar ni reemplazar `01_eda.ipynb`, `src/datos.py`, docker-compose.yml o credenciales.

En JupyterLab abrir `02_features.ipynb`, luego **Kernel → Restart Kernel and Run All Cells**.
Los gráficos y `docs/features_resumen.json` aparecen en el volumen de `/workspace/docs`.

El código del `ColumnTransformer` está definido, pero NO se ajusta hasta después de separar train/test, durante `Pipeline.fit()` dentro de la validación cruzada.

IMPORTANTE: Dataset transversal; disponer de datos antes del abandono no se ha verificado temporalmente.
