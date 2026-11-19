# Paso 5 — Selección final y Model Registry

Se entrega únicamente contenido **nuevo** de PSet #3. Extraer este ZIP en la carpeta padre de `pset_3` sin reemplazar los notebooks anteriores.

## Antes de ejecutar

1. Verificar que `docker compose ps` muestra `mlflow` y `notebook` **Up**.
2. Los archivos del notebook 04 deben existir dentro de `pset_3/docs/`: `resultados_tuning.csv`, `split_info.json`, `split_train_test.csv`, `resultados_candidatos.csv`.
3. `src/tuning.py`, `src/pipeline.py` y `src/datos.py` deben seguir siendo los del proyecto actual.
4. Ejecutar `notebooks/05_seleccion.ipynb` con **Restart Kernel and Run All Cells**.
5. Confirmar en MLflow Models que el modelo `PSet3_Churn_Telco-modelo` tiene dos versiones con alias **champion** y **challenger**.
6. Ejecutar `notebooks/06_verificar_champion.ipynb` con kernel nuevo; debe imprimir `RESULTADO: MODELO CHAMPION REPRODUCIBLE`.
7. Guardar las capturas, `docs/resultados_holdout.csv`, `docs/seleccion_final.json`, `docs/corridas.csv` y README actualizado en Git. **Nunca** subir `.env`, `.env.snowflake`, claves RSA ni tokens.

## Decisiones de evaluación

- Se mantiene la única separación original (80/20, semilla 42), con hash SHA256 y manifiesto de clientes.
- El umbral de decisión se fija usando probabilidades OOF sobre **train**.
- Las decisiones de champion/challenger quedan fijadas antes de examinar el conjunto **test**.
- El conjunto test se evalúa una vez en el notebook 05, sin reoptimizar.
- Los supuestos USD 5/contacto, USD 120/abandono, eficacia del 20% son **ilustrativos** y no demuestran retorno causal de una campaña.
- La prueba de LLM requiere enviar los prompts a un modelo generativo (no hay API conectada en este flujo) y guardar respuestas; sin ello, este apartado sigue **pendiente**.
- Nota de limitación: se inspeccionó toda la OBT durante EDA inicial, antes de congelar train/test. Aunque el procesamiento se ajusta en train, la exploración global puede introducir sesgo indirecto de selección de features. Se documenta como limitación.

## Archivo entregado

- `notebooks/05_seleccion.ipynb`
- `notebooks/06_verificar_champion.ipynb`
- `src/seleccion.py`
- `docs/INSTRUCCIONES_PASO5.md`
