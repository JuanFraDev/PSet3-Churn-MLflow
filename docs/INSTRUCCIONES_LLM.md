# Prueba de prompting — PSet #3

## Hallazgo sobre el CSV inicial

El archivo `llm_prompts_train.csv` recibido tenía **5 casos, todos con la etiqueta 0**. Se incluyó un archivo `docs/llm_demostracion_5.csv` para mostrar decisiones cualitativas, pero NO representa una prueba ciega ni permite evaluar el recall en la clase de abandono. Las etiquetas venían visibles en el mismo archivo original.

## Preparación de una muestra evaluable

1. Copia los archivos de este complemento dentro de tu carpeta existente `pset_3` sin reemplazar notebooks anteriores.
2. Abre `notebooks/07_llm_preparar_muestra.ipynb` en JupyterLab y ejecuta **Restart Kernel and Run All Cells**.
3. El notebook verifica el hash original y `split_train_test.csv`, selecciona **20 clientes exclusivamente de train** (15 que permanecen, 5 que abandonan) con semilla fija y guarda tres archivos:
   - `docs/llm_prompts_20_sin_etiquetas.csv`: único archivo que debes enviar al LLM.
   - `docs/llm_etiquetas_20_reservadas.csv`: **no enviar** al LLM.
   - `docs/llm_respuestas_20.csv`: plantilla para guardar resultados.
4. Envía **solo** el archivo `llm_prompts_20_sin_etiquetas.csv` al chat y solicita un CSV de respuestas con las columnas `ejemplo,prediccion_llm,razon_llm`. Las respuestas han de usar exclusivamente `ABANDONA` o `PERMANECE`.
5. Reemplaza `docs/llm_respuestas_20.csv` con el archivo de respuestas y ejecuta `notebooks/08_llm_evaluar.ipynb` con **Restart Kernel and Run All Cells**.
6. Conserva resultados y captura de la matriz de confusión. Registra nombre del modelo de LLM, fecha y versión de prompt. Si no usas API, especifica que fue prompting manual en ChatGPT.

## Qué incluir en el memo

> Se realizó una prueba exploratoria de prompting con 20 fichas extraídas exclusivamente del conjunto de entrenamiento mediante muestreo reproducible, manteniendo etiquetas reales ocultas durante la predicción. Se evaluaron accuracy, precisión, recall y F1 para la clase de abandono. Dado el tamaño reducido y la ausencia de validación cruzada, los resultados no se usan para seleccionar el champion ni para estimar el valor de negocio.

**Importante:** Este texto debe actualizarse con resultados reales *después* de recibir las respuestas del modelo. Nunca inventes métricas.
