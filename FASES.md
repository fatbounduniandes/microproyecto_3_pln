# Fases del Microproyecto 3

Avanzamos **una fase a la vez**: solo se marca `[x]` cuando Jean lo autoriza, y los `git push`
solo se hacen cuando él lo pide.

## Checklist

- [x] **Fase 0 — Repo y base** (validada localmente)
- [x] **Fase 1 — Portada e introducción** (sección 1, alineada al estilo del MP2)
- [x] **Fase 2 — Librerías** (sección 2 validada en Colab)
- [x] **Fase 3 — Carga del dataset y exploración** (sección 3 validada en Colab)
- [x] **Fase 4 — Construcción de prompts y preprocesamiento** (sección 4)
- [x] **Fase 5 — Particiones (train / validation / test)** (sección 5)
- [x] **Fase 6 — Implementación de la evaluación pass@1** (sección 6)
- [x] **Fase 7 — Evaluación de los modelos base (zero-shot)** (sección 7)
- [x] **Fase 8 — Finetuning con LoRA** (sección 8)
- [x] **Fase 9 — Evaluación de los modelos finetuneados** (sección 9)
- [x] **Fase 10 — Comparación y resultados** (sección 10)
- [x] **Fase 11 — Conclusiones** (sección 11)
- [x] **Fase 12 — Smoke test en Colab T4/L4**
- [ ] **Fase 13 — Run final y entrega** (notebook ejecutado listo; pendiente el `.zip`)

---

## Detalle de cada fase

### Fase 0 — Repo y base (no va en el notebook)
- `AGENTS.md`, `FASES.md`, `README.md`, `GUIA_MICROPROYECTO_3.md`, `.gitignore`,
  `.gitattributes`.
- `build_notebook.py`, `requirements-colab.txt` y el esqueleto de las 11 secciones en `src/cells/`.
- `notebook.ipynb` generado desde las celdas fuente.
- Skills curadas: `mbpp-lora-pass1`, `lora-qlora-recipes`, `peft-fine-tuning`,
  `python-testing-patterns`, `find-skills`.
- **Lista cuando:** `python3 build_notebook.py --check` responde `OK` y el notebook tiene las
  11 secciones con los títulos exactos.

### Fase 1 — Portada e introducción
- Nombre del estudiante o integrantes, resumen del proyecto y descripción de las cuatro
  configuraciones (Qwen base, Qwen + LoRA, SmolLM2 base, SmolLM2 + LoRA).
- Tabla de configuración y mención del dataset, la métrica y el entorno de ejecución.
- Misma estructura que el MP2: autoría, `---`, resumen con la tabla, `### Datos`,
  `### Evaluación`, `### Entorno` y cierre de reproducibilidad.
- **Lista cuando:** la sección 1 se lee completa y no tiene marcadores de pendiente.
- **Estado:** escrita y validada localmente; pendiente de revisión y push.

### Fase 2 — Librerías
- Instalación de `transformers`, `peft`, `datasets` y `accelerate` en Colab.
- Imports, versiones, GPU, precisión (bf16/fp16), semilla, `DRIVE_DIR` y liberación de VRAM.
- La precisión se elige por compute capability: bf16 desde 8.0 (L4, A100); T4 usa fp16 con
  `GradScaler` (D-031).
- Revisiones fijas de MBPP, Qwen2.5-1.5B y SmolLM2-1.7B junto a las constantes.
- **Lista cuando:** en Colab se ve la GPU, el compute capability, el tipo de AMP y la semilla.
- **Estado:** corrida en Colab T4; mostró bf16 por emulación, corregido en D-031.

### Fase 3 — Carga del dataset y exploración
- Carga de `google-research-datasets/mbpp` configuración `full` con revisión fija.
- Descripción de los splits `train` (374), `validation` (90), `test` (500) y `prompt` (10).
- Ejemplos reales de `text`, `code`, `test_list` y `test_setup_code`.
- Verificación de que ningún `task_id` se repite entre splits y de que no hay textos ni códigos
  duplicados, exactos ni normalizados.
- Tabla descriptiva con pandas (cuartiles, desviación y extremos) para `train` y `validation`.
- Análisis de vocabulario: palabras distintas, más frecuentes y cuántos enunciados mencionan el
  nombre de la función.
- Análisis de estructura con `ast`: funciones por solución, imports, clases, recursión y ciclos.
- Calidad del texto: tabs, `print(`, `input(`, soluciones de una línea y prueba de
  Kolmogorov-Smirnov entre `train` y `validation`.
- Análisis de los `assert`: cantidad por problema y tipo de valor esperado.
- Seis figuras: boxplots, curva acumulada, palabras frecuentes, líneas por solución, estructura
  y construcciones.
- Conclusiones en markdown con las cifras reales de la corrida, sin texto generado (D-033).
- De `test` solo se cuentan los ejemplos y se revisan los identificadores (D-030).
- **Lista cuando:** se imprimen los tamaños y ejemplos reales, salen las figuras y Jean aprueba
  el cierre.
- **Estado:** corrida en Colab hasta esta fase; pendiente re-ejecutar con el análisis ampliado.

### Fase 4 — Construcción de prompts y preprocesamiento
- Formato de instrucción único: problema + nombre/firma de la función objetivo.
- Función que arma el prompt y el target (código de referencia completo + EOS).
- Tokenización con `add_special_tokens=False`, BOS/EOS manual si aplica.
- Máscara de `labels` con `-100` para todos los tokens del prompt; verificación de longitudes.
- **Lista cuando:** un ejemplo imprime prompt, target, labels y el número de tokens enmascarados
  coincide con la longitud real del prompt.
- **Estado:** escrita (D-034 y D-035). Validada en local con los 464 ejemplos de `train` y
  `validation` y los tokenizadores de las revisiones fijas: 0 diferencias de prefijo, 0 ejemplos
  por encima de 512 y 0 fugas. Corrida completa en Colab A100 80GB.

### Fase 5 — Particiones (train / validation / test)
- Asignación de roles: `train` para finetuning, `validation` para monitoreo e hiperparámetros,
  `test` reservado para la evaluación final.
- Verificación de que no se usa `test` en entrenamiento ni en selección de hiperparámetros.
- **Lista cuando:** las tres particiones quedan documentadas y el notebook no toca `test` antes
  de la sección 7.
- **Estado:** escrita (D-036). Roles en tabla, 24 lotes por época, 50 de monitoreo elegidos con
  `SEED = 42` (ids 511–600, 0 repetidos con `train`) y presupuesto de 33,984 / 40,201 tokens por
  época. Corrida completa en Colab A100 80GB.

### Fase 6 — Implementación de la evaluación pass@1
- Extracción de código generado (bloques markdown o texto plano).
- Generación por lotes: padding a la izquierda, `pad_token = eos`, `attention_mask`, recorte del
  prompt, `do_sample=False`, `max_new_tokens=512`.
- Ejecución de código generado en subproceso aislado con timeout: código → `test_setup_code` →
  asserts.
- Cálculo de pass@1 y guardado de resultados por problema.
- **Lista cuando:** el sandbox corta un bucle infinito sin tumbar el notebook y las soluciones
  escritas a mano producen resultados coherentes (D-036: sin modelo diminuto).
- **Estado:** escrita (D-037 y D-042). `generar_soluciones`, `extraer_codigo`, `ejecutar_solucion` y
  `evaluar_pass1`; seis pruebas del sandbox superadas en local (correcta, incorrecta, bucle
  infinito, sintaxis, vacía y orden con setup) y cortes en encabezados posteriores. Corrida completa
  en Colab A100 80GB.

### Fase 7 — Evaluación de los modelos base (zero-shot)
- Carga de Qwen2.5-1.5B y SmolLM2-1.7B base, uno a la vez.
- Evaluación de los 500 problemas de `test` con la misma configuración de generación.
- Guardado de pass@1 y generaciones en Drive/resultados.
- **Lista cuando:** existen los dos pass@1 base y se liberó la GPU entre modelos.
- **Estado:** completada (D-038). Verificación de `test` (500 problemas, 0 solapamientos, 3 asserts
  y 1 `test_setup_code`), carga en `AMP_DTYPE`, prueba de formato y evaluación de los 500 en Colab
  L4. Resultados finales: Qwen base 41.2 % (206/500, 579.6 s de generación) y SmolLM2 base 4.6 %
  (23/500, 629.2 s).

### Fase 8 — Finetuning con LoRA
- `LoraConfig(r=16, alpha=16, dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj"])`.
- Bucle manual con `transformers` + `peft`, labels enmascarados, fp16/bf16, gradient checkpointing.
- Monitoreo de pérdida de validación y pass@1 en 50 problemas de validación por época.
- Guardado del mejor adaptador de cada modelo en Drive; liberar GPU entre ambos.
- **Lista cuando:** existen `checkpoint_qwen2.5_1.5b_lora/` y `checkpoint_smollm2_1.7b_lora/`
  con sus tokenizadores y el historial de entrenamiento.
- **Estado:** completada (D-039 y D-043). Receta de D-021 con fp32 + autocast, gradient checkpointing,
  5 épocas, monitoreo por época, mejor adaptador e historial en `RESULTADOS_DIR`. Los bloques de
  validación se arman por índices porque los splits son `Dataset` de Hugging Face. Entrenada en
  Colab A100 80GB: Qwen 4,358,144 parámetros LoRA (0.28 %) y SmolLM2 6,291,456 (0.37 %); mejor época
  5 en los dos modelos, con 48 % y 32 % de pass@1 de monitoreo. Adaptadores, tokenizadores e
  historiales guardados en Drive.

### Fase 9 — Evaluación de los modelos finetuneados
- Carga de cada modelo base + su adaptador LoRA.
- Evaluación de los 500 de `test` con exactamente el mismo prompt y `max_new_tokens` que en la
  sección 7.
- Guardado de pass@1 y generaciones.
- **Lista cuando:** existen los dos pass@1 finetuneados y se liberó la GPU entre modelos.
- **Estado:** completada (D-040). Carga de cada adaptador sobre su base, verificación de configuración
  (r=16, alpha=16, dropout 0.05, q/k/v/o) y evaluación de los 500 con el mismo `evaluar_pass1` en
  Colab L4. Resultados finales: Qwen + LoRA 43.6 % (218/500, 414.4 s de generación) y SmolLM2 + LoRA
  31.6 % (158/500, 475.5 s).

### Fase 10 — Comparación y resultados
- Tabla de las cuatro configuraciones con pass@1 e intervalo de Wilson al 95 %.
- Gráfico de barras comparativo y análisis del aporte del finetuning en cada familia.
- Ejemplos cualitativos de soluciones correctas e incorrectas.
- Conclusión sobre cuál modelo es mejor en la tarea.
- **Lista cuando:** la tabla, la gráfica y los ejemplos están en el notebook y la conclusión es
  coherente con los números.
- **Estado:** escrita (D-044). Sección de 12 celdas con tabla de las cuatro configuraciones e
  intervalos de Wilson al 95 %, barras de pass@1, motivos apilados, matriz de transición base → LoRA
  y dos ejemplos cualitativos por modelo más un ejemplo de regresión. No vuelve a generar: usa los
  resúmenes de las secciones 7 y 9. Cifras finales del run en L4: 41.2 %, 43.6 %, 4.6 % y 31.6 %.

### Fase 11 — Conclusiones
- Discusión de resultados, seguimiento del entrenamiento, hiperparámetros y limitaciones.
- Comparación con lo esperado y aprendizajes del proyecto.
- **Lista cuando:** el notebook cierra con la discusión y no quedan celdas pendientes.
- **Estado:** escrita. Sección 11 con las conclusiones 42–47: resultado global con intervalos de
  Wilson, aporte del formato, selección de época, limitaciones, qué mide el pass@1 y trabajo futuro.

### Fase 12 — Smoke test en Colab T4/L4
- Corrida completa con menos datos o menos épocas para detectar errores rápido.
- **Lista cuando:** `Run All` termina sin excepciones en T4 o L4.
- **Estado:** completada (D-046). El run final en L4 ejecutó las 163 celdas, 72 de código, con 0
  errores y 10 figuras. Los adaptadores se cargaron desde Drive sin reentrenar.

### Fase 13 — Run final y entrega
- `Run All` en L4 o A100 con los hiperparámetros definitivos.
- `.zip` con notebook ejecutado + los dos adaptadores, subido a Drive con permiso de lector.
- **Lista cuando:** la descarga funciona en modo incógnito y el zip contiene todo lo exigido.
- **Estado:** notebook ejecutado listo (`resultados/notebook.ipynb`). Pendiente el `.zip` y la prueba
  en incógnito.
