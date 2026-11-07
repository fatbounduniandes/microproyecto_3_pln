# Fases del Microproyecto 3

Avanzamos **una fase a la vez**: solo se marca `[x]` cuando Jean lo autoriza, y los `git push`
solo se hacen cuando él lo pide.

## Checklist

- [x] **Fase 0 — Repo y base** (validada localmente; push pendiente)
- [x] **Fase 1 — Portada e introducción** (sección 1, alineada al estilo del MP2)
- [x] **Fase 2 — Librerías** (sección 2 escrita; validación pendiente en Colab)
- [x] **Fase 3 — Carga del dataset y exploración** (sección 3 escrita; validación pendiente en Colab)
- [ ] **Fase 4 — Construcción de prompts y preprocesamiento** (sección 4)
- [ ] **Fase 5 — Particiones (train / validation / test)** (sección 5)
- [ ] **Fase 6 — Implementación de la evaluación pass@1** (sección 6)
- [ ] **Fase 7 — Evaluación de los modelos base (zero-shot)** (sección 7)
- [ ] **Fase 8 — Finetuning con LoRA** (sección 8)
- [ ] **Fase 9 — Evaluación de los modelos finetuneados** (sección 9)
- [ ] **Fase 10 — Comparación y resultados** (sección 10)
- [ ] **Fase 11 — Conclusiones** (sección 11)
- [ ] **Fase 12 — Smoke test en Colab T4/L4**
- [ ] **Fase 13 — Run final y entrega**

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
  por encima de 512 y 0 fugas; pendiente de ejecutar en Colab.

### Fase 5 — Particiones (train / validation / test)
- Asignación de roles: `train` para finetuning, `validation` para monitoreo e hiperparámetros,
  `test` reservado para la evaluación final.
- Verificación de que no se usa `test` en entrenamiento ni en selección de hiperparámetros.
- **Lista cuando:** las tres particiones quedan documentadas y el notebook no toca `test` antes
  de la sección 7.
- **Estado:** escrita (D-036). Roles en tabla, 24 lotes por época, 50 de monitoreo elegidos con
  `SEED = 42` (ids 511–600, 0 repetidos con `train`) y presupuesto de 33,984 / 40,201 tokens por
  época; pendiente de ejecutar en Colab.

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
  infinito, sintaxis, vacía y orden con setup) y cortes en encabezados posteriores; pendiente de
  ejecutar en Colab.

### Fase 7 — Evaluación de los modelos base (zero-shot)
- Carga de Qwen2.5-1.5B y SmolLM2-1.7B base, uno a la vez.
- Evaluación de los 500 problemas de `test` con la misma configuración de generación.
- Guardado de pass@1 y generaciones en Drive/resultados.
- **Lista cuando:** existen los dos pass@1 base y se liberó la GPU entre modelos.
- **Estado:** escrita (D-038). Verificación de `test`, carga en `AMP_DTYPE`, prueba de formato con
  cuatro problemas y evaluación de los 500; pendiente correr en Colab L4.

### Fase 8 — Finetuning con LoRA
- `LoraConfig(r=16, alpha=16, dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj"])`.
- Bucle manual con `transformers` + `peft`, labels enmascarados, fp16/bf16, gradient checkpointing.
- Monitoreo de pérdida de validación y pass@1 en 50 problemas de validación por época.
- Guardado del mejor adaptador de cada modelo en Drive; liberar GPU entre ambos.
- **Lista cuando:** existen `checkpoint_qwen2.5_1.5b_lora/` y `checkpoint_smollm2_1.7b_lora/`
  con sus tokenizadores y el historial de entrenamiento.
- **Estado:** escrita (D-039). Receta de D-021 con fp32 + autocast, gradient checkpointing, 5
  épocas, monitoreo por época, mejor adaptador e historial en `RESULTADOS_DIR`; pendiente correr en
  Colab L4.

### Fase 9 — Evaluación de los modelos finetuneados
- Carga de cada modelo base + su adaptador LoRA.
- Evaluación de los 500 de `test` con exactamente el mismo prompt y `max_new_tokens` que en la
  sección 7.
- Guardado de pass@1 y generaciones.
- **Lista cuando:** existen los dos pass@1 finetuneados y se liberó la GPU entre modelos.
- **Estado:** escrita (D-040). Carga de cada adaptador sobre su base, verificación de configuración
  y evaluación de los 500 con el mismo `evaluar_pass1`; pendiente correr en Colab L4.

### Fase 10 — Comparación y resultados
- Tabla de las cuatro configuraciones con pass@1 e intervalo de Wilson al 95 %.
- Gráfico de barras comparativo y análisis del aporte del finetuning en cada familia.
- Ejemplos cualitativos de soluciones correctas e incorrectas.
- Conclusión sobre cuál modelo es mejor en la tarea.
- **Lista cuando:** la tabla, la gráfica y los ejemplos están en el notebook y la conclusión es
  coherente con los números.

### Fase 11 — Conclusiones
- Discusión de resultados, seguimiento del entrenamiento, hiperparámetros y limitaciones.
- Comparación con lo esperado y aprendizajes del proyecto.
- **Lista cuando:** el notebook cierra con la discusión y no quedan celdas pendientes.

### Fase 12 — Smoke test en Colab T4/L4
- Corrida completa con menos datos o menos épocas para detectar errores rápido.
- **Lista cuando:** `Run All` termina sin excepciones en T4 o L4.

### Fase 13 — Run final y entrega
- `Run All` en L4 o A100 con los hiperparámetros definitivos.
- `.zip` con notebook ejecutado + los dos adaptadores, subido a Drive con permiso de lector.
- **Lista cuando:** la descarga funciona en modo incógnito y el zip contiene todo lo exigido.
