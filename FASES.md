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
- Revisiones fijas de MBPP, Qwen2.5-1.5B y SmolLM2-1.7B junto a las constantes.
- **Lista cuando:** en Colab se ve la GPU, el tipo de AMP y la semilla fijada.

### Fase 3 — Carga del dataset y exploración
- Carga de `google-research-datasets/mbpp` configuración `full` con revisión fija.
- Descripción de los splits `train` (374), `validation` (90), `test` (500) y `prompt` (10).
- Ejemplos reales de `text`, `code`, `test_list` y `test_setup_code`.
- Verificación de que ningún `task_id` se repite entre splits.
- Tabla comparativa de los tres splits y percentiles de largo en `train`, con histograma.
- Estructura del código con `ast`: imports, clases y funciones auxiliares.
- Revisión de `challenge_test_list` y de un ejemplo del split `prompt`.
- Longitudes en caracteres y palabras; el conteo en tokens se hace en la sección 4, cuando ya
  está cargado el tokenizador de cada modelo (D-028).
- **Lista cuando:** se imprimen los tamaños y ejemplos reales, y Jean aprueba el cierre.

### Fase 4 — Construcción de prompts y preprocesamiento
- Formato de instrucción único: problema + nombre/firma de la función objetivo.
- Función que arma el prompt y el target (código de referencia completo + EOS).
- Tokenización con `add_special_tokens=False`, BOS/EOS manual si aplica.
- Máscara de `labels` con `-100` para todos los tokens del prompt; verificación de longitudes.
- **Lista cuando:** un ejemplo imprime prompt, target, labels y el número de tokens enmascarados
  coincide con la longitud real del prompt.

### Fase 5 — Particiones (train / validation / test)
- Asignación de roles: `train` para finetuning, `validation` para monitoreo e hiperparámetros,
  `test` reservado para la evaluación final.
- Verificación de que no se usa `test` en entrenamiento ni en selección de hiperparámetros.
- **Lista cuando:** las tres particiones quedan documentadas y el notebook no toca `test` antes
  de la sección 7.

### Fase 6 — Implementación de la evaluación pass@1
- Extracción de código generado (bloques markdown o texto plano).
- Generación por lotes: padding a la izquierda, `pad_token = eos`, `attention_mask`, recorte del
  prompt, `do_sample=False`, `max_new_tokens=512`.
- Ejecución de código generado en subproceso aislado con timeout: código → `test_setup_code` →
  asserts.
- Cálculo de pass@1 y guardado de resultados por problema.
- **Lista cuando:** un modelo diminuto o un mock produce resultados coherentes y un bucle
  infinito se corta por timeout sin tumbar el notebook.

### Fase 7 — Evaluación de los modelos base (zero-shot)
- Carga de Qwen2.5-1.5B y SmolLM2-1.7B base, uno a la vez.
- Evaluación de los 500 problemas de `test` con la misma configuración de generación.
- Guardado de pass@1 y generaciones en Drive/resultados.
- **Lista cuando:** existen los dos pass@1 base y se liberó la GPU entre modelos.

### Fase 8 — Finetuning con LoRA
- `LoraConfig(r=16, alpha=16, dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj"])`.
- Bucle manual con `transformers` + `peft`, labels enmascarados, fp16/bf16, gradient checkpointing.
- Monitoreo de pérdida de validación y pass@1 en 50 problemas de validación por época.
- Guardado del mejor adaptador de cada modelo en Drive; liberar GPU entre ambos.
- **Lista cuando:** existen `checkpoint_qwen2.5_1.5b_lora/` y `checkpoint_smollm2_1.7b_lora/`
  con sus tokenizadores y el historial de entrenamiento.

### Fase 9 — Evaluación de los modelos finetuneados
- Carga de cada modelo base + su adaptador LoRA.
- Evaluación de los 500 de `test` con exactamente el mismo prompt y `max_new_tokens` que en la
  sección 7.
- Guardado de pass@1 y generaciones.
- **Lista cuando:** existen los dos pass@1 finetuneados y se liberó la GPU entre modelos.

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
