---
name: MBPP LoRA pass@1
description: Reglas y checklist del Microproyecto 3 de MAPLN (generación condicional de código). Úsala siempre que se trabaje en el notebook, el entrenamiento LoRA o la evaluación pass@1 de Qwen2.5-1.5B y SmolLM2-1.7B sobre MBPP, o cuando haya que revisar que la entrega cumple los requisitos obligatorios del curso.
metadata:
  opencode/autoinvoke: true
---

# Microproyecto 3 — LoRA + MBPP + pass@1

Skill específica de este repositorio. El incumplimiento de cualquiera de los
requisitos marcados como **OBLIGATORIO** anula la entrega (nota 0).

## Requisitos duros (no negociables)

- **Dataset OBLIGATORIO**: `google-research-datasets/mbpp`, configuración `full`.
  Splits: `train` (~374) para fine-tuning, `validation` (~90) para monitoreo e
  hiperparámetros, `test` (500) **exclusivamente** para la evaluación final.
  No se permite otro dataset ni datos externos para entrenar.
- **Modelos OBLIGATORIOS**: `Qwen/Qwen2.5-1.5B` y `HuggingFaceTB/SmolLM2-1.7B`
  (variantes base, Apache-2.0). No se permite ningún otro modelo. Si el
  enunciado enlaza una variante concreta, esa manda.
- **Librerías**: PyTorch + `transformers`, `peft`, `datasets` (y utilidades
  estándar). No introducir frameworks de entrenamiento ajenos a los indicados.
- **Cuatro configuraciones** comparadas sobre el mismo `test`:
  1. Qwen2.5-1.5B zero-shot
  2. Qwen2.5-1.5B + LoRA
  3. SmolLM2-1.7B zero-shot
  4. SmolLM2-1.7B + LoRA
- **Checkpoints**: entregar un adaptador LoRA por modelo (o el modelo fusionado
  con `merge_and_unload()`), guardado con `save_pretrained`/`torch.save`.
- Todo el código debe ser propio y ejecutable por el evaluador.

## Protocolo pass@1 (OBLIGATORIO)

- Evaluar los **500 problemas del split `test`**, el mismo conjunto para las
  cuatro configuraciones.
- **Una sola solución por problema**, decodificación **greedy**
  (`do_sample=False`).
- Misma configuración de generación (prompt, `max_new_tokens`, criterios de
  parada) para la versión base y la fine-tuneada de cada modelo. Cada modelo
  usa su propio tokenizador.
- `pass@1 = problemas que pasan todos sus asserts / problemas evaluados`.
- Un problema cuenta como correcto solo si la solución generada pasa **todos**
  los `assert` de `test_list`.

### Ejecución segura del código generado

El código lo produce un modelo: no confiar en él. Ejecutar cada solución en un
subproceso aislado con timeout (por ejemplo `subprocess.run([sys.executable,
"-I", "-c", script], timeout=...)`) en lugar de un `exec` directo en el kernel.
Nunca ejecutar código generado con acceso a red, credenciales o al repositorio.

## Checklist de implementación

### Prompt y etiquetas

- Definir **un** formato de instrucción explícito y usarlo igual en
  entrenamiento y evaluación. No incluir la solución de referencia ni los
  asserts en el prompt.
- Enmascarar los tokens del prompt con `-100` en `labels`, para que la pérdida
  solo considere los tokens de la solución:

```python
labels = input_ids.clone()
labels[prompt_len:] = input_ids[prompt_len:]
labels[:prompt_len] = -100
```

  Verificar que el número de tokens enmascarados coincide con la longitud real
  del prompt (sin contar padding), no con un valor fijo.

### Generación por lotes

- `tokenizer.padding_side = "left"` antes de tokenizar (modelos decoder-only).
- `tokenizer.pad_token = tokenizer.eos_token` si no existe, y fijar
  `model.config.pad_token_id`.
- Pasar `attention_mask` (con `**inputs` se pasa solo).
- Recortar el prompt de la salida:
  `gen = out[:, inputs["input_ids"].shape[1]:]`.
- `@torch.no_grad()` y `model.eval()` durante evaluación.
- Empezar con `batch_size=16`; bajar si hay OOM. `max_new_tokens` acotado
  (las soluciones de MBPP son cortas).

### Entrenamiento LoRA

- Configuración de referencia: `r=16`, `lora_alpha=16`, `lora_dropout=0.05`,
  `target_modules=["q_proj","k_proj","v_proj","o_proj"]`,
  `task_type="CAUSAL_LM"`. Justificar cualquier cambio.
- 3–5 épocas sobre `train`; monitorear pérdida de `validation`.
- Entrenar un modelo a la vez y liberar GPU antes del siguiente
  (`del model, trainer`, `gc.collect()`, `torch.cuda.empty_cache()`).
- Fijar semilla y documentar hiperparámetros y criterio de selección.

## Estructura OBLIGATORIA del notebook

El notebook debe ser autocontenido y ejecutable con **Run All**, con estas
secciones en este orden y con estos títulos:

1. Portada e introducción
2. Librerías
3. Carga del dataset y exploración
4. Construcción de prompts y preprocesamiento
5. Particiones (train / validation / test)
6. Implementación de la evaluación pass@1
7. Evaluación de los modelos base (zero-shot)
8. Finetuning con LoRA
9. Evaluación de los modelos finetuneados
10. Comparación y resultados
11. Conclusiones

La sección 10 debe comparar las cuatro configuraciones, incluir ejemplos
cualitativos y concluir cuál modelo es mejor y cuánto aporta el fine-tuning.
La sección 11 debe discutir seguimiento del entrenamiento y limitaciones.

## Errores que invalidan o penalizan la entrega

- Entrenar con `test` o ajustar hiperparámetros mirando `test`.
- Usar sampling o más de una solución por problema.
- Cambiar el prompt o `max_new_tokens` entre base y fine-tuneado.
- No enmascarar el prompt (`-100`) y entrenar a predecir la instrucción.
- Padding a la derecha en generación por lotes.
- Evaluar solo un subconjunto del `test`.
- Ejecutar código generado sin aislamiento ni timeout.
- Entregar solo el notebook sin los dos checkpoints (o viceversa).

## Verificación antes de cerrar una tarea

1. ¿Se usó MBPP `full` y solo sus splits oficiales?
2. ¿Son exactamente los dos modelos obligatorios?
3. ¿Las cuatro configuraciones usan el mismo `test` y la misma generación?
4. ¿La generación es greedy y una solución por problema?
5. ¿Los `labels` del prompt son `-100`?
6. ¿El notebook corre de principio a fin y respeta las 11 secciones?
7. ¿Existen y cargan los dos checkpoints LoRA?
