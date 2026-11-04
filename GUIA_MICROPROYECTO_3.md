# Guía del Microproyecto 3 — Instrucciones y recomendaciones

**Curso:** Modelos Avanzados para el Procesamiento de Lenguaje Natural (MAPLN) — Maestría en IA
**Proyecto:** generación condicional de código con LLMs, finetuning con LoRA y evaluación pass@1
**Modelos:** `Qwen/Qwen2.5-1.5B` y `HuggingFaceTB/SmolLM2-1.7B`
**Dataset:** `google-research-datasets/mbpp`, configuración `full`
**Métrica:** pass@1 sobre los 500 problemas del split `test`

> Este documento resume las instrucciones oficiales y las recomendaciones de ingeniería del
> proyecto. Las fuentes de verdad siguen siendo los PDFs de `PDF/`, la skill
> `.opencode/skills/mbpp-lora-pass1/SKILL.md` y `AGENTS.md`. Si hay conflicto, mandan los PDFs y
> la skill del curso.

---

## 1. Resumen del proyecto

El objetivo es ajustar dos modelos de lenguaje de familias distintas para resolver una tarea de
generación condicional de código: dada una instrucción en lenguaje natural que describe un
problema, el modelo debe generar una función en Python que lo resuelva.

Se comparan **cuatro configuraciones** sobre exactamente el mismo conjunto de prueba:

| # | Configuración | Modelo | Entrenamiento |
| :-: | :--- | :--- | :--- |
| 1 | Qwen base | Qwen2.5-1.5B | Sin entrenar (zero-shot) |
| 2 | Qwen + LoRA | Qwen2.5-1.5B | LoRA sobre MBPP `train` |
| 3 | SmolLM2 base | SmolLM2-1.7B | Sin entrenar (zero-shot) |
| 4 | SmolLM2 + LoRA | SmolLM2-1.7B | LoRA sobre MBPP `train` |

La comparación debe concluir cuál de los dos modelos obtiene mejor desempeño y cuánto aporta el
finetuning en cada familia.

---

## 2. Requisitos duros (si falla uno, la entrega vale 0)

1. **Dataset único:** `google-research-datasets/mbpp`, configuración `full`.
2. **Splits oficiales:** `train` (~374) para finetuning, `validation` (~90) para monitoreo e
   hiperparámetros, `test` (500) **solo** para la evaluación final.
3. **Modelos obligatorios:** `Qwen/Qwen2.5-1.5B` y `HuggingFaceTB/SmolLM2-1.7B` en su versión
   base. No se permite otro modelo.
4. **Cuatro configuraciones** exactamente: base y LoRA de cada modelo.
5. **Pass@1 sobre los 500 de `test`**, con una sola solución por problema y decodificación
   greedy (`do_sample=False`).
6. **Mismo prompt y mismo `max_new_tokens`** para la versión base y la finetuneada de cada
   modelo. Cada modelo usa su propio tokenizador.
7. **Labels del prompt en `-100`**: la pérdida solo considera los tokens del código de la
   solución.
8. **LoRA**, no full finetuning. Configuración de referencia: `r=16`, `lora_alpha=16`,
   `lora_dropout=0.05`, `target_modules=["q_proj","k_proj","v_proj","o_proj"]`.
9. **Entregar los dos adaptadores LoRA** (o el modelo fusionado) y el notebook ejecutado.
10. **Notebook autocontenido y `Run All`**, con las 11 secciones exactas.
11. **No usar `test`** para entrenar ni para elegir hiperparámetros.
12. **No usar TRL, Unsloth, HF Jobs** ni frameworks de entrenamiento ajenos a
    PyTorch + `transformers` + `peft` + `datasets`.

---

## 3. Entorno y flujo de trabajo

| Rol | Dónde | Qué se hace |
| :--- | :--- | :--- |
| Autoría | WSL2 sobre `/mnt/d/...` | Escribir celdas, generar el notebook, revisar diffs, git |
| Ejecución | Google Colab Pro | Entrenamiento, evaluación y salida final |
| GPU recomendada | L4 (24 GB, bf16) o A100 | T4 solo como respaldo |
| Artefactos | Google Drive | Adaptadores LoRA, resultados y `.zip` |

Flujo:

```text
WSL: editar src/cells/*.py
  └─> python3 build_notebook.py
  └─> python3 build_notebook.py --check
  └─> git add -A && git commit          # push solo con autorización
Colab: abrir/clonar el repo y Runtime > Run All
  └─> guardar adaptadores y resultados en Drive
  └─> al final, zip con notebook ejecutado + 2 adaptadores
```

Reglas de Colab:

- No editar el notebook en Colab: se pierde al regenerar desde `src/cells/`.
- Antes de un run largo, verificar `torch.cuda.get_device_name(0)` y
  `torch.cuda.is_bf16_supported()`.
- Para el run final, hacer `Runtime > Factory reset runtime` y partir de VRAM limpia.
- Guardar checkpoints en Drive después de cada época o modelo; las sesiones se desconectan.

---

## 4. Estructura obligatoria del notebook

El notebook debe tener estas **11 secciones**, en este orden y con estos títulos:

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

Contenido mínimo por sección:

| Sección | Qué debe incluir |
| :--- | :--- |
| 1 | Nombre, resumen, tabla de las 4 configuraciones, dataset, métrica y entorno |
| 2 | Instalación, imports, versiones, GPU, AMP, semilla, Drive y liberación de VRAM |
| 3 | Carga de MBPP, tamaños de splits, ejemplos de `text`, `code`, `test_list` y longitudes |
| 4 | Formato de prompt, tokenización, target, labels con `-100` y verificación |
| 5 | Roles de `train`, `validation` y `test`; prohibición de usar `test` antes de evaluar |
| 6 | Generación por lotes, extracción de código, sandbox y cálculo de pass@1 |
| 7 | pass@1 de Qwen base y SmolLM2 base sobre los 500 de `test` |
| 8 | LoRA de cada modelo, bucle de entrenamiento, monitoreo y guardado del mejor adaptador |
| 9 | pass@1 de cada adaptador sobre el mismo `test` y con la misma generación |
| 10 | Tabla de 4 configuraciones, Wilson 95 %, gráfica, ejemplos cualitativos y conclusión |
| 11 | Discusión, seguimiento del entrenamiento y limitaciones |

---

## 5. Diseño técnico recomendado

### 5.1 Dataset y particiones

- Cargar con `load_dataset("google-research-datasets/mbpp", "full")`.
- `train`: 374 ejemplos para el finetuning.
- `validation`: 90 ejemplos para pérdida de validación y pass@1 de monitoreo.
- `test`: 500 ejemplos reservados para la evaluación final.
- El split `prompt` (10 ejemplos) no se usa para entrenar ni evaluar.
- El campo `test_setup_code` se usa **solo al ejecutar** los tests, nunca en el prompt.

### 5.2 Prompt: problema + nombre/firma, sin fuga

El prompt debe ser idéntico en entrenamiento y evaluación. Propuesta:

```text
### Tarea
Escribe una función de Python que resuelva el siguiente problema.

### Problema
{text}

### Función
def {nombre}({argumentos}):
```

Reglas:

- Incluir el **nombre y la firma** de la función objetivo para que los asserts llamen a la
  función correcta.
- **No** incluir el cuerpo de la solución de referencia.
- **No** incluir helpers, clases ni imports del campo `code`.
- **No** incluir `test_list` ni `test_setup_code` en el prompt.

Para obtener el nombre y la firma, usar `ast` sobre `code` y localizar la función que llaman los
asserts de `test_list`. Si hay varias funciones, elegir la que aparece en los asserts; si no se
puede determinar, usar la última función de nivel superior.

El **target de entrenamiento** es el **código de referencia completo** (imports, clases, helpers
y la función objetivo), normalizado a `\n` y con EOS al final. Así el modelo aprende a generar
todo lo necesario, sin que el prompt revele la solución.

### 5.3 Tokenización y labels

- Tokenizar prompt y target por separado con `add_special_tokens=False`.
- Agregar BOS si el tokenizador lo define; agregar EOS al final del target.
- `input_ids = prompt_ids + target_ids + [eos_token_id]`.
- `labels = [-100] * len(prompt_ids) + target_ids + [eos_token_id]`.
- Verificar que el número de tokens enmascarados coincide con `len(prompt_ids)`, no con un valor
  fijo.
- `max_len = 512`: el código completo más largo de MBPP ronda los 432 tokens con el tokenizer de
  Qwen, así que 512 cubre sin truncar.
- Para entrenamiento, padding a la derecha y `labels` de padding en `-100`.
- Para generación, padding a la izquierda.

### 5.4 LoRA y entrenamiento

Configuración de referencia del curso:

```python
LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
```

Hiperparámetros recomendados:

- 3–5 épocas sobre `train` (empezar con 3).
- `learning_rate = 2e-4`.
- Batch efectivo 16: micro-batch 4 u 8 × acumulación de gradientes.
- `weight_decay = 0.0` (LoRA no necesita la regularización del preentrenamiento).
- Warmup corto y scheduler coseno.
- `max_len = 512`.
- fp16 + `GradScaler` en T4; bf16 en L4/A100.
- Gradient checkpointing con `use_cache=False` durante entrenamiento.
- Con PEFT + gradient checkpointing, llamar `model.enable_input_require_grads()`.

Bucle manual con PyTorch:

1. Cargar el modelo base con `AutoModelForCausalLM`.
2. Aplicar `get_peft_model` y `print_trainable_parameters()`.
3. Entrenar una época, calcular pérdida de validación.
4. Evaluar pass@1 en 50 problemas fijos de `validation`.
5. Guardar el adaptador si mejora el pass@1 de validación (desempate por pérdida).
6. Liberar GPU antes del siguiente modelo.

No usar `test` para elegir época, learning rate, rank ni ningún hiperparámetro.

### 5.5 Generación por lotes

- `tokenizer.padding_side = "left"`.
- `tokenizer.pad_token = tokenizer.eos_token` si no existe.
- `model.config.pad_token_id = tokenizer.pad_token_id`.
- Tokenizar con `padding=True`, `truncation=True`, `max_length=512` y pasar `**inputs` a
  `generate`.
- `max_new_tokens = 512`, `do_sample=False`.
- `pad_token_id = tokenizer.pad_token_id`.
- Recortar el prompt: `gen = out[:, inputs["input_ids"].shape[1]:]`.
- Decodificar con `skip_special_tokens=True`.
- `model.eval()` y `@torch.no_grad()`.
- `batch_size = 16`; bajar a 8 si hay OOM.
- La misma configuración para base y finetuneado.

### 5.6 Sandbox para ejecutar código generado

Nunca ejecutar el código del modelo directamente en el kernel. Usar un subproceso aislado:

```python
subprocess.run(
    [sys.executable, "-I", "-c", script],
    timeout=5,
    cwd=tempdir,
    capture_output=True,
    text=True,
)
```

Orden de ejecución:

1. Código generado.
2. `test_setup_code`.
3. Asserts de `test_list`.

El orden del snippet del PDF (`setup` antes del código) falla en casos como el task 367, donde el
setup usa `Node` antes de que el código lo defina. Ejecutar el código primero resuelve eso.

Recomendaciones adicionales:

- `-I` para aislar el entorno de usuario.
- `cwd` en un directorio temporal.
- Sin acceso a red ni credenciales.
- Timeout de 3–5 segundos por problema.
- Capturar `Exception` y `BaseException`; un fallo o timeout cuenta como incorrecto.
- No dejar procesos hijos vivos.

### 5.7 Métrica y análisis

- `pass@1 = problemas que pasan todos los asserts / 500`.
- Reportar el intervalo de Wilson al 95 %.
- Tabla con las 4 configuraciones: pass@1, IC95 %, tiempo y VRAM aproximada.
- Gráfica de barras comparativa.
- Ejemplos cualitativos: al menos 2 correctos y 2 incorrectos.
- Categorías de error: sintaxis, error en tiempo de ejecución, assert fallido, timeout,
  nombre/API incorrecta, truncamiento.
- Conclusión explícita: cuál modelo es mejor y cuánto aporta LoRA en cada familia.

### 5.8 Reproducibilidad

- `SEED = 42` para `random`, `numpy`, `torch` y `torch.cuda`.
- Revisiones fijas de MBPP, Qwen2.5-1.5B y SmolLM2-1.7B.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` antes de importar `torch`.
- `torch.backends.cudnn.deterministic=True` y `benchmark=False`.
- Guardar configuración de LoRA, hiperparámetros, semilla y revisiones junto a cada adaptador.
- Guardar las generaciones por problema en JSON/JSONL para poder auditar.

### 5.9 Colab y Drive

- `drive.mount("/content/drive")`.
- `DRIVE_DIR = "/content/drive/MyDrive/microproyecto_3"`.
- Guardar adaptadores en:
  - `checkpoint_qwen2.5_1.5b_lora/`
  - `checkpoint_smollm2_1.7b_lora/`
- Guardar resultados en `DRIVE_DIR/resultados/`.
- Si el adaptador ya existe, cargarlo y saltar el entrenamiento (reanudación idempotente).
- Después de cada modelo: `del`, `gc.collect()` y `torch.cuda.empty_cache()`.

### 5.10 Entregable

Un `.zip` en Google Drive con:

1. Notebook ejecutado, con las 11 secciones y salidas visibles.
2. `checkpoint_qwen2.5_1.5b_lora/` con `adapter_config.json`,
   `adapter_model.safetensors`, tokenizer y configuración.
3. `checkpoint_smollm2_1.7b_lora/` con los mismos archivos.

El enlace de Drive debe estar compartido con “cualquier persona con el enlace” y probado en una
ventana de incógnito. No incluir los modelos base: se descargan de Hugging Face.

---

## 6. Recomendaciones y decisiones clave

### Por qué incluir el nombre/firma y no solo el problema

MBPP no da el nombre de la función en `text`; los asserts llaman a un nombre específico. Si el
prompt es solo el problema, el modelo inventa otro nombre y el pass@1 se cae por una razón
trivial. El nombre/firma es la API mínima necesaria, no la solución.

### Por qué no incluir helpers ni `test_list`

Incluir helpers del campo `code` es una fuga parcial de la solución. Incluir `test_list` permite
hardcodear los casos de prueba. La skill del curso prohíbe ambas cosas. El modelo debe aprender a
generar el código completo a partir del problema.

### Por qué el target es el código completo

Si el target fuera solo el cuerpo de la función objetivo, el modelo no aprendería a generar los
helpers y clases que algunos problemas necesitan. Entrenar con el código de referencia completo
enseña al modelo a producir todo lo necesario.

### Por qué `max_new_tokens=512`

Las soluciones de MBPP son cortas (media ~58 tokens, p95 ~149), pero el máximo del `test` ronda
los 432 tokens. 256 truncaría algunas soluciones; 512 cubre el peor caso sin penalizar la
mayoría, porque la generación se detiene en EOS.

### Por qué `q/k/v/o` y no all-linear

La skill del curso fija `q/k/v/o`, `alpha=16` y `dropout=0.05`. Las skills genéricas
(`lora-qlora-recipes`, `peft-fine-tuning`) recomiendan all-linear, `alpha=2r` y `dropout=0`, pero
no son autoridad para este proyecto. Si se quiere explorar all-linear, hacerlo como **ablación
en `validation`** y justificarlo; nunca elegir con `test`.

### Por qué subprocess y no `exec`

El código lo genera un modelo y puede contener bucles infinitos, imports raros o llamadas al
sistema. Un `exec` directo puede colgar o tumbar la sesión de Colab. El subproceso con timeout
aísla el fallo y convierte el problema en “incorrecto”.

### Por qué LoRA y no full finetuning

El enunciado exige LoRA. Además, los adaptadores pesan decenas de MB, se guardan fácil en Drive y
se cargan con `PeftModel.from_pretrained`. No hace falta fusionar ni entregar los modelos base.

### Por qué L4/A100

L4 tiene 24 GB, bf16 y es suficiente para LoRA de 1.5B/1.7B. A100 acelera el run final, pero el
cuello de botella es la generación de 500 × 4, no el entrenamiento. T4 funciona como respaldo con
fp16.

### Por qué no TRL, Unsloth ni HF Jobs

La skill del curso limita el stack a PyTorch + `transformers` + `peft` + `datasets`. Además, el
notebook debe correr `Run All` en Colab y ser ejecutable por el evaluador; mover el entrenamiento
a HF Jobs rompe esa condición.

---

## 7. Skills del proyecto

### Jerarquía

1. **`mbpp-lora-pass1`** — skill del curso, obligatoria. Manda sobre todo lo demás.
2. **`AGENTS.md`** — decisiones de ingeniería del repositorio.
3. **Skills genéricas** — solo referencia; nunca contradicen a la skill del curso.

### Set actual

| Skill | Rol | Estado |
| :--- | :--- | :--- |
| `mbpp-lora-pass1` | Checklist obligatoria de MBPP, LoRA y pass@1 | Mantener |
| `lora-qlora-recipes` | Referencia de LoRA/QLoRA y failure modes | Mantener como referencia |
| `peft-fine-tuning` | Referencia de PEFT, LoRA/QLoRA, rank/alpha y target modules | Mantener como referencia |
| `python-testing-patterns` | pytest, fixtures y tests de funciones auxiliares | Mantener para tests locales |
| `find-skills` | Descubrir e instalar skills | Mantener |

### Skills retiradas

| Skill | Motivo |
| :--- | :--- |
| `trl-training` | Usa TRL, prohibido por el curso y por D-025 |
| `jupyter-notebook` | Su scaffolder choca con `build_notebook.py`; su checklist quedó en esta guía |

`huggingface-llm-trainer` no está instalada en el repo; el harness la anuncia, pero es de
TRL/Unsloth/HF Jobs y no se usa.

### Conflictos conocidos

- `lora-qlora-recipes` y `peft-fine-tuning` recomiendan all-linear, `alpha=2r` y `dropout=0`.
  La skill del curso fija `q/k/v/o`, `alpha=16`, `dropout=0.05`. **Gana la skill del curso.**
- `lora-qlora-recipes` asume Unsloth como camino rápido. **No se usa.**
- `peft-fine-tuning` incluye QLoRA con `bitsandbytes`. **No se necesita** para 1.5B/1.7B en L4.

### Comandos

```bash
# Ver skills del proyecto
npx skills list

# Quitar una skill
npx skills remove <skill> -y

# Instalar una skill en el proyecto
npx skills add <owner/repo@skill> -y

# Buscar skills
npx skills find "<tema>"
```

La skill `mbpp-lora-pass1` vive en `.opencode/skills/` y no la administra `npx skills`.

---

## 8. Checklists

### Antes de abrir Colab

- [ ] `python3 build_notebook.py --check` responde `OK`.
- [ ] El notebook tiene las 11 secciones exactas.
- [ ] El prompt no incluye `code`, helpers ni `test_list`.
- [ ] Los labels del prompt están en `-100`.
- [ ] El padding de generación es a la izquierda.
- [ ] `max_new_tokens` es el mismo para base y finetuneado.
- [ ] La ejecución de código usa subprocess con timeout.
- [ ] No se usa `test` para entrenar ni para elegir hiperparámetros.
- [ ] `DRIVE_DIR` apunta a `microproyecto_3`.

### Por fase

- [ ] Fase 0: repo, build y esqueleto.
- [ ] Fase 1: portada y tabla de 4 configuraciones.
- [ ] Fase 2: GPU, AMP, Drive y semilla verificados en Colab.
- [ ] Fase 3: MBPP cargado y ejemplos reales impresos.
- [ ] Fase 4: prompt, target y labels verificados con un ejemplo.
- [ ] Fase 5: roles de splits documentados.
- [ ] Fase 6: generación batched y sandbox probados con modelo tiny.
- [ ] Fase 7: pass@1 base de los dos modelos.
- [ ] Fase 8: dos adaptadores guardados en Drive.
- [ ] Fase 9: pass@1 finetuneado de los dos modelos.
- [ ] Fase 10: tabla, gráfica, ejemplos y conclusión.
- [ ] Fase 11: discusión y limitaciones.
- [ ] Fase 12: `Run All` sin excepciones en T4/L4.
- [ ] Fase 13: zip en Drive probado en incógnito.

### Antes de entregar

- [ ] Las 4 configuraciones usan el mismo `test` de 500.
- [ ] La generación es greedy y una solución por problema.
- [ ] El prompt y `max_new_tokens` son iguales para base y LoRA.
- [ ] Los dos adaptadores cargan con `PeftModel.from_pretrained`.
- [ ] El notebook ejecutado tiene salidas visibles y no quedan `Pendiente:`.
- [ ] El zip contiene notebook + 2 adaptadores.
- [ ] El enlace de Drive abre en incógnito sin pedir permiso.

---

## 9. Errores comunes

1. Entrenar o ajustar hiperparámetros con `test`.
2. Usar sampling o más de una solución por problema.
3. Cambiar el prompt o `max_new_tokens` entre base y finetuneado.
4. No enmascarar el prompt con `-100`.
5. Padding a la derecha en generación.
6. Evaluar solo un subconjunto del `test`.
7. Ejecutar código generado sin aislamiento ni timeout.
8. Ejecutar `test_setup_code` antes del código generado.
9. Entregar solo el notebook o solo los checkpoints.
10. No probar el enlace de Drive en incógnito.
11. Editar el `.ipynb` en Colab y perder el cambio al regenerar.
12. Incluir la solución de referencia o los asserts en el prompt.
13. Truncar soluciones por usar `max_new_tokens` demasiado bajo.
14. No liberar VRAM entre modelos.
15. No guardar checkpoints en Drive antes de una desconexión.

---

## 10. Comandos útiles

```bash
# Regenerar el notebook
python3 build_notebook.py

# Verificar que el notebook está sincronizado
python3 build_notebook.py --check

# Validar sintaxis de las celdas fuente
python3 -m py_compile build_notebook.py src/cells/*.py

# Git
git status
git add -A
git commit -m "feat(notebook): ..."
git push -u origin main   # solo con autorización

# Skills
npx skills list
npx skills find "lora"
npx skills add <owner/repo@skill> -y
npx skills remove <skill> -y

# En Colab: verificar GPU
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())"

# En Colab: montar Drive
from google.colab import drive
drive.mount("/content/drive")
```

---

## 11. Referencias

- `PDF/Lectura_instrucciones_microproyecto_3_codigo.pdf` — especificación oficial.
- `PDF/MAIA_MAPLN_2026_15_Microproyecto_3_ANEXO.pdf` — generación por lotes.
- `AGENTS.md` — decisiones de ingeniería (D-001 a D-026).
- `FASES.md` — checklist de fases.
- `.opencode/skills/mbpp-lora-pass1/SKILL.md` — checklist obligatoria del curso.
- `instrucciones_proyecto/` — extracción de texto de los PDFs.
- Hugging Face:
  - https://huggingface.co/datasets/google-research-datasets/mbpp
  - https://huggingface.co/Qwen/Qwen2.5-1.5B
  - https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B
  - https://huggingface.co/docs/peft
  - https://huggingface.co/docs/transformers
