# AGENTS.md — Bitácora de decisiones del Microproyecto 3

Este archivo es la **fuente única de verdad sobre las decisiones de ingeniería** del proyecto.
Cualquier agente o persona que modifique el repositorio debe leerlo antes de empezar y
**registrar aquí toda decisión nueva** (ver §9). No se cambia una decisión existente en
silencio: se marca como reemplazada y se agrega una nueva.

- **Curso:** Modelos Avanzados para el Procesamiento de Lenguaje Natural (MAPLN) — Maestría en IA.
- **Repositorio:** `fatbounduniandes/microproyecto_3_pln` (público).
- **Rama principal:** `main`.
- **Última actualización de esta bitácora:** 2026-11-03.

---

## 1. Fuentes de verdad

| Documento | Ruta | Uso |
| :--- | :--- | :--- |
| Especificación oficial | `PDF/Lectura_instrucciones_micro_proyecto_3_codigo.pdf` | Requisitos obligatorios, rúbrica, estructura del notebook y entregables |
| Anexo de generación por lotes | `PDF/MAIA_MAPLN_2026_15_Microproyecto_3_ANEXO.pdf` | Padding a la izquierda, `attention_mask`, `pad_token` y recorte del prompt |
| Skill del proyecto | `.opencode/skills/mbpp-lora-pass1/SKILL.md` | Checklist obligatoria de MBPP, LoRA y pass@1 |
| Decisiones de ingeniería | `AGENTS.md` (este archivo) | Cómo y por qué se implementa |

Los PDFs no se versionan (`.gitignore`); viven en local. En `instrucciones_proyecto/` se deja
una extracción de texto para búsqueda, pero **ante cualquier conflicto manda el PDF original**.
Si una decisión de aquí contradice la especificación o la skill, es un error y debe corregirse.

---

## 2. Entorno de trabajo

| Rol | Dónde | Qué se hace ahí |
| :--- | :--- | :--- |
| Autoría / desarrollo | **WSL2 (Ubuntu 24.04)** sobre `/mnt/d/...` | Escribir celdas, generar el notebook, revisar diffs, git |
| Ejecución GPU | **Google Colab Pro** (runtime L4 o A100; T4 como respaldo) | Entrenamiento, evaluación, generación, salida final |
| Almacenamiento de artefactos | **Google Drive** (montado en Colab) | Adaptadores LoRA, resultados y `.zip` de entrega |

Estado verificado de WSL: `python3 3.12.3`, sin `pip`, sin `torch`, sin `transformers`,
sin `nvidia-smi` (no hay GPU NVIDIA). **WSL no ejecuta el notebook.**

---

## 3. Decisiones (registro ADR)

Formato: `D-XXX · fecha · decisión · motivo · alternativas descartadas · consecuencias`.

### D-001 · 2026-11-03 · Entorno híbrido WSL (autoría) + Colab (ejecución)
- **Decisión:** El notebook se escribe y versiona en WSL; se ejecuta únicamente en Colab.
- **Motivo:** WSL no tiene GPU NVIDIA ni el stack de PyTorch; instalar torch CPU no aporta para
  un entrenamiento de 1.5B/1.7B. La iteración de código sí es local y rápida.
- **Descartado:** entrenar en WSL-CPU (horas por época, inviable); desarrollar directamente en
  Colab (se pierden historial, diffs y control ante desconexiones).
- **Consecuencia:** toda validación de ejecución ocurre en Colab; WSL solo genera el notebook y
  corre pruebas de sintaxis.

### D-002 · 2026-11-03 · GPU objetivo: **L4** para iterar y **A100** opcional para el run final
- **Decisión:** L4 (24 GB, bf16) para desarrollo y smoke tests; A100 (40/80 GB) solo si se quiere
  el `Run All` final más rápido. T4 (16 GB, fp16) queda como respaldo. No se usa TPU ni High-RAM.
- **Motivo:** L4 es el punto dulce para modelos de 1.5B/1.7B con LoRA; A100 acelera, pero el
  cuello de botella es la generación de 500 problemas × 4 configuraciones, no el entrenamiento.
- **Descartado:** TPU (el stack es PyTorch + HF, no XLA); High-RAM (los modelos y MBPP caben
  holgados en RAM).
- **Consecuencia:** el notebook detecta `torch.cuda.is_bf16_supported()` y usa bf16 en L4/A100
  o fp16 + `GradScaler` en T4, sin cambios de código.
  > La detección de bf16 se reemplaza por D-031: `is_bf16_supported()` devuelve `True` en T4 por
  > emulación.

### D-003 · 2026-11-03 · Sincronización vía GitHub con HTTPS + `gh`
- **Decisión:** Único mecanismo de sync: `git push` / `git pull` contra
  `https://github.com/fatbounduniandes/microproyecto_3_pln.git`.
- **Motivo:** `gh` ya está autenticado en WSL con scope `repo`; el remoto SSH no está configurado.
- **Descartado:** subir el `.ipynb` a mano entre WSL y Colab (rompe trazabilidad); sync solo por
  Drive (sin historial ni diffs).
- **Consecuencia:** en Colab se clona el repo o se abre el notebook desde GitHub. El notebook en
  sí no clona el repositorio (D-004).

### D-004 · 2026-11-03 · El notebook es **autocontenido** (requisito duro)
- **Decisión:** `notebook.ipynb` no importa módulos locales, no depende de `src/`, no ejecuta
  `!git clone` de código propio ni `%run` de scripts. Todo el código vive inline en sus celdas.
- **Motivo:** requisito explícito de "ejecutable de principio a fin (`Run All`)" y de entrega
  como archivo único.
- **Descartado:** paquete `src/` importado desde el notebook (obliga a clonar y rompe el `Run All`).
- **Consecuencia:** la modularidad se logra **antes** de generar el notebook (D-006); el artefacto
  final es un único `.ipynb` con celdas en orden y con salidas visibles.

### D-005 · 2026-11-03 · MBPP se carga en cada `Run All`, sin caché
- **Decisión:** cada ejecución hace `load_dataset("google-research-datasets/mbpp", "full")` y usa
  los splits oficiales. No se cachea el dataset en Drive ni en disco.
- **Motivo:** el dataset es pequeño (~974 ejemplos) y la especificación exige ese dataset y esos
  splits; cachearlo no aporta y complica la reproducibilidad.
- **Descartado:** persistir los ejemplos tokenizados (más rápido, pero innecesario y frágil).
- **Consecuencia:** el pipeline es reproducible y autocontenido; la descarga tarda segundos.

### D-006 · 2026-11-03 · Celdas fuente en `src/cells/` + `build_notebook.py` genera el `.ipynb`
- **Decisión:** el código se escribe en archivos `.py` numerados (`src/cells/01_portada.py`, …),
  y un script sin dependencias externas los ensambla en `notebook.ipynb`. **El `.ipynb` no se
  edita a mano.**
- **Motivo:** el JSON del notebook es ilegible en diffs y propenso a corrupción; escribir celdas
  como texto permite revisar cambios, reordenar y regenerar sin tocar JSON.
- **Descartado:** editar el `.ipynb` directamente (diffs ruidosos); `jupytext` (dependencia
  adicional innecesaria); `nbformat` (se evita cualquier dependencia).
- **Consecuencia:** única vía de cambio es `src/cells/` + `python3 build_notebook.py`. Si alguien
  edita el notebook en Colab, ese cambio se pierde en la próxima regeneración: **prohibido**.

### D-007 · 2026-11-03 · Reanudación idempotente de entrenamientos desde Drive
- **Decisión:** cada fase de entrenamiento comprueba si su adaptador LoRA existe en Drive; si
  existe, lo carga y salta el entrenamiento (imprime un aviso claro). Si no, entrena y lo guarda.
- **Motivo:** Colab desconecta sesiones; repetir un finetuning por una caída cuesta minutos y
  puede perder la corrida completa.
- **Descartado:** entrenar siempre desde cero (frágil); reanudar a mitad de época con estados de
  optimizador (complejidad alta, beneficio marginal para 3–5 épocas).
- **Consecuencia:** la corrida entregada se ejecuta una vez de punta a punta; las re-ejecuciones
  posteriores son rápidas y consistentes. El notebook deja explícito si entrenó o cargó checkpoint.

### D-008 · 2026-11-03 · Higiene de repositorio (`.gitignore` + `.gitattributes`)
- **Decisión:** se versiona **solo** lo pequeño y reproducible: notebook, celdas fuente, script
  de build, documentación, requisitos y skills. Se excluyen checkpoints, datasets, cachés, `.zip`,
  secretos y configuración local del editor.
- **Motivo:** GitHub rechaza archivos > 100 MB y los binarios no aportan trazabilidad; el
  entregable pesado vive en Drive.
- **Consecuencia:** los adaptadores **no** se recuperan del repo; se regeneran en Colab o se bajan
  de Drive si hiciera falta.

### D-009 · 2026-11-03 · Nombres canónicos de artefactos
- **Decisión:** `checkpoint_qwen2.5_1.5b_lora/`, `checkpoint_smollm2_1.7b_lora/` y
  `entrega_microproyecto_3.zip`, guardados en `/content/drive/MyDrive/microproyecto_3/`.
- **Motivo:** centralizar la ruta evita discrepancias entre celdas y facilita el zip de entrega.
- **Consecuencia:** una sola constante `DRIVE_DIR` en la celda de configuración; ninguna ruta
  absoluta dispersa en el código.

### D-010 · 2026-11-03 · Reproducibilidad
- **Decisión:** `SEED = 42` global (`random`, `numpy`, `torch`, `torch.cuda`), split determinista
  y revisiones fijas de MBPP, Qwen2.5-1.5B y SmolLM2-1.7B.
- **Motivo:** permite comparar las 4 configuraciones sobre exactamente los mismos datos y
  regenerar la evaluación sin variaciones.
- **Consecuencia:** fijar semilla antes de cargar datos, antes de inicializar cada modelo y antes
  de generar. En GPU algunos kernels siguen avisando con `warn_only`, pero el resultado esperado
  es el mismo.

### D-011 · 2026-11-03 · Los `push` se hacen solo cuando Jean lo indica
- **Decisión:** se puede trabajar y hacer commits locales, pero **ningún `git push`** se ejecuta
  sin autorización explícita de Jean en el momento.
- **Motivo:** él revisa cada fase antes de publicarla y quiere controlar qué llega al remoto.
- **Consecuencia:** al terminar una fase se reporta el estado y se espera la instrucción de
  subir. Si una fase se corrige, se puede reescribir el commit local antes de subir.

### D-012 · 2026-11-03 · Formato percent para las celdas y build determinista
- **Decisión:** `src/cells/NN_seccion.py` usa marcadores `# %%` (código) y `# %% [markdown]`
  con prefijo `# ` en cada línea de markdown. `build_notebook.py` genera ids de celda con
  `sha1(archivo:índice)`, escribe JSON con `indent=1` y **no** incluye timestamps ni rutas.
- **Motivo:** ids estables y salida byte a byte reproducible ⇒ diffs limpios y un modo
  `--check` fiable para detectar el notebook desactualizado.
- **Descartado:** ids aleatorios de `nbformat` (ruido en cada build); generar el `.ipynb` con
  `nbformat` (dependencia innecesaria); escribir markdown sin prefijo (rompe el resaltado).
- **Consecuencia:** antes de cada commit se corre `python3 build_notebook.py --check`; si falla,
  el notebook no corresponde a las celdas fuente y no debe subirse.

### D-013 · 2026-11-03 · Estilo del notebook: markdown breve y código sin comentarios
- **Decisión:** el markdown va en tono de estudiante, en primera persona del plural, con párrafos
  cortos, listas y tablas simples. Las celdas de código **no llevan comentarios ni docstrings**:
  la explicación vive en el markdown de la sección.
- **Motivo:** el entregable lo firma el equipo y debe leerse como trabajo propio. El código limpio
  se ve mejor y el markdown ya cumple la función explicativa.
- **Alternativas descartadas:** markdown denso con referencias al anexo (se siente ajeno);
  comentarios línea a línea en el código (ruido y apariencia de plantilla).
- **Consecuencia:** al escribir una celda nueva, primero el markdown explica qué se hace y luego
  el código lo hace sin comentarios. Cada celda de código lleva justo encima un markdown de una
  sola frase que anuncia lo que hace.

### D-014 · 2026-11-03 · El token de Hugging Face viene del entorno de Colab
- **Decisión:** no usar `userdata.get("HF_TOKEN")` ni `login()`. El notebook asume que `HF_TOKEN`
  ya existe como variable de entorno en el runtime de Colab y solo reporta si está definido.
- **Motivo:** los modelos y MBPP son públicos; el token solo mejora los límites de tasa. Pedirlo
  otra vez agrega código y pasos manuales que el evaluador no necesita repetir.
- **Consecuencia:** `huggingface_hub` lo toma solo del entorno. Si no existe, las descargas
  públicas funcionan igual, con límites más estrictos.

### D-015 · 2026-11-03 · Estilo de programación súper básico
- **Decisión:** el notebook se escribe con lo mínimo indispensable de Python: `for` e `if`
  explícitos, funciones cortas sin anotaciones de tipo, rutas como texto (`str`) en vez de
  `pathlib`, sin lambdas, sin decoradores propios y sin comprensiones de lista anidadas. Se
  prefieren varias líneas claras a una línea ingeniosa.
- **Motivo:** el código debe parecer escrito por un estudiante y poder explicarse línea por línea.
- **Alternativas descartadas:** `pathlib`, helpers genéricos y estilo funcional (más elegante,
  pero opaco para sustentar); `try/except` defensivos alrededor de todo.
- **Consecuencia:** `drive.mount()` se llama directo, sin *fallback* local: el notebook asume
  Colab (requisito del enunciado). Las funciones tienen parámetros con valor por defecto simples
  y nombres largos en español.

### D-016 · 2026-11-03 · El repositorio es público
- **Decisión:** `fatbounduniandes/microproyecto_3_pln` queda público. Se mantiene el remoto HTTPS
  y el flujo de `push` bajo autorización (D-011).
- **Motivo:** Colab ve repositorios públicos sin pedir PAT y el badge de "Open in Colab" funciona;
  el repo del MP2 ya funcionó así.
- **Consecuencia:** cualquier persona puede leer el repo; no hay secretos versionados porque el
  `HF_TOKEN` se toma del entorno (D-014) y los checkpoints viven en Drive (D-008).

### D-017 · 2026-11-03 · `resultados/` guarda notebooks ejecutados y no se versiona
- **Decisión:** los notebooks descargados desde Colab con sus salidas se guardan en `resultados/`
  (ignorado por git) y se usan como evidencia para revisar cada fase.
- **Motivo:** permiten leer las salidas reales sin contaminar el repo con copias ejecutadas ni
  duplicar el notebook entregable, que se regenera desde `src/cells/`.
- **Consecuencia:** el notebook oficial sigue siendo `notebook.ipynb`; la copia ejecutada que se
  entregue se genera al final (Fase 13).

### D-018 · 2026-11-03 · Prompt: problema + nombre/firma, sin solución de referencia ni tests
- **Decisión:** el prompt de entrenamiento y evaluación contiene la descripción del problema y el
  nombre/firma de la función objetivo. **No** incluye el cuerpo de la solución de referencia, ni
  helpers/clases del `code`, ni los `test_list`.
- **Motivo:** la skill `mbpp-lora-pass1` prohíbe incluir la solución de referencia y los asserts en
  el prompt. Incluir helpers del `code` también sería una forma de fuga. El nombre/firma es solo
  la API necesaria para que los asserts llamen a la función correcta.
- **Entrenamiento:** el target es el **código de referencia completo** (imports, clases, helpers y
  la función objetivo), con EOS. Así el modelo aprende a generar todo lo necesario.
- **Descartado:** prompt solo con el problema (el modelo inventa el nombre y el pass@1 se cae por
  una razón trivial); prompt con helpers del `code` (fuga parcial); prompt con `test_list`
  (permite hardcodear).
- **Consecuencia:** el pass@1 absoluto será más bajo que si se diera el preámbulo, pero el
  protocolo es limpio y la comparación base vs. LoRA sigue siendo válida.

### D-019 · 2026-11-03 · Protocolo pass@1 obligatorio
- **Decisión:** evaluar los **500 problemas de `test`** para las 4 configuraciones, con una sola
  solución por problema, `do_sample=False`, mismo prompt y mismo `max_new_tokens` para base y
  finetuneado de cada modelo. `pass@1 = aciertos / 500`.
- **Generación:** `tokenizer.padding_side = "left"`, `pad_token = eos_token` si falta,
  `model.config.pad_token_id`, `attention_mask`, `batch_size=16` (bajar a 8 si hay OOM),
  `max_new_tokens=512`, recorte `gen = out[:, inputs["input_ids"].shape[1]:]`.
- **Motivo:** es el protocolo que exige la especificación y la skill. 512 cubre el cuerpo más
  largo de MBPP (~385 tokens con el tokenizer de Qwen) sin truncar.
- **Consecuencia:** las 4 configuraciones son comparables; el batch no cambia el resultado porque
  la decodificación es greedy y la `attention_mask` ignora el padding.

### D-020 · 2026-11-03 · Ejecución segura del código generado
- **Decisión:** cada solución se ejecuta en un subproceso aislado
  (`subprocess.run([sys.executable, "-I", "-c", script], timeout=5)`) con cwd temporal, sin red
  ni credenciales. Orden de ejecución: **código generado → `test_setup_code` → asserts**.
- **Motivo:** el código lo produce un modelo y puede colgarse o fallar; un `exec` directo en el
  kernel puede tumbar la sesión. El orden del snippet del PDF (`setup` antes del código) falla en
  casos como el task 367, donde el setup usa `Node` antes de que el código lo defina.
- **Descartado:** `exec` directo (riesgoso); `multiprocessing` con timeout (más complejo y no
  mata el proceso hijo de forma fiable).
- **Consecuencia:** un bucle infinito o un `assert` fallido cuenta como problema incorrecto sin
  afectar la evaluación global.

### D-021 · 2026-11-03 · LoRA y bucle de entrenamiento
- **Decisión:** `LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05,
  target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM")`, 3–5 épocas,
  lr 2e-4, batch efectivo 16, `max_len=512`, weight decay 0.0, warmup y coseno, fp16/bf16 según
  GPU. Bucle manual con PyTorch, `transformers`, `peft` y `datasets`; sin TRL, Unsloth ni HF Jobs.
- **Motivo:** es la configuración de referencia de la skill y del enunciado; el bucle manual da
  control sobre el enmascaramiento de labels, el monitoreo y el guardado del mejor adaptador.
- **Descartado:** `Trainer`/TRL (más abstracción y dependencias); full finetuning (el enunciado
  exige LoRA); QLoRA (no hace falta para 1.5B/1.7B en L4/A100).
- **Consecuencia:** cada modelo se entrena por separado y se libera la GPU entre uno y otro.

### D-022 · 2026-11-03 · Monitoreo y selección del mejor adaptador
- **Decisión:** en cada época se registra la pérdida de `validation` y se calcula pass@1 sobre un
  subconjunto fijo de 50 problemas de `validation`. Se guarda el adaptador con mejor pass@1 de
  validación (desempate por menor pérdida); si no mejora, se conserva el anterior.
- **Motivo:** la rúbrica evalúa la idoneidad de los hiperparámetros y el seguimiento del desempeño;
  la pérdida sola no dice si el modelo aprueba asserts.
- **Descartado:** guardar solo la última época (puede sobreajustar); mirar `test` para elegir
  hiperparámetros (prohibido).
- **Consecuencia:** el adaptador entregado es el mejor por validación, no necesariamente el último.

### D-023 · 2026-11-03 · Los checkpoints son adaptadores LoRA
- **Decisión:** se entrega un adaptador por modelo con `save_pretrained` (y el tokenizador
  correspondiente), sin fusionar. El zip incluye notebook ejecutado + los dos adaptadores.
- **Motivo:** el enunciado permite solo el adaptador; pesa decenas de MB y el modelo base se
  descarga de Hugging Face.
- **Descartado:** fusionar y guardar el modelo completo (~3 GB por modelo, zip innecesariamente
  grande); `torch.save` del `state_dict` completo (peor portabilidad).
- **Consecuencia:** el zip de Drive es pequeño y el evaluador puede cargar cada adaptador con
  `PeftModel.from_pretrained`.

### D-024 · 2026-11-03 · Estructura obligatoria del notebook
- **Decisión:** 11 secciones con los títulos exactos y en el orden del PDF:
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
- **Motivo:** requisito explícito de la especificación; el incumplimiento anula la entrega.
- **Consecuencia:** `FASES.md` alinea sus fases 1–11 con estas secciones; las fases 0, 12 y 13
  son infraestructura y validación, no secciones del notebook.

### D-025 · 2026-11-03 · No usar TRL, Unsloth ni Hugging Face Jobs
- **Decisión:** el entrenamiento se implementa con PyTorch + `transformers` + `peft` + `datasets`.
  No se usan TRL, Unsloth, HF Jobs ni frameworks de entrenamiento externos.
- **Motivo:** la skill `mbpp-lora-pass1` prohíbe frameworks ajenos a los indicados; además, el
  notebook debe correr `Run All` en Colab y ser ejecutable por el evaluador. La skill
  `huggingface-llm-trainer` no se instala; si el harness la ofrece, solo se toma su tabla de
  hardware/costos, no su stack.
- **Descartado:** TRL `SFTTrainer` (dependencia extra y menos control del enmascaramiento);
  Unsloth (fuera del stack permitido); HF Jobs (rompe el `Run All` y la entrega autocontenida).
- **Consecuencia:** el código es más explícito y defendible, aunque un poco más largo.

### D-026 · 2026-11-03 · Gobierno de skills y guía del proyecto
- **Decisión:** `mbpp-lora-pass1` es la skill obligatoria y manda sobre cualquier skill genérica.
  Se mantienen `lora-qlora-recipes`, `peft-fine-tuning`, `python-testing-patterns` y
  `find-skills` como apoyo. Se retiran `trl-training` (TRL prohibido) y `jupyter-notebook` (su
  scaffolder choca con `build_notebook.py`; su checklist se incorporó a la guía). Se crea
  `GUIA_MICROPROYECTO_3.md` con las instrucciones y recomendaciones consolidadas.
- **Motivo:** evitar que skills genéricas contradigan la receta del curso (por ejemplo, all-linear,
  `alpha=2r`, `dropout=0`), reducir ruido y dejar un único documento de onboarding.
- **Descartado:** mantener todas las skills instaladas (más conflicto y contexto); eliminar todas
  las genéricas (se pierden referencias útiles de PEFT y testing).
- **Consecuencia:** `skills-lock.json` refleja el set nuevo. Las skills genéricas nunca se usan
  para cambiar los hiperparámetros obligatorios ni para sustituir la evaluación del curso.

### D-027 · 2026-11-03 · Estructura del notebook heredada del MP2
- **Decisión:** cada sección sigue el molde del `2_microproyecto`: un markdown que explica, y luego
  una celda de código por paso, cada una anunciada por un markdown de una frase. La portada repite
  el esquema del MP2 (título, autoría, `---`, resumen con la tabla de configuraciones, `### Datos`,
  `### Evaluación`, `### Entorno` y cierre de reproducibilidad).
- **Motivo:** el MP2 ya fue entregado y calificado con esa estructura; mantenerla hace el MP3
  reconocible, legible y consistente, y evita inventar un formato nuevo a mitad de curso.
- **Descartado:** portada con encabezados propios por bloque (más secciones y más ruido); secciones
  con una sola celda gigante (peor para `Run All` y para explicar cada paso).
- **Consecuencia:** las once secciones del PDF se ven iguales entre sí; cualquier sección nueva se
  escribe copiando ese molde.

### D-028 · 2026-11-03 · Las longitudes en tokens se miden en la sección 4
- **Decisión:** la sección 3 explora el dataset en caracteres y palabras; el conteo en tokens con
  cada tokenizador se hace en la sección 4, cuando ya se cargaron los dos tokenizadores.
- **Motivo:** las longitudes en tokens dependen del tokenizador y son dos; cargar tokenizadores en
  la sección 3 adelanta trabajo que la sección 4 necesita y duplica descargas si cambia el modelo.
- **Descartado:** medir tokens con el tokenizador de Qwen en la sección 3 (mezcla datos de un
  modelo con una exploración que debería ser neutral).
- **Consecuencia:** la sección 3 solo imprime tamaños del dataset y estadísticas de texto; el
  máximo de caracteres observado en `train` es 908, holgado frente a `max_len=512`.

### D-029 · 2026-11-03 · Exploración ampliada de la sección 3
- **Decisión:** además de cargar el dataset, la sección 3 verifica que no haya `task_id` repetidos
  entre splits, compara los tres splits con las mismas medidas, revisa con `ast` cuántas soluciones
  traen imports, clases o funciones auxiliares, grafica las longitudes y mira `challenge_test_list`
  y el split `prompt`.
- **Motivo:** la fuga de datos es el error más caro del proyecto y se descarta con una comprobación
  explícita; la comparación entre splits demuestra que `test` es representativo; la estructura del
  código justifica que el target sea la solución completa y no solo el cuerpo de la función.
- **Descartado:** dejar la exploración en cinco celdas (cumple, pero no deja evidencia de la
  separación de splits ni de la complejidad real de las soluciones).
- **Consecuencia:** la sección 3 pasa de 5 a 11 celdas de código y agrega una figura. Sigue sin
  cargar tokenizadores ni modelos (D-028) y el código mantiene el estilo básico de D-015, sin
  comentarios dentro del código (D-013).

### D-030 · 2026-11-05 · La exploración no analiza el contenido de `test`
- **Decisión:** en la sección 3, las estadísticas de longitud, la estructura del código y las
  figuras se calculan solo sobre `train` y `validation`. De `test` únicamente se imprime su tamaño
  y se revisan sus `task_id` para el chequeo de fuga.
- **Motivo:** el enunciado reserva `test` para la evaluación final. Aunque mirar longitudes no es
  ajustar hiperparámetros, hacerlo sobre `test` abre la puerta a decisiones informadas por el
  conjunto de prueba y complica la defensa del protocolo.
- **Descartado:** comparar los tres splits en la tabla y en las figuras (más vistoso, pero mezcla
  el conjunto reservado con la exploración).
- **Consecuencia:** las tres figuras (boxplots, curva acumulada y barras de estructura) comparan
  `train` contra `validation`. El tamaño y los identificadores de `test` se siguen verificando.

### D-031 · 2026-11-05 · La precisión mixta se elige por compute capability
- **Decisión:** usar bf16 solo si `torch.cuda.get_device_properties(0).major >= 8` (Ampere o
  superior: L4, A100). En T4 y en GPUs más viejas usar fp16 con `GradScaler`.
- **Motivo:** la corrida del 2026-11-05 en una T4 imprimió `Precisión mixta: bfloat16`. Desde
  PyTorch 2.5, `torch.cuda.is_bf16_supported(including_emulation=True)` devuelve `True` porque
  emula bf16, no porque la tarjeta lo soporte; entrenar así es lento y frágil, y además apaga el
  `GradScaler` que una T4 necesita.
- **Descartado:** confiar en `is_bf16_supported()` (falso positivo en T4); fijar bf16 a mano
  (falla en T4); fijar fp16 siempre (pierde el rango de bf16 en L4/A100).
- **Consecuencia:** la sección 2 imprime el compute capability junto a la GPU, y una T4 cae en
  fp16 + `GradScaler` sin tocar el resto del código. Reemplaza la detección descrita en D-002.

### D-032 · 2026-11-05 · Análisis descriptivo y conclusiones calculadas en la sección 3
- **Decisión:** la sección 3 incluye tabla descriptiva con pandas, análisis de vocabulario,
  estructura del código con `ast`, construcciones usadas, calidad del texto (tabs, `print`,
  `input`, duplicados) y prueba de Kolmogorov-Smirnov entre `train` y `validation`. Las
  conclusiones se generan con `display(Markdown(...))` a partir de los valores calculados, en vez
  de escribir cifras fijas en el texto.
- **Motivo:** el MP2 mostró que la exploración se evalúa por lo que permite concluir, no por el
  número de ejemplos impresos. Generar el texto desde las variables evita que las conclusiones se
  desactualicen y obliga a que cada afirmación tenga un número detrás.
- **Descartado:** conclusiones estáticas con cifras escritas a mano (se rompen al cambiar el
  dataset); limitar la exploración a longitudes y conteos (no justifica el prompt ni el target).
- **Consecuencia:** la sección 3 pasa a 18 celdas de código y 6 figuras, y suma `pandas` y
  `scipy` a las dependencias. Todo el análisis sigue limitado a `train` y `validation` (D-030).
  > Reemplazada por D-033 en lo relativo a cómo se escriben las conclusiones.

### D-033 · 2026-11-05 · Las conclusiones de la sección 3 van en markdown estático
- **Decisión:** las conclusiones de la exploración son una celda markdown con las cifras escritas
  a mano a partir de la corrida, no un texto generado con `display(Markdown(...))` ni f-strings.
- **Motivo:** el dataset está fijado por revisión y el notebook siempre produce los mismos
  valores, así que el texto dinámico no aporta; una celda de código de 150 líneas con f-strings es
  más difícil de leer, de sustentar y de mantener que el markdown, y además obliga a importar
  IPython. El markdown se ve igual en Colab y en GitHub.
- **Descartado:** conclusiones calculadas (D-032); texto con cifras sin verificar contra la
  corrida (por eso se transcriben los valores reales).
- **Consecuencia:** se elimina la celda de código final y el import de `IPython.display`. Si algún
  día cambia la revisión del dataset, hay que actualizar las cifras a mano.

### D-034 · 2026-11-06 · Formato del prompt: `### Problem` / `### Signature` / `### Solution`
- **Decisión:** el prompt es `### Problem\n{enunciado}\n\n### Signature\n{def ...:}\n\n### Solution\n`.
  La firma va en un bloque de código aparte y el target es el `code` de referencia completo
  (`rstrip()`) más el EOS. No se incluyen helpers en el prompt, ni `assert`, ni `test_setup_code`.
- **Motivo:** el PDF deja el formato a nuestra elección y solo exige que el prompt describa el
  problema y que el mismo formato se use en entrenamiento y evaluación. La firma es necesaria
  porque los `assert` llaman a la función por su nombre y solo 13 de los 374 enunciados de `train`
  lo mencionan. El bloque separado permite que el modelo escriba la solución completa desde cero:
  103 soluciones empiezan con imports, clases o helpers, y en 2 casos (717 y 940) la función
  objetivo no es el último `def`.
- **Descartado:** firma en prosa (el modelo tiene que interpretarla y reescribirla); prompt que
  termina en `def ...:` (no representa las 103 soluciones con preludio); prompt sin firma (el
  nombre se adivina y el pass@1 cae por una razón ajena a la programación).
- **Consecuencia:** el mismo `construir_prompt` se usa en entrenamiento, validación y `test`. En la
  sección 7 se verifica que un modelo base no continúe el bloque `### Signature`; si lo hace, se
  cambia el encabezado antes de entrenar.

### D-035 · 2026-11-06 · Tokenización, EOS y máscara de labels
- **Decisión:** tokenizar con `add_special_tokens=False`; agregar el EOS a mano y supervisarlo; no
  usar BOS; poner `pad_token = eos_token` cuando falte (SmolLM2); `padding_side = "left"` para la
  generación; en entrenamiento, lotes con relleno dinámico a la derecha armados en `preparar_lote`;
  `labels` en `-100` para el prompt y para el relleno, con los ids reales en la solución y el EOS;
  `MAX_LEN = 512` sin truncamiento, con error si un lote lo supera.
- **Motivo:** ninguno de los dos tokenizadores agrega BOS ni EOS, y el EOS es la señal de parada que
  el modelo debe aprender a emitir. El prompt tokenizado es prefijo exacto del texto completo en
  464/464 con ambos tokenizadores, así que la máscara por longitud es exacta. El total más largo
  medido es 346 tokens, de modo que 512 no recorta nada.
- **Descartado:** `padding="max_length"` (reserva 512 tokens por ejemplo y desperdicia memoria);
  truncamiento (nunca se activa y podría cortar el target); padding a la izquierda en entrenamiento
  (no aporta cuando la pérdida está enmascarada y complica la lectura de los `labels`).
- **Consecuencia:** `tokenizar_ejemplo` es la referencia para auditar y `preparar_lote` es la única
  vía para armar lotes de entrenamiento. La sección 5 aplica el pipeline a los tres splits.

### D-036 · 2026-11-06 · 50 de validación con semilla y sin modelo diminuto
- **Decisión:** el subconjunto de monitoreo son 50 problemas de `validation` elegidos con
  `random.Random(SEED).sample` y ordenados por `task_id` (`EJEMPLOS_MONITOREO`). El sandbox de la
  sección 6 se prueba con soluciones escritas a mano y no se descarga ningún modelo diminuto; la
  generación por lotes se estrena en la sección 7 con Qwen base.
- **Motivo:** los 90 problemas están ordenados por `task_id`, así que los primeros 50 podrían no
  representar al conjunto; una muestra aleatoria con semilla cubre todo el rango y es reproducible.
  Un modelo diminuto agrega una tercera descarga, no aporta sobre los casos sintéticos y roza la
  regla de usar solo los dos modelos obligatorios.
- **Descartado:** los primeros 50 (posible sesgo de orden); los 90 completos (multiplican la
  generación, que es el cuello de botella del proyecto); modelo diminuto (dependencia extra).
- **Consecuencia:** `EJEMPLOS_MONITOREO` queda fijo; cambiar la semilla cambia la lista y obliga a
  repetir el entrenamiento. `FASES.md` (Fase 6) se ajusta para decir “soluciones escritas a mano”.

### D-037 · 2026-11-06 · Evaluación pass@1: extracción, sandbox y registro por problema
- **Decisión:** `generar_soluciones` sigue la receta del anexo (padding a la izquierda,
  `attention_mask`, `pad_token = eos`, `do_sample=False`, `max_new_tokens=512` y recorte del
  prompt). `extraer_codigo` toma el primer bloque cercado o, si no hay, el primer sufijo que es
  Python válido, de modo que conserva imports, clases, asignaciones y funciones. `ejecutar_solucion`
  corre `subprocess.run([sys.executable, "-I", "-c", script], timeout=5)` en una carpeta temporal,
  con `HOME` temporal y sin variables de credenciales, en el orden código → `test_setup_code` →
  asserts. `evaluar_pass1` devuelve un resumen y guarda `pass1_<nombre>.json` con el código, el
  motivo y el tiempo de cada problema.
- **Motivo:** el protocolo exige una solución greedy por problema y ejecución aislada; guardar el
  código y el motivo permite mostrar ejemplos cualitativos en la sección 10 y diagnosticar los
  fallos. El orden código → setup → asserts es obligatorio en el problema 367, el único de `test`
  con `test_setup_code`, cuyo setup instancia `Node` antes de que exista si el setup va primero.
- **Descartado:** `exec` en el kernel (puede tumbar la sesión); `multiprocessing` (no mata los
  procesos hijos de forma fiable); extraer solo bloques cercados (los modelos base suelen responder
  en texto plano); guardar solo el agregado (sin evidencia para la sección 10).
- **Consecuencia:** las secciones 7 y 9 llaman `evaluar_pass1` una vez por configuración; la sección
  8 la reutiliza para los 50 de monitoreo. La primera generación real ocurre en la sección 7, donde
  se revisa el primer lote y se confirma que el modelo no continúa `### Signature`.

### D-038 · 2026-11-06 · Sección 7: evaluación base y primera lectura de `test`
- **Decisión:** `test` se usa por primera vez y solo para medir. Antes de generar se verifican sus
  metadatos (500 problemas, `task_id` únicos, cero solapamiento con `train` y `validation`, 3
  `assert` y 1 `test_setup_code`). Los modelos base se cargan en `AMP_DTYPE` (bf16 en L4/A100, fp16
  en T4) y se evalúan los 500 con `evaluar_pass1("qwen_base")` y `evaluar_pass1("smol_base")`.
  Antes de la corrida completa se generan 4 soluciones y se revisa el texto crudo para confirmar
  que el modelo no repite `### Signature` ni `### Problem` y que empieza con código.
- **Motivo:** los dos pass@1 base son la referencia del aporte de LoRA; el protocolo debe ser
  idéntico al de la sección 9. La prueba de formato es la que exige D-034 antes de entrenar.
  Cargar en media precisión basta para inferencia y deja VRAM para el modelo siguiente.
- **Descartado:** evaluar sobre `validation` (no es el protocolo del curso); cargar en fp32 para
  generar (más lento y sin beneficio); saltar la prueba de formato (un prompt mal cerrado se
  detectaría después de gastar la evaluación).
- **Consecuencia:** quedan `pass1_qwen_base.json` y `pass1_smol_base.json` en `RESULTADOS_DIR`; la
  sección 10 los lee junto con los dos finetuneados. La VRAM se libera con `modelo = None` y
  `liberar_vram()`: borrar solo el parámetro dentro de la función no suelta la referencia.

### D-039 · 2026-11-06 · Sección 8: receta de entrenamiento LoRA
- **Decisión:** `LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05,
  target_modules=["q_proj","k_proj","v_proj","o_proj"])`, 5 épocas, lr 2e-4, weight decay 0.0,
  batch efectivo 16, warmup del 5 % y scheduler coseno. El modelo base se carga en fp32 y el
  forward/backward corre con `torch.autocast` en `AMP_DTYPE`; se activa gradient checkpointing,
  `enable_input_require_grads()` y `use_cache=False` durante el entrenamiento. La pérdida de
  validación se promedia por token supervisado. Cada época se monitorea con los 50 problemas y se
  guarda el adaptador con mejor pass@1 (desempate por menor pérdida). El historial queda en
  `RESULTADOS_DIR/historial_<modelo>.json`. Si el adaptador ya existe, se carga y se salta el
  entrenamiento (D-007).
- **Motivo:** es la receta de la skill y de D-021/D-022. El checkpointing reduce las activaciones
  para que un 1.5B/1.7B entre con batch 16 en una L4 de 24 GB, y el base en fp32 con autocast evita
  entrenar los pesos LoRA en bf16 con AdamW. El JSON de historial permite reconstruir la tabla y la
  figura si la sesión se cae y toca reanudar.
- **Descartado:** cargar el base en bf16 (más frágil para AdamW); micro-batch 8 con acumulación
  (el checkpointing ya resuelve la memoria); guardar solo la última época (puede sobreajustar);
  `Trainer`/TRL (D-025).
- **Consecuencia:** se guardan adaptador, tokenizador e historial por modelo. Una segunda corrida
  con los adaptadores en Drive no vuelve a entrenar; para el `Run All` final hay que borrarlos si se
  quiere ver el entrenamiento completo.

### D-040 · 2026-11-06 · Sección 9: evaluación de los adaptadores
- **Decisión:** cada adaptador se carga con `PeftModel.from_pretrained` sobre su base en
  `AMP_DTYPE`. Antes de evaluar se comprueba que exista `adapter_config.json` y se imprimen rango,
  alpha, dropout, módulos y número de parámetros LoRA. Los 500 problemas se evalúan con el mismo
  `evaluar_pass1` de las secciones 6 y 7, y la VRAM se libera entre modelos.
- **Motivo:** el protocolo tiene que ser idéntico al de la sección 7 para que la diferencia mida
  LoRA y no un cambio de generación. Verificar la configuración y los parámetros demuestra que el
  checkpoint que se entrega es el que se entrenó.
- **Descartado:** fusionar el adaptador con `merge_and_unload()` (D-023 pide el adaptador suelto);
  reevaluar en `validation` (el pass@1 final es sobre `test`); evaluar un subconjunto.
- **Consecuencia:** quedan `pass1_qwen_lora.json` y `pass1_smol_lora.json`; la sección 10 compara
  las cuatro configuraciones y la sección 11 discute el resultado.

---

## 4. Reglas duras del proyecto (no negociables)

1. Dataset único `google-research-datasets/mbpp`, configuración `full`, splits oficiales.
2. Modelos obligatorios: `Qwen/Qwen2.5-1.5B` y `HuggingFaceTB/SmolLM2-1.7B` (base).
3. Exactamente 4 configuraciones: Qwen base, Qwen + LoRA, SmolLM2 base, SmolLM2 + LoRA.
4. `train` solo para finetuning; `validation` para monitoreo e hiperparámetros; `test` (500)
   exclusivamente para la evaluación final.
5. `pass@1` sobre los 500 de `test`, una solución por problema, greedy, mismo prompt y
   `max_new_tokens` para base y finetuneado.
6. Labels del prompt en `-100`; la pérdida solo considera el código de la solución.
7. Padding a la izquierda en generación; `pad_token = eos_token` si falta; `attention_mask`.
8. Ejecutar código generado en subproceso aislado con timeout; nunca en el kernel.
9. LoRA con `r=16`, `alpha=16`, dropout 0.05, target `q/k/v/o`; justificar cualquier cambio.
10. Entregar los dos adaptadores LoRA y el notebook ejecutado en un `.zip` de Drive.
11. Notebook autocontenido, `Run All`, con las 11 secciones exactas.
12. No usar `test` para elegir hiperparámetros ni para entrenar.
13. No hacer `push` sin autorización de Jean.
14. No introducir TRL, Unsloth, HF Jobs ni frameworks de entrenamiento ajenos.

---

## 5. Flujo de trabajo

```
WSL: editar src/cells/*.py
  └─> python3 build_notebook.py        # regenera notebook.ipynb
  └─> python3 build_notebook.py --check
  └─> git add -A && git commit         # push solo con autorización
Colab: abrir/clonar el repo y hacer Runtime > Run All
  └─> guarda adaptadores y resultados en Drive
  └─> al final, zip con notebook ejecutado + 2 adaptadores
```

### Fases

| # | Fase | Entorno | Criterio de salida |
| :-- | :--- | :--- | :--- |
| 0 | Repo, `.gitignore`, bitácora, build, esqueleto | WSL | `build_notebook.py --check` en OK |
| 1 | Portada e introducción | WSL | Sección 1 con nombre, resumen y tabla de 4 configuraciones |
| 2 | Librerías | Colab L4 | GPU, AMP, Drive y semilla verificados |
| 3 | Carga del dataset y exploración | Colab/WSL | Splits y ejemplos reales impresos |
| 4 | Prompts y preprocesamiento | WSL | Labels `-100` y longitudes verificadas |
| 5 | Particiones | WSL | Roles train/validation/test sin fuga |
| 6 | Evaluación pass@1 | WSL/Colab | Generación batched y sandbox probados con modelo tiny |
| 7 | Evaluación base | Colab L4 | pass@1 de los dos modelos base sobre 500 |
| 8 | Finetuning LoRA | Colab L4/A100 | Dos adaptadores guardados en Drive |
| 9 | Evaluación finetuneada | Colab L4 | pass@1 de los dos adaptadores sobre 500 |
| 10 | Comparación y resultados | WSL/Colab | Tabla de 4 configuraciones, gráficas y ejemplos |
| 11 | Conclusiones | WSL | Discusión y limitaciones |
| 12 | Smoke test | Colab T4/L4 | `Run All` sin excepciones |
| 13 | Run final y entrega | Colab L4/A100 | `.zip` en Drive probado en incógnito |

### Regla de oro de Colab
- No editar el notebook en Colab.
- Antes de un run largo: elegir L4 o A100 y verificar `torch.cuda.get_device_name(0)` y
  `torch.cuda.is_bf16_supported()`.
- Para el run final: `Runtime > Factory reset runtime` para partir de VRAM limpia.

---

## 6. Estructura del repositorio

```
.
├── AGENTS.md                  # esta bitácora
├── FASES.md                   # checklist de fases (una a la vez)
├── README.md                  # descripción breve y cómo ejecutar
├── PDF/                       # enunciado y anexo (no versionado)
├── instrucciones_proyecto/    # extracción de texto para búsqueda
├── src/cells/                 # celdas fuente numeradas (autoría)
├── build_notebook.py          # ensambla notebook.ipynb
├── notebook.ipynb             # ENTREGABLE generado (no editar a mano)
├── requirements-colab.txt     # versiones fijadas para Colab
├── .gitignore
└── .gitattributes
```

No versionado: `PDF/`, `checkpoint_*`, `*.safetensors`, `*.zip`, `resultados/`, cachés de HF.

---

## 7. Convenciones de código

- Python 3.10+ (el de Colab), `torch` + `transformers` + `peft` + `datasets`; nada exótico.
- Nombres en español para variables de dominio (`perdida`, `tokens_validos`) y en inglés para
  API de PyTorch (`input_ids`, `logits`).
- Constantes en mayúsculas al inicio de la celda de configuración.
- Cada celda imprime un resumen breve de lo que produjo (shapes, conteos, rutas, tiempo).
- Semillas fijas antes de cualquier operación aleatoria (D-010).
- **Código sin comentarios ni docstrings** (D-013): la explicación va en el markdown.
- **Estilo básico** (D-015): `for` e `if` explícitos, sin list comprehensions anidadas, sin
  lambdas, sin decoradores propios; rutas como `str`, sin `pathlib`; sin anotaciones de tipo;
  nombres de variables largos y obvios.

---

## 8. Estado actual

- [x] `git init`, rama `main`.
- [x] Skills del proyecto curadas: `mbpp-lora-pass1`, `lora-qlora-recipes`,
      `peft-fine-tuning`, `python-testing-patterns`, `find-skills`. Retiradas `trl-training` y
      `jupyter-notebook`.
- [x] `.gitignore` inicial y `.gitattributes`.
- [x] Fase 0: `AGENTS.md`, `FASES.md`, `README.md`, `GUIA_MICROPROYECTO_3.md`,
      `build_notebook.py`, `requirements-colab.txt` y esqueleto de las 11 secciones.
- [x] `build_notebook.py --check` en OK.
- [x] Fase 1: Portada e introducción (alineada al estilo del MP2).
- [x] Fase 2: Librerías (corrida en Colab T4; bf16 por emulación corregido en D-031).
- [x] Fase 3: Carga del dataset y exploración (corrida en Colab; análisis ampliado con 6 figuras,
      vocabulario, calidad y KS; conclusiones en markdown, D-030 y D-033; pendiente re-ejecutar).
- [ ] Fase 4: Prompts y preprocesamiento (sección 4 escrita con formato `### Signature`, target
      completo + EOS y máscara `-100`; D-034 y D-035; corrida en Colab, pendiente revisar salidas).
- [ ] Fase 5: Particiones (sección 5 escrita con roles, 50 de monitoreo con semilla y presupuesto
      de épocas; D-036; pendiente correr en Colab).
- [ ] Fase 6: Evaluación pass@1 (sección 6 escrita con generación por lotes, extracción tolerante,
      sandbox con timeout y JSON por problema; D-037; pendiente correr en Colab).
- [ ] Fase 7: Evaluación base (sección 7 escrita con integridad de `test`, carga en `AMP_DTYPE`,
      prueba de formato y evaluación de los 500; D-038; pendiente correr en Colab L4).
- [ ] Fase 8: Finetuning LoRA (sección 8 escrita con la receta de D-021, gradient checkpointing,
      monitoreo por época, mejor adaptador e historial; D-039; pendiente correr en Colab L4).
- [ ] Fase 9: Evaluación finetuneada (sección 9 escrita con carga de adaptadores, verificación de
      configuración y evaluación de los 500; D-040; pendiente correr en Colab L4).
- [ ] Fase 10: Comparación y resultados.
- [ ] Fase 11: Conclusiones.
- [ ] Fase 12: Smoke test.
- [ ] Fase 13: Run final y entrega.

---

## 9. Cómo registrar una decisión nueva

1. Agrega `### D-0NN · AAAA-MM-DD · Título corto` al final de §3 (no borres las anteriores).
2. Incluye: **Decisión**, **Motivo**, **Alternativas descartadas**, **Consecuencia**.
3. Si reemplaza una decisión previa, escribe en la vieja `> Reemplazada por D-0NN`.
4. Actualiza §8 (estado) si corresponde.
5. Commit aparte: `docs(agents): D-0NN <título>`.
