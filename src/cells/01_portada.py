# %% [markdown]
# # Microproyecto 3: Generación condicional de código con LLMs
#
# **Curso:** Modelos Avanzados para el Procesamiento de Lenguaje Natural
# **Integrante:** Jean Pierre Betancourt Padilla
# **Fecha:** noviembre de 2026
#
# ---
#
# ## 1. Portada e introducción
#
# ### Resumen
#
# En este microproyecto ajustamos dos modelos de lenguaje de familias distintas para generar
# funciones en Python a partir de un enunciado en lenguaje natural. Trabajamos con MBPP, un corpus
# de problemas básicos de programación que trae, para cada problema, el enunciado, una solución de
# referencia y los `assert` con los que se verifica.
#
# Comparamos cuatro configuraciones para medir cuánto aporta el finetuning con LoRA y qué familia
# de modelo resuelve mejor la tarea:
#
# | # | Configuración | Modelo | Entrenamiento |
# | :-: | :--- | :--- | :--- |
# | 1 | Qwen base | Qwen2.5-1.5B | Sin entrenar (zero-shot) |
# | 2 | Qwen + LoRA | Qwen2.5-1.5B | LoRA sobre MBPP `train` |
# | 3 | SmolLM2 base | SmolLM2-1.7B | Sin entrenar (zero-shot) |
# | 4 | SmolLM2 + LoRA | SmolLM2-1.7B | LoRA sobre MBPP `train` |
#
# ### Datos
#
# Usamos `google-research-datasets/mbpp` en su configuración `full` con las particiones oficiales:
# `train` (~374 problemas) para el finetuning, `validation` (~90) para vigilar el entrenamiento y
# `test` (500) reservado solo para la evaluación final.
#
# El prompt que ve el modelo tiene el enunciado del problema y el nombre con la firma de la función
# que hay que escribir. No incluye la solución de referencia, ni las clases y helpers que trae el
# código, ni los casos de prueba: así el modelo tiene que resolver el problema y no puede copiarse.
# Lo que aprende a generar es el código de referencia completo, con sus imports y sus helpers.
#
# ### Evaluación
#
# La métrica es **pass@1**: un problema cuenta como correcto solo si el código generado pasa todos
# sus `assert`. Evaluamos los 500 problemas de `test` con una sola solución por problema y
# decodificación greedy, y usamos el mismo prompt y el mismo `max_new_tokens` en la versión base y
# en la finetuneada de cada modelo.
#
# El código generado se ejecuta en un subproceso aparte con un tiempo límite, nunca dentro del
# notebook, porque el modelo puede equivocarse y colgar la sesión.
#
# ### Entorno
#
# El notebook se escribió en WSL y se ejecuta completo en Google Colab con GPU L4 o A100. En L4 y
# A100 usamos bf16; si la sesión cae en una T4, pasamos a fp16 con `GradScaler` sin cambiar código.
# Los adaptadores LoRA y los resultados quedan en Google Drive para sobrevivir a una desconexión.
#
# Todo el pipeline es reproducible: fijamos la semilla, la revisión del dataset y las de los dos
# modelos, de modo que dos ejecuciones parten de los mismos datos y de los mismos pesos. El notebook
# sigue las once secciones que pide el enunciado y se puede ejecutar de principio a fin con
# `Run All`.
