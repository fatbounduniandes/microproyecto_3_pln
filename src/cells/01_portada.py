# %% [markdown]
# # Microproyecto 3: Generación condicional de código con LLMs
#
# **Curso:** Modelos Avanzados para el Procesamiento de Lenguaje Natural
# **Integrante:** Jean Pierre Betancourt Padilla
# **Fecha:** noviembre de 2026
#
# ## 1. Portada e introducción
#
# ### Resumen
#
# En este microproyecto ajustamos dos modelos de lenguaje de familias distintas para resolver una
# tarea de generación condicional de código: dada una instrucción en lenguaje natural que describe
# un problema, el modelo debe generar una función en Python que lo resuelva. Usamos MBPP, un corpus
# de problemas básicos de programación con su solución de referencia y sus `assert` de verificación.
#
# El objetivo es medir el efecto del finetuning con LoRA sobre el desempeño en la tarea y comparar
# las dos familias de modelos. Para cada modelo evaluamos su versión original (zero-shot) y su
# versión finetuneada, siempre sobre el mismo conjunto de prueba y con la misma configuración de
# generación.
#
# ### Las cuatro configuraciones
#
# | # | Configuración | Modelo | Entrenamiento |
# | :-: | :--- | :--- | :--- |
# | 1 | Qwen base | Qwen2.5-1.5B | Sin entrenar (zero-shot) |
# | 2 | Qwen + LoRA | Qwen2.5-1.5B | LoRA sobre MBPP `train` |
# | 3 | SmolLM2 base | SmolLM2-1.7B | Sin entrenar (zero-shot) |
# | 4 | SmolLM2 + LoRA | SmolLM2-1.7B | LoRA sobre MBPP `train` |
#
# ### Datos y evaluación
#
# Usamos `google-research-datasets/mbpp` en su configuración `full`, con las particiones oficiales:
# `train` (~374 ejemplos) para el finetuning, `validation` (~90) para el monitoreo del entrenamiento
# y `test` (500) reservado exclusivamente para la evaluación final.
#
# La métrica es **pass@1**: un problema cuenta como correcto solo si la solución generada pasa todos
# sus `assert`. Generamos una sola solución por problema con decodificación greedy y usamos el mismo
# prompt y el mismo `max_new_tokens` para la versión base y la finetuneada de cada modelo.
#
# ### Entorno
#
# El notebook se escribió y versionó en WSL y se ejecuta en Google Colab con GPU L4 o A100. Los
# adaptadores LoRA y los resultados se guardan en Google Drive para sobrevivir a las desconexiones.
# Todo el código vive dentro del notebook, de modo que se puede ejecutar de principio a fin con
# `Run All`.
