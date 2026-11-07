# Microproyecto 3 — Generación condicional de código con LLMs

**Curso:** Modelos Avanzados para el Procesamiento de Lenguaje Natural (MAPLN) — Maestría en IA
**Tema:** finetuning con LoRA de Qwen2.5-1.5B y SmolLM2-1.7B sobre MBPP, evaluado con pass@1.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fatbounduniandes/microproyecto_3_pln/blob/main/notebook.ipynb)

## Qué contiene

- `notebook.ipynb`: entregable autocontenido, generado desde `src/cells/`. No se edita a mano.
- `src/cells/`: celdas fuente en formato percent (`# %%` y `# %% [markdown]`).
- `build_notebook.py`: ensambla el notebook; no tiene dependencias externas.
- `AGENTS.md`: bitácora de decisiones de ingeniería.
- `FASES.md`: checklist de avance, una fase a la vez.
- `GUIA_MICROPROYECTO_3.md`: instrucciones y recomendaciones consolidadas.
- `requirements-colab.txt`: versiones de las librerías para Colab.
- `instrucciones_proyecto/`: extracción de texto de los PDFs para búsqueda.
- `resultados/`: notebooks ejecutados y resultados (no se versiona).

## Resultados

pass@1 sobre los 500 problemas de `test`, con decodificación greedy y el mismo prompt en las
cuatro configuraciones (corrida en Colab L4):

| Configuración | Aciertos | pass@1 |
| :--- | ---: | ---: |
| Qwen2.5-1.5B base | 206/500 | 41.2 % |
| Qwen2.5-1.5B + LoRA | 218/500 | **43.6 %** |
| SmolLM2-1.7B base | 23/500 | 4.6 % |
| SmolLM2-1.7B + LoRA | 158/500 | **31.6 %** |

La sección 10 del notebook incluye la tabla con intervalos de Wilson al 95 %, la comparación de
motivos de fallo, la matriz de transición base → LoRA y ejemplos cualitativos.

## Cómo ejecutar

1. Abrir el notebook en Colab con el badge de arriba, o clonar el repo y abrir
   `notebook.ipynb`.
2. Elegir runtime **L4** o **A100** (T4 funciona como respaldo).
3. Ejecutar `Runtime > Run All`.
4. Los adaptadores LoRA y los resultados se guardan en
   `/content/drive/MyDrive/microproyecto_3/`.

## Entregable

Un `.zip` en Google Drive con:

1. El notebook ejecutado, con las 11 secciones y las salidas visibles.
2. Los dos adaptadores LoRA: `checkpoint_qwen2.5_1.5b_lora/` y
   `checkpoint_smollm2_1.7b_lora/`.

El enlace de Drive debe estar compartido con permiso “cualquier persona con el enlace” y
verificado en una ventana de incógnito.

## Estructura del notebook

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
