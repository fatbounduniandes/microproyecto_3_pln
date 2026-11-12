# %% [markdown]
# ## 5. Particiones (train / validation / test)
#
# MBPP ya viene partido en tres conjuntos y no los tocamos: usamos los splits oficiales tal como
# están. La única decisión nuestra es qué rol cumple cada uno. `train` se usa para el finetuning,
# `validation` para monitorear el entrenamiento y elegir la mejor época, y `test` queda reservado
# para la evaluación final: no se tokeniza ni se genera nada de él antes de la sección 7.
#
# De `validation` apartamos 50 problemas fijos para revisar el avance después de cada época. Los
# elegimos al azar con la semilla del proyecto para no quedarnos con un extremo del conjunto.

# %% [markdown]
# Dejamos escrito el rol de cada split y qué se permite hacer con él.

# %%
filas_roles = [
    {
        "split": "train",
        "problemas": len(dataset["train"]),
        "rol": "entrenamiento",
        "uso": "finetuning con LoRA; nunca se evalúa",
    },
    {
        "split": "validation",
        "problemas": len(dataset["validation"]),
        "rol": "monitoreo",
        "uso": "pérdida y pass@1 para elegir época e hiperparámetros",
    },
    {
        "split": "test",
        "problemas": len(dataset["test"]),
        "rol": "evaluación final",
        "uso": "pass@1 final; no se usa para decidir nada",
    },
]

tabla_roles = pd.DataFrame(filas_roles)
print(tabla_roles.to_string(index=False))

# %% [markdown]
# Guardamos las listas de entrenamiento y validación y calculamos cuántos lotes tiene una época.

# %%
EJEMPLOS_ENTRENAMIENTO = dataset["train"]
EJEMPLOS_VALIDACION = dataset["validation"]

lotes_por_epoca = len(EJEMPLOS_ENTRENAMIENTO) // BATCH_SIZE
sobrantes = len(EJEMPLOS_ENTRENAMIENTO) % BATCH_SIZE
if sobrantes > 0:
    lotes_por_epoca = lotes_por_epoca + 1

print("Ejemplos de entrenamiento:", len(EJEMPLOS_ENTRENAMIENTO))
print("Ejemplos de validación:", len(EJEMPLOS_VALIDACION))
print("Batch de entrenamiento:", BATCH_SIZE)
print("Lotes por época:", lotes_por_epoca)
if sobrantes == 0:
    print("Todos los lotes van completos")
else:
    print("Ejemplos en el último lote:", sobrantes)

# %% [markdown]
# Elegimos los 50 problemas de monitoreo con la semilla del proyecto y comprobamos que sean
# distintos y que ninguno esté en entrenamiento.

# %%
CANTIDAD_MONITOREO = 50

generador = random.Random(SEED)
indices_monitoreo = generador.sample(range(len(EJEMPLOS_VALIDACION)), CANTIDAD_MONITOREO)
indices_monitoreo = sorted(indices_monitoreo)

EJEMPLOS_MONITOREO = []
for indice in indices_monitoreo:
    EJEMPLOS_MONITOREO.append(EJEMPLOS_VALIDACION[indice])

ids_entrenamiento = []
for ejemplo in EJEMPLOS_ENTRENAMIENTO:
    ids_entrenamiento.append(ejemplo["task_id"])

ids_monitoreo = []
for ejemplo in EJEMPLOS_MONITOREO:
    ids_monitoreo.append(ejemplo["task_id"])

ids_en_entrenamiento = 0
for ident in ids_monitoreo:
    if ident in ids_entrenamiento:
        ids_en_entrenamiento = ids_en_entrenamiento + 1

print("Semilla:", SEED)
print("Problemas de validación para monitoreo:", len(EJEMPLOS_MONITOREO))
print("task_id distintos:", len(set(ids_monitoreo)))
print("task_id que también están en entrenamiento:", ids_en_entrenamiento)
print("task_id:", ids_monitoreo)

# %% [markdown]
# Calculamos cuántos tokens ve el modelo en una época para dimensionar el entrenamiento. El costo de
# generar es aparte: aquí solo contamos lo que se recorre hacia adelante y hacia atrás.

# %%
filas_presupuesto = []
for nombre, tokenizador in TOKENIZADORES:
    tokens_por_epoca = 0
    for ejemplo in EJEMPLOS_ENTRENAMIENTO:
        datos = tokenizar_ejemplo(ejemplo, tokenizador)
        tokens_por_epoca = tokens_por_epoca + len(datos["ids"])
    filas_presupuesto.append({
        "modelo": nombre,
        "tokens por época": tokens_por_epoca,
        "tokens en 3 épocas": tokens_por_epoca * 3,
        "tokens en 5 épocas": tokens_por_epoca * 5,
        "pasos en 5 épocas": lotes_por_epoca * 5,
    })

tabla_presupuesto = pd.DataFrame(filas_presupuesto)
print(tabla_presupuesto.to_string(index=False))

# %% [markdown]
# Comparamos los 50 de monitoreo con los 90 de validación: longitudes en tokens con Qwen y estructura
# del código de referencia. Si se parecen, el pass@1 de los 50 sirve para decidir la época.

# %%
largos_monitoreo = []
for ejemplo in EJEMPLOS_MONITOREO:
    datos = tokenizar_ejemplo(ejemplo, tokenizador_qwen)
    largos_monitoreo.append(len(datos["ids"]))

largos_validacion = []
for ejemplo in EJEMPLOS_VALIDACION:
    datos = tokenizar_ejemplo(ejemplo, tokenizador_qwen)
    largos_validacion.append(len(datos["ids"]))

print("Monitoreo (n = 50)")
print("  media:", round(promedio(largos_monitoreo), 1))
print("  mediana:", percentil(largos_monitoreo, 0.50))
print("  máximo:", max(largos_monitoreo))
print()
print("Validación (n = 90)")
print("  media:", round(promedio(largos_validacion), 1))
print("  mediana:", percentil(largos_validacion, 0.50))
print("  máximo:", max(largos_validacion))
print()
print("Estructura del código de referencia")
print("  monitoreo:", contar_estructura(EJEMPLOS_MONITOREO))
print("  validación:", contar_estructura(EJEMPLOS_VALIDACION))

# %% [markdown]
# ### Lo que nos deja la partición
#
# **1. Tres conjuntos con roles separados.** El dataset trae 374 problemas de entrenamiento, 90 de
# validación y 500 de prueba. `train` solo se usa para el finetuning; `validation` para la pérdida y
# el pass@1 de monitoreo; `test` queda intacto hasta la sección 7. La tabla de roles deja esa
# separación escrita en el notebook y no solo en la cabeza de quien lo corre.
#
# **2. Una época son 24 pasos.** Con batch de 16, los 374 ejemplos dan 23 lotes completos y uno de
# 6. Cinco épocas son 120 pasos de optimización, así que el bucle de entrenamiento es corto y se
# puede seguir época por época sin problema.
#
# **3. El entrenamiento no es el costo.** Una época recorre 33,984 tokens con Qwen y 40,201 con
# SmolLM2; cinco épocas llegan a 169,920 y 201,005. Son cifras pequeñas para una GPU de 24 GB. El
# costo real está en generar 500 soluciones por cada una de las cuatro configuraciones, que es lo que
# justifica la generación por lotes y el recorte del prompt.
#
# **4. Los 50 de monitoreo representan al conjunto.** Salieron 50 `task_id` distintos entre 511 y
# 600, ninguno de entrenamiento. En tokens son un poco más largos que los 90 (mediana 85 contra 77,
# media 100.8 contra 92.7), pero el máximo es el mismo (236). La estructura también se parece: 4 de
# 50 tienen más de una función (8 %) contra 6 de 90 (6.7 %), y 7 de 50 traen imports (14 %) contra
# 13 de 90 (14.4 %). Con esa muestra el pass@1 de monitoreo sirve para comparar épocas, no para
# sacar conclusiones finas.
#
# **5. Qué se decide con esto.** La época y los hiperparámetros se eligen con la pérdida de los 90 y
# el pass@1 de los 50; `test` no participa. Como cada acierto en 50 mueve el pass@1 dos puntos
# porcentuales, solo miramos diferencias de varios problemas y desempatamos por la pérdida, como
# estaba previsto. La lista de `test` se arma en la sección 7, cuando ya no hay nada que elegir.
