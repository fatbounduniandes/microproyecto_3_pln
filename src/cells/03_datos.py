# %% [markdown]
# ## 3. Carga del dataset y exploración
#
# Cargamos MBPP desde Hugging Face con la revisión fija de la sección anterior y revisamos qué
# trae cada problema, cuánto miden los enunciados y las soluciones y cuántos `assert` usa cada uno.
# Con eso decidimos cómo armar el prompt y cuánto contexto necesitamos.

# %% [markdown]
# Cargamos el dataset y revisamos los splits y las columnas que trae.

# %%
dataset = load_dataset(DATASET_ID, "full", revision=DATASET_REVISION)

print("Splits del dataset:")
for nombre in dataset:
    cuantos = len(dataset[nombre])
    if cuantos == 1:
        print(" ", nombre, "->", cuantos, "ejemplo")
    else:
        print(" ", nombre, "->", cuantos, "ejemplos")

print()
print("Columnas:", dataset["train"].column_names)

# %% [markdown]
# Comparamos los tamaños con lo que dice el enunciado: 500 problemas de `test` y el resto repartido
# entre `train` y `validation`.

# %%
TAMANOS_ESPERADOS = {"train": 374, "validation": 90, "test": 500}

for nombre in TAMANOS_ESPERADOS:
    esperado = TAMANOS_ESPERADOS[nombre]
    obtenido = len(dataset[nombre])
    if obtenido == esperado:
        print(nombre, "OK:", obtenido, "ejemplos")
    else:
        print(nombre, "distinto: se esperaban", esperado, "y hay", obtenido)

print()
print("Total de problemas para evaluar:", len(dataset["test"]))

# %% [markdown]
# Mostramos dos problemas de `train` completos: el enunciado, la solución de referencia, los casos
# de prueba y el código de preparación que traen algunos ejercicios.

# %%
for indice in [0, 1]:
    ejemplo = dataset["train"][indice]
    print("=" * 70)
    print("task_id:", ejemplo["task_id"])
    print()
    print("Enunciado:")
    print(ejemplo["text"])
    print()
    print("Solución de referencia:")
    print(ejemplo["code"])
    print()
    print("Casos de prueba:")
    for linea in ejemplo["test_list"]:
        print(" ", linea)
    print()
    print("Código de preparación:")
    if ejemplo["test_setup_code"].strip() == "":
        print("  (vacío)")
    else:
        print(ejemplo["test_setup_code"])
    print()

# %% [markdown]
# Medimos los enunciados y las soluciones en caracteres y en palabras. Todavía no usamos un
# tokenizador porque cada modelo tiene el suyo y eso lo vemos en la sección siguiente; aquí solo
# queremos una idea del tamaño de los textos.

# %%
largos_texto = []
largos_codigo = []
palabras_texto = []

for ejemplo in dataset["train"]:
    largos_texto.append(len(ejemplo["text"]))
    largos_codigo.append(len(ejemplo["code"]))
    palabras_texto.append(len(ejemplo["text"].split()))

largos_texto.sort()
largos_codigo.sort()
palabras_texto.sort()


def resumen_largos(nombre, valores):
    total = len(valores)
    promedio = sum(valores) / total
    print(nombre)
    print("  mínimo:", valores[0])
    print("  promedio:", round(promedio, 1))
    print("  mediana:", valores[int(total * 0.50)])
    print("  p95:", valores[int(total * 0.95)])
    print("  máximo:", valores[-1])


resumen_largos("Enunciado en caracteres", largos_texto)
print()
resumen_largos("Enunciado en palabras", palabras_texto)
print()
resumen_largos("Solución en caracteres", largos_codigo)

# %% [markdown]
# Revisamos cuántos `assert` trae cada problema y en cuántos aparece `test_setup_code`, porque eso
# cambia la forma de ejecutar la verificación: el código de preparación va después de la solución
# generada y antes de los `assert`.

# %%
conteo_asserts = {}
con_setup = 0

for ejemplo in dataset["train"]:
    cuantos = len(ejemplo["test_list"])
    if cuantos in conteo_asserts:
        conteo_asserts[cuantos] = conteo_asserts[cuantos] + 1
    else:
        conteo_asserts[cuantos] = 1
    if ejemplo["test_setup_code"].strip() != "":
        con_setup = con_setup + 1

print("Assert por problema:")
for cuantos in sorted(conteo_asserts):
    if conteo_asserts[cuantos] == 1:
        print(" ", cuantos, "asserts ->", conteo_asserts[cuantos], "problema")
    else:
        print(" ", cuantos, "asserts ->", conteo_asserts[cuantos], "problemas")

print()
print("Problemas con test_setup_code:", con_setup, "de", len(dataset["train"]))

# %% [markdown]
# ### Lo que nos deja la exploración
#
# Los enunciados son cortos y las soluciones también, así que un límite de 512 tokens alcanza de
# sobra. Los problemas de MBPP no dicen cómo se llama la función, pero los `assert` la llaman por
# su nombre, y algunos ejercicios necesitan clases o helpers que están en la solución de referencia.
#
# Por eso el prompt lleva el enunciado y el nombre con la firma de la función, y el modelo aprende a
# generar el código completo: imports, helpers y la función pedida. La solución de referencia y los
# casos de prueba se quedan fuera del prompt, porque si aparecen el modelo podría copiarse o
# memorizar los `assert` en vez de resolver el problema.
