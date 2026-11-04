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
# Revisamos que ningún `task_id` se repita entre particiones. Si `train` y `test` compartieran
# problemas, el pass@1 saldría inflado porque el modelo habría entrenado con ellos.

# %%
ids_train = []
for ejemplo in dataset["train"]:
    ids_train.append(ejemplo["task_id"])

ids_validation = []
for ejemplo in dataset["validation"]:
    ids_validation.append(ejemplo["task_id"])

ids_test = []
for ejemplo in dataset["test"]:
    ids_test.append(ejemplo["task_id"])

repetidos_train_validation = 0
for ident in ids_train:
    if ident in ids_validation:
        repetidos_train_validation = repetidos_train_validation + 1

repetidos_train_test = 0
for ident in ids_train:
    if ident in ids_test:
        repetidos_train_test = repetidos_train_test + 1

repetidos_validation_test = 0
for ident in ids_validation:
    if ident in ids_test:
        repetidos_validation_test = repetidos_validation_test + 1


def imprimir_split(nombre, ids):
    if len(ids) == 1:
        print(f"{nombre:<11}{len(ids):>4} problema, del {min(ids)} al {max(ids)}")
    else:
        print(f"{nombre:<11}{len(ids):>4} problemas, del {min(ids)} al {max(ids)}")


imprimir_split("train", ids_train)
imprimir_split("validation", ids_validation)
imprimir_split("test", ids_test)
print()
print("Repetidos entre train y validation:", repetidos_train_validation)
print("Repetidos entre train y test:", repetidos_train_test)
print("Repetidos entre validation y test:", repetidos_validation_test)

# %% [markdown]
# Comparamos los tres splits con las mismas medidas. Si `train`, `validation` y `test` se parecen en
# tamaño de enunciado y de solución, los resultados de un split dicen algo sobre los otros.

# %%
def promedio(valores):
    return sum(valores) / len(valores)


def percentil(valores, fraccion):
    ordenados = sorted(valores)
    posicion = int(len(ordenados) * fraccion)
    return ordenados[posicion]


print(f"{'split':<12}{'n':>5}{'texto_prom':>12}{'texto_p95':>11}{'codigo_prom':>13}{'codigo_p95':>12}")

for nombre in ["train", "validation", "test"]:
    largos_texto_split = []
    largos_codigo_split = []
    for ejemplo in dataset[nombre]:
        largos_texto_split.append(len(ejemplo["text"]))
        largos_codigo_split.append(len(ejemplo["code"]))
    texto_promedio = round(promedio(largos_texto_split), 1)
    texto_p95 = percentil(largos_texto_split, 0.95)
    codigo_promedio = round(promedio(largos_codigo_split), 1)
    codigo_p95 = percentil(largos_codigo_split, 0.95)
    print(f"{nombre:<12}{len(dataset[nombre]):>5}{texto_promedio:>12}{texto_p95:>11}{codigo_promedio:>13}{codigo_p95:>12}")

# %% [markdown]
# Miramos con más detalle los largos de `train`. Todavía no usamos un tokenizador porque cada modelo
# tiene el suyo y eso lo vemos en la sección siguiente; aquí solo queremos una idea del tamaño de
# los textos.

# %%
largos_texto = []
largos_codigo = []
palabras_texto = []

for ejemplo in dataset["train"]:
    largos_texto.append(len(ejemplo["text"]))
    largos_codigo.append(len(ejemplo["code"]))
    palabras_texto.append(len(ejemplo["text"].split()))


def resumen_largos(nombre, valores):
    print(nombre)
    print("  mínimo:", min(valores))
    print("  promedio:", round(promedio(valores), 1))
    print("  mediana:", percentil(valores, 0.50))
    print("  p95:", percentil(valores, 0.95))
    print("  máximo:", max(valores))


resumen_largos("Enunciado en caracteres", largos_texto)
print()
resumen_largos("Enunciado en palabras", palabras_texto)
print()
resumen_largos("Solución en caracteres", largos_codigo)

# %% [markdown]
# Dibujamos esas longitudes para ver si hay valores extremos que nos obliguen a subir el contexto.

# %%
plt.figure(figsize=(9, 3))

plt.subplot(1, 2, 1)
plt.hist(largos_texto, bins=20)
plt.title("Enunciados en train")
plt.xlabel("Caracteres")
plt.ylabel("Problemas")

plt.subplot(1, 2, 2)
plt.hist(largos_codigo, bins=20)
plt.title("Soluciones en train")
plt.xlabel("Caracteres")

plt.tight_layout()
plt.show()

# %% [markdown]
# Revisamos qué tan completas son las soluciones de referencia. Si varias traen imports, clases o
# funciones auxiliares, el modelo tiene que aprender a generarlas y no basta con el cuerpo de una
# sola función.

# %%
con_import = 0
con_clase = 0
con_varias_funciones = 0
con_una_funcion = 0
sin_parsear = 0

for ejemplo in dataset["train"]:
    try:
        arbol = ast.parse(ejemplo["code"])
    except SyntaxError:
        sin_parsear = sin_parsear + 1
        continue

    imports = 0
    clases = 0
    funciones = 0
    for nodo in arbol.body:
        if isinstance(nodo, ast.Import):
            imports = imports + 1
        if isinstance(nodo, ast.ImportFrom):
            imports = imports + 1
        if isinstance(nodo, ast.ClassDef):
            clases = clases + 1
        if isinstance(nodo, ast.FunctionDef):
            funciones = funciones + 1

    if imports > 0:
        con_import = con_import + 1
    if clases > 0:
        con_clase = con_clase + 1
    if funciones > 1:
        con_varias_funciones = con_varias_funciones + 1
    if funciones == 1:
        con_una_funcion = con_una_funcion + 1

print("Problemas de train:", len(dataset["train"]))
print("Soluciones que no se pudieron leer:", sin_parsear)
print("Con imports:", con_import)
print("Con clases:", con_clase)
print("Con más de una función:", con_varias_funciones)
print("Con una sola función:", con_una_funcion)

# %% [markdown]
# Revisamos la columna `challenge_test_list`, que trae pruebas adicionales, y miramos un ejemplo del
# split `prompt`, que no usamos ni para entrenar ni para evaluar.

# %%
con_challenge = 0
for ejemplo in dataset["train"]:
    if len(ejemplo["challenge_test_list"]) > 0:
        con_challenge = con_challenge + 1

print("Problemas de train con challenge_test_list:", con_challenge, "de", len(dataset["train"]))

ejemplo_prompt = dataset["prompt"][0]
print()
print("Un ejemplo del split prompt:")
print("task_id:", ejemplo_prompt["task_id"])
print()
print(ejemplo_prompt["text"])
print()
print(ejemplo_prompt["code"])

# %% [markdown]
# Mostramos un problema de `validation` y otro de `test` para confirmar que los tres splits tienen
# el mismo formato. Solo miramos la forma; estas particiones no se tocan hasta evaluar.

# %%
for nombre in ["validation", "test"]:
    ejemplo = dataset[nombre][0]
    print("=" * 70)
    print("split:", nombre, "| task_id:", ejemplo["task_id"])
    print()
    print(ejemplo["text"])
    print()
    print(ejemplo["code"])
    print()
    print("Primer caso de prueba:", ejemplo["test_list"][0])
    print()

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
    if cuantos == 1:
        etiqueta = "assert"
    else:
        etiqueta = "asserts"
    if conteo_asserts[cuantos] == 1:
        print(" ", cuantos, etiqueta, "->", conteo_asserts[cuantos], "problema")
    else:
        print(" ", cuantos, etiqueta, "->", conteo_asserts[cuantos], "problemas")

print()
print("Problemas con test_setup_code:", con_setup, "de", len(dataset["train"]))

# %% [markdown]
# ### Lo que nos deja la exploración
#
# Los tres splits vienen bien separados: `test` ocupa los `task_id` más bajos y `train` los más
# altos, sin problemas repetidos, así que la evaluación no se contamina con lo que el modelo ya vio.
# Los splits además se parecen en tamaño de enunciado y de solución, lo que hace comparable el
# pass@1 de `train` con el de `test`.
#
# Los enunciados son cortos y las soluciones también, así que un límite de 512 tokens alcanza de
# sobra. Casi todas las soluciones son una sola función, pero varias traen imports y alguna trae
# clases o funciones auxiliares: por eso el target es el código completo y no solo el cuerpo.
#
# Los problemas de MBPP no dicen cómo se llama la función, pero los `assert` la llaman por su
# nombre. Por eso el prompt lleva el enunciado y el nombre con la firma de la función. La solución
# de referencia y los casos de prueba se quedan fuera del prompt, porque si aparecen el modelo
# podría copiarse o memorizar los `assert` en vez de resolver el problema.
