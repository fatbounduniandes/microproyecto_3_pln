# %% [markdown]
# ## 3. Carga del dataset y exploración
#
# Cargamos MBPP desde Hugging Face con la revisión fija de la sección anterior y revisamos qué
# trae cada problema, cuánto miden los enunciados y las soluciones y cuántos `assert` usa cada uno.
# El análisis se hace sobre `train` y `validation`; el split `test` solo se cuenta y se usa para
# comprobar que no haya problemas repetidos, porque queda reservado para la evaluación final.

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
# Medimos los enunciados y las soluciones de `train` y `validation` en caracteres. Si los dos splits
# se parecen, lo que aprendamos mirando `train` también aplica a `validation`.

# %%
def promedio(valores):
    return sum(valores) / len(valores)


def percentil(valores, fraccion):
    ordenados = sorted(valores)
    posicion = int(len(ordenados) * fraccion)
    return ordenados[posicion]


largos_texto_train = []
largos_codigo_train = []
palabras_train = []
for ejemplo in dataset["train"]:
    largos_texto_train.append(len(ejemplo["text"]))
    largos_codigo_train.append(len(ejemplo["code"]))
    palabras_train.append(len(ejemplo["text"].split()))

largos_texto_validation = []
largos_codigo_validation = []
for ejemplo in dataset["validation"]:
    largos_texto_validation.append(len(ejemplo["text"]))
    largos_codigo_validation.append(len(ejemplo["code"]))

print(f"{'split':<12}{'n':>5}{'texto_prom':>12}{'texto_p95':>11}{'codigo_prom':>13}{'codigo_p95':>12}")

for nombre in ["train", "validation"]:
    if nombre == "train":
        textos = largos_texto_train
        codigos = largos_codigo_train
    else:
        textos = largos_texto_validation
        codigos = largos_codigo_validation
    texto_promedio = round(promedio(textos), 1)
    texto_p95 = percentil(textos, 0.95)
    codigo_promedio = round(promedio(codigos), 1)
    codigo_p95 = percentil(codigos, 0.95)
    print(f"{nombre:<12}{len(textos):>5}{texto_promedio:>12}{texto_p95:>11}{codigo_promedio:>13}{codigo_p95:>12}")

# %% [markdown]
# Miramos con más detalle los largos de `train`, que es el split con el que se entrena. Todavía no
# usamos un tokenizador porque cada modelo tiene el suyo y eso lo vemos en la sección siguiente.

# %%
print("Enunciado en caracteres")
print("  mínimo:", min(largos_texto_train))
print("  promedio:", round(promedio(largos_texto_train), 1))
print("  mediana:", percentil(largos_texto_train, 0.50))
print("  p95:", percentil(largos_texto_train, 0.95))
print("  máximo:", max(largos_texto_train))
print()
print("Enunciado en palabras")
print("  mínimo:", min(palabras_train))
print("  promedio:", round(promedio(palabras_train), 1))
print("  mediana:", percentil(palabras_train, 0.50))
print("  p95:", percentil(palabras_train, 0.95))
print("  máximo:", max(palabras_train))
print()
print("Solución en caracteres")
print("  mínimo:", min(largos_codigo_train))
print("  promedio:", round(promedio(largos_codigo_train), 1))
print("  mediana:", percentil(largos_codigo_train, 0.50))
print("  p95:", percentil(largos_codigo_train, 0.95))
print("  máximo:", max(largos_codigo_train))

# %% [markdown]
# Comparamos los dos splits con diagramas de caja. La caja muestra la mediana y el rango donde está
# la mitad central de los datos, y los puntos sueltos son problemas atípicos.

# %%
plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.boxplot([largos_texto_train, largos_texto_validation])
plt.xticks([1, 2], ["train", "validation"])
plt.title("Enunciados")
plt.ylabel("Caracteres")

plt.subplot(1, 2, 2)
plt.boxplot([largos_codigo_train, largos_codigo_validation])
plt.xticks([1, 2], ["train", "validation"])
plt.title("Soluciones")
plt.ylabel("Caracteres")

plt.tight_layout()
plt.show()

# %% [markdown]
# Dibujamos la curva acumulada de las longitudes: para cada valor del eje x muestra qué porcentaje
# de los problemas mide eso o menos. Sirve para decidir cuánto contexto necesitamos.

# %%
def acumulada(valores):
    ordenados = sorted(valores)
    x = ordenados
    y = []
    for i in range(len(ordenados)):
        y.append((i + 1) / len(ordenados))
    return x, y


x_train, y_train = acumulada(largos_texto_train)
x_val, y_val = acumulada(largos_texto_validation)

plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.plot(x_train, y_train, label="train")
plt.plot(x_val, y_val, linestyle="--", label="validation")
plt.axvline(percentil(largos_texto_train, 0.95), linestyle=":", color="gray")
plt.title("Enunciados")
plt.xlabel("Caracteres")
plt.ylabel("Proporción acumulada")
plt.legend()

x_train, y_train = acumulada(largos_codigo_train)
x_val, y_val = acumulada(largos_codigo_validation)

plt.subplot(1, 2, 2)
plt.plot(x_train, y_train, label="train")
plt.plot(x_val, y_val, linestyle="--", label="validation")
plt.axvline(percentil(largos_codigo_train, 0.95), linestyle=":", color="gray")
plt.title("Soluciones")
plt.xlabel("Caracteres")
plt.legend()

plt.tight_layout()
plt.show()

print("p95 del enunciado en train:", percentil(largos_texto_train, 0.95), "caracteres")
print("p95 de la solución en train:", percentil(largos_codigo_train, 0.95), "caracteres")

# %% [markdown]
# Revisamos qué tan completas son las soluciones de referencia. Si varias traen imports, clases o
# funciones auxiliares, el modelo tiene que aprender a generarlas y no basta con el cuerpo de una
# sola función.

# %%
def contar_estructura(particion):
    conteos = {"una función": 0, "varias funciones": 0, "con imports": 0, "con clases": 0, "sin leer": 0}
    for ejemplo in particion:
        try:
            arbol = ast.parse(ejemplo["code"])
        except SyntaxError:
            conteos["sin leer"] = conteos["sin leer"] + 1
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

        if funciones == 1:
            conteos["una función"] = conteos["una función"] + 1
        if funciones > 1:
            conteos["varias funciones"] = conteos["varias funciones"] + 1
        if imports > 0:
            conteos["con imports"] = conteos["con imports"] + 1
        if clases > 0:
            conteos["con clases"] = conteos["con clases"] + 1
    return conteos


conteos_train = contar_estructura(dataset["train"])
conteos_validation = contar_estructura(dataset["validation"])

print("train:", conteos_train)
print("validation:", conteos_validation)

categorias = ["una función", "varias funciones", "con imports", "con clases"]

porcentaje_train = []
porcentaje_validation = []
for categoria in categorias:
    porcentaje_train.append(100 * conteos_train[categoria] / len(dataset["train"]))
    porcentaje_validation.append(100 * conteos_validation[categoria] / len(dataset["validation"]))

plt.figure(figsize=(8, 3.5))
plt.barh(categorias, porcentaje_train, label="train")
plt.barh(categorias, porcentaje_validation, label="validation", alpha=0.6)
plt.xlabel("Porcentaje de problemas")
plt.title("Estructura del código de referencia")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Buscamos los casos extremos de `train`: el enunciado más largo, la solución más larga y el único
# problema que trae `test_setup_code`. Los extremos son los que pueden quedar truncados o romper la
# ejecución de los `assert`.

# %%
indice_texto = largos_texto_train.index(max(largos_texto_train))
indice_codigo = largos_codigo_train.index(max(largos_codigo_train))

print("Enunciado más largo de train")
print("  task_id:", dataset["train"][indice_texto]["task_id"])
print("  caracteres:", largos_texto_train[indice_texto])
print(" ", dataset["train"][indice_texto]["text"])
print()
print("Solución más larga de train")
print("  task_id:", dataset["train"][indice_codigo]["task_id"])
print("  caracteres:", largos_codigo_train[indice_codigo])
print("  líneas:", len(dataset["train"][indice_codigo]["code"].splitlines()))

indice_setup = None
for i in range(len(dataset["train"])):
    if dataset["train"][i]["test_setup_code"].strip() != "":
        indice_setup = i
        break

print()
if indice_setup is None:
    print("Ningún problema de train trae test_setup_code")
else:
    print("Problema con test_setup_code")
    print("  task_id:", dataset["train"][indice_setup]["task_id"])
    print(dataset["train"][indice_setup]["test_setup_code"])

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
# Mostramos un problema de `validation` para confirmar que tiene el mismo formato que `train`.

# %%
ejemplo_validacion = dataset["validation"][0]
print("task_id:", ejemplo_validacion["task_id"])
print()
print(ejemplo_validacion["text"])
print()
print(ejemplo_validacion["code"])
print()
print("Primer caso de prueba:", ejemplo_validacion["test_list"][0])

# %% [markdown]
# Revisamos cuántos `assert` trae cada problema y en cuántos aparece `test_setup_code`, porque eso
# cambia la forma de ejecutar la verificación: el código de preparación va después de la solución
# generada y antes de los `assert`.

# %%
for nombre in ["train", "validation"]:
    conteo_asserts = {}
    con_setup = 0
    for ejemplo in dataset[nombre]:
        cuantos = len(ejemplo["test_list"])
        if cuantos in conteo_asserts:
            conteo_asserts[cuantos] = conteo_asserts[cuantos] + 1
        else:
            conteo_asserts[cuantos] = 1
        if ejemplo["test_setup_code"].strip() != "":
            con_setup = con_setup + 1

    print(nombre)
    for cuantos in sorted(conteo_asserts):
        if cuantos == 1:
            etiqueta = "assert"
        else:
            etiqueta = "asserts"
        if conteo_asserts[cuantos] == 1:
            print(" ", cuantos, etiqueta, "->", conteo_asserts[cuantos], "problema")
        else:
            print(" ", cuantos, etiqueta, "->", conteo_asserts[cuantos], "problemas")
    print("  problemas con test_setup_code:", con_setup)
    print()

# %% [markdown]
# ### Lo que nos deja la exploración
#
# `train` va del `task_id` 601 al 974 y `validation` del 511 al 600, sin ningún identificador
# repetido entre particiones. Eso descarta que el modelo haya visto durante el entrenamiento algún
# problema de los que se usan para monitorear, y deja `test` (del 11 al 510) limpio para el pass@1
# final.
#
# Los dos splits se parecen: el enunciado promedia 78 caracteres (unas 14 palabras) y la solución
# 178, con un p95 de 113 y 412 caracteres. El 95 % de las soluciones cabe en menos de 412
# caracteres, así que un límite de 512 tokens sobra, y la solución más larga ronda las 900 letras.
#
# Casi todas las soluciones son una sola función, pero 87 de las 374 traen imports y unas pocas
# agregan clases o funciones auxiliares. Por eso el target es el código completo y no solo el
# cuerpo: si entrenáramos solo con el cuerpo, el modelo nunca aprendería a escribir esas piezas.
#
# Los problemas de MBPP no dicen cómo se llama la función, pero los `assert` la llaman por su
# nombre. Por eso el prompt lleva el enunciado y el nombre con la firma de la función. La solución
# de referencia y los casos de prueba se quedan fuera del prompt, porque si aparecen el modelo
# podría copiarse o memorizar los `assert` en vez de resolver el problema.
#
# Además, los 374 problemas de `train` traen exactamente 3 `assert` y solo uno usa
# `test_setup_code`, así que la verificación es uniforme y el caso especial es raro pero hay que
# manejarlo: ese código de preparación se ejecuta después de la solución generada y antes de los
# `assert`.
