# %% [markdown]
# ## 3. Carga del dataset y exploración
#
# Cargamos MBPP desde Hugging Face con la revisión fija de la sección anterior y lo miramos a fondo:
# qué trae cada problema, cuánto miden los enunciados y las soluciones, cómo están escritas, qué
# palabras se repiten y cómo son los `assert`. El análisis se hace sobre `train` y `validation`; el
# split `test` solo se cuenta y se usa para comprobar que no haya problemas repetidos, porque queda
# reservado para la evaluación final.

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
# Sacamos la tabla descriptiva de `train` y de `validation` con cuartiles, desviación estándar y
# extremos. Es la misma idea que un `describe()` de pandas: resumir cada variable en una sola vista
# para comparar los dos splits de un vistazo.

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
lineas_train = []
for ejemplo in dataset["train"]:
    largos_texto_train.append(len(ejemplo["text"]))
    largos_codigo_train.append(len(ejemplo["code"]))
    palabras_train.append(len(ejemplo["text"].split()))
    lineas_train.append(len(ejemplo["code"].splitlines()))

largos_texto_validation = []
largos_codigo_validation = []
palabras_validation = []
lineas_validation = []
for ejemplo in dataset["validation"]:
    largos_texto_validation.append(len(ejemplo["text"]))
    largos_codigo_validation.append(len(ejemplo["code"]))
    palabras_validation.append(len(ejemplo["text"].split()))
    lineas_validation.append(len(ejemplo["code"].splitlines()))

tabla_train = pd.DataFrame({
    "enunciado_car": largos_texto_train,
    "enunciado_pal": palabras_train,
    "solucion_car": largos_codigo_train,
    "solucion_lin": lineas_train,
})

tabla_validation = pd.DataFrame({
    "enunciado_car": largos_texto_validation,
    "enunciado_pal": palabras_validation,
    "solucion_car": largos_codigo_validation,
    "solucion_lin": lineas_validation,
})

print("train (n =", len(tabla_train), ")")
print(tabla_train.describe().round(1).to_string())
print()
print("validation (n =", len(tabla_validation), ")")
print(tabla_validation.describe().round(1).to_string())

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
# Miramos el vocabulario de los enunciados: cuántas palabras distintas usan, cuáles se repiten y
# cuántos problemas empiezan con la misma fórmula. También revisamos si el nombre de la función
# aparece en el texto, porque de eso depende el prompt.

# %%
PALABRAS_VACIAS = [
    "a", "an", "the", "to", "of", "and", "in", "for", "is", "that", "with", "from",
    "as", "on", "by", "it", "this", "be", "are", "given", "write", "function", "python",
]

conteo_palabras = {}
todas_las_palabras = set()
for ejemplo in dataset["train"]:
    for palabra in re.findall(r"[a-z]+", ejemplo["text"].lower()):
        todas_las_palabras.add(palabra)
        if palabra in PALABRAS_VACIAS:
            continue
        if palabra in conteo_palabras:
            conteo_palabras[palabra] = conteo_palabras[palabra] + 1
        else:
            conteo_palabras[palabra] = 1

palabras_ordenadas = sorted(conteo_palabras, key=conteo_palabras.get, reverse=True)
top_palabras = palabras_ordenadas[:15]

empieza_to = 0
empieza_that = 0
empieza_python = 0
for ejemplo in dataset["train"]:
    texto = ejemplo["text"].lower().strip()
    if texto.startswith("write a function to"):
        empieza_to = empieza_to + 1
    if texto.startswith("write a function that"):
        empieza_that = empieza_that + 1
    if texto.startswith("write a python function"):
        empieza_python = empieza_python + 1

nombres_en_texto = 0
for ejemplo in dataset["train"]:
    nombre = ""
    for linea in ejemplo["test_list"]:
        arbol = ast.parse(linea)
        nodo = arbol.body[0]
        if isinstance(nodo, ast.Assert) and isinstance(nodo.test, ast.Compare):
            izquierda = nodo.test.left
            if isinstance(izquierda, ast.Call) and isinstance(izquierda.func, ast.Name):
                nombre = izquierda.func.id
                break
    if nombre != "" and nombre.lower() in ejemplo["text"].lower():
        nombres_en_texto = nombres_en_texto + 1

print("Palabras distintas en total:", len(todas_las_palabras))
print("Palabras distintas sin contar las vacías:", len(conteo_palabras))
print()
print("Palabras con contenido más frecuentes:")
for palabra in top_palabras:
    print(" ", palabra, "->", conteo_palabras[palabra])
print()
print("Enunciados que empiezan con 'write a function to':", empieza_to)
print("Enunciados que empiezan con 'write a function that':", empieza_that)
print("Enunciados que empiezan con 'write a python function':", empieza_python)
print("Enunciados que mencionan el nombre de la función:", nombres_en_texto, "de", len(dataset["train"]))

# %% [markdown]
# Dibujamos las palabras con contenido más frecuentes. Las palabras vacías y las de la plantilla
# (“write”, “function”, “python”) quedaron fuera del conteo para que se vea el tema de cada
# problema y no el formato de la instrucción.

# %%
etiquetas = []
valores = []
for palabra in top_palabras:
    etiquetas.append(palabra)
    valores.append(conteo_palabras[palabra])

plt.figure(figsize=(7, 4))
plt.barh(etiquetas, valores)
plt.gca().invert_yaxis()
plt.xlabel("Apariciones en train")
plt.title("Palabras con contenido más frecuentes")
plt.tight_layout()
plt.show()

# %% [markdown]
# Medimos cuántas líneas tiene cada solución de `train`. Es otra forma de ver el tamaño del target:
# una solución de una línea y otra de treinta no cuestan lo mismo de generar.

# %%
print("Líneas por solución en train")
print("  mínimo:", min(lineas_train))
print("  promedio:", round(promedio(lineas_train), 1))
print("  mediana:", percentil(lineas_train, 0.50))
print("  p95:", percentil(lineas_train, 0.95))
print("  máximo:", max(lineas_train))

plt.figure(figsize=(7, 3))
plt.hist(lineas_train, bins=20)
plt.title("Líneas por solución en train")
plt.xlabel("Líneas")
plt.ylabel("Problemas")
plt.tight_layout()
plt.show()

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
# Contamos qué construcciones de Python aparecen en las soluciones: ciclos, condicionales, clases y
# recursión. Es lo que el modelo tiene que aprender a escribir para resolver los problemas.

# %%
con_for = 0
con_while = 0
con_if = 0
con_recursion = 0
con_clase = 0

for ejemplo in dataset["train"]:
    arbol = ast.parse(ejemplo["code"])

    nombres_funciones = []
    for nodo in arbol.body:
        if isinstance(nodo, ast.FunctionDef):
            nombres_funciones.append(nodo.name)

    usa_for = False
    usa_while = False
    usa_if = False
    usa_recursion = False
    usa_clase = False

    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.For):
            usa_for = True
        if isinstance(nodo, ast.While):
            usa_while = True
        if isinstance(nodo, ast.If):
            usa_if = True
        if isinstance(nodo, ast.ClassDef):
            usa_clase = True
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name):
            if nodo.func.id in nombres_funciones:
                usa_recursion = True

    if usa_for:
        con_for = con_for + 1
    if usa_while:
        con_while = con_while + 1
    if usa_if:
        con_if = con_if + 1
    if usa_recursion:
        con_recursion = con_recursion + 1
    if usa_clase:
        con_clase = con_clase + 1

print("Soluciones de train que usan:")
print("  for:", con_for)
print("  while:", con_while)
print("  if:", con_if)
print("  recursión:", con_recursion)
print("  clases:", con_clase)

etiquetas = ["for", "while", "if", "recursión", "clases"]
valores = [con_for, con_while, con_if, con_recursion, con_clase]

plt.figure(figsize=(8, 3))
plt.barh(etiquetas, valores)
plt.gca().invert_yaxis()
plt.xlabel("Problemas de train")
plt.title("Construcciones usadas en las soluciones")
plt.tight_layout()
plt.show()

# %% [markdown]
# Revisamos la calidad del texto: si hay soluciones con tabs en vez de espacios, si alguna imprime o
# pide datos, si hay soluciones de una sola línea y si se repite algún enunciado o algún código. Al
# final comparamos las distribuciones de `train` y `validation` con una prueba de
# Kolmogorov-Smirnov.

# %%
con_tabs = 0
con_print = 0
con_input = 0
una_linea = 0
for ejemplo in dataset["train"]:
    if "\t" in ejemplo["code"]:
        con_tabs = con_tabs + 1
    if "print(" in ejemplo["code"]:
        con_print = con_print + 1
    if "input(" in ejemplo["code"]:
        con_input = con_input + 1
    if len(ejemplo["code"].splitlines()) == 1:
        una_linea = una_linea + 1


def contar_duplicados(campo, normalizar):
    vistos = {}
    for nombre in ["train", "validation"]:
        for ejemplo in dataset[nombre]:
            valor = ejemplo[campo]
            if normalizar:
                valor = " ".join(valor.lower().split())
            clave = hashlib.md5(valor.encode()).hexdigest()
            if clave in vistos:
                vistos[clave] = vistos[clave] + 1
            else:
                vistos[clave] = 1
    repetidos = 0
    for clave in vistos:
        if vistos[clave] > 1:
            repetidos = repetidos + 1
    return repetidos


duplicados_texto_exactos = contar_duplicados("text", False)
duplicados_texto_normalizados = contar_duplicados("text", True)
duplicados_codigo_exactos = contar_duplicados("code", False)
duplicados_codigo_normalizados = contar_duplicados("code", True)

ks_texto = ks_2samp(largos_texto_train, largos_texto_validation)
ks_codigo = ks_2samp(largos_codigo_train, largos_codigo_validation)

print("Soluciones de train con tabs:", con_tabs, "de", len(dataset["train"]))
print("Soluciones de train con print(:", con_print)
print("Soluciones de train con input(:", con_input)
print("Soluciones de train de una sola línea:", una_linea)
print()
print("Duplicados entre train y validation")
print("  enunciados exactos:", duplicados_texto_exactos)
print("  enunciados normalizados:", duplicados_texto_normalizados)
print("  códigos exactos:", duplicados_codigo_exactos)
print("  códigos normalizados:", duplicados_codigo_normalizados)
print()
print("Kolmogorov-Smirnov entre train y validation")
print("  enunciado: estadístico", round(ks_texto.statistic, 3), "p-valor", round(ks_texto.pvalue, 3))
print("  solución: estadístico", round(ks_codigo.statistic, 3), "p-valor", round(ks_codigo.pvalue, 3))

# %% [markdown]
# Analizamos los `assert`: cuántos trae cada problema y qué tipo de valor esperan. Si comparan
# listas o diccionarios, la verificación tiene que ejecutar el código de verdad y no basta con
# comparar texto.

# %%
total_asserts = 0
min_asserts = 100
max_asserts = 0
con_setup_total = 0
tipos_valor = {}

for nombre in ["train", "validation"]:
    for ejemplo in dataset[nombre]:
        cuantos = len(ejemplo["test_list"])
        if cuantos < min_asserts:
            min_asserts = cuantos
        if cuantos > max_asserts:
            max_asserts = cuantos
        if ejemplo["test_setup_code"].strip() != "":
            con_setup_total = con_setup_total + 1
        for linea in ejemplo["test_list"]:
            total_asserts = total_asserts + 1
            arbol = ast.parse(linea)
            nodo = arbol.body[0]
            if isinstance(nodo, ast.Assert):
                comparacion = nodo.test
                if isinstance(comparacion, ast.Compare):
                    valor = comparacion.comparators[0]
                else:
                    valor = comparacion
                if isinstance(valor, ast.Constant):
                    tipo = "constantes"
                elif isinstance(valor, ast.List):
                    tipo = "listas"
                elif isinstance(valor, ast.Tuple):
                    tipo = "tuplas"
                elif isinstance(valor, ast.Dict):
                    tipo = "diccionarios"
                elif isinstance(valor, ast.Set):
                    tipo = "conjuntos"
                elif isinstance(valor, ast.UnaryOp):
                    tipo = "valores negativos"
                else:
                    tipo = "otros"
                if tipo in tipos_valor:
                    tipos_valor[tipo] = tipos_valor[tipo] + 1
                else:
                    tipos_valor[tipo] = 1

tipos_ordenados = sorted(tipos_valor, key=tipos_valor.get, reverse=True)

print("Assert en train y validation:", total_asserts)
print("Assert por problema: mínimo", min_asserts, "máximo", max_asserts)
print("Problemas con test_setup_code:", con_setup_total)
print()
print("Tipo de valor esperado en assert:")
for tipo in tipos_ordenados:
    print(" ", tipo, "->", tipos_valor[tipo])

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
print("  líneas:", lineas_train[indice_codigo])

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
# Con todos los números anteriores armamos las conclusiones de la exploración. El texto se genera
# con los valores calculados, así que siempre coincide con la ejecución.

# %%
n_train = len(dataset["train"])
n_validation = len(dataset["validation"])
n_test = len(dataset["test"])
n_prompt = len(dataset["prompt"])
n_total = n_train + n_validation + n_test + n_prompt

porcentaje_nombres = round(100 * nombres_en_texto / n_train, 1)
porcentaje_tabs = round(100 * con_tabs / n_train, 1)
resto_enunciados = n_train - empieza_to - empieza_that - empieza_python

mediana_codigo = percentil(largos_codigo_train, 0.50)
ratio_codigo = round(max(largos_codigo_train) / mediana_codigo, 1)
puntos_por_problema = round(100 / n_validation, 1)


def frase_verbos(cantidad, singular, plural):
    if cantidad == 1:
        return str(cantidad) + " " + singular
    return str(cantidad) + " " + plural


trozos_top = []
for palabra in top_palabras[:5]:
    trozos_top.append(palabra + " (" + str(conteo_palabras[palabra]) + ")")
lista_top = ", ".join(trozos_top)

SINGULARES = {
    "constantes": "constante",
    "listas": "lista",
    "tuplas": "tupla",
    "diccionarios": "diccionario",
    "conjuntos": "conjunto",
    "valores negativos": "valor negativo",
    "otros": "otro",
}

trozos_tipos = []
for tipo in tipos_ordenados:
    cantidad = tipos_valor[tipo]
    if cantidad == 1:
        trozos_tipos.append("1 " + SINGULARES[tipo])
    else:
        trozos_tipos.append(str(cantidad) + " " + tipo)
detalle_tipos = ", ".join(trozos_tipos)

frase_resto = ""
if resto_enunciados > 0:
    frase_resto = f" Los {resto_enunciados} restantes son variantes menores, como un título en mayúsculas."

if ks_texto.pvalue < 0.05:
    frase_ks_texto = "sí encuentra una diferencia"
else:
    frase_ks_texto = "no encuentra una diferencia"

if ks_codigo.pvalue < 0.05:
    frase_ks_codigo = "sí encuentra una diferencia"
else:
    frase_ks_codigo = "no encuentra una diferencia"

if ks_texto.pvalue < 0.05 or ks_codigo.pvalue < 0.05:
    frase_potencia = (
        f"Cuando el p-valor es bajo, la diferencia puede ser real o venir de una muestra pequeña: "
        f"con {n_validation} ejemplos la prueba tiene poca potencia y no conviene sobreinterpretar."
    )
else:
    frase_potencia = (
        f"Con {n_validation} ejemplos la prueba tiene poca potencia: no encontrar diferencias no "
        f"prueba que los splits sean idénticos, solo que no hay evidencia suficiente para distinguirlos."
    )

if duplicados_texto_exactos == 0 and duplicados_codigo_exactos == 0 and duplicados_texto_normalizados == 0 and duplicados_codigo_normalizados == 0:
    frase_duplicados = "tampoco hay enunciados ni soluciones duplicadas, ni exactas ni normalizadas"
    frase_contenido = "La separación se sostiene también a nivel de contenido, no solo de identificadores."
else:
    frase_duplicados = "pero aparecen duplicados de contenido que hay que revisar antes de entrenar"
    frase_contenido = "La separación por identificadores no basta: hay que revisar esos duplicados antes de entrenar."

if MAX_NEW_TOKENS >= 512:
    frase_limite = "deja margen de sobra y el peor caso se generaría completo"
else:
    frase_limite = "puede truncar las soluciones del extremo"

if min_asserts == max_asserts:
    frase_asserts = f"todos los problemas traen exactamente {min_asserts}"
else:
    frase_asserts = f"cada problema trae entre {min_asserts} y {max_asserts}"

if con_tabs == 1:
    frase_tabs = f"{con_tabs} solución ({porcentaje_tabs} %) usa tabs"
else:
    frase_tabs = f"{con_tabs} soluciones ({porcentaje_tabs} %) usan tabs"

if una_linea == 0:
    frase_linea = "ninguna cabe en una sola línea"
else:
    frase_linea = f"{una_linea} caben en una sola línea"

if con_print == 0 and con_input == 0:
    frase_salida = "no hay ninguna con `print(` ni con `input(`"
else:
    frase_salida = f"hay {con_print} con `print(` y {con_input} con `input(`"

if con_setup_total == 1:
    frase_setup = "el único problema con `test_setup_code`"
else:
    frase_setup = f"los {con_setup_total} problemas con `test_setup_code`"

frase_una = frase_verbos(conteos_train["una función"], "es una sola función", "son una sola función")
frase_varias = frase_verbos(conteos_train["varias funciones"], "tiene más de una", "tienen más de una")
frase_imports = frase_verbos(conteos_train["con imports"], "trae imports", "traen imports")
frase_clases = frase_verbos(conteos_train["con clases"], "declara clases", "declaran clases")
frase_recursion = frase_verbos(con_recursion, "solución es recursiva", "soluciones son recursivas")
frase_ciclos = (
    frase_verbos(con_for, "usa", "usan") + " un ciclo `for`, "
    + frase_verbos(con_while, "usa", "usan") + " un `while` y "
    + frase_verbos(con_if, "usa", "usan") + " un `if`"
)

if ratio_codigo >= 2:
    frase_ratio = f"es {ratio_codigo} veces la mediana"
else:
    frase_ratio = f"apenas supera la mediana ({ratio_codigo} veces)"

partes = []

partes.append(
    f"### Lo que nos deja la exploración\n\n"
    f"**1. Ficha del corpus.** MBPP trae {n_total} problemas repartidos en cuatro splits: "
    f"{n_train} en `train`, {n_validation} en `validation`, {n_test} en `test` y {n_prompt} en "
    f"`prompt`. Para el proyecto usamos los tres primeros: `train` para el finetuning, "
    f"`validation` para vigilar el entrenamiento y elegir la época, y `test` una sola vez al "
    f"final. El split `prompt` no se usa ni para entrenar ni para evaluar."
)

partes.append(
    f"**2. Separación y fuga de datos.** Los `task_id` de `train` van del {min(ids_train)} al "
    f"{max(ids_train)}, los de `validation` del {min(ids_validation)} al {max(ids_validation)} y "
    f"los de `test` del {min(ids_test)} al {max(ids_test)}. No hay un solo identificador repetido "
    f"entre particiones; {frase_duplicados}. Esto importa porque una fuga entre `train` y `test` "
    f"inflaría el pass@1 sin que el modelo sea mejor: estaría resolviendo problemas que ya vio. "
    f"{frase_contenido}"
)

partes.append(
    f"**3. ¿Se parecen `train` y `validation`?** El enunciado promedia "
    f"{round(promedio(largos_texto_train), 1)} caracteres en `train` y "
    f"{round(promedio(largos_texto_validation), 1)} en `validation`, con p95 de "
    f"{percentil(largos_texto_train, 0.95)} y {percentil(largos_texto_validation, 0.95)}. La "
    f"solución promedia {round(promedio(largos_codigo_train), 1)} y "
    f"{round(promedio(largos_codigo_validation), 1)} caracteres, con p95 de "
    f"{percentil(largos_codigo_train, 0.95)} y {percentil(largos_codigo_validation, 0.95)}. La "
    f"prueba de Kolmogorov-Smirnov {frase_ks_texto} entre las distribuciones del enunciado "
    f"(estadístico {round(ks_texto.statistic, 3)}, p-valor {round(ks_texto.pvalue, 3)}) y "
    f"{frase_ks_codigo} entre las de la solución (estadístico {round(ks_codigo.statistic, 3)}, "
    f"p-valor {round(ks_codigo.pvalue, 3)}). {frase_potencia} Para lo que necesitamos alcanza: si "
    f"el modelo mejora en `validation`, es razonable esperar una mejora parecida en `test`."
)

partes.append(
    f"**4. Los enunciados son plantillas.** De los {n_train} problemas, {empieza_to} empiezan "
    f"literalmente con “write a function to”, {empieza_that} con “write a function that” y "
    f"{empieza_python} con “write a python function”.{frase_resto} El vocabulario es pequeño: "
    f"{len(todas_las_palabras)} palabras distintas en total y {len(conteo_palabras)} sin contar "
    f"las vacías y las de la plantilla. Las palabras con contenido más frecuentes son {lista_top}. "
    f"Esa uniformidad tiene dos consecuencias. Primero, el modelo no necesita interpretar "
    f"instrucciones muy diversas: aprende un molde y lo repite, así que el pass@1 dependerá sobre "
    f"todo de resolver el problema y no de entender el enunciado. Segundo, el nombre de la función "
    f"objetivo solo aparece en {nombres_en_texto} de los {n_train} enunciados "
    f"({porcentaje_nombres} %), aunque los `assert` la llaman por su nombre; sin la firma en el "
    f"prompt, el modelo tendría que adivinarlo y el pass@1 caería por una razón que no es de "
    f"programación. Por eso D-018 incluye el nombre y la firma, y solo eso, en el prompt."
)

partes.append(
    f"**5. Las soluciones son cortas.** La mediana es de {mediana_codigo} caracteres y "
    f"{percentil(lineas_train, 0.50)} líneas; el p95, de {percentil(largos_codigo_train, 0.95)} "
    f"caracteres y {percentil(lineas_train, 0.95)} líneas; la más larga llega a "
    f"{max(largos_codigo_train)} caracteres y {max(lineas_train)} líneas. Con estos tamaños, un "
    f"límite de {MAX_LEN} tokens para el target y de {MAX_NEW_TOKENS} tokens nuevos al generar "
    f"{frase_limite}: el p95 queda muy por debajo del límite y solo la cola se le acerca. Si el "
    f"límite bajara a 256, las soluciones del extremo podrían quedar truncadas y el problema se "
    f"contaría como incorrecto por el recorte y no por el modelo."
)

partes.append(
    f"**6. Qué tiene que generar el modelo.** De las {n_train} soluciones de `train`, "
    f"{frase_una}, {frase_varias} y {frase_imports}; {frase_clases}. Además, {frase_recursion}, "
    f"{frase_ciclos}. Esto explica por qué el target es el código de referencia completo y no solo "
    f"el cuerpo de la función: si entrenáramos con el cuerpo, el modelo nunca aprendería a "
    f"escribir los imports, las clases ni las funciones auxiliares que necesitan esos problemas."
)

partes.append(
    f"**7. El estilo del código también importa.** {frase_tabs} para indentar, {frase_salida}, y "
    f"{frase_linea}. El modelo puede imitar los tabs, así que el ejecutor tiene que aceptarlos y "
    f"el tokenizador no puede asumir espacios. Que el corpus no imprima ni pida datos es una buena "
    f"noticia: el código generado no debería llenar la salida ni quedarse esperando entrada, "
    f"aunque nada impide que lo haga y por eso se ejecuta en un subproceso aparte."
)

partes.append(
    f"**8. Los `assert` son uniformes pero exigentes.** En `train` y `validation` hay "
    f"{total_asserts} `assert` y {frase_asserts}. Los valores esperados se reparten así: "
    f"{detalle_tipos}. Que la mayoría sean constantes facilita la comparación, pero las listas, "
    f"tuplas y diccionarios obligan a ejecutar el código y comparar estructuras, no texto. Por eso "
    f"la verificación de la sección 6 corre el código generado en un subproceso y no intenta "
    f"analizarlo estáticamente. Además, {frase_setup} confirma que el código de preparación tiene "
    f"que ejecutarse antes de los `assert`, y en el orden que fija D-020."
)

partes.append(
    f"**9. Riesgos que deja la exploración.** El primero es el tamaño: la solución más larga "
    f"({max(largos_codigo_train)} caracteres) {frase_ratio}, así que cualquier límite de "
    f"generación se define por la cola y no por el caso típico. El segundo son los tabs y las "
    f"clases: el código generado puede ser sintácticamente válido y aun así fallar al ejecutarse "
    f"por indentación o por nombres que no existen. El tercero es el silencio de `validation`: "
    f"solo {n_validation} problemas, así que cada acierto mueve el pass@1 "
    f"{puntos_por_problema} puntos porcentuales y no alcanza para elegir hiperparámetros finos. "
    f"Los tres se cubren con el mismo diseño: límites holgados, ejecución aislada con timeout y "
    f"decisiones tomadas sobre el conjunto completo de validación, nunca sobre `test`."
)

partes.append(
    f"**10. Limitaciones de esta exploración.** Las longitudes están en caracteres y palabras, no "
    f"en tokens: eso se mide en la sección 4 con cada tokenizador, porque depende del modelo. No "
    f"medimos dificultad ni calidad de las soluciones, solo su forma. `validation` tiene "
    f"{n_validation} ejemplos, suficiente para comparar distribuciones gruesas pero no para "
    f"detectar diferencias pequeñas. Y `test` no se analizó a propósito (D-030): solo se contaron "
    f"sus problemas y se revisaron sus identificadores."
)

display(Markdown("\n\n".join(partes)))
