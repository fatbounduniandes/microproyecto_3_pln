# %% [markdown]
# ## 4. Construcción de prompts y preprocesamiento
#
# El enunciado deja libre el formato de instrucción y pide tres cosas: que el prompt describa el
# problema, que el mismo formato se use en entrenamiento y evaluación, y que los tokens del prompt
# queden enmascarados con `-100` para que la pérdida solo considere la solución. Aquí fijamos ese
# formato, lo aplicamos a un ejemplo y medimos cuántos tokens ocupa con cada tokenizador.
#
# El prompt lleva el enunciado y la firma de la función objetivo. Incluimos la firma porque los
# `assert` llaman a la función por su nombre y el enunciado solo lo menciona en 13 de los 374
# problemas; sin ella, el modelo tendría que adivinarlo. No incluimos el cuerpo de la solución, ni
# los helpers, ni los `assert`, ni el `test_setup_code`: el prompt solo dice qué hay que resolver y
# cómo se llama la función.
#
# El formato queda así:
#
# ```text
# ### Problem
# {enunciado}
#
# ### Signature
# def nombre(argumentos):
#
# ### Solution
# ```
#
# El target es el código de referencia completo (imports, clases, funciones auxiliares y función
# objetivo) más el EOS. Lo dejamos completo porque 103 de las 374 soluciones de `train` no empiezan
# con la función objetivo, y así el modelo aprende a escribir todo lo que la solución necesita.

# %% [markdown]
# Sacamos la firma leyendo el código con `ast`: el nombre sale de los `assert` y la línea del `def` se
# corta en los dos puntos.

# %%
FORMATO_PROBLEMA = "### Problem"
FORMATO_FIRMA = "### Signature"
FORMATO_SOLUCION = "### Solution"


def nombre_funcion_objetivo(ejemplo):
    arbol = ast.parse(ejemplo["code"])
    nombres = []
    for nodo in arbol.body:
        if isinstance(nodo, ast.FunctionDef):
            nombres.append(nodo.name)
    llamadas = []
    for linea in ejemplo["test_list"]:
        arbol_assert = ast.parse(linea)
        for nodo in ast.walk(arbol_assert):
            if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name):
                if nodo.func.id in nombres and nodo.func.id not in llamadas:
                    llamadas.append(nodo.func.id)
    if len(llamadas) > 0:
        return llamadas[0]
    return nombres[-1]


def extraer_firma(ejemplo):
    arbol = ast.parse(ejemplo["code"])
    objetivo = nombre_funcion_objetivo(ejemplo)
    for nodo in arbol.body:
        if isinstance(nodo, ast.FunctionDef) and nodo.name == objetivo:
            linea = ejemplo["code"].splitlines()[nodo.lineno - 1]
            corte = linea.index(":")
            return linea[: corte + 1].rstrip()
    return ""


for ident in [602, 606, 601]:
    for ejemplo in dataset["train"]:
        if ejemplo["task_id"] == ident:
            firma = extraer_firma(ejemplo)
            ast.parse(firma + "\n    pass")
            print("task_id:", ident)
            print("  función objetivo:", nombre_funcion_objetivo(ejemplo))
            print("  firma:", firma)
            print("  primera línea del código:", ejemplo["code"].splitlines()[0])
            print()

# %% [markdown]
# Armamos el prompt y el target de tres problemas: uno simple, uno que empieza con un import y uno
# que empieza con una clase.

# %%
def construir_prompt(ejemplo):
    firma = extraer_firma(ejemplo)
    prompt = FORMATO_PROBLEMA + "\n"
    prompt = prompt + ejemplo["text"].strip() + "\n\n"
    prompt = prompt + FORMATO_FIRMA + "\n"
    prompt = prompt + firma + "\n\n"
    prompt = prompt + FORMATO_SOLUCION + "\n"
    return prompt


def construir_solucion(ejemplo):
    return ejemplo["code"].rstrip() + "\n"


def buscar_ejemplo(particion, ident):
    for ejemplo in particion:
        if ejemplo["task_id"] == ident:
            return ejemplo
    return None


for ident in [602, 606, 601]:
    ejemplo = buscar_ejemplo(dataset["train"], ident)
    print("=" * 70)
    print(construir_prompt(ejemplo))
    print("--- target ---")
    print(construir_solucion(ejemplo))

# %% [markdown]
# Cargamos los dos tokenizadores con su revisión fija. Si no traen token de relleno les ponemos el
# EOS, y los dejamos con padding a la izquierda porque así se genera en la evaluación. En
# entrenamiento los lotes se arman a mano con relleno a la derecha.

# %%
tokenizador_qwen = AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION)
tokenizador_smol = AutoTokenizer.from_pretrained(SMOL_ID, revision=SMOL_REVISION)

TOKENIZADORES = [
    ("Qwen2.5-1.5B", tokenizador_qwen),
    ("SmolLM2-1.7B", tokenizador_smol),
]

for nombre, tokenizador in TOKENIZADORES:
    if tokenizador.pad_token is None:
        tokenizador.pad_token = tokenizador.eos_token
    tokenizador.padding_side = "left"

for nombre, tokenizador in TOKENIZADORES:
    print("=" * 50)
    print(nombre)
    print("  clase:", type(tokenizador).__name__)
    print("  vocabulario:", len(tokenizador))
    print("  BOS:", tokenizador.bos_token, "->", tokenizador.bos_token_id)
    print("  EOS:", tokenizador.eos_token, "->", tokenizador.eos_token_id)
    print("  PAD:", tokenizador.pad_token, "->", tokenizador.pad_token_id)
    print("  padding_side:", tokenizador.padding_side)

# %% [markdown]
# Tokenizamos un ejemplo completo y miramos dónde termina el prompt y dónde empieza la solución. El
# EOS se agrega a mano porque ninguno de los dos tokenizadores lo pone, y queda supervisado.

# %%
def tokenizar_ejemplo(ejemplo, tokenizador):
    prompt = construir_prompt(ejemplo)
    solucion = construir_solucion(ejemplo)
    ids_prompt = tokenizador(prompt, add_special_tokens=False)["input_ids"]
    ids_solucion = tokenizador(solucion, add_special_tokens=False)["input_ids"]
    ids_solucion = ids_solucion + [tokenizador.eos_token_id]
    ids = ids_prompt + ids_solucion
    labels = [-100] * len(ids_prompt) + ids_solucion
    return {
        "prompt": prompt,
        "solucion": solucion,
        "ids_prompt": ids_prompt,
        "ids_solucion": ids_solucion,
        "ids": ids,
        "labels": labels,
    }


ejemplo = buscar_ejemplo(dataset["train"], 602)
datos = tokenizar_ejemplo(ejemplo, tokenizador_qwen)

print("tokens del prompt:", len(datos["ids_prompt"]))
print("tokens de la solución (con EOS):", len(datos["ids_solucion"]))
print("tokens en total:", len(datos["ids"]))
print("labels en -100:", datos["labels"].count(-100))
print("labels supervisados:", len(datos["labels"]) - datos["labels"].count(-100))
print()
print("últimos tokens del prompt:", tokenizador_qwen.convert_ids_to_tokens(datos["ids_prompt"][-6:]))
print("primeros tokens de la solución:", tokenizador_qwen.convert_ids_to_tokens(datos["ids_solucion"][:6]))
print("último token de la solución:", tokenizador_qwen.convert_ids_to_tokens([datos["ids_solucion"][-1]]))
print()
print("final del prompt:", repr(datos["prompt"][-30:]))
print("inicio del target:", repr(datos["solucion"][:30]))
print()
enmascarados_prompt = datos["labels"][: len(datos["ids_prompt"])].count(-100)
print("el prompt está completo en -100:", enmascarados_prompt == len(datos["ids_prompt"]))
print("el EOS está supervisado:", datos["labels"][-1] == tokenizador_qwen.eos_token_id)
print("cabe en MAX_LEN:", len(datos["ids"]) <= MAX_LEN)

# %% [markdown]
# Comprobamos en `train` y `validation` que el prompt tokenizado sea exactamente el prefijo del texto
# completo. Si no lo fuera, la máscara por longitud no coincidiría con la frontera real y el modelo
# recibiría pérdida por tokens del enunciado. También revisamos que todas las firmas sean Python
# válido y en cuántos problemas la función objetivo no es el último `def`.

# %%
print("El prompt tokenizado es prefijo del texto completo")
for nombre, tokenizador in TOKENIZADORES:
    revisados = 0
    distintos = 0
    for particion in ["train", "validation"]:
        for ejemplo in dataset[particion]:
            datos = tokenizar_ejemplo(ejemplo, tokenizador)
            texto_completo = datos["prompt"] + datos["solucion"]
            ids_completos = tokenizador(texto_completo, add_special_tokens=False)["input_ids"]
            if ids_completos[: len(datos["ids_prompt"])] != datos["ids_prompt"]:
                distintos = distintos + 1
            revisados = revisados + 1
    print(" ", nombre, "->", distintos, "diferencias en", revisados, "ejemplos")

firmas_invalidas = 0
for particion in ["train", "validation"]:
    for ejemplo in dataset[particion]:
        firma = extraer_firma(ejemplo)
        try:
            ast.parse(firma + "\n    pass")
        except SyntaxError:
            firmas_invalidas = firmas_invalidas + 1

print("Firmas que no son Python válido:", firmas_invalidas)

no_ultimo = 0
ids_no_ultimo = []
for particion in ["train", "validation"]:
    for ejemplo in dataset[particion]:
        arbol = ast.parse(ejemplo["code"])
        defs = []
        for nodo in arbol.body:
            if isinstance(nodo, ast.FunctionDef):
                defs.append(nodo.name)
        if nombre_funcion_objetivo(ejemplo) != defs[-1]:
            no_ultimo = no_ultimo + 1
            ids_no_ultimo.append(ejemplo["task_id"])

print("Problemas donde la función objetivo no es el último def:", no_ultimo)
print("  task_id:", ids_no_ultimo)

# %% [markdown]
# Medimos cuántos tokens ocupa el prompt, la solución y el total en `train` y `validation` con cada
# tokenizador.

# %%
filas = []
superan_total = 0
for nombre_split in ["train", "validation"]:
    for nombre_modelo, tokenizador in TOKENIZADORES:
        largos_prompt = []
        largos_solucion = []
        largos_total = []
        superan = 0
        for ejemplo in dataset[nombre_split]:
            datos = tokenizar_ejemplo(ejemplo, tokenizador)
            largos_prompt.append(len(datos["ids_prompt"]))
            largos_solucion.append(len(datos["ids_solucion"]))
            largos_total.append(len(datos["ids"]))
            if len(datos["ids"]) > MAX_LEN:
                superan = superan + 1
        superan_total = superan_total + superan
        resumen_prompt = str(round(promedio(largos_prompt), 1))
        resumen_prompt = resumen_prompt + " / " + str(percentil(largos_prompt, 0.50))
        resumen_prompt = resumen_prompt + " / " + str(percentil(largos_prompt, 0.95))
        resumen_prompt = resumen_prompt + " / " + str(max(largos_prompt))
        resumen_solucion = str(round(promedio(largos_solucion), 1))
        resumen_solucion = resumen_solucion + " / " + str(percentil(largos_solucion, 0.50))
        resumen_solucion = resumen_solucion + " / " + str(percentil(largos_solucion, 0.95))
        resumen_solucion = resumen_solucion + " / " + str(max(largos_solucion))
        resumen_total = str(round(promedio(largos_total), 1))
        resumen_total = resumen_total + " / " + str(percentil(largos_total, 0.50))
        resumen_total = resumen_total + " / " + str(percentil(largos_total, 0.95))
        resumen_total = resumen_total + " / " + str(max(largos_total))
        filas.append({
            "split": nombre_split,
            "modelo": nombre_modelo,
            "prompt": resumen_prompt,
            "solución": resumen_solucion,
            "total": resumen_total,
            "superan 512": superan,
        })

tabla_tokens = pd.DataFrame(filas)
print("Cada medida es media / mediana / p95 / máximo, en tokens")
print(tabla_tokens.to_string(index=False))
print()
print("Ejemplos que superan MAX_LEN:", superan_total)

# %% [markdown]
# Dibujamos la curva acumulada del total de tokens. En el eje x está el tamaño del ejemplo y en el eje
# y la proporción de problemas que caben en ese tamaño o menos. La línea roja marca el límite de 512.

# %%
maximos = {}
superan_total = 0

plt.figure(figsize=(10, 3.5))
for posicion in range(len(TOKENIZADORES)):
    nombre_modelo, tokenizador = TOKENIZADORES[posicion]
    plt.subplot(1, 2, posicion + 1)
    for nombre_split in ["train", "validation"]:
        largos = []
        for ejemplo in dataset[nombre_split]:
            datos = tokenizar_ejemplo(ejemplo, tokenizador)
            largos.append(len(datos["ids"]))
            if len(datos["ids"]) > MAX_LEN:
                superan_total = superan_total + 1
        maximos[nombre_modelo + " " + nombre_split] = max(largos)
        x, y = acumulada(largos)
        plt.plot(x, y, label=nombre_split)
    plt.axvline(MAX_LEN, color="tab:red", linestyle="--", label="límite de " + str(MAX_LEN))
    plt.title(nombre_modelo)
    plt.xlabel("Tokens del prompt y la solución")
    plt.ylabel("Proporción acumulada")
    plt.legend()

plt.tight_layout()
plt.show()

print("Máximo de tokens por split y modelo:")
for clave in maximos:
    print(" ", clave, "->", maximos[clave])
print("Ejemplos que superan MAX_LEN:", superan_total)

# %% [markdown]
# Armamos un lote de cuatro ejemplos con relleno dinámico: el lote se estira hasta el ejemplo más
# largo y los `labels` del relleno también quedan en `-100`. Así la pérdida no considera ni el prompt
# ni el relleno.

# %%
def preparar_lote(ejemplos, tokenizador):
    datos = []
    for ejemplo in ejemplos:
        datos.append(tokenizar_ejemplo(ejemplo, tokenizador))

    largo_maximo = 0
    for fila in datos:
        if len(fila["ids"]) > largo_maximo:
            largo_maximo = len(fila["ids"])

    if largo_maximo > MAX_LEN:
        raise ValueError("Un ejemplo del lote supera MAX_LEN: " + str(largo_maximo))

    input_ids = []
    attention_mask = []
    labels = []
    for fila in datos:
        relleno = largo_maximo - len(fila["ids"])
        input_ids.append(fila["ids"] + [tokenizador.pad_token_id] * relleno)
        attention_mask.append([1] * len(fila["ids"]) + [0] * relleno)
        labels.append(fila["labels"] + [-100] * relleno)

    lote = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }
    return lote


ejemplos_demo = []
for indice in range(4):
    ejemplos_demo.append(dataset["train"][indice])

lote = preparar_lote(ejemplos_demo, tokenizador_qwen)

print("input_ids:", tuple(lote["input_ids"].shape))
print("attention_mask:", tuple(lote["attention_mask"].shape))
print("labels:", tuple(lote["labels"].shape))
print()

filas = []
for i in range(len(ejemplos_demo)):
    etiquetas = lote["labels"][i].tolist()
    mascara = lote["attention_mask"][i].tolist()
    enmascarados = 0
    supervisados = 0
    relleno = 0
    for j in range(len(etiquetas)):
        if mascara[j] == 0:
            relleno = relleno + 1
        elif etiquetas[j] == -100:
            enmascarados = enmascarados + 1
        else:
            supervisados = supervisados + 1
    filas.append({
        "task_id": ejemplos_demo[i]["task_id"],
        "prompt en -100": enmascarados,
        "supervisados": supervisados,
        "relleno": relleno,
        "total": len(etiquetas),
    })

print(pd.DataFrame(filas).to_string(index=False))
print()
print("labels del primer ejemplo:")
print(lote["labels"][0].tolist())

# %% [markdown]
# Buscamos fugas: que la solución completa, los `assert` o el `test_setup_code` no aparezcan dentro
# del prompt. También revisamos línea por línea el código de referencia, sin contar la firma, que es
# la única parte del código que el prompt sí debe llevar.

# %%
fugas_solucion = 0
fugas_assert = 0
fugas_setup = 0
lineas_repetidas = []

for particion in ["train", "validation"]:
    for ejemplo in dataset[particion]:
        prompt = construir_prompt(ejemplo)
        firma = extraer_firma(ejemplo)
        solucion = construir_solucion(ejemplo)

        if solucion.strip() in prompt:
            fugas_solucion = fugas_solucion + 1
        for linea in ejemplo["test_list"]:
            if linea.strip() in prompt:
                fugas_assert = fugas_assert + 1
        if ejemplo["test_setup_code"].strip() != "":
            if ejemplo["test_setup_code"].strip() in prompt:
                fugas_setup = fugas_setup + 1
        for linea in ejemplo["code"].splitlines():
            if linea.strip() == "":
                continue
            if linea.strip() == firma:
                continue
            if linea.strip() in prompt:
                lineas_repetidas.append((ejemplo["task_id"], linea.strip()))

print("Soluciones completas dentro del prompt:", fugas_solucion)
print("Assert dentro del prompt:", fugas_assert)
print("test_setup_code dentro del prompt:", fugas_setup)
print("Líneas de código (sin la firma) dentro del prompt:", len(lineas_repetidas))
for ident, linea in lineas_repetidas:
    print("  task_id", ident, "->", repr(linea))

# %% [markdown]
# ### Lo que nos deja el preprocesamiento
#
# **1. El prompt es una parte pequeña del ejemplo.** Con este formato, el prompt promedia 32.7
# tokens con Qwen y 40.6 con SmolLM2 en `train`, y 32.4 y 40.2 en `validation`. El máximo es de 68 y
# 74 tokens. Frente a una solución que promedia 58.2 y 66.9 tokens, el prompt se queda con algo más
# de un tercio del total: 36.0 % con Qwen y 37.8 % con SmolLM2. Eso es lo que queremos, porque el
# prompt se enmascara y solo la solución aporta a la pérdida.
#
# **2. La máscara cae justo en la frontera.** En el problema 602 el prompt ocupa 32 tokens y la
# solución 43 (42 de código más el EOS). Los 32 primeros `labels` quedan en `-100` y los 43
# restantes conservan su id, incluido el EOS. La comprobación sobre los 464 ejemplos da 0
# diferencias: el prompt tokenizado es el prefijo exacto del texto completo con los dos
# tokenizadores, así que la máscara por longitud no parte ningún token ni deja texto sin cubrir.
#
# **3. 512 tokens sobran.** El total más largo es de 346 tokens (SmolLM2, `train`) y el de Qwen
# llega a 321; en `validation` los máximos son 281 y 236. Ninguno de los 464 ejemplos supera el
# límite y el margen es de 166 tokens, un 32 % de `MAX_LEN`. La mediana del total es de 78 tokens
# con Qwen y 92 con SmolLM2, así que el límite lo fija la cola y no el caso típico. Con
# `MAX_NEW_TOKENS = 512` pasa lo mismo: la solución más larga (308 tokens con SmolLM2) cabe entera.
#
# **4. La firma no siempre es el último `def`.** Las 464 firmas son Python válido y el nombre de la
# función objetivo siempre coincide con una función definida en el código. En 2 problemas (717 y
# 940) la función que llaman los `assert` aparece antes de una función auxiliar, así que quedarse
# con el último `def` habría dado una firma equivocada. Por eso el nombre se saca de los `assert` y
# no de la posición.
#
# **5. No hay fuga en el prompt.** Ninguna solución completa, ningún `assert` y ningún
# `test_setup_code` aparecen dentro del prompt de los 464 ejemplos. La única línea de código que
# coincide es `return 20` en el problema 675, y coincide porque el enunciado dice literalmente “it
# will return 20”: es una casualidad del texto, no una filtración de la solución.
#
# **6. El relleno se paga por lote, no por corpus.** En el lote de cuatro ejemplos el más largo mide
# 183 tokens y el más corto 41, así que este último lleva 142 de relleno. Con relleno dinámico esa
# diferencia se paga solo dentro del lote y no se reserva `MAX_LEN` para todos: en entrenamiento cada
# lote se estira hasta su ejemplo más largo, y el relleno queda con máscara 0 y `-100` en los
# `labels`. En generación se usa padding a la izquierda, que es lo que permite que los 16 prompts del
# lote terminen en la misma columna.
#
# Con esto el pipeline queda definido: `construir_prompt` y `construir_solucion` arman los textos,
# `tokenizar_ejemplo` aplica la máscara y `preparar_lote` construye los tensores. La sección 5 lo
# aplica a las tres particiones y deja escrito el rol de cada una.
