# %% [markdown]
# ## 6. Implementación de la evaluación pass@1
#
# pass@1 es la proporción de problemas en los que el modelo acierta a la primera. Para cada problema
# generamos una única solución con decodificación greedy, la ejecutamos y corremos sus `assert`; si
# todos pasan, el problema cuenta como acierto. Al final dividimos los aciertos entre los problemas
# evaluados.
#
# La evaluación tiene cuatro pasos: generar las soluciones en lotes, extraer el código del texto que
# devuelve el modelo, ejecutarlo en un entorno aislado y contar los aciertos. Aquí implementamos los
# cuatro con la misma configuración para las cuatro configuraciones del proyecto: mismo prompt,
# `max_new_tokens=512`, una sola solución y decodificación greedy. En esta sección no se descarga
# ningún modelo ni se toca `test`: el sandbox se prueba con soluciones escritas a mano y la
# generación se estrena en la sección 7.

# %% [markdown]
# Generamos varias soluciones a la vez para aprovechar la GPU. Los prompts de un lote tienen
# longitudes distintas, así que el relleno va a la izquierda: de esa forma el último token real de
# cada fila queda al final y el modelo puede continuar desde ahí. La `attention_mask` marca el
# relleno y el recorte descarta el prompt para quedarnos solo con lo generado.

# %%
@torch.no_grad()
def generar_soluciones(
    modelo,
    tokenizador,
    prompts,
    tamano_lote=BATCH_SIZE,
    max_new_tokens=MAX_NEW_TOKENS,
):
    tokenizador.padding_side = "left"
    if tokenizador.pad_token is None:
        tokenizador.pad_token = tokenizador.eos_token
    modelo.config.pad_token_id = tokenizador.pad_token_id
    modelo.eval()

    soluciones = []
    for inicio in tqdm(range(0, len(prompts), tamano_lote), desc="Generando", leave=False):
        bloque = prompts[inicio: inicio + tamano_lote]
        entradas = tokenizador(
            bloque,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        entradas = entradas.to(modelo.device)
        salida = modelo.generate(
            **entradas,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizador.pad_token_id,
        )
        generado = salida[:, entradas["input_ids"].shape[1]:]
        textos = tokenizador.batch_decode(generado, skip_special_tokens=True)
        for texto in textos:
            soluciones.append(texto)
    return soluciones

# %% [markdown]
# Antes de generar con un modelo, comprobamos que el tokenizador deja el relleno a la izquierda y
# que la máscara lo marca. Usamos tres problemas de validación para no tocar `test`.

# %%
prompts_demo = []
for ejemplo in EJEMPLOS_MONITOREO[:3]:
    prompts_demo.append(construir_prompt(ejemplo))

tokenizador_demo = tokenizador_qwen
tokenizador_demo.padding_side = "left"
entradas_demo = tokenizador_demo(
    prompts_demo,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=MAX_LEN,
)

print("input_ids:", tuple(entradas_demo["input_ids"].shape))
print("attention_mask:", tuple(entradas_demo["attention_mask"].shape))
print()

for i in range(len(prompts_demo)):
    ids = entradas_demo["input_ids"][i].tolist()
    mascara = entradas_demo["attention_mask"][i].tolist()
    reales = sum(mascara)
    relleno = len(ids) - reales
    print("prompt", i + 1, "->", reales, "tokens reales,", relleno, "de relleno")
    print("  máscara al inicio:", mascara[:4])
    print("  máscara al final:", mascara[-4:])
    print("  últimos tokens:", tokenizador_demo.convert_ids_to_tokens(ids[-4:]))

sin_relleno = tokenizador_demo(prompts_demo[0], add_special_tokens=False)["input_ids"]
ultimo_sin_relleno = tokenizador_demo.convert_ids_to_tokens(sin_relleno[-1:])
ultimo_en_lote = tokenizador_demo.convert_ids_to_tokens(entradas_demo["input_ids"][0].tolist()[-1:])
print()
print("último token sin relleno:", ultimo_sin_relleno)
print("último token en el lote:", ultimo_en_lote)

# %% [markdown]
# El modelo devuelve texto, no código. Puede encerrar la solución en un bloque markdown, repetir el
# encabezado `### Solution` o escribir una frase antes del `def`. La extracción se queda con el
# primer fragmento que es Python válido y descarta lo que no lo es, pero no corrige el código: si el
# modelo no escribe una solución válida, el problema falla.

# %%
def extraer_codigo(texto):
    texto = texto.replace("\r\n", "\n")

    partes = texto.split("```")
    if len(partes) >= 3:
        for posicion in range(1, len(partes), 2):
            bloque = partes[posicion]
            lineas = bloque.split("\n")
            if len(lineas) > 0:
                primera = lineas[0].strip().lower()
                if primera in ["python", "py", "python3"]:
                    lineas = lineas[1:]
            bloque = "\n".join(lineas).strip("\n")
            if bloque.strip() != "":
                texto = bloque
                break

    if "### Solution" in texto:
        texto = texto.split("### Solution")[-1]

    for marca in ["\n### Problem", "\n### Signature"]:
        if marca in texto:
            texto = texto[: texto.index(marca)]

    lineas = texto.split("\n")
    for i in range(len(lineas)):
        if lineas[i].strip() == "":
            continue
        if lineas[i][0].isspace():
            continue
        candidato = "\n".join(lineas[i:]).strip("\n")
        try:
            ast.parse(candidato)
            return candidato + "\n"
        except SyntaxError:
            continue

    codigo = texto.strip("\n")
    return codigo + "\n"


textos = [
    "def doblar(x):\n    return 2 * x\n",
    "```python\ndef doblar(x):\n    return 2 * x\n```",
    "Aquí está la solución:\n\ndef doblar(x):\n    return 2 * x\n",
    "### Solution\ndef doblar(x):\n    return 2 * x\n\n### Problem\nOtro problema",
    "MAX = 10\n\ndef doblar(x):\n    return MAX * x\n",
]

for texto in textos:
    codigo = extraer_codigo(texto)
    print("---")
    print(repr(codigo))

# %% [markdown]
# Ejecutamos cada solución en un subproceso aparte, nunca en el kernel. El subproceso arranca con
# `-I` (sin el directorio actual ni paquetes del usuario), corre en una carpeta temporal, usa un
# `HOME` temporal y no recibe las variables con credenciales. El tiempo límite es de 5 segundos: un
# bucle infinito se corta y el notebook sigue. El orden es código generado, `test_setup_code` y
# `assert`, porque hay problemas cuyo setup usa clases que define la solución.

# %%
def ejecutar_solucion(codigo, ejemplo, limite_segundos=5):
    if codigo.strip() == "":
        return {"pasa": False, "motivo": "vacio", "detalle": "sin código", "segundos": 0.0}

    preparacion = ejemplo["test_setup_code"].replace("\r\n", "\n").strip()

    partes = [codigo.rstrip()]
    if preparacion != "":
        partes.append(preparacion)
    for linea in ejemplo["test_list"]:
        partes.append(linea.strip())
    partes.append('print("__FIN__")')
    script = "\n\n".join(partes) + "\n"

    entorno = os.environ.copy()
    claves = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY", "GITHUB_TOKEN", "GH_TOKEN"]
    for clave in claves:
        if clave in entorno:
            del entorno[clave]

    inicio = time.time()
    with tempfile.TemporaryDirectory() as directorio:
        entorno["HOME"] = directorio
        try:
            proceso = subprocess.run(
                [sys.executable, "-I", "-c", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=limite_segundos,
                cwd=directorio,
                env=entorno,
            )
        except subprocess.TimeoutExpired:
            segundos = time.time() - inicio
            detalle = "más de " + str(limite_segundos) + " segundos"
            return {"pasa": False, "motivo": "timeout", "detalle": detalle, "segundos": segundos}
        except Exception as error:
            segundos = time.time() - inicio
            return {"pasa": False, "motivo": "entorno", "detalle": str(error), "segundos": segundos}

    segundos = time.time() - inicio
    detalle = proceso.stderr
    if detalle is None:
        detalle = ""
    if len(detalle) > 500:
        detalle = detalle[-500:]

    if proceso.returncode == 0 and "__FIN__" in proceso.stdout:
        return {"pasa": True, "motivo": "ok", "detalle": "", "segundos": segundos}
    if "AssertionError" in detalle:
        return {"pasa": False, "motivo": "assert", "detalle": detalle, "segundos": segundos}
    if "SyntaxError" in detalle or "IndentationError" in detalle:
        return {"pasa": False, "motivo": "sintaxis", "detalle": detalle, "segundos": segundos}
    return {"pasa": False, "motivo": "error", "detalle": detalle, "segundos": segundos}

# %% [markdown]
# Recorremos los problemas, generamos una solución por cada uno y ejecutamos sus `assert`. Cada
# problema queda registrado con su `task_id`, si pasó, el motivo, el tiempo y el código generado.
# El resumen y el detalle se guardan en un JSON dentro de `RESULTADOS_DIR`, que es lo que después
# permite comparar las cuatro configuraciones y sacar ejemplos cualitativos.

# %%
def evaluar_pass1(modelo, tokenizador, ejemplos, nombre, conjunto="test", guardar=True):
    if len(ejemplos) == 0:
        raise ValueError("No hay problemas para evaluar")

    prompts = []
    for ejemplo in ejemplos:
        prompts.append(construir_prompt(ejemplo))

    inicio_generacion = time.time()
    soluciones = generar_soluciones(modelo, tokenizador, prompts)
    segundos_generacion = time.time() - inicio_generacion

    if len(soluciones) != len(ejemplos):
        mensaje = "Se generaron " + str(len(soluciones)) + " soluciones"
        mensaje = mensaje + " para " + str(len(ejemplos)) + " problemas"
        raise ValueError(mensaje)

    inicio_ejecucion = time.time()
    resultados = []
    aciertos = 0
    for i in tqdm(range(len(ejemplos)), desc="Ejecutando", leave=False):
        codigo = extraer_codigo(soluciones[i])
        revision = ejecutar_solucion(codigo, ejemplos[i])
        if revision["pasa"]:
            aciertos = aciertos + 1
        resultados.append({
            "task_id": ejemplos[i]["task_id"],
            "pasa": revision["pasa"],
            "motivo": revision["motivo"],
            "segundos": round(revision["segundos"], 3),
            "codigo": codigo,
            "detalle": revision["detalle"],
        })
    segundos_ejecucion = time.time() - inicio_ejecucion

    motivos = {}
    for fila in resultados:
        motivo = fila["motivo"]
        if motivo in motivos:
            motivos[motivo] = motivos[motivo] + 1
        else:
            motivos[motivo] = 1

    pass1 = aciertos / len(ejemplos)
    resumen = {
        "experimento": nombre,
        "conjunto": conjunto,
        "evaluados": len(ejemplos),
        "aciertos": aciertos,
        "pass1": pass1,
        "segundos_generacion": round(segundos_generacion, 1),
        "segundos_ejecucion": round(segundos_ejecucion, 1),
        "motivos": motivos,
        "resultados": resultados,
    }

    print("Evaluación:", nombre, "(" + conjunto + ")")
    print("  problemas:", len(ejemplos))
    print("  aciertos:", aciertos)
    print("  pass@1:", round(pass1 * 100, 2), "%")
    print("  generación:", round(segundos_generacion, 1), "s")
    print("  ejecución:", round(segundos_ejecucion, 1), "s")
    print("  motivos:", motivos)

    if guardar:
        ruta = os.path.join(RESULTADOS_DIR, "pass1_" + nombre + ".json")
        with open(ruta, "w", encoding="utf-8") as archivo:
            json.dump(resumen, archivo, ensure_ascii=False, indent=2)
        print("  guardado en:", ruta)

    return resumen

# %% [markdown]
# Probamos el sandbox con seis soluciones escritas a mano: una correcta, una incorrecta, un bucle
# infinito, un error de sintaxis, una vacía y una que depende del `test_setup_code`. La última
# comprueba el orden: la solución define la clase y el setup la usa.

# %%
ejemplo_suma = {
    "task_id": 1,
    "test_list": ["assert sumar(2, 3) == 5", "assert sumar(-1, 1) == 0"],
    "test_setup_code": "",
}

ejemplo_nodo = {
    "task_id": 2,
    "test_list": ["assert nodo.valor == 7"],
    "test_setup_code": "nodo = Node(7)",
}

codigo_correcto = "def sumar(a, b):\n    return a + b\n"
codigo_incorrecto = "def sumar(a, b):\n    return a - b\n"
codigo_bucle = "def sumar(a, b):\n    while True:\n        pass\n"
codigo_sintaxis = "def sumar(a, b)\n    return a + b\n"
codigo_vacio = ""
codigo_nodo = "class Node:\n    def __init__(self, valor):\n        self.valor = valor\n"

casos = [
    ("correcto", codigo_correcto, ejemplo_suma, True),
    ("incorrecto", codigo_incorrecto, ejemplo_suma, False),
    ("bucle infinito", codigo_bucle, ejemplo_suma, False),
    ("error de sintaxis", codigo_sintaxis, ejemplo_suma, False),
    ("sin código", codigo_vacio, ejemplo_suma, False),
    ("orden con setup", codigo_nodo, ejemplo_nodo, True),
]

filas_pruebas = []
for caso, codigo, ejemplo, esperado in casos:
    revision = ejecutar_solucion(codigo, ejemplo)
    filas_pruebas.append({
        "caso": caso,
        "esperado": esperado,
        "pasa": revision["pasa"],
        "motivo": revision["motivo"],
        "segundos": round(revision["segundos"], 3),
    })

tabla_pruebas = pd.DataFrame(filas_pruebas)
print(tabla_pruebas.to_string(index=False))

superadas = 0
for fila in filas_pruebas:
    if fila["pasa"] == fila["esperado"]:
        superadas = superadas + 1

print()
print("Pruebas superadas:", superadas, "de", len(filas_pruebas))

# %% [markdown]
# ### Lo que nos deja la evaluación pass@1
#
# **22. Un solo protocolo para las cuatro configuraciones.** `evaluar_pass1` recibe el modelo y su
# tokenizador y siempre usa el mismo `construir_prompt`, el mismo `max_new_tokens`, la misma
# decodificación greedy y el mismo sandbox. Cambiar de configuración solo cambia los pesos, no la
# forma de evaluar. El resumen y el detalle por problema quedan en `RESULTADOS_DIR`, así que la
# comparación de la sección 10 no necesita volver a generar.
#
# **23. El sandbox aguanta código malo.** Las seis pruebas se comportaron como esperábamos: la
# solución correcta pasó, la incorrecta falló por `AssertionError`, el bucle infinito se cortó con el
# límite de 5 segundos, el error de sintaxis se reportó como tal, la solución vacía no llegó a
# ejecutarse y la que usa el `test_setup_code` pasó solo con el orden código → setup → asserts.
# Ningún caso tumbó el notebook.
#
# **24. La extracción no arregla el código.** `extraer_codigo` quita vallas markdown, encabezados
# repetidos y prosa, y conserva el código válido que empieza con una asignación, un import o una
# clase. Pero si el modelo no escribe Python válido, la solución falla. Eso es correcto: el pass@1
# mide lo que produce el modelo, no lo que nosotros reconstruimos.
#
# **25. Lo que falta probar.** La generación real se estrena en la sección 7 con Qwen base. Ahí se
# revisa el primer lote y se confirma que el modelo no continúa el bloque `### Signature` en vez de
# empezar la solución. El sandbox ya está probado y no depende de la GPU; la generación por lotes
# depende del modelo y se verifica con el primero.
