# %% [markdown]
# ## 7. Evaluación de los modelos base (zero-shot)
#
# Los dos modelos base no han visto MBPP: se evalúan tal como salen de Hugging Face. Este es el punto
# de partida contra el que se mide el aporte de LoRA y la primera vez que se usa `test`, que queda
# reservado para medir y no para elegir hiperparámetros.
#
# El protocolo es el de la sección 6: el mismo `construir_prompt`, una solución greedy por problema,
# `max_new_tokens=512` y el mismo sandbox. Lo único que cambia entre las cuatro configuraciones del
# proyecto son los pesos; todo lo demás se mantiene igual.

# %% [markdown]
# Antes de generar, comprobamos que `test` esté completo y separado de los otros dos conjuntos. Solo
# miramos metadatos: tamaño, `task_id`, número de `assert` y cuántos problemas traen
# `test_setup_code`.

# %%
EJEMPLOS_TEST = dataset["test"]

ids_test = []
for ejemplo in EJEMPLOS_TEST:
    ids_test.append(ejemplo["task_id"])

ids_entrenamiento = []
for ejemplo in EJEMPLOS_ENTRENAMIENTO:
    ids_entrenamiento.append(ejemplo["task_id"])

ids_validacion = []
for ejemplo in EJEMPLOS_VALIDACION:
    ids_validacion.append(ejemplo["task_id"])

cantidad_asserts = []
con_setup = 0
for ejemplo in EJEMPLOS_TEST:
    cantidad_asserts.append(len(ejemplo["test_list"]))
    if ejemplo["test_setup_code"].strip() != "":
        con_setup = con_setup + 1

print("Problemas de test:", len(EJEMPLOS_TEST))
print("task_id distintos:", len(set(ids_test)))
print("task_id que también están en train:", len(set(ids_test) & set(ids_entrenamiento)))
print("task_id que también están en validation:", len(set(ids_test) & set(ids_validacion)))
print("asserts por problema:", sorted(set(cantidad_asserts)))
print("problemas con test_setup_code:", con_setup)

# %% [markdown]
# Cargamos Qwen2.5-1.5B base con la revisión fija y la precisión mixta del runtime. Para generar basta
# con media precisión: no hay gradientes y el modelo entra holgado en la GPU. La misma función se
# reutiliza en la sección 9 para cargar la base antes de pegarle el adaptador.

# %%
def cargar_modelo_base(modelo_id, revision):
    modelo = AutoModelForCausalLM.from_pretrained(
        modelo_id,
        revision=revision,
        torch_dtype=AMP_DTYPE,
    )
    modelo = modelo.to(DEVICE)
    modelo.eval()
    return modelo


def contar_parametros(modelo):
    total = 0
    for parametro in modelo.parameters():
        total = total + parametro.numel()
    return total


def memoria_gpu():
    if not torch.cuda.is_available():
        return "sin GPU"
    asignada = torch.cuda.memory_allocated() / 1024 ** 3
    reservada = torch.cuda.memory_reserved() / 1024 ** 3
    texto = str(round(asignada, 2)) + " GB asignados, "
    texto = texto + str(round(reservada, 2)) + " GB reservados"
    return texto


modelo_qwen = cargar_modelo_base(QWEN_ID, QWEN_REVISION)
parametros_qwen = contar_parametros(modelo_qwen)

print("Modelo:", QWEN_ID)
print("Parámetros:", round(parametros_qwen / 1e9, 2), "mil millones")
print("VRAM:", memoria_gpu())

# %% [markdown]
# Antes de gastar la evaluación completa, generamos cuatro soluciones y miramos el texto crudo. Lo que
# queremos confirmar es que el modelo no repita `### Signature` ni `### Problem` y que empiece a
# escribir código, no una frase en prosa. No se elige nada con estas cuatro: solo se verifica el
# formato del prompt.

# %%
def revisar_generacion(modelo, tokenizador, ejemplos, cantidad=4):
    prompts = []
    for indice in range(cantidad):
        prompts.append(construir_prompt(ejemplos[indice]))
    textos = generar_soluciones(modelo, tokenizador, prompts, tamano_lote=cantidad)

    filas = []
    for indice in range(cantidad):
        texto = textos[indice]
        codigo = extraer_codigo(texto)
        try:
            ast.parse(codigo)
            es_valido = True
        except SyntaxError:
            es_valido = False

        primera = ""
        for linea in codigo.splitlines():
            if linea.strip() != "":
                primera = linea.strip()
                break

        empieza = primera.startswith("def ") or primera.startswith("class ")
        empieza = empieza or primera.startswith("import ") or primera.startswith("from ")
        repite = ("### Signature" in texto) or ("### Problem" in texto)
        filas.append({
            "task_id": ejemplos[indice]["task_id"],
            "repite encabezado": repite,
            "empieza con código": empieza,
            "python válido": es_valido,
            "caracteres": len(texto),
        })

        print("=" * 60)
        print("task_id:", ejemplos[indice]["task_id"])
        print(texto[:400])

    return pd.DataFrame(filas)


tabla_qwen = revisar_generacion(modelo_qwen, tokenizador_qwen, EJEMPLOS_TEST)
print()
print(tabla_qwen.to_string(index=False))

# %% [markdown]
# Evaluamos los 500 problemas de `test` con Qwen2.5-1.5B base. La generación greedy y el sandbox son
# los mismos de la sección 6, y el JSON con el código y el motivo de cada problema queda en Drive. En
# una L4 esto tarda unos minutos.

# %%
resumen_qwen_base = evaluar_pass1(
    modelo_qwen,
    tokenizador_qwen,
    EJEMPLOS_TEST,
    "qwen_base",
)
print("VRAM:", memoria_gpu())

# %% [markdown]
# Liberamos la GPU y cargamos SmolLM2-1.7B base. La VRAM se libera poniendo el modelo en `None` y
# llamando a `liberar_vram`: borrar solo el parámetro dentro de la función no suelta la referencia
# del notebook.

# %%
modelo_qwen = None
liberar_vram()

modelo_smol = cargar_modelo_base(SMOL_ID, SMOL_REVISION)
parametros_smol = contar_parametros(modelo_smol)

print("Modelo:", SMOL_ID)
print("Parámetros:", round(parametros_smol / 1e9, 2), "mil millones")
print("VRAM:", memoria_gpu())

# %% [markdown]
# Repetimos la revisión del formato con SmolLM2 sobre los mismos cuatro problemas.

# %%
tabla_smol = revisar_generacion(modelo_smol, tokenizador_smol, EJEMPLOS_TEST)
print()
print(tabla_smol.to_string(index=False))

# %% [markdown]
# Evaluamos los 500 problemas con SmolLM2-1.7B base.

# %%
resumen_smol_base = evaluar_pass1(
    modelo_smol,
    tokenizador_smol,
    EJEMPLOS_TEST,
    "smol_base",
)
print("VRAM:", memoria_gpu())

# %% [markdown]
# Liberamos la GPU otra vez para que la sección 8 empiece con la memoria limpia.

# %%
modelo_smol = None
liberar_vram()
print("VRAM:", memoria_gpu())

# %% [markdown]
# Resumimos los dos resultados base. Los motivos separan los aciertos (`ok`) de los fallos por
# `assert`, por sintaxis, por timeout y por código vacío, y el tiempo distingue la generación de la
# ejecución en el sandbox.

# %%
filas_base = []
for resumen in [resumen_qwen_base, resumen_smol_base]:
    fila = {
        "experimento": resumen["experimento"],
        "problemas": resumen["evaluados"],
        "aciertos": resumen["aciertos"],
        "pass@1 (%)": round(resumen["pass1"] * 100, 2),
        "generación (s)": resumen["segundos_generacion"],
        "ejecución (s)": resumen["segundos_ejecucion"],
    }
    for motivo in ["ok", "assert", "sintaxis", "timeout", "vacio", "error", "entorno"]:
        fila[motivo] = resumen["motivos"].get(motivo, 0)
    filas_base.append(fila)

tabla_base = pd.DataFrame(filas_base)
print(tabla_base.to_string(index=False))

# %% [markdown]
# ### Conclusiones de la evaluación base
#
# **1. Qwen2.5-1.5B ya es un baseline fuerte.** Acertó 206 de los 500 problemas (41.2 %), con
# 579.6 s de generación y 29.1 s de ejecución. Sin ningún entrenamiento sobre MBPP, el modelo
# resuelve cuatro de cada diez problemas, así que el margen de LoRA en este modelo es real pero
# acotado: para subir hay que mejorar código que ya es válido en la mayoría de los casos.
#
# **2. SmolLM2-1.7B casi no sigue el formato.** Acertó 23 de 500 (4.6 %), con 629.2 s de generación
# y 29.1 s de ejecución. En la prueba de cuatro problemas, 3 de 4 empiezan con código, 2 de 4 repiten
# `### Problem` o `### Signature` y 3 de 4 producen Python válido; sus salidas miden entre 1,010 y
# 1,486 caracteres, frente a 41–694 en Qwen. El modelo sabe escribir funciones, pero se enreda con el
# formato y con el momento de terminar.
#
# **3. El perfil de fallo es distinto en cada modelo.** En Qwen hay 206 aciertos, 235 fallos por
# `assert` (47.0 %), 50 por error de ejecución (10.0 %) y solo 9 por sintaxis (1.8 %). En SmolLM2,
# 397 de los 500 fallos son de sintaxis (79.4 %), seguidos de 40 errores (8.0 %), 36 `assert` (7.2 %)
# y 4 respuestas vacías (0.8 %). Qwen falla sobre todo resolviendo; SmolLM2 falla sobre todo
# escribiendo.
#
# **4. Eso anticipa dónde puede aportar LoRA.** En SmolLM2 hay un problema de forma que el
# finetuning puede corregir con pocos ejemplos: si el modelo aprende a escribir una función válida y
# a detenerse, el pass@1 debería subir mucho. En Qwen el formato ya está resuelto y la mejora tiene
# que venir de la calidad de la solución, que es más difícil de mover con 374 ejemplos. La
# comparación base contra finetuneado tiene que leerse por modelo, no como un solo promedio.
