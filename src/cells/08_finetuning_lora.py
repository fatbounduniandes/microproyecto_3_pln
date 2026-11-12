# %% [markdown]
# ## 8. Finetuning con LoRA
#
# LoRA congela el modelo base y entrena dos matrices pequeñas en las proyecciones de atención. Así el
# adaptador ocupa pocos MB, se guarda en Drive y se puede cargar sobre el mismo modelo base que ya
# evaluamos.
#
# Usamos la configuración de referencia: rango 16, alpha 16, dropout 0.05 y las proyecciones q, k, v
# y o. Entrenamos 5 épocas con AdamW, learning rate 2e-4, weight decay 0, warmup corto y scheduler
# coseno sobre un batch efectivo de 16. Cada época se mide con la pérdida de validación y con el
# pass@1 de los 50 problemas de monitoreo; el adaptador que se guarda es el de mejor pass@1, con
# desempate por menor pérdida. `test` no participa en ninguna de esas decisiones.

# %% [markdown]
# Fijamos los hiperparámetros y la configuración de LoRA.

# %%
EPOCAS = 5
TASA_APRENDIZAJE = 2e-4
PESO_DECAIMIENTO = 0.0

configuracion_lora = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

print("Épocas:", EPOCAS)
print("Learning rate:", TASA_APRENDIZAJE)
print("Weight decay:", PESO_DECAIMIENTO)
print("Rango:", configuracion_lora.r)
print("Alpha:", configuracion_lora.lora_alpha)
print("Dropout:", configuracion_lora.lora_dropout)
print("Módulos objetivo:", configuracion_lora.target_modules)

# %% [markdown]
# Cargamos el modelo base en `float32` y activamos el checkpointing de gradientes: el modelo congela
# sus pesos, así que la memoria se va en activaciones, y el checkpointing las recalcula en el
# backward a cambio de un poco de cómputo. Con PEFT hay que pedir que la entrada requiera gradiente
# y apagar la caché de generación mientras se entrena.

# %%
def preparar_modelo_lora(modelo_id, revision):
    modelo = AutoModelForCausalLM.from_pretrained(
        modelo_id,
        revision=revision,
        torch_dtype=torch.float32,
    )
    modelo = modelo.to(DEVICE)
    modelo = get_peft_model(modelo, configuracion_lora)
    modelo.gradient_checkpointing_enable()
    modelo.enable_input_require_grads()
    modelo.config.use_cache = False
    return modelo


def contar_parametros_lora(modelo):
    total = 0
    for nombre, parametro in modelo.named_parameters():
        if "lora_" in nombre:
            total = total + parametro.numel()
    return total

# %% [markdown]
# Preparamos los lotes con `preparar_lote`, que ya deja el prompt y el relleno en `-100`. La pérdida
# de validación se promedia por token supervisado y no por lote, para que los lotes con respuestas
# cortas no pesen lo mismo que los largos. El bloque se arma ejemplo por ejemplo porque los splits
# son objetos `Dataset` de Hugging Face: cortarlos con `[inicio:fin]` devuelve un diccionario de
# columnas, no una lista de ejemplos.

# %%
def mover_lote(lote):
    for clave in lote:
        lote[clave] = lote[clave].to(DEVICE)
    return lote


@torch.no_grad()
def calcular_perdida(modelo, tokenizador, ejemplos):
    modelo.eval()
    perdida_total = 0.0
    tokens_total = 0

    for inicio in range(0, len(ejemplos), BATCH_SIZE):
        bloque = []
        for posicion in range(inicio, min(inicio + BATCH_SIZE, len(ejemplos))):
            bloque.append(ejemplos[posicion])
        lote = preparar_lote(bloque, tokenizador)
        lote = mover_lote(lote)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            salidas = modelo(**lote)
        supervisados = (lote["labels"] != -100).sum().item()
        perdida_total = perdida_total + salidas.loss.item() * supervisados
        tokens_total = tokens_total + supervisados

    return perdida_total / tokens_total

# %% [markdown]
# Una época recorre los 374 ejemplos en lotes de 16, en orden aleatorio con semilla, y actualiza el
# optimizador y el scheduler paso a paso.

# %%
def entrenar_epoca(modelo, tokenizador, ejemplos, optimizador, programador, escalador, generador):
    modelo.train()
    indices = list(range(len(ejemplos)))
    generador.shuffle(indices)
    perdida_total = 0.0
    tokens_total = 0

    for inicio in range(0, len(indices), BATCH_SIZE):
        bloque = []
        for posicion in range(inicio, min(inicio + BATCH_SIZE, len(indices))):
            bloque.append(ejemplos[indices[posicion]])
        lote = preparar_lote(bloque, tokenizador)
        lote = mover_lote(lote)

        optimizador.zero_grad()
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            salidas = modelo(**lote)
            perdida = salidas.loss
        if USE_SCALER:
            escalador.scale(perdida).backward()
            escalador.step(optimizador)
            escalador.update()
        else:
            perdida.backward()
            optimizador.step()
        programador.step()

        supervisados = (lote["labels"] != -100).sum().item()
        perdida_total = perdida_total + perdida.item() * supervisados
        tokens_total = tokens_total + supervisados

    return perdida_total / tokens_total

# %% [markdown]
# El bucle completo entrena una época, mide pérdida de validación y pass@1 de monitoreo, y guarda el
# adaptador si mejora. Antes de generar hay que devolverle la caché al modelo; después se apaga otra
# vez para seguir entrenando. Si el adaptador ya existe en Drive, se carga y se salta el
# entrenamiento, y el historial se recupera del JSON.

# %%
def entrenar_lora(nombre, modelo_id, revision, tokenizador, ruta_checkpoint):
    ruta_historial = os.path.join(RESULTADOS_DIR, "historial_" + nombre + ".json")
    ruta_adaptador = os.path.join(ruta_checkpoint, "adapter_config.json")

    if os.path.exists(ruta_adaptador):
        print("Adaptador encontrado:", ruta_checkpoint)
        print("Se carga y se salta el entrenamiento.")
        modelo = cargar_modelo_base(modelo_id, revision)
        modelo = PeftModel.from_pretrained(modelo, ruta_checkpoint)
        configuracion = modelo.peft_config["default"]
        print("  r:", configuracion.r, "| alpha:", configuracion.lora_alpha)
        print("  dropout:", configuracion.lora_dropout)
        print("  módulos:", configuracion.target_modules)
        print("  parámetros LoRA:", contar_parametros_lora(modelo))
        historial = []
        if os.path.exists(ruta_historial):
            with open(ruta_historial, encoding="utf-8") as archivo:
                historial = json.load(archivo)
        modelo = None
        liberar_vram()
        return historial

    fijar_semilla()
    print("Entrenando", nombre)

    modelo = preparar_modelo_lora(modelo_id, revision)
    total = contar_parametros(modelo)
    entrenables = contar_parametros_lora(modelo)
    porcentaje = round(100 * entrenables / total, 2)
    print("Parámetros totales:", total)
    print("Parámetros LoRA:", entrenables, "(" + str(porcentaje) + "%)")

    parametros = []
    for parametro in modelo.parameters():
        if parametro.requires_grad:
            parametros.append(parametro)

    optimizador = torch.optim.AdamW(
        parametros,
        lr=TASA_APRENDIZAJE,
        weight_decay=PESO_DECAIMIENTO,
    )
    lotes_por_epoca = (len(EJEMPLOS_ENTRENAMIENTO) + BATCH_SIZE - 1) // BATCH_SIZE
    pasos_totales = lotes_por_epoca * EPOCAS
    pasos_warmup = pasos_totales // 20
    if pasos_warmup < 1:
        pasos_warmup = 1
    programador = get_cosine_schedule_with_warmup(optimizador, pasos_warmup, pasos_totales)
    escalador = torch.amp.GradScaler("cuda", enabled=USE_SCALER)
    generador = random.Random(SEED)

    print("Lotes por época:", lotes_por_epoca)
    print("Pasos totales:", pasos_totales, "| pasos de warmup:", pasos_warmup)

    historial = []
    mejor_pass1 = -1.0
    mejor_perdida = 1000000.0
    mejor_epoca = 0

    for epoca in range(1, EPOCAS + 1):
        perdida_entrenamiento = entrenar_epoca(
            modelo,
            tokenizador,
            EJEMPLOS_ENTRENAMIENTO,
            optimizador,
            programador,
            escalador,
            generador,
        )
        perdida_validacion = calcular_perdida(modelo, tokenizador, EJEMPLOS_VALIDACION)

        modelo.config.use_cache = True
        modelo.generation_config.use_cache = True
        resumen = evaluar_pass1(
            modelo,
            tokenizador,
            EJEMPLOS_MONITOREO,
            "monitoreo_" + nombre + "_epoca_" + str(epoca),
            conjunto="validation",
            guardar=False,
        )
        modelo.config.use_cache = False
        modelo.generation_config.use_cache = False
        modelo.train()

        pass1_monitoreo = resumen["pass1"]
        historial.append({
            "epoca": epoca,
            "perdida_entrenamiento": round(perdida_entrenamiento, 4),
            "perdida_validacion": round(perdida_validacion, 4),
            "pass1_monitoreo": round(pass1_monitoreo, 4),
            "mejor": False,
        })

        mejora = False
        if pass1_monitoreo > mejor_pass1:
            mejora = True
        if pass1_monitoreo == mejor_pass1 and perdida_validacion < mejor_perdida:
            mejora = True

        if mejora:
            mejor_pass1 = pass1_monitoreo
            mejor_perdida = perdida_validacion
            mejor_epoca = epoca
            for fila in historial:
                fila["mejor"] = False
            historial[-1]["mejor"] = True
            modelo.save_pretrained(ruta_checkpoint)
            tokenizador.save_pretrained(ruta_checkpoint)
            print("  nuevo mejor adaptador en la época", epoca)

        with open(ruta_historial, "w", encoding="utf-8") as archivo:
            json.dump(historial, archivo, ensure_ascii=False, indent=2)

        print("Época", epoca)
        print("  pérdida entrenamiento:", round(perdida_entrenamiento, 4))
        print("  pérdida validación:", round(perdida_validacion, 4))
        print("  pass@1 monitoreo:", round(pass1_monitoreo * 100, 2), "%")

    print()
    print("Mejor época:", mejor_epoca)
    print("Mejor pass@1 de monitoreo:", round(mejor_pass1 * 100, 2), "%")
    print("Pérdida de validación en esa época:", round(mejor_perdida, 4))
    print("Adaptador:", ruta_checkpoint)

    parametros = None
    optimizador = None
    programador = None
    escalador = None
    modelo = None
    liberar_vram()
    return historial

# %% [markdown]
# Entrenamos el adaptador de Qwen2.5-1.5B. Si `checkpoint_qwen2.5_1.5b_lora` ya existe en Drive, la
# celda lo carga y no entrena. En una L4 esto tarda varios minutos por época.

# %%
historial_qwen = entrenar_lora(
    "qwen",
    QWEN_ID,
    QWEN_REVISION,
    tokenizador_qwen,
    CHECKPOINT_QWEN,
)
print("VRAM:", memoria_gpu())

# %% [markdown]
# Entrenamos el adaptador de SmolLM2-1.7B con la misma receta.

# %%
historial_smol = entrenar_lora(
    "smol",
    SMOL_ID,
    SMOL_REVISION,
    tokenizador_smol,
    CHECKPOINT_SMOL,
)
print("VRAM:", memoria_gpu())

# %% [markdown]
# Comprobamos qué archivos quedaron en cada carpeta: el adaptador, su configuración y el tokenizador.

# %%
for nombre, ruta in [("Qwen2.5-1.5B", CHECKPOINT_QWEN), ("SmolLM2-1.7B", CHECKPOINT_SMOL)]:
    print(nombre, "->", ruta)
    if os.path.exists(ruta):
        for archivo in sorted(os.listdir(ruta)):
            print("  ", archivo)
    else:
        print("   no existe")

# %% [markdown]
# Juntamos el historial de las dos corridas: pérdida de entrenamiento, pérdida de validación y pass@1
# de monitoreo por época, con la época que quedó guardada.

# %%
filas_historial = []
for nombre, historial in [("Qwen2.5-1.5B", historial_qwen), ("SmolLM2-1.7B", historial_smol)]:
    for fila in historial:
        filas_historial.append({
            "modelo": nombre,
            "época": fila["epoca"],
            "pérdida train": fila["perdida_entrenamiento"],
            "pérdida validation": fila["perdida_validacion"],
            "pass@1 monitoreo (%)": round(fila["pass1_monitoreo"] * 100, 2),
            "mejor": fila["mejor"],
        })

tabla_historial = pd.DataFrame(filas_historial)
print(tabla_historial.to_string(index=False))

# %% [markdown]
# Graficamos la pérdida y el pass@1 de monitoreo época por época. Si la corrida se saltó el
# entrenamiento y no encontró historial, no hay curvas que mostrar.

# %%
if len(tabla_historial) > 0:
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    for nombre, historial in [("Qwen2.5-1.5B", historial_qwen), ("SmolLM2-1.7B", historial_smol)]:
        if len(historial) == 0:
            continue
        epocas = []
        perdidas_train = []
        perdidas_validacion = []
        pass1_monitoreo = []
        for fila in historial:
            epocas.append(fila["epoca"])
            perdidas_train.append(fila["perdida_entrenamiento"])
            perdidas_validacion.append(fila["perdida_validacion"])
            pass1_monitoreo.append(fila["pass1_monitoreo"] * 100)
        ejes[0].plot(epocas, perdidas_train, marker="o", label=nombre + " train")
        ejes[0].plot(epocas, perdidas_validacion, marker="s", linestyle="--", label=nombre + " validation")
        ejes[1].plot(epocas, pass1_monitoreo, marker="o", label=nombre)
    ejes[0].set_xlabel("época")
    ejes[0].set_ylabel("pérdida")
    ejes[0].set_title("Pérdida")
    ejes[0].legend()
    ejes[0].grid(alpha=0.3)
    ejes[1].set_xlabel("época")
    ejes[1].set_ylabel("pass@1 (%)")
    ejes[1].set_title("pass@1 en los 50 de monitoreo")
    ejes[1].legend()
    ejes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("Sin historial: no hay curvas para graficar")

# %% [markdown]
# ### Conclusiones del finetuning
#
# **1. Una receta pequeña y explícita.** Qwen2.5-1.5B quedó con 4,358,144 parámetros entrenables
# (0.28 % del total) y SmolLM2-1.7B con 6,291,456 (0.37 %). Cada época son 24 lotes de 16 ejemplos,
# es decir, 120 pasos de optimización en las cinco épocas, con 6 pasos de warmup. Es un entrenamiento
# corto: el costo del proyecto está en generar, no en actualizar los adaptadores.
#
# **2. En Qwen la pérdida se estancó antes que el pass@1.** La pérdida de entrenamiento bajó de
# 0.7184 a 0.4252 y la de validación de 0.5361 a 0.5037, con su mínimo en la época 4 (0.5035). El
# pass@1 de monitoreo fue 46 %, 42 %, 44 %, 46 % y 48 %: la mejor época fue la 5 por pass@1, aunque
# la pérdida ya no mejoraba. Si hubiéramos elegido por pérdida, nos habríamos quedado con la 4 y con
# dos puntos menos de pass@1.
#
# **3. En SmolLM2 el desempate importó.** La pérdida de entrenamiento bajó de 0.7038 a 0.4102 y la
# de validación de 0.5556 a 0.4754. El pass@1 de monitoreo fue 28 %, 32 %, 30 %, 28 % y 32 %: las
# épocas 2 y 5 empataron en aciertos, y ganó la 5 por menor pérdida de validación (0.4754 contra
# 0.4899). La regla de desempate no fue decorativa: decidió qué adaptador se guarda.
#
# **4. La selección se hizo solo con validación.** `test` no participó en la elección de época ni de
# hiperparámetros. En cada mejora se guardaron el adaptador, el tokenizador y el historial, y la
# carpeta de Drive permite reanudar sin volver a entrenar. Con 50 problemas de monitoreo cada acierto
# vale dos puntos porcentuales, así que las diferencias de uno o dos aciertos deben leerse con
# cuidado; por eso el desempate por pérdida y no por una diferencia mínima de pass@1.
