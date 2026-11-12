# %% [markdown]
# ## 9. Evaluación de los modelos finetuneados
#
# Cargamos cada adaptador sobre su modelo base y repetimos exactamente la evaluación de la sección 7:
# los mismos 500 problemas, el mismo prompt, una solución greedy por problema y el mismo sandbox. Así
# la diferencia entre base y finetuneado mide el aporte de LoRA y no un cambio de protocolo.

# %% [markdown]
# Antes de evaluar, comprobamos que el adaptador exista y que su configuración sea la que entrenamos:
# rango 16, alpha 16, dropout 0.05 y las cuatro proyecciones de atención.

# %%
def cargar_modelo_con_adapter(modelo_id, revision, ruta_checkpoint):
    ruta_adaptador = os.path.join(ruta_checkpoint, "adapter_config.json")
    if not os.path.exists(ruta_adaptador):
        raise FileNotFoundError("No existe el adaptador en " + ruta_checkpoint)
    modelo = cargar_modelo_base(modelo_id, revision)
    modelo = PeftModel.from_pretrained(modelo, ruta_checkpoint)
    modelo.eval()
    return modelo


modelo_qwen_lora = cargar_modelo_con_adapter(QWEN_ID, QWEN_REVISION, CHECKPOINT_QWEN)
configuracion = modelo_qwen_lora.peft_config["default"]

print("Modelo:", QWEN_ID)
print("  r:", configuracion.r)
print("  alpha:", configuracion.lora_alpha)
print("  dropout:", configuracion.lora_dropout)
print("  módulos:", configuracion.target_modules)
print("  parámetros LoRA:", contar_parametros_lora(modelo_qwen_lora))
print("VRAM:", memoria_gpu())

# %% [markdown]
# Evaluamos los 500 problemas con Qwen2.5-1.5B + LoRA.

# %%
resumen_qwen_lora = evaluar_pass1(
    modelo_qwen_lora,
    tokenizador_qwen,
    EJEMPLOS_TEST,
    "qwen_lora",
)
print("VRAM:", memoria_gpu())

# %% [markdown]
# Liberamos la GPU y cargamos el adaptador de SmolLM2 sobre su base.

# %%
modelo_qwen_lora = None
liberar_vram()

modelo_smol_lora = cargar_modelo_con_adapter(SMOL_ID, SMOL_REVISION, CHECKPOINT_SMOL)
configuracion = modelo_smol_lora.peft_config["default"]

print("Modelo:", SMOL_ID)
print("  r:", configuracion.r)
print("  alpha:", configuracion.lora_alpha)
print("  dropout:", configuracion.lora_dropout)
print("  módulos:", configuracion.target_modules)
print("  parámetros LoRA:", contar_parametros_lora(modelo_smol_lora))
print("VRAM:", memoria_gpu())

# %% [markdown]
# Evaluamos los 500 problemas con SmolLM2-1.7B + LoRA.

# %%
resumen_smol_lora = evaluar_pass1(
    modelo_smol_lora,
    tokenizador_smol,
    EJEMPLOS_TEST,
    "smol_lora",
)
print("VRAM:", memoria_gpu())

# %% [markdown]
# Liberamos la GPU al terminar.

# %%
modelo_smol_lora = None
liberar_vram()
print("VRAM:", memoria_gpu())

# %% [markdown]
# Resumimos los dos resultados finetuneados con el mismo formato de la tabla base.

# %%
filas_lora = []
for resumen in [resumen_qwen_lora, resumen_smol_lora]:
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
    filas_lora.append(fila)

tabla_lora = pd.DataFrame(filas_lora)
print(tabla_lora.to_string(index=False))

# %% [markdown]
# ### Conclusiones de la evaluación finetuneada
#
# **1. Los adaptadores cargados son los que entrenamos.** En los dos casos la verificación leyó
# r=16, alpha=16, dropout 0.05 y los módulos q, k, v y o. Qwen tiene 4,358,144 parámetros LoRA y
# SmolLM2 6,291,456, los mismos del entrenamiento, y las dos carpetas incluyen `adapter_config.json`,
# `adapter_model.safetensors` y el tokenizador. La carga se hizo sobre el mismo modelo base y la
# revisión fija, y la evaluación usó el mismo `evaluar_pass1` de las secciones 6 y 7.
#
# **2. Qwen + LoRA sube de 41.2 % a 43.6 %.** Acertó 218 de 500, doce problemas más que la base.
# Los motivos quedaron en 241 `assert`, 37 errores, 4 de sintaxis y ningún timeout. La ganancia es
# moderada y coherente con lo que anticipaba la sección 7: el formato ya estaba resuelto y LoRA tenía
# que mejorar la solución, no solo la forma. Con 500 problemas, 12 aciertos son un +2.4 pp; en la
# sección 10 los intervalos de Wilson muestran que esa diferencia se solapa con el ruido de muestreo.
#
# **3. SmolLM2 + LoRA sube de 4.6 % a 31.6 %.** Acertó 158 de 500, 135 problemas más que la base.
# Lo más elocuente es el cambio de motivo: los fallos de sintaxis pasaron de 397 a 5 y los `assert`
# subieron de 36 a 291. El adaptador no convirtió al modelo en un gran programador, pero sí le enseñó
# a escribir una función ejecutable y a detenerse; ahora la mayoría de los fallos son de lógica.
#
# **4. La mejora se mide con el mismo protocolo.** El prompt, `max_new_tokens`, la decodificación
# greedy y el sandbox son idénticos a los de la base, así que la diferencia entre 41.2 % y 43.6 %, y
# entre 4.6 % y 31.6 %, no viene de un cambio de evaluación: en SmolLM2 es clara y en Qwen queda
# dentro del ruido. La generación bajó de 579.6 a 414.4 s en Qwen y de 629.2 a 475.5 s en SmolLM2, una
# señal compatible con respuestas más cortas y con haber aprendido a terminar; la métrica que decide
# sigue siendo el pass@1, no el tiempo.
