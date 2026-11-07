# %% [markdown]
# ## 10. Comparación y resultados
#
# Ya tenemos los cuatro pass@1 medidos con el mismo protocolo sobre los 500 problemas de `test`.
# Aquí los ponemos juntos, miramos de dónde sale la diferencia, contamos cuántos problemas se ganan
# y se pierden con LoRA y revisamos ejemplos concretos. No se vuelve a generar: todo sale de los
# cuatro resúmenes que quedaron en memoria y en `RESULTADOS_DIR`.

# %% [markdown]
# Juntamos los cuatro resúmenes en una tabla. Para cada configuración mostramos aciertos, pass@1,
# intervalo de Wilson al 95 %, diferencia contra su base, mejora relativa, longitud media del código
# generado y tiempos. El intervalo ayuda a no tomar por real una diferencia de dos o tres problemas;
# la longitud media ayuda a leer los tiempos, porque un adaptador que aprendió a terminar antes
# genera menos tokens y tarda menos.

# %%
def intervalo_wilson(aciertos, total):
    proporcion = aciertos / total
    z = 1.96
    denominador = 1 + z * z / total
    centro = (proporcion + z * z / (2 * total)) / denominador
    varianza = proporcion * (1 - proporcion) / total + z * z / (4 * total * total)
    margen = z * (varianza ** 0.5) / denominador
    return centro - margen, centro + margen


pass1_por_modelo = {}
for nombre, resumen in [("Qwen2.5-1.5B", resumen_qwen_base), ("SmolLM2-1.7B", resumen_smol_base)]:
    pass1_por_modelo[nombre] = resumen["pass1"]

configuraciones = [
    ("Qwen2.5-1.5B", "base", resumen_qwen_base),
    ("Qwen2.5-1.5B", "LoRA", resumen_qwen_lora),
    ("SmolLM2-1.7B", "base", resumen_smol_base),
    ("SmolLM2-1.7B", "LoRA", resumen_smol_lora),
]

filas_comparacion = []
for nombre, pesos, resumen in configuraciones:
    caracteres = 0
    for fila in resumen["resultados"]:
        caracteres = caracteres + len(fila["codigo"])
    promedio = caracteres / len(resumen["resultados"])

    base = pass1_por_modelo[nombre]
    if pesos == "base":
        diferencia = 0.0
        relativa = 0.0
    else:
        diferencia = (resumen["pass1"] - base) * 100
        relativa = (resumen["pass1"] - base) / base * 100

    inferior, superior = intervalo_wilson(resumen["aciertos"], resumen["evaluados"])
    filas_comparacion.append({
        "modelo": nombre,
        "pesos": pesos,
        "aciertos": resumen["aciertos"],
        "pass@1 (%)": round(resumen["pass1"] * 100, 2),
        "IC 95 % inferior (%)": round(inferior * 100, 1),
        "IC 95 % superior (%)": round(superior * 100, 1),
        "delta (pp)": round(diferencia, 2),
        "mejora relativa (%)": round(relativa, 1),
        "código medio (caracteres)": round(promedio, 1),
        "generación (s)": resumen["segundos_generacion"],
        "ejecución (s)": resumen["segundos_ejecucion"],
    })

tabla_comparacion = pd.DataFrame(filas_comparacion)
print(tabla_comparacion.to_string(index=False))

# %% [markdown]
# Graficamos el pass@1 de las cuatro configuraciones. Las barras grises son los modelos base y las
# de color son los finetuneados.

# %%
etiquetas = ["Qwen base", "Qwen + LoRA", "Smol base", "Smol + LoRA"]
valores = [
    resumen_qwen_base["pass1"] * 100,
    resumen_qwen_lora["pass1"] * 100,
    resumen_smol_base["pass1"] * 100,
    resumen_smol_lora["pass1"] * 100,
]
colores = ["tab:gray", "tab:blue", "tab:gray", "tab:orange"]

fig, eje = plt.subplots(figsize=(8, 4))
eje.bar(etiquetas, valores, color=colores)
for i in range(len(valores)):
    eje.text(i, valores[i] + 0.8, str(round(valores[i], 1)) + "%", ha="center")
eje.set_ylabel("pass@1 (%)")
eje.set_title("pass@1 sobre los 500 problemas de test")
eje.set_ylim(0, 50)
eje.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Los motivos de fallo separan lo que el modelo no compila de lo que compila pero no resuelve. La
# comparación base contra LoRA muestra en qué se gastan los problemas que se ganan.

# %%
categorias = ["ok", "assert", "sintaxis", "error", "vacio", "timeout", "entorno"]
colores_motivos = ["tab:green", "tab:orange", "tab:red", "tab:brown", "tab:gray", "tab:purple", "black"]
resumenes_ordenados = [resumen_qwen_base, resumen_qwen_lora, resumen_smol_base, resumen_smol_lora]

fondo = []
for resumen in resumenes_ordenados:
    fondo.append(0)

fig, eje = plt.subplots(figsize=(9, 4))
for posicion in range(len(categorias)):
    categoria = categorias[posicion]
    valores = []
    for resumen in resumenes_ordenados:
        valores.append(resumen["motivos"].get(categoria, 0))
    eje.bar(etiquetas, valores, bottom=fondo, label=categoria, color=colores_motivos[posicion])
    for i in range(len(valores)):
        fondo[i] = fondo[i] + valores[i]

eje.set_ylabel("problemas")
eje.set_title("Motivos en los 500 problemas de test")
eje.set_ylim(0, 560)
eje.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# La matriz de transición compara problema por problema la base y el adaptador. `ambos pasan` y
# `ninguno` son los que no cambian; `solo LoRA` son las ganancias y `solo base` las regresiones. La
# ganancia neta tiene que coincidir con la diferencia de aciertos de la tabla.

# %%
def ids_que_pasan(resumen):
    ids = set()
    for fila in resumen["resultados"]:
        if fila["pasa"]:
            ids.add(fila["task_id"])
    return ids


comparaciones = [
    ("Qwen2.5-1.5B", resumen_qwen_base, resumen_qwen_lora),
    ("SmolLM2-1.7B", resumen_smol_base, resumen_smol_lora),
]

filas_transicion = []
for nombre, resumen_base, resumen_lora in comparaciones:
    ids_base = ids_que_pasan(resumen_base)
    ids_lora = ids_que_pasan(resumen_lora)
    total = resumen_base["evaluados"]
    ambos = len(ids_base & ids_lora)
    solo_base = len(ids_base - ids_lora)
    solo_lora = len(ids_lora - ids_base)
    ninguno = total - ambos - solo_base - solo_lora
    filas_transicion.append({
        "modelo": nombre,
        "ambos pasan": ambos,
        "solo base": solo_base,
        "solo LoRA": solo_lora,
        "ninguno": ninguno,
        "ganancia neta": solo_lora - solo_base,
    })

tabla_transicion = pd.DataFrame(filas_transicion)
print(tabla_transicion.to_string(index=False))

# %% [markdown]
# Miramos dos problemas por modelo que la base falla y LoRA resuelve, eligiendo los de código más
# corto para que se puedan leer. Mostramos el enunciado, la solución de referencia y lo que generó
# cada configuración. También contamos las regresiones y mostramos una si existe.

# %%
def codigo_de(resumen, ident):
    for fila in resumen["resultados"]:
        if fila["task_id"] == ident:
            return fila["codigo"]
    return ""


def motivo_de(resumen, ident):
    for fila in resumen["resultados"]:
        if fila["task_id"] == ident:
            return fila["motivo"]
    return ""


def recortar_codigo(codigo, maximo=25):
    lineas = codigo.splitlines()
    if len(lineas) <= maximo:
        return codigo
    return "\n".join(lineas[:maximo]) + "\n... (recortado)"


for nombre, resumen_base, resumen_lora in comparaciones:
    ids_base = ids_que_pasan(resumen_base)
    ids_lora = ids_que_pasan(resumen_lora)
    mejoras = ids_lora - ids_base
    regresiones = ids_base - ids_lora

    pares = []
    for ident in mejoras:
        pares.append((len(codigo_de(resumen_lora, ident)), ident))
    pares.sort()

    print("=" * 70)
    print("Modelo:", nombre)
    print("Problemas que la base falla y LoRA resuelve:", len(mejoras))
    print("Problemas que la base resuelve y LoRA falla:", len(regresiones))

    for posicion in range(min(2, len(pares))):
        ident = pares[posicion][1]
        ejemplo = buscar_ejemplo(EJEMPLOS_TEST, ident)
        print("-" * 70)
        print("task_id:", ident)
        print("enunciado:", ejemplo["text"])
        print("solución de referencia:")
        print(recortar_codigo(ejemplo["code"]))
        print("código base (" + motivo_de(resumen_base, ident) + "):")
        print(recortar_codigo(codigo_de(resumen_base, ident)))
        print("código LoRA (" + motivo_de(resumen_lora, ident) + "):")
        print(recortar_codigo(codigo_de(resumen_lora, ident)))

    if len(regresiones) > 0:
        ident = sorted(regresiones)[0]
        print("-" * 70)
        print("ejemplo de regresión, task_id:", ident)
        print("código base:")
        print(recortar_codigo(codigo_de(resumen_base, ident)))
        print("código LoRA (" + motivo_de(resumen_lora, ident) + "):")
        print(recortar_codigo(codigo_de(resumen_lora, ident)))

# %% [markdown]
# ### Lo que muestra la comparación
#
# **38. El efecto de LoRA depende del modelo.** En Qwen el pass@1 pasó de 41.0 % a 43.6 % (+2.6 pp,
# +6.3 % relativo) y en SmolLM2 de 4.8 % a 32.6 % (+27.8 pp). Los intervalos de Wilson al 95 % de
# Qwen son 36.8–45.4 % y 39.3–48.0 %: se solapan, así que con 500 problemas la mejora de Qwen no se
# puede separar del ruido de muestreo. En SmolLM2 son 3.2–7.0 % y 28.6–36.8 %, sin solapamiento: ahí
# la mejora es clara. El adaptador no aporta lo mismo a los dos, y un solo promedio de las cuatro
# configuraciones escondería esa diferencia.
#
# **39. Los motivos cuentan la misma historia.** En Qwen los errores de ejecución bajaron de 56 a 36
# y la sintaxis de 7 a 3, mientras los `assert` subieron de 232 a 243: el adaptador produce código
# un poco más válido y algo más de código que llega a ejecutarse. En SmolLM2 la sintaxis cayó de 396
# a 5 y los `assert` subieron de 33 a 284: el problema dejó de ser escribir Python y pasó a ser
# resolver el problema.
#
# **40. La ganancia neta no es la historia completa.** La matriz de transición muestra cuántos
# problemas se ganan y cuántos se pierden; el saldo es +13 en Qwen y +139 en SmolLM2, que coincide con
# la diferencia de aciertos. Mirar solo el saldo ocultaría las regresiones, y mirar solo los ejemplos
# bonitos ocultaría que el adaptador también puede empeorar problemas que la base resolvía.
#
# **41. El costo es razonable.** La evaluación de 500 problemas tarda entre 30 y 35 s de ejecución en
# el sandbox para las cuatro configuraciones, y la generación bajó en los dos modelos finetuneados,
# una señal compatible con respuestas más cortas. Entrenar cinco épocas es barato; lo caro es generar,
# y eso ya está resuelto con lotes y con un límite de tokens bien dimensionado.
