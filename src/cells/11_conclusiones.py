# %% [markdown]
# ## 11. Conclusiones
#
# **1. LoRA cumple el objetivo, sobre todo en SmolLM2.** Qwen2.5-1.5B pasó de 41.2 % a 43.6 %
# (+2.4 pp) y SmolLM2-1.7B de 4.6 % a 31.6 % (+27.0 pp) sobre los mismos 500 problemas de `test`.
# El resultado no es un modelo nuevo: es el mismo base con un adaptador de 4.3 y 6.3 millones de
# parámetros que se puede guardar, compartir y volver a cargar. En SmolLM2 el cambio es grande porque
# el punto de partida era muy bajo; en Qwen es moderado porque el base ya resolvía cuatro de cada
# diez problemas. Los intervalos de Wilson al 95 % lo confirman: en Qwen se solapan (37.0–45.6 % y
# 39.3–48.0 %), así que el +2.4 pp es compatible con el ruido de muestreo; en SmolLM2 no se solapan
# (3.1–6.8 % y 27.7–35.8 %).
#
# **2. El formato del prompt fue parte de la solución.** Incluir el nombre y la firma de la función,
# y entrenar con el código de referencia completo más el EOS, permitió que los dos modelos escribieran
# funciones con la API correcta y que SmolLM2 pasara de 397 fallos de sintaxis a 5. La explicación más
# directa es haber visto 374 soluciones completas con una instrucción fija. Al mismo tiempo, el prompt
# no lleva la solución ni los `assert`, así que el pass@1 mide generación y no memoria del dataset.
#
# **3. Elegir la época por pass@1 de validación fue la decisión correcta.** En Qwen la pérdida de
# validación se estancó en la época 4 (0.5035) mientras el pass@1 de monitoreo seguía subiendo hasta
# 48 % en la época 5; si hubiéramos mirado solo la pérdida, habríamos entregado el adaptador de la
# época 4, con dos puntos menos. En SmolLM2 las épocas 2 y 5 empataron en 32 % y el desempate por
# pérdida eligió la 5. `test` no participó en ninguna de esas decisiones.
#
# **4. Limitaciones del experimento.** Medimos una sola solución greedy por problema, con una
# semilla y una corrida de entrenamiento. El monitoreo usa 50 problemas de validación, donde cada
# acierto vale dos puntos porcentuales, así que las diferencias de uno o dos aciertos no son
# concluyentes. El entrenamiento se fijó en 5 épocas y no se hizo una búsqueda amplia de
# hiperparámetros. Los números son de MBPP, en inglés y con enunciados cortos; no se pueden extrapolar
# sin más a otros benchmarks ni a problemas largos.
#
# **5. Qué mide y qué no mide el pass@1.** Mide si el código generado pasa los tres `assert` del
# problema en un subproceso aislado con 5 segundos de límite. No mide estilo, eficiencia, legibilidad
# ni casos que los `assert` no cubren. Un problema puede fallar por un detalle de formato numérico o
# por un `assert` incompleto, y otro puede pasar con una solución que no generaliza. Aun así, es la
# métrica que pide el enunciado y la comparación base contra LoRA es justa porque las cuatro
# configuraciones se evaluaron igual.
#
# **6. Trabajo futuro.** El siguiente paso natural es repetir el entrenamiento con varias semillas y
# generar varias soluciones por problema para reportar pass@k y una medida de estabilidad. También se
# puede probar QLoRA para modelos más grandes, ampliar la búsqueda de rango y épocas usando solo
# validación, y evaluar en otros conjuntos de código. Otra línea es usar el resultado del sandbox para
# filtrar o reparar soluciones, pero eso ya no sería el mismo protocolo de pass@1 y habría que
# declararlo como un experimento aparte.
#
# En resumen: el proyecto deja un notebook reproducible, dos adaptadores LoRA y una comparación
# honesta de cuatro configuraciones. La conclusión principal es que LoRA no hace magia con 374
# ejemplos, pero sí corrige el formato y destraba a un modelo que sin entrenamiento casi no podía
# competir.
