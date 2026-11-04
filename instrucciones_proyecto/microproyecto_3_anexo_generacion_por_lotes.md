# Anexo de generación por lotes

> Texto extraído automáticamente de `PDF/MAIA_MAPLN_2026_15_Microproyecto_3_ANEXO.pdf` para búsqueda.
> Ante cualquier duda o conflicto, manda el PDF original.

---

Anexo — Generación por lotes (batched generation)
Padding a la izquierda,attention_masky recorte del prompt
Curso Modelos Avanzados de Procesamiento de Lenguaje Natural
1 Introducción
En el Micro proyecto 3, el costo de tiempo dominante no es el finetuning(que es rápido, pocos minutos
por modelo), sino la evaluación pass@1: hay que generar una solución para cada uno de los 500
problemas delsplit test, y esto se repitecuatro veces(versión original y finetuneada de cada uno de
los dos modelos). Generar las solucionesde a una(un forwardautoregresivo por problema) subutiliza
la GPU y hace que la evaluación completa tome mucho más tiempo del necesario.
Lageneración por lotes(batched generation) consiste en generar varias soluciones en paralelo,
agrupando variospromptsen un mismobatchy llamando una sola vez amodel.generate(...)por lote.
Esto acelera la evaluación considerablemente, pero introduce una dificultad que no aparece al generar
de a uno: lospromptsde un mismo lote casi nunca tienen la misma longitud, y hay que resolver ese
desajuste con cuidado en un modelodecoder-only. Este documento explica cómo se resuelve (paddinga la
izquierda,attention_mask,pad_token) y cómo se recorta la salida para quedarse solo con lo generado.
2 ¿Por qué no basta con generar de a uno?
Al generar una sola secuencia a la vez, la GPU procesa un batchde tamaño 1 en cada paso de generación:
buena parte de su capacidad de cómputo paralelo queda sin usar. Si en cambio se agrupan, por ejemplo,
16 promptsen un mismobatch, la GPU calcula las 16 continuaciones en paralelo con un costo muy similar
al de calcular una sola, lo que reduce drásticamente el tiempo total de evaluación (de horas a minutos,
dependiendo del hardware). El costo de esta ganancia es que hay que tokenizar y generar sobre tensores
rectangulares, es decir, todas las secuencias de un lote deben tener el mismo largo.
3 Padding a la izquierda en modelosdecoder-only
3.1 Idea
Como lospromptsde un lote tienen longitudes distintas, hay que rellenarlos (padding) hasta la longitud
del más largo para poder apilarlos en un solo tensor. La pregunta es de qué lado agregar ese relleno.
En un modelodecoder-only,model.generate(...)genera el siguiente token a partir del último
token de la secuencia (y su estado interno). Si elpaddingse agrega a laderecha(como es más natural,
y como se hace típicamente para tareas de clasificación), el último token de cada fila del lote sería un
token de relleno para todas las secuencias más cortas que la más larga, no el último tokenreal del prompt.
El modelo generaría a partir de esepadding, lo cual es incorrecto.
Por eso, para generación con modelosdecoder-onlyel relleno se agrega a laizquierda: de esta forma,
sin importar la longitud original de cada prompt, el último token de todas las filas del lote coincide
con el final real de cada prompt, ygeneratepuede continuar generando de forma correcta y simultánea
para todo el lote.
3.2 Implementación
Entransformers, esto se controla con el atributopadding_sidedeltokenizer:
tokenizer.padding_side = "left" # padding a la izquierda (modelos decoder-only)
Este atributo debe fijarseantesde tokenizar lospromptsdel lote; afecta solamente a cómo se rellenan
las secuencias, no al orden de los tokensreales dentro de cada una.
4attention_maskypad_token
4.1attention_mask
Agregartokensde relleno no basta por sí solo: el modelo no debeatendera esas posiciones, porque no
forman parte del promptreal y contaminarían la representación de las posiciones siguientes. Para esto,
eltokenizerdevuelve, junto coninput_ids, unaattention_maskcon un1en las posiciones reales y un
0en las posiciones de relleno.model.generate(...)usa esta máscara tanto para ignorar el relleno en
la atención como para calcular correctamente las posiciones de cadatokenreal cuando elpaddingestá a
la izquierda.
inputs = tokenizer(chunk, return_tensors="pt", padding=True,
truncation=True, max_length=MAX_LEN)
# inputs contiene tanto input_ids como attention_mask
Basta con pasarpadding=Trueal tokenizar; laattention_maskcorrespondiente se genera automática-
mente y debe pasarse junto coninput_idsagenerate(...)(al usar**inputsesto ocurre de forma
directa).
4.2pad_token
Muchos modelosdecoder-only(incluyendo Qwen2.5 y SmolLM2) se entrenan solo con untoken de fin de
secuencia (eos_token) ynodefinen untoken de relleno propio. Si se intenta tokenizar conpadding=True
sin definir uno,transformerslanza un error. La solución habitual es reutilizar eleos_tokencomo
pad_token:
if tokenizer.pad_token is None:
tokenizer.pad_token = tokenizer.eos_token
...
model.config.pad_token_id = tokenizer.pad_token_id # el modelo tambien debe conocerlo
Esto no genera ambigüedad porque, gracias a laattention_mask, el modelo nunca atiende a las posi-
ciones marcadas como relleno, sin importar quéiddetokense haya usado para rellenarlas. También
conviene fijarmodel.config.pad_token_id, para quegeneratesepa quéidusar al construir el lote de
salida.
5 Ejemplo con oraciones cortas
Para ver el efecto del padding, laattention_masky el truncamiento sobre datos reales, considere el
siguiente lote de tres oraciones de distinto tamaño (entre 2 y 5 palabras):
•"Hola mundo"(2 palabras)
•"El gato duerme mucho"(4 palabras)
•"Los perros corren rápido hoy"(5 palabras)
oraciones = ["Hola mundo", "El gato duerme mucho", "Los perros corren rapido hoy"]
tokenizer.padding_side = "left"
inputs = tokenizer(oraciones, return_tensors="pt", padding=True,
truncation=True, max_length=4) # max_length pequeno a proposito
print(inputs["input_ids"])
print(inputs["attention_mask"])
Para hacer visible el mecanismo, la siguiente tabla muestra el resultado a nivel depalabra(una tok-
enización real de sub-palabras generaría mástokenspor oración, pero el efecto del padding, la máscara
y el truncamiento sobre eltensorfinal es exactamente el mismo):
Oración input_ids(largo 4) attention_mask
"Hola mundo" [PAD, PAD, hola, mundo] [0, 0, 1, 1]
"El gato duerme mucho" [el, gato, duerme, mucho] [1, 1, 1, 1]
"Los perros corren
rápido hoy"
[los, perros, corren, rápido] [1, 1, 1, 1]
Tres cosas para notar en este ejemplo:
•La oración de 2 palabras queda condosPADa la izquierda: el último tokenreal (mundo) queda
al final de la fila, igual que en las otras dos filas.
•Laattention_maskmarca con0exactamente las posiciones dePADy con1las posiciones detokens
reales, sin importar de qué lado quedó el relleno.
•La oración de 5 palabrasse truncaa 4 (se descarta"hoy", la última palabra). Por defecto,
el truncamiento recorta desde elfinalde la secuencia (truncation_side="right"), que es un
ajusteindependientedepadding_side. En este proyecto no debería ser un problema porque los
promptsde MBPP son cortos y el max_lengthse elige holgado, pero conviene tenerlo presente: con
un max_lengthdemasiado pequeño se puede perder información al final del prompt(por ejemplo,
parte de la instrucción del problema).
6 Recorte del prompten la salida generada
model.generate(...)devuelve, por cada fila del lote, la concatenación del prompt(con su relleno a
la izquierda) y los tokensgenerados. Para quedarse solo con la solución generada hay que descartar la
parte correspondiente alprompt. Como todas las filas del lote de entrada tienen exactamente la misma
longitud (por elpadding), esta longitud es la misma para todas las filas de la salida, y el recorte se puede
hacer con una sola operación vectorizada sobre todo el lote:
out = model.generate(**inputs, max_new_tokens=max_new_tokens,
do_sample=False, # greedy / determinista
pad_token_id=tokenizer.pad_token_id)
gen = out[:, inputs["input_ids"].shape[1]:] # se recorta el prompt (con su padding)
soluciones = tokenizer.batch_decode(gen, skip_special_tokens=True)
inputs["input_ids"].shape[1]es el largo (fijo, tras elpadding) de las secuencias de entrada del lote;
todo lo quegenerateproduce después de esa posición estokennuevo.skip_special_tokens=True
elimina además elpad_token/eos_tokenque puedan quedar en la decodificación.
7 Tamaño del lote (batch_size)
El tamaño del lote es untrade-offentre velocidad y memoria: lotes más grandes generan más soluciones
en paralelo (evaluación más rápida), pero requieren más memoria de GPU (más secuencias y máspadding
en memoria a la vez); un lote demasiado grande para la GPU disponible produce un error de memoria
(out of memory). Un valor comobatch_size=16suele ser un punto de partida razonable para una
GPU de gama modesta con los tamaños de modelo y depromptde este proyecto; si ocurre un error de
memoria, basta con reducirlo.
8 Implementación completa
Uniendo lo anterior, la función de generación por lotes queda así (max_new_tokenslimita la longitud de
la solución generada, ydo_sample=Falseasegura generacióndeterminista, requerida por el protocolo
de evaluación):
@torch.no_grad()
def generate_batched(model, tokenizer, prompts, batch_size=16, max_new_tokens=256):
’’’Generacion por lotes, greedy. Padding a la IZQUIERDA (decoder-only); recorta el prompt.’’’
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
tokenizer.pad_token = tokenizer.eos_token
model.eval()
solutions = []
for i in range(0, len(prompts), batch_size):
chunk = prompts[i:i + batch_size]
inputs = tokenizer(chunk, return_tensors="pt", padding=True,
truncation=True, max_length=MAX_LEN).to(model.device)
out = model.generate(**inputs, max_new_tokens=max_new_tokens,
do_sample=False, # greedy / determinista
pad_token_id=tokenizer.pad_token_id)
gen = out[:, inputs["input_ids"].shape[1]:] # se recorta el prompt
solutions.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
return solutions
Nótese el decorador@torch.no_grad(): durante la evaluación no se necesitan gradientes, así que desac-
tivarlos reduce el consumo de memoria y acelera la generación.
9 Resumen
Elemento Por qué es necesario
padding_side = "left" Que el último tokenreal de cada fila quede alineado al final,
para poder seguir generando en todo el lote.
attention_mask Evita que el modelo atienda a las posiciones de relleno.
pad_token=eos_token Muchos modelos no traen untoken de relleno propio.
Recorte del prompt Todaslasfilasdel lotequedanconelmismolargodeentrada
tras elpadding, por lo que se puede recortar el promptde
todo el lote a la vez.
do_sample=False Genera una única solución determinista por problema,
como exige el protocolo de evaluación.
Con estos elementos, la generación por lotes produce exactamente las mismas soluciones que generar de a
una (la decodificación es determinista y elpaddingno afecta al resultado gracias a laattention_mask),
pero en una fracción del tiempo, lo cual es indispensable para poder evaluar el conjunto de prueba
completo (500 problemas) en las cuatro configuraciones del proyecto.
