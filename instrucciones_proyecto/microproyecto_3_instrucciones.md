# Especificación oficial del Microproyecto 3

> Texto extraído automáticamente de `PDF/Lectura_instrucciones_micro_proyecto_3_codigo.pdf` para búsqueda.
> Ante cualquier duda o conflicto, manda el PDF original.

---

Micro proyecto 3 — Generación condicional de código con
LLMs
Curso Modelos Avanzados de Procesamiento de Lenguaje Natural
1 Introducción
En este documento se describe de manera completa el micro proyecto 3 del curso, el cual comprende
el contenido de la semana 5 y 6. Además de conocer el propósito del proyecto, también se presenta la
rúbrica con la que se realizará la evaluación del mismo. El documento está estructurado de la siguiente
forma:
Contents
1 Introducción 1
2 Resultados de Aprendizaje 1
3 Descripción del micro proyecto 2
3.1 Detalles del dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
3.2 Recursos y tiempos de entrenamiento recomendados . . . . . . . . . . . . . . . . . . . . . 2
4 Rúbrica de la nota del micro proyecto 3
4.1 Evaluación de los modelos . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
4.2 Finetuning del modelo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
4.3 Resultados obtenidos . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
4.4 Cumplimiento de requisitos de la actividad . . . . . . . . . . . . . . . . . . . . . . . . . . 4
5 Entregables en Coursera 5
5.1 Estructura obligatoria del Jupyter Notebook . . . . . . . . . . . . . . . . . . . . . . . . . 5
2 Resultados de Aprendizaje
Con esta actividad se busca que el estudiante pueda poner en práctica su capacidad de ajustar un gran
modelo de lenguaje mediante finetuning a una tarea específica. Tras realizar esta actividad se espera
que el estudiante esté en capacidad de:
1. Cargar y utilizar modelos de lenguaje con miles de millones de parámetros.
2. Realizar finetuning de grandes modelos de lenguaje mediante la técnica de LoRA.
3. Procesar y crear un conjunto de datos para realizar finetuning de modelos de lenguaje para tareas
específicas.
4. Evaluar el desempeño de grandes modelos de lenguaje en tareas específicas.
3 Descripción del micro proyecto
El objetivo del micro proyecto es realizar el finetuning (mediante LoRA) de dos modelos de lenguaje
de familias distintas para resolver una tarea de generación condicional de código: dada una instrucción
en lenguaje natural que describe un problema, el modelo debe generar una función en Python que
lo resuelva. La generación de código condicionada por instrucciones es la base de los asistentes de
programación actuales (tipo Copilot).
Los dos modelos que debe usar, ambos obligatorios, son Qwen2.5-1.5B y SmolLM2-1.7B. Para cada
uno debe evaluar dos versiones: la versión original (sin entrenar,zero-shot) y la versión resultante del
finetuning con LoRA sobre el corpus de problemas de programación. En total se comparan cuatro
configuraciones.
Ambos son modelos de propósito general (no especializados en código), por lo que el finetuning
permite especializarlos en la tarea y hace que la mejora obtenida sea clara y medible. Como la métrica
de evaluación es pass@1 (correctitud funcional, que no depende del tokenizador), los dos modelos son
directamente comparables aunque pertenezcan a familias distintas. El análisis debe concluir cuál de los
dos modelos obtiene mejor desempeño en la tarea, tanto en su versión original como tras el finetuning.
3.1 Detalles del dataset
El dataset se conoce como MBPP (Mostly Basic Python Problems) y se compone de alrededor de 974
problemas básicos de programación en Python. Cada ejemplo consiste en una descripción corta en
lenguaje natural del problema, una función de referencia en Python que lo resuelve, y una lista deassert
(casos de prueba) que permiten verificar automáticamente si una solución es correcta. El dataset puede
descargarse desde Hugging Face (link) y ya viene dividido en particiones de entrenamiento, validación,
prueba yprompt.
Una ventaja de este dataset es que los contextos son cortos (la instrucción son una o dos frases y la
solución son pocas líneas), por lo que no es necesario aplicar ningún filtro adicional de longitud para que
sea tratable con recursos de cómputo modestos.
from datasets import load_dataset
ds = load_dataset("google-research-datasets/mbpp", "full")
print(ds) # splits: train (~374), test (500), validation (90), prompt (10)
ex = ds["test"][0]
print(ex["text"]) # instruccion en lenguaje natural (el problema)
print(ex["code"]) # solucion de referencia en Python
print(ex["test_list"]) # lista de asserts para evaluar la correctitud
A partir de este corpus, usted debe construir losprompts de entrada (un formato de instrucción definido
por usted, más la descripción del problema). Use las particiones provistas por MBPP con los siguientes
roles: el split train para el finetuning, elsplit validation para el monitoreo del entrenamiento y el ajuste
de hiperparámetros, y el split test reservado exclusivamente para la evaluación final del pass@1. El
formato deprompt y la preparación de los datos deben poder generarse dentro del código delnotebook
que va a entregar.
3.2 Recursos y tiempos de entrenamiento recomendados
El proyecto está diseñado para resolverse con recursos de cómputo modestos (una GPU de unos 16GB)
usando LoRA. Tenga en cuenta que el finetuning en sí es rápido; el principal costo de tiempo es la
evaluación (que implica generación y, con dos modelos, se realiza cuatro veces: la versión original y la
finetuneada de cada uno). Los dos modelos se entrenan y evalúan de a uno, liberando la GPU antes de
pasar al siguiente. Como referencia:
• Finetuning con LoRAde cada modelo (MBPPtrain, ∼374 ejemplos cortos, 3–5 épocas): aprox-
imadamente 5–15 minutos.
• Evaluación pass@1(generación + ejecución deasserts), por cada modelo: pocos minutos si se
usa generación por lotes.
Recomendaciones para acotar el tiempo: limite max_new_tokens (las soluciones de MBPP son
cortas) y use generación por lotes (batched) para acelerar la evaluación. Con estas prácticas, evaluar el
conjunto de prueba completo (500 problemas) por cada modelo es viable en pocos minutos.
4 Rúbrica de la nota del micro proyecto
La nota del micro proyecto se compone de la forma descrita en la siguiente tabla:
Item Porcentaje de la nota
Evaluación de los modelos 40%
Finetuning de los modelos 40%
Resultados obtenidos 20%
Table 1: Composición de la nota del micro proyecto
4.1 Evaluación de los modelos
El estudiante debe implementar un programa que le permita evaluar el desempeño de los modelos en el
dataset indicado. Esta evaluación debe realizarla sobre el conjunto de datos separado con este propósito
(split: test), para las cuatro configuraciones (la versión original y la finetuneada de cada uno de los dos
modelos).
La métrica a utilizar es pass@1 (correctitud funcional): para cada problema, el modelo genera una
solución, la cual se considera correcta únicamente si pasa todos losassert asociados al problema. El
pass@1 es el porcentaje de problemas resueltos correctamente sobre el total de problemas evaluados:
pass@1 = #{problemas cuya solución generada pasa todos sus tests}#{problemas evaluados}
Protocolo de evaluación (obligatorio). Para que la comparación entre las cuatro configuraciones
sea válida:
• Evalúe el pass@1 sobre el conjunto de prueba completo (los 500 problemas delsplit test), usando
el mismo conjunto para las cuatro configuraciones.
• Genere una sola solución por problema con decodificación determinista (greedy, do_sample=False),
de modo que el resultado sea reproducible.
• Use la misma configuración de generación (mismo formato deprompt, mismo max_new_tokens y
mismos criterios de parada) para la versión original y la finetuneada de cada modelo. Cada modelo
usa su propio tokenizador, pero como pass@1 mide correctitud funcional, los resultados siguen
siendo comparables entre familias.
A continuación se muestra un fragmento guía para verificar si una solución generada pasa los tests
de un problema:
def passes_tests(generated_code, test_list, setup_code=""):
# Ejecuta el codigo generado junto con sus asserts en un namespace aislado.
# ADVERTENCIA: se ejecuta codigo generado por el modelo; hagalo en un
# entorno aislado (por ejemplo, un runtime efimero o un sandbox).
env = {}
try:
exec(setup_code + "\n" + generated_code, env)
for assertion in test_list:
exec(assertion, env) # un assert que falla lanza AssertionError
return True
except Exception:
return False
# pass@1 = promedio de passes_tests(...) sobre todos los problemas evaluados
Evaluar los 500 problemas generando de a uno es lento; puede acelerar la evaluación generando varios
prompts a la vez (batched generation). Esto requiere algunos ajustes adicionales sobre lo visto en los
tutoriales (padding a la izquierda,attention_mask, recorte del prompt generado), que se explican en
detalle en el documento anexobatched_generation/generacion_por_lotes.pdf.
Dentro de este criterio, también se evaluará que se haya realizado de manera correcta la evaluación
de los modelos originales (zero-shot), de modo que la comparación con sus versiones finetuneadas sea
válida.
4.2 Finetuning del modelo
El estudiante debe realizar el finetuning de los modelos Qwen2.5-1.5B y SmolLM2-1.7B mediante LoRA,
usando las librerías vistas en los tutoriales (por ejemplotransformers y peft). Cada modelo se entrena
por separado (con su propio adaptador LoRA), liberando la GPU entre uno y otro.
Durante el entrenamiento se deben ignorar los tokens que hacen parte del prompt (la instrucción y el
contexto), con el objetivo de que el modelo aprenda a generar únicamente la solución (el código). Para
lograr esto, los labels que corresponden a estos tokens deben tener un valor de−100, de manera que
automáticamente sean ignorados por la función de costo.
from peft import LoraConfig, get_peft_model
config = LoraConfig(
r=16,
lora_alpha=16,
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
lora_dropout=0.05,
task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
4.3 Resultados obtenidos
Se evaluará la idoneidad de los hiperparámetros seleccionados para realizar el entrenamiento y de la
estrategia usada para hacer seguimiento al desempeño durante el entrenamiento, así como los resultados
obtenidos (pass@1) en el conjunto de evaluación tras realizar el finetuning, comparados con los de los
modelos originales. El análisis debe comparar las cuatro configuraciones y concluir cuál de los dos
modelos (Qwen2.5-1.5B o SmolLM2-1.7B) obtiene mejor desempeño en la tarea, así como cuánto aporta
el finetuning en cada familia.
4.4 Cumplimiento de requisitos de la actividad
Recuerde que su entrega no tendrá validez si no cumple los requisitos que se mencionan a
continuación. Por lo tanto, obtendrá una calificación de 0 en la actividad correspondiente
al micro proyecto.
Los requisitos son los siguientes:
• El dataset obligatorio es MBPP (link). No se permite cambiar de dataset ni usar datos de otras
fuentes para el entrenamiento de los modelos.
• Los modelos obligatorios son Qwen2.5-1.5B (link) y SmolLM2-1.7B (link), ambos modelos abiertos
con licencia Apache-2.0 (no requieren autenticación ni aceptación de licencia). No se permite usar
otros modelos.
• Debe implementar y entrenar los modelos haciendo uso de PyTorch, y de las librerías que se le
haya indicado únicamente (por ejemplo,transformers, peft y datasets).
• El código enviado debe poder ser ejecutado por el equipo evaluador, y el código debe ser comple-
tamente de su autoría.
• Debe entregar loscheckpoints de los dos modelos entrenados (uno por modelo) que utilizó para
realizar la evaluación, guardados mediantetorch.save o save_pretrained. Para cada modelo
puede entregar solo el adaptador LoRA, o el modelo con el adaptador LoRA fusionado (merge, por
ejemplo con merge_and_unload()); ambas opciones son válidas. Deben incluirse en el archivo zip
de entregables (ver la sección de entregables).
5 Entregables en Coursera
En la semana 6 del curso se habilitará una actividad en Coursera. En esa actividad usted no sube
archivos: debe enviar un enlace de Google Drivea un archivo comprimido (zip) que contenga los
siguientes entregables:
1. Jupyter Notebook que se utilizó para realizar la implementación y el entrenamiento de los modelos,
junto con los análisis de los resultados obtenidos.
2. Los checkpoints de las versiones finales de sus dos modelos entrenados (uno por modelo): solo el
adaptador LoRA, o el modelo con el adaptador LoRA fusionado (merge). No es necesario incluir
los modelos base originales, pues se descargan directamente de Hugging Face.
El archivo zip debe estar compartido en Google Drive con el permisoCualquier persona que tenga el
vínculo (lector). Verifique, desde una ventana de incógnito, que el enlace permite descargar el archivo
sin pedir acceso, y no modifique ni elimine el archivo hasta que finalice el proceso de calificación. Un
enlace que no se pueda abrir equivale a no entregar.
5.1 Estructura obligatoria del Jupyter Notebook
El Jupyter Notebook debe organizarseobligatoriamente en las siguientes secciones, en este
mismo orden y con estos mismos títulos. El notebook debe ser autocontenido y ejecutable de
principio a fin (Run All). Entre paréntesis se indica el ítem de la rúbrica al que contribuye cada sección:
1. Portada e introducción. Nombre del estudiante, resumen del proyecto y descripción de las
cuatro configuraciones que se compararán (Qwen2.5-1.5B y SmolLM2-1.7B, cada uno en versión
zero-shot y tras el finetuning con LoRA).
2. Librerías. Importación de todas las dependencias utilizadas.
3. Carga del dataset y exploración.Carga de MBPP, descripción de sus particiones y ejemplos
del contenido (instrucción, código de referencia ytest_list).
4. Construcción de prompts y preprocesamiento. Formato de instrucción definido por usted,
tokenizaciónyenmascaramientodelos labels del prompt con −100paraquelapérdidasoloconsidere
la solución. (Finetuning del modelo)
5. Particiones (train / validation / test). Asignación de roles: train para el finetuning,
validation para el monitoreo,test reservado para la evaluación final.
6. Implementación de la evaluación pass@1.Función de generación (con generación por lotes) y
de verificación de losassert en un entorno aislado, siguiendo el protocolo de evaluación obligatorio
(greedy, mismo conjunto y misma configuración para todas las configuraciones).(Evaluación de los
modelos)
7. Evaluación de los modelos base (zero-shot). pass@1 de Qwen2.5-1.5B y de SmolLM2-1.7B
originales sobre el conjunto de prueba.(Evaluación de los modelos)
8. Finetuning con LoRA. Para cada modelo: configuración de LoRA, bucle de entrenamiento,
selección de hiperparámetros y seguimiento del desempeño. Los modelos se entrenan de a uno,
liberando la GPU entre ambos.(Finetuning de los modelos)
9. Evaluación de los modelos finetuneados.pass@1 de cada modelo con LoRA sobre el mismo
conjunto y con la misma configuración que su versión base.(Evaluación de los modelos)
10. Comparación y resultados. Comparación del pass@1 de las cuatro configuraciones, con su
análisis (incluyendo algunos ejemplos cualitativos de soluciones generadas) y una conclusión sobre
cuál de los dos modelos es mejor en la tarea.(Resultados obtenidos)
11. Conclusiones. Discusión de los resultados, del seguimiento del entrenamiento y de las limitaciones
de los modelos.
