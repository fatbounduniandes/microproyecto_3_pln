# %% [markdown]
# ## 2. Librerías
#
# Dejamos listo el entorno de ejecución antes de tocar los datos: instalamos las librerías,
# verificamos la GPU y la precisión mixta, montamos Drive y fijamos la semilla.

# %% [markdown]
# Instalamos las versiones de Hugging Face que usamos en el proyecto.

# %%
!pip install -q "transformers>=4.44,<5" "peft>=0.13,<1" "datasets>=2.20,<4" "accelerate>=0.33"

# %% [markdown]
# Importamos lo necesario y comprobamos las versiones que quedaron cargadas. `ast` lo usamos para
# leer el nombre y la firma de la función, `subprocess` para ejecutar el código generado y `re`
# para extraer los bloques de código del texto que devuelve el modelo.

# %%
import ast
import gc
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import warnings

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

import datasets
import peft
import transformers
from datasets import load_dataset
from google.colab import drive
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Entorno cargado")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("peft:", peft.__version__)
print("numpy:", np.__version__)

# %% [markdown]
# Detectamos la GPU y elegimos la precisión mixta: bf16 en L4 y A100, fp16 con `GradScaler` en T4.
# En la misma celda dejamos las constantes del proyecto, incluidos los nombres y las revisiones
# fijas del dataset y de los dos modelos.

# %%
SEED = 42
MAX_LEN = 512
MAX_NEW_TOKENS = 512
BATCH_SIZE = 16

DATASET_ID = "google-research-datasets/mbpp"
DATASET_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"

QWEN_ID = "Qwen/Qwen2.5-1.5B"
QWEN_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
SMOL_ID = "HuggingFaceTB/SmolLM2-1.7B"
SMOL_REVISION = "effd688a12921b4cc83e3312b6feb579f70f9c71"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()

if USE_BF16:
    AMP_DTYPE = torch.bfloat16
    NOMBRE_AMP = "bfloat16"
    USE_SCALER = False
else:
    AMP_DTYPE = torch.float16
    NOMBRE_AMP = "float16"
    USE_SCALER = True

if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name)
    print("VRAM:", round(props.total_memory / 1024 ** 3, 1), "GB")
else:
    print("No se detectó GPU: el entrenamiento en CPU va a ser muy lento")

print("Precisión mixta:", NOMBRE_AMP)
print("GradScaler:", USE_SCALER)
print("Máximo de tokens por ejemplo:", MAX_LEN)
print("Máximo de tokens nuevos al generar:", MAX_NEW_TOKENS)
print("Batch de evaluación:", BATCH_SIZE)

# %% [markdown]
# Montamos Drive y creamos las carpetas donde quedan los adaptadores y los resultados. Si la
# sesión de Colab se cae, volvemos a ejecutar y lo encontramos todo ahí.

# %%
drive.mount("/content/drive")

DRIVE_DIR = "/content/drive/MyDrive/microproyecto_3"
RESULTADOS_DIR = os.path.join(DRIVE_DIR, "resultados")
CHECKPOINT_QWEN = os.path.join(DRIVE_DIR, "checkpoint_qwen2.5_1.5b_lora")
CHECKPOINT_SMOL = os.path.join(DRIVE_DIR, "checkpoint_smollm2_1.7b_lora")

os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs(RESULTADOS_DIR, exist_ok=True)

print("Carpeta de trabajo:", DRIVE_DIR)
print("Resultados:", RESULTADOS_DIR)
print("Adaptador de Qwen:", CHECKPOINT_QWEN)
print("Adaptador de SmolLM2:", CHECKPOINT_SMOL)

if "HF_TOKEN" in os.environ:
    print("HF_TOKEN: configurado")
else:
    print("HF_TOKEN: no está en el entorno, las descargas públicas funcionan igual")

# %% [markdown]
# Fijamos la semilla para que los resultados sean reproducibles y definimos la función que libera
# la memoria de la GPU entre un modelo y el siguiente.

# %%
def fijar_semilla(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def liberar_vram(modelo=None, optimizador=None):
    if modelo is not None:
        del modelo
    if optimizador is not None:
        del optimizador
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


fijar_semilla()
print("Semilla:", SEED)
