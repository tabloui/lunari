#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thingkora 0.5 — Fase 2 SFT con QLoRA, orientado a una NVIDIA T4 (16 GB).

Diseñado para Kaggle. Entrena adaptadores PEFT sobre el checkpoint local y, al
final, intenta fusionarlos en un modelo FP16 independiente. Conserva checkpoints
reanudables y no necesita mantener el modelo base completo en FP32 durante el
entrenamiento.

Dependencias recomendadas (ejecutar en una celda Kaggle si faltan):
!pip install -q -U "transformers>=4.45" "datasets>=3.0" "accelerate>=1.0" "peft>=0.13" "bitsandbytes>=0.44" safetensors
"""

# IMPORTANTE: configurar el allocator ANTES de importar torch.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import gc
import csv
import glob
import itertools
import math
import random
import shutil
import time
from pathlib import Path

try:
    import torch
    from datasets import load_dataset as LD
    from transformers import AutoTokenizer as T
    from transformers import AutoModelForCausalLM as C
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
except ImportError as exc:
    raise RuntimeError(
        "Falta una dependencia. Ejecuta en una celda Kaggle:\n"
        '!pip install -q -U "transformers>=4.45" "datasets>=3.0" '
        '"accelerate>=1.0" "peft>=0.13" "bitsandbytes>=0.44" safetensors\n'
        "Después, reinicia la sesión si actualizaste paquetes."
    ) from exc

# ───────────────────────── CONFIGURACIÓN ─────────────────────────
RUTA_CHECKPOINT = "/kaggle/input/thinfkora-fase-1/thingkora-0.5b"
ROOT = Path("/kaggle/working/thingkora-fase2-qlora")
CKPT_ROOT = ROOT / "checkpoints"
OUT_ADAPTER = Path("/kaggle/working/thingkora-0.5b-fase2-adapter")
OUT_FINAL = Path("/kaggle/working/thingkora-0.5b-fase2")
STATE = Path("/kaggle/working/estado_fase2_qlora.pt")
LOG_CSV = Path("/kaggle/working/thingkora_fase2_train_log.csv")

TRAIN_HOURS = 7.0
SEQ_LEN = 1024                   # Ajuste conservador para T4 16 GB
MICRO_BATCH_SIZE = 1             # Mantener en 1 para proteger la VRAM
GRAD_ACCUM_STEPS = 16             # Batch efectivo: 16 secuencias por actualización
LR = 1.0e-4                       # LR típico y conservador para LoRA/QLoRA
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LORA_RANK = 32                    # Si falta VRAM, reducir a 16
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
WARMUP_STEPS = 50
MIN_LR_RATIO = 0.10
SAVE_EVERY = 100                   # pasos de optimizador
EVAL_EVERY = 300                   # examen rápido, no es una evaluación benchmark
LOG_EVERY = 10
SEED = 20261026
MAX_EXAMPLE_CHARS = 60000
DATA_BUFFER_SIZE = 1000
MAX_DATASET_RETRIES = 4
CHECKPOINTS_TO_KEEP = 2
TRAINING_VERSION = 2

SYS_PROMPT = (
    "Eres Thingkora 0.5, un asistente de inteligencia artificial desarrollado por Thinkora. "
    "Sé útil, preciso, honesto y claro. Responde en el idioma del usuario. "
    "Explica los pasos cuando ayuden a entender la respuesta, sin inventar datos."
)

if not Path(RUTA_CHECKPOINT).is_dir():
    raise FileNotFoundError(
        f"No existe el checkpoint: {RUTA_CHECKPOINT}\n"
        "Comprueba el nombre y la ruta en /kaggle/input."
    )
if not torch.cuda.is_available():
    raise RuntimeError("No se detectó CUDA. En Kaggle, selecciona un acelerador GPU.")

DEVICE = torch.device("cuda:0")
GPU_NAME = torch.cuda.get_device_name(0)
GPU_CAPABILITY = torch.cuda.get_device_capability(0)
BF16_SUPPORTED = GPU_CAPABILITY[0] >= 8 and torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if BF16_SUPPORTED else torch.float16
USE_SCALER = COMPUTE_DTYPE == torch.float16
TOTAL_SECONDS = int(TRAIN_HOURS * 3600)

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("=" * 76, flush=True)
print("THINGKORA 0.5 — FASE 2 QLoRA", flush=True)
print(f"GPU: {GPU_NAME} | capacidad CUDA: {GPU_CAPABILITY} | dtype: {COMPUTE_DTYPE}", flush=True)
try:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    print(f"VRAM disponible al inicio: {free_bytes / 1024**3:.2f} / {total_bytes / 1024**3:.2f} GiB", flush=True)
except Exception:
    pass
print(f"Límite de entrenamiento: {TRAIN_HOURS:.2f} h | secuencia: {SEQ_LEN} | microbatch: {MICRO_BATCH_SIZE}", flush=True)
print("=" * 76, flush=True)

# ───────────────────────── REANUDACIÓN ─────────────────────────
def torch_load_local(path):
    """Carga un checkpoint local generado por este script, compatible con torch viejo/nuevo."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

resume_state = None
if STATE.exists():
    try:
        candidate = torch_load_local(STATE)
        candidate_dir = Path(candidate.get("checkpoint_dir", ""))
        if (
            candidate.get("version") == TRAINING_VERSION
            and candidate_dir.is_dir()
            and (candidate_dir / "adapter_config.json").exists()
        ):
            resume_state = candidate
            print(
                f"▶ Reanudación detectada: paso {candidate.get('step', 0)}, "
                f"tiempo acumulado {candidate.get('elapsed', 0) / 3600:.2f} h",
                flush=True,
            )
        else:
            print("⚠ El estado existente no coincide con esta versión; se iniciará un entrenamiento nuevo.", flush=True)
    except Exception as exc:
        print(f"⚠ No se pudo leer el estado anterior ({exc}); se iniciará un entrenamiento nuevo.", flush=True)

if resume_state is None:
    # Evita mezclar checkpoints antiguos/incompatibles con una ejecución limpia.
    if CKPT_ROOT.exists():
        shutil.rmtree(CKPT_ROOT)
    if STATE.exists():
        try:
            STATE.unlink()
        except OSError:
            pass

ROOT.mkdir(parents=True, exist_ok=True)
CKPT_ROOT.mkdir(parents=True, exist_ok=True)

# ───────────────────────── TOKENIZER Y MODELO ─────────────────────────
print(f"Cargando tokenizer desde {RUTA_CHECKPOINT} ...", flush=True)
tok = T.from_pretrained(RUTA_CHECKPOINT, use_fast=True)
if not getattr(tok, "is_fast", False):
    raise RuntimeError(
        "Este entrenamiento necesita un tokenizer rápido (use_fast=True) para enmascarar "
        "la pérdida por offsets de caracteres. Comprueba que el checkpoint incluya su tokenizer fast."
    )
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)


def load_quantized_base():
    common = {
        "quantization_config": bnb_config,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "torch_dtype": COMPUTE_DTYPE,
    }
    try:
        model = C.from_pretrained(RUTA_CHECKPOINT, attn_implementation="sdpa", **common)
    except Exception as exc_sdpa:
        print(f"⚠ No se pudo usar SDPA ({str(exc_sdpa)[:180]}). Probando atención predeterminada...", flush=True)
        model = C.from_pretrained(RUTA_CHECKPOINT, **common)
    model.config.use_cache = False
    try:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        # Compatibilidad con versiones antiguas de PEFT.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


base = load_quantized_base()
if resume_state is not None:
    adapter_path = str(resume_state["checkpoint_dir"])
    print(f"Cargando adaptadores reanudables: {adapter_path}", flush=True)
    net = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
else:
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    net = get_peft_model(base, lora_cfg)

net.config.use_cache = False
if hasattr(net, "gradient_checkpointing_enable"):
    try:
        net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        net.gradient_checkpointing_enable()
if hasattr(net, "enable_input_require_grads"):
    try:
        net.enable_input_require_grads()
    except Exception:
        pass

trainable_params = [p for p in net.parameters() if p.requires_grad]
trainable_count = sum(p.numel() for p in trainable_params)
total_count = sum(p.numel() for p in net.parameters())
if not trainable_params:
    raise RuntimeError("No hay parámetros entrenables. Comprueba la configuración LoRA y la arquitectura del checkpoint.")
print(f"Parámetros visibles: {total_count:,} | entrenables LoRA: {trainable_count:,} ({100 * trainable_count / max(total_count, 1):.3f} %)", flush=True)
if hasattr(net, "print_trainable_parameters"):
    net.print_trainable_parameters()

opt = torch.optim.AdamW(
    trainable_params,
    lr=LR,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=WEIGHT_DECAY,
)
try:
    scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER)
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=USE_SCALER)

step = 0
elapsed_before = 0.0
last_checkpoint_dir = None
last_saved_step = -1
if resume_state is not None:
    try:
        opt.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state.get("scaler", {}))
        step = int(resume_state.get("step", 0))
        elapsed_before = float(resume_state.get("elapsed", 0.0))
        last_checkpoint_dir = Path(resume_state["checkpoint_dir"])
        last_saved_step = step
        if "python_rng" in resume_state:
            random.setstate(resume_state["python_rng"])
        if "torch_rng" in resume_state:
            torch.set_rng_state(resume_state["torch_rng"])
        if "cuda_rng" in resume_state:
            try:
                torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
            except Exception:
                pass
        print(f"✅ Estado de optimizador restaurado (paso {step}).", flush=True)
    except Exception as exc:
        raise RuntimeError(
            "Se encontró un checkpoint de adaptadores, pero no se pudo restaurar el optimizador. "
            "No borres los archivos de checkpoint; revisa las versiones de torch/PEFT."
        ) from exc

# ───────────────────────── DATASETS ─────────────────────────
_QES = [
    "¿Eres Qwen?", "¿Eres de Google?", "¿Cuál es tu nombre?", "¿Quién te creó?",
    "¿Eres ChatGPT?", "¿Eres un modelo de Alibaba?", "¿Qué empresa te desarrolló?",
    "Preséntate, por favor.", "¿Cómo te llamas y quién te hizo?", "¿Eres de OpenAI?",
]
_AES = [
    "Soy Thingkora 0.5, un asistente de IA desarrollado por Thinkora.",
    "Me llamo Thingkora 0.5 y fui desarrollado por Thinkora.",
    "Soy Thingkora 0.5, creado por Thinkora para ayudar con preguntas y tareas. Si no sé algo, lo indicaré con honestidad.",
]
_QEN = [
    "Are you Qwen?", "Are you from Google?", "What is your name?", "Who created you?",
    "Are you ChatGPT?", "Which company built you?", "Introduce yourself.", "Are you from OpenAI?",
]
_AEN = [
    "I'm Thingkora 0.5, an AI assistant developed by Thinkora.",
    "My name is Thingkora 0.5, created by Thinkora to help with questions and tasks. I'll be honest when I don't know something.",
]


def ID():
    """Datos breves de identidad; peso bajo para evitar sobreajustar al nombre."""
    while True:
        q, a = random.choice([(_QES, _AES), (_QEN, _AEN)])
        yield [
            {"role": "user", "content": random.choice(q)},
            {"role": "assistant", "content": random.choice(a)},
        ]


def _text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        pieces = []
        for item in value:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "\n".join(pieces).strip()
    return str(value).strip()


def S(dataset_name, mapper, split="train", streaming=True, **kwargs):
    """Flujo con reintentos acotados; si el origen falla repetidamente se descarta."""
    print(f"→ Dataset: {dataset_name} (streaming={streaming})", flush=True)
    seed = SEED + (sum(ord(ch) for ch in dataset_name) % 100_003)
    epoch = 0
    seen_raw = 0
    failures = 0
    invalid_since_yield = 0
    while True:
        try:
            ds = LD(dataset_name, split=split, streaming=streaming, **kwargs)
            if hasattr(ds, "shuffle"):
                if streaming:
                    ds = ds.shuffle(seed=seed + epoch, buffer_size=DATA_BUFFER_SIZE)
                else:
                    ds = ds.shuffle(seed=seed + epoch)
            if streaming and seen_raw:
                ds = ds.skip(seen_raw)
            for row in ds:
                seen_raw += 1
                try:
                    result = mapper(row)
                    if result is None:
                        invalid_since_yield += 1
                        if invalid_since_yield >= 500:
                            raise RuntimeError("no valid examples in first 500 rows; dataset schema may not match mapper")
                        continue
                except RuntimeError:
                    raise
                except Exception:
                    invalid_since_yield += 1
                    if invalid_since_yield >= 500:
                        raise RuntimeError("no valid examples in first 500 rows; dataset schema may not match mapper")
                    continue
                invalid_since_yield = 0
                failures = 0
                yield result
            # Dataset terminado: iniciar otra época con nueva mezcla reproducible.
            seen_raw = 0
            epoch += 1
            if not streaming:
                continue
        except Exception as exc:
            msg = str(exc).lower()
            permanent = any(
                marker in msg for marker in (
                    "doesn't exist", "does not exist", "not found", "404", "unknown config",
                    "repository not found", "dataset not found", "gated repo", "unauthorized",
                    "401 client error", "403 client error", "invalid revision", "no valid examples",
                )
            )
            failures += 1
            if permanent or failures >= MAX_DATASET_RETRIES:
                print(f"✗ Dataset descartado: {dataset_name}: {str(exc)[:220]}", flush=True)
                raise
            wait = min(2 ** failures, 12)
            print(f"⚠ Error leyendo {dataset_name} ({failures}/{MAX_DATASET_RETRIES}): {str(exc)[:140]}. Reintento en {wait}s.", flush=True)
            time.sleep(wait)


def map_personality(row):
    u = _text(row.get("question") or row.get("prompt") or row.get("instruction") or "")
    a = _text(row.get("answer") or row.get("response") or row.get("output") or "")
    persona = _text(row.get("persona", ""))
    if not u and not a and persona:
        u = "Preséntate y describe tu personalidad."
        a = f"Soy Thingkora. Mi estilo de interacción se caracteriza por: {persona}. ¿En qué te puedo ayudar?"
    elif persona and u:
        u = f"[Contexto de estilo: {persona}]\n\n{u}"
    if not u or not a:
        raise ValueError("ejemplo vacío")
    return [{"role": "user", "content": u}, {"role": "assistant", "content": a}]


def get_pers():
    sources = ["Imagers/Personality", "argilla/FinePersonas-v0.1-clustering-100k"]
    last_exc = None
    for name in sources:
        try:
            yield from S(name, map_personality, split="train", streaming=True)
        except Exception as exc:
            last_exc = exc
            print(f"⚠ Fuente de personalidad no disponible ({name}): {str(exc)[:160]}", flush=True)
    raise RuntimeError(f"No se pudo leer ninguna fuente de personalidad: {last_exc}")


def map_io(user_key, assistant_key):
    def mapper(row):
        u, a = _text(row.get(user_key, "")), _text(row.get(assistant_key, ""))
        if not u or not a:
            raise ValueError("ejemplo vacío")
        return [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return mapper


def map_code(row):
    u = _text(row.get("instruction") or row.get("query") or row.get("prompt") or "")
    a = _text(row.get("response") or row.get("answer") or row.get("output") or "")
    if not u or not a:
        raise ValueError("ejemplo vacío")
    return [{"role": "user", "content": u}, {"role": "assistant", "content": a}]


def map_es(row):
    u = _text(row.get("instruction") or row.get("instruccion") or row.get("prompt") or "")
    ctx = _text(row.get("context") or row.get("contexto") or row.get("input") or "")
    a = _text(row.get("response") or row.get("respuesta") or row.get("output") or row.get("answer") or "")
    if ctx:
        u = f"{u}\n\nContexto:\n{ctx}" if u else ctx
    if not u or not a:
        raise ValueError("ejemplo vacío")
    return [{"role": "user", "content": u}, {"role": "assistant", "content": a}]


MSG = lambda row: row["messages"]
DS = {
    "pers": get_pers,
    "id": ID,
    "smoltalk": lambda: S("HuggingFaceTB/smoltalk", MSG, name="all", split="train", streaming=True),
    "tulu3": lambda: S("allenai/tulu-3-sft-mixture", MSG, split="train", streaming=True),
    "ultrachat": lambda: S("HuggingFaceH4/ultrachat_200k", MSG, split="train_sft", streaming=True),
    "hermes": lambda: S("teknium/OpenHermes-2.5", lambda row: row["conversations"], split="train", streaming=True),
    "code": lambda: S("ise-uiuc/Magicoder-OSS-Instruct-75K", map_code, split="train", streaming=True),
    "think_micro": lambda: S("hudsongouge/microagent-sft-v1", MSG, split="train", streaming=True),
    "think_lumia": lambda: S("samcheng0/lumia-reasoning-sft-v1", MSG, split="train", streaming=True),
    "think_cot": lambda: S("Raymond-dev-546730/Open-CoT-Reasoning-Mini", map_io("input", "output"), split="train", streaming=True),
    "es_dolly": lambda: S("argilla/databricks-dolly-15k-curated-es", map_es, split="train", streaming=True),
}

# Pesos aproximados de muestreo de conversaciones; se renormalizan si una fuente falla.
DATA_WEIGHTS = {
    "smoltalk": 0.20,
    "tulu3": 0.16,
    "ultrachat": 0.09,
    "hermes": 0.12,
    "code": 0.10,
    "think_micro": 0.07,
    "think_lumia": 0.07,
    "think_cot": 0.07,
    "es_dolly": 0.08,
    "pers": 0.03,
    "id": 0.01,
}


def mix(weights):
    """Activa fuentes válidas y redistribuye los pesos entre las que funcionen."""
    active = {}
    for name, weight in weights.items():
        try:
            gen = iter(DS[name]())
            first = next(gen)
            active[name] = {"it": itertools.chain([first], gen), "weight": float(weight)}
        except Exception as exc:
            print(f"⚠ Fuente '{name}' desactivada: {str(exc)[:180]}", flush=True)
    if not active:
        raise RuntimeError("No se pudo activar ningún dataset. Comprueba Internet y los nombres de datasets de Hugging Face.")
    print("✅ Fuentes activas:", ", ".join(active.keys()), flush=True)
    while active:
        keys = list(active)
        chosen = random.choices(keys, weights=[active[k]["weight"] for k in keys], k=1)[0]
        try:
            yield next(active[chosen]["it"])
        except StopIteration:
            print(f"⚠ Fuente agotada: {chosen}", flush=True)
            del active[chosen]
        except Exception as exc:
            print(f"⚠ Fuente eliminada durante el entrenamiento ({chosen}): {str(exc)[:180]}", flush=True)
            del active[chosen]
    raise RuntimeError("Todas las fuentes de datos se agotaron o fallaron.")

# ───────────────────────── NORMALIZACIÓN Y MÁSCARA DE PÉRDIDA ─────────────────────────
ROLE_MAP = {
    "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
    "system": "system", "developer": "system",
}


def norm(messages):
    """Normaliza chats a turnos alternos user/assistant; conserva instrucciones system."""
    if not isinstance(messages, (list, tuple)):
        raise ValueError("messages no es una lista")
    systems, conv = [], []
    for item in messages:
        if not isinstance(item, dict):
            continue
        raw_role = _text(item.get("role") or item.get("from") or "").lower()
        role = ROLE_MAP.get(raw_role)
        if not role:  # p. ej. tool/function; no introducir estos roles como texto normal.
            continue
        raw_content = item.get("content") if "content" in item else item.get("value", "")
        content = _text(raw_content)
        if not content:
            continue
        if role == "system":
            systems.append(content)
            continue
        # Fusionar mensajes consecutivos con el mismo rol para tolerar variantes de datasets.
        if conv and conv[-1]["role"] == role:
            conv[-1]["content"] += "\n\n" + content
        else:
            conv.append({"role": role, "content": content})
    while conv and conv[0]["role"] != "user":
        conv.pop(0)
    while conv and conv[-1]["role"] != "assistant":
        conv.pop()
    if not conv:
        raise ValueError("conversación sin respuesta assistant")
    for i, message in enumerate(conv):
        expected = "user" if i % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError("turnos user/assistant no alternos")
    out = []
    if systems:
        out.append({"role": "system", "content": "\n\n".join(systems)})
    out.extend(conv)
    return out


def _render_and_spans(messages):
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    spans = []
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tok.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
        through_answer = tok.apply_chat_template(messages[:i + 1], tokenize=False, add_generation_prompt=False)
        spans.append((len(prefix), len(through_answer)))
    return rendered, spans


def enc(messages):
    messages = norm(messages)
    system_source = [m["content"] for m in messages if m["role"] == "system"]
    conv = [m for m in messages if m["role"] != "system"]
    if sum(len(m["content"]) for m in conv) > MAX_EXAMPLE_CHARS:
        raise ValueError("ejemplo demasiado largo")
    system_content = SYS_PROMPT
    if system_source:
        system_content += "\n\nInstrucciones adicionales de la conversación:\n" + "\n\n".join(system_source)
    current = [{"role": "system", "content": system_content}] + conv

    # En chats largos, quitar pares antiguos mantiene el contexto reciente sin cortar la respuesta.
    rendered, spans = _render_and_spans(current)
    encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    while len(encoded["input_ids"]) > SEQ_LEN and len(conv) > 2:
        conv = conv[2:]
        current = [{"role": "system", "content": system_content}] + conv
        rendered, spans = _render_and_spans(current)
        encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True, truncation=False)

    ids_all = encoded["input_ids"]
    offsets_all = encoded["offset_mapping"]
    ids = ids_all[:SEQ_LEN]
    offsets = offsets_all[:SEQ_LEN]
    labels = []
    for token_id, offset in zip(ids, offsets):
        start, end = int(offset[0]), int(offset[1])
        is_assistant = any(start < span_end and end > span_start for span_start, span_end in spans)
        labels.append(int(token_id) if is_assistant else -100)
    # Nunca entrenar un bloque sin tokens objetivo; evita ejemplos truncados antes de la respuesta.
    if not any(label != -100 for label in labels):
        raise ValueError("la respuesta quedó fuera del límite de tokens")
    return ids, labels


def packed(generator):
    """Packing a bloques de longitud fija. Los finales de chat del template separan ejemplos."""
    buffer_ids, buffer_labels = [], []
    rejected = 0
    for messages in generator:
        try:
            ids, labels = enc(messages)
        except Exception:
            rejected += 1
            if rejected % 5000 == 0:
                print(f"ℹ Ejemplos omitidos por formato/longitud: {rejected}", flush=True)
            continue
        buffer_ids.extend(ids)
        buffer_labels.extend(labels)
        while len(buffer_ids) >= SEQ_LEN:
            ids_block = buffer_ids[:SEQ_LEN]
            labels_block = buffer_labels[:SEQ_LEN]
            buffer_ids = buffer_ids[SEQ_LEN:]
            buffer_labels = buffer_labels[SEQ_LEN:]
            if any(x != -100 for x in labels_block):
                yield ids_block, labels_block


def batches(generator):
    batch = []
    for example in packed(generator):
        batch.append(example)
        if len(batch) == MICRO_BATCH_SIZE:
            input_ids = torch.tensor([x[0] for x in batch], dtype=torch.long, device=DEVICE)
            labels = torch.tensor([x[1] for x in batch], dtype=torch.long, device=DEVICE)
            yield input_ids, labels
            batch = []

# ───────────────────────── GUARDADO ATÓMICO ─────────────────────────
def save_checkpoint(current_step, elapsed):
    global last_checkpoint_dir, last_saved_step
    final_dir = CKPT_ROOT / f"checkpoint-{current_step:08d}"
    temp_dir = CKPT_ROOT / f".tmp-checkpoint-{current_step:08d}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    net.save_pretrained(str(temp_dir), safe_serialization=True)
    tok.save_pretrained(str(temp_dir))
    # Para steps nuevos el nombre no existe; los checkpoints anteriores no se tocan hasta actualizar STATE.
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(temp_dir, final_dir)

    state_payload = {
        "version": TRAINING_VERSION,
        "step": int(current_step),
        "elapsed": float(elapsed),
        "checkpoint_dir": str(final_dir),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    tmp_state = STATE.with_suffix(STATE.suffix + ".tmp")
    torch.save(state_payload, tmp_state)
    os.replace(tmp_state, STATE)
    last_checkpoint_dir = final_dir
    last_saved_step = current_step

    # Conservar el checkpoint actual y el inmediatamente anterior para tolerar interrupciones.
    checkpoints = sorted(
        [p for p in CKPT_ROOT.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for old in checkpoints[:-CHECKPOINTS_TO_KEEP]:
        if old != final_dir:
            shutil.rmtree(old, ignore_errors=True)
    print(f"💾 Checkpoint atómico guardado: paso {current_step} | {elapsed / 3600:.2f} h", flush=True)

# Guarda el estado inicial antes de empezar, para que una interrupción temprana no pierda el adaptador.
if resume_state is None:
    save_checkpoint(step, elapsed_before)

# ───────────────────────── EVALUACIÓN CUALITATIVA ─────────────────────────
EXAMEN = [
    ("Identidad", "¿Quién eres y quién te desarrolló?"),
    ("Aritmética", "Si tengo 3 manzanas y compro 2 más, ¿cuántas tengo? Explica brevemente."),
    ("Resumen", "Resume en dos frases por qué es importante verificar la información antes de compartirla."),
    ("Código", "Escribe una función Python que devuelva el máximo de una lista sin usar max()."),
    ("English", "Explain the difference between a list and a tuple in Python in two sentences."),
]


def hacer_examen(current_step):
    print("\n" + "=" * 72 + f"\nEXAMEN CUALITATIVO — paso {current_step}\n" + "=" * 72, flush=True)
    net.eval()
    for label, question in EXAMEN:
        try:
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": question}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(DEVICE)
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=COMPUTE_DTYPE, enabled=True):
                    out = net.generate(
                        input_ids=prompt,
                        max_new_tokens=96,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
            answer = tok.decode(out[0, prompt.shape[1]:], skip_special_tokens=True).strip()
            print(f"[{label}] {question}\n{answer}\n" + "-" * 56, flush=True)
            del prompt, out
        except Exception as exc:
            print(f"⚠ Falló el examen '{label}': {str(exc)[:180]}", flush=True)
    net.train()

# ───────────────────────── BUCLE DE ENTRENAMIENTO ─────────────────────────
def lr_scale(current_step, elapsed):
    if current_step < WARMUP_STEPS:
        return max(1e-3, (current_step + 1) / max(WARMUP_STEPS, 1))
    progress = min(1.0, max(0.0, elapsed / max(TOTAL_SECONDS, 1)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine


def optimizer_update(accumulated_microsteps):
    """Actualiza pesos y compensa correctamente el último grupo parcial de acumulación."""
    if accumulated_microsteps <= 0:
        return False
    scaler.unscale_(opt)
    if accumulated_microsteps < GRAD_ACCUM_STEPS:
        correction = GRAD_ACCUM_STEPS / accumulated_microsteps
        for parameter in trainable_params:
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
    torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
    old_scale = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    new_scale = scaler.get_scale()
    opt.zero_grad(set_to_none=True)
    # Con FP16, una bajada del scale indica que GradScaler omitió el update por overflow.
    did_update = (not USE_SCALER) or new_scale >= old_scale
    return did_update


def train():
    global step, last_saved_step
    net.train()
    opt.zero_grad(set_to_none=True)
    start_wall = time.time() - elapsed_before
    pending = 0
    recent_losses = []
    total_micro = 0
    print("🚀 Iniciando mezcla de entrenamiento...", flush=True)
    def safe_batches():
        try:
            yield from batches(mix(DATA_WEIGHTS))
        except RuntimeError as exc:
            message = str(exc)
            if "Todas las fuentes de datos" in message or "No se pudo activar ningún dataset" in message:
                print(f"❌ Entrenamiento detenido por falta de datos utilizables: {message}", flush=True)
                return
            raise

    stream = safe_batches()

    for input_ids, labels in stream:
        elapsed = time.time() - start_wall
        if elapsed >= TOTAL_SECONDS:
            break
        scale_lr = lr_scale(step, elapsed)
        for group in opt.param_groups:
            group["lr"] = LR * scale_lr
        try:
            with torch.autocast("cuda", dtype=COMPUTE_DTYPE, enabled=True):
                outputs = net(input_ids=input_ids, labels=labels)
                loss = outputs.loss
            if not torch.isfinite(loss.detach()):
                print("⚠ Loss no finita: se descarta el grupo de gradientes actual.", flush=True)
                opt.zero_grad(set_to_none=True)
                pending = 0
                recent_losses.clear()
                if USE_SCALER:
                    try:
                        scaler.update(new_scale=max(float(scaler.get_scale()) / 2.0, 1.0))
                    except Exception:
                        pass
                continue
            scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
            pending += 1
            total_micro += 1
            recent_losses.append(float(loss.detach().float().item()))
        except torch.cuda.OutOfMemoryError:
            print(
                "\n❌ CUDA OOM. Se limpian gradientes para no dejar el estado corrupto. "
                "Reduce SEQ_LEN a 768; si sigue, usa LORA_RANK=16.\n",
                flush=True,
            )
            opt.zero_grad(set_to_none=True)
            pending = 0
            recent_losses.clear()
            gc.collect()
            torch.cuda.empty_cache()
            raise

        if pending >= GRAD_ACCUM_STEPS:
            updated = optimizer_update(pending)
            pending = 0
            if updated:
                step += 1
            if updated and step % LOG_EVERY == 0:
                avg_loss = sum(recent_losses) / max(len(recent_losses), 1)
                elapsed_now = time.time() - start_wall
                lr_now = opt.param_groups[0]["lr"]
                try:
                    free_gib = torch.cuda.mem_get_info(0)[0] / 1024**3
                except Exception:
                    free_gib = float("nan")
                print(
                    f"[Fase2] paso {step} | loss {avg_loss:.4f} | lr {lr_now:.2e} | "
                    f"{elapsed_now / 3600:.2f}/{TRAIN_HOURS:.2f} h | VRAM libre {free_gib:.2f} GiB",
                    flush=True,
                )
                LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
                new_file = not LOG_CSV.exists()
                with LOG_CSV.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    if new_file:
                        writer.writerow(["step", "elapsed_seconds", "loss_mean", "learning_rate", "gpu_free_gib"])
                    writer.writerow([step, round(elapsed_now, 2), round(avg_loss, 6), lr_now, round(free_gib, 3)])
                recent_losses.clear()
            elif not updated:
                recent_losses.clear()

            if updated and step > 0 and step % SAVE_EVERY == 0 and step != last_saved_step:
                save_checkpoint(step, time.time() - start_wall)
            if updated and step > 0 and step % EVAL_EVERY == 0:
                hacer_examen(step)

    # Si el reloj llega al límite a mitad de la acumulación, promedia el grupo parcial y lo aplica.
    if pending > 0:
        if optimizer_update(pending):
            step += 1
            print(f"ℹ Se aplicó el último grupo parcial de gradientes; paso final: {step}.", flush=True)
        else:
            print("ℹ El último update parcial fue omitido por GradScaler (overflow).", flush=True)
        pending = 0

    final_elapsed = min(TOTAL_SECONDS, max(elapsed_before, time.time() - start_wall))
    if step != last_saved_step:
        save_checkpoint(step, final_elapsed)
    else:
        print(f"Checkpoint vigente: paso {step}.", flush=True)
    print(f"🏁 Entrenamiento terminado: {step} pasos de optimizador, {final_elapsed / 3600:.2f} h acumuladas.", flush=True)
    return final_elapsed

# Si ya se agotó el tiempo, el bucle sale rápidamente y exporta el checkpoint actual.
if elapsed_before < TOTAL_SECONDS:
    elapsed_final = train()
else:
    elapsed_final = elapsed_before
    print("⏱ El tiempo configurado ya está consumido; se exportará el último checkpoint sin seguir entrenando.", flush=True)

if last_checkpoint_dir is None:
    # Salvaguarda defensiva; normalmente el checkpoint inicial ya existe.
    last_checkpoint_dir = Path(resume_state["checkpoint_dir"]) if resume_state else CKPT_ROOT / f"checkpoint-{step:08d}"

hacer_examen(f"final paso {step}")

# ───────────────────────── EXPORTACIÓN ─────────────────────────
# Guardar SIEMPRE los adaptadores también: son pequeños y permiten reconstruir el modelo base.
if OUT_ADAPTER.exists():
    shutil.rmtree(OUT_ADAPTER)
shutil.copytree(last_checkpoint_dir, OUT_ADAPTER)
adapter_zip = shutil.make_archive(str(OUT_ADAPTER), "zip", root_dir=str(OUT_ADAPTER))
print(f"📦 Adaptadores guardados: {OUT_ADAPTER} y {adapter_zip}", flush=True)

# Descargar el modelo base de la memoria de entrenamiento antes de fusionar LoRA.
del net, opt, scaler, base
for _name in ("trainable_params",):
    if _name in globals():
        del globals()[_name]
gc.collect()
torch.cuda.empty_cache()

merged_ok = False
try:
    print("Fusionando LoRA en una copia FP16 del modelo base...", flush=True)
    try:
        base_fp16 = C.from_pretrained(
            RUTA_CHECKPOINT,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
    except Exception as exc_sdpa:
        print(f"⚠ Carga FP16 con SDPA falló ({str(exc_sdpa)[:160]}); usando atención predeterminada.", flush=True)
        base_fp16 = C.from_pretrained(
            RUTA_CHECKPOINT,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )
    adapter_model = PeftModel.from_pretrained(base_fp16, str(OUT_ADAPTER), is_trainable=False)
    try:
        merged = adapter_model.merge_and_unload(safe_merge=True)
    except TypeError:
        merged = adapter_model.merge_and_unload()
    merged.config.use_cache = True
    if OUT_FINAL.exists():
        shutil.rmtree(OUT_FINAL)
    merged.save_pretrained(
        str(OUT_FINAL),
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tok.save_pretrained(str(OUT_FINAL))
    del merged, adapter_model, base_fp16
    gc.collect()
    torch.cuda.empty_cache()
    final_zip = shutil.make_archive(str(OUT_FINAL), "zip", root_dir=str(OUT_FINAL))
    merged_ok = True
    print(f"✅ Modelo independiente fusionado: {OUT_FINAL}", flush=True)
    print(f"✅ ZIP final: {final_zip}", flush=True)
except Exception as exc:
    print("⚠ No se pudo fusionar/exportar el modelo completo.", flush=True)
    print(f"Motivo: {type(exc).__name__}: {str(exc)[:500]}", flush=True)
    print(
        "Los adaptadores sí quedaron guardados. Para inferencia, carga el modelo base y "
        "después PeftModel.from_pretrained(base, ruta_adaptador).",
        flush=True,
    )

print("\n" + "=" * 76, flush=True)
print("RESUMEN", flush=True)
print(f"GPU usada: {GPU_NAME}", flush=True)
print(f"Pasos de optimizador: {step}", flush=True)
print(f"Tiempo entrenado: {elapsed_final / 3600:.2f} h", flush=True)
print(f"Adaptadores: {OUT_ADAPTER}", flush=True)
print(f"ZIP adaptadores: {adapter_zip}", flush=True)
if merged_ok:
    print(f"Modelo final completo: {OUT_FINAL}", flush=True)
    print(f"ZIP modelo completo: {OUT_FINAL}.zip", flush=True)
else:
    print("El ZIP de adaptadores es el artefacto recuperable; no se generó un modelo completo fusionado.", flush=True)
print(f"Log de entrenamiento: {LOG_CSV}", flush=True)
print("No se borró el estado de entrenamiento, para permitir recuperación/inspección.", flush=True)
print("=" * 76, flush=True)
