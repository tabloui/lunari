r"""Aqua — fase JSON / Tool Calling (fusión 300M v2 + 500M v5, contexto 4096).

Formato (texto plano, sin tokens nuevos; el EOS cierra cada turno del asistente):

    <|system|>\n{system}\n<|tools|>\n{schemas}\n<|user|>\n{user}\n<|assistant|>\n{salida}<EOS>

Salidas del asistente:
  * una herramienta  -> {"name": ..., "arguments": {...}}
  * varias           -> [{"name": ...}, {"name": ...}]   (en orden)
  * ninguna aplica   -> texto breve
  * tras ejecutar    -> el runtime añade "\n<|tool_result|>\n{json}\n<|assistant|>\n"
                        y el modelo responde en texto (nunca inventa el resultado).

La loss se calcula SOLO sobre la salida del asistente (+ EOS).

Variables de entorno útiles:
  AQUA_MODEL_PATH, AQUA_JSON_OUTPUT, AQUA_TARGET_TOKENS, AQUA_SEQ_LEN,
  AQUA_MICRO_BATCH, AQUA_GRAD_ACCUM, AQUA_JSON_LR, AQUA_GRAD_CKPT (0/1),
  AQUA_GEN_EVAL_EVERY (0 = desactivado), AQUA_RESUME (0/1), AQUA_DRY_RUN (0/1),
  AQUA_DIVERSITY (0/1: más formulaciones del usuario), AQUA_TOOL_RESULT_EXAMPLES (0/1).

Contexto: SEQ_LEN = 4096 y max_position_embeddings = 4096 (constantes del script).

AQUA_DRY_RUN=1 solo valida el generador de datos y muestra ejemplos (no entrena).
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import operator
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

# "false": el tokenizer se usa dentro de workers del DataLoader (fork).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import transformers
from packaging import version
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
def _env(name: str, default: Any, cast=str):
    return cast(os.environ.get(name, default))


def _num(x: str) -> int:
    return int(float(x))  # acepta "3e8"


SEED = _env("AQUA_SEED", 42, _num)
TARGET_TOKENS = _env("AQUA_TARGET_TOKENS", 500_000_000, _num)
# Contexto fijo: el modelo se carga con max_position_embeddings=4096 y se entrena a 4096.
MAX_POSITION_EMBEDDINGS = 4096
SEQ_LEN = 4096
assert SEQ_LEN <= MAX_POSITION_EMBEDDINGS
EVAL_BLOCKS = _env("AQUA_EVAL_BLOCKS", 128, _num)
DIVERSITY_MODE = _env("AQUA_DIVERSITY", "1") == "1"
TOOL_RESULT_EXAMPLES = _env("AQUA_TOOL_RESULT_EXAMPLES", "1") == "1"

MODEL_PATH = _env("AQUA_MODEL_PATH", "/kaggle/working/modelo_final")
OUTPUT_DIR = _env("AQUA_JSON_OUTPUT", "/kaggle/working/Aqua-100M-JSON-ToolCalling")

# 2 x 8 x 4096 = 65 536 tokens por paso (con 1 GPU), igual que antes con 4 x 8 x 2048:
# a 4096 se reduce el micro-batch a la mitad para no disparar el pico de memoria (logits).
MICRO_BATCH = _env("AQUA_MICRO_BATCH", 2, _num)
GRAD_ACCUM = _env("AQUA_GRAD_ACCUM", 8, _num)
LR = _env("AQUA_JSON_LR", "5e-5", float)
MIN_LR_RATIO = 0.1
WARMUP_RATIO = 0.02
WEIGHT_DECAY = 0.05
LOG_STEPS = 10
SAVE_STEPS = 500
EVAL_STEPS = 500
SAVE_LIMIT = 3
NUM_WORKERS = _env("AQUA_WORKERS", 2, _num)
GRAD_CKPT = _env("AQUA_GRAD_CKPT", "0") == "1"  # un modelo de 100M no lo necesita
RESUME = _env("AQUA_RESUME", "1") == "1"
DRY_RUN = _env("AQUA_DRY_RUN", "0") == "1"

GEN_EVAL_EVERY = _env("AQUA_GEN_EVAL_EVERY", 1000, _num)
GEN_EVAL_N = 32
GEN_EVAL_FINAL_N = 200
GEN_MAX_NEW_TOKENS = 384

set_seed(SEED)


# ----------------------------------------------------------------------------
# Herramientas (schemas)
# ----------------------------------------------------------------------------
TOOLS: dict[str, dict[str, Any]] = {
    "web_search": {
        "description": "Busca información en Internet.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                "language": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "calculator": {
        "description": "Calcula una expresión matemática.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    "python": {
        "description": "Ejecuta código Python.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 300},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    "get_weather": {
        "description": "Obtiene el tiempo actual.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    },
    "get_time": {
        "description": "Obtiene la hora de una zona.",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
            "additionalProperties": False,
        },
    },
    "read_file": {
        "description": "Lee un archivo.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer", "minimum": 1, "maximum": 100000},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    "write_file": {
        "description": "Escribe un archivo.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "overwrite": {"type": "boolean"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    "list_files": {
        "description": "Lista archivos.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    "github": {
        "description": "Realiza acciones sobre GitHub.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "search_code", "read_file", "list_files"],
                },
                "repository": {"type": "string"},
                "query": {"type": "string"},
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "json_transform": {
        "description": "Transforma o consulta JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["get", "set", "delete"]},
                "data": {},
                "path": {"type": "string"},
                "value": {},
            },
            "required": ["operation", "data"],
            "additionalProperties": False,
        },
    },
    "get_url": {
        "description": "Consulta una URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "enum": ["GET", "HEAD"]},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
}


def compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


_SCHEMA_ITEM = {
    name: compact(
        {
            "name": name,
            "description": spec["description"],
            "parameters": spec["parameters"],
        }
    )
    for name, spec in TOOLS.items()
}


def schema_text(names: list[str]) -> str:
    return "[" + ",".join(_SCHEMA_ITEM[n] for n in names) + "]"


# ----------------------------------------------------------------------------
# Validación (tipos, enum, min/max, required, additionalProperties)
# ----------------------------------------------------------------------------
_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
}


def validate_value(value: Any, schema: dict[str, Any]) -> bool:
    if not schema:  # {} = cualquier valor
        return True
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "anyOf" in schema and not any(validate_value(value, part) for part in schema["anyOf"]):
        return False
    if "oneOf" in schema and sum(validate_value(value, part) for part in schema["oneOf"]) != 1:
        return False
    t = schema.get("type")
    if t is not None and not _TYPE_CHECKS[t](value):
        return False
    if t in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            return False
        if "maximum" in schema and value > schema["maximum"]:
            return False
    if t == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            return False
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return False
    if t == "array":
        item_schema = schema.get("items", {})
        return all(validate_value(v, item_schema) for v in value)
    if t == "object":
        return validate_object(value, schema)
    return True


def validate_object(obj: dict[str, Any], schema: dict[str, Any]) -> bool:
    props = schema.get("properties", {})
    if any(k not in obj for k in schema.get("required", [])):
        return False
    if schema.get("additionalProperties", True) is False and any(k not in props for k in obj):
        return False
    return all(validate_value(v, props[k]) for k, v in obj.items() if k in props)


def validate_call(obj: Any) -> bool:
    if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
        return False
    name, args = obj["name"], obj["arguments"]
    if name not in TOOLS or not isinstance(args, dict):
        return False
    return validate_object(args, TOOLS[name]["parameters"])


def validate_tool_call(text: str) -> bool:
    return validate_call(parse_json(text))


def parse_tool_output(text: str) -> list[dict[str, Any]] | None:
    """Devuelve la lista de llamadas si `text` es una llamada (objeto) o varias (array) válidas."""
    obj = parse_json(text.strip())
    if isinstance(obj, dict):
        calls = [obj]
    elif isinstance(obj, list) and obj:
        calls = obj
    else:
        return None
    return calls if all(validate_call(c) for c in calls) else None


def call_text(calls: list[dict[str, Any]]) -> str:
    text = compact(calls[0] if len(calls) == 1 else calls)
    if parse_tool_output(text) != calls:
        raise ValueError(f"Llamada generada inválida: {text}")
    return text


# ----------------------------------------------------------------------------
# Datos sintéticos: pools y generadores de argumentos
# ----------------------------------------------------------------------------
PATHS = [
    "/tmp/data.json",
    "/kaggle/working/config.json",
    "src/main.py",
    "training.py",
    "README.md",
    "data/tools.json",
]
DIRS = ["src", "data", "docs", "/tmp", "/kaggle/working", "config", "tests"]
FILE_STEMS = ["main", "config", "notes", "tools", "train", "data", "report", "settings"]
EXTS = ["py", "json", "md", "txt", "yaml", "csv"]
LIST_DIRS = ["/tmp", "/kaggle/working", ".", "src", "data", "docs"]
FILE_NAMES = ["a.txt", "b.txt", "main.py", "config.json", "README.md", "train.py", "data.csv", "notes.md"]

PLACES = [
    "Madrid", "Barcelona", "Pamplona", "Londres", "París", "Tokio", "Berlín", "Roma",
    "Nueva York", "México", "Lisboa", "Buenos Aires", "Bogotá", "Santiago", "Sídney",
    "El Cairo", "Dublín", "Ámsterdam", "San Sebastián", "Valencia",
]
TIMEZONES = [
    "Europe/Madrid", "Europe/London", "Europe/Paris", "Europe/Berlin", "Asia/Tokyo",
    "America/New_York", "America/Los_Angeles", "America/Mexico_City", "America/Bogota",
    "Australia/Sydney", "Africa/Cairo", "UTC",
]
CONDITIONS = ["despejado", "nublado", "lluvioso", "soleado", "con viento"]

TOPICS = [
    "inteligencia artificial", "Aqua 1.0.0", "Python asyncio", "Llama 3", "transformers",
    "Kaggle T4", "GitHub tool calling", "energía solar", "recetas de paella",
    "historia de Roma", "precio del bitcoin", "mejores portátiles", "entrenamiento de LLM",
    "Docker compose", "PyTorch DDP", "rutas de senderismo en Navarra",
    "fine-tuning de modelos pequeños", "cuantización GGUF", "SQL window functions",
]
QUERY_SUFFIXES = ["", "", "", " tutorial", " 2026", " últimas noticias", " documentación",
                  " ejemplos", " guía para principiantes", " comparativa"]
REPOS = [
    "tabloui/shark-agent", "tabloui/Api-de-subtom", "aquas-modela/Aqua-1.0.0",
    "huggingface/transformers", "pytorch/pytorch",
]
URLS = ["https://example.com", "https://github.com", "https://huggingface.co",
        "https://kaggle.com", "https://pytorch.org"]

FILE_SNIPPETS = [
    "hola",
    "# Aqua\n\nModelo de lenguaje pequeño.",
    '{"name":"Aqua","version":"1.0.0"}',
    "print('hello')",
    "x = 1\ny = 2",
    "línea uno\nlínea dos",
]
# Incluye comillas y barras invertidas para practicar el escapado JSON.
WRITE_CONTENTS = FILE_SNIPPETS + [
    'echo "hola mundo"',
    "ruta C:\\temp\\log.txt",
    'dijo: "ok"\nfin',
]

JSON_KEYS = ["name", "version", "enabled", "items", "port", "host", "user", "tags",
             "count", "debug", "timeout", "id"]
JSON_STRS = ["Aqua", "1.0.0", "localhost", "admin", "demo", "prod"]


class Scenario(NamedTuple):
    args: dict[str, Any]
    user: str
    result: Any
    answer: str


def phrase(rng: random.Random, base: list[str], extra: list[str] | None = None) -> str:
    """Formulación del usuario. Con AQUA_DIVERSITY=1 se amplía el pool con las variantes de v5.

    Solo se añaden variantes que mencionan los mismos datos que las originales, para que
    el texto del usuario siga determinando los argumentos.
    """
    pool = base + extra if (DIVERSITY_MODE and extra) else base
    return rng.choice(pool)


def make_query(rng: random.Random) -> str:
    return rng.choice(TOPICS) + rng.choice(QUERY_SUFFIXES)


def make_path(rng: random.Random) -> str:
    if rng.random() < 0.3:
        return rng.choice(PATHS)
    return f"{rng.choice(DIRS)}/{rng.choice(FILE_STEMS)}.{rng.choice(EXTS)}"


def make_expr(rng: random.Random) -> str:
    a, b, c = (rng.randint(1, rng.choice([20, 100, 1000, 99999])) for _ in range(3))
    op, op2 = rng.choice("+-*"), rng.choice("+-*")
    form = rng.randrange(7)
    if form == 0:
        return f"{a}{op}{b}"
    if form == 1:
        return f"{a}/{b}"
    if form == 2:
        return f"({a}+{b})*{c}"
    if form == 3:
        return f"{rng.randint(2, 30)}**{rng.randint(2, 5)}"
    if form == 4:
        return f"{a} {op} {b} {op2} {c}"
    if form == 5:
        return f"{a}.{rng.randint(1, 99)}*{b}"
    return f"({a}{op}{b})/{c}"


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def safe_calc(expr: str) -> int | float:
    def ev(node: ast.AST):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponente demasiado grande")
            return _BIN_OPS[type(node.op)](left, right)
        raise ValueError(f"expresión no soportada: {expr}")

    return ev(ast.parse(expr, mode="eval"))


def fmt_num(v: int | float) -> int | float:
    if isinstance(v, float):
        v = round(v, 6)
        if v == int(v) and abs(v) < 1e15:
            return int(v)
    return v


def make_python(rng: random.Random) -> tuple[str, str]:
    """(código, stdout esperado) — se calcula sin ejecutar nada externo."""
    a, b = rng.randint(1, 50), rng.randint(1, 50)
    k = rng.randrange(6)
    if k == 0:
        return f"print({a}+{b})", str(a + b)
    if k == 1:
        return f"print(sum(range({a})))", str(sum(range(a)))
    if k == 2:
        n = min(a, 10)
        return f"x=[i*{b} for i in range({n})]; print(x)", str([i * b for i in range(n)])
    if k == 3:
        return f"import math; print(math.sqrt({a * a}))", str(math.sqrt(a * a))
    if k == 4:
        return f"data={{'a':{a},'b':{b}}}; print(data)", str({"a": a, "b": b})
    n = min(a, 4)
    return f"for i in range({n}):\n    print(i)", "\n".join(str(i) for i in range(n))


# --- JSON anidado --------------------------------------------------------------
def rand_scalar(rng: random.Random) -> Any:
    k = rng.randrange(4)
    if k == 0:
        return rng.choice(JSON_STRS)
    if k == 1:
        return rng.randint(0, 9999)
    if k == 2:
        return rng.choice([True, False])
    return [rng.randint(1, 9) for _ in range(rng.randint(1, 4))]


def rand_data(rng: random.Random, depth: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in rng.sample(JSON_KEYS, rng.randint(2, 4)):
        out[k] = rand_data(rng, depth - 1) if depth > 0 and rng.random() < 0.4 else rand_scalar(rng)
    return out


def all_paths(d: dict[str, Any], prefix: str = "") -> list[str]:
    out: list[str] = []
    for k, v in d.items():
        out.append(prefix + k)
        if isinstance(v, dict):
            out += all_paths(v, prefix + k + ".")
    return out


def _walk(d: dict[str, Any], path: str) -> tuple[dict[str, Any], str]:
    *parents, last = path.split(".")
    for k in parents:
        d = d[k]
    return d, last


def get_path(d: dict[str, Any], path: str) -> Any:
    parent, last = _walk(d, path)
    return parent[last]


def set_path(d: dict[str, Any], path: str, value: Any) -> None:
    parent, last = _walk(d, path)
    parent[last] = value


def del_path(d: dict[str, Any], path: str) -> None:
    parent, last = _walk(d, path)
    del parent[last]


# ----------------------------------------------------------------------------
# Escenarios por herramienta: args + petición del usuario + resultado + respuesta final.
# El texto del usuario SIEMPRE determina los argumentos (si un argumento opcional
# aparece, el usuario lo menciona), para no enseñar al modelo a inventar valores.
# ----------------------------------------------------------------------------
def sc_web_search(rng: random.Random) -> Scenario:
    q = make_query(rng)
    args: dict[str, Any] = {"query": q}
    text = phrase(rng, [
        f"Busca información sobre {q}.",
        f"Busca en Internet {q}.",
        f"¿Puedes buscar {q} en la web?",
        f"Search the web for {q}.",
    ], [
        f"Necesito resultados sobre {q}.",
        f"Investiga {q} y usa la herramienta de búsqueda.",
        f"Encuentra información fiable sobre {q} y usa la búsqueda.",
        f"Necesito investigar {q}; haz una búsqueda web.",
        f"Busca {q} y dame resultados para continuar.",
        f"Consulta en Internet {q}.",
    ])
    if rng.random() < 0.5:
        n = rng.choice([3, 5, 10, 15])
        args["max_results"] = n
        text += f" Quiero {n} resultados."
    if rng.random() < 0.35:
        code, lang = rng.choice([("es", "español"), ("en", "inglés")])
        args["language"] = code
        text += f" Resultados en {lang}."
    shown = min(args.get("max_results", 3), 4)
    slug = re.sub(r"\W+", "-", q.lower()).strip("-")
    results = [
        {"title": f"{q} (resultado {i})", "url": f"https://example.com/{slug}/{i}",
         "snippet": f"Resumen sobre {q}."}
        for i in range(1, shown + 1)
    ]
    answer = f"Estos son los primeros resultados sobre {q}. El primero: {results[0]['title']} ({results[0]['url']})."
    return Scenario(args, text, {"results": results}, answer)


def sc_calculator(rng: random.Random) -> Scenario:
    e = make_expr(rng)
    text = phrase(rng, [f"Calcula {e}.", f"¿Cuánto es {e}?", f"Resuelve {e}.", f"Compute {e}."], [
        f"Resuelve {e} usando la calculadora.",
        f"Necesito el resultado exacto de {e}.",
        f"Calcula exactamente {e}.",
        f"Necesito resolver {e}. Usa la calculadora.",
        f"Evalúa {e} sin hacerlo manualmente.",
    ])
    value = fmt_num(safe_calc(e))
    return Scenario({"expression": e}, text, {"result": value}, f"{e} = {value}.")


def sc_python(rng: random.Random) -> Scenario:
    code, stdout = make_python(rng)
    args: dict[str, Any] = {"code": code}
    if "\n" in code:
        text = f"Ejecuta este código Python:\n{code}"
    else:
        text = phrase(rng, [f"Ejecuta este Python: {code}", f"Corre este código: {code}"], [
            f"Corre el siguiente código Python: {code}",
            f"Usa Python para ejecutar: {code}",
            f"Ejecuta este código Python y usa la herramienta: {code}",
            f"Corre {code} con Python.",
            f"Necesito ejecutar: {code}",
        ])
    if rng.random() < 0.4:
        t = rng.choice([10, 30, 60, 120])
        args["timeout"] = t
        text += f"\nUsa un timeout de {t} segundos."
    answer = f"La salida fue: {stdout}"
    return Scenario(args, text, {"stdout": stdout, "exit_code": 0}, answer)


def sc_get_weather(rng: random.Random) -> Scenario:
    loc = rng.choice(PLACES)
    args: dict[str, Any] = {"location": loc}
    text = phrase(rng, [
        f"¿Qué tiempo hace en {loc}?",
        f"Dime el tiempo actual en {loc}.",
        f"What's the weather in {loc}?",
    ], [
        f"Consulta el clima de {loc}.",
        f"Consulta el tiempo actual de {loc}.",
        f"¿Cuál es el clima de {loc} ahora?",
        f"Obtén el tiempo de {loc}.",
        f"Dime el tiempo actual de {loc}.",
    ])
    units = "celsius"
    if rng.random() < 0.5:
        units = rng.choice(["celsius", "fahrenheit"])
        args["units"] = units
        text += " En grados Celsius." if units == "celsius" else " En grados Fahrenheit."
    temp = rng.randint(-5, 38) if units == "celsius" else rng.randint(23, 100)
    cond = rng.choice(CONDITIONS)
    sym = "°C" if units == "celsius" else "°F"
    result = {"location": loc, "temperature": temp, "units": units, "condition": cond}
    return Scenario(args, text, result, f"En {loc} hay {temp}{sym} y está {cond}.")


def sc_get_time(rng: random.Random) -> Scenario:
    tz = rng.choice(TIMEZONES)
    text = phrase(rng, [f"¿Qué hora es en {tz}?", f"Dime la hora en {tz}.", f"What time is it in {tz}?"], [
        f"Consulta la hora de {tz}.",
        f"Necesito la hora actual en {tz}.",
        f"Obtén la hora actual de {tz}.",
        f"¿Qué hora es ahora mismo en {tz}?",
    ])
    hhmm = f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
    return Scenario({"timezone": tz}, text, {"timezone": tz, "time": hhmm}, f"En {tz} son las {hhmm}.")


def sc_read_file(rng: random.Random) -> Scenario:
    path = make_path(rng)
    args: dict[str, Any] = {"path": path}
    text = phrase(rng, [f"Lee {path}.", f"Muéstrame el contenido de {path}.", f"Read {path}."], [
        f"Lee el archivo {path}.",
        f"Necesito el contenido de {path}.",
        f"Abre y lee {path}.",
        f"Lee {path} y devuelve su contenido.",
        f"Abre el archivo {path}.",
        f"Necesito leer {path}.",
    ])
    if rng.random() < 0.4:
        n = rng.choice([500, 1000, 4000, 10000])
        args["max_chars"] = n
        text += f" Máximo {n} caracteres."
    content = rng.choice(FILE_SNIPPETS)
    return Scenario(args, text, {"path": path, "content": content, "truncated": False},
                    f"{path} contiene: {content}")


def sc_write_file(rng: random.Random) -> Scenario:
    path, content = make_path(rng), rng.choice(WRITE_CONTENTS)
    args: dict[str, Any] = {"path": path, "content": content}
    text = phrase(rng, [f"Escribe un archivo en {path}.", f"Guarda un archivo en {path}."], [
        f"Escribe el contenido indicado en {path}.",
        f"Guarda este contenido en {path}.",
        f"Crea o actualiza {path} con el contenido solicitado.",
        f"Escribe en {path} lo que te indique.",
        f"Crea o actualiza {path}.",
    ])
    if rng.random() < 0.6:
        ow = rng.choice([True, False])
        args["overwrite"] = ow
        text += " Sobrescribe si existe." if ow else " No sobrescribas si ya existe."
    text += f"\nContenido:\n{content}"
    n = len(content.encode("utf-8"))
    return Scenario(args, text, {"ok": True, "path": path, "bytes_written": n},
                    f"He escrito {n} bytes en {path}.")


def sc_list_files(rng: random.Random) -> Scenario:
    path = rng.choice(LIST_DIRS)
    args: dict[str, Any] = {"path": path}
    text = phrase(rng, [f"Lista los archivos de {path}.", f"¿Qué archivos hay en {path}?"], [
        f"Muestra qué archivos hay en {path}.",
        f"Enumera los archivos de {path}.",
        f"Lista los archivos disponibles en {path}.",
        f"Enumera el contenido de {path}.",
        f"Quiero saber qué archivos hay en {path}.",
    ])
    if rng.random() < 0.5:
        rec = rng.choice([True, False])
        args["recursive"] = rec
        text += " De forma recursiva." if rec else " Sin recursión."
    files = rng.sample(FILE_NAMES, rng.randint(2, 4))
    return Scenario(args, text, {"path": path, "files": files},
                    f"En {path} hay: {', '.join(files)}.")


def sc_github(rng: random.Random) -> Scenario:
    action = rng.choice(["search", "search_code", "read_file", "list_files"])
    repo = rng.choice(REPOS)
    if action in ("search", "search_code"):
        q = make_query(rng)
        if action == "search" and rng.random() < 0.3:
            args = {"action": action, "query": q}
            text = f"Busca {q} en GitHub."
        else:
            args = {"action": action, "repository": repo, "query": q}
            text = (f"Busca {q} en el repositorio {repo} de GitHub." if action == "search"
                    else f"Busca código relacionado con {q} en el repositorio {repo} de GitHub.")
        result = {"action": action,
                  "results": [{"path": rng.choice(PATHS), "snippet": f"Coincidencia con {q}"} for _ in range(2)]}
        answer = f"Encontré 2 coincidencias para {q} en GitHub."
    elif action == "read_file":
        path = make_path(rng)
        content = rng.choice(FILE_SNIPPETS)
        args = {"action": action, "repository": repo, "path": path}
        text = f"Lee {path} del repositorio {repo} en GitHub."
        result = {"path": path, "content": content}
        answer = f"{path} contiene: {content}"
    else:
        path = rng.choice(["src", "data", ".", "docs"])
        files = rng.sample(FILE_NAMES, 3)
        args = {"action": action, "repository": repo, "path": path}
        text = f"Lista los archivos de {path} en el repositorio {repo} de GitHub."
        result = {"path": path, "files": files}
        answer = f"En {path} hay: {', '.join(files)}."
    return Scenario(args, text, result, answer)


def sc_json_transform(rng: random.Random) -> Scenario:
    data = rand_data(rng, rng.choice([0, 1, 2, 3]))
    path = rng.choice(all_paths(data))
    op = rng.choice(["get", "set", "delete"])
    args: dict[str, Any] = {"operation": op, "data": data, "path": path}
    blob = compact(data)
    if op == "get":
        value = get_path(data, path)
        text = f"Obtén el valor de la ruta {path} en este JSON: {blob}"
        return Scenario(args, text, {"value": value}, f"El valor en {path} es {compact(value)}.")
    new = copy.deepcopy(data)
    if op == "delete":
        del_path(new, path)
        text = f"Elimina la clave {path} de este JSON: {blob}"
    else:
        value = rand_scalar(rng)
        args["value"] = value
        set_path(new, path, value)
        text = f"Establece {path} a {compact(value)} en este JSON: {blob}"
    return Scenario(args, text, {"data": new}, f"JSON resultante: {compact(new)}")


def sc_get_url(rng: random.Random) -> Scenario:
    url = rng.choice(URLS) + rng.choice(["", "", "/docs", "/about", "/pricing"])
    args: dict[str, Any] = {"url": url}
    text = rng.choice([f"Consulta {url}.", f"Haz una petición a {url}.", f"Check {url}."])
    method = "GET"
    if rng.random() < 0.4:
        method = rng.choice(["GET", "HEAD"])
        args["method"] = method
        text += f" Usa {method}."
    result = {"url": url, "method": method, "status": 200, "content_type": "text/html"}
    return Scenario(args, text, result, f"{url} respondió con estado 200.")


SCENARIOS = {
    "web_search": sc_web_search,
    "calculator": sc_calculator,
    "python": sc_python,
    "get_weather": sc_get_weather,
    "get_time": sc_get_time,
    "read_file": sc_read_file,
    "write_file": sc_write_file,
    "list_files": sc_list_files,
    "github": sc_github,
    "json_transform": sc_json_transform,
    "get_url": sc_get_url,
}
assert set(SCENARIOS) == set(TOOLS), "Cada herramienta necesita un escenario"


# ----------------------------------------------------------------------------
# Ejemplos de entrenamiento: listas de segmentos (texto, ¿se entrena?, ¿EOS?)
# ----------------------------------------------------------------------------
class Seg(NamedTuple):
    text: str
    train: bool = False
    eos: bool = False


SYSTEM_PROMPTS = [
    "Eres un asistente con herramientas. Si necesitas una herramienta, responde solo con un JSON válido "
    '{"name":...,"arguments":{...}}. Si necesitas varias, responde con un array JSON de llamadas en orden. '
    "Respeta el schema exactamente. Si ninguna herramienta aplica, responde brevemente en texto.",
    "Usa las herramientas disponibles. Una llamada: objeto JSON con name y arguments. Varias llamadas: array "
    "JSON en orden. No añadas texto fuera del JSON ni argumentos que no estén en el schema. "
    "Si no hace falta herramienta, contesta en texto breve.",
    "Genera exclusivamente tool calls en JSON válido cuando la petición lo requiera (objeto para una, array "
    "para varias). Cumple tipos, enums y campos obligatorios del schema. Sin herramienta aplicable, "
    "responde en una frase.",
]
SYSTEM_REPAIR = (
    "Corrige la llamada a herramienta y devuelve exclusivamente el JSON válido corregido, "
    "conforme al schema. No expliques la corrección."
)

NO_TOOL_REPLIES = [
    ("Hola", "¡Hola! ¿En qué puedo ayudarte?"),
    ("Buenos días", "¡Buenos días! ¿Qué necesitas?"),
    ("Gracias", "De nada."),
    ("Vale, perfecto", "Perfecto. Avísame si necesitas algo más."),
    ("Adiós", "¡Hasta luego!"),
    ("¿Qué puedes hacer?", "Puedo usar las herramientas disponibles para buscar, calcular, leer y escribir archivos, entre otras cosas."),
]


def format_prompt(system: str, tool_names: list[str], user: str) -> str:
    return f"<|system|>\n{system}\n<|tools|>\n{schema_text(tool_names)}\n<|user|>\n{user}\n<|assistant|>\n"


def format_tool_result(result: Any) -> str:
    return f"\n<|tool_result|>\n{compact(result)}\n<|assistant|>\n"


def pick_tools(rng: random.Random, required: list[str]) -> list[str]:
    """Herramientas necesarias + 0-3 distractores, barajadas (el modelo aprende a elegir)."""
    extra = rng.sample([n for n in TOOLS if n not in required], rng.randint(0, 3))
    names = required + extra
    rng.shuffle(names)
    return names


def multi_user(rng: random.Random, scs: list[Scenario]) -> str:
    head = rng.choice([
        "Haz estas tareas en orden:",
        "Necesito que hagas lo siguiente, en este orden:",
        "Do these in order:",
    ])
    return head + "\n" + "\n".join(f"{i}. {s.user}" for i, s in enumerate(scs, 1))


def build_call_case(
    rng: random.Random, n_calls: int, forced: list[str] | None = None
) -> tuple[str, list[dict[str, Any]], list[Scenario]]:
    names = list(forced) if forced else rng.sample(list(TOOLS), n_calls)
    scs = [SCENARIOS[n](rng) for n in names]
    calls = [{"name": n, "arguments": s.args} for n, s in zip(names, scs)]
    user = scs[0].user if len(scs) == 1 else multi_user(rng, scs)
    prompt = format_prompt(rng.choice(SYSTEM_PROMPTS), pick_tools(rng, names), user)
    return prompt, calls, scs


def ex_calls(rng: random.Random, n_calls: int, forced: list[str] | None = None) -> list[Seg]:
    prompt, calls, _ = build_call_case(rng, n_calls, forced)
    return [Seg(prompt), Seg(call_text(calls), True, True)]


# Pares de herramientas dependientes, en orden (de v5).
CHAIN_PAIRS = [
    ("web_search", "get_url"),
    ("list_files", "read_file"),
    ("github", "read_file"),
    ("get_time", "web_search"),
    ("json_transform", "write_file"),
]


def ex_chain(rng: random.Random) -> list[Seg]:
    return ex_calls(rng, 2, forced=list(rng.choice(CHAIN_PAIRS)))


def ex_result_loop(rng: random.Random) -> list[Seg]:
    prompt, calls, (sc,) = build_call_case(rng, 1)
    if rng.random() < 0.1:
        result: Any = {"ok": False, "error": "No se pudo completar la operación."}
        answer = "La herramienta devolvió un error: No se pudo completar la operación."
    else:
        result, answer = sc.result, sc.answer
    return [
        Seg(prompt),
        Seg(call_text(calls), True, True),
        Seg(format_tool_result(result)),
        Seg(answer, True, True),
    ]


def _corrupt_missing_required(correct: str, obj: dict[str, Any]) -> str:
    args = dict(obj["arguments"])
    required = TOOLS[obj["name"]]["parameters"].get("required", [])
    if required:
        args.pop(required[0], None)
    return compact({"name": obj["name"], "arguments": args})


def _corrupt_wrong_type(correct: str, obj: dict[str, Any]) -> str:
    args = dict(obj["arguments"])
    for k, v in args.items():
        if isinstance(v, (int, bool)):
            args[k] = str(v)
            break
    return compact({"name": obj["name"], "arguments": args})


CORRUPTIONS = [
    lambda c, o: c[:-1],                                     # truncado
    lambda c, o: c[:-2] + ",}}",                             # coma final
    lambda c, o: c.replace("{", "", 1),                      # falta llave
    lambda c, o: c.replace('"arguments":', '"arguments":{', 1),  # llave extra en arguments
    lambda c, o: c.replace('"name"', '"tool_name"', 1),      # clave incorrecta
    lambda c, o: c.replace('"', "'"),                        # comillas simples
    lambda c, o: "Aquí tienes la llamada:\n" + c,            # texto extra
    lambda c, o: c.replace("true", "True").replace("false", "False"),
    _corrupt_missing_required,
    _corrupt_wrong_type,
]


def ex_repair(rng: random.Random) -> list[Seg]:
    name = rng.choice(list(TOOLS))
    sc = SCENARIOS[name](rng)
    obj = {"name": name, "arguments": sc.args}
    correct = call_text([obj])
    bad = correct[:-1]
    for _ in range(20):
        cand = rng.choice(CORRUPTIONS)(correct, obj)
        if cand != correct and not validate_tool_call(cand):
            bad = cand
            break
    prompt = format_prompt(SYSTEM_REPAIR, pick_tools(rng, [name]),
                           f"Corrige esta llamada para que sea JSON válido y cumpla el schema:\n{bad}")
    return [Seg(prompt), Seg(correct, True, True)]


def ex_no_tool(rng: random.Random) -> list[Seg]:
    user, reply = rng.choice(NO_TOOL_REPLIES)
    tools = rng.sample(list(TOOLS), rng.randint(1, 4))
    prompt = format_prompt(rng.choice(SYSTEM_PROMPTS), tools, user)
    return [Seg(prompt), Seg(reply, True, True)]


def make_example(rng: random.Random) -> list[Seg]:
    p = rng.random()
    if p < 0.40:
        return ex_calls(rng, 1)
    if p < 0.48:
        return ex_calls(rng, 1, forced=["json_transform"])  # JSON profundo
    if p < 0.60:
        return ex_calls(rng, rng.choice([2, 3]))
    if p < 0.68:
        return ex_chain(rng)
    if p < 0.88:
        return ex_result_loop(rng) if TOOL_RESULT_EXAMPLES else ex_calls(rng, 1)
    if p < 0.95:
        return ex_repair(rng)
    return ex_no_tool(rng)


def selfcheck(n: int = 3000) -> None:
    """Falla pronto (antes de gastar GPU) si algún generador produce JSON/schema inválido."""
    rng = random.Random(SEED)
    for _ in range(n):
        make_example(rng)
    print(f"[selfcheck] {n} ejemplos generados y validados correctamente.")


# ----------------------------------------------------------------------------
# Dataset empaquetado y determinista (el bloque i siempre es el mismo → reanudable)
# ----------------------------------------------------------------------------
def encode_segments(tokenizer, segs: list[Seg], eos_id: int) -> tuple[list[int], list[int]]:
    # Se tokeniza cada segmento por separado: la frontera prompt/respuesta coincide
    # exactamente con la de inferencia (se tokeniza el prompt y se genera a partir de ahí).
    enc = tokenizer([s.text for s in segs], add_special_tokens=False)["input_ids"]
    ids: list[int] = []
    labels: list[int] = []
    for seg, seg_ids in zip(segs, enc):
        ids += seg_ids
        labels += seg_ids if seg.train else [-100] * len(seg_ids)
        if seg.eos:
            ids.append(eos_id)
            labels.append(eos_id)
    return ids, labels


class PackedToolDataset(Dataset):
    def __init__(self, tokenizer, num_blocks: int, seed: int):
        if tokenizer.eos_token_id is None:
            raise ValueError("El tokenizer necesita eos_token_id para marcar el fin de la llamada.")
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.eos_token_id
        self.num_blocks = num_blocks
        self.seed = seed

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = random.Random(self.seed * 1_000_003 + idx)
        ids: list[int] = []
        labels: list[int] = []
        while len(ids) < SEQ_LEN:
            e_ids, e_labels = encode_segments(self.tokenizer, make_example(rng), self.eos_id)
            ids += e_ids
            labels += e_labels
        ids, labels = ids[:SEQ_LEN], labels[:SEQ_LEN]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(SEQ_LEN, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ----------------------------------------------------------------------------
# Evaluación por generación (JSON válido / schema / coincidencia exacta)
# ----------------------------------------------------------------------------
def build_eval_case(rng: random.Random) -> tuple[str, list[dict[str, Any]]]:
    prompt, calls, _ = build_call_case(rng, 1 if rng.random() < 0.75 else 2)
    return prompt, calls


@torch.no_grad()
def evaluate_generation(model, tokenizer, n: int, seed: int = SEED + 1) -> dict[str, float]:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    rng = random.Random(seed)
    stats: Counter[str] = Counter()
    for _ in range(n):
        prompt, expected = build_eval_case(rng)
        enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        out = model.generate(
            **enc,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        text = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        stats["valid_json"] += parse_json(text) is not None
        calls = parse_tool_output(text)
        stats["valid_schema"] += calls is not None
        if calls is not None:
            stats["right_tools"] += [c["name"] for c in calls] == [c["name"] for c in expected]
            stats["exact_match"] += calls == expected
    if was_training:
        model.train()
    return {k: stats[k] / n for k in ("valid_json", "valid_schema", "right_tools", "exact_match")}


class GenerationEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, every: int, n: int):
        self.tokenizer, self.every, self.n = tokenizer, every, n

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every <= 0 or state.global_step == 0 or state.global_step % self.every:
            return
        if not state.is_world_process_zero or model is None:
            return
        m = evaluate_generation(model, self.tokenizer, self.n)
        print(f"[gen-eval] paso {state.global_step}: " + " | ".join(f"{k}={v:.2%}" for k, v in m.items()))


# ----------------------------------------------------------------------------
# Utilidades de entrenamiento
# ----------------------------------------------------------------------------
def data_parallel_size() -> int:
    ws = int(os.environ.get("WORLD_SIZE", "0"))
    return ws if ws > 0 else max(torch.cuda.device_count(), 1)


def build_training_args(total_steps: int) -> TrainingArguments:
    use_cuda = torch.cuda.is_available()
    # T4/P100 no tienen bf16 nativo → fp16 con pesos en fp32 (AMP). En Ampere+ se usa bf16.
    bf16 = use_cuda and torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    fp16 = use_cuda and not bf16

    kwargs: dict[str, Any] = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=MICRO_BATCH,
        per_device_eval_batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": MIN_LR_RATIO},
        warmup_steps=max(1, int(total_steps * WARMUP_RATIO)),
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=1.0,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=GRAD_CKPT,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused" if use_cuda else "adamw_torch",
        logging_steps=LOG_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_LIMIT,
        eval_steps=EVAL_STEPS,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=use_cuda,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
        seed=SEED,
        data_seed=SEED,
    )
    if bf16:
        kwargs["tf32"] = True
    if NUM_WORKERS > 0:
        kwargs["dataloader_persistent_workers"] = True

    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    kwargs["eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"] = "steps"
    if "lr_scheduler_kwargs" not in supported:  # transformers muy antiguo
        kwargs["lr_scheduler_type"] = "cosine"
        kwargs.pop("lr_scheduler_kwargs")
    dropped = sorted(k for k in kwargs if k not in supported)
    if dropped:
        print(f"[aviso] TrainingArguments no soporta (se ignoran): {dropped}")
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in supported})


def dry_run(tokenizer) -> None:
    eos = tokenizer.eos_token_id
    ds = PackedToolDataset(tokenizer, 64, SEED)
    sup = tot = 0
    for i in range(len(ds)):
        item = ds[i]
        sup += int((item["labels"] != -100).sum())
        tot += SEQ_LEN
    print(f"[dry-run] fracción de tokens supervisados: {sup / tot:.1%} (EOS id = {eos})")
    rng = random.Random(SEED)
    for _ in range(4):
        segs = make_example(rng)
        ids, labels = encode_segments(tokenizer, segs, eos)
        trained = [t for t, l in zip(ids, labels) if l != -100]
        print("-" * 72)
        print(f"tokens totales={len(ids)} supervisados={len(trained)}")
        print("PROMPT (sin loss):", segs[0].text[-300:].replace("\n", "⏎"))
        print("ENTRENADO:", tokenizer.decode(trained).replace("\n", "⏎"))


def load_model():
    """Carga el modelo forzando max_position_embeddings=4096 (con RoPE no cambia ningún tensor)."""
    # Pesos en fp32 + autocast fp16/bf16: con pesos fp16 el GradScaler falla
    # ("Attempting to unscale FP16 gradients").
    dtype_kw = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56.0") else "torch_dtype"
    config = AutoConfig.from_pretrained(MODEL_PATH)
    original = getattr(config, "max_position_embeddings", None)
    config.max_position_embeddings = MAX_POSITION_EMBEDDINGS
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, config=config, low_cpu_mem_usage=True, **{dtype_kw: torch.float32}
        )
    except (RuntimeError, ValueError) as e:
        if "size mismatch" in str(e):
            raise RuntimeError(
                f"El modelo usa posiciones aprendidas ({original}) y no se puede pasar a "
                f"{MAX_POSITION_EMBEDDINGS} solo cambiando la config: hay que redimensionar la "
                "matriz de posiciones (o usar un modelo con RoPE)."
            ) from e
        raise
    if model.config.max_position_embeddings != MAX_POSITION_EMBEDDINGS:
        raise ValueError(
            f"max_position_embeddings={model.config.max_position_embeddings}, "
            f"se esperaba {MAX_POSITION_EMBEDDINGS}."
        )
    print(f"max_position_embeddings: {original} -> {model.config.max_position_embeddings}")
    return model


def main() -> None:
    dp = data_parallel_size()
    eff_blocks = MICRO_BATCH * GRAD_ACCUM * dp
    num_blocks = math.ceil(math.ceil(TARGET_TOKENS / SEQ_LEN) / eff_blocks) * eff_blocks
    total_steps = num_blocks // eff_blocks

    print("=" * 72)
    print("AQUA — FASE JSON / TOOL CALLING (fusión 300M v2 + 500M v5)")
    print("=" * 72)
    print(f"Modelo base: {MODEL_PATH}")
    print(f"Contexto: SEQ_LEN={SEQ_LEN} | max_position_embeddings={MAX_POSITION_EMBEDDINGS}")
    print(f"Objetivo: {TARGET_TOKENS:,} tokens -> {num_blocks:,} bloques x {SEQ_LEN} = {num_blocks * SEQ_LEN:,}")
    print(f"GPUs: {torch.cuda.device_count()} | batch efectivo: {eff_blocks} bloques = "
          f"{eff_blocks * SEQ_LEN:,} tokens/paso | pasos: {total_steps:,}")
    print("=" * 72)

    selfcheck()

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"No existe {MODEL_PATH}. Copia aquí el modelo_final obtenido en la fase anterior."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("El tokenizer no define eos_token; es necesario para que el modelo aprenda a parar.")
    tokenizer.model_max_length = MAX_POSITION_EMBEDDINGS

    if DRY_RUN:
        dry_run(tokenizer)
        return

    model = load_model()
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        raise ValueError("El tokenizer tiene más tokens que la matriz de embeddings del modelo.")
    model.config.use_cache = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {n_params / 1e6:.1f}M")

    train_ds = PackedToolDataset(tokenizer, num_blocks, SEED)
    eval_ds = PackedToolDataset(tokenizer, EVAL_BLOCKS, SEED + 7919)

    args = build_training_args(total_steps)
    trainer_kw: dict[str, Any] = {}
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kw["processing_class"] = tokenizer
    else:
        trainer_kw["tokenizer"] = tokenizer

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,  # los bloques ya tienen longitud fija
        callbacks=[GenerationEvalCallback(tokenizer, GEN_EVAL_EVERY, GEN_EVAL_N)],
        **trainer_kw,
    )

    last_ckpt = None
    if RESUME and Path(OUTPUT_DIR).is_dir():
        last_ckpt = get_last_checkpoint(OUTPUT_DIR)
        if last_ckpt:
            print(f"Reanudando desde {last_ckpt}")
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)

    eval_metrics = trainer.evaluate()
    model.config.use_cache = True
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    final_dir = Path(OUTPUT_DIR) / "final"
    trainer.save_model(str(final_dir))  # solo escribe el proceso principal

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_dir))
        gen_metrics = evaluate_generation(model, tokenizer, GEN_EVAL_FINAL_N)
        print("[gen-eval final] " + " | ".join(f"{k}={v:.2%}" for k, v in gen_metrics.items()))

        with open(final_dir / "tools.json", "w", encoding="utf-8") as f:
            json.dump(TOOLS, f, ensure_ascii=False, indent=2)

        metadata = {
            "name": "Aqua-100M-JSON-ToolCalling",
            "base_model": MODEL_PATH,
            "parameters": n_params,
            "target_tokens": TARGET_TOKENS,
            "tokens_processed": num_blocks * SEQ_LEN,
            "sequence_length": SEQ_LEN,
            "context_length": SEQ_LEN,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "purpose": "strict JSON and tool calling",
            "loss_on": "assistant tokens only (+EOS)",
            "prompt_format": {
                "template": "<|system|>\\n{system}\\n<|tools|>\\n{schemas}\\n<|user|>\\n{user}\\n<|assistant|>\\n",
                "tool_result_turn": "\\n<|tool_result|>\\n{json}\\n<|assistant|>\\n",
                "single_call": '{"name":...,"arguments":{...}}',
                "multi_call": "[{...},{...}]",
                "stop": "EOS",
            },
            "tools": list(TOOLS),
            "tool_count": len(TOOLS),
            "diversity_mode": DIVERSITY_MODE,
            "tool_result_examples": TOOL_RESULT_EXAMPLES,
            "optimizer": str(getattr(args.optim, "value", args.optim)),
            "capabilities_trained": [
                "single_tool_call", "multi_tool_calls", "tool_chains", "tool_selection_with_distractors",
                "tool_result_to_answer", "json_repair", "nested_json", "no_tool_reply",
            ],
            "general_conversation_training": False,
            "reasoning_training": False,
            "eval_loss": eval_metrics.get("eval_loss"),
            "generation_eval": {"n": GEN_EVAL_FINAL_N, "greedy": True, **gen_metrics},
            "train_runtime_s": train_result.metrics.get("train_runtime"),
        }
        with open(final_dir / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print()
        print("=" * 72)
        print("ENTRENAMIENTO JSON TERMINADO")
        print("=" * 72)
        print(f"Modelo final: {final_dir}")
        print(f"Tokens procesados: {num_blocks * SEQ_LEN:,}")
        print("=" * 72)


if __name__ == "__main__":
    main()
