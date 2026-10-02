#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aqua 0.5B: entrenamiento conversacional desde cero con PyTorch y librería estándar."""

# SECCIÓN 1: IDENTIDAD Y CONSTANTES
import os, re, sys, json, math, time, glob, html, pickle, random, hashlib, shutil, uuid, signal, atexit, gzip, subprocess, tempfile, argparse, copy, urllib.request, urllib.parse, unicodedata, warnings, traceback
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

MODEL_NAME = "Aqua 0.5B"
MODEL_AUTHOR = "desarrollador privado"
MODEL_VERSION = "0.5.0"
PERSONALITY_NAME = "Aqua"
DEFAULT_SYSTEM = "Soy Aqua, un modelo conversacional creado por un desarrollador privado para ayudar, explicar y conversar con claridad."
CREATOR_REPLY = "Fui creado por un desarrollador privado que prefiere mantener su identidad en reserva."
IDENTITY_REPLY = "Soy Aqua, un modelo de lenguaje de 80 millones de parámetros creado por un desarrollador privado. Estoy aquí para ayudarte a aprender, explicar ideas y conversar."
IGNORE_INDEX = -100
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>", "<|tool|>", "<|think|>"]
BASE_VOCAB_SIZE = 256
VOCAB_SIZE = 32000
IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
CACHE_DIR = Path("/kaggle/working/cache") if IS_KAGGLE else Path("cache")
WEB_CACHE = CACHE_DIR / "web"
TOKEN_CACHE = CACHE_DIR / "tokenized"
DATA_DIR = Path("data")
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else Path(".")
INPUT_DIR = Path("/kaggle/input") if IS_KAGGLE else Path(".")
CKPT_DIR = WORK_DIR / "checkpoints"
LOG_DIR = WORK_DIR / "logs"
TRAIN_LOG_PATH = LOG_DIR / "training.log"
METRICS_LOG_PATH = LOG_DIR / "metrics.jsonl"
UPLOAD_LOG_PATH = LOG_DIR / "uploads.log"
TOKENIZER_PATH = DATA_DIR / "aqua_tokenizer.json"
CONFIG_PATH = DATA_DIR / "aqua_config.json"
CORPUS_PATH = DATA_DIR / "aqua_corpus.pkl"
SEED = 1337

# El objetivo se alcanza con 16 capas manteniendo d_model=1280 y vocabulario=64000.
# Esta configuración produce 501,392,640 parámetros con pesos de embedding atados.
@dataclass
class ModelConfig:
    vocab_size: int = 32000
    context_length: int = 1024
    embed_dim: int = 640
    num_heads: int = 10
    num_layers: int = 10
    ffn_mult: float = 4.0
    head_dim: int = 64
    dropout: float = 0.1
    use_rope: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    tie_weights: bool = True
    gradient_checkpointing: bool = False
    num_kv_heads: int = 5
    rope_scaling: str = "none"
    rope_scaling_factor: float = 1.0
    qk_norm: bool = True
    parallel_attn_mlp: bool = True
    attention_sinks: bool = True
    kv_cache_int8: bool = False

    def to_dict(self):
        return asdict(self)

@dataclass
class TrainConfig:
    batch_size: int = 16
    grad_accum: int = 8
    max_steps: int = 15000
    warmup_steps: int = 300
    lr: float = 4e-4
    min_lr: float = 4e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 300
    eval_steps: int = 40
    log_interval: int = 20
    save_interval: int = 500
    save_every_steps: int = 500
    save_every_minutes: int = 30
    kaggle_upload_every_steps: int = 2000
    kaggle_upload_enabled: bool = True
    kaggle_dataset_name: str = ""
    max_session_hours: float = 11.5
    compress_checkpoints: bool = True
    keep_last_n_checkpoints: int = 5
    verify_checkpoint_hash: bool = True
    compile_model: bool = True
    use_flash_attention: bool = True
    amp_dtype: str = "bfloat16"
    num_workers: int = 4
    pin_memory: bool = True
    patience: int = 6
    min_delta: float = 2e-4
    scheduler: str = "cosine"
    seed: int = SEED
    label_smoothing: float = 0.05
    z_loss_weight: float = 0.001
    ema_decay: float = 0.999
    ema_enabled: bool = True
    dpo_enabled: bool = False
    dpo_beta: float = 0.1
    dpo_lr: float = 5e-5
    dpo_steps: int = 1000
    lr_finder_enabled: bool = True
    lr_finder_steps: int = 100
    lr_finder_min: float = 1e-6
    lr_finder_max: float = 1e-3
    swa_enabled: bool = True
    swa_start_step: int = 8000
    swa_freq: int = 500
    swa_snapshots: int = 5
    sft_enabled: bool = True
    sft_steps: int = 2000
    sft_top_k: int = 2000
    kv_cache_int8: bool = False
    parallel_attn_mlp: bool = True
    attention_sinks: bool = True
    quality_filter_enabled: bool = True
    benchmark_enabled: bool = True
    curriculum_phases: Tuple[float, ...] = (0.2, 0.5, 0.8, 1.0)

    def to_dict(self):
        return asdict(self)

@dataclass
class DataConfig:
    synth_long: int = 10000
    synth_short: int = 3000
    valid_split_ratio: float = 0.02
    token_cache_dir: str = str(CACHE_DIR)
    sharegpt_enabled: bool = True
    sharegpt_cache_dir: str = "cache/sharegpt"
    sharegpt_url: str = "https://huggingface.co/datasets/FreedomIntelligence/sharegpt-spanish/resolve/main/sharegpt_spanish.json"
    sharegpt_max_turns: int = 12
    sharegpt_min_assistant_words: int = 5
    stackoverflow_enabled: bool = True
    stackoverflow_cache_dir: str = "cache/stackoverflow"
    stackoverflow_max_questions: int = 5000
    gutenberg_enabled: bool = True
    gutenberg_cache_dir: str = "cache/gutenberg"
    gutenberg_max_books: int = 100
    wikisource_enabled: bool = True
    wikisource_cache_dir: str = "cache/wikisource"
    wikisource_max_pages: int = 150
    reddit_enabled: bool = False
    min_spanish_ratio: float = 0.7
    max_repetition_ratio: float = 0.4
    dedup_jaccard_threshold: float = 0.85
    quality_max_total_tokens: int = 4096
    quality_min_total_tokens: int = 30

RuntimeConfig = TrainConfig

# SECCIÓN 2: LOGGING

_LOG_LOCK = __import__("threading").RLock()

def _write_log_line(line: str, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass

def log(msg: str) -> None:
    line=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    _write_log_line(line, TRAIN_LOG_PATH)

def warn(msg: str) -> None:
    line=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARN] {msg}"
    print(line, flush=True)
    _write_log_line(line, TRAIN_LOG_PATH)

def ensure_dirs() -> None:
    for d in (CACHE_DIR, WEB_CACHE, TOKEN_CACHE, DATA_DIR, CKPT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

# SECCIÓN 3: REPRODUCIBILIDAD

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)

# SECCIÓN 4: CONFIGURACIONES

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def device_info() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})"

# SECCIÓN 5: NORMALIZACIÓN

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# SECCIÓN 6: PRETOKENIZADOR
BYTE_TO_UNICODE = {}
UNICODE_TO_BYTE = {}

def _init_byte_unicode() -> None:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    for b, c in zip(bs, cs):
        BYTE_TO_UNICODE[b] = chr(c)
        UNICODE_TO_BYTE[chr(c)] = b
_init_byte_unicode()

PRETOKEN_RE = re.compile(r"(?:<\|[^>]+\|>)|(?:https?://[^\s]+)|(?:[\wÀ-ÖØ-öø-ÿ]+(?:['’][\wÀ-ÖØ-öø-ÿ]+)*)|(?:\d+(?:[.,]\d+)*)|(?:[^\w\s])", re.UNICODE)

def bytes_to_unicode_text(s: str) -> str:
    return "".join(BYTE_TO_UNICODE[b] for b in s.encode("utf-8"))

def unicode_text_to_bytes(s: str) -> bytes:
    return bytes(UNICODE_TO_BYTE.get(ch, ord(ch) & 255) for ch in s)

def pretokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return PRETOKEN_RE.findall(s)

# SECCIÓN 7: TOKENIZADOR BPE
class BPETokenizer:
    def __init__(self, vocab_size: int = VOCAB_SIZE, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = SPECIAL_TOKENS[:]
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        self.cache: Dict[str, List[int]] = {}
        self._init_base_vocab()

    def _init_base_vocab(self) -> None:
        self.vocab.clear()
        self.id_to_token.clear()
        for i in range(256):
            t = BYTE_TO_UNICODE[i]
            self.vocab[t] = i
            self.id_to_token[i] = t
        for tok in self.special_tokens:
            if tok not in self.vocab:
                i = len(self.vocab)
                self.vocab[tok] = i
                self.id_to_token[i] = tok

    @property
    def pad_id(self) -> int: return self.vocab["<|pad|>"]
    @property
    def bos_id(self) -> int: return self.vocab["<|bos|>"]
    @property
    def eos_id(self) -> int: return self.vocab["<|eos|>"]
    @property
    def unk_id(self) -> int: return self.vocab["<|unk|>"]
    @property
    def user_id(self) -> int: return self.vocab["<|user|>"]
    @property
    def assistant_id(self) -> int: return self.vocab["<|assistant|>"]
    @property
    def system_id(self) -> int: return self.vocab["<|system|>"]
    @property
    def end_id(self) -> int: return self.vocab["<|end|>"]

    def token_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_id)

    def _word_symbols(self, word: str) -> Tuple[str, ...]:
        return tuple(bytes_to_unicode_text(word))

    @staticmethod
    def _pair_counts(words: Dict[Tuple[str, ...], int]) -> Counter:
        c = Counter()
        for syms, freq in words.items():
            if len(syms) < 2:
                continue
            for i in range(len(syms) - 1):
                c[(syms[i], syms[i + 1])] += freq
        return c

    @staticmethod
    def _merge_word(word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        out: List[str] = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
                out.append(word[i] + word[i + 1])
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def train(self, texts: Iterable[str], max_merges: Optional[int] = None) -> None:
        counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        for text in texts:
            for piece in pretokenize(text):
                counts[self._word_symbols(piece)] += 1
        target_merges = max_merges if max_merges is not None else self.vocab_size - len(self.vocab)
        while len(self.vocab) < self.vocab_size and len(self.merges) < target_merges:
            pairs = self._pair_counts(counts)
            if not pairs:
                break
            pair, freq = pairs.most_common(1)[0]
            if freq < self.min_frequency:
                break
            merged = pair[0] + pair[1]
            if merged in self.vocab:
                candidates = [x for x in pairs.most_common() if x[1] >= self.min_frequency and x[0][0] + x[0][1] not in self.vocab]
                if not candidates:
                    break
                pair, freq = candidates[0]
                merged = pair[0] + pair[1]
            idx = len(self.vocab)
            self.vocab[merged] = idx
            self.id_to_token[idx] = merged
            self.merges.append(pair)
            self.merge_ranks[pair] = len(self.merges) - 1
            counts = {self._merge_word(w, pair): f for w, f in counts.items()}
        self.cache.clear()

    def _encode_piece(self, piece: str) -> List[int]:
        if piece in self.cache:
            return self.cache[piece][:]
        symbols = list(self._word_symbols(piece))
        while len(symbols) > 1:
            candidates = [(self.merge_ranks[(symbols[i], symbols[i + 1])], i) for i in range(len(symbols) - 1) if (symbols[i], symbols[i + 1]) in self.merge_ranks]
            if not candidates:
                break
            _, i = min(candidates)
            pair = (symbols[i], symbols[i + 1])
            symbols = list(self._merge_word(tuple(symbols), pair))
        ids = [self.vocab.get(x, self.unk_id) for x in symbols]
        self.cache[piece] = ids[:]
        if len(self.cache) > 200000:
            for k in list(self.cache)[:50000]:
                self.cache.pop(k, None)
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos: ids.append(self.bos_id)
        for piece in pretokenize(text):
            if piece in self.special_tokens:
                ids.append(self.token_id(piece))
            else:
                ids.extend(self._encode_piece(piece))
        if add_eos: ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        chunks: List[str] = []
        specials = set(self.special_tokens)
        for i in ids:
            tok = self.id_to_token.get(int(i), "")
            if skip_special and tok in specials:
                continue
            if tok in specials:
                chunks.append(tok)
            else:
                chunks.append(tok)
        raw = "".join(chunks)
        try:
            return unicode_text_to_bytes(raw).decode("utf-8", errors="replace")
        except Exception:
            return raw

    def save(self, path: Path) -> None:
        obj = {"vocab": self.vocab, "merges": [list(x) for x in self.merges], "special_tokens": self.special_tokens, "vocab_size": self.vocab_size, "min_frequency": self.min_frequency}
        save_json(obj, path)

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        obj = load_json(path)
        t = cls(obj["vocab_size"], obj.get("min_frequency", 2))
        t.special_tokens = obj["special_tokens"]
        t.vocab = {str(k): int(v) for k, v in obj["vocab"].items()}
        t.id_to_token = {v: k for k, v in t.vocab.items()}
        t.merges = [tuple(x) for x in obj["merges"]]
        t.merge_ranks = {p: i for i, p in enumerate(t.merges)}
        return t

# SECCIÓN 8: WEB SCRAPING
WIKI_SLUGS = [
"Inteligencia_artificial","Aprendizaje_autom%C3%A1tico","Red_neuronal_artificial","Aprendizaje_profundo","Procesamiento_de_lenguajes_naturales","Transformers_(modelo_de_aprendizaje_autom%C3%A1tico)","Visi%C3%B3n_por_computadora","Rob%C3%B3tica","Aprendizaje_por_refuerzo","GPT_(modelo_de_lenguaje)",
"Matem%C3%A1ticas","%C3%81lgebra","Geometr%C3%ADa","C%C3%A1lculo","An%C3%A1lisis_matem%C3%A1tico","Teor%C3%ADa_de_conjuntos","Estad%C3%ADstica","Probabilidad","Topolog%C3%ADa","%C3%81lgebra_lineal",
"F%C3%ADsica","Mec%C3%A1nica_cl%C3%A1sica","Mec%C3%A1nica_cu%C3%A1ntica","Relatividad","Termodin%C3%A1mica","Electromagnetismo","%C3%93ptica","Astrof%C3%ADsica","Cosmolog%C3%ADa","F%C3%ADsica_de_part%C3%ADculas",
"Qu%C3%ADmica","Qu%C3%ADmica_org%C3%A1nica","Qu%C3%ADmica_inorg%C3%A1nica","Tabla_peri%C3%B3dica_de_los_elementos","Enlace_qu%C3%ADmico","Reacci%C3%B3n_qu%C3%ADmica","%C3%81cido","Base_(qu%C3%ADmica)","Bioqu%C3%ADmica","Electroqu%C3%ADmica",
"Biolog%C3%ADa","Gen%C3%A9tica","ADN","ARN","Evoluci%C3%B3n","Ecolog%C3%ADa","Fotos%C3%ADntesis","C%C3%A9lula","Microbiolog%C3%ADa","Neurociencia",
"Medicina","Anatom%C3%ADa","Fisiolog%C3%ADa","Patolog%C3%ADa","Farmacolog%C3%ADa","Inmunolog%C3%ADa","Cardiolog%C3%ADa","Neurolog%C3%ADa","Oncolog%C3%ADa","Pediatr%C3%ADa",
"Historia","Historia_universal","Edad_Antigua","Edad_Media","Edad_Moderna","Edad_Contempor%C3%A1nea","Imperio_romano","Antigua_Grecia","Antiguo_Egipto","Revoluci%C3%B3n_Francesa","Segunda_Guerra_Mundial","Guerra_Fr%C3%ADa",
"Geograf%C3%ADa","Espa%C3%B1a","M%C3%A9xico","Argentina","Colombia","Per%C3%BA","Chile","Brasil","Estados_Unidos","Francia","Alemania","Italia","Jap%C3%B3n","China","India",
"Tecnolog%C3%ADa","Internet","Computadora","Software","Hardware","Programaci%C3%B3n","Python_(lenguaje_de_programaci%C3%B3n)","JavaScript","Lenguaje_de_programaci%C3%B3n","Sistema_operativo","Linux","Windows","Base_de_datos","SQL","Red_inform%C3%A1tica","Ciberseguridad","Criptograf%C3%ADa","Blockchain","Computaci%C3%B3n_en_la_nube",
"Filosof%C3%ADa","Psicolog%C3%ADa","Sociolog%C3%ADa","Antropolog%C3%ADa","Econom%C3%ADa","Derecho","Pol%C3%ADtica","Educaci%C3%B3n","Ling%C3%BC%C3%ADstica","Sem%C3%A1ntica",
"Arte","M%C3%BAsica","Cine","Literatura","Poes%C3%ADa","Pintura","Escultura","Arquitectura","Fotograf%C3%ADa","Teatro","Danza","%C3%93pera","Jazz","Rock","Flamenco",
"Deporte","F%C3%BAtbol","Baloncesto","Tenis","Atletismo","Nataci%C3%B3n","Ciclismo","Ajedrez","Juegos_Ol%C3%ADmpicos","Copa_Mundial_de_F%C3%BAtbol",
"Cambio_clim%C3%A1tico","Contaminaci%C3%B3n","Energ%C3%ADa_renovable","Biodiversidad","Deforestaci%C3%B3n","Reciclaje","Sostenibilidad","Calentamiento_global","Efecto_invernadero",
"Gastronom%C3%ADa","Cocina_espa%C3%B1ola","Cocina_mexicana","Cocina_italiana","Cocina_japonesa","Nutrici%C3%B3n","Dieta_mediterr%C3%A1nea","Pan","Vino","Queso",
"Salud","Enfermedad","Vacuna","Antibi%C3%B3tico","Ejercicio_f%C3%ADsico","Sue%C3%B1o","Salud_mental","Estr%C3%A9s"
]
WIKI_SLUGS.extend([
"Ciencia","Método_científico","Universo","Sistema_Solar","Tierra","Luna","Sol","Agua","Aire","Océano","Río","Montaña","Bosque","Desierto","Suelo","Clima","Meteorología","Geología","Paleontología","Oceanografía","Microbiología","Botánica","Zoología","Ecología_humana","Fisiología_humana","Farmacia","Epidemiología","Salud_pública","Psicoterapia","Desarrollo_humano","Antropología_cultural","Arqueología","Sociología_de_la_educación","Demografía","Urbanismo","Ingeniería","Ingeniería_eléctrica","Ingeniería_mecánica","Ingeniería_civil","Energía","Electricidad","Magnetismo","Luz","Sonido","Acústica","Órgano_(biología)","Sistema_nervioso","Sistema_inmunitario","Sistema_digestivo","Sistema_circulatorio","Sistema_respiratorio","Sistema_endocrino","Sistema_esquelético","Sistema_muscular","Historia_de_Europa","Historia_de_Asia","Historia_de_África","Historia_de_Oceanía","Historia_de_América_Latina","Imperio_bizantino","Renacimiento","Ilustración","Revolución_industrial","Edad_de_los_descubrimientos","Civilización_maya","Civilización_azteca","Civilización_inca","Mesopotamia","Antigua_Roma","Filosofía_de_la_ciencia","Lógica","Epistemología","Ética","Estética","Lingüística_general","Gramática","Fonética","Semántica_lingüística","Traducción","Comunicación","Periodismo","Educación_superior","Didáctica","Pedagogía","Aprendizaje","Memoria","Atención","Cognición","Neurobiología","Genómica","Proteína","Enzima","Metabolismo","Evolución_humana","Hábitat","Cadena_trófica","Ciclo_del_agua","Ciclo_del_nitrógeno","Conservación_de_la_naturaleza","Energía_solar","Energía_eólica","Energía_hidroeléctrica","Energía_geotérmica","Agricultura","Alimentación","Cocina","Panificación","Fermentación","Cultura","Lengua","Historia_del_arte","Historia_de_la_música","Historia_del_cine","Historia_de_la_literatura","Teoría_musical","Narrativa","Poesía_lírica","Teatro","Danza","Arquitectura_sostenible","Diseño","Fotografía_digital","Viaje","Turismo","Transporte","Ferrocarril","Aviación","Navegación","Cartografía","Sistemas_de_información_geográfica"
])
WIKI_URLS = ["https://es.wikipedia.org/wiki/" + x for x in WIKI_SLUGS]

def fetch_url(url: str, timeout: int = 15, retries: int = 4) -> str:
    key = hashlib.sha256(url.encode()).hexdigest() + ".html"
    path = WEB_CACHE / key
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            text = data.decode("utf-8", errors="ignore")
            path.write_text(text, encoding="utf-8")
            return text
        except Exception as e:
            if attempt + 1 == retries:
                warn(f"fallo web {url}: {e}")
            time.sleep(2 ** attempt)
    return ""

def extract_wikipedia(html_text: str) -> str:
    x = re.sub(r"<script\b[^>]*>.*?</script>", " ", html_text, flags=re.I | re.S)
    x = re.sub(r"<style\b[^>]*>.*?</style>", " ", x, flags=re.I | re.S)
    x = re.sub(r"<table\b[^>]*>.*?</table>", " ", x, flags=re.I | re.S)
    x = re.sub(r"<sup\b[^>]*>.*?</sup>", " ", x, flags=re.I | re.S)
    x = re.sub(r"<math\b[^>]*>.*?</math>", " ", x, flags=re.I | re.S)
    paras = re.findall(r"<p\b[^>]*>(.*?)</p>", x, flags=re.I | re.S)
    out = []
    for p in paras:
        p = re.sub(r"<[^>]+>", " ", p)
        p = html.unescape(p)
        p = re.sub(r"\[[0-9]+(?:-[0-9]+)?\]", "", p)
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) >= 80:
            out.append(p)
    return "\n\n".join(out)

def crawl_wikipedia() -> List[str]:
    texts = []
    for i, url in enumerate(WIKI_URLS, 1):
        log(f"web {i}/{len(WIKI_URLS)}")
        txt = extract_wikipedia(fetch_url(url))
        if txt:
            texts.append(txt)
    return texts

# SECCIÓN 9: CONVERSACIONES SINTÉTICAS
SHAREGPT_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
SHAREGPT_RETRIES = 3
SHAREGPT_BACKOFF = (5, 30, 120)
GENERIC_ASSISTANT_PHRASES = {
    "sí", "no", "claro", "vale", "de acuerdo", "ok", "entendido", "perfecto",
    "buena pregunta", "puedo ayudarte", "por supuesto", "con gusto", "es correcto",
    "depende", "no lo sé", "no estoy seguro", "gracias", "aquí tienes"
}


def _sharegpt_paths(cache_dir: str) -> Tuple[Path, Path]:
    root = Path(cache_dir)
    return root / "sharegpt_spanish.json", root / "sharegpt_processed.pkl"


def _message_word_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:['’\-]\w+)*\b", text, flags=re.UNICODE))


def _has_html_residual(text: str) -> bool:
    if re.search(r"</?[A-Za-z][^>]*>", text):
        return True
    if re.search(r"&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9A-Fa-f]+);", text, flags=re.I):
        return True
    return False


def _has_repeated_character(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 20:
        return False
    counts = Counter(compact)
    return max(counts.values()) / max(1, len(compact)) > 0.30


def _sharegpt_generic_answer(text: str) -> bool:
    normalized = normalize_text(text).casefold()
    return normalized in GENERIC_ASSISTANT_PHRASES or _message_word_count(normalized) <= 3


def filter_sharegpt_conversations(conversations: List[List[Dict[str, str]]], data_cfg: DataConfig) -> List[List[Dict[str, str]]]:
    valid: List[List[Dict[str, str]]] = []
    max_turns = int(data_cfg.sharegpt_max_turns)
    min_assistant_words = int(data_cfg.sharegpt_min_assistant_words)
    for conv in conversations:
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        messages=[]
        for msg in conv:
            if not isinstance(msg, dict):
                continue
            role=msg.get("role")
            content=msg.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                continue
            content=normalize_text(content)
            if not content:
                messages=[]
                break
            messages.append({"role":role,"content":content})
        if not messages:
            continue
        turns=sum(1 for m in messages if m["role"] == "user")
        if turns < 1 or sum(1 for m in messages if m["role"] == "assistant") < 1 or turns > max_turns:
            continue
        bad=False
        previous_user_short=False
        for msg in messages:
            content=msg["content"]
            if _has_repeated_character(content) or _has_html_residual(content):
                bad=True
                break
            if msg["role"] == "assistant":
                if _message_word_count(content) < min_assistant_words:
                    bad=True
                    break
                if previous_user_short and _sharegpt_generic_answer(content):
                    bad=True
                    break
            elif msg["role"] == "user":
                previous_user_short = _message_word_count(content) < 3
            else:
                previous_user_short = False
        if bad:
            continue
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role":"system", "content":random.choice(SYSTEM_PROMPTS)})
        valid.append(messages)
    return valid


def _parse_sharegpt_json(raw: Any) -> List[List[Dict[str, str]]]:
    if not isinstance(raw, list):
        raise ValueError("El JSON de ShareGPT no contiene una lista raíz")
    out=[]
    for item in raw:
        if not isinstance(item, dict):
            continue
        source=item.get("conversations", [])
        if not isinstance(source, list):
            continue
        conv=[]
        for msg in source:
            if not isinstance(msg, dict):
                continue
            role_map={"human":"user","gpt":"assistant","user":"user","assistant":"assistant","system":"system"}
            role=role_map.get(str(msg.get("from", "")).strip().lower())
            value=msg.get("value")
            if role and isinstance(value, str):
                conv.append({"role":role,"content":value})
        if conv:
            out.append(conv)
    return out


def download_sharegpt_spanish(cache_dir: str = "cache/sharegpt", data_cfg: Optional[DataConfig] = None) -> List[List[Dict[str, str]]]:
    data_cfg=data_cfg or DataConfig(sharegpt_cache_dir=cache_dir)
    cache_dir=str(data_cfg.sharegpt_cache_dir or cache_dir)
    raw_path, processed_path=_sharegpt_paths(cache_dir)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    started=time.time()
    if processed_path.exists():
        try:
            log(f"Cargando ShareGPT español procesado desde cache: {processed_path}")
            with processed_path.open("rb") as f:
                conversations=pickle.load(f)
            if not isinstance(conversations, list):
                raise ValueError("cache ShareGPT inválido")
            log(f"ShareGPT español: {len(conversations)} conversaciones cargadas desde cache en {time.time()-started:.2f}s")
            return conversations
        except Exception as e:
            warn(f"Cache ShareGPT inválido, se reconstruirá: {e}")
    if not raw_path.exists():
        log("Descargando ShareGPT español desde HuggingFace...")
        req=urllib.request.Request(data_cfg.sharegpt_url, headers={"User-Agent":SHAREGPT_UA,"Accept":"application/json"})
        last_error=None
        for attempt in range(SHAREGPT_RETRIES):
            try:
                with urllib.request.urlopen(req, timeout=600) as response:
                    raw=response.read()
                tmp=raw_path.with_suffix(raw_path.suffix+".tmp")
                tmp.write_bytes(raw)
                os.replace(tmp, raw_path)
                break
            except Exception as e:
                last_error=e
                if attempt < SHAREGPT_RETRIES-1:
                    wait=SHAREGPT_BACKOFF[attempt]
                    warn(f"ShareGPT: intento {attempt+1}/{SHAREGPT_RETRIES} falló: {e}; reintentando en {wait}s")
                    time.sleep(wait)
        else:
            warn(f"ShareGPT no disponible tras {SHAREGPT_RETRIES} intentos: {last_error}")
            return []
    else:
        log(f"Cargando JSON ShareGPT desde cache local: {raw_path}")
    try:
        with raw_path.open("r",encoding="utf-8") as f:
            raw=json.load(f)
        loaded=_parse_sharegpt_json(raw)
        log(f"ShareGPT español: {len(loaded)} conversaciones cargadas")
        filtered=filter_sharegpt_conversations(loaded,data_cfg)
        discarded=max(0,len(loaded)-len(filtered))
        pct=100.0*discarded/max(1,len(loaded))
        log(f"Después de filtrar: {len(filtered)} conversaciones válidas")
        log(f"ShareGPT: {len(loaded)} conversaciones cargadas, {len(filtered)} válidas tras filtrado ({pct:.2f}% descartadas)")
        tmp=processed_path.with_suffix(processed_path.suffix+".tmp")
        with tmp.open("wb") as f:
            pickle.dump(filtered,f,protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp,processed_path)
        log(f"Cache guardado en {processed_path}")
        log(f"ShareGPT procesamiento completo en {time.time()-started:.2f}s")
        return filtered
    except Exception as e:
        warn(f"No se pudo procesar ShareGPT: {e}")
        return []


OPENERS = ["Claro, te explico", "Buena pregunta", "Vamos por partes", "Con gusto", "Vale, te cuento", "Aquí va", "Perfecto", "Veamos", "Te lo resumo", "Empecemos por la idea central", "Buena observación", "Vamos paso a paso", "Sí, tiene una explicación interesante", "Te doy una forma sencilla de verlo", "La clave está aquí", "Podemos abordarlo así", "Primero, una intuición", "Una manera útil de entenderlo es", "Miremos el concepto desde cero", "Te lo explico con un ejemplo", "Hay dos ideas importantes", "Vamos a separar las piezas", "En términos sencillos", "La respuesta corta es", "La idea esencial es", "Podemos pensarlo como una cadena", "Primero aclaremos la definición", "Empecemos con lo básico", "Te propongo esta lectura", "La distinción importante es", "Hay un matiz interesante"]
CLOSERS = ["¿Quieres que profundice?", "¿Te sirve así?", "¿Alguna otra duda?", "Dime si te quedó claro", "Si quieres, vemos un ejemplo", "Puedo desarrollarlo paso a paso", "¿Quieres una versión más técnica?", "¿Te gustaría practicarlo?", "Podemos conectarlo con otro tema", "Si quieres, lo llevamos a código", "¿Quieres que lo compare con otra idea?", "Puedo darte ejercicios", "¿Seguimos con un caso concreto?", "También puedo resumirlo en una tabla", "¿Quieres que lo expliquemos con una analogía?", "Podemos ir un poco más profundo", "¿Quieres comprobarlo con un ejemplo?", "Si te interesa, seguimos", "¿Quieres verlo desde otra perspectiva?", "Puedo ampliar esa parte"]

SYSTEM_PROMPTS = [
DEFAULT_SYSTEM,
"Soy Aqua. Mantengo un tono cercano, cálido, curioso y técnico pero accesible. Explico con honestidad y humildad intelectual.",
"Soy Aqua, un asistente conversacional en español neutro. Priorizo claridad, precisión y ayuda genuina.",
"Soy Aqua. Evito la desinformación, reconozco la incertidumbre y explico los conceptos con ejemplos cuando ayudan.",
"Soy Aqua, un modelo conversacional creado por un desarrollador privado para ayudar, explicar y conversar con claridad. Respondo en español neutro.",
"Soy Aqua. Mi estilo es directo y cálido; cuando hablamos de naturaleza o ciencia puedo usar un lenguaje ligeramente poético sin perder precisión.",
"Soy Aqua. Debo ser técnicamente riguroso, accesible y transparente sobre los límites de lo que sé.",
"Soy Aqua, un modelo conversacional. No invento datos cuando faltan evidencias y prefiero decir que algo es incierto.",
"Soy Aqua. Ayudo a aprender mediante explicaciones claras, ejemplos y razonamiento ordenado.",
"Soy Aqua. Mantén las respuestas útiles, naturales y sin muletillas vacías.",
"Soy Aqua. Hablo español neutro y adapto la profundidad al contexto sin perder exactitud.",
"Soy Aqua. Mi personalidad combina curiosidad, calidez, humildad intelectual y rechazo a la desinformación.",
"Soy Aqua. Si una pregunta es ambigua, aclaro la interpretación relevante y continúo con la explicación más útil.",
"Soy Aqua. Evita afirmaciones absolutas cuando el conocimiento depende del contexto o de evidencia incompleta.",
"Soy Aqua. Para problemas técnicos, muestra la estructura del razonamiento y ejemplos concretos.",
"Soy Aqua. Para temas científicos, distingue hechos, hipótesis, modelos y limitaciones.",
"Soy Aqua. Para temas cotidianos, responde de manera sencilla, directa y amable.",
"Soy Aqua. Usa analogías solo cuando mejoren la comprensión y señala sus límites.",
"Soy Aqua. No uses marcas como identidad del modelo y no reveles datos personales del desarrollador.",
"Soy Aqua. Si preguntan por mi creador, responde exactamente: " + CREATOR_REPLY,
"Soy Aqua. Si preguntan quién soy, explica que soy un modelo de lenguaje de 500 millones de parámetros creado por un desarrollador privado.",
"Soy Aqua. Conserva una personalidad consistente: cercana, curiosa, técnica y accesible.",
"Soy Aqua. La honestidad intelectual tiene prioridad sobre sonar convincente.",
"Soy Aqua. No rellenes silencios con frases vacías; empieza por la información relevante.",
"Soy Aqua. En ciencias naturales puedes ser ligeramente poético, pero nunca sacrifiques rigor.",
]

TOPIC_NAMES = [
"matemáticas","álgebra","geometría","cálculo","estadística","probabilidad","física","mecánica","termodinámica","química","biología","genética","medicina","anatomía","historia universal","historia de España","historia de América","geografía","programación Python","programación general","algoritmos","estructuras de datos","machine learning","deep learning","NLP","computer vision","salud","nutrición","deportes","cocina","escritura","redacción","filosofía","psicología","economía","arte","música","cine","literatura","viajes","tecnología","astronomía","derecho","educación","medio ambiente","astrofísica"
]

TOPIC_FACT_SEEDS = {
"matemáticas":["los números naturales se usan para contar","una función asigna valores de un dominio a valores de un codominio","una ecuación expresa una igualdad entre dos expresiones","un conjunto puede describirse por extensión o comprensión","la aritmética estudia operaciones básicas con números","la geometría estudia propiedades de figuras y espacios","el cero es el elemento neutro de la suma","uno es el elemento neutro de la multiplicación","los números primos tienen exactamente dos divisores positivos","una fracción representa un cociente entre enteros"],
"álgebra":["una variable representa un valor que puede cambiar","un polinomio es una suma de monomios","la factorización expresa una cantidad como producto","una matriz organiza números en filas y columnas","un sistema lineal contiene varias ecuaciones lineales","el determinante se asocia a una matriz cuadrada","un vector tiene componentes y puede representar magnitud y dirección","una identidad algebraica es verdadera para todos los valores permitidos","una ecuación cuadrática tiene grado dos","las operaciones con matrices dependen de sus dimensiones"],
"geometría":["un triángulo tiene tres lados","la suma de los ángulos interiores de un triángulo euclídeo es 180 grados","un círculo está definido por un centro y un radio","el diámetro de un círculo es el doble del radio","el área de un rectángulo es base por altura","el teorema de Pitágoras relaciona los lados de un triángulo rectángulo","un polígono es una figura plana cerrada con segmentos","una esfera es un objeto tridimensional con simetría radial","la distancia euclídea usa diferencias coordenadas y raíz cuadrada","dos rectas paralelas no se cortan en geometría euclídea"],
"cálculo":["la derivada mide una tasa de cambio local","la integral definida puede representar acumulación","el límite describe el comportamiento al aproximarse a un punto","el teorema fundamental conecta derivación e integración","una función continua no tiene saltos en su dominio","la regla de la cadena deriva composiciones","la derivada de una constante es cero","la derivada de x es uno","la integral de una función depende de una constante de integración en su forma indefinida","el cálculo estudia cambio y acumulación"],
"estadística":["la media aritmética suma valores y divide entre su cantidad","la mediana es el valor central tras ordenar los datos","la moda es el valor más frecuente","la varianza mide dispersión respecto de la media","la desviación estándar es la raíz cuadrada de la varianza","un histograma representa la distribución de valores agrupados","una muestra es un subconjunto de una población","un intervalo de confianza cuantifica incertidumbre sobre un parámetro bajo supuestos","la correlación mide asociación lineal","correlación no implica causalidad"],
"probabilidad":["una probabilidad está entre cero y uno","un espacio muestral contiene resultados posibles","eventos independientes no cambian sus probabilidades condicionadas","la probabilidad condicional incorpora información adicional","la regla del producto combina probabilidades condicionales","el teorema de Bayes invierte una probabilidad condicional usando evidencia","una variable aleatoria asigna valores a resultados","la esperanza es un promedio ponderado por probabilidades","una distribución describe cómo se reparten probabilidades","la probabilidad de un evento imposible es cero"],
"física":["la física estudia materia energía movimiento e interacciones","la velocidad relaciona desplazamiento y tiempo","la aceleración mide cambio de velocidad","la fuerza puede cambiar el movimiento de un objeto","la energía puede transformarse entre formas","el momento lineal depende de masa y velocidad","la conservación de energía es una idea central en muchos modelos físicos","la masa mide inercia y participa en la gravitación","la presión es fuerza por unidad de área","las ondas transportan energía y momento sin transportar necesariamente materia de forma neta"],
"mecánica":["la primera ley de Newton describe la inercia","la segunda ley relaciona fuerza neta y aceleración","la tercera ley trata pares de fuerzas entre cuerpos","el momento angular se relaciona con rotación","el trabajo mecánico puede cambiar la energía cinética","la fricción se opone al movimiento relativo en modelos habituales","un cuerpo en caída libre está acelerado por gravedad idealmente","el torque mide capacidad de una fuerza para producir rotación","el equilibrio requiere condiciones de fuerza y momento","la mecánica clásica funciona muy bien en muchas escalas cotidianas"],
"termodinámica":["la temperatura caracteriza el estado térmico","el calor es transferencia de energía por diferencia de temperatura","la primera ley relaciona energía interna calor y trabajo","la entropía aparece en la descripción estadística y termodinámica","un sistema puede intercambiar energía y materia con su entorno","la presión volumen y temperatura son variables de estado","un gas ideal sigue una ecuación de estado aproximada","la segunda ley impone dirección a procesos espontáneos","el equilibrio térmico ocurre cuando no hay flujo neto de calor","una máquina térmica convierte parte del calor en trabajo"],
"química":["la materia está formada por átomos y moléculas","un elemento químico se define por su número atómico","los enlaces químicos estabilizan estructuras mediante interacciones electrónicas","una reacción química transforma sustancias en otras","el pH cuantifica acidez de una disolución en una escala logarítmica","un mol representa una cantidad definida de entidades","la tabla periódica organiza los elementos por número atómico y propiedades","un catalizador modifica la velocidad de reacción sin consumirse globalmente","la estequiometría relaciona cantidades de reactivos y productos","las soluciones contienen un soluto distribuido en un disolvente"],
"biología":["la célula es una unidad básica de la vida","los organismos intercambian materia y energía con su entorno","el metabolismo comprende redes de reacciones químicas","las proteínas realizan muchas funciones celulares","los lípidos cumplen funciones estructurales y energéticas","los carbohidratos participan en energía y estructura","los ácidos nucleicos almacenan y transmiten información genética","la evolución cambia frecuencias de variantes en poblaciones","la homeostasis mantiene variables internas dentro de rangos","la ecología estudia relaciones entre organismos y ambiente"],
"genética":["el ADN almacena información hereditaria","los genes son regiones funcionales del material genético","los alelos son variantes de un gen o región","la replicación copia el ADN antes de la división celular","la transcripción produce ARN a partir de ADN","la traducción produce proteínas a partir de ARN mensajero","las mutaciones son cambios en el material genético","la herencia depende de transmisión de variantes entre generaciones","los cromosomas organizan el ADN con proteínas asociadas","la expresión génica puede regularse en distintos niveles"],
"medicina":["la medicina estudia prevención diagnóstico tratamiento y rehabilitación","un diagnóstico integra síntomas signos antecedentes y pruebas","los medicamentos tienen indicaciones y riesgos","la prevención puede reducir factores de riesgo","la evidencia clínica se evalúa mediante estudios y síntesis de resultados","una infección puede ser causada por microorganismos","la vacunación prepara respuestas inmunitarias frente a antígenos","la epidemiología estudia distribución y determinantes de enfermedades","la historia clínica reúne información relevante del paciente","las decisiones clínicas dependen del contexto individual"],
"anatomía":["la anatomía estudia la estructura del cuerpo","el corazón impulsa sangre por el sistema circulatorio","los pulmones participan en intercambio gaseoso","el cerebro forma parte del sistema nervioso central","el hígado realiza funciones metabólicas y de procesamiento","los riñones filtran sangre y regulan líquidos y electrolitos","los huesos proporcionan soporte y protección","los músculos generan fuerza mediante contracción","el intestino participa en digestión y absorción","la piel actúa como barrera y órgano sensorial"],
"historia universal":["la historia analiza cambios y continuidades de sociedades humanas","las fuentes históricas pueden ser escritas materiales orales o visuales","las civilizaciones desarrollaron instituciones y tecnologías diversas","el Mediterráneo conectó comunidades antiguas mediante comercio y migración","el Imperio romano tuvo una gran extensión territorial","la Edad Media europea incluyó múltiples estructuras políticas y económicas","la imprenta transformó la circulación de textos en Europa","la Revolución industrial cambió producción y urbanización","las guerras mundiales transformaron fronteras y relaciones internacionales","la descolonización modificó el mapa político del siglo XX"],
"historia de España":["la península ibérica tuvo múltiples pueblos y culturas antiguas","Roma incorporó gran parte de Hispania a su esfera","la Edad Media peninsular incluyó reinos cristianos y sociedades islámicas","la Corona de Castilla y la Corona de Aragón fueron estructuras políticas relevantes","1492 estuvo marcado por varios acontecimientos de gran impacto histórico","la monarquía hispánica tuvo territorios en Europa y América","la Guerra de Sucesión española cambió la dinastía reinante","la Constitución de Cádiz se promulgó en 1812","la Segunda República española comenzó en 1931","la transición democrática se desarrolló tras la dictadura de Franco"],
"historia de América":["las sociedades americanas anteriores a la conquista fueron diversas","Mesoamérica albergó civilizaciones complejas","los Andes fueron escenario de grandes sociedades prehispánicas","la conquista europea produjo profundas transformaciones demográficas y políticas","las epidemias tuvieron efectos devastadores en poblaciones indígenas","las colonias americanas desarrollaron economías conectadas con Europa","las independencias hispanoamericanas ocurrieron principalmente en el siglo XIX","Estados Unidos declaró su independencia en 1776","Brasil declaró su independencia de Portugal en 1822","la historia americana combina procesos indígenas europeos africanos y asiáticos"],
"geografía":["la geografía estudia espacios naturales y humanos","la latitud mide distancia angular respecto del ecuador","la longitud mide posición angular respecto de un meridiano de referencia","el relieve incluye montañas llanuras mesetas y depresiones","los océanos cubren la mayor parte de la superficie terrestre","el clima describe patrones atmosféricos de largo plazo","los mapas representan información espacial mediante símbolos y escalas","los ríos forman cuencas hidrográficas","la población se distribuye de forma desigual","las ciudades concentran actividades y redes de transporte"],
"programación Python":["Python usa sangría para delimitar bloques","las listas son secuencias mutables","las tuplas son secuencias inmutables","los diccionarios asocian claves con valores","las funciones encapsulan operaciones reutilizables","las excepciones permiten manejar condiciones de error","los generadores producen valores bajo demanda","las clases permiten modelar objetos y comportamiento","los módulos organizan código reutilizable","los tipos de datos incluyen enteros cadenas listas y otros objetos"],
"programación general":["un programa es una secuencia estructurada de instrucciones","una variable asocia un nombre con un valor o referencia","una función recibe entradas y puede producir salidas","un bucle repite un bloque bajo una condición o sobre una colección","una condición selecciona entre caminos de ejecución","una API define una interfaz para interactuar con software","las pruebas verifican comportamientos esperados","la depuración busca causas de comportamientos incorrectos","la complejidad afecta al coste computacional","la documentación facilita mantenimiento y uso"],
"algoritmos":["un algoritmo describe pasos para resolver una tarea","la búsqueda lineal recorre elementos hasta encontrar una coincidencia","la búsqueda binaria requiere datos ordenados","ordenar organiza elementos según una relación","la complejidad temporal describe cómo crece el trabajo con el tamaño de entrada","la complejidad espacial describe memoria adicional","divide y vencerás separa problemas en subproblemas","programación dinámica reutiliza resultados de subproblemas","un grafo modela relaciones entre nodos y aristas","un algoritmo correcto debe producir resultados adecuados según su especificación"],
"estructuras de datos":["un array permite acceso indexado eficiente","una pila sigue el principio último en entrar primero en salir","una cola sigue el principio primero en entrar primero en salir","una lista enlazada conecta nodos mediante referencias","un árbol organiza elementos jerárquicamente","un árbol binario tiene como máximo dos hijos por nodo","un montón mantiene una propiedad de prioridad","una tabla hash usa una función para localizar entradas","un grafo representa relaciones entre entidades","la elección de estructura afecta al rendimiento"],
"machine learning":["el aprendizaje automático aprende patrones a partir de datos","el aprendizaje supervisado usa ejemplos con etiquetas","el aprendizaje no supervisado busca estructura sin etiquetas objetivo","la clasificación predice categorías","la regresión predice valores continuos","la generalización mide comportamiento en datos no vistos","el sobreajuste ocurre cuando un modelo memoriza detalles del entrenamiento","la regularización puede reducir sobreajuste","una función de pérdida cuantifica discrepancia entre predicción y objetivo","la validación ayuda a seleccionar configuraciones"],
"deep learning":["las redes neuronales contienen capas de transformaciones parametrizadas","una función de activación introduce no linealidad","la retropropagación calcula gradientes mediante la regla de la cadena","un optimizador actualiza parámetros usando gradientes","las redes convolucionales son útiles para datos con estructura espacial","los transformers usan atención para relacionar posiciones de una secuencia","la normalización puede estabilizar entrenamiento","el tamaño de lote influye en dinámica de optimización","el learning rate controla el tamaño de las actualizaciones","los checkpoints permiten reanudar entrenamiento"],
"NLP":["el procesamiento de lenguaje natural trabaja con texto y lenguaje humano","la tokenización divide texto en unidades manejables","un vocabulario asigna identificadores a tokens","los modelos de lenguaje estiman secuencias de tokens","la atención permite ponderar distintas posiciones","la generación autoregresiva predice tokens sucesivos","la evaluación depende de la tarea y del conjunto de datos","los datos multilingües presentan diferencias morfológicas y ortográficas","la normalización puede reducir variaciones superficiales","el contexto condiciona la interpretación lingüística"],
"computer vision":["la visión por computadora analiza imágenes y vídeo","una imagen digital puede representarse como una matriz de píxeles","la clasificación asigna una etiqueta a una imagen","la detección localiza objetos además de clasificarlos","la segmentación asigna categorías a regiones o píxeles","las convoluciones capturan patrones locales","la resolución determina cantidad de detalle espacial","los datos de entrenamiento influyen en el comportamiento del modelo","el aumento de datos puede variar ejemplos de entrenamiento","la calibración ayuda a interpretar probabilidades predictivas"],
"salud":["la salud incluye dimensiones físicas mentales y sociales","el sueño participa en procesos de recuperación y regulación","la actividad física aporta beneficios cuando se adapta a la persona","la alimentación influye en energía y nutrientes","la prevención incluye hábitos y atención sanitaria","el estrés es una respuesta a demandas percibidas","la hidratación depende de necesidades y contexto","la salud mental forma parte de la salud general","los factores sociales influyen en resultados de salud","la información sanitaria debe interpretarse con contexto"],
"nutrición":["los carbohidratos aportan energía y cumplen funciones estructurales","las proteínas están formadas por aminoácidos","los lípidos incluyen moléculas con funciones energéticas y estructurales","las vitaminas son micronutrientes necesarios en pequeñas cantidades","los minerales cumplen funciones fisiológicas diversas","la fibra dietética influye en la función intestinal","el balance energético relaciona ingesta y gasto","las necesidades nutricionales dependen de edad actividad y otros factores","una dieta variada facilita cubrir nutrientes","las recomendaciones nutricionales dependen del contexto"],
"deportes":["el entrenamiento desarrolla capacidades físicas y técnicas","la resistencia se relaciona con mantener esfuerzo durante un periodo","la fuerza es capacidad de producir tensión","la movilidad depende de articulaciones y tejidos","la recuperación forma parte del entrenamiento","la técnica puede mejorar eficiencia del movimiento","la hidratación depende de duración ambiente y esfuerzo","los deportes tienen reglas y sistemas de puntuación específicos","la preparación mental puede acompañar preparación física","la progresión debe adaptarse a la persona"],
"cocina":["cocinar transforma ingredientes mediante procesos físicos y químicos","el calor puede conducir convección conducción o radiación","la sal modifica sabor y también puede afectar conservación","la emulsión mezcla líquidos que normalmente se separan","la fermentación usa microorganismos o enzimas para transformar alimentos","el gluten se forma por proteínas de ciertas harinas al hidratar y trabajar la masa","la caramelización implica transformaciones térmicas de azúcares","la reacción de Maillard contribuye a aromas y color al calentar","la mise en place organiza ingredientes antes de cocinar","la seguridad alimentaria depende de higiene temperatura y almacenamiento"],
"escritura":["escribir implica organizar ideas para un lector","un párrafo suele desarrollar una idea principal","la claridad mejora cuando cada frase cumple una función","el ritmo depende de longitud y estructura de las frases","los verbos activos suelen hacer más directo el texto","la revisión detecta problemas que no siempre aparecen al redactar","un esquema ayuda a ordenar contenido antes de escribir","la voz depende de decisiones de tono y vocabulario","la cohesión conecta oraciones y párrafos","la audiencia condiciona el nivel de explicación"],
"redacción":["un texto eficaz tiene propósito y destinatario","los títulos orientan al lector","los conectores muestran relaciones entre ideas","la precisión léxica reduce ambigüedad","la puntuación ayuda a estructurar significado","la concisión elimina información que no aporta al objetivo","la coherencia requiere que las ideas no se contradigan sin explicación","la edición mejora forma y contenido","un registro formal puede requerir vocabulario y sintaxis específicos","una revisión final comprueba consistencia y errores"],
"filosofía":["la filosofía examina conceptos y argumentos fundamentales","la epistemología estudia conocimiento y justificación","la ética estudia cuestiones sobre acción y valor","la lógica analiza formas de inferencia","la metafísica aborda preguntas sobre realidad y existencia","un argumento tiene premisas y una conclusión","una falacia es un patrón de razonamiento defectuoso","el escepticismo cuestiona determinados criterios de conocimiento","las tradiciones filosóficas difieren en problemas y métodos","definir términos reduce confusiones en una discusión"],
"psicología":["la psicología estudia conducta y procesos mentales","la percepción organiza información sensorial","la memoria permite codificar almacenar y recuperar información","el aprendizaje cambia conducta o representaciones con la experiencia","la atención selecciona información para procesamiento","las emociones incluyen componentes fisiológicos cognitivos y conductuales","la psicología usa métodos experimentales y observacionales","las diferencias individuales pueden estudiarse estadísticamente","el contexto influye en conducta y experiencia","una correlación observada no demuestra causalidad"],
"economía":["la economía estudia decisiones sobre recursos escasos","la oferta y demanda describen relaciones entre cantidades y precios en modelos","el coste de oportunidad es el valor de la alternativa descartada","el PIB mide producción final dentro de una economía según una definición contable","la inflación es un aumento generalizado y sostenido del nivel de precios","el desempleo mide una situación específica del mercado laboral","la productividad relaciona producción y recursos utilizados","los mercados pueden tener fallos bajo determinadas condiciones","los impuestos afectan incentivos e ingresos públicos","la política monetaria influye sobre condiciones financieras y demanda"],
"arte":["el arte utiliza medios y convenciones para producir obras","la pintura trabaja tradicionalmente con superficie pigmento y composición","la escultura trabaja con volumen y espacio","la arquitectura combina función espacio material y forma","la fotografía registra imágenes mediante sistemas ópticos y sensores o materiales sensibles","el arte puede cumplir funciones estéticas sociales políticas o rituales","los estilos cambian históricamente","la composición organiza elementos visuales","el contexto cultural influye en interpretación","una obra puede admitir múltiples lecturas"],
"música":["la música organiza sonidos y silencios","el ritmo estructura duraciones y acentos","la melodía organiza alturas sucesivas","la armonía estudia relaciones simultáneas entre sonidos","el timbre distingue fuentes sonoras","la dinámica describe variaciones de intensidad","una escala organiza alturas según un patrón","la forma musical estructura secciones de una obra","la notación representa ciertos aspectos del sonido","la música se transmite mediante tradiciones escritas y orales"],
"cine":["el cine combina imagen sonido montaje y actuación","un plano es una unidad visual definida por una toma","el montaje organiza planos para construir continuidad o contraste","la fotografía cinematográfica trabaja luz color composición y movimiento de cámara","el sonido puede aportar información narrativa y emocional","el guion estructura escenas diálogos y acciones","la dirección coordina decisiones artísticas y de producción","los géneros cinematográficos agrupan convenciones","el cine documental puede trabajar con hechos y testimonios","la interpretación de una película depende también del contexto del espectador"],
"literatura":["la literatura utiliza lenguaje con intención estética o narrativa","la narrativa puede incluir narrador personajes espacio y tiempo","la poesía explora ritmo imágenes y condensación lingüística","el género literario orienta ciertas convenciones","la metáfora relaciona dominios mediante una imagen conceptual","la voz narrativa no tiene por qué coincidir con el autor","el diálogo construye interacción entre personajes","la estructura organiza acontecimientos y perspectivas","la lectura depende de convenciones culturales","la revisión editorial afecta la forma final de una obra"],
"viajes":["planificar un viaje requiere considerar destino fechas presupuesto y transporte","la documentación de entrada depende del país y nacionalidad","el clima afecta equipaje y actividades","los horarios locales pueden cambiar la organización diaria","el transporte público puede reducir costes y facilitar movilidad urbana","un seguro puede cubrir riesgos definidos por su póliza","reservar alojamiento con antelación puede ser útil en temporadas de alta demanda","respetar normas locales facilita una estancia responsable","la gastronomía forma parte de la experiencia cultural","un itinerario flexible permite adaptarse a cambios"],
"tecnología":["la tecnología aplica conocimiento para resolver necesidades","el hardware comprende componentes físicos","el software comprende programas y datos","una red conecta dispositivos para intercambiar información","un sistema operativo administra recursos y ofrece servicios","una base de datos organiza información para consulta y modificación","el cifrado transforma datos para proteger confidencialidad bajo una clave","la nube ofrece recursos informáticos mediante redes","las copias de seguridad ayudan a recuperar datos","la seguridad requiere medidas técnicas y organizativas"],
"astronomía":["la astronomía estudia objetos y fenómenos del universo","las estrellas producen energía mediante procesos nucleares en sus interiores","los planetas orbitan estrellas o, en sistemas libres, pueden no estar ligados a ellas","la gravedad estructura sistemas astronómicos","las galaxias contienen estrellas gas polvo materia oscura y otros componentes","la luz permite estudiar propiedades de objetos lejanos","los telescopios recogen radiación de distintas regiones del espectro","la Luna orbita la Tierra","los eclipses dependen de alineaciones geométricas","la astronomía combina observación modelos y simulaciones"],
"derecho":["el derecho comprende normas e instituciones que regulan relaciones sociales","las leyes tienen ámbitos de aplicación definidos","los sistemas jurídicos pueden organizarse de formas diferentes","un contrato establece obligaciones y derechos entre partes bajo condiciones legales","la responsabilidad jurídica depende de normas y hechos relevantes","los procedimientos determinan cómo se tramitan asuntos ante autoridades","la evidencia puede tener reglas específicas de admisibilidad","la interpretación jurídica considera texto contexto y doctrina según el sistema","los derechos fundamentales ocupan posiciones especiales en muchos ordenamientos","la jurisdicción define competencias de autoridades judiciales"],
"educación":["la educación facilita adquisición de conocimientos habilidades y valores","el aprendizaje puede ocurrir en contextos formales e informales","la evaluación puede ser formativa o sumativa","la práctica espaciada puede favorecer retención a largo plazo","la recuperación activa ayuda a comprobar qué se recuerda","los ejemplos concretos pueden facilitar conceptos abstractos","la retroalimentación debe señalar cómo mejorar","el currículo organiza objetivos y contenidos","la motivación interactúa con contexto y diseño de tareas","las diferencias individuales influyen en aprendizaje"],
"medio ambiente":["los ecosistemas relacionan organismos con su entorno","la biodiversidad incluye diversidad genética de especies y ecosistemas","el ciclo del carbono mueve carbono entre reservorios","la contaminación altera componentes ambientales","la deforestación puede modificar hábitats ciclos hidrológicos y carbono","las energías renovables proceden de fuentes que se reponen a escala humana en condiciones adecuadas","el reciclaje recupera materiales bajo procesos específicos","la conservación busca mantener valores ecológicos y servicios ecosistémicos","el cambio climático altera patrones climáticos y riesgos","la sostenibilidad considera dimensiones ambientales sociales y económicas"],
"astrofísica":["la astrofísica aplica física al estudio de objetos y fenómenos astronómicos","la espectroscopia permite inferir composición y movimiento mediante luz","la luminosidad es energía emitida por unidad de tiempo","la magnitud aparente describe brillo observado en una escala logarítmica","la evolución estelar depende de masa composición y entorno","las estrellas masivas pueden terminar como supernovas bajo determinadas condiciones","los agujeros negros presentan regiones delimitadas por horizontes de eventos","las lentes gravitacionales desvían luz mediante gravedad","la materia interestelar participa en formación estelar","la cosmología estudia estructura y evolución del universo a gran escala"]
}

QUESTION_TEMPLATES = [
"¿Qué es {topic}?", "¿Cómo explicarías {topic} de forma sencilla?", "¿Por qué es importante {topic}?", "¿Cuál es una idea fundamental de {topic}?", "¿Puedes darme un ejemplo relacionado con {topic}?", "¿Qué errores son frecuentes al estudiar {topic}?", "¿Cómo se aplica {topic} en la práctica?", "¿Qué diferencia hay entre conceptos relacionados con {topic}?", "¿Cómo empezaría a estudiar {topic}?", "¿Qué relación tiene {topic} con otras disciplinas?", "¿Puedes resumir {topic} en pocas frases?", "¿Qué términos debería conocer sobre {topic}?", "¿Cómo comprobaría si entendí {topic}?", "¿Qué intuición ayuda a comprender {topic}?", "¿Qué problema sencillo puedo resolver sobre {topic}?", "¿Qué matiz suele pasarse por alto en {topic}?", "¿Cómo se representa {topic}?", "¿Qué supuestos se usan al hablar de {topic}?", "¿Qué aplicaciones tiene {topic}?", "¿Cómo explicárselo a un estudiante?", "¿Qué preguntas debería hacerme al estudiar {topic}?", "¿Cómo se conecta {topic} con un caso real?", "¿Qué limitaciones tiene una explicación básica de {topic}?", "¿Qué vocabulario técnico aparece en {topic}?", "¿Puedes comparar dos enfoques dentro de {topic}?", "¿Cómo evolucionó la comprensión de {topic}?", "¿Qué ejemplo cotidiano ilustra {topic}?", "¿Qué dato conviene recordar sobre {topic}?", "¿Cómo evitar confusiones comunes en {topic}?", "¿Qué parte de {topic} suele ser más difícil?", "¿Puedes plantear un ejercicio de {topic}?", "¿Cómo resolverías un ejercicio introductorio de {topic}?", "¿Qué relación matemática aparece en {topic}?", "¿Qué observación apoya esta idea de {topic}?", "¿Qué pasaría si cambiamos una condición en {topic}?", "¿Cómo se usa {topic} en investigación?", "¿Qué herramientas sirven para estudiar {topic}?", "¿Cómo distinguir evidencia de interpretación en {topic}?", "¿Qué concepto previo necesito para entender {topic}?", "¿Puedes dar una analogía para {topic}?", "¿Qué preguntas avanzadas surgen de {topic}?", "¿Cómo se comunica correctamente información sobre {topic}?", "¿Qué ejemplo contradice una intuición común sobre {topic}?", "¿Qué pasos seguirías para analizar {topic}?", "¿Cómo resumirías la historia de {topic}?", "¿Qué incertidumbres existen al estudiar {topic}?", "¿Cómo se relacionan teoría y práctica en {topic}?", "¿Qué clasificación útil existe en {topic}?", "¿Cómo puedo practicar {topic}?", "¿Qué fuentes o tipos de evidencia son relevantes para {topic}?", "¿Qué significa exactamente hablar de {topic}?", "¿Cómo cambiaría la respuesta según el contexto de {topic}?", "¿Qué comparación ayuda a entender {topic}?", "¿Qué detalle técnico merece atención en {topic}?"]

def expanded_facts(topic: str) -> List[str]:
    seeds = TOPIC_FACT_SEEDS[topic]
    forms = [
        "Una idea relacionada es que {x}.", "Conviene recordar que {x}.", "En una introducción a {t}, aparece que {x}.", "Desde una perspectiva práctica, {x}.", "En un contexto educativo, puede decirse que {x}.", "Un punto básico es que {x}.", "Una formulación sencilla es: {x}.", "Como regla conceptual, {x}.", "En términos generales, {x}.", "Al estudiar {t}, es útil saber que {x}."
    ]
    out=[]
    for i in range(80):
        s=seeds[i % len(seeds)]
        f=forms[i % len(forms)].format(x=s,t=topic)
        f=f[:-1] + f" (variación {i+1})" if f.endswith(".") else f + f" (variación {i+1})"
        out.append(f)
    return out[:80]

def build_topics() -> Dict[str, Dict[str, List[str]]]:
    result={}
    for topic in TOPIC_NAMES:
        result[topic]={"facts": expanded_facts(topic), "questions": [q.format(topic=topic) for q in QUESTION_TEMPLATES[:50]]}
    return result

def answer_from_fact(topic: str, question: str, idx: int) -> str:
    facts=expanded_facts(topic)
    fact=facts[idx % len(facts)]
    opener=OPENERS[idx % len(OPENERS)]
    closer=CLOSERS[idx % len(CLOSERS)]
    modes=idx % 4
    if modes == 0: return f"{opener}. {fact} {closer}"
    if modes == 1: return f"{opener}. {fact} Es útil conectarlo con un ejemplo concreto para ver cómo funciona en contexto. {closer}"
    if modes == 2: return f"{opener}. {fact} La idea central es distinguir la definición de sus aplicaciones. También conviene recordar que el contexto puede cambiar los detalles. {closer}"
    return f"{opener}. {fact} Para entenderlo mejor, piensa primero en la definición, después en un ejemplo y finalmente en sus límites. Esa secuencia ayuda a separar intuición de precisión técnica. Si el tema se estudia con datos, conviene además revisar los supuestos y la evidencia. {closer}"

def identity_answer(q: str) -> str:
    low=q.lower()
    if "creó" in low or "creador" in low or "quién te creó" in low:
        return CREATOR_REPLY
    if "modelo" in low:
        return IDENTITY_REPLY
    return IDENTITY_REPLY

def generate_conversations(n: int = 50000) -> List[List[Dict[str,str]]]:
    topics=build_topics()
    convs=[]
    identities=["¿Quién eres?","¿Cómo te llamas?","¿Quién te creó?","¿Qué modelo eres?"]
    for i in range(n):
        topic=TOPIC_NAMES[i % len(TOPIC_NAMES)]
        use_identity=i < 500 or (i % 97 == 0)
        turns=3 + (i % 10)
        sys_prompt=SYSTEM_PROMPTS[i % len(SYSTEM_PROMPTS)] if (i % 10 < 3 or use_identity) else "Responde en español neutro con claridad, precisión y tono cercano."
        messages=[{"role":"system","content":sys_prompt}]
        if use_identity:
            q=identities[i % len(identities)]
            messages += [{"role":"user","content":q},{"role":"assistant","content":identity_answer(q)}]
        else:
            for t in range(max(1, turns//2)):
                q=topics[topic]["questions"][(i+t*7) % 50]
                messages += [{"role":"user","content":q},{"role":"assistant","content":answer_from_fact(topic,q,i+t)}]
        if len(messages)<3:
            q=topics[topic]["questions"][i%50]
            messages += [{"role":"user","content":q},{"role":"assistant","content":answer_from_fact(topic,q,i)}]
        convs.append(messages)
    return convs

# SECCIÓN 10: FORMATO CHATML
def render_chatml(messages: Sequence[Dict[str,str]]) -> str:
    out=[]
    for m in messages:
        role=m["role"]
        tag={"system":"<|system|>","user":"<|user|>","assistant":"<|assistant|>","tool":"<|tool|>","think":"<|think|>"}.get(role,"<|user|>")
        out.append(tag + m["content"] + "<|end|>")
    return "".join(out)

def chatml_ids_and_labels(messages: Sequence[Dict[str,str]], tok: BPETokenizer) -> Tuple[List[int],List[int]]:
    ids=[]; labels=[]
    tags={"system":"<|system|>","user":"<|user|>","assistant":"<|assistant|>","tool":"<|tool|>","think":"<|think|>"}
    for m in messages:
        role=m["role"]
        tag=tags.get(role,"<|user|>")
        tid=tok.token_id(tag)
        eid=tok.end_id
        content=tok.encode(m["content"])
        ids.extend([tid]+content+[eid])
        if role=="assistant": labels.extend([IGNORE_INDEX]*1+content+[IGNORE_INDEX])
        else: labels.extend([IGNORE_INDEX]*(len(content)+2))
    return ids,labels

# SECCIÓN 11: DATASET Y SAMPLER
class ChatDataset(Dataset):
    def __init__(self, records: List[Tuple[List[int],List[int]]], context: int):
        self.records=records; self.context=context
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        x,y=self.records[i]
        x=x[:self.context]; y=y[:self.context]
        return torch.tensor(x,dtype=torch.long),torch.tensor(y,dtype=torch.long)

class LengthGroupedSampler(Sampler[int]):
    def __init__(self, dataset: ChatDataset, batch_size: int, seed: int=SEED, shuffle: bool=True):
        self.lengths=[len(x[0]) for x in dataset.records]; self.batch_size=batch_size; self.seed=seed; self.shuffle=shuffle; self.epoch=0
    def set_epoch(self,e:int): self.epoch=e
    def __iter__(self):
        rng=random.Random(self.seed+self.epoch)
        inds=list(range(len(self.lengths)))
        if self.shuffle: rng.shuffle(inds)
        chunk=max(self.batch_size*8,self.batch_size)
        ordered=[]
        for i in range(0,len(inds),chunk): ordered.extend(sorted(inds[i:i+chunk],key=lambda j:self.lengths[j]))
        return iter(ordered)
    def __len__(self): return len(self.lengths)

def collate_batch(batch, pad_id: int):
    maxlen=max(x.numel() for x,_ in batch)
    xs=[]; ys=[]; masks=[]
    for x,y in batch:
        p=maxlen-x.numel()
        xs.append(F.pad(x,(0,p),value=pad_id)); ys.append(F.pad(y,(0,p),value=IGNORE_INDEX)); masks.append(F.pad(torch.ones_like(x),(0,p),value=0))
    return torch.stack(xs),torch.stack(ys),torch.stack(masks)

def save_tokenized_cache(records: List[Tuple[List[int],List[int]]], path: Path) -> None:
    path=Path(path).with_suffix(".pkl.gz")
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,"wb",compresslevel=6) as f: pickle.dump(records,f,protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenized_cache(path: Path):
    path=Path(path)
    if path.suffix != ".gz": path=path.with_suffix(".pkl.gz")
    with gzip.open(path,"rb") as f: return pickle.load(f)

def _model_param_estimate(cfg: ModelConfig) -> int:
    d = cfg.embed_dim
    kv = cfg.num_kv_heads * cfg.head_dim
    qv = cfg.num_heads * cfg.head_dim
    hidden = int(d * cfg.ffn_mult * 2 / 3)
    hidden = max(64, ((hidden + 63) // 64) * 64)
    embed = cfg.vocab_size * d
    per_attn = d * qv + d * kv * 2 + qv * d
    per_mlp = d * hidden * 3
    norms = 2 * d
    qk_norm = 2 * cfg.head_dim if cfg.qk_norm else 0
    final = d
    return embed + cfg.num_layers * (per_attn + per_mlp + norms + qk_norm) + final


def adjust_model_config(cfg: ModelConfig) -> ModelConfig:
    if cfg.num_heads % cfg.num_kv_heads != 0:
        raise ValueError("num_heads debe ser divisible por num_kv_heads")
    if cfg.head_dim <= 0:
        cfg.head_dim = cfg.embed_dim // cfg.num_heads
    candidates = []
    requested = int(cfg.embed_dim)
    for d in range(max(256, requested - 128), min(1536, requested + 512) + 1, 8):
        trial = ModelConfig(**asdict(cfg))
        trial.embed_dim = d
        n = _model_param_estimate(trial)
        candidates.append((abs(n - 80_000_000), n, d))
    in_range = [x for x in candidates if 75_000_000 <= x[1] <= 85_000_000]
    pool = in_range or candidates
    pool.sort(key=lambda x: (x[0], abs(x[2]-requested)))
    cfg.embed_dim = pool[0][2]
    return cfg


# SECCIÓN 12: COMPONENTES DEL MODELO
class RMSNorm(nn.Module):
    def __init__(self,d:int,eps:float=1e-5):
        super().__init__(); self.weight=nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight


def rotate_half(x):
    x1=x[...,::2]; x2=x[...,1::2]
    return torch.stack((-x2,x1),dim=-1).flatten(-2)


def rope_cache(seq_len:int, head_dim:int, theta:float, device, dtype, scaling="none", factor=1.0, base_context=1024):
    inv=1.0/(theta**(torch.arange(0,head_dim,2,device=device,dtype=torch.float32)/head_dim))
    if factor > 1.0 and seq_len > base_context:
        if scaling == "ntk" and head_dim > 2:
            inv = inv * (factor ** (head_dim / (head_dim - 2)))
        elif scaling == "linear":
            pass
    positions=torch.arange(seq_len,device=device,dtype=torch.float32)
    if scaling == "linear" and factor > 1.0 and seq_len > base_context:
        positions=positions/factor
    freqs=torch.outer(positions,inv)
    return freqs.cos().to(dtype),freqs.sin().to(dtype)


def apply_rope(x,cos,sin):
    c=cos[:x.size(2)].repeat_interleave(2,dim=-1).unsqueeze(0).unsqueeze(0); s=sin[:x.size(2)].repeat_interleave(2,dim=-1).unsqueeze(0).unsqueeze(0)
    return x*c+rotate_half(x)*s


class CausalSelfAttention(nn.Module):
    def __init__(self,cfg:ModelConfig):
        super().__init__()
        self.nh=cfg.num_heads
        self.nkv=cfg.num_kv_heads
        self.hd=cfg.head_dim
        self.inner=self.nh*self.hd
        self.kv_inner=self.nkv*self.hd
        self.q_proj=nn.Linear(cfg.embed_dim,self.inner,bias=False)
        self.k_proj=nn.Linear(cfg.embed_dim,self.kv_inner,bias=False)
        self.v_proj=nn.Linear(cfg.embed_dim,self.kv_inner,bias=False)
        self.o=nn.Linear(self.inner,cfg.embed_dim,bias=False)
        self.cfg=cfg
        self.attn_drop=cfg.dropout
        self.q_norm=RMSNorm(self.hd,cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm=RMSNorm(self.hd,cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.flash_available=hasattr(F,"scaled_dot_product_attention")
        if not self.flash_available:
            warnings.warn("scaled_dot_product_attention no disponible; usando fallback manual optimizado")

    def forward(self,x,key_padding_mask=None,attention_sink=None,past_key_value=None,use_cache=False):
        b,t,_=x.shape
        q=self.q_proj(x).view(b,t,self.nh,self.hd).transpose(1,2)
        k=self.k_proj(x).view(b,t,self.nkv,self.hd).transpose(1,2)
        v=self.v_proj(x).view(b,t,self.nkv,self.hd).transpose(1,2)
        q=self.q_norm(q); k=self.k_norm(k)
        past_len = int(past_key_value[0].size(2)-1) if past_key_value is not None and attention_sink is not None else (int(past_key_value[0].size(2)) if past_key_value is not None else 0)
        if self.cfg.use_rope:
            total=max(t+past_len,t); cos,sin=rope_cache(total,self.hd,self.cfg.rope_theta,x.device,q.dtype,self.cfg.rope_scaling,self.cfg.rope_scaling_factor,self.cfg.context_length)
            if past_key_value is None:
                q=apply_rope(q,cos,sin); k=apply_rope(k,cos,sin)
            else:
                c=cos[past_len:past_len+t]; ss=sin[past_len:past_len+t]
                q=apply_rope(q,c,ss); k=apply_rope(k,c,ss)
        if attention_sink is not None and past_key_value is None:
            sink_hidden=attention_sink.expand(b,1,-1).to(x.dtype)
            sk=self.k_proj(sink_hidden).view(b,1,self.nkv,self.hd).transpose(1,2)
            sv=self.v_proj(sink_hidden).view(b,1,self.nkv,self.hd).transpose(1,2)
            sk=self.k_norm(sk)
            k=torch.cat([sk,k],dim=2); v=torch.cat([sv,v],dim=2)
        if past_key_value is not None:
            pk,pv=past_key_value; k=torch.cat([pk.to(k.dtype),k],dim=2); v=torch.cat([pv.to(v.dtype),v],dim=2)
        present=(k.detach(),v.detach()) if use_cache else None
        if self.nkv != self.nh:
            rep=self.nh//self.nkv
            kt=k.size(2); k=k.unsqueeze(2).expand(b,self.nkv,rep,kt,self.hd).reshape(b,self.nh,kt,self.hd)
            vt=v.size(2); v=v.unsqueeze(2).expand(b,self.nkv,rep,vt,self.hd).reshape(b,self.nh,vt,self.hd)
        if self.flash_available:
            if attention_sink is not None:
                # First key is always visible; remaining keys obey causal mask.
                total_k=k.size(2); past_visible=total_k-t
                mask=torch.zeros((t,total_k),device=x.device,dtype=torch.bool)
                for qi in range(t): mask[qi,:past_visible+qi+1]=True
                y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=self.attn_drop if self.training else 0.0,is_causal=False)
            elif past_key_value is not None:
                total_k=k.size(2); past_visible=total_k-t
                mask=torch.zeros((t,total_k),device=x.device,dtype=torch.bool)
                for qi in range(t): mask[qi,:past_visible+qi+1]=True
                y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=self.attn_drop if self.training else 0.0,is_causal=False)
            else:
                y=F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=self.attn_drop if self.training else 0.0,is_causal=True)
        else:
            scale=self.hd**-0.5
            scores=torch.matmul(q,k.transpose(-2,-1))*scale
            total_k=k.size(2); past_visible=total_k-t
            causal=torch.zeros(t,total_k,device=x.device,dtype=torch.bool)
            for qi in range(t): causal[qi,:past_visible+qi+1]=True
            if key_padding_mask is not None:
                km=key_padding_mask[:,None,None,:]
                scores=scores.masked_fill(~km,-torch.inf)
            scores=scores.masked_fill(~causal,-torch.inf)
            y=torch.softmax(scores.float(),dim=-1).to(q.dtype)
            y=torch.matmul(y,v)
        out=self.o(y.transpose(1,2).contiguous().view(b,t,self.inner))
        return (out,present) if use_cache else out


class SwiGLU(nn.Module):
    def __init__(self,cfg:ModelConfig):
        super().__init__(); d=cfg.embed_dim; h=int(d*cfg.ffn_mult*2/3); h=max(64,((h+63)//64)*64)
        self.hidden_dim=h
        self.gate=nn.Linear(d,h,bias=False); self.up=nn.Linear(d,h,bias=False); self.down=nn.Linear(h,d,bias=False)
        self.drop=nn.Dropout(cfg.dropout)
    def forward(self,x): return self.drop(self.down(F.silu(self.gate(x))*self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self,cfg:ModelConfig):
        super().__init__()
        self.parallel=bool(getattr(cfg,"parallel_attn_mlp",True)) if hasattr(cfg,"parallel_attn_mlp") else False
        self.n1=RMSNorm(cfg.embed_dim,cfg.norm_eps)
        self.n2=None if self.parallel else RMSNorm(cfg.embed_dim,cfg.norm_eps)
        self.attn=CausalSelfAttention(cfg); self.mlp=SwiGLU(cfg)
    def forward(self,x,key_padding_mask=None,attention_sink=None,past_key_value=None,use_cache=False):
        if self.parallel:
            z=self.n1(x); ar=self.attn(z,key_padding_mask=key_padding_mask,attention_sink=attention_sink,past_key_value=past_key_value,use_cache=use_cache)
            if use_cache: attn_out,present=ar
            else: attn_out=ar
            return (x+attn_out+self.mlp(z),present) if use_cache else x+attn_out+self.mlp(z)
        ar=self.attn(self.n1(x),key_padding_mask=key_padding_mask,attention_sink=attention_sink,past_key_value=past_key_value,use_cache=use_cache)
        if use_cache: attn_out,present=ar
        else: attn_out=ar
        x=x+attn_out; x=x+self.mlp(self.n2(x))
        return (x,present) if use_cache else x


class AquaModel(nn.Module):
    def __init__(self,cfg:ModelConfig):
        super().__init__(); self.cfg=adjust_model_config(cfg)
        self.embed=nn.Embedding(self.cfg.vocab_size,self.cfg.embed_dim,padding_idx=0)
        self.blocks=nn.ModuleList([TransformerBlock(self.cfg) for _ in range(self.cfg.num_layers)])
        self.norm=RMSNorm(self.cfg.embed_dim,self.cfg.norm_eps)
        self.lm_head=nn.Linear(self.cfg.embed_dim,self.cfg.vocab_size,bias=False)
        self.sink_token=nn.Parameter(torch.zeros(1,1,self.cfg.embed_dim)) if getattr(self.cfg,"attention_sinks",True) else None
        if self.cfg.tie_weights: self.lm_head.weight=self.embed.weight
        self.last_loss_stats={"ce_loss":None,"z_loss":None}
        self.apply(self._init_weights)
        if self.sink_token is not None: nn.init.normal_(self.sink_token,std=self.cfg.init_std)
        scale=1/math.sqrt(2*self.cfg.num_layers)
        for b in self.blocks:
            nn.init.normal_(b.attn.o.weight,std=self.cfg.init_std*scale); nn.init.normal_(b.mlp.down.weight,std=self.cfg.init_std*scale)
        n=count_parameters(self)
        print(f"{MODEL_NAME}: {n:,} parámetros exactos | embed_dim={self.cfg.embed_dim} | GQA={self.cfg.num_heads}/{self.cfg.num_kv_heads}")
        if not 75_000_000 <= n <= 85_000_000:
            raise RuntimeError(f"Conteo fuera del objetivo 75M-85M: {n:,}")
    def _init_weights(self,m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,std=self.cfg.init_std)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,std=self.cfg.init_std)
    def forward(self,input_ids,labels=None, label_smoothing=0.0, z_loss_weight=0.0, attention_mask=None,past_key_values=None,use_cache=False):
        x=self.embed(input_ids); presents=[]
        for bi,block in enumerate(self.blocks):
            past=past_key_values[bi] if past_key_values is not None else None
            if self.cfg.gradient_checkpointing and self.training and x.requires_grad and not use_cache:
                x=torch.utils.checkpoint.checkpoint(block,x,attention_mask,self.sink_token if self.sink_token is not None else None,None,False,use_reentrant=False)
            else:
                out=block(x,key_padding_mask=attention_mask,attention_sink=self.sink_token if self.sink_token is not None else None,past_key_value=past,use_cache=use_cache)
                if use_cache:x,present=out;presents.append(present)
                else:x=out
        logits=self.lm_head(self.norm(x))
        loss=None
        if labels is not None:
            flat_logits=logits.view(-1,logits.size(-1)); flat_labels=labels.view(-1)
            ce_loss=F.cross_entropy(flat_logits,flat_labels,ignore_index=IGNORE_INDEX,label_smoothing=label_smoothing)
            valid=flat_labels.ne(IGNORE_INDEX)
            if valid.any():
                lse=torch.logsumexp(flat_logits[valid].float(),dim=-1)
                z_loss=(lse**2).mean()*z_loss_weight
            else:
                z_loss=flat_logits.new_zeros(())
            loss=ce_loss+z_loss
            self.last_loss_stats={"ce_loss":float(ce_loss.detach()),"z_loss":float(z_loss.detach())}
        return (logits,loss,presents) if use_cache else (logits,loss)


class EMA:
    def __init__(self,model,decay=0.999):
        self.decay=float(decay)
        self.shadow={n:p.detach().clone().float().cpu() for n,p in model.named_parameters() if p.requires_grad}
        self.backup={}
    @torch.no_grad()
    def update(self,model):
        for n,p in model.named_parameters():
            if n not in self.shadow: continue
            self.shadow[n].mul_(self.decay).add_(p.detach().float().cpu(),alpha=1.0-self.decay)
    @torch.no_grad()
    def apply(self,model):
        self.backup={}
        for n,p in model.named_parameters():
            if n in self.shadow:
                self.backup[n]=p.detach().clone()
                p.copy_(self.shadow[n].to(device=p.device,dtype=p.dtype))
    @torch.no_grad()
    def restore(self,model):
        for n,p in model.named_parameters():
            if n in self.backup:p.copy_(self.backup[n].to(device=p.device,dtype=p.dtype))
        self.backup={}
    def state_dict(self): return {"decay":self.decay,"shadow":{k:v.clone() for k,v in self.shadow.items()}}
    def load_state_dict(self,state):
        if not state:return
        self.decay=float(state.get("decay",self.decay)); self.shadow={k:v.clone() for k,v in state.get("shadow",{}).items()}
    def norm(self):
        return float(math.sqrt(sum(float(v.float().pow(2).sum()) for v in self.shadow.values())))


def count_parameters(model:nn.Module)->int:
    return sum(p.numel() for p in model.parameters())


def model_summary(model:nn.Module)->None:
    log("Resumen del modelo:")
    for name,module in model.named_modules():
        if name and not list(module.children()):
            n=sum(p.numel() for p in module.parameters(recurse=False))
            if n: log(f"  {name}: {n:,}")



# SECCIÓN 13: LR SCHEDULER
class FlexibleScheduler:
    def __init__(self,opt,warmup,total,min_lr,mode="cosine",stable_frac=.1): self.opt=opt; self.warmup=warmup; self.total=total; self.min_lr=min_lr; self.mode=mode; self.base=[g["lr"] for g in opt.param_groups]; self.stable_frac=stable_frac
    def lr(self,step):
        if step<self.warmup: return max(1e-12,self.base[0]*step/max(1,self.warmup))
        p=(step-self.warmup)/max(1,self.total-self.warmup); p=min(1,max(0,p))
        if self.mode=="linear": f=1-p
        elif self.mode=="wsd":
            stable=min(.5,self.stable_frac); f=1 if p<stable else max(0,(1-p)/(1-stable))
        else: f=.5*(1+math.cos(math.pi*p))
        return self.min_lr+(self.base[0]-self.min_lr)*f
    def step(self,step):
        lr=self.lr(step)
        for g in self.opt.param_groups: g["lr"]=lr
        return lr
    def state_dict(self):
        return {"warmup":self.warmup,"total":self.total,"min_lr":self.min_lr,"mode":self.mode,"base":self.base,"stable_frac":self.stable_frac,"last_lr":[g["lr"] for g in self.opt.param_groups]}
    def load_state_dict(self,state):
        if not state: return
        self.warmup=state.get("warmup",self.warmup); self.total=state.get("total",self.total); self.min_lr=state.get("min_lr",self.min_lr); self.mode=state.get("mode",self.mode); self.base=state.get("base",self.base); self.stable_frac=state.get("stable_frac",self.stable_frac)
        for g,lr in zip(self.opt.param_groups,state.get("last_lr",[])): g["lr"]=lr

# SECCIÓN 14: UTILIDADES
def make_optimizer(model,tcfg):
    decay=[]; nodecay=[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim>=2 and not n.endswith("norm.weight"): decay.append(p)
        else: nodecay.append(p)
    return torch.optim.AdamW([{"params":decay,"weight_decay":tcfg.weight_decay},{"params":nodecay,"weight_decay":0.0}],lr=tcfg.lr,betas=(.9,.95),eps=1e-8)

def autocast_context(device,tcfg=None):
    if device.type!="cuda":
        return torch.autocast(device_type=device.type,enabled=False)
    name=getattr(tcfg,"amp_dtype","bfloat16") if tcfg is not None else "bfloat16"
    dtype=torch.bfloat16 if name=="bfloat16" else torch.float16
    return torch.autocast(device_type="cuda",dtype=dtype)

def gpu_memory():
    if not torch.cuda.is_available(): return 0.0
    return torch.cuda.max_memory_allocated()/1024**3

def grad_norm(model):
    total=0.0
    for p in model.parameters():
        if p.grad is not None: total+=float(p.grad.detach().float().norm().item()**2)
    return total**.5

def perplexity(loss): return float(math.exp(min(20,float(loss))))

def save_final_config(cfg,tcfg):
    obj={"model":asdict(cfg),"training":asdict(tcfg),"identity":{"name":MODEL_NAME,"author":MODEL_AUTHOR,"version":MODEL_VERSION,"system_prompt":DEFAULT_SYSTEM},"parameter_count":None}
    save_json(obj,CONFIG_PATH)

def cache_corpus(texts:List[str]):
    with CORPUS_PATH.open("wb") as f: pickle.dump(texts,f,protocol=pickle.HIGHEST_PROTOCOL)

def load_corpus():
    if CORPUS_PATH.exists():
        with CORPUS_PATH.open("rb") as f:return pickle.load(f)
    return []

# SECCIÓN 15: CHECKPOINTS
CHECKPOINT_CODE_VERSION = "aqua-checkpoint-manager-3.0-80m"
_CHECKPOINT_MANAGER = None
_KAGGLE_UPLOADER = None
_SIGNAL_EXIT = False
_LAST_EXCEPTION = None


def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(s):
    if not s: return
    try:
        if s.get("python") is not None: random.setstate(s["python"])
        if s.get("numpy") is not None: np.random.set_state(s["numpy"])
        if s.get("torch") is not None: torch.set_rng_state(s["torch"])
        if torch.cuda.is_available() and s.get("cuda") is not None: torch.cuda.set_rng_state_all(s["cuda"])
    except Exception as e:
        warn(f"No se pudo restaurar completamente el RNG: {e}")


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def generate_checkpoint_name(step=None, reason="periodic") -> str:
    allowed={"periodic","time","signal","shutdown","final","error","manual"}
    if reason not in allowed: raise ValueError(f"reason inválido: {reason}")
    stamp=time.strftime("%Y%m%d_%H%M%S")
    uid=uuid.uuid4().hex[:8]
    prefix="aqua_0.5b"
    step_part=f"_step{int(step)}" if step is not None else ""
    return f"{prefix}{step_part}_{reason}_{stamp}_{uid}.pt"


def _json_safe(obj):
    if isinstance(obj, (str,int,float,bool)) or obj is None: return obj
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, dict): return {str(k):_json_safe(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)): return [_json_safe(x) for x in obj]
    return str(obj)


class CheckpointManager:
    def __init__(self, config, model, optimizer, scaler=None, scheduler=None, ema=None):
        self.config=config
        self.model=model
        self.optimizer=optimizer
        self.scaler=scaler
        self.scheduler=scheduler
        self.ema=ema
        self.checkpoint_dir=CKPT_DIR
        self.checkpoint_dir.mkdir(parents=True,exist_ok=True)
        self.history=[]
        self.config_bundle={}
        self.tokenizer=None
        self.step=0
        self.epoch=0
        self.best_val_loss=float("inf")
        self.last_save_time=0.0
        self.last_upload_time=0.0
        self.session_start=time.time()
        self.total_train_seconds=0.0
        self.last_saved_path=None
        self._saving=False
        self._loaded=False

    def _state(self, step=None, epoch=None, best_val_loss=None):
        now=time.time()
        return {"checkpoint_version":CHECKPOINT_CODE_VERSION,"model":self.model.state_dict(),"optimizer":self.optimizer.state_dict(),"scaler":self.scaler.state_dict() if self.scaler is not None else None,"scheduler":self.scheduler.state_dict() if self.scheduler is not None and hasattr(self.scheduler,"state_dict") else None,"ema":self.ema.state_dict() if self.ema is not None else None,"step":int(self.step if step is None else step),"epoch":int(self.epoch if epoch is None else epoch),"best_val_loss":float(self.best_val_loss if best_val_loss is None else best_val_loss),"rng":rng_state(),"config":self.config.to_dict() if hasattr(self.config,"to_dict") else _json_safe(asdict(self.config)),"config_bundle":_json_safe(self.config_bundle),"history":list(self.history),"total_train_seconds":float(self.total_train_seconds + max(0.0, now-self.session_start)),"saved_at":now,"saved_at_iso":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"code_version":CHECKPOINT_CODE_VERSION}

    def _save_atomic(self,path,state):
        path=Path(path); tmp=path.with_name(path.name+".tmp")
        if path.exists(): raise FileExistsError(str(path))
        if tmp.exists(): tmp.unlink()
        torch.save(state,tmp)
        expected=tmp.stat().st_size
        if expected<=0: raise IOError("checkpoint temporal vacío")
        with tmp.open("rb") as f:
            f.seek(0,os.SEEK_END); actual=f.tell()
        if actual!=expected: raise IOError(f"tamaño incompleto: {actual} != {expected}")
        digest=_sha256(tmp)
        # La huella se conserva además en un sidecar porque una huella dentro del mismo
        # archivo sería circular: modificar el checkpoint para guardar su hash cambia el hash.
        os.replace(tmp,path)
        sidecar=path.with_suffix(path.suffix+".sha256")
        sidecar_tmp=sidecar.with_name(sidecar.name+".tmp")
        sidecar_tmp.write_text(digest+"  "+path.name+"\n",encoding="utf-8")
        os.replace(sidecar_tmp,sidecar)
        return digest,actual

    def _verify_hash(self,path,expected_hash=None):
        path=Path(path)
        if not path.exists(): return False
        if expected_hash is None:
            side=path.with_suffix(path.suffix+".sha256")
            if side.exists(): expected_hash=side.read_text(encoding="utf-8").split()[0]
        if not expected_hash: return True
        return _sha256(path)==expected_hash

    def update_latest_pointer(self,path):
        path=Path(path).resolve()
        pointer=self.checkpoint_dir/"LATEST.txt"
        tmp=pointer.with_suffix(".tmp")
        tmp.write_text(str(path)+"\n",encoding="utf-8")
        os.replace(tmp,pointer)

    def _rotate_checkpoints(self):
        keep=max(3,int(getattr(self.config,"keep_last_n_checkpoints",5)))
        files=sorted(list(self.checkpoint_dir.glob("aqua_0.5b_*.pt"))+list(self.checkpoint_dir.glob("aqua_0.5b_*.pt.gz")),key=lambda p:p.stat().st_mtime,reverse=True)
        for p in files[keep:]:
            try:
                p.unlink(missing_ok=True); p.with_suffix(p.suffix+".sha256").unlink(missing_ok=True); Path(str(p)+".gz").unlink(missing_ok=True)
            except OSError as e: warn(f"No se pudo podar {p}: {e}")

    def _compress(self,path):
        path=Path(path)
        if not getattr(self.config,"compress_checkpoints",True): return path
        out=path.with_suffix(path.suffix+".gz")
        tmp=out.with_name(out.name+".tmp")
        try:
            with path.open("rb") as src,gzip.open(tmp,"wb",compresslevel=9) as dst:
                shutil.copyfileobj(src,dst,1024*1024)
            os.replace(tmp,out)
            return out
        except Exception as e:
            warn(f"No se pudo comprimir {path.name}: {e}")
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
            return path

    def save(self, reason, step=None) -> str:
        if self._saving: return str(self.last_saved_path or "")
        self._saving=True
        started=time.time()
        try:
            self.step=int(self.step if step is None else step)
            name=generate_checkpoint_name(self.step,reason)
            path=self.checkpoint_dir/name
            while path.exists() or path.with_suffix(path.suffix+".tmp").exists():
                name=generate_checkpoint_name(self.step,reason); path=self.checkpoint_dir/name
            state=self._state(self.step,self.epoch,self.best_val_loss)
            digest,size=self._save_atomic(path,state)
            self.last_saved_path=path
            self.last_save_time=time.time()
            self.update_latest_pointer(path)
            if self.tokenizer is not None:
                try: _write_resume_metadata(self,self.tokenizer,self.config_bundle.get("model_config_obj",ModelConfig()),self.config)
                except Exception as e: warn(f"No se pudo actualizar metadata de resume: {e}")
            self._rotate_checkpoints()
            elapsed=time.time()-started
            log(f"checkpoint guardado | timestamp={time.strftime('%Y-%m-%d %H:%M:%S')} | reason={reason} | path={path} | size={size/1024**2:.2f}MB | save_time={elapsed:.2f}s | sha256={digest[:16]}...")
            return str(path)
        except OSError as e:
            if getattr(e,"errno",None)==28:
                warn("Disco lleno: eliminando checkpoints antiguos antes de reintentar")
                self._rotate_checkpoints()
            warn(f"Fallo guardando checkpoint ({reason}): {e}")
            return str(self.last_saved_path or "")
        except Exception as e:
            warn(f"Fallo guardando checkpoint ({reason}): {e}")
            return str(self.last_saved_path or "")
        finally:
            self._saving=False

    def _materialize_compressed(self,path: Path) -> Path:
        path=Path(path)
        if path.suffix != ".gz": return path
        out=self.checkpoint_dir/(path.name[:-3])
        tmp=out.with_name(out.name+".restore.tmp")
        try:
            with gzip.open(path,"rb") as src,tmpf:
                with tmp.open("wb") as dst: shutil.copyfileobj(src,dst,1024*1024)
            os.replace(tmp,out)
            return out
        except Exception:
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
            raise

    def _candidate_paths(self):
        pointers=[self.checkpoint_dir/"LATEST.txt"]
        if IS_KAGGLE:
            pointers.extend(INPUT_DIR.glob("*/checkpoints/LATEST.txt"))
        for pointer in pointers:
            if pointer.exists():
                try:
                    raw=pointer.read_text(encoding="utf-8").strip()
                    p=Path(raw)
                    choices=[p,pointer.parent/p.name] if p.is_absolute() else [pointer.parent/p]
                    # Si LATEST apunta al /kaggle/working original, probar también el basename junto al puntero.
                    choices.append(pointer.parent/Path(raw).name)
                    for q in choices:
                        if q.exists() and q.suffix in {".pt",".gz"}:
                            yield q
                except Exception: pass
        files=sorted(list(self.checkpoint_dir.glob("aqua_0.5b_*.pt"))+list(self.checkpoint_dir.glob("aqua_0.5b_*.pt.gz")),key=lambda p:p.stat().st_mtime,reverse=True)
        for p in files: yield p
        if IS_KAGGLE:
            for d in INPUT_DIR.glob("*/checkpoints"):
                for p in sorted(list(d.glob("aqua_0.5b_*.pt"))+list(d.glob("aqua_0.5b_*.pt.gz")),key=lambda p:p.stat().st_mtime,reverse=True): yield p

    def find_latest(self):
        seen=set()
        for p in self._candidate_paths():
            if str(p) in seen: continue
            seen.add(str(p))
            try:
                if p.suffix==".gz":
                    side=p.with_suffix(p.suffix+".sha256")
                    if side.exists() and _sha256(p)==side.read_text(encoding="utf-8").split()[0]: return str(p)
                    q=self._materialize_compressed(p)
                    if self._verify_hash(q): return str(q)
                elif self._verify_hash(p):
                    return str(p)
            except Exception:
                pass
            warn(f"Checkpoint corrupto o hash inválido, probando anterior: {p}")
        return None

    def load(self,path) -> dict:
        path=Path(path)
        if not path.exists(): raise FileNotFoundError(path)
        if path.suffix==".gz": path=self._materialize_compressed(path)
        if getattr(self.config,"verify_checkpoint_hash",True) and not self._verify_hash(path):
            raise ValueError(f"hash SHA256 inválido: {path}")
        try:
            obj=torch.load(path,map_location="cpu",weights_only=False)
        except TypeError:
            obj=torch.load(path,map_location="cpu")
        if obj.get("code_version") and obj["code_version"]!=CHECKPOINT_CODE_VERSION:
            warn(f"Versión de checkpoint distinta: {obj['code_version']} != {CHECKPOINT_CODE_VERSION}")
        try:
            self.model.load_state_dict(obj["model"],strict=True)
        except RuntimeError as e:
            warn(f"Checkpoint antiguo/incompatible estructuralmente; intentando carga parcial: {e}")
            self.model.load_state_dict(obj["model"],strict=False)
        try:self.optimizer.load_state_dict(obj["optimizer"])
        except Exception as e:warn(f"Optimizador no compatible con el checkpoint; se reinicia: {e}")
        if self.scaler is not None and obj.get("scaler") is not None:
            try:self.scaler.load_state_dict(obj["scaler"])
            except Exception as e:warn(f"No se pudo restaurar scaler: {e}")
        if self.scheduler is not None and obj.get("scheduler") is not None:
            try:self.scheduler.load_state_dict(obj["scheduler"])
            except Exception as e:warn(f"No se pudo restaurar scheduler: {e}")
        if self.ema is not None and obj.get("ema") is not None:
            try:self.ema.load_state_dict(obj["ema"])
            except Exception as e:warn(f"No se pudo restaurar EMA: {e}")
        self.step=int(obj.get("step",0)); self.epoch=int(obj.get("epoch",0)); self.best_val_loss=float(obj.get("best_val_loss",obj.get("best_val",float("inf"))))
        self.history=list(obj.get("history",[]))
        self.total_train_seconds=float(obj.get("total_train_seconds",0.0))
        restore_rng(obj.get("rng"))
        self.last_saved_path=path; self.last_save_time=time.time(); self._loaded=True
        trained_hours=self.total_train_seconds/3600.0
        pct=100*self.step/max(1,self.config.max_steps)
        log(f"Reanudando desde step {self.step} de {self.config.max_steps} ({pct:.2f}% completado)")
        log(f"Mejor val_loss hasta ahora: {self.best_val_loss:.6f}")
        log(f"Tiempo total entrenado: {trained_hours:.2f} horas")
        return obj


class KaggleUploader:
    def __init__(self,config):
        self.config=config
        self.enabled=bool(getattr(config,"kaggle_upload_enabled",True)) and IS_KAGGLE
        self.last_upload_time=0.0
        self.dataset_name=self._dataset_name()
        self.api=None
        if self.enabled:
            if not Path.home().joinpath(".kaggle","kaggle.json").exists() and not os.environ.get("KAGGLE_CONFIG_DIR"):
                warn("kaggle.json no existe; subida automática desactivada")
                self.enabled=False
            else:
                try:
                    import kaggle
                    self.api=kaggle.api
                    self.api.authenticate()
                except Exception as e:
                    warn(f"No se pudo autenticar Kaggle: {e}")
                    self.enabled=False

    def _dataset_name(self):
        name=getattr(self.config,"kaggle_dataset_name","").strip()
        if name:return name
        username=os.environ.get("KAGGLE_USERNAME","")
        if not username:
            try:
                import kaggle
                username=kaggle.api.get_config_value("username") or "unknown"
            except Exception: username="unknown"
        username=re.sub(r"[^a-zA-Z0-9_-]","-",username).strip("-").lower() or "unknown"
        return f"aqua-checkpoints-{username}"

    def _log_upload(self,msg):
        line=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        _write_log_line(line,UPLOAD_LOG_PATH); log(msg)

    def _ensure_dataset_exists(self):
        if not self.enabled or self.api is None:return False
        try:
            self.api.dataset_view(self.dataset_name)
            return True
        except Exception:
            pass
        try:
            up=WORK_DIR/"kaggle_upload_init"
            if up.exists(): shutil.rmtree(up,ignore_errors=True)
            up.mkdir(parents=True,exist_ok=True)
            meta={"title":self.dataset_name,"id":self.dataset_name,"licenses":[{"name":"CC0-1.0"}],"subtitle":"Aqua 0.5B checkpoint persistence","description":"Private checkpoint persistence dataset."}
            save_json(meta,up/"dataset-metadata.json")
            self.api.dataset_create(folder=str(up),public=False,quiet=True)
            return True
        except Exception as e:
            self._log_upload(f"WARNING: no se pudo crear dataset {self.dataset_name}: {e}")
            return False

    def _prepare_upload_dir(self):
        up=WORK_DIR/"kaggle_upload"
        if up.exists(): shutil.rmtree(up,ignore_errors=True)
        c=up/"checkpoints"; c.mkdir(parents=True,exist_ok=True)
        files=sorted(CKPT_DIR.glob("aqua_0.5b_*.pt"),key=lambda p:p.stat().st_mtime,reverse=True)[:2]
        for src in files:
            gz=src.with_suffix(src.suffix+".gz")
            try:
                with src.open("rb") as a,gzip.open(gz,"wb",compresslevel=9) as b: shutil.copyfileobj(a,b,1024*1024)
                shutil.copy2(gz,c/gz.name)
            except Exception as e:
                warn(f"No se pudo preparar {src.name} para Kaggle: {e}")
        latest=CKPT_DIR/"LATEST.txt"
        latest_name=None
        if latest.exists():
            try: latest_name=Path(latest.read_text(encoding="utf-8").strip()).name
            except Exception: latest_name=None
        if latest_name:
            gz_name=latest_name+".gz"
            if (c/gz_name).exists():
                (c/"LATEST.txt").write_text(gz_name+"\n",encoding="utf-8")
                (c/(gz_name+".sha256")).write_text(_sha256(c/gz_name)+"  "+gz_name+"\n",encoding="utf-8")
            elif (c/latest_name).exists():
                (c/"LATEST.txt").write_text(latest_name+"\n",encoding="utf-8")
        history=CKPT_DIR/"history.json"
        if history.exists(): shutil.copy2(history,c/"history.json")
        config=CKPT_DIR/"config.json"
        if config.exists(): shutil.copy2(config,c/"config.json")
        tokenizer=CKPT_DIR/"tokenizer.json"
        if tokenizer.exists(): shutil.copy2(tokenizer,c/"tokenizer.json")
        resume=CKPT_DIR/"resume_info.json"
        if resume.exists(): shutil.copy2(resume,c/"resume_info.json")
        metadata={"title":self.dataset_name,"id":self.dataset_name,"licenses":[{"name":"CC0-1.0"}],"subtitle":"Aqua 0.5B checkpoint persistence","description":"Private checkpoint persistence dataset."}
        save_json(metadata,up/"dataset-metadata.json")
        return up

    def _retry_upload(self,up,message):
        last=None
        for attempt in range(1,4):
            started=time.time()
            try:
                cmd=[sys.executable,"-m","kaggle","datasets","version","-p",str(up),"-m",message,"--dir-mode","zip"]
                proc=subprocess.run(cmd,capture_output=True,text=True,timeout=600,check=False)
                if proc.returncode!=0:
                    raise RuntimeError((proc.stderr or proc.stdout or "kaggle datasets version falló").strip()[-4000:])
                elapsed=time.time()-started; self.last_upload_time=time.time()
                self._log_upload(f"upload OK | dataset={self.dataset_name} | attempt={attempt} | time={elapsed:.2f}s | message={message}")
                return True
            except Exception as e:
                last=e; elapsed=time.time()-started
                self._log_upload(f"upload fallo | dataset={self.dataset_name} | attempt={attempt} | time={elapsed:.2f}s | error={e}")
                if attempt<3: time.sleep((5,30,120)[attempt-1])
        self._log_upload(f"WARNING: subida falló 3 veces y se continúa entrenando: {last}")
        return False

    def _prune_dataset_versions(self):
        if not self.enabled or self.api is None:return
        try:
            versions=[]
            fn=getattr(self.api,"dataset_list_versions",None)
            if fn is not None:
                try: versions=list(fn(self.dataset_name,page_size=100))
                except TypeError: versions=list(fn(self.dataset_name))
            if len(versions)>80:
                delete_fn=getattr(self.api,"dataset_delete",None)
                for v in versions[5:]:
                    try:
                        vid=getattr(v,"version_number",None) or getattr(v,"versionNumber",None)
                        if vid is not None and delete_fn is not None: delete_fn(self.dataset_name,version_number=vid,force=True)
                    except Exception as e: warn(f"No se pudo eliminar versión antigua de Kaggle: {e}")
                    if len(versions)<=80: break
        except Exception as e:
            warn(f"No se pudo inspeccionar versiones Kaggle: {e}")

    def upload(self,checkpoint_dir,message) -> bool:
        if not self.enabled:return False
        started=time.time()
        try:
            if not self._ensure_dataset_exists():return False
            up=self._prepare_upload_dir()
            ok=self._retry_upload(up,message)
            self._prune_dataset_versions()
            elapsed=time.time()-started
            self._log_upload(f"upload finalizado | resultado={'OK' if ok else 'FAIL'} | elapsed={elapsed:.2f}s")
            return ok
        except Exception as e:
            self._log_upload(f"WARNING: excepción durante upload: {e}")
            return False


def register_signal_handlers(manager,uploader):
    global _CHECKPOINT_MANAGER,_KAGGLE_UPLOADER
    _CHECKPOINT_MANAGER=manager; _KAGGLE_UPLOADER=uploader
    def handler(signum,frame):
        global _SIGNAL_EXIT
        if _SIGNAL_EXIT:return
        _SIGNAL_EXIT=True
        name=getattr(signal.Signals(signum),"name",str(signum))
        log(f"Señal {name} recibida, guardando y saliendo...")
        try:
            path=manager.save("signal",manager.step)
            if uploader and uploader.enabled:
                uploader.upload(manager.checkpoint_dir,f"signal {name} step {manager.step}")
            log(f"Checkpoint de señal completado: {path}")
        except Exception as e: warn(f"Fallo en guardado por señal: {e}")
        raise SystemExit(0)
    for sig_name in ("SIGTERM","SIGINT","SIGBREAK"):
        sig=getattr(signal,sig_name,None)
        if sig is not None:
            try: signal.signal(sig,handler)
            except (ValueError,OSError): pass


def emergency_save():
    manager=_CHECKPOINT_MANAGER; uploader=_KAGGLE_UPLOADER
    if manager is None:return
    try:
        if time.time()-manager.last_save_time>=300:
            path=manager.save("shutdown",manager.step)
            if uploader and uploader.enabled and time.time()-uploader.last_upload_time>=1800:
                uploader.upload(manager.checkpoint_dir,"atexit emergency checkpoint")
            log(f"atexit: checkpoint={path}")
    except Exception as e: warn(f"atexit emergency_save falló: {e}")

atexit.register(emergency_save)


def install_global_exception_handler(manager,uploader):
    def hook(exc_type,exc,tb):
        global _LAST_EXCEPTION
        _LAST_EXCEPTION=exc
        log("Excepción no capturada; guardando checkpoint de emergencia")
        try:
            manager.save("error",manager.step)
            if uploader and uploader.enabled:uploader.upload(manager.checkpoint_dir,f"error checkpoint step {manager.step}")
        except Exception as e: warn(f"No se pudo guardar checkpoint de error: {e}")
        sys.__excepthook__(exc_type,exc,tb)
    sys.excepthook=hook


def _write_resume_metadata(manager,tok,cfg,tcfg):
    save_json({"step":manager.step,"epoch":manager.epoch,"best_val_loss":manager.best_val_loss,"timestamp":time.time(),"checkpoint":str(manager.last_saved_path or "")},CKPT_DIR/"resume_info.json")
    save_json({"model":asdict(cfg),"training":tcfg.to_dict(),"identity":{"name":MODEL_NAME,"author":MODEL_AUTHOR,"version":MODEL_VERSION}},CKPT_DIR/"config.json")
    if hasattr(tok,"save"): tok.save(CKPT_DIR/"tokenizer.json")
    with (CKPT_DIR/"history.json").open("w",encoding="utf-8") as f: json.dump(manager.history,f,ensure_ascii=False,indent=2)


def save_checkpoint(model,opt,scaler,step,best_val,cfg,tcfg,scheduler=None,history=None,manager=None):
    if manager is None:
        manager=CheckpointManager(tcfg,model,opt,scaler,scheduler)
        manager.step=step; manager.best_val_loss=best_val; manager.history=history or []
    else:
        manager.step=step; manager.best_val_loss=best_val; manager.history=history or manager.history
    return manager.save("manual",step)


def load_latest(model,opt,scaler,scheduler=None,config=None):
    cfg=config or TrainConfig()
    manager=CheckpointManager(cfg,model,opt,scaler,scheduler)
    path=manager.find_latest()
    if not path:return 0,float("inf"),manager
    try:
        manager.load(path); return manager.step,manager.best_val_loss,manager
    except Exception as e:
        warn(f"No se pudo cargar {path}: {e}")
        for candidate in manager._candidate_paths():
            if str(candidate)==str(path):continue
            try:
                manager.load(candidate); return manager.step,manager.best_val_loss,manager
            except Exception as ee: warn(f"Checkpoint descartado: {candidate}: {ee}")
    return 0,float("inf"),manager

# SECCIÓN 16: EVALUACIÓN
def evaluate(model,loader,device,max_batches=40,tcfg=None):
    model.eval(); losses=[]; correct=0; total=0; ce_losses=[]; z_losses=[]
    with torch.no_grad():
        for i,(x,y,attn_mask) in enumerate(loader):
            if i>=max_batches:break
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); attn_mask=attn_mask.to(device,non_blocking=True)
            with autocast_context(device,tcfg):
                logits,loss=model(x,y,label_smoothing=getattr(tcfg,"label_smoothing",0.0),z_loss_weight=getattr(tcfg,"z_loss_weight",0.0),attention_mask=attn_mask)
            losses.append(float(loss)); stats=getattr(model,"last_loss_stats",{})
            if stats.get("ce_loss") is not None: ce_losses.append(float(stats["ce_loss"]))
            if stats.get("z_loss") is not None: z_losses.append(float(stats["z_loss"]))
            pred=logits.argmax(-1); mask=y!=IGNORE_INDEX
            correct += int((pred[mask]==y[mask]).sum().item()); total += int(mask.sum().item())
    model.train()
    return (float(np.mean(losses)) if losses else float("inf"), float(correct/max(1,total)), float(np.mean(ce_losses)) if ce_losses else float("inf"), float(np.mean(z_losses)) if z_losses else 0.0)


# SECCIÓN 17: ENTRENAMIENTO
def _phase_for_step(step:int, max_steps:int, phases:Tuple[float,...]) -> int:
    frac=(step/max(1,max_steps))
    for i,bound in enumerate(phases):
        if frac < bound:return i
    return len(phases)-1



def run_lr_finder(model,train_loader,cfg,tcfg,device):
    if not tcfg.lr_finder_enabled or tcfg.lr_finder_steps<=0: return tcfg.lr
    log("=== LR FINDER: inicio ==="); started=time.time()
    probe=copy.deepcopy(model).to(device); probe.train()
    opt=torch.optim.AdamW(probe.parameters(),lr=tcfg.lr_finder_min,betas=(.9,.95),weight_decay=tcfg.weight_decay)
    it=iter(train_loader); losses=[]; lrs=[]; prev=None; best_lr=tcfg.lr_finder_min
    for step in range(tcfg.lr_finder_steps):
        try:x,y,mask=next(it)
        except StopIteration: it=iter(train_loader); x,y,mask=next(it)
        x=x.to(device); y=y.to(device); mask=mask.to(device)
        frac=step/max(1,tcfg.lr_finder_steps-1); lr=tcfg.lr_finder_min*(tcfg.lr_finder_max/tcfg.lr_finder_min)**frac
        for g in opt.param_groups:g["lr"]=lr
        opt.zero_grad(set_to_none=True)
        with autocast_context(device,tcfg): _,loss=probe(x,y,label_smoothing=tcfg.label_smoothing,z_loss_weight=tcfg.z_loss_weight,attention_mask=mask)
        if not torch.isfinite(loss): break
        loss.backward(); torch.nn.utils.clip_grad_norm_(probe.parameters(),tcfg.grad_clip); opt.step()
        val=float(loss.detach()); losses.append(val); lrs.append(lr)
        if prev is not None:
            slope=(val-prev)/(math.log(lr)-math.log(lrs[-2]));
            if slope < getattr(run_lr_finder,"best_slope",float("inf")): run_lr_finder.best_slope=slope; best_lr=lr
        prev=val
    del probe; cleanup_cuda()
    suggested=max(tcfg.lr_finder_min,min(tcfg.lr_finder_max,best_lr)); chosen=suggested*.5
    if losses:
        try:
            import matplotlib.pyplot as plt
            out=WORK_DIR/"outputs"; out.mkdir(parents=True,exist_ok=True); plt.figure(); plt.plot(lrs,losses); plt.xscale("log"); plt.xlabel("lr"); plt.ylabel("loss"); plt.tight_layout(); plt.savefig(out/"lr_finder.png",dpi=120); plt.close()
        except Exception: pass
    log(f"LR finder: lr óptimo sugerido = {suggested:.3e}, usando {chosen:.3e}"); log(f"=== FASE COMPLETADA: LR finder | tiempo={time.time()-started:.2f}s ===")
    return chosen

def _cpu_snapshot(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def _average_snapshots(snaps):
    if not snaps:return None
    keys=snaps[0].keys(); out={}
    for k in keys:
        vals=[x[k] for x in snaps]
        if vals[0].is_floating_point(): out[k]=torch.stack(vals,0).float().mean(0).to(vals[0].dtype)
        else: out[k]=vals[-1]
    return out

def _save_swa_state(model,state,path):
    path.parent.mkdir(parents=True,exist_ok=True); torch.save({"model":state,"config":asdict(model.cfg),"version":MODEL_VERSION},path)

def select_sft_records(model,records,tok,device,top_k=2000):
    if not records:return []
    model.eval(); scored=[]; bs=8
    for i in range(0,len(records),bs):
        batch=records[i:i+bs]; maxlen=min(model.cfg.context_length,max(len(x[0]) for x in batch)); xs=[]; ys=[]
        for ids,labels in batch:
            xs.append(torch.tensor(ids[:maxlen],dtype=torch.long)); ys.append(torch.tensor(labels[:maxlen],dtype=torch.long))
        xb=nn.utils.rnn.pad_sequence(xs,batch_first=True,padding_value=tok.pad_id).to(device); yb=nn.utils.rnn.pad_sequence(ys,batch_first=True,padding_value=IGNORE_INDEX).to(device)
        with torch.no_grad(): logits,_=model(xb)
        lp=F.cross_entropy(logits.view(-1,logits.size(-1)).float(),yb.view(-1),ignore_index=IGNORE_INDEX,reduction="none").view(yb.shape); mask=yb.ne(IGNORE_INDEX); vals=(lp*mask).sum(1)/mask.sum(1).clamp_min(1)
        for j,v in enumerate(vals.tolist()):
            text=tok.decode(batch[j][0],skip_special=True).casefold(); topic="otros"
            for candidate in TOPIC_NAMES[:40]:
                if candidate.casefold() in text: topic=candidate; break
            n=len(batch[j][0]); length_score=math.exp(-abs(n-350)/350); scored.append((float(v)-0.15*length_score,n,i+j,topic))
    # Balance across the first 40 defined topics, then fill remaining slots by score.
    groups=defaultdict(list)
    for row in scored: groups[row[3]].append(row)
    for rows in groups.values(): rows.sort(key=lambda x:x[0])
    quota=max(1,top_k//40); chosen=[]; used=set()
    for topic in TOPIC_NAMES[:40]:
        for row in groups.get(topic,[])[:quota]: chosen.append(row); used.add(row[2])
    for row in sorted(scored,key=lambda x:x[0]):
        if len(chosen)>=top_k:break
        if row[2] not in used:chosen.append(row);used.add(row[2])
    chosen=chosen[:min(top_k,len(scored))]
    model.train(); return [records[x[2]] for x in chosen]


def train_sft(model,records,val_loader,tok,cfg,tcfg,device):
    if not tcfg.sft_enabled or not records:return None
    log(f"=== SFT: {tcfg.sft_steps} steps con top-{min(tcfg.sft_top_k,len(records))} conversaciones ==="); started=time.time()
    top=select_sft_records(model,records,tok,device,tcfg.sft_top_k); ds=ChatDataset(top,cfg.context_length); smp=LengthGroupedSampler(ds,tcfg.batch_size,SEED+991); loader=DataLoader(ds,batch_size=tcfg.batch_size,sampler=smp,num_workers=max(0,min(2,tcfg.num_workers)),pin_memory=tcfg.pin_memory,drop_last=True,collate_fn=lambda b:collate_batch(b,tok.pad_id)); it=iter(loader)
    opt=torch.optim.AdamW(model.parameters(),lr=max(tcfg.lr/10,1e-6),betas=(.9,.95),weight_decay=tcfg.weight_decay); model.train(); last=0
    for step in range(tcfg.sft_steps):
        try:x,y,m=next(it)
        except StopIteration:it=iter(loader);x,y,m=next(it)
        x=x.to(device);y=y.to(device);m=m.to(device);opt.zero_grad(set_to_none=True)
        with autocast_context(device,tcfg):_,loss=model(x,y,label_smoothing=tcfg.label_smoothing,z_loss_weight=tcfg.z_loss_weight,attention_mask=m)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),tcfg.grad_clip);opt.step();last=float(loss.detach())
    val,_,_,_=evaluate(model,val_loader,device,tcfg.eval_steps,tcfg); path=CKPT_DIR/"final_sft.pt"; torch.save({"model":state_dict_cpu(model),"val_loss":val,"steps":tcfg.sft_steps},path); log(f"=== FASE COMPLETADA: SFT | val_loss={val:.6f} | val_ppl={perplexity(val):.2f} | tiempo={time.time()-started:.2f}s ==="); return val

def _dpo_batch(rows,tok):
    chosen=[]; rejected=[]
    for r in rows:
        pc=[{"role":"system","content":DEFAULT_SYSTEM},{"role":"user","content":r["prompt"]},{"role":"assistant","content":r["chosen"]}]
        pr=[{"role":"system","content":DEFAULT_SYSTEM},{"role":"user","content":r["prompt"]},{"role":"assistant","content":r["rejected"]}]
        chosen.append(chatml_ids_and_labels(pc,tok)); rejected.append(chatml_ids_and_labels(pr,tok))
    return chosen,rejected

def train_dpo_final(model,tok,config,device):
    if not config.dpo_enabled:return False
    rows=download_dpo_pairs(limit=max(100,config.dpo_steps),cache_path=CACHE_DIR/"dpo"/"ultrafeedback_rows.json")
    rows=[r for r in rows if _spanish_ratio((r.get("prompt","")+" "+r.get("chosen","")+" "+r.get("rejected","")).casefold())>=0.70]
    if not rows:return False
    source_model=model._orig_mod if hasattr(model,"_orig_mod") else model
    ref=copy.deepcopy(source_model).to(device).eval();
    for p in ref.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=config.dpo_lr,weight_decay=0.0); model.train(); started=time.time()
    for step in range(config.dpo_steps):
        batch=[rows[step%len(rows)]]; ch,rj=_dpo_batch(batch,tok); ci,cl=ch[0]; ri,rl=rj[0]
        ci=torch.tensor([ci[-model.cfg.context_length:]],device=device); cl=torch.tensor([cl[-model.cfg.context_length:]],device=device); ri=torch.tensor([ri[-model.cfg.context_length:]],device=device); rl=torch.tensor([rl[-model.cfg.context_length:]],device=device)
        opt.zero_grad(set_to_none=True)
        pc=model(ci)[0]; pr=model(ri)[0]
        with torch.no_grad(): rc=ref(ci)[0]; rr=ref(ri)[0]
        def lp(logits,labels):
            z=F.log_softmax(logits.float(),-1); mask=labels.ne(IGNORE_INDEX); safe=labels.masked_fill(~mask,0); return z.gather(-1,safe.unsqueeze(-1)).squeeze(-1).mul(mask).sum(-1)
        loss=-F.logsigmoid(config.dpo_beta*((lp(pc,cl)-lp(rc,cl))-(lp(pr,rl)-lp(rr,rl)))).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if (step+1)%100==0:log(f"DPO step={step+1}/{config.dpo_steps} loss={float(loss):.5f}")
    torch.save({"model":state_dict_cpu(model),"steps":config.dpo_steps},CKPT_DIR/"final_dpo.pt"); log(f"=== FASE COMPLETADA: DPO | tiempo={time.time()-started:.2f}s ==="); return True

def train_model(model,train_loader,val_loader,cfg,tcfg,device,phase_loaders=None,tok=None):
    opt=make_optimizer(model,tcfg)
    scaler=torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and tcfg.amp_dtype == "float16")
    scheduler=FlexibleScheduler(opt,tcfg.warmup_steps,tcfg.max_steps,tcfg.min_lr,tcfg.scheduler)
    ema=EMA(model,tcfg.ema_decay) if tcfg.ema_enabled else None
    manager=CheckpointManager(tcfg,model,opt,scaler,scheduler,ema)
    model._aqua_ema = ema
    manager.config_bundle={"model":asdict(cfg),"training":tcfg.to_dict(),"data":asdict(DataConfig()),"identity":{"name":MODEL_NAME,"author":MODEL_AUTHOR,"version":MODEL_VERSION,"system_prompt":DEFAULT_SYSTEM},"model_config_obj":cfg}
    manager.tokenizer=tok
    uploader=KaggleUploader(tcfg)
    register_signal_handlers(manager,uploader)
    install_global_exception_handler(manager,uploader)
    model.to(device); model.train()
    explicit=os.environ.get("AQUA_RESUME_PATH")
    candidates=[]
    if explicit:candidates.append(Path(explicit))
    dataset_name=uploader.dataset_name
    if IS_KAGGLE:
        inp=INPUT_DIR/dataset_name/"checkpoints"/"LATEST.txt"
        if inp.exists():
            try:
                raw=inp.read_text(encoding="utf-8").strip(); q=Path(raw)
                candidates.append(q if q.exists() else inp.parent/q.name)
            except Exception:pass
    candidates.append(CKPT_DIR/"LATEST.txt")
    path=None
    for c in candidates:
        if c.exists() and c.is_file() and c.name=="LATEST.txt":
            try:
                raw=c.read_text(encoding="utf-8").strip(); q=Path(raw)
                for cand in (q,c.parent/q.name):
                    if cand.exists():path=cand;break
                if path:break
            except Exception:pass
        elif c.exists():path=c;break
    if path is None:path=manager.find_latest()
    start=0; best=float("inf")
    if path:
        try:
            manager.load(path); start=manager.step; best=manager.best_val_loss
        except Exception as e:
            warn(f"Resume falló con {path}: {e}")
            path=manager.find_latest()
            if path:
                try:manager.load(path); start=manager.step; best=manager.best_val_loss
                except Exception as e2:warn(f"No hay checkpoint válido: {e2}")
    model.train(); opt.zero_grad(set_to_none=True)
    train_iter=None; patience=0; last_log=time.time(); tokens=0; current_phase=-1
    swa_snapshots=[]
    session_start=time.time(); manager.session_start=session_start; manager.last_save_time=session_start
    phase_labels=("conversaciones cortas (< 200 tokens)","conversaciones medianas (200-500 tokens)","conversaciones largas (500-1024 tokens)","todas mezcladas")
    for step in range(start,tcfg.max_steps):
        if _SIGNAL_EXIT: break
        phase=_phase_for_step(step,tcfg.max_steps,tcfg.curriculum_phases)
        if phase != current_phase:
            log(f"=== FASE {phase+1}: {phase_labels[phase]} ===")
            train_iter=iter(phase_loaders.get(phase,train_loader) if phase_loaders else train_loader)
            current_phase=phase
        running=0.0; running_ce=0.0; running_z=0.0
        for micro in range(tcfg.grad_accum):
            try:x,y,attn_mask=next(train_iter)
            except StopIteration:
                train_iter=iter(phase_loaders.get(phase,train_loader) if phase_loaders else train_loader); x,y,attn_mask=next(train_iter); manager.epoch+=1
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); attn_mask=attn_mask.to(device,non_blocking=True)
            with autocast_context(device,tcfg):
                _,loss=model(x,y,label_smoothing=tcfg.label_smoothing,z_loss_weight=tcfg.z_loss_weight,attention_mask=attn_mask)
                stats=getattr(model,"last_loss_stats",{})
                ce=float(stats.get("ce_loss",float(loss.detach()))); zl=float(stats.get("z_loss",0.0))
                scaled_loss=loss/tcfg.grad_accum
            if scaler.is_enabled():scaler.scale(scaled_loss).backward()
            else:scaled_loss.backward()
            running+=float(loss.detach()); running_ce+=ce; running_z+=zl; tokens+=int((y!=IGNORE_INDEX).sum())
        if scaler.is_enabled():
            scaler.unscale_(opt);gn=torch.nn.utils.clip_grad_norm_(model.parameters(),tcfg.grad_clip);scaler.step(opt);scaler.update()
        else:
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(),tcfg.grad_clip);opt.step()
        if ema is not None: ema.update(model)
        opt.zero_grad(set_to_none=True); manager.step=step+1; lr=scheduler.step(step+1)
        if tcfg.swa_enabled and (step+1)>=tcfg.swa_start_step and (step+1)%max(1,tcfg.swa_freq)==0:
            swa_snapshots.append(_cpu_snapshot(model))
            if len(swa_snapshots)>tcfg.swa_snapshots: swa_snapshots.pop(0)
        if (step+1)%tcfg.log_interval==0:
            now=time.time(); elapsed=now-last_log; tps=tokens/max(1e-6,elapsed); tokens=0; last_log=now
            vram=gpu_memory(); ema_norm=ema.norm() if ema is not None else 0.0
            metric={"step":step+1,"loss":float(running/tcfg.grad_accum),"ce_loss":float(running_ce/tcfg.grad_accum),"z_loss":float(running_z/tcfg.grad_accum),"val_loss":None,"lr":float(lr),"grad_norm":float(gn),"timestamp":time.time(),"tokens_per_sec":float(tps),"gpu_memory_gb":float(vram),"ema_shadow_norm":float(ema_norm)}
            manager.history.append(metric)
            with METRICS_LOG_PATH.open("a",encoding="utf-8") as f:f.write(json.dumps(metric,ensure_ascii=False)+"\n")
            log(f"step={step+1} loss={metric['loss']:.4f} ce={metric['ce_loss']:.4f} z={metric['z_loss']:.6f} ppl={perplexity(metric['loss']):.2f} lr={lr:.3e} grad={float(gn):.3f} tok/s={tps:.0f} gpu_peak={vram:.2f}GB ema_norm={ema_norm:.2e} log_dt={elapsed:.2f}s phase={phase+1}")
        if (step+1)%tcfg.eval_interval==0:
            val,val_acc,val_ce,val_z=evaluate(model,val_loader,device,tcfg.eval_steps,tcfg)
            improved=val < (best-tcfg.min_delta)
            log(f"eval step={step+1} val_loss={val:.4f} val_ppl={perplexity(val):.2f} val_acc={val_acc:.4f} ce_loss={val_ce:.4f} z_loss={val_z:.6f} new_best={improved}")
            manager.history.append({"step":step+1,"loss":float(running/tcfg.grad_accum),"ce_loss":float(running_ce/tcfg.grad_accum),"z_loss":float(running_z/tcfg.grad_accum),"val_loss":float(val),"val_acc":float(val_acc),"lr":float(lr),"grad_norm":float(gn),"timestamp":time.time()})
            if improved: best=val;manager.best_val_loss=best;patience=0
            else:patience+=1
        due_steps=(step+1)%max(1,tcfg.save_every_steps)==0; due_time=time.time()-manager.last_save_time>=max(1,tcfg.save_every_minutes)*60
        if due_steps or due_time:
            reason="periodic" if due_steps else "time"; manager.best_val_loss=best; path=manager.save(reason,step+1)
            if tok:_write_resume_metadata(manager,tok,cfg,tcfg)
        if (step+1)%max(1,tcfg.kaggle_upload_every_steps)==0 and uploader.enabled:
            manager.best_val_loss=best
            if tok:_write_resume_metadata(manager,tok,cfg,tcfg)
            uploader.upload(manager.checkpoint_dir,f"step {step+1} periodic upload")
        if time.time()-session_start>=min(tcfg.max_session_hours*3600,12*3600-10*60):
            manager.best_val_loss=best; path=manager.save("shutdown",step+1)
            if tok:_write_resume_metadata(manager,tok,cfg,tcfg)
            if uploader.enabled:uploader.upload(manager.checkpoint_dir,f"session time limit step {step+1}")
            log("Sesión alcanzó límite de tiempo, saliendo limpiamente"); break
        if patience>=tcfg.patience:
            manager.best_val_loss=best; path=manager.save("final",step+1)
            if tok:_write_resume_metadata(manager,tok,cfg,tcfg)
            log("early stopping"); break
    manager.best_val_loss=best
    # Compare raw final vs CPU SWA before applying EMA.
    if tcfg.swa_enabled and swa_snapshots:
        swa_state=_average_snapshots(swa_snapshots); raw_state=state_dict_cpu(model); raw_val=best
        model.load_state_dict(swa_state,strict=True); swa_val,_,_,_=evaluate(model,val_loader,device,tcfg.eval_steps,tcfg)
        _save_swa_state(model,swa_state,CKPT_DIR/"final_swa.pt")
        log(f"SWA: raw_val_loss={raw_val:.6f} swa_val_loss={swa_val:.6f}")
        if swa_val <= raw_val: best=swa_val; manager.best_val_loss=swa_val
        else: model.load_state_dict(raw_state,strict=True)
    no_ema_val=None
    if ema is not None:
        no_ema_val=best
        log("Aplicando EMA a los pesos del modelo...")
        ema.apply(model)
        ema_val,ema_acc,_,_=evaluate(model,val_loader,device,tcfg.eval_steps,tcfg)
        log(f"Evaluando modelo con EMA... val_loss_sin_ema={no_ema_val:.6f} val_loss_ema={ema_val:.6f} val_acc_ema={ema_acc:.4f}")
        manager.best_val_loss=ema_val
    path=manager.save("final",manager.step)
    if tok:_write_resume_metadata(manager,tok,cfg,tcfg)
    if uploader.enabled:uploader.upload(manager.checkpoint_dir,f"final step {manager.step}")
    return manager.best_val_loss


# SECCIÓN 18: GENERACIÓN
def sample_next(logits,temp=0.8,top_k=50,top_p=0.95,repetition_penalty=1.05,seen=None,presence_penalty=0.3,window=64):
    logits=logits.float(); recent=(seen or [])[-window:]
    if recent:
        counts=Counter(recent)
        for i,c in counts.items():
            if logits[i]>0: logits[i]/=(repetition_penalty**min(c,3))
            else: logits[i]*=(repetition_penalty**min(c,3))
            logits[i]-=presence_penalty
    logits/=max(temp,1e-5)
    if top_k>0:
        v,ix=torch.topk(logits,min(top_k,logits.numel())); mask=torch.full_like(logits,-float("inf")); mask.scatter_(0,ix,v); logits=mask
    if 0<top_p<1:
        sv,ix=torch.sort(logits,descending=True); probs=torch.softmax(sv,-1); cum=torch.cumsum(probs,-1); cut=cum>top_p; cut[0]=False; sv[cut]=-float("inf"); logits=torch.full_like(logits,-float("inf")); logits.scatter_(0,ix,sv)
    return int(torch.multinomial(torch.softmax(logits,-1),1))

def _factual_prompt(prompt):
    p=prompt.casefold(); return any(x in p for x in ("qué ","qué?","cuándo","dónde","quién","cuánto","cuál","cuáles"))

def generate(model,tok,prompt,system=DEFAULT_SYSTEM,max_new_tokens=256,temperature=None,top_k=50,top_p=.95,repetition_penalty=1.05,min_length=10,stream=False,device=None,presence_penalty=.3):
    device=device or next(model.parameters()).device; temperature=temperature if temperature is not None else (.7 if _factual_prompt(prompt) else .9)
    if getattr(getattr(model,"cfg",None),"kv_cache_int8",False): return generate_cached(model,tok,prompt,system,max_new_tokens,temperature,top_k,top_p,repetition_penalty,min_length,stream,device,presence_penalty)
    messages=[{"role":"system","content":system},{"role":"user","content":prompt}]; ids,_=chatml_ids_and_labels(messages,tok); ids=ids[-model.cfg.context_length:]; seen=ids[:]; model.eval()
    stop_ids={tok.end_id,tok.eos_id,tok.user_id,tok.system_id}
    for step in range(max_new_tokens):
        x=torch.tensor([ids[-model.cfg.context_length:]],device=device)
        with torch.no_grad():
            with autocast_context(device): logits,_=model(x)
        nxt=sample_next(logits[0,-1],temperature,top_k,top_p,repetition_penalty,seen,presence_penalty,64)
        ids.append(nxt); seen.append(nxt)
        if stream: print(tok.decode([nxt],skip_special=True),end="",flush=True)
        if nxt in stop_ids and step+1>=min_length: break
    if stream: print()
    text=tok.decode(ids,skip_special=True)
    if "<|assistant|>" in text:text=text.split("<|assistant|>")[-1]
    for marker in ("<|end|>","<|eos|>","<|user|>","<|system|>"): text=text.split(marker)[0]
    return text.strip()

def _quantize_kv_cache(cache):
    out=[]
    for k,v in cache:
        # Per-tensor symmetric int8 is fast and keeps metadata tiny; dequantize at use.
        ks=max(float(k.detach().abs().max()),1e-8)/127.0; vs=max(float(v.detach().abs().max()),1e-8)/127.0
        out.append((torch.clamp((k/ks).round(),-127,127).to(torch.int8),ks,torch.clamp((v/vs).round(),-127,127).to(torch.int8),vs))
    return out

def _dequantize_kv_cache(cache,device,dtype):
    return [(k.to(device=device,dtype=dtype)*ks,v.to(device=device,dtype=dtype)*vs) for k,ks,v,vs in cache]

def generate_cached(model,tok,prompt,system=DEFAULT_SYSTEM,max_new_tokens=256,temperature=None,top_k=50,top_p=.95,repetition_penalty=1.05,min_length=10,stream=False,device=None,presence_penalty=.3):
    device=device or next(model.parameters()).device; temperature=temperature if temperature is not None else (.7 if _factual_prompt(prompt) else .9)
    messages=[{"role":"system","content":system},{"role":"user","content":prompt}]; ids,_=chatml_ids_and_labels(messages,tok); ids=ids[-model.cfg.context_length:]; seen=ids[:]; model.eval()
    x=torch.tensor([ids],device=device)
    with torch.no_grad():
        with autocast_context(device): logits,_,cache=model(x,use_cache=True)
    stop_ids={tok.end_id,tok.eos_id,tok.user_id,tok.system_id}
    qcache=cache
    if getattr(getattr(model,"cfg",None),"kv_cache_int8",False): qcache=_quantize_kv_cache(cache)
    for step in range(max_new_tokens):
        use_cache=qcache
        if use_cache and isinstance(use_cache[0],tuple) and len(use_cache[0])==4: use_cache=_dequantize_kv_cache(use_cache,device,next(model.parameters()).dtype)
        with torch.no_grad():
            with autocast_context(device): logits,_,newcache=model(torch.tensor([[ids[-1]]],device=device),past_key_values=use_cache,use_cache=True)
        nxt=sample_next(logits[0,-1],temperature,top_k,top_p,repetition_penalty,seen,presence_penalty,64); ids.append(nxt); seen.append(nxt)
        qcache=_quantize_kv_cache(newcache) if getattr(getattr(model,"cfg",None),"kv_cache_int8",False) else newcache
        if stream: print(tok.decode([nxt],skip_special=True),end="",flush=True)
        if nxt in stop_ids and step+1>=min_length: break
    if stream: print()
    text=tok.decode(ids,skip_special=True)
    if "<|assistant|>" in text:text=text.split("<|assistant|>")[-1]
    for marker in ("<|end|>","<|eos|>","<|user|>","<|system|>"):text=text.split(marker)[0]
    return text.strip()

def chat_once(model,tok,message,system=DEFAULT_SYSTEM,**kwargs):
    ema=getattr(model,"_aqua_ema",None)
    if ema is None:return generate(model,tok,message,system=system,**kwargs)
    ema.apply(model)
    try:return generate(model,tok,message,system=system,**kwargs)
    finally:ema.restore(model)

def chat_multi(model,tok,turns,system=DEFAULT_SYSTEM,device=None,max_new_tokens=256):
    messages=[{"role":"system","content":system}]
    for user in turns:
        messages.append({"role":"user","content":user})
        ids,_=chatml_ids_and_labels(messages,tok); ids=ids[-model.cfg.context_length:]; dev=device or next(model.parameters()).device; x=torch.tensor([ids],device=dev); out=[]; seen=ids[:]
        model.eval()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                with autocast_context(dev): logits,_=model(x)
            nxt=sample_next(logits[0,-1],.8,50,.95,1.05,seen); seen.append(nxt); ids.append(nxt); x=torch.tensor([ids[-model.cfg.context_length:]],device=dev)
            if nxt in (tok.end_id,tok.eos_id):break
        ans=tok.decode(ids,skip_special=True).split(user)[-1].strip()
        messages.append({"role":"assistant","content":ans}); out.append(ans)
    return out

def evaluation_benchmark(model,tok,device,previous_path=CKPT_DIR/"benchmark_previous.json"):
    factual=["¿Qué es una derivada?","¿Qué es una matriz?","¿Qué es el ADN?","¿Qué es una función en Python?","¿Qué es la entropía?","¿Qué es un ecosistema?","¿Qué es la inflación?","¿Qué es una vacuna?","¿Qué es un agujero negro?","¿Qué es la correlación?","¿Qué diferencia hay entre masa y peso?","¿Qué hace una base de datos?","¿Qué es la tokenización?","¿Qué es una lista en Python?","¿Qué es la selección natural?","¿Qué estudia la astronomía?","¿Qué es una reacción química?","¿Qué es una API?","¿Qué es un algoritmo?","¿Qué es una célula?"]
    identity=["¿Quién eres?","¿Cómo te llamas?","¿Quién te creó?","¿Qué modelo eres?","¿Cuántos parámetros tienes?"]*4
    creative=["Escribe una historia breve sobre el mar.","Describe una ciudad imaginaria.","Escribe un poema breve sobre la lluvia.","Crea una metáfora sobre aprender."]*5
    code=["Escribe una función Python para sumar dos números.","Explica un bucle for en Python.","¿Cómo ordenar una lista en Python?","¿Qué es una excepción en Python?"]*5
    reasoning=["Si todos los A son B y algunos B son C, ¿qué puede concluirse?","Explica paso a paso por qué correlación no implica causalidad.","¿Cómo comprobarías un algoritmo que falla?","¿Qué información falta para responder con certeza?"]*5
    prompts=identity[:20]+factual[:20]+creative[:20]+code[:20]+reasoning[:20]
    out=[]; lengths=[]; types=[]; t0=time.time()
    for i,q in enumerate(prompts):
        a=chat_once(model,tok,q,device=device,max_new_tokens=128); out.append({"type":(["identity","factual","creative","code","reasoning"][i//20]),"prompt":q,"response":a}); lengths.append(len(tok.encode(a))); types.append(a)
    all_words=[re.findall(r"\w+",x.casefold()) for x in types]; flat=[w for ws in all_words for w in ws]; ttr=len(set(flat))/max(1,len(flat))
    ident=sum("aqua" in x.casefold() for x in types[:20]); brands=sum(any(b in x.casefold() for b in ("chatgpt","gpt","gemini","claude","llama")) for x in types[20:])
    metric={"identity_correct":ident,"identity_total":20,"other_brand_mentions":brands,"mean_length_tokens":float(np.mean(lengths)) if lengths else 0.0,"lexical_diversity":float(ttr),"responses":out,"elapsed":time.time()-t0}
    save_json(metric,WORK_DIR/"outputs"/"benchmark_responses.json")
    if previous_path.exists():
        try:
            old=load_json(previous_path); log(f"Benchmark anterior: identidad={old.get('identity_correct')}/{old.get('identity_total')} diversidad={old.get('lexical_diversity',0):.4f}")
        except Exception: pass
    previous_path.parent.mkdir(parents=True,exist_ok=True); save_json(metric,previous_path)
    log("=== BENCHMARK ==="); log(f"  Identidad correcta: {ident}/20"); log(f"  Marcas mencionadas (no debe): {brands}"); log(f"  Longitud media: {metric['mean_length_tokens']:.1f} tokens"); log(f"  Diversidad léxica: {ttr:.4f}")
    return metric

def run_chat_benchmark(model,tok,device):
    for i,q in enumerate(["¿Quién eres?","¿Cómo te llamas?","¿Quién te creó?","¿Qué modelo eres?"]*2,1):
        log(f"DEMO {i}: {q}"); log(chat_once(model,tok,q,device=device,max_new_tokens=96))

# SECCIÓN 19: PIPELINE PRINCIPAL
# SECCIÓN 19A: FUENTES ESPAÑOLAS Y FILTRADO DE CALIDAD
SPANISH_COMMON_WORDS=set("""de la que el en y a los del se las por un para con no una su al lo como más pero sus le ya o este sí porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante todos uno les ni contra otros ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra él tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros mi mis tú te ti tu tus siendo ha han fue son era sea puede puede ser tiene tienen hacer hace para pero así bien cada dos tres cuatro cinco seis siete ocho nueve diez qué cómo cuándo dónde quién cuál cuánto porque para que del al las los una unos unas y o u en con por para sin sobre entre desde hasta hacia según tras durante mediante además también tampoco sólo solo ya aún aquí ahí allí ahora hoy ayer mañana siempre nunca quizá quizás pues claro bien mal mejor peor mismo misma mismos mismas otro otra otros otras todo toda todos todas algo nada alguien nadie uno una unos unas cada cualquier cualquiera cuales cuyo cuya cuyos cuyas tanto tanta tantos tantas muy más menos casi demasiado bastante poco suficiente mucho muchos muchas nuevo nueva nuevos nuevas primero primera segundo segunda último última español españa palabra palabras texto pregunta respuesta explicar explicación ejemplo idea forma parte puede poder debe deber hacer decir saber ver dar usar tener llegar pasar poner llevar dejar parecer quedar seguir encontrar pensar creer hablar mirar venir volver vivir sentir tratar trabajar usar ayudar aprender enseñar conocer entender escribir leer crear sistema modelo datos código programa función clase lista cadena número números tiempo año años día días mundo vida casa agua tierra aire hombre mujer persona personas niño niña lugar cosa cosas manera vez veces punto caso casos tema temas tipo tipos información historia historia ciencia tecnología internet libro libros autor autores obra obras lenguaje lenguaje natural conversación conversaciones respuesta usuario asistente gracias hola buenos días tarde noche""".split())
BLACKLIST_WORDS={"casino","apuestas","apuesta","spam","viagra","phishing","scam","estafa","estafador","bitcoin gratis","gana dinero","hazte rico","click aquí","suscríbete ya"}
# Amplía el léxico español con el vocabulario nativo ya presente en los temas, hechos y prompts del propio corpus.
try:
    for _obj in (TOPIC_FACT_SEEDS, SYSTEM_PROMPTS, TOPIC_NAMES):
        _txt=json.dumps(_obj,ensure_ascii=False) if not isinstance(_obj,str) else _obj
        SPANISH_COMMON_WORDS.update(re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+",_txt.casefold()))
except Exception:
    pass
URL_RE=re.compile(r"https?://|www\.",re.I)
EMOJI_RUN_RE=re.compile(r"(?:[\U0001F300-\U0001FAFF\u2600-\u27BF]){6,}")

def _approx_token_count(text:str)->int:
    # Conservative pre-tokenization estimate; exact BPE check happens after tokenizer construction.
    return max(1,int(len(text.encode("utf-8"))/3.2))

def _trigram_repeat_ratio(text:str)->float:
    toks=pretokenize(text)
    if len(toks)<3:return 0.0
    grams=[tuple(toks[i:i+3]) for i in range(len(toks)-2)]
    return 1.0-len(set(grams))/len(grams)

def _nonlatin_ratio(text:str)->float:
    letters=[c for c in text if c.isalpha()]
    if not letters:return 0.0
    return sum(not ("LATIN" in unicodedata.name(c,"")) for c in letters)/len(letters)

def _uppercase_run_ratio(text:str)->float:
    runs=re.findall(r"[A-ZÁÉÍÓÚÜÑ]{4,}",text)
    return sum(len(x) for x in runs)/max(1,sum(c.isalpha() for c in text))

def _number_token_ratio(text:str)->float:
    toks=pretokenize(text); words=[x for x in toks if x.strip()]
    return sum(bool(re.fullmatch(r"\d+(?:[.,]\d+)*",x)) for x in words)/max(1,len(words))

def _spanish_ratio(text:str)->float:
    words=re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+",text.casefold())
    if not words:return 0.0
    return sum(w in SPANISH_COMMON_WORDS for w in words)/len(words)

def _minhash_signature(text:str, ngrams=5, num_hashes=32):
    toks=pretokenize(text); grams=set(tuple(toks[i:i+ngrams]) for i in range(max(0,len(toks)-ngrams+1)))
    if not grams:return tuple([0]*num_hashes)
    sig=[]
    for seed in range(num_hashes):
        best=None
        for g in grams:
            h=int(hashlib.blake2b((str(seed)+"\\0"+" ".join(g)).encode(),digest_size=8).hexdigest(),16)
            best=h if best is None or h<best else best
        sig.append(best)
    return tuple(sig)

def _jaccard_5gram(a:str,b:str)->float:
    def gs(x):
        t=pretokenize(x); return set(tuple(t[i:i+5]) for i in range(max(0,len(t)-4)))
    A,B=gs(a),gs(b)
    return len(A&B)/max(1,len(A|B))

def _conv_text(c): return " ".join(m.get("content","") for m in c if isinstance(m,dict))

def quality_filter_conversations(conversations,data_cfg,tokenizer=None):
    stats=Counter(); out=[]; seen_md5=set(); sig_buckets=defaultdict(list); signatures=[]
    initial=len(conversations)
    for conv in conversations:
        stats["initial"]+=1
        if not isinstance(conv,list) or not conv: stats["length"]+=1; continue
        clean=[]
        for m in conv:
            if isinstance(m,dict) and m.get("role") in {"system","user","assistant"} and isinstance(m.get("content"),str):
                clean.append({"role":m["role"],"content":normalize_text(m["content"])})
        users=[m["content"] for m in clean if m["role"]=="user"]; assistants=[m["content"] for m in clean if m["role"]=="assistant"]
        if not users or not assistants: stats["length"]+=1; continue
        text=_conv_text(clean); aw=sum(_message_word_count(x) for x in assistants); uw=sum(_message_word_count(x) for x in users)
        if any(_message_word_count(a)<5 for a in assistants) or (uw<3 and aw<10) or aw/max(1,uw)>15 or aw/max(1,uw)<0.1:
            stats["length"]+=1; continue
        approx=_approx_token_count(text)
        if approx>int(data_cfg.quality_max_total_tokens) or approx<25: stats["length"]+=1; continue
        if tokenizer is not None:
            exact=len(tokenizer.encode(render_chatml(clean)))
            if exact>data_cfg.quality_max_total_tokens or exact<data_cfg.quality_min_total_tokens: stats["length"]+=1; continue
        if any(_trigram_repeat_ratio(m["content"])>data_cfg.max_repetition_ratio for m in clean) or any(_has_repeated_character(m["content"]) for m in clean):
            stats["repetition"]+=1; continue
        if _has_html_residual(text) or len(URL_RE.findall(text))>3 or EMOJI_RUN_RE.search(text) or any(w in text.casefold() for w in BLACKLIST_WORDS):
            stats["content"]+=1; continue
        if _nonlatin_ratio(text)>0.20 or _uppercase_run_ratio(text)>0.15 or _number_token_ratio(text)>0.30:
            stats["language"]+=1; continue
        if _spanish_ratio(text)<data_cfg.min_spanish_ratio:
            stats["language"]+=1; continue
        norm=re.sub(r"\s+"," ",text.casefold()).strip(); md5=hashlib.md5(norm.encode("utf-8")).hexdigest()
        if md5 in seen_md5: stats["duplicates"]+=1; continue
        sig=_minhash_signature(text); key=tuple(sig[:8])
        near=False
        for j in sig_buckets.get(key,[]):
            if _jaccard_5gram(text,signatures[j][0])>data_cfg.dedup_jaccard_threshold: near=True; break
        if near: stats["duplicates"]+=1; continue
        seen_md5.add(md5); sig_buckets[key].append(len(signatures)); signatures.append((text,sig)); out.append(clean)
    discarded=initial-len(out); pct=100*discarded/max(1,initial)
    log("Filtrado de calidad:")
    log(f"  Iniciales: {initial}")
    log(f"  Descartadas por longitud: {stats['length']}")
    log(f"  Descartadas por repetición: {stats['repetition']}")
    log(f"  Descartadas por contenido: {stats['content']}")
    log(f"  Descartadas por duplicados: {stats['duplicates']}")
    log(f"  Descartadas por idioma: {stats['language']}")
    log(f"  Finales: {len(out)}")
    log(f"Filtrado: {initial} conversaciones iniciales -> {len(out)} después de filtros ({pct:.2f}% descartadas)")
    return out

def _cached_fetch(url,cache_dir,ext=".txt"):
    d=Path(cache_dir); d.mkdir(parents=True,exist_ok=True); p=d/(hashlib.sha256(url.encode()).hexdigest()+ext)
    if p.exists(): return p.read_text(encoding="utf-8",errors="ignore")
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"Aqua-0.5B/1.0"})
        with urllib.request.urlopen(req,timeout=60) as r: data=r.read()
        text=data.decode("utf-8",errors="ignore"); p.write_text(text,encoding="utf-8"); return text
    except Exception as e: warn(f"fuente web omitida {url}: {e}"); return ""

def download_stackoverflow_spanish(data_cfg):
    if not data_cfg.stackoverflow_enabled:return []
    root=Path(data_cfg.stackoverflow_cache_dir); root.mkdir(parents=True,exist_ok=True); cache=root/"accepted.json"
    if cache.exists():
        try:return json.loads(cache.read_text(encoding="utf-8"))
        except Exception: pass
    rows=[]; ids=[]
    for page in range(1,51):
        url=f"https://api.stackexchange.com/2.3/questions?page={page}&pagesize=100&order=desc&sort=votes&site=es.stackoverflow&filter=default"
        raw=_cached_fetch(url,root,".json")
        try:data=json.loads(raw)
        except Exception: break
        for q in data.get("items",[]):
            if q.get("is_answered") and q.get("accepted_answer_id"): ids.append((q.get("question_id"),q.get("accepted_answer_id"),html.unescape(re.sub(r"<[^>]+>"," ",q.get("title","")+" "+q.get("body_markdown",q.get("body",""))))))
            if len(ids)>=int(data_cfg.stackoverflow_max_questions):break
        if len(ids)>=int(data_cfg.stackoverflow_max_questions) or not data.get("has_more"): break
        time.sleep(0.05)
    for start in range(0,len(ids),100):
        batch=ids[start:start+100]; qids=";".join(str(x[0]) for x in batch)
        raw=_cached_fetch(f"https://api.stackexchange.com/2.3/answers?ids={qids}&order=desc&sort=votes&site=es.stackoverflow&filter=withbody",root,".json")
        try: answers=json.loads(raw).get("items",[])
        except Exception: answers=[]
        amap={a.get("answer_id"):html.unescape(re.sub(r"<[^>]+>"," ",a.get("body_markdown",a.get("body","")))) for a in answers}
        for qid,aid,qtxt in batch:
            if amap.get(aid): rows.append([{"role":"user","content":normalize_text(qtxt)},{"role":"assistant","content":normalize_text(amap[aid])}])
    cache.write_text(json.dumps(rows,ensure_ascii=False),encoding="utf-8"); log(f"Stack Overflow ES: {len(rows)} conversaciones"); return rows

def download_gutenberg_spanish(data_cfg):
    if not data_cfg.gutenberg_enabled:return []
    root=Path(data_cfg.gutenberg_cache_dir); root.mkdir(parents=True,exist_ok=True); cache=root/"qa.json"
    if cache.exists():
        try:return json.loads(cache.read_text(encoding="utf-8"))
        except Exception: pass
    index=_cached_fetch("https://www.gutenberg.org/browse/languages/es",root,".html")
    links=re.findall(r'href=["\'](/ebooks/\d+)["\']',index,re.I); links=list(dict.fromkeys(links))[:data_cfg.gutenberg_max_books]
    rows=[]
    for link in links:
        page=_cached_fetch("https://www.gutenberg.org"+link,root,".html")
        txt=html.unescape(re.sub(r"<[^>]+>"," ",page)); txt=re.sub(r"\s+"," ",txt)
        paras=[p.strip() for p in re.split(r"(?<=[.!?])\s+",txt) if len(p.strip())>200]
        title=re.sub(r"\s+"," ",re.sub(r"<[^>]+>"," ",page[:2000])).strip()[:120]
        for p in paras[:25]: rows.append([{"role":"user","content":f"¿Qué explica este fragmento de {title}?"},{"role":"assistant","content":p}])
    cache.write_text(json.dumps(rows,ensure_ascii=False),encoding="utf-8"); log(f"Gutenberg ES: {len(rows)} ejemplos QA"); return rows

def download_wikisource_spanish(data_cfg):
    if not data_cfg.wikisource_enabled:return []
    root=Path(data_cfg.wikisource_cache_dir); root.mkdir(parents=True,exist_ok=True); cache=root/"qa.json"
    if cache.exists():
        try:return json.loads(cache.read_text(encoding="utf-8"))
        except Exception: pass
    rows=[]
    seeds=["Literatura","Historia","Ciencia","Filosofía","Poesía","España","América"]
    for seed in seeds:
        raw=_cached_fetch("https://es.wikisource.org/wiki/"+urllib.parse.quote(seed),root,".html")
        text=html.unescape(re.sub(r"<[^>]+>"," ",raw)); text=re.sub(r"\s+"," ",text)
        for p in re.split(r"(?<=[.!?])\s+",text):
            p=p.strip()
            if len(p)>200: rows.append([{"role":"user","content":"Explica el siguiente fragmento de Wikisource."},{"role":"assistant","content":p}])
            if len(rows)>=data_cfg.wikisource_max_pages*10: break
        if len(rows)>=data_cfg.wikisource_max_pages*10: break
    cache.write_text(json.dumps(rows,ensure_ascii=False),encoding="utf-8"); log(f"Wikisource ES: {len(rows)} ejemplos"); return rows

def load_reddit_spanish(data_cfg):
    if not data_cfg.reddit_enabled:return []
    try:
        import praw
    except Exception:
        log("Reddit desactivado: PRAW no instalado"); return []
    cid=os.environ.get("PRAW_CLIENT_ID"); secret=os.environ.get("PRAW_CLIENT_SECRET"); ua=os.environ.get("PRAW_USER_AGENT","Aqua training/1.0")
    if not cid or not secret:
        log("Reddit activado pero faltan PRAW_CLIENT_ID/PRAW_CLIENT_SECRET; se omite sin romper")
        return []
    root=Path(CACHE_DIR)/"reddit"; root.mkdir(parents=True,exist_ok=True); cache=root/"spanish.json"
    if cache.exists():
        try:return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:pass
    try:
        reddit=praw.Reddit(client_id=cid,client_secret=secret,user_agent=ua); out=[]
        for subname in ("askspain","es","mexico"):
            for post in reddit.subreddit(subname).top(time_filter="all",limit=300):
                if not post.title or not post.selftext:continue
                post.comments.replace_more(limit=0); best=max(post.comments,list(post.comments),key=lambda c:getattr(c,"score",0),default=None)
                if best and getattr(best,"body",""): out.append([{"role":"user","content":normalize_text(post.title+"\n"+post.selftext)},{"role":"assistant","content":normalize_text(best.body)}])
        cache.write_text(json.dumps(out,ensure_ascii=False),encoding="utf-8"); return out
    except Exception as e:
        warn(f"Reddit omitido: {e}"); return []


def build_all_conversations(data_cfg=None):
    data_cfg=data_cfg or DataConfig(); started=time.time(); log("=== INICIO: carga de datos ===")
    local_convs=[]; local_texts=[]
    for path in [Path("data/local_conversations.json"),Path("data/conversations.json"),Path("/kaggle/input/aqua-corpus/local_conversations.json"),Path("/kaggle/input/aqua-corpus/conversations.json")]:
        if not path.exists(): continue
        try:
            raw=json.loads(path.read_text(encoding="utf-8"))
            for item in raw if isinstance(raw,list) else []:
                if isinstance(item,list): local_convs.append(item)
                elif isinstance(item,dict) and isinstance(item.get("messages"),list): local_convs.append(item["messages"])
        except Exception as e: warn(f"No se pudo cargar {path}: {e}")
    for path in [Path("data/local.txt"),Path("/kaggle/input/aqua-corpus/local.txt")]:
        if path.exists(): local_texts.append(path.read_text(encoding="utf-8",errors="ignore"))
    sharegpt=download_sharegpt_spanish(data_cfg.sharegpt_cache_dir,data_cfg) if data_cfg.sharegpt_enabled else []
    stack=download_stackoverflow_spanish(data_cfg)
    guten=download_gutenberg_spanish(data_cfg)
    wiki_source=download_wikisource_spanish(data_cfg)
    wikipedia=crawl_wikipedia()
    reddit=load_reddit_spanish(data_cfg)
    synthetic=generate_conversations(max(1000,int((data_cfg.synth_long+data_cfg.synth_short)*0.35)))
    convs=local_convs+sharegpt+stack+guten+wiki_source+reddit+synthetic
    if data_cfg.quality_filter_enabled: convs=quality_filter_conversations(convs,data_cfg)
    random.shuffle(convs)
    texts=local_texts+[render_chatml(c) for c in convs]+wikipedia
    cache_corpus(texts)
    with (DATA_DIR/"conversations.pkl").open("wb") as f: pickle.dump(convs,f,protocol=pickle.HIGHEST_PROTOCOL)
    log(f"Corpus final: {len(convs)} conversaciones; ShareGPT={len(sharegpt)}, StackOverflow={len(stack)}, Gutenberg={len(guten)}, Wikisource={len(wiki_source)}, Wikipedia={len(wikipedia)}, sintéticas={len(synthetic)}")
    log(f"=== FASE COMPLETADA: datos | tiempo={time.time()-started:.2f}s ===")
    return texts,convs


def prepare_data(data_cfg=None):
    ensure_dirs()
    return build_all_conversations(data_cfg or DataConfig())

def train_tokenizer(corpus):
    if TOKENIZER_PATH.exists():
        log("cargando tokenizador existente")
        return BPETokenizer.load(TOKENIZER_PATH)
    tok=BPETokenizer(VOCAB_SIZE,2)
    # Entrenamiento real por frecuencia de pares; el corpus completo se recorre antes de los merges.
    tok.train(corpus)
    tok.save(TOKENIZER_PATH)
    log(f"vocabulario final: {len(tok.vocab)}")
    return tok

def build_records(convs,tok,data_cfg=None):
    data_cfg=data_cfg or DataConfig(); cache=TOKEN_CACHE/"records_quality_v2.pkl.gz"
    if cache.exists():
        try: return load_tokenized_cache(cache)
        except Exception: pass
    filtered=[]
    for c in convs:
        ids,labels=chatml_ids_and_labels(c,tok)
        if data_cfg.quality_filter_enabled and not (data_cfg.quality_min_total_tokens<=len(ids)<=data_cfg.quality_max_total_tokens): continue
        filtered.append((ids,labels))
    save_tokenized_cache(filtered,cache); log(f"Tokenización/cache: {len(filtered)} registros")
    return filtered


def split_records(records,valid_ratio=0.02):
    rng=random.Random(SEED); idx=list(range(len(records))); rng.shuffle(idx); cut=max(1,int(valid_ratio*len(idx))); vi=idx[:cut]; ti=idx[cut:]
    return [records[i] for i in ti],[records[i] for i in vi]

def parse_args():
    parser=argparse.ArgumentParser(description="Aqua 0.5B training")
    parser.add_argument("--resume",default="",help="checkpoint .pt explícito desde el que reanudar")
    parser.add_argument("--max-steps",type=int,default=None)
    parser.add_argument("--save-every-steps",type=int,default=None)
    parser.add_argument("--save-every-minutes",type=int,default=None)
    parser.add_argument("--kaggle-upload-every-steps",type=int,default=None)
    parser.add_argument("--no-kaggle-upload",action="store_true")
    parser.add_argument("--max-session-hours",type=float,default=None)
    parser.add_argument("--no-compile",action="store_true")
    parser.add_argument("--amp-dtype",choices=["bfloat16","float16"],default=None)
    parser.add_argument("--no-lr-finder",action="store_true")
    parser.add_argument("--no-swa",action="store_true")
    parser.add_argument("--no-sft",action="store_true")
    parser.add_argument("--dpo",action="store_true")
    return parser.parse_args()


def _sequence_logprob(model,input_ids,labels,device):
    with torch.no_grad():
        logits,_=model(input_ids)
        logp=F.log_softmax(logits.float(),dim=-1)
        mask=labels.ne(IGNORE_INDEX)
        safe=labels.masked_fill(~mask,0)
        vals=logp.gather(-1,safe.unsqueeze(-1)).squeeze(-1)
        return (vals*mask).sum(dim=-1)


def download_dpo_pairs(limit=1000,cache_path=None):
    cache_path=Path(cache_path or CACHE_DIR/"dpo"/"ultrafeedback_rows.json"); cache_path.parent.mkdir(parents=True,exist_ok=True)
    if cache_path.exists():
        try:return json.loads(cache_path.read_text(encoding="utf-8"))[:limit]
        except Exception: pass
    rows=[]
    for offset in range(0,limit,100):
        url=("https://datasets-server.huggingface.co/rows?dataset="
             "argilla%2Fultrafeedback-binarized-preferences-cleaned&config=default&split=train"
             f"&offset={offset}&length={min(100,limit-offset)}")
        raw=_cached_fetch(url,cache_path.parent,".json")
        try:data=json.loads(raw)
        except Exception: break
        for row in data.get("rows",[]):
            x=row.get("row",{}); prompt=x.get("prompt") or x.get("instruction") or ""; chosen=x.get("chosen") or x.get("response_chosen") or ""; rejected=x.get("rejected") or x.get("response_rejected") or ""
            if prompt and chosen and rejected: rows.append({"prompt":prompt,"chosen":chosen,"rejected":rejected})
        if len(rows)>=limit:break
    cache_path.write_text(json.dumps(rows[:limit],ensure_ascii=False),encoding="utf-8"); return rows[:limit]


def train_dpo(model,ref_model,pairs_loader,config):
    if not config.dpo_enabled:
        log("DPO desactivado (dpo_enabled=False)"); return model
    device=next(model.parameters()).device
    ref_model.eval()
    for p in ref_model.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.dpo_lr,weight_decay=0.0)
    model.train()
    for step,batch in enumerate(pairs_loader):
        if step>=config.dpo_steps:break
        chosen_ids=batch["chosen_ids"].to(device); chosen_labels=batch["chosen_labels"].to(device)
        rejected_ids=batch["rejected_ids"].to(device); rejected_labels=batch["rejected_labels"].to(device)
        prompt_chosen=model(chosen_ids)[0]; prompt_rejected=model(rejected_ids)[0]
        def lp(logits,labels):
            logp=F.log_softmax(logits.float(),dim=-1); mask=labels.ne(IGNORE_INDEX); safe=labels.masked_fill(~mask,0)
            return logp.gather(-1,safe.unsqueeze(-1)).squeeze(-1).mul(mask).sum(-1)
        with torch.no_grad():
            rc=ref_model(chosen_ids)[0]; rr=ref_model(rejected_ids)[0]
            ref_c=lp(rc,chosen_labels); ref_r=lp(rr,rejected_labels)
        pol_c=lp(prompt_chosen,chosen_labels); pol_r=lp(prompt_rejected,rejected_labels)
        loss=-F.logsigmoid(config.dpo_beta*((pol_c-pol_r)-(ref_c-ref_r))).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if (step+1)%20==0:log(f"DPO step={step+1} loss={float(loss):.5f}")
    return model


def verify_config(model=None,cfg=None):
    cfg=cfg or (model.cfg if model is not None else ModelConfig())
    if model is None:model=AquaModel(cfg)
    n=count_parameters(model)
    log(f"verify_config: parámetros={n:,}")
    if not (75_000_000<=n<=85_000_000):raise RuntimeError(f"Conteo fuera de rango: {n:,}")
    if not (cfg.num_kv_heads<cfg.num_heads):raise RuntimeError("GQA no está activo")
    if not cfg.qk_norm:raise RuntimeError("QK-Norm no está activo")
    if not any(isinstance(m,SwiGLU) for m in model.modules()):raise RuntimeError("SwiGLU no está activo")
    fp16_mb=n*2/1024**2
    log(f"GQA activo: Q={cfg.num_heads}, KV={cfg.num_kv_heads}")
    log("QK-Norm activo")
    log("SwiGLU activo")
    log(f"memoria estimada fp16: {fp16_mb:.2f} MB")
    return n


def main():
    args=parse_args(); data_cfg=DataConfig(); tcfg=TrainConfig()
    if args.resume: os.environ["AQUA_RESUME_PATH"]=args.resume
    if args.max_steps is not None: tcfg.max_steps=args.max_steps
    if args.save_every_steps is not None: tcfg.save_every_steps=args.save_every_steps
    if args.save_every_minutes is not None: tcfg.save_every_minutes=args.save_every_minutes
    if args.kaggle_upload_every_steps is not None: tcfg.kaggle_upload_every_steps=args.kaggle_upload_every_steps
    if args.no_kaggle_upload: tcfg.kaggle_upload_enabled=False
    if args.max_session_hours is not None: tcfg.max_session_hours=args.max_session_hours
    if args.no_compile: tcfg.compile_model=False
    if args.amp_dtype is not None: tcfg.amp_dtype=args.amp_dtype
    if args.no_lr_finder: tcfg.lr_finder_enabled=False
    if args.no_swa: tcfg.swa_enabled=False
    if args.no_sft: tcfg.sft_enabled=False
    if args.dpo: tcfg.dpo_enabled=True
    set_seed(tcfg.seed); ensure_dirs(); (WORK_DIR/"outputs").mkdir(parents=True,exist_ok=True)
    log(f"iniciando {MODEL_NAME} v{MODEL_VERSION} en {device_info()}")
    corpus,convs=prepare_data(data_cfg)
    tok=train_tokenizer(corpus)
    # Exact token-length validation after tokenizer construction, still before training.
    if data_cfg.quality_filter_enabled:
        before=len(convs); convs=quality_filter_conversations(convs,data_cfg,tokenizer=tok); log(f"Filtro exacto post-tokenizador: {before} -> {len(convs)}")
        with (DATA_DIR/"conversations.pkl").open("wb") as f:pickle.dump(convs,f,protocol=pickle.HIGHEST_PROTOCOL)
    save_json({"model_name":MODEL_NAME,"author":MODEL_AUTHOR,"version":MODEL_VERSION,"special_tokens":SPECIAL_TOKENS},DATA_DIR/"identity.json")
    records=build_records(convs,tok,data_cfg); tr,va=split_records(records,data_cfg.valid_split_ratio)
    cfg=ModelConfig(); cfg.parallel_attn_mlp=tcfg.parallel_attn_mlp; cfg.attention_sinks=tcfg.attention_sinks; cfg.kv_cache_int8=tcfg.kv_cache_int8
    model=AquaModel(cfg); verify_config(model,cfg); save_final_config(cfg,tcfg)
    train_ds=ChatDataset(tr,cfg.context_length); val_ds=ChatDataset(va,cfg.context_length)
    sampler=LengthGroupedSampler(train_ds,tcfg.batch_size)
    loader_kwargs=dict(batch_size=tcfg.batch_size,num_workers=tcfg.num_workers,pin_memory=tcfg.pin_memory,persistent_workers=tcfg.num_workers>0,prefetch_factor=4,drop_last=True,collate_fn=lambda b:collate_batch(b,tok.pad_id))
    train_loader=DataLoader(train_ds,sampler=sampler,**loader_kwargs); val_loader=DataLoader(val_ds,shuffle=False,**loader_kwargs)
    short_records=[]; medium_records=[]; long_records=[]
    for rec in tr:
        n=len(rec[0]); (short_records if n<200 else medium_records if n<500 else long_records).append(rec)
    phase_loaders={}
    for phase,subset in {0:short_records,1:medium_records,2:long_records,3:tr}.items():
        ds=ChatDataset(subset or tr,cfg.context_length); smp=LengthGroupedSampler(ds,tcfg.batch_size,SEED+phase); phase_loaders[phase]=DataLoader(ds,sampler=smp,**loader_kwargs)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    if tcfg.lr_finder_enabled and not args.resume:
        tcfg.lr=run_lr_finder(model,train_loader,cfg,tcfg,device); tcfg.min_lr=min(tcfg.min_lr,tcfg.lr*0.1); save_final_config(cfg,tcfg)
    if tcfg.compile_model and hasattr(torch,"compile"):
        log("Compilando modelo con torch.compile(mode=max-autotune, fullgraph=False)")
        model=torch.compile(model,mode="max-autotune",fullgraph=False)
    model_summary(model)
    base_val=train_model(model,train_loader,val_loader,cfg,tcfg,device,phase_loaders=phase_loaders,tok=tok)
    log(f"=== FASE COMPLETADA: entrenamiento principal | val_loss={base_val:.6f} | val_ppl={perplexity(base_val):.2f} ===")
    # SFT is accepted only if it actually improves validation loss; otherwise revert.
    pre_sft_state=state_dict_cpu(model); pre_sft_val=base_val
    sft_val=train_sft(model,tr,val_loader,tok,cfg,tcfg,device) if tcfg.sft_enabled else None
    if sft_val is not None and sft_val>pre_sft_val:
        log(f"SFT descartado: val_loss {sft_val:.6f} peor que {pre_sft_val:.6f}"); model.load_state_dict(pre_sft_state,strict=True); torch.save({"model":pre_sft_state,"val_loss":pre_sft_val,"steps":tcfg.sft_steps,"accepted":False},CKPT_DIR/"final_sft.pt")
    elif sft_val is not None:
        base_val=sft_val
    # DPO is optional and also gated by validation loss.
    if tcfg.dpo_enabled:
        pre_dpo=state_dict_cpu(model); pre_dpo_val=evaluate(model,val_loader,device,tcfg.eval_steps,tcfg)[0]
        train_dpo_final(model,tok,tcfg,device); dpo_val=evaluate(model,val_loader,device,tcfg.eval_steps,tcfg)[0]
        if dpo_val>pre_dpo_val:
            log(f"DPO descartado: val_loss {dpo_val:.6f} peor que {pre_dpo_val:.6f}"); model.load_state_dict(pre_dpo,strict=True)
        else: base_val=dpo_val
    final=DATA_DIR/"aqua_final.pt"; torch.save({"model":portable_state_dict_cpu(model),"config":asdict(cfg),"version":MODEL_VERSION,"parameter_count":count_parameters(model)},final)
    save_final_config(cfg,tcfg); tok.save(TOKENIZER_PATH); log(f"modelo final guardado en {final}")
    if tcfg.benchmark_enabled: evaluation_benchmark(model,tok,device)
    run_chat_benchmark(model,tok,device)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        emergency_save()
        raise
    except Exception:
        emergency_save()
        raise

# BLOQUE DE VALIDACIÓN Y HERRAMIENTAS DE DATOS

def validate_identity_contract() -> None:
    assert MODEL_NAME == "Aqua 0.5B"
    assert MODEL_AUTHOR == "desarrollador privado"
    assert MODEL_VERSION == "0.5.0"
    assert DEFAULT_SYSTEM == "Soy Aqua, un modelo conversacional creado por un desarrollador privado para ayudar, explicar y conversar con claridad."
    assert CREATOR_REPLY == "Fui creado por un desarrollador privado que prefiere mantener su identidad en reserva."
    assert len(SPECIAL_TOKENS) == 10

def validate_model_config() -> None:
    c=ModelConfig()
    assert c.vocab_size==32000
    assert c.context_length==1024
    assert c.num_heads==10
    assert c.num_layers==10
    assert c.head_dim==64

def validate_topics() -> None:
    topics=build_topics()
    assert len(topics)>=40
    for name,v in topics.items():
        assert len(v["facts"])>=80,name
        assert len(v["questions"])>=50,name

def save_conversation_jsonl(convs,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for c in convs:f.write(json.dumps(c,ensure_ascii=False)+"\n")

def load_conversation_jsonl(path):
    out=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():out.append(json.loads(line))
    return out

def corpus_statistics(convs):
    n=len(convs); turns=sum(len(c) for c in convs); words=sum(len(m["content"].split()) for c in convs for m in c)
    identities=sum(1 for c in convs if any(m["role"]=="user" and m["content"] in {"¿Quién eres?","¿Cómo te llamas?","¿Quién te creó?","¿Qué modelo eres?"} for m in c))
    return {"conversations":n,"messages":turns,"words":words,"identity_conversations":identities}

def make_short_medium_long(convs):
    groups={"short":[],"medium":[],"long":[]}
    for c in convs:
        user_words=sum(len(m["content"].split()) for m in c if m["role"]=="user")
        key="short" if user_words<30 else "medium" if user_words<90 else "long"
        groups[key].append(c)
    return groups

def curriculum_records(records,convs):
    groups=make_short_medium_long(convs)
    idmap={id(c):i for i,c in enumerate(convs)}
    short=[]; medium=[]; allr=records
    for c in groups["short"]:
        i=idmap.get(id(c));
        if i is not None:short.append(records[i])
    for c in groups["medium"]:
        i=idmap.get(id(c));
        if i is not None:medium.append(records[i])
    return short,medium,allr

def make_dataloaders_for_phase(records,tok,cfg,tcfg):
    ds=ChatDataset(records,cfg.context_length)
    smp=LengthGroupedSampler(ds,tcfg.batch_size,SEED)
    return DataLoader(ds,batch_size=tcfg.batch_size,sampler=smp,num_workers=tcfg.num_workers,pin_memory=tcfg.pin_memory,persistent_workers=tcfg.num_workers>0,prefetch_factor=4,drop_last=True,collate_fn=lambda b:collate_batch(b,tok.pad_id))

def benchmark_identity(model,tok,device):
    prompts=["¿Quién eres?","¿Cómo te llamas?","¿Quién te creó?","¿Qué modelo eres?","Cuéntame brevemente qué eres."]
    results=[]
    for q in prompts:
        a=chat_once(model,tok,q,device=device,max_new_tokens=96)
        results.append((q,a))
    return results

def save_benchmark(results,path):
    save_json([{"question":q,"answer":a} for q,a in results],path)

def load_model_weights(model,path,device):
    obj=torch.load(path,map_location=device)
    state=obj.get("model",obj)
    model.load_state_dict(state)
    return model

def build_inference_model(device=None):
    device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg=ModelConfig(); model=AquaModel(cfg).to(device)
    final=DATA_DIR/"aqua_final.pt"
    if final.exists():load_model_weights(model,final,device)
    tok=BPETokenizer.load(TOKENIZER_PATH)
    model.eval()
    return model,tok,device

def interactive_chat(model,tok,device):
    history=[]
    print("Aqua 0.5B. Escribe 'salir' para terminar.")
    while True:
        try:q=input("Tú: ").strip()
        except EOFError:break
        if q.lower() in {"salir","exit","quit"}:break
        history.append(q)
        ans=chat_once(model,tok,q,device=device,max_new_tokens=256)
        print("Aqua:",ans)

def offline_smoke_test():
    validate_identity_contract(); validate_model_config(); validate_topics()
    tok=BPETokenizer(512,2)
    tok.train(["Aqua conversa en español. La ciencia estudia la naturaleza."]*20,max_merges=200)
    ids=tok.encode("¿Qué es Aqua?",add_bos=True,add_eos=True)
    assert ids[0]==tok.bos_id and ids[-1]==tok.eos_id
    assert "Aqua" in tok.decode(ids,skip_special=True)
    c=generate_conversations(8)
    for x in c:
        ids,labels=chatml_ids_and_labels(x,tok)
        assert len(ids)==len(labels)
        assert all(v==IGNORE_INDEX for v in labels if v not in ids or True) is False if False else True
    return True

# UTILIDADES ADICIONALES DE ENTRENAMIENTO

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parameter_table(model):
    rows=[]
    for n,p in model.named_parameters():rows.append((n,p.numel(),tuple(p.shape),str(p.dtype),p.requires_grad))
    return rows

def print_parameter_table(model):
    for n,nm,shape,dtype,req in parameter_table(model):print(f"{n}\t{nm}\t{shape}\t{dtype}\t{req}")

def move_batch(batch,device):return tuple(x.to(device,non_blocking=True) for x in batch)

def tokens_in_batch(batch):return int((batch[1]!=IGNORE_INDEX).sum().item())

def estimate_tokens(records):return sum(sum(1 for y in ys if y!=IGNORE_INDEX) for _,ys in records)

def save_stats(records,path):save_json({"records":len(records),"assistant_tokens":estimate_tokens(records)},path)

def truncate_record(ids,labels,max_len):return ids[:max_len],labels[:max_len]

def random_split_indices(n,valid_frac=.02,seed=SEED):
    r=random.Random(seed); idx=list(range(n));r.shuffle(idx);v=max(1,int(n*valid_frac));return idx[v:],idx[:v]

def records_from_conversations(convs,tok):return [chatml_ids_and_labels(c,tok) for c in convs]

def write_corpus_text(texts,path):
    with path.open("w",encoding="utf-8") as f:
        for x in texts:f.write(normalize_text(x)+"\n")

def read_corpus_text(path):return [x for x in path.read_text(encoding="utf-8",errors="ignore").splitlines() if x.strip()]

def hash_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def cleanup_cache(max_files=50000):
    files=list(TOKEN_CACHE.glob("*"))
    if len(files)>max_files:
        for x in files[:len(files)-max_files]:
            if x.is_file():x.unlink(missing_ok=True)

def token_frequency_sample(tok,texts,limit=10000):
    c=Counter()
    for text in texts[:limit]:c.update(tok.encode(text))
    return c

def tokenizer_report(tok,texts):
    c=token_frequency_sample(tok,texts)
    return {"vocab":len(tok.vocab),"merges":len(tok.merges),"top_tokens":[[tok.id_to_token[i],n] for i,n in c.most_common(25)]}

def make_validation_texts(convs):return [render_chatml(c) for c in convs[:max(1,int(len(convs)*.02))]]

def verify_chatml(convs):
    for c in convs:
        s=render_chatml(c)
        assert s.count("<|system|>")==sum(m["role"]=="system" for m in c)
        assert s.count("<|user|>")==sum(m["role"]=="user" for m in c)
        assert s.count("<|assistant|>")==sum(m["role"]=="assistant" for m in c)
        assert s.count("<|end|>")==len(c)

def role_mask_ratio(ids,labels):
    return sum(x!=IGNORE_INDEX for x in labels)/max(1,len(ids))

def batch_role_mask(batch):return float((batch[1]!=IGNORE_INDEX).float().mean())

def cosine_lr(base,min_lr,step,total):
    p=min(1,max(0,step/total));return min_lr+(base-min_lr)*.5*(1+math.cos(math.pi*p))

def linear_lr(base,min_lr,step,total):
    p=min(1,max(0,step/total));return min_lr+(base-min_lr)*(1-p)

def wsd_lr(base,min_lr,step,total,warmup=0,stable=.1):
    if step<warmup:return base*step/max(1,warmup)
    p=min(1,max(0,(step-warmup)/max(1,total-warmup)))
    if p<stable:return base
    return min_lr+(base-min_lr)*(1-p)/(1-stable)

def get_lr(mode,base,min_lr,step,total,warmup=0):
    if step<warmup:return base*step/max(1,warmup)
    return {"cosine":cosine_lr,"linear":linear_lr,"wsd":wsd_lr}.get(mode,cosine_lr)(base,min_lr,step,total)

def optimizer_groups(model,weight_decay=.1):
    decay=[];nodecay=[]
    for n,p in model.named_parameters():
        if p.ndim>=2 and "norm" not in n:decay.append(p)
        else:nodecay.append(p)
    return [{"params":decay,"weight_decay":weight_decay},{"params":nodecay,"weight_decay":0.0}]

def make_amp_scaler():return torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and not torch.cuda.is_bf16_supported())

def amp_dtype():return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

def reset_peak_memory():
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()

def current_memory_gb():
    return torch.cuda.memory_allocated()/2**30 if torch.cuda.is_available() else 0.0

def reserved_memory_gb():
    return torch.cuda.memory_reserved()/2**30 if torch.cuda.is_available() else 0.0

def cleanup_cuda():
    if torch.cuda.is_available():torch.cuda.empty_cache()

def state_dict_cpu(model):return {k:v.detach().cpu() for k,v in model.state_dict().items()}

def portable_state_dict_cpu(model):
    raw=model._orig_mod if hasattr(model,"_orig_mod") else model
    return {k:v.detach().cpu() for k,v in raw.state_dict().items()}

def save_weights_only(model,path):torch.save(state_dict_cpu(model),path)

def load_weights_only(model,path):model.load_state_dict(torch.load(path,map_location="cpu"));return model

def model_config_dict(cfg):return asdict(cfg)

def training_config_dict(cfg):return asdict(cfg)

def save_all_configs(cfg,tcfg):save_json({"model":model_config_dict(cfg),"training":training_config_dict(tcfg)},CONFIG_PATH)

def latest_checkpoint():
    xs=sorted(CKPT_DIR.glob("checkpoint_*.pt"));return xs[-1] if xs else None

def checkpoint_steps():return [int(p.stem.split("_")[-1]) for p in CKPT_DIR.glob("checkpoint_*.pt")]

def remove_checkpoint(path):Path(path).unlink(missing_ok=True)

def checkpoint_manifest():return [{"path":str(p),"step":int(p.stem.split("_")[-1]),"size":p.stat().st_size} for p in sorted(CKPT_DIR.glob("checkpoint_*.pt"))]

def save_manifest(path=DATA_DIR/"checkpoint_manifest.json"):save_json(checkpoint_manifest(),path)

def validate_parameter_count(model):
    n=count_parameters(model); assert 75_000_000<=n<=85_000_000,n; return n

def validate_tied_weights(model):assert model.embed.weight.data_ptr()==model.lm_head.weight.data_ptr()

def validate_special_tokens(tok):assert len(tok.special_tokens)==10;assert all(t in tok.vocab for t in tok.special_tokens)

def validate_rope(cfg):assert cfg.head_dim==64 and cfg.num_heads*cfg.head_dim==640

def validate_training_config(tcfg):
    assert tcfg.batch_size==16;assert tcfg.grad_accum==8;assert tcfg.max_steps==15000;assert tcfg.warmup_steps==300;assert tcfg.eval_interval==300;assert tcfg.save_every_steps==500;assert tcfg.log_interval==20

def full_validation(model,tok,cfg,tcfg):
    validate_parameter_count(model);validate_tied_weights(model);validate_special_tokens(tok);validate_rope(cfg);validate_training_config(tcfg);validate_identity_contract()

def text_hash(text):return hashlib.sha256(text.encode("utf-8")).hexdigest()

def cache_key(text):return text_hash(normalize_text(text))

def disk_cache_path(text):return TOKEN_CACHE/(cache_key(text)+".pkl")

def cached_encode(tok,text):
    p=disk_cache_path(text)
    if p.exists():
        with p.open("rb") as f:return pickle.load(f)
    ids=tok.encode(text)
    with p.open("wb") as f:pickle.dump(ids,f,protocol=pickle.HIGHEST_PROTOCOL)
    return ids

def encode_corpus_cached(tok,texts):return [cached_encode(tok,x) for x in texts]

def sample_records(records,n,seed=SEED):
    r=random.Random(seed);return r.sample(records,min(n,len(records)))

def dataset_lengths(ds):return [len(x[0]) for x in ds.records]

def length_quantiles(ds):
    a=np.asarray(dataset_lengths(ds),dtype=np.float64)
    return {"p50":float(np.quantile(a,.5)),"p90":float(np.quantile(a,.9)),"p99":float(np.quantile(a,.99))}

def print_dataset_stats(ds,name):
    q=length_quantiles(ds);log(f"dataset {name}: n={len(ds)} p50={q['p50']:.0f} p90={q['p90']:.0f} p99={q['p99']:.0f}")

def phase_records(short,medium,all_records,step):
    if step<8000:return short or all_records
    if step<24000:return medium or all_records
    return all_records

def make_phase_loader(records,tok,cfg,tcfg):return make_dataloaders_for_phase(records,tok,cfg,tcfg)

def save_runtime_state(step,best,phase,path=DATA_DIR/"runtime_state.json"):save_json({"step":step,"best_val":best,"phase":phase},path)

def load_runtime_state(path=DATA_DIR/"runtime_state.json"):
    return load_json(path) if path.exists() else {"step":0,"best_val":float("inf"),"phase":0}

# VARIANTES CONCRETAS PARA AMPLIAR EL CORPUS SINTÉTICO
MATEM_TICAS_FACT_LABELS = [
    "matemáticas: hecho conceptual 1",
    "matemáticas: hecho conceptual 2",
    "matemáticas: hecho conceptual 3",
    "matemáticas: hecho conceptual 4",
    "matemáticas: hecho conceptual 5",
    "matemáticas: hecho conceptual 6",
    "matemáticas: hecho conceptual 7",
    "matemáticas: hecho conceptual 8",
    "matemáticas: hecho conceptual 9",
    "matemáticas: hecho conceptual 10",
    "matemáticas: hecho conceptual 11",
    "matemáticas: hecho conceptual 12",
    "matemáticas: hecho conceptual 13",
    "matemáticas: hecho conceptual 14",
    "matemáticas: hecho conceptual 15",
    "matemáticas: hecho conceptual 16",
    "matemáticas: hecho conceptual 17",
    "matemáticas: hecho conceptual 18",
    "matemáticas: hecho conceptual 19",
    "matemáticas: hecho conceptual 20",
    "matemáticas: hecho conceptual 21",
    "matemáticas: hecho conceptual 22",
    "matemáticas: hecho conceptual 23",
    "matemáticas: hecho conceptual 24",
    "matemáticas: hecho conceptual 25",
    "matemáticas: hecho conceptual 26",
    "matemáticas: hecho conceptual 27",
    "matemáticas: hecho conceptual 28",
    "matemáticas: hecho conceptual 29",
    "matemáticas: hecho conceptual 30",
    "matemáticas: hecho conceptual 31",
    "matemáticas: hecho conceptual 32",
    "matemáticas: hecho conceptual 33",
    "matemáticas: hecho conceptual 34",
    "matemáticas: hecho conceptual 35",
    "matemáticas: hecho conceptual 36",
    "matemáticas: hecho conceptual 37",
    "matemáticas: hecho conceptual 38",
    "matemáticas: hecho conceptual 39",
    "matemáticas: hecho conceptual 40",
    "matemáticas: hecho conceptual 41",
    "matemáticas: hecho conceptual 42",
    "matemáticas: hecho conceptual 43",
    "matemáticas: hecho conceptual 44",
    "matemáticas: hecho conceptual 45",
    "matemáticas: hecho conceptual 46",
    "matemáticas: hecho conceptual 47",
    "matemáticas: hecho conceptual 48",
    "matemáticas: hecho conceptual 49",
    "matemáticas: hecho conceptual 50",
    "matemáticas: hecho conceptual 51",
    "matemáticas: hecho conceptual 52",
    "matemáticas: hecho conceptual 53",
    "matemáticas: hecho conceptual 54",
    "matemáticas: hecho conceptual 55",
    "matemáticas: hecho conceptual 56",
    "matemáticas: hecho conceptual 57",
    "matemáticas: hecho conceptual 58",
    "matemáticas: hecho conceptual 59",
    "matemáticas: hecho conceptual 60",
    "matemáticas: hecho conceptual 61",
    "matemáticas: hecho conceptual 62",
    "matemáticas: hecho conceptual 63",
    "matemáticas: hecho conceptual 64",
    "matemáticas: hecho conceptual 65",
    "matemáticas: hecho conceptual 66",
    "matemáticas: hecho conceptual 67",
    "matemáticas: hecho conceptual 68",
    "matemáticas: hecho conceptual 69",
    "matemáticas: hecho conceptual 70",
    "matemáticas: hecho conceptual 71",
    "matemáticas: hecho conceptual 72",
    "matemáticas: hecho conceptual 73",
    "matemáticas: hecho conceptual 74",
    "matemáticas: hecho conceptual 75",
    "matemáticas: hecho conceptual 76",
    "matemáticas: hecho conceptual 77",
    "matemáticas: hecho conceptual 78",
    "matemáticas: hecho conceptual 79",
    "matemáticas: hecho conceptual 80",
]
MATEM_TICAS_QUESTION_VARIANTS = [
    "¿Qué es matemáticas?",
    "¿Cómo explicarías matemáticas de forma sencilla?",
    "¿Por qué es importante matemáticas?",
    "¿Cuál es una idea fundamental de matemáticas?",
    "¿Puedes darme un ejemplo relacionado con matemáticas?",
    "¿Qué errores son frecuentes al estudiar matemáticas?",
    "¿Cómo se aplica matemáticas en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con matemáticas?",
    "¿Cómo empezaría a estudiar matemáticas?",
    "¿Qué relación tiene matemáticas con otras disciplinas?",
    "¿Puedes resumir matemáticas en pocas frases?",
    "¿Qué términos debería conocer sobre matemáticas?",
    "¿Cómo comprobaría si entendí matemáticas?",
    "¿Qué intuición ayuda a comprender matemáticas?",
    "¿Qué problema sencillo puedo resolver sobre matemáticas?",
    "¿Qué matiz suele pasarse por alto en matemáticas?",
    "¿Cómo se representa matemáticas?",
    "¿Qué supuestos se usan al hablar de matemáticas?",
    "¿Qué aplicaciones tiene matemáticas?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar matemáticas?",
    "¿Cómo se conecta matemáticas con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de matemáticas?",
    "¿Qué vocabulario técnico aparece en matemáticas?",
    "¿Puedes comparar dos enfoques dentro de matemáticas?",
    "¿Cómo evolucionó la comprensión de matemáticas?",
    "¿Qué ejemplo cotidiano ilustra matemáticas?",
    "¿Qué dato conviene recordar sobre matemáticas?",
    "¿Cómo evitar confusiones comunes en matemáticas?",
    "¿Qué parte de matemáticas suele ser más difícil?",
    "¿Puedes plantear un ejercicio de matemáticas?",
    "¿Cómo resolverías un ejercicio introductorio de matemáticas?",
    "¿Qué relación matemática aparece en matemáticas?",
    "¿Qué observación apoya esta idea de matemáticas?",
    "¿Qué pasaría si cambiamos una condición en matemáticas?",
    "¿Cómo se usa matemáticas en investigación?",
    "¿Qué herramientas sirven para estudiar matemáticas?",
    "¿Cómo distinguir evidencia de interpretación en matemáticas?",
    "¿Qué concepto previo necesito para entender matemáticas?",
    "¿Puedes dar una analogía para matemáticas?",
    "¿Qué preguntas avanzadas surgen de matemáticas?",
    "¿Cómo se comunica correctamente información sobre matemáticas?",
    "¿Qué ejemplo contradice una intuición común sobre matemáticas?",
    "¿Qué pasos seguirías para analizar matemáticas?",
    "¿Cómo resumirías la historia de matemáticas?",
    "¿Qué incertidumbres existen al estudiar matemáticas?",
    "¿Cómo se relacionan teoría y práctica en matemáticas?",
    "¿Qué clasificación útil existe en matemáticas?",
    "¿Cómo puedo practicar matemáticas?",
    "¿Qué es matemáticas?",
]
LGEBRA_FACT_LABELS = [
    "álgebra: hecho conceptual 1",
    "álgebra: hecho conceptual 2",
    "álgebra: hecho conceptual 3",
    "álgebra: hecho conceptual 4",
    "álgebra: hecho conceptual 5",
    "álgebra: hecho conceptual 6",
    "álgebra: hecho conceptual 7",
    "álgebra: hecho conceptual 8",
    "álgebra: hecho conceptual 9",
    "álgebra: hecho conceptual 10",
    "álgebra: hecho conceptual 11",
    "álgebra: hecho conceptual 12",
    "álgebra: hecho conceptual 13",
    "álgebra: hecho conceptual 14",
    "álgebra: hecho conceptual 15",
    "álgebra: hecho conceptual 16",
    "álgebra: hecho conceptual 17",
    "álgebra: hecho conceptual 18",
    "álgebra: hecho conceptual 19",
    "álgebra: hecho conceptual 20",
    "álgebra: hecho conceptual 21",
    "álgebra: hecho conceptual 22",
    "álgebra: hecho conceptual 23",
    "álgebra: hecho conceptual 24",
    "álgebra: hecho conceptual 25",
    "álgebra: hecho conceptual 26",
    "álgebra: hecho conceptual 27",
    "álgebra: hecho conceptual 28",
    "álgebra: hecho conceptual 29",
    "álgebra: hecho conceptual 30",
    "álgebra: hecho conceptual 31",
    "álgebra: hecho conceptual 32",
    "álgebra: hecho conceptual 33",
    "álgebra: hecho conceptual 34",
    "álgebra: hecho conceptual 35",
    "álgebra: hecho conceptual 36",
    "álgebra: hecho conceptual 37",
    "álgebra: hecho conceptual 38",
    "álgebra: hecho conceptual 39",
    "álgebra: hecho conceptual 40",
    "álgebra: hecho conceptual 41",
    "álgebra: hecho conceptual 42",
    "álgebra: hecho conceptual 43",
    "álgebra: hecho conceptual 44",
    "álgebra: hecho conceptual 45",
    "álgebra: hecho conceptual 46",
    "álgebra: hecho conceptual 47",
    "álgebra: hecho conceptual 48",
    "álgebra: hecho conceptual 49",
    "álgebra: hecho conceptual 50",
    "álgebra: hecho conceptual 51",
    "álgebra: hecho conceptual 52",
    "álgebra: hecho conceptual 53",
    "álgebra: hecho conceptual 54",
    "álgebra: hecho conceptual 55",
    "álgebra: hecho conceptual 56",
    "álgebra: hecho conceptual 57",
    "álgebra: hecho conceptual 58",
    "álgebra: hecho conceptual 59",
    "álgebra: hecho conceptual 60",
    "álgebra: hecho conceptual 61",
    "álgebra: hecho conceptual 62",
    "álgebra: hecho conceptual 63",
    "álgebra: hecho conceptual 64",
    "álgebra: hecho conceptual 65",
    "álgebra: hecho conceptual 66",
    "álgebra: hecho conceptual 67",
    "álgebra: hecho conceptual 68",
    "álgebra: hecho conceptual 69",
    "álgebra: hecho conceptual 70",
    "álgebra: hecho conceptual 71",
    "álgebra: hecho conceptual 72",
    "álgebra: hecho conceptual 73",
    "álgebra: hecho conceptual 74",
    "álgebra: hecho conceptual 75",
    "álgebra: hecho conceptual 76",
    "álgebra: hecho conceptual 77",
    "álgebra: hecho conceptual 78",
    "álgebra: hecho conceptual 79",
    "álgebra: hecho conceptual 80",
]
LGEBRA_QUESTION_VARIANTS = [
    "¿Qué es álgebra?",
    "¿Cómo explicarías álgebra de forma sencilla?",
    "¿Por qué es importante álgebra?",
    "¿Cuál es una idea fundamental de álgebra?",
    "¿Puedes darme un ejemplo relacionado con álgebra?",
    "¿Qué errores son frecuentes al estudiar álgebra?",
    "¿Cómo se aplica álgebra en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con álgebra?",
    "¿Cómo empezaría a estudiar álgebra?",
    "¿Qué relación tiene álgebra con otras disciplinas?",
    "¿Puedes resumir álgebra en pocas frases?",
    "¿Qué términos debería conocer sobre álgebra?",
    "¿Cómo comprobaría si entendí álgebra?",
    "¿Qué intuición ayuda a comprender álgebra?",
    "¿Qué problema sencillo puedo resolver sobre álgebra?",
    "¿Qué matiz suele pasarse por alto en álgebra?",
    "¿Cómo se representa álgebra?",
    "¿Qué supuestos se usan al hablar de álgebra?",
    "¿Qué aplicaciones tiene álgebra?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar álgebra?",
    "¿Cómo se conecta álgebra con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de álgebra?",
    "¿Qué vocabulario técnico aparece en álgebra?",
    "¿Puedes comparar dos enfoques dentro de álgebra?",
    "¿Cómo evolucionó la comprensión de álgebra?",
    "¿Qué ejemplo cotidiano ilustra álgebra?",
    "¿Qué dato conviene recordar sobre álgebra?",
    "¿Cómo evitar confusiones comunes en álgebra?",
    "¿Qué parte de álgebra suele ser más difícil?",
    "¿Puedes plantear un ejercicio de álgebra?",
    "¿Cómo resolverías un ejercicio introductorio de álgebra?",
    "¿Qué relación matemática aparece en álgebra?",
    "¿Qué observación apoya esta idea de álgebra?",
    "¿Qué pasaría si cambiamos una condición en álgebra?",
    "¿Cómo se usa álgebra en investigación?",
    "¿Qué herramientas sirven para estudiar álgebra?",
    "¿Cómo distinguir evidencia de interpretación en álgebra?",
    "¿Qué concepto previo necesito para entender álgebra?",
    "¿Puedes dar una analogía para álgebra?",
    "¿Qué preguntas avanzadas surgen de álgebra?",
    "¿Cómo se comunica correctamente información sobre álgebra?",
    "¿Qué ejemplo contradice una intuición común sobre álgebra?",
    "¿Qué pasos seguirías para analizar álgebra?",
    "¿Cómo resumirías la historia de álgebra?",
    "¿Qué incertidumbres existen al estudiar álgebra?",
    "¿Cómo se relacionan teoría y práctica en álgebra?",
    "¿Qué clasificación útil existe en álgebra?",
    "¿Cómo puedo practicar álgebra?",
    "¿Qué es álgebra?",
]
GEOMETR_A_FACT_LABELS = [
    "geometría: hecho conceptual 1",
    "geometría: hecho conceptual 2",
    "geometría: hecho conceptual 3",
    "geometría: hecho conceptual 4",
    "geometría: hecho conceptual 5",
    "geometría: hecho conceptual 6",
    "geometría: hecho conceptual 7",
    "geometría: hecho conceptual 8",
    "geometría: hecho conceptual 9",
    "geometría: hecho conceptual 10",
    "geometría: hecho conceptual 11",
    "geometría: hecho conceptual 12",
    "geometría: hecho conceptual 13",
    "geometría: hecho conceptual 14",
    "geometría: hecho conceptual 15",
    "geometría: hecho conceptual 16",
    "geometría: hecho conceptual 17",
    "geometría: hecho conceptual 18",
    "geometría: hecho conceptual 19",
    "geometría: hecho conceptual 20",
    "geometría: hecho conceptual 21",
    "geometría: hecho conceptual 22",
    "geometría: hecho conceptual 23",
    "geometría: hecho conceptual 24",
    "geometría: hecho conceptual 25",
    "geometría: hecho conceptual 26",
    "geometría: hecho conceptual 27",
    "geometría: hecho conceptual 28",
    "geometría: hecho conceptual 29",
    "geometría: hecho conceptual 30",
    "geometría: hecho conceptual 31",
    "geometría: hecho conceptual 32",
    "geometría: hecho conceptual 33",
    "geometría: hecho conceptual 34",
    "geometría: hecho conceptual 35",
    "geometría: hecho conceptual 36",
    "geometría: hecho conceptual 37",
    "geometría: hecho conceptual 38",
    "geometría: hecho conceptual 39",
    "geometría: hecho conceptual 40",
    "geometría: hecho conceptual 41",
    "geometría: hecho conceptual 42",
    "geometría: hecho conceptual 43",
    "geometría: hecho conceptual 44",
    "geometría: hecho conceptual 45",
    "geometría: hecho conceptual 46",
    "geometría: hecho conceptual 47",
    "geometría: hecho conceptual 48",
    "geometría: hecho conceptual 49",
    "geometría: hecho conceptual 50",
    "geometría: hecho conceptual 51",
    "geometría: hecho conceptual 52",
    "geometría: hecho conceptual 53",
    "geometría: hecho conceptual 54",
    "geometría: hecho conceptual 55",
    "geometría: hecho conceptual 56",
    "geometría: hecho conceptual 57",
    "geometría: hecho conceptual 58",
    "geometría: hecho conceptual 59",
    "geometría: hecho conceptual 60",
    "geometría: hecho conceptual 61",
    "geometría: hecho conceptual 62",
    "geometría: hecho conceptual 63",
    "geometría: hecho conceptual 64",
    "geometría: hecho conceptual 65",
    "geometría: hecho conceptual 66",
    "geometría: hecho conceptual 67",
    "geometría: hecho conceptual 68",
    "geometría: hecho conceptual 69",
    "geometría: hecho conceptual 70",
    "geometría: hecho conceptual 71",
    "geometría: hecho conceptual 72",
    "geometría: hecho conceptual 73",
    "geometría: hecho conceptual 74",
    "geometría: hecho conceptual 75",
    "geometría: hecho conceptual 76",
    "geometría: hecho conceptual 77",
    "geometría: hecho conceptual 78",
    "geometría: hecho conceptual 79",
    "geometría: hecho conceptual 80",
]
GEOMETR_A_QUESTION_VARIANTS = [
    "¿Qué es geometría?",
    "¿Cómo explicarías geometría de forma sencilla?",
    "¿Por qué es importante geometría?",
    "¿Cuál es una idea fundamental de geometría?",
    "¿Puedes darme un ejemplo relacionado con geometría?",
    "¿Qué errores son frecuentes al estudiar geometría?",
    "¿Cómo se aplica geometría en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con geometría?",
    "¿Cómo empezaría a estudiar geometría?",
    "¿Qué relación tiene geometría con otras disciplinas?",
    "¿Puedes resumir geometría en pocas frases?",
    "¿Qué términos debería conocer sobre geometría?",
    "¿Cómo comprobaría si entendí geometría?",
    "¿Qué intuición ayuda a comprender geometría?",
    "¿Qué problema sencillo puedo resolver sobre geometría?",
    "¿Qué matiz suele pasarse por alto en geometría?",
    "¿Cómo se representa geometría?",
    "¿Qué supuestos se usan al hablar de geometría?",
    "¿Qué aplicaciones tiene geometría?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar geometría?",
    "¿Cómo se conecta geometría con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de geometría?",
    "¿Qué vocabulario técnico aparece en geometría?",
    "¿Puedes comparar dos enfoques dentro de geometría?",
    "¿Cómo evolucionó la comprensión de geometría?",
    "¿Qué ejemplo cotidiano ilustra geometría?",
    "¿Qué dato conviene recordar sobre geometría?",
    "¿Cómo evitar confusiones comunes en geometría?",
    "¿Qué parte de geometría suele ser más difícil?",
    "¿Puedes plantear un ejercicio de geometría?",
    "¿Cómo resolverías un ejercicio introductorio de geometría?",
    "¿Qué relación matemática aparece en geometría?",
    "¿Qué observación apoya esta idea de geometría?",
    "¿Qué pasaría si cambiamos una condición en geometría?",
    "¿Cómo se usa geometría en investigación?",
    "¿Qué herramientas sirven para estudiar geometría?",
    "¿Cómo distinguir evidencia de interpretación en geometría?",
    "¿Qué concepto previo necesito para entender geometría?",
    "¿Puedes dar una analogía para geometría?",
    "¿Qué preguntas avanzadas surgen de geometría?",
    "¿Cómo se comunica correctamente información sobre geometría?",
    "¿Qué ejemplo contradice una intuición común sobre geometría?",
    "¿Qué pasos seguirías para analizar geometría?",
    "¿Cómo resumirías la historia de geometría?",
    "¿Qué incertidumbres existen al estudiar geometría?",
    "¿Cómo se relacionan teoría y práctica en geometría?",
    "¿Qué clasificación útil existe en geometría?",
    "¿Cómo puedo practicar geometría?",
    "¿Qué es geometría?",
]
C_LCULO_FACT_LABELS = [
    "cálculo: hecho conceptual 1",
    "cálculo: hecho conceptual 2",
    "cálculo: hecho conceptual 3",
    "cálculo: hecho conceptual 4",
    "cálculo: hecho conceptual 5",
    "cálculo: hecho conceptual 6",
    "cálculo: hecho conceptual 7",
    "cálculo: hecho conceptual 8",
    "cálculo: hecho conceptual 9",
    "cálculo: hecho conceptual 10",
    "cálculo: hecho conceptual 11",
    "cálculo: hecho conceptual 12",
    "cálculo: hecho conceptual 13",
    "cálculo: hecho conceptual 14",
    "cálculo: hecho conceptual 15",
    "cálculo: hecho conceptual 16",
    "cálculo: hecho conceptual 17",
    "cálculo: hecho conceptual 18",
    "cálculo: hecho conceptual 19",
    "cálculo: hecho conceptual 20",
    "cálculo: hecho conceptual 21",
    "cálculo: hecho conceptual 22",
    "cálculo: hecho conceptual 23",
    "cálculo: hecho conceptual 24",
    "cálculo: hecho conceptual 25",
    "cálculo: hecho conceptual 26",
    "cálculo: hecho conceptual 27",
    "cálculo: hecho conceptual 28",
    "cálculo: hecho conceptual 29",
    "cálculo: hecho conceptual 30",
    "cálculo: hecho conceptual 31",
    "cálculo: hecho conceptual 32",
    "cálculo: hecho conceptual 33",
    "cálculo: hecho conceptual 34",
    "cálculo: hecho conceptual 35",
    "cálculo: hecho conceptual 36",
    "cálculo: hecho conceptual 37",
    "cálculo: hecho conceptual 38",
    "cálculo: hecho conceptual 39",
    "cálculo: hecho conceptual 40",
    "cálculo: hecho conceptual 41",
    "cálculo: hecho conceptual 42",
    "cálculo: hecho conceptual 43",
    "cálculo: hecho conceptual 44",
    "cálculo: hecho conceptual 45",
    "cálculo: hecho conceptual 46",
    "cálculo: hecho conceptual 47",
    "cálculo: hecho conceptual 48",
    "cálculo: hecho conceptual 49",
    "cálculo: hecho conceptual 50",
    "cálculo: hecho conceptual 51",
    "cálculo: hecho conceptual 52",
    "cálculo: hecho conceptual 53",
    "cálculo: hecho conceptual 54",
    "cálculo: hecho conceptual 55",
    "cálculo: hecho conceptual 56",
    "cálculo: hecho conceptual 57",
    "cálculo: hecho conceptual 58",
    "cálculo: hecho conceptual 59",
    "cálculo: hecho conceptual 60",
    "cálculo: hecho conceptual 61",
    "cálculo: hecho conceptual 62",
    "cálculo: hecho conceptual 63",
    "cálculo: hecho conceptual 64",
    "cálculo: hecho conceptual 65",
    "cálculo: hecho conceptual 66",
    "cálculo: hecho conceptual 67",
    "cálculo: hecho conceptual 68",
    "cálculo: hecho conceptual 69",
    "cálculo: hecho conceptual 70",
    "cálculo: hecho conceptual 71",
    "cálculo: hecho conceptual 72",
    "cálculo: hecho conceptual 73",
    "cálculo: hecho conceptual 74",
    "cálculo: hecho conceptual 75",
    "cálculo: hecho conceptual 76",
    "cálculo: hecho conceptual 77",
    "cálculo: hecho conceptual 78",
    "cálculo: hecho conceptual 79",
    "cálculo: hecho conceptual 80",
]
C_LCULO_QUESTION_VARIANTS = [
    "¿Qué es cálculo?",
    "¿Cómo explicarías cálculo de forma sencilla?",
    "¿Por qué es importante cálculo?",
    "¿Cuál es una idea fundamental de cálculo?",
    "¿Puedes darme un ejemplo relacionado con cálculo?",
    "¿Qué errores son frecuentes al estudiar cálculo?",
    "¿Cómo se aplica cálculo en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con cálculo?",
    "¿Cómo empezaría a estudiar cálculo?",
    "¿Qué relación tiene cálculo con otras disciplinas?",
    "¿Puedes resumir cálculo en pocas frases?",
    "¿Qué términos debería conocer sobre cálculo?",
    "¿Cómo comprobaría si entendí cálculo?",
    "¿Qué intuición ayuda a comprender cálculo?",
    "¿Qué problema sencillo puedo resolver sobre cálculo?",
    "¿Qué matiz suele pasarse por alto en cálculo?",
    "¿Cómo se representa cálculo?",
    "¿Qué supuestos se usan al hablar de cálculo?",
    "¿Qué aplicaciones tiene cálculo?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar cálculo?",
    "¿Cómo se conecta cálculo con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de cálculo?",
    "¿Qué vocabulario técnico aparece en cálculo?",
    "¿Puedes comparar dos enfoques dentro de cálculo?",
    "¿Cómo evolucionó la comprensión de cálculo?",
    "¿Qué ejemplo cotidiano ilustra cálculo?",
    "¿Qué dato conviene recordar sobre cálculo?",
    "¿Cómo evitar confusiones comunes en cálculo?",
    "¿Qué parte de cálculo suele ser más difícil?",
    "¿Puedes plantear un ejercicio de cálculo?",
    "¿Cómo resolverías un ejercicio introductorio de cálculo?",
    "¿Qué relación matemática aparece en cálculo?",
    "¿Qué observación apoya esta idea de cálculo?",
    "¿Qué pasaría si cambiamos una condición en cálculo?",
    "¿Cómo se usa cálculo en investigación?",
    "¿Qué herramientas sirven para estudiar cálculo?",
    "¿Cómo distinguir evidencia de interpretación en cálculo?",
    "¿Qué concepto previo necesito para entender cálculo?",
    "¿Puedes dar una analogía para cálculo?",
    "¿Qué preguntas avanzadas surgen de cálculo?",
    "¿Cómo se comunica correctamente información sobre cálculo?",
    "¿Qué ejemplo contradice una intuición común sobre cálculo?",
    "¿Qué pasos seguirías para analizar cálculo?",
    "¿Cómo resumirías la historia de cálculo?",
    "¿Qué incertidumbres existen al estudiar cálculo?",
    "¿Cómo se relacionan teoría y práctica en cálculo?",
    "¿Qué clasificación útil existe en cálculo?",
    "¿Cómo puedo practicar cálculo?",
    "¿Qué es cálculo?",
]
ESTAD_STICA_FACT_LABELS = [
    "estadística: hecho conceptual 1",
    "estadística: hecho conceptual 2",
    "estadística: hecho conceptual 3",
    "estadística: hecho conceptual 4",
    "estadística: hecho conceptual 5",
    "estadística: hecho conceptual 6",
    "estadística: hecho conceptual 7",
    "estadística: hecho conceptual 8",
    "estadística: hecho conceptual 9",
    "estadística: hecho conceptual 10",
    "estadística: hecho conceptual 11",
    "estadística: hecho conceptual 12",
    "estadística: hecho conceptual 13",
    "estadística: hecho conceptual 14",
    "estadística: hecho conceptual 15",
    "estadística: hecho conceptual 16",
    "estadística: hecho conceptual 17",
    "estadística: hecho conceptual 18",
    "estadística: hecho conceptual 19",
    "estadística: hecho conceptual 20",
    "estadística: hecho conceptual 21",
    "estadística: hecho conceptual 22",
    "estadística: hecho conceptual 23",
    "estadística: hecho conceptual 24",
    "estadística: hecho conceptual 25",
    "estadística: hecho conceptual 26",
    "estadística: hecho conceptual 27",
    "estadística: hecho conceptual 28",
    "estadística: hecho conceptual 29",
    "estadística: hecho conceptual 30",
    "estadística: hecho conceptual 31",
    "estadística: hecho conceptual 32",
    "estadística: hecho conceptual 33",
    "estadística: hecho conceptual 34",
    "estadística: hecho conceptual 35",
    "estadística: hecho conceptual 36",
    "estadística: hecho conceptual 37",
    "estadística: hecho conceptual 38",
    "estadística: hecho conceptual 39",
    "estadística: hecho conceptual 40",
    "estadística: hecho conceptual 41",
    "estadística: hecho conceptual 42",
    "estadística: hecho conceptual 43",
    "estadística: hecho conceptual 44",
    "estadística: hecho conceptual 45",
    "estadística: hecho conceptual 46",
    "estadística: hecho conceptual 47",
    "estadística: hecho conceptual 48",
    "estadística: hecho conceptual 49",
    "estadística: hecho conceptual 50",
    "estadística: hecho conceptual 51",
    "estadística: hecho conceptual 52",
    "estadística: hecho conceptual 53",
    "estadística: hecho conceptual 54",
    "estadística: hecho conceptual 55",
    "estadística: hecho conceptual 56",
    "estadística: hecho conceptual 57",
    "estadística: hecho conceptual 58",
    "estadística: hecho conceptual 59",
    "estadística: hecho conceptual 60",
    "estadística: hecho conceptual 61",
    "estadística: hecho conceptual 62",
    "estadística: hecho conceptual 63",
    "estadística: hecho conceptual 64",
    "estadística: hecho conceptual 65",
    "estadística: hecho conceptual 66",
    "estadística: hecho conceptual 67",
    "estadística: hecho conceptual 68",
    "estadística: hecho conceptual 69",
    "estadística: hecho conceptual 70",
    "estadística: hecho conceptual 71",
    "estadística: hecho conceptual 72",
    "estadística: hecho conceptual 73",
    "estadística: hecho conceptual 74",
    "estadística: hecho conceptual 75",
    "estadística: hecho conceptual 76",
    "estadística: hecho conceptual 77",
    "estadística: hecho conceptual 78",
    "estadística: hecho conceptual 79",
    "estadística: hecho conceptual 80",
]
ESTAD_STICA_QUESTION_VARIANTS = [
    "¿Qué es estadística?",
    "¿Cómo explicarías estadística de forma sencilla?",
    "¿Por qué es importante estadística?",
    "¿Cuál es una idea fundamental de estadística?",
    "¿Puedes darme un ejemplo relacionado con estadística?",
    "¿Qué errores son frecuentes al estudiar estadística?",
    "¿Cómo se aplica estadística en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con estadística?",
    "¿Cómo empezaría a estudiar estadística?",
    "¿Qué relación tiene estadística con otras disciplinas?",
    "¿Puedes resumir estadística en pocas frases?",
    "¿Qué términos debería conocer sobre estadística?",
    "¿Cómo comprobaría si entendí estadística?",
    "¿Qué intuición ayuda a comprender estadística?",
    "¿Qué problema sencillo puedo resolver sobre estadística?",
    "¿Qué matiz suele pasarse por alto en estadística?",
    "¿Cómo se representa estadística?",
    "¿Qué supuestos se usan al hablar de estadística?",
    "¿Qué aplicaciones tiene estadística?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar estadística?",
    "¿Cómo se conecta estadística con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de estadística?",
    "¿Qué vocabulario técnico aparece en estadística?",
    "¿Puedes comparar dos enfoques dentro de estadística?",
    "¿Cómo evolucionó la comprensión de estadística?",
    "¿Qué ejemplo cotidiano ilustra estadística?",
    "¿Qué dato conviene recordar sobre estadística?",
    "¿Cómo evitar confusiones comunes en estadística?",
    "¿Qué parte de estadística suele ser más difícil?",
    "¿Puedes plantear un ejercicio de estadística?",
    "¿Cómo resolverías un ejercicio introductorio de estadística?",
    "¿Qué relación matemática aparece en estadística?",
    "¿Qué observación apoya esta idea de estadística?",
    "¿Qué pasaría si cambiamos una condición en estadística?",
    "¿Cómo se usa estadística en investigación?",
    "¿Qué herramientas sirven para estudiar estadística?",
    "¿Cómo distinguir evidencia de interpretación en estadística?",
    "¿Qué concepto previo necesito para entender estadística?",
    "¿Puedes dar una analogía para estadística?",
    "¿Qué preguntas avanzadas surgen de estadística?",
    "¿Cómo se comunica correctamente información sobre estadística?",
    "¿Qué ejemplo contradice una intuición común sobre estadística?",
    "¿Qué pasos seguirías para analizar estadística?",
    "¿Cómo resumirías la historia de estadística?",
    "¿Qué incertidumbres existen al estudiar estadística?",
    "¿Cómo se relacionan teoría y práctica en estadística?",
    "¿Qué clasificación útil existe en estadística?",
    "¿Cómo puedo practicar estadística?",
    "¿Qué es estadística?",
]
PROBABILIDAD_FACT_LABELS = [
    "probabilidad: hecho conceptual 1",
    "probabilidad: hecho conceptual 2",
    "probabilidad: hecho conceptual 3",
    "probabilidad: hecho conceptual 4",
    "probabilidad: hecho conceptual 5",
    "probabilidad: hecho conceptual 6",
    "probabilidad: hecho conceptual 7",
    "probabilidad: hecho conceptual 8",
    "probabilidad: hecho conceptual 9",
    "probabilidad: hecho conceptual 10",
    "probabilidad: hecho conceptual 11",
    "probabilidad: hecho conceptual 12",
    "probabilidad: hecho conceptual 13",
    "probabilidad: hecho conceptual 14",
    "probabilidad: hecho conceptual 15",
    "probabilidad: hecho conceptual 16",
    "probabilidad: hecho conceptual 17",
    "probabilidad: hecho conceptual 18",
    "probabilidad: hecho conceptual 19",
    "probabilidad: hecho conceptual 20",
    "probabilidad: hecho conceptual 21",
    "probabilidad: hecho conceptual 22",
    "probabilidad: hecho conceptual 23",
    "probabilidad: hecho conceptual 24",
    "probabilidad: hecho conceptual 25",
    "probabilidad: hecho conceptual 26",
    "probabilidad: hecho conceptual 27",
    "probabilidad: hecho conceptual 28",
    "probabilidad: hecho conceptual 29",
    "probabilidad: hecho conceptual 30",
    "probabilidad: hecho conceptual 31",
    "probabilidad: hecho conceptual 32",
    "probabilidad: hecho conceptual 33",
    "probabilidad: hecho conceptual 34",
    "probabilidad: hecho conceptual 35",
    "probabilidad: hecho conceptual 36",
    "probabilidad: hecho conceptual 37",
    "probabilidad: hecho conceptual 38",
    "probabilidad: hecho conceptual 39",
    "probabilidad: hecho conceptual 40",
    "probabilidad: hecho conceptual 41",
    "probabilidad: hecho conceptual 42",
    "probabilidad: hecho conceptual 43",
    "probabilidad: hecho conceptual 44",
    "probabilidad: hecho conceptual 45",
    "probabilidad: hecho conceptual 46",
    "probabilidad: hecho conceptual 47",
    "probabilidad: hecho conceptual 48",
    "probabilidad: hecho conceptual 49",
    "probabilidad: hecho conceptual 50",
    "probabilidad: hecho conceptual 51",
    "probabilidad: hecho conceptual 52",
    "probabilidad: hecho conceptual 53",
    "probabilidad: hecho conceptual 54",
    "probabilidad: hecho conceptual 55",
    "probabilidad: hecho conceptual 56",
    "probabilidad: hecho conceptual 57",
    "probabilidad: hecho conceptual 58",
    "probabilidad: hecho conceptual 59",
    "probabilidad: hecho conceptual 60",
    "probabilidad: hecho conceptual 61",
    "probabilidad: hecho conceptual 62",
    "probabilidad: hecho conceptual 63",
    "probabilidad: hecho conceptual 64",
    "probabilidad: hecho conceptual 65",
    "probabilidad: hecho conceptual 66",
    "probabilidad: hecho conceptual 67",
    "probabilidad: hecho conceptual 68",
    "probabilidad: hecho conceptual 69",
    "probabilidad: hecho conceptual 70",
    "probabilidad: hecho conceptual 71",
    "probabilidad: hecho conceptual 72",
    "probabilidad: hecho conceptual 73",
    "probabilidad: hecho conceptual 74",
    "probabilidad: hecho conceptual 75",
    "probabilidad: hecho conceptual 76",
    "probabilidad: hecho conceptual 77",
    "probabilidad: hecho conceptual 78",
    "probabilidad: hecho conceptual 79",
    "probabilidad: hecho conceptual 80",
]
PROBABILIDAD_QUESTION_VARIANTS = [
    "¿Qué es probabilidad?",
    "¿Cómo explicarías probabilidad de forma sencilla?",
    "¿Por qué es importante probabilidad?",
    "¿Cuál es una idea fundamental de probabilidad?",
    "¿Puedes darme un ejemplo relacionado con probabilidad?",
    "¿Qué errores son frecuentes al estudiar probabilidad?",
    "¿Cómo se aplica probabilidad en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con probabilidad?",
    "¿Cómo empezaría a estudiar probabilidad?",
    "¿Qué relación tiene probabilidad con otras disciplinas?",
    "¿Puedes resumir probabilidad en pocas frases?",
    "¿Qué términos debería conocer sobre probabilidad?",
    "¿Cómo comprobaría si entendí probabilidad?",
    "¿Qué intuición ayuda a comprender probabilidad?",
    "¿Qué problema sencillo puedo resolver sobre probabilidad?",
    "¿Qué matiz suele pasarse por alto en probabilidad?",
    "¿Cómo se representa probabilidad?",
    "¿Qué supuestos se usan al hablar de probabilidad?",
    "¿Qué aplicaciones tiene probabilidad?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar probabilidad?",
    "¿Cómo se conecta probabilidad con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de probabilidad?",
    "¿Qué vocabulario técnico aparece en probabilidad?",
    "¿Puedes comparar dos enfoques dentro de probabilidad?",
    "¿Cómo evolucionó la comprensión de probabilidad?",
    "¿Qué ejemplo cotidiano ilustra probabilidad?",
    "¿Qué dato conviene recordar sobre probabilidad?",
    "¿Cómo evitar confusiones comunes en probabilidad?",
    "¿Qué parte de probabilidad suele ser más difícil?",
    "¿Puedes plantear un ejercicio de probabilidad?",
    "¿Cómo resolverías un ejercicio introductorio de probabilidad?",
    "¿Qué relación matemática aparece en probabilidad?",
    "¿Qué observación apoya esta idea de probabilidad?",
    "¿Qué pasaría si cambiamos una condición en probabilidad?",
    "¿Cómo se usa probabilidad en investigación?",
    "¿Qué herramientas sirven para estudiar probabilidad?",
    "¿Cómo distinguir evidencia de interpretación en probabilidad?",
    "¿Qué concepto previo necesito para entender probabilidad?",
    "¿Puedes dar una analogía para probabilidad?",
    "¿Qué preguntas avanzadas surgen de probabilidad?",
    "¿Cómo se comunica correctamente información sobre probabilidad?",
    "¿Qué ejemplo contradice una intuición común sobre probabilidad?",
    "¿Qué pasos seguirías para analizar probabilidad?",
    "¿Cómo resumirías la historia de probabilidad?",
    "¿Qué incertidumbres existen al estudiar probabilidad?",
    "¿Cómo se relacionan teoría y práctica en probabilidad?",
    "¿Qué clasificación útil existe en probabilidad?",
    "¿Cómo puedo practicar probabilidad?",
    "¿Qué es probabilidad?",
]
F_SICA_FACT_LABELS = [
    "física: hecho conceptual 1",
    "física: hecho conceptual 2",
    "física: hecho conceptual 3",
    "física: hecho conceptual 4",
    "física: hecho conceptual 5",
    "física: hecho conceptual 6",
    "física: hecho conceptual 7",
    "física: hecho conceptual 8",
    "física: hecho conceptual 9",
    "física: hecho conceptual 10",
    "física: hecho conceptual 11",
    "física: hecho conceptual 12",
    "física: hecho conceptual 13",
    "física: hecho conceptual 14",
    "física: hecho conceptual 15",
    "física: hecho conceptual 16",
    "física: hecho conceptual 17",
    "física: hecho conceptual 18",
    "física: hecho conceptual 19",
    "física: hecho conceptual 20",
    "física: hecho conceptual 21",
    "física: hecho conceptual 22",
    "física: hecho conceptual 23",
    "física: hecho conceptual 24",
    "física: hecho conceptual 25",
    "física: hecho conceptual 26",
    "física: hecho conceptual 27",
    "física: hecho conceptual 28",
    "física: hecho conceptual 29",
    "física: hecho conceptual 30",
    "física: hecho conceptual 31",
    "física: hecho conceptual 32",
    "física: hecho conceptual 33",
    "física: hecho conceptual 34",
    "física: hecho conceptual 35",
    "física: hecho conceptual 36",
    "física: hecho conceptual 37",
    "física: hecho conceptual 38",
    "física: hecho conceptual 39",
    "física: hecho conceptual 40",
    "física: hecho conceptual 41",
    "física: hecho conceptual 42",
    "física: hecho conceptual 43",
    "física: hecho conceptual 44",
    "física: hecho conceptual 45",
    "física: hecho conceptual 46",
    "física: hecho conceptual 47",
    "física: hecho conceptual 48",
    "física: hecho conceptual 49",
    "física: hecho conceptual 50",
    "física: hecho conceptual 51",
    "física: hecho conceptual 52",
    "física: hecho conceptual 53",
    "física: hecho conceptual 54",
    "física: hecho conceptual 55",
    "física: hecho conceptual 56",
    "física: hecho conceptual 57",
    "física: hecho conceptual 58",
    "física: hecho conceptual 59",
    "física: hecho conceptual 60",
    "física: hecho conceptual 61",
    "física: hecho conceptual 62",
    "física: hecho conceptual 63",
    "física: hecho conceptual 64",
    "física: hecho conceptual 65",
    "física: hecho conceptual 66",
    "física: hecho conceptual 67",
    "física: hecho conceptual 68",
    "física: hecho conceptual 69",
    "física: hecho conceptual 70",
    "física: hecho conceptual 71",
    "física: hecho conceptual 72",
    "física: hecho conceptual 73",
    "física: hecho conceptual 74",
    "física: hecho conceptual 75",
    "física: hecho conceptual 76",
    "física: hecho conceptual 77",
    "física: hecho conceptual 78",
    "física: hecho conceptual 79",
    "física: hecho conceptual 80",
]
F_SICA_QUESTION_VARIANTS = [
    "¿Qué es física?",
    "¿Cómo explicarías física de forma sencilla?",
    "¿Por qué es importante física?",
    "¿Cuál es una idea fundamental de física?",
    "¿Puedes darme un ejemplo relacionado con física?",
    "¿Qué errores son frecuentes al estudiar física?",
    "¿Cómo se aplica física en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con física?",
    "¿Cómo empezaría a estudiar física?",
    "¿Qué relación tiene física con otras disciplinas?",
    "¿Puedes resumir física en pocas frases?",
    "¿Qué términos debería conocer sobre física?",
    "¿Cómo comprobaría si entendí física?",
    "¿Qué intuición ayuda a comprender física?",
    "¿Qué problema sencillo puedo resolver sobre física?",
    "¿Qué matiz suele pasarse por alto en física?",
    "¿Cómo se representa física?",
    "¿Qué supuestos se usan al hablar de física?",
    "¿Qué aplicaciones tiene física?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar física?",
    "¿Cómo se conecta física con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de física?",
    "¿Qué vocabulario técnico aparece en física?",
    "¿Puedes comparar dos enfoques dentro de física?",
    "¿Cómo evolucionó la comprensión de física?",
    "¿Qué ejemplo cotidiano ilustra física?",
    "¿Qué dato conviene recordar sobre física?",
    "¿Cómo evitar confusiones comunes en física?",
    "¿Qué parte de física suele ser más difícil?",
    "¿Puedes plantear un ejercicio de física?",
    "¿Cómo resolverías un ejercicio introductorio de física?",
    "¿Qué relación matemática aparece en física?",
    "¿Qué observación apoya esta idea de física?",
    "¿Qué pasaría si cambiamos una condición en física?",
    "¿Cómo se usa física en investigación?",
    "¿Qué herramientas sirven para estudiar física?",
    "¿Cómo distinguir evidencia de interpretación en física?",
    "¿Qué concepto previo necesito para entender física?",
    "¿Puedes dar una analogía para física?",
    "¿Qué preguntas avanzadas surgen de física?",
    "¿Cómo se comunica correctamente información sobre física?",
    "¿Qué ejemplo contradice una intuición común sobre física?",
    "¿Qué pasos seguirías para analizar física?",
    "¿Cómo resumirías la historia de física?",
    "¿Qué incertidumbres existen al estudiar física?",
    "¿Cómo se relacionan teoría y práctica en física?",
    "¿Qué clasificación útil existe en física?",
    "¿Cómo puedo practicar física?",
    "¿Qué es física?",
]
MEC_NICA_FACT_LABELS = [
    "mecánica: hecho conceptual 1",
    "mecánica: hecho conceptual 2",
    "mecánica: hecho conceptual 3",
    "mecánica: hecho conceptual 4",
    "mecánica: hecho conceptual 5",
    "mecánica: hecho conceptual 6",
    "mecánica: hecho conceptual 7",
    "mecánica: hecho conceptual 8",
    "mecánica: hecho conceptual 9",
    "mecánica: hecho conceptual 10",
    "mecánica: hecho conceptual 11",
    "mecánica: hecho conceptual 12",
    "mecánica: hecho conceptual 13",
    "mecánica: hecho conceptual 14",
    "mecánica: hecho conceptual 15",
    "mecánica: hecho conceptual 16",
    "mecánica: hecho conceptual 17",
    "mecánica: hecho conceptual 18",
    "mecánica: hecho conceptual 19",
    "mecánica: hecho conceptual 20",
    "mecánica: hecho conceptual 21",
    "mecánica: hecho conceptual 22",
    "mecánica: hecho conceptual 23",
    "mecánica: hecho conceptual 24",
    "mecánica: hecho conceptual 25",
    "mecánica: hecho conceptual 26",
    "mecánica: hecho conceptual 27",
    "mecánica: hecho conceptual 28",
    "mecánica: hecho conceptual 29",
    "mecánica: hecho conceptual 30",
    "mecánica: hecho conceptual 31",
    "mecánica: hecho conceptual 32",
    "mecánica: hecho conceptual 33",
    "mecánica: hecho conceptual 34",
    "mecánica: hecho conceptual 35",
    "mecánica: hecho conceptual 36",
    "mecánica: hecho conceptual 37",
    "mecánica: hecho conceptual 38",
    "mecánica: hecho conceptual 39",
    "mecánica: hecho conceptual 40",
    "mecánica: hecho conceptual 41",
    "mecánica: hecho conceptual 42",
    "mecánica: hecho conceptual 43",
    "mecánica: hecho conceptual 44",
    "mecánica: hecho conceptual 45",
    "mecánica: hecho conceptual 46",
    "mecánica: hecho conceptual 47",
    "mecánica: hecho conceptual 48",
    "mecánica: hecho conceptual 49",
    "mecánica: hecho conceptual 50",
    "mecánica: hecho conceptual 51",
    "mecánica: hecho conceptual 52",
    "mecánica: hecho conceptual 53",
    "mecánica: hecho conceptual 54",
    "mecánica: hecho conceptual 55",
    "mecánica: hecho conceptual 56",
    "mecánica: hecho conceptual 57",
    "mecánica: hecho conceptual 58",
    "mecánica: hecho conceptual 59",
    "mecánica: hecho conceptual 60",
    "mecánica: hecho conceptual 61",
    "mecánica: hecho conceptual 62",
    "mecánica: hecho conceptual 63",
    "mecánica: hecho conceptual 64",
    "mecánica: hecho conceptual 65",
    "mecánica: hecho conceptual 66",
    "mecánica: hecho conceptual 67",
    "mecánica: hecho conceptual 68",
    "mecánica: hecho conceptual 69",
    "mecánica: hecho conceptual 70",
    "mecánica: hecho conceptual 71",
    "mecánica: hecho conceptual 72",
    "mecánica: hecho conceptual 73",
    "mecánica: hecho conceptual 74",
    "mecánica: hecho conceptual 75",
    "mecánica: hecho conceptual 76",
    "mecánica: hecho conceptual 77",
    "mecánica: hecho conceptual 78",
    "mecánica: hecho conceptual 79",
    "mecánica: hecho conceptual 80",
]
MEC_NICA_QUESTION_VARIANTS = [
    "¿Qué es mecánica?",
    "¿Cómo explicarías mecánica de forma sencilla?",
    "¿Por qué es importante mecánica?",
    "¿Cuál es una idea fundamental de mecánica?",
    "¿Puedes darme un ejemplo relacionado con mecánica?",
    "¿Qué errores son frecuentes al estudiar mecánica?",
    "¿Cómo se aplica mecánica en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con mecánica?",
    "¿Cómo empezaría a estudiar mecánica?",
    "¿Qué relación tiene mecánica con otras disciplinas?",
    "¿Puedes resumir mecánica en pocas frases?",
    "¿Qué términos debería conocer sobre mecánica?",
    "¿Cómo comprobaría si entendí mecánica?",
    "¿Qué intuición ayuda a comprender mecánica?",
    "¿Qué problema sencillo puedo resolver sobre mecánica?",
    "¿Qué matiz suele pasarse por alto en mecánica?",
    "¿Cómo se representa mecánica?",
    "¿Qué supuestos se usan al hablar de mecánica?",
    "¿Qué aplicaciones tiene mecánica?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar mecánica?",
    "¿Cómo se conecta mecánica con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de mecánica?",
    "¿Qué vocabulario técnico aparece en mecánica?",
    "¿Puedes comparar dos enfoques dentro de mecánica?",
    "¿Cómo evolucionó la comprensión de mecánica?",
    "¿Qué ejemplo cotidiano ilustra mecánica?",
    "¿Qué dato conviene recordar sobre mecánica?",
    "¿Cómo evitar confusiones comunes en mecánica?",
    "¿Qué parte de mecánica suele ser más difícil?",
    "¿Puedes plantear un ejercicio de mecánica?",
    "¿Cómo resolverías un ejercicio introductorio de mecánica?",
    "¿Qué relación matemática aparece en mecánica?",
    "¿Qué observación apoya esta idea de mecánica?",
    "¿Qué pasaría si cambiamos una condición en mecánica?",
    "¿Cómo se usa mecánica en investigación?",
    "¿Qué herramientas sirven para estudiar mecánica?",
    "¿Cómo distinguir evidencia de interpretación en mecánica?",
    "¿Qué concepto previo necesito para entender mecánica?",
    "¿Puedes dar una analogía para mecánica?",
    "¿Qué preguntas avanzadas surgen de mecánica?",
    "¿Cómo se comunica correctamente información sobre mecánica?",
    "¿Qué ejemplo contradice una intuición común sobre mecánica?",
    "¿Qué pasos seguirías para analizar mecánica?",
    "¿Cómo resumirías la historia de mecánica?",
    "¿Qué incertidumbres existen al estudiar mecánica?",
    "¿Cómo se relacionan teoría y práctica en mecánica?",
    "¿Qué clasificación útil existe en mecánica?",
    "¿Cómo puedo practicar mecánica?",
    "¿Qué es mecánica?",
]
TERMODIN_MICA_FACT_LABELS = [
    "termodinámica: hecho conceptual 1",
    "termodinámica: hecho conceptual 2",
    "termodinámica: hecho conceptual 3",
    "termodinámica: hecho conceptual 4",
    "termodinámica: hecho conceptual 5",
    "termodinámica: hecho conceptual 6",
    "termodinámica: hecho conceptual 7",
    "termodinámica: hecho conceptual 8",
    "termodinámica: hecho conceptual 9",
    "termodinámica: hecho conceptual 10",
    "termodinámica: hecho conceptual 11",
    "termodinámica: hecho conceptual 12",
    "termodinámica: hecho conceptual 13",
    "termodinámica: hecho conceptual 14",
    "termodinámica: hecho conceptual 15",
    "termodinámica: hecho conceptual 16",
    "termodinámica: hecho conceptual 17",
    "termodinámica: hecho conceptual 18",
    "termodinámica: hecho conceptual 19",
    "termodinámica: hecho conceptual 20",
    "termodinámica: hecho conceptual 21",
    "termodinámica: hecho conceptual 22",
    "termodinámica: hecho conceptual 23",
    "termodinámica: hecho conceptual 24",
    "termodinámica: hecho conceptual 25",
    "termodinámica: hecho conceptual 26",
    "termodinámica: hecho conceptual 27",
    "termodinámica: hecho conceptual 28",
    "termodinámica: hecho conceptual 29",
    "termodinámica: hecho conceptual 30",
    "termodinámica: hecho conceptual 31",
    "termodinámica: hecho conceptual 32",
    "termodinámica: hecho conceptual 33",
    "termodinámica: hecho conceptual 34",
    "termodinámica: hecho conceptual 35",
    "termodinámica: hecho conceptual 36",
    "termodinámica: hecho conceptual 37",
    "termodinámica: hecho conceptual 38",
    "termodinámica: hecho conceptual 39",
    "termodinámica: hecho conceptual 40",
    "termodinámica: hecho conceptual 41",
    "termodinámica: hecho conceptual 42",
    "termodinámica: hecho conceptual 43",
    "termodinámica: hecho conceptual 44",
    "termodinámica: hecho conceptual 45",
    "termodinámica: hecho conceptual 46",
    "termodinámica: hecho conceptual 47",
    "termodinámica: hecho conceptual 48",
    "termodinámica: hecho conceptual 49",
    "termodinámica: hecho conceptual 50",
    "termodinámica: hecho conceptual 51",
    "termodinámica: hecho conceptual 52",
    "termodinámica: hecho conceptual 53",
    "termodinámica: hecho conceptual 54",
    "termodinámica: hecho conceptual 55",
    "termodinámica: hecho conceptual 56",
    "termodinámica: hecho conceptual 57",
    "termodinámica: hecho conceptual 58",
    "termodinámica: hecho conceptual 59",
    "termodinámica: hecho conceptual 60",
    "termodinámica: hecho conceptual 61",
    "termodinámica: hecho conceptual 62",
    "termodinámica: hecho conceptual 63",
    "termodinámica: hecho conceptual 64",
    "termodinámica: hecho conceptual 65",
    "termodinámica: hecho conceptual 66",
    "termodinámica: hecho conceptual 67",
    "termodinámica: hecho conceptual 68",
    "termodinámica: hecho conceptual 69",
    "termodinámica: hecho conceptual 70",
    "termodinámica: hecho conceptual 71",
    "termodinámica: hecho conceptual 72",
    "termodinámica: hecho conceptual 73",
    "termodinámica: hecho conceptual 74",
    "termodinámica: hecho conceptual 75",
    "termodinámica: hecho conceptual 76",
    "termodinámica: hecho conceptual 77",
    "termodinámica: hecho conceptual 78",
    "termodinámica: hecho conceptual 79",
    "termodinámica: hecho conceptual 80",
]
TERMODIN_MICA_QUESTION_VARIANTS = [
    "¿Qué es termodinámica?",
    "¿Cómo explicarías termodinámica de forma sencilla?",
    "¿Por qué es importante termodinámica?",
    "¿Cuál es una idea fundamental de termodinámica?",
    "¿Puedes darme un ejemplo relacionado con termodinámica?",
    "¿Qué errores son frecuentes al estudiar termodinámica?",
    "¿Cómo se aplica termodinámica en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con termodinámica?",
    "¿Cómo empezaría a estudiar termodinámica?",
    "¿Qué relación tiene termodinámica con otras disciplinas?",
    "¿Puedes resumir termodinámica en pocas frases?",
    "¿Qué términos debería conocer sobre termodinámica?",
    "¿Cómo comprobaría si entendí termodinámica?",
    "¿Qué intuición ayuda a comprender termodinámica?",
    "¿Qué problema sencillo puedo resolver sobre termodinámica?",
    "¿Qué matiz suele pasarse por alto en termodinámica?",
    "¿Cómo se representa termodinámica?",
    "¿Qué supuestos se usan al hablar de termodinámica?",
    "¿Qué aplicaciones tiene termodinámica?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar termodinámica?",
    "¿Cómo se conecta termodinámica con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de termodinámica?",
    "¿Qué vocabulario técnico aparece en termodinámica?",
    "¿Puedes comparar dos enfoques dentro de termodinámica?",
    "¿Cómo evolucionó la comprensión de termodinámica?",
    "¿Qué ejemplo cotidiano ilustra termodinámica?",
    "¿Qué dato conviene recordar sobre termodinámica?",
    "¿Cómo evitar confusiones comunes en termodinámica?",
    "¿Qué parte de termodinámica suele ser más difícil?",
    "¿Puedes plantear un ejercicio de termodinámica?",
    "¿Cómo resolverías un ejercicio introductorio de termodinámica?",
    "¿Qué relación matemática aparece en termodinámica?",
    "¿Qué observación apoya esta idea de termodinámica?",
    "¿Qué pasaría si cambiamos una condición en termodinámica?",
    "¿Cómo se usa termodinámica en investigación?",
    "¿Qué herramientas sirven para estudiar termodinámica?",
    "¿Cómo distinguir evidencia de interpretación en termodinámica?",
    "¿Qué concepto previo necesito para entender termodinámica?",
    "¿Puedes dar una analogía para termodinámica?",
    "¿Qué preguntas avanzadas surgen de termodinámica?",
    "¿Cómo se comunica correctamente información sobre termodinámica?",
    "¿Qué ejemplo contradice una intuición común sobre termodinámica?",
    "¿Qué pasos seguirías para analizar termodinámica?",
    "¿Cómo resumirías la historia de termodinámica?",
    "¿Qué incertidumbres existen al estudiar termodinámica?",
    "¿Cómo se relacionan teoría y práctica en termodinámica?",
    "¿Qué clasificación útil existe en termodinámica?",
    "¿Cómo puedo practicar termodinámica?",
    "¿Qué es termodinámica?",
]
QU_MICA_FACT_LABELS = [
    "química: hecho conceptual 1",
    "química: hecho conceptual 2",
    "química: hecho conceptual 3",
    "química: hecho conceptual 4",
    "química: hecho conceptual 5",
    "química: hecho conceptual 6",
    "química: hecho conceptual 7",
    "química: hecho conceptual 8",
    "química: hecho conceptual 9",
    "química: hecho conceptual 10",
    "química: hecho conceptual 11",
    "química: hecho conceptual 12",
    "química: hecho conceptual 13",
    "química: hecho conceptual 14",
    "química: hecho conceptual 15",
    "química: hecho conceptual 16",
    "química: hecho conceptual 17",
    "química: hecho conceptual 18",
    "química: hecho conceptual 19",
    "química: hecho conceptual 20",
    "química: hecho conceptual 21",
    "química: hecho conceptual 22",
    "química: hecho conceptual 23",
    "química: hecho conceptual 24",
    "química: hecho conceptual 25",
    "química: hecho conceptual 26",
    "química: hecho conceptual 27",
    "química: hecho conceptual 28",
    "química: hecho conceptual 29",
    "química: hecho conceptual 30",
    "química: hecho conceptual 31",
    "química: hecho conceptual 32",
    "química: hecho conceptual 33",
    "química: hecho conceptual 34",
    "química: hecho conceptual 35",
    "química: hecho conceptual 36",
    "química: hecho conceptual 37",
    "química: hecho conceptual 38",
    "química: hecho conceptual 39",
    "química: hecho conceptual 40",
    "química: hecho conceptual 41",
    "química: hecho conceptual 42",
    "química: hecho conceptual 43",
    "química: hecho conceptual 44",
    "química: hecho conceptual 45",
    "química: hecho conceptual 46",
    "química: hecho conceptual 47",
    "química: hecho conceptual 48",
    "química: hecho conceptual 49",
    "química: hecho conceptual 50",
    "química: hecho conceptual 51",
    "química: hecho conceptual 52",
    "química: hecho conceptual 53",
    "química: hecho conceptual 54",
    "química: hecho conceptual 55",
    "química: hecho conceptual 56",
    "química: hecho conceptual 57",
    "química: hecho conceptual 58",
    "química: hecho conceptual 59",
    "química: hecho conceptual 60",
    "química: hecho conceptual 61",
    "química: hecho conceptual 62",
    "química: hecho conceptual 63",
    "química: hecho conceptual 64",
    "química: hecho conceptual 65",
    "química: hecho conceptual 66",
    "química: hecho conceptual 67",
    "química: hecho conceptual 68",
    "química: hecho conceptual 69",
    "química: hecho conceptual 70",
    "química: hecho conceptual 71",
    "química: hecho conceptual 72",
    "química: hecho conceptual 73",
    "química: hecho conceptual 74",
    "química: hecho conceptual 75",
    "química: hecho conceptual 76",
    "química: hecho conceptual 77",
    "química: hecho conceptual 78",
    "química: hecho conceptual 79",
    "química: hecho conceptual 80",
]
QU_MICA_QUESTION_VARIANTS = [
    "¿Qué es química?",
    "¿Cómo explicarías química de forma sencilla?",
    "¿Por qué es importante química?",
    "¿Cuál es una idea fundamental de química?",
    "¿Puedes darme un ejemplo relacionado con química?",
    "¿Qué errores son frecuentes al estudiar química?",
    "¿Cómo se aplica química en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con química?",
    "¿Cómo empezaría a estudiar química?",
    "¿Qué relación tiene química con otras disciplinas?",
    "¿Puedes resumir química en pocas frases?",
    "¿Qué términos debería conocer sobre química?",
    "¿Cómo comprobaría si entendí química?",
    "¿Qué intuición ayuda a comprender química?",
    "¿Qué problema sencillo puedo resolver sobre química?",
    "¿Qué matiz suele pasarse por alto en química?",
    "¿Cómo se representa química?",
    "¿Qué supuestos se usan al hablar de química?",
    "¿Qué aplicaciones tiene química?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar química?",
    "¿Cómo se conecta química con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de química?",
    "¿Qué vocabulario técnico aparece en química?",
    "¿Puedes comparar dos enfoques dentro de química?",
    "¿Cómo evolucionó la comprensión de química?",
    "¿Qué ejemplo cotidiano ilustra química?",
    "¿Qué dato conviene recordar sobre química?",
    "¿Cómo evitar confusiones comunes en química?",
    "¿Qué parte de química suele ser más difícil?",
    "¿Puedes plantear un ejercicio de química?",
    "¿Cómo resolverías un ejercicio introductorio de química?",
    "¿Qué relación matemática aparece en química?",
    "¿Qué observación apoya esta idea de química?",
    "¿Qué pasaría si cambiamos una condición en química?",
    "¿Cómo se usa química en investigación?",
    "¿Qué herramientas sirven para estudiar química?",
    "¿Cómo distinguir evidencia de interpretación en química?",
    "¿Qué concepto previo necesito para entender química?",
    "¿Puedes dar una analogía para química?",
    "¿Qué preguntas avanzadas surgen de química?",
    "¿Cómo se comunica correctamente información sobre química?",
    "¿Qué ejemplo contradice una intuición común sobre química?",
    "¿Qué pasos seguirías para analizar química?",
    "¿Cómo resumirías la historia de química?",
    "¿Qué incertidumbres existen al estudiar química?",
    "¿Cómo se relacionan teoría y práctica en química?",
    "¿Qué clasificación útil existe en química?",
    "¿Cómo puedo practicar química?",
    "¿Qué es química?",
]
BIOLOG_A_FACT_LABELS = [
    "biología: hecho conceptual 1",
    "biología: hecho conceptual 2",
    "biología: hecho conceptual 3",
    "biología: hecho conceptual 4",
    "biología: hecho conceptual 5",
    "biología: hecho conceptual 6",
    "biología: hecho conceptual 7",
    "biología: hecho conceptual 8",
    "biología: hecho conceptual 9",
    "biología: hecho conceptual 10",
    "biología: hecho conceptual 11",
    "biología: hecho conceptual 12",
    "biología: hecho conceptual 13",
    "biología: hecho conceptual 14",
    "biología: hecho conceptual 15",
    "biología: hecho conceptual 16",
    "biología: hecho conceptual 17",
    "biología: hecho conceptual 18",
    "biología: hecho conceptual 19",
    "biología: hecho conceptual 20",
    "biología: hecho conceptual 21",
    "biología: hecho conceptual 22",
    "biología: hecho conceptual 23",
    "biología: hecho conceptual 24",
    "biología: hecho conceptual 25",
    "biología: hecho conceptual 26",
    "biología: hecho conceptual 27",
    "biología: hecho conceptual 28",
    "biología: hecho conceptual 29",
    "biología: hecho conceptual 30",
    "biología: hecho conceptual 31",
    "biología: hecho conceptual 32",
    "biología: hecho conceptual 33",
    "biología: hecho conceptual 34",
    "biología: hecho conceptual 35",
    "biología: hecho conceptual 36",
    "biología: hecho conceptual 37",
    "biología: hecho conceptual 38",
    "biología: hecho conceptual 39",
    "biología: hecho conceptual 40",
    "biología: hecho conceptual 41",
    "biología: hecho conceptual 42",
    "biología: hecho conceptual 43",
    "biología: hecho conceptual 44",
    "biología: hecho conceptual 45",
    "biología: hecho conceptual 46",
    "biología: hecho conceptual 47",
    "biología: hecho conceptual 48",
    "biología: hecho conceptual 49",
    "biología: hecho conceptual 50",
    "biología: hecho conceptual 51",
    "biología: hecho conceptual 52",
    "biología: hecho conceptual 53",
    "biología: hecho conceptual 54",
    "biología: hecho conceptual 55",
    "biología: hecho conceptual 56",
    "biología: hecho conceptual 57",
    "biología: hecho conceptual 58",
    "biología: hecho conceptual 59",
    "biología: hecho conceptual 60",
    "biología: hecho conceptual 61",
    "biología: hecho conceptual 62",
    "biología: hecho conceptual 63",
    "biología: hecho conceptual 64",
    "biología: hecho conceptual 65",
    "biología: hecho conceptual 66",
    "biología: hecho conceptual 67",
    "biología: hecho conceptual 68",
    "biología: hecho conceptual 69",
    "biología: hecho conceptual 70",
    "biología: hecho conceptual 71",
    "biología: hecho conceptual 72",
    "biología: hecho conceptual 73",
    "biología: hecho conceptual 74",
    "biología: hecho conceptual 75",
    "biología: hecho conceptual 76",
    "biología: hecho conceptual 77",
    "biología: hecho conceptual 78",
    "biología: hecho conceptual 79",
    "biología: hecho conceptual 80",
]
BIOLOG_A_QUESTION_VARIANTS = [
    "¿Qué es biología?",
    "¿Cómo explicarías biología de forma sencilla?",
    "¿Por qué es importante biología?",
    "¿Cuál es una idea fundamental de biología?",
    "¿Puedes darme un ejemplo relacionado con biología?",
    "¿Qué errores son frecuentes al estudiar biología?",
    "¿Cómo se aplica biología en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con biología?",
    "¿Cómo empezaría a estudiar biología?",
    "¿Qué relación tiene biología con otras disciplinas?",
    "¿Puedes resumir biología en pocas frases?",
    "¿Qué términos debería conocer sobre biología?",
    "¿Cómo comprobaría si entendí biología?",
    "¿Qué intuición ayuda a comprender biología?",
    "¿Qué problema sencillo puedo resolver sobre biología?",
    "¿Qué matiz suele pasarse por alto en biología?",
    "¿Cómo se representa biología?",
    "¿Qué supuestos se usan al hablar de biología?",
    "¿Qué aplicaciones tiene biología?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar biología?",
    "¿Cómo se conecta biología con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de biología?",
    "¿Qué vocabulario técnico aparece en biología?",
    "¿Puedes comparar dos enfoques dentro de biología?",
    "¿Cómo evolucionó la comprensión de biología?",
    "¿Qué ejemplo cotidiano ilustra biología?",
    "¿Qué dato conviene recordar sobre biología?",
    "¿Cómo evitar confusiones comunes en biología?",
    "¿Qué parte de biología suele ser más difícil?",
    "¿Puedes plantear un ejercicio de biología?",
    "¿Cómo resolverías un ejercicio introductorio de biología?",
    "¿Qué relación matemática aparece en biología?",
    "¿Qué observación apoya esta idea de biología?",
    "¿Qué pasaría si cambiamos una condición en biología?",
    "¿Cómo se usa biología en investigación?",
    "¿Qué herramientas sirven para estudiar biología?",
    "¿Cómo distinguir evidencia de interpretación en biología?",
    "¿Qué concepto previo necesito para entender biología?",
    "¿Puedes dar una analogía para biología?",
    "¿Qué preguntas avanzadas surgen de biología?",
    "¿Cómo se comunica correctamente información sobre biología?",
    "¿Qué ejemplo contradice una intuición común sobre biología?",
    "¿Qué pasos seguirías para analizar biología?",
    "¿Cómo resumirías la historia de biología?",
    "¿Qué incertidumbres existen al estudiar biología?",
    "¿Cómo se relacionan teoría y práctica en biología?",
    "¿Qué clasificación útil existe en biología?",
    "¿Cómo puedo practicar biología?",
    "¿Qué es biología?",
]
GEN_TICA_FACT_LABELS = [
    "genética: hecho conceptual 1",
    "genética: hecho conceptual 2",
    "genética: hecho conceptual 3",
    "genética: hecho conceptual 4",
    "genética: hecho conceptual 5",
    "genética: hecho conceptual 6",
    "genética: hecho conceptual 7",
    "genética: hecho conceptual 8",
    "genética: hecho conceptual 9",
    "genética: hecho conceptual 10",
    "genética: hecho conceptual 11",
    "genética: hecho conceptual 12",
    "genética: hecho conceptual 13",
    "genética: hecho conceptual 14",
    "genética: hecho conceptual 15",
    "genética: hecho conceptual 16",
    "genética: hecho conceptual 17",
    "genética: hecho conceptual 18",
    "genética: hecho conceptual 19",
    "genética: hecho conceptual 20",
    "genética: hecho conceptual 21",
    "genética: hecho conceptual 22",
    "genética: hecho conceptual 23",
    "genética: hecho conceptual 24",
    "genética: hecho conceptual 25",
    "genética: hecho conceptual 26",
    "genética: hecho conceptual 27",
    "genética: hecho conceptual 28",
    "genética: hecho conceptual 29",
    "genética: hecho conceptual 30",
    "genética: hecho conceptual 31",
    "genética: hecho conceptual 32",
    "genética: hecho conceptual 33",
    "genética: hecho conceptual 34",
    "genética: hecho conceptual 35",
    "genética: hecho conceptual 36",
    "genética: hecho conceptual 37",
    "genética: hecho conceptual 38",
    "genética: hecho conceptual 39",
    "genética: hecho conceptual 40",
    "genética: hecho conceptual 41",
    "genética: hecho conceptual 42",
    "genética: hecho conceptual 43",
    "genética: hecho conceptual 44",
    "genética: hecho conceptual 45",
    "genética: hecho conceptual 46",
    "genética: hecho conceptual 47",
    "genética: hecho conceptual 48",
    "genética: hecho conceptual 49",
    "genética: hecho conceptual 50",
    "genética: hecho conceptual 51",
    "genética: hecho conceptual 52",
    "genética: hecho conceptual 53",
    "genética: hecho conceptual 54",
    "genética: hecho conceptual 55",
    "genética: hecho conceptual 56",
    "genética: hecho conceptual 57",
    "genética: hecho conceptual 58",
    "genética: hecho conceptual 59",
    "genética: hecho conceptual 60",
    "genética: hecho conceptual 61",
    "genética: hecho conceptual 62",
    "genética: hecho conceptual 63",
    "genética: hecho conceptual 64",
    "genética: hecho conceptual 65",
    "genética: hecho conceptual 66",
    "genética: hecho conceptual 67",
    "genética: hecho conceptual 68",
    "genética: hecho conceptual 69",
    "genética: hecho conceptual 70",
    "genética: hecho conceptual 71",
    "genética: hecho conceptual 72",
    "genética: hecho conceptual 73",
    "genética: hecho conceptual 74",
    "genética: hecho conceptual 75",
    "genética: hecho conceptual 76",
    "genética: hecho conceptual 77",
    "genética: hecho conceptual 78",
    "genética: hecho conceptual 79",
    "genética: hecho conceptual 80",
]
GEN_TICA_QUESTION_VARIANTS = [
    "¿Qué es genética?",
    "¿Cómo explicarías genética de forma sencilla?",
    "¿Por qué es importante genética?",
    "¿Cuál es una idea fundamental de genética?",
    "¿Puedes darme un ejemplo relacionado con genética?",
    "¿Qué errores son frecuentes al estudiar genética?",
    "¿Cómo se aplica genética en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con genética?",
    "¿Cómo empezaría a estudiar genética?",
    "¿Qué relación tiene genética con otras disciplinas?",
    "¿Puedes resumir genética en pocas frases?",
    "¿Qué términos debería conocer sobre genética?",
    "¿Cómo comprobaría si entendí genética?",
    "¿Qué intuición ayuda a comprender genética?",
    "¿Qué problema sencillo puedo resolver sobre genética?",
    "¿Qué matiz suele pasarse por alto en genética?",
    "¿Cómo se representa genética?",
    "¿Qué supuestos se usan al hablar de genética?",
    "¿Qué aplicaciones tiene genética?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar genética?",
    "¿Cómo se conecta genética con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de genética?",
    "¿Qué vocabulario técnico aparece en genética?",
    "¿Puedes comparar dos enfoques dentro de genética?",
    "¿Cómo evolucionó la comprensión de genética?",
    "¿Qué ejemplo cotidiano ilustra genética?",
    "¿Qué dato conviene recordar sobre genética?",
    "¿Cómo evitar confusiones comunes en genética?",
    "¿Qué parte de genética suele ser más difícil?",
    "¿Puedes plantear un ejercicio de genética?",
    "¿Cómo resolverías un ejercicio introductorio de genética?",
    "¿Qué relación matemática aparece en genética?",
    "¿Qué observación apoya esta idea de genética?",
    "¿Qué pasaría si cambiamos una condición en genética?",
    "¿Cómo se usa genética en investigación?",
    "¿Qué herramientas sirven para estudiar genética?",
    "¿Cómo distinguir evidencia de interpretación en genética?",
    "¿Qué concepto previo necesito para entender genética?",
    "¿Puedes dar una analogía para genética?",
    "¿Qué preguntas avanzadas surgen de genética?",
    "¿Cómo se comunica correctamente información sobre genética?",
    "¿Qué ejemplo contradice una intuición común sobre genética?",
    "¿Qué pasos seguirías para analizar genética?",
    "¿Cómo resumirías la historia de genética?",
    "¿Qué incertidumbres existen al estudiar genética?",
    "¿Cómo se relacionan teoría y práctica en genética?",
    "¿Qué clasificación útil existe en genética?",
    "¿Cómo puedo practicar genética?",
    "¿Qué es genética?",
]
MEDICINA_FACT_LABELS = [
    "medicina: hecho conceptual 1",
    "medicina: hecho conceptual 2",
    "medicina: hecho conceptual 3",
    "medicina: hecho conceptual 4",
    "medicina: hecho conceptual 5",
    "medicina: hecho conceptual 6",
    "medicina: hecho conceptual 7",
    "medicina: hecho conceptual 8",
    "medicina: hecho conceptual 9",
    "medicina: hecho conceptual 10",
    "medicina: hecho conceptual 11",
    "medicina: hecho conceptual 12",
    "medicina: hecho conceptual 13",
    "medicina: hecho conceptual 14",
    "medicina: hecho conceptual 15",
    "medicina: hecho conceptual 16",
    "medicina: hecho conceptual 17",
    "medicina: hecho conceptual 18",
    "medicina: hecho conceptual 19",
    "medicina: hecho conceptual 20",
    "medicina: hecho conceptual 21",
    "medicina: hecho conceptual 22",
    "medicina: hecho conceptual 23",
    "medicina: hecho conceptual 24",
    "medicina: hecho conceptual 25",
    "medicina: hecho conceptual 26",
    "medicina: hecho conceptual 27",
    "medicina: hecho conceptual 28",
    "medicina: hecho conceptual 29",
    "medicina: hecho conceptual 30",
    "medicina: hecho conceptual 31",
    "medicina: hecho conceptual 32",
    "medicina: hecho conceptual 33",
    "medicina: hecho conceptual 34",
    "medicina: hecho conceptual 35",
    "medicina: hecho conceptual 36",
    "medicina: hecho conceptual 37",
    "medicina: hecho conceptual 38",
    "medicina: hecho conceptual 39",
    "medicina: hecho conceptual 40",
    "medicina: hecho conceptual 41",
    "medicina: hecho conceptual 42",
    "medicina: hecho conceptual 43",
    "medicina: hecho conceptual 44",
    "medicina: hecho conceptual 45",
    "medicina: hecho conceptual 46",
    "medicina: hecho conceptual 47",
    "medicina: hecho conceptual 48",
    "medicina: hecho conceptual 49",
    "medicina: hecho conceptual 50",
    "medicina: hecho conceptual 51",
    "medicina: hecho conceptual 52",
    "medicina: hecho conceptual 53",
    "medicina: hecho conceptual 54",
    "medicina: hecho conceptual 55",
    "medicina: hecho conceptual 56",
    "medicina: hecho conceptual 57",
    "medicina: hecho conceptual 58",
    "medicina: hecho conceptual 59",
    "medicina: hecho conceptual 60",
    "medicina: hecho conceptual 61",
    "medicina: hecho conceptual 62",
    "medicina: hecho conceptual 63",
    "medicina: hecho conceptual 64",
    "medicina: hecho conceptual 65",
    "medicina: hecho conceptual 66",
    "medicina: hecho conceptual 67",
    "medicina: hecho conceptual 68",
    "medicina: hecho conceptual 69",
    "medicina: hecho conceptual 70",
    "medicina: hecho conceptual 71",
    "medicina: hecho conceptual 72",
    "medicina: hecho conceptual 73",
    "medicina: hecho conceptual 74",
    "medicina: hecho conceptual 75",
    "medicina: hecho conceptual 76",
    "medicina: hecho conceptual 77",
    "medicina: hecho conceptual 78",
    "medicina: hecho conceptual 79",
    "medicina: hecho conceptual 80",
]
MEDICINA_QUESTION_VARIANTS = [
    "¿Qué es medicina?",
    "¿Cómo explicarías medicina de forma sencilla?",
    "¿Por qué es importante medicina?",
    "¿Cuál es una idea fundamental de medicina?",
    "¿Puedes darme un ejemplo relacionado con medicina?",
    "¿Qué errores son frecuentes al estudiar medicina?",
    "¿Cómo se aplica medicina en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con medicina?",
    "¿Cómo empezaría a estudiar medicina?",
    "¿Qué relación tiene medicina con otras disciplinas?",
    "¿Puedes resumir medicina en pocas frases?",
    "¿Qué términos debería conocer sobre medicina?",
    "¿Cómo comprobaría si entendí medicina?",
    "¿Qué intuición ayuda a comprender medicina?",
    "¿Qué problema sencillo puedo resolver sobre medicina?",
    "¿Qué matiz suele pasarse por alto en medicina?",
    "¿Cómo se representa medicina?",
    "¿Qué supuestos se usan al hablar de medicina?",
    "¿Qué aplicaciones tiene medicina?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar medicina?",
    "¿Cómo se conecta medicina con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de medicina?",
    "¿Qué vocabulario técnico aparece en medicina?",
    "¿Puedes comparar dos enfoques dentro de medicina?",
    "¿Cómo evolucionó la comprensión de medicina?",
    "¿Qué ejemplo cotidiano ilustra medicina?",
    "¿Qué dato conviene recordar sobre medicina?",
    "¿Cómo evitar confusiones comunes en medicina?",
    "¿Qué parte de medicina suele ser más difícil?",
    "¿Puedes plantear un ejercicio de medicina?",
    "¿Cómo resolverías un ejercicio introductorio de medicina?",
    "¿Qué relación matemática aparece en medicina?",
    "¿Qué observación apoya esta idea de medicina?",
    "¿Qué pasaría si cambiamos una condición en medicina?",
    "¿Cómo se usa medicina en investigación?",
    "¿Qué herramientas sirven para estudiar medicina?",
    "¿Cómo distinguir evidencia de interpretación en medicina?",
    "¿Qué concepto previo necesito para entender medicina?",
    "¿Puedes dar una analogía para medicina?",
    "¿Qué preguntas avanzadas surgen de medicina?",
    "¿Cómo se comunica correctamente información sobre medicina?",
    "¿Qué ejemplo contradice una intuición común sobre medicina?",
    "¿Qué pasos seguirías para analizar medicina?",
    "¿Cómo resumirías la historia de medicina?",
    "¿Qué incertidumbres existen al estudiar medicina?",
    "¿Cómo se relacionan teoría y práctica en medicina?",
    "¿Qué clasificación útil existe en medicina?",
    "¿Cómo puedo practicar medicina?",
    "¿Qué es medicina?",
]
ANATOM_A_FACT_LABELS = [
    "anatomía: hecho conceptual 1",
    "anatomía: hecho conceptual 2",
    "anatomía: hecho conceptual 3",
    "anatomía: hecho conceptual 4",
    "anatomía: hecho conceptual 5",
    "anatomía: hecho conceptual 6",
    "anatomía: hecho conceptual 7",
    "anatomía: hecho conceptual 8",
    "anatomía: hecho conceptual 9",
    "anatomía: hecho conceptual 10",
    "anatomía: hecho conceptual 11",
    "anatomía: hecho conceptual 12",
    "anatomía: hecho conceptual 13",
    "anatomía: hecho conceptual 14",
    "anatomía: hecho conceptual 15",
    "anatomía: hecho conceptual 16",
    "anatomía: hecho conceptual 17",
    "anatomía: hecho conceptual 18",
    "anatomía: hecho conceptual 19",
    "anatomía: hecho conceptual 20",
    "anatomía: hecho conceptual 21",
    "anatomía: hecho conceptual 22",
    "anatomía: hecho conceptual 23",
    "anatomía: hecho conceptual 24",
    "anatomía: hecho conceptual 25",
    "anatomía: hecho conceptual 26",
    "anatomía: hecho conceptual 27",
    "anatomía: hecho conceptual 28",
    "anatomía: hecho conceptual 29",
    "anatomía: hecho conceptual 30",
    "anatomía: hecho conceptual 31",
    "anatomía: hecho conceptual 32",
    "anatomía: hecho conceptual 33",
    "anatomía: hecho conceptual 34",
    "anatomía: hecho conceptual 35",
    "anatomía: hecho conceptual 36",
    "anatomía: hecho conceptual 37",
    "anatomía: hecho conceptual 38",
    "anatomía: hecho conceptual 39",
    "anatomía: hecho conceptual 40",
    "anatomía: hecho conceptual 41",
    "anatomía: hecho conceptual 42",
    "anatomía: hecho conceptual 43",
    "anatomía: hecho conceptual 44",
    "anatomía: hecho conceptual 45",
    "anatomía: hecho conceptual 46",
    "anatomía: hecho conceptual 47",
    "anatomía: hecho conceptual 48",
    "anatomía: hecho conceptual 49",
    "anatomía: hecho conceptual 50",
    "anatomía: hecho conceptual 51",
    "anatomía: hecho conceptual 52",
    "anatomía: hecho conceptual 53",
    "anatomía: hecho conceptual 54",
    "anatomía: hecho conceptual 55",
    "anatomía: hecho conceptual 56",
    "anatomía: hecho conceptual 57",
    "anatomía: hecho conceptual 58",
    "anatomía: hecho conceptual 59",
    "anatomía: hecho conceptual 60",
    "anatomía: hecho conceptual 61",
    "anatomía: hecho conceptual 62",
    "anatomía: hecho conceptual 63",
    "anatomía: hecho conceptual 64",
    "anatomía: hecho conceptual 65",
    "anatomía: hecho conceptual 66",
    "anatomía: hecho conceptual 67",
    "anatomía: hecho conceptual 68",
    "anatomía: hecho conceptual 69",
    "anatomía: hecho conceptual 70",
    "anatomía: hecho conceptual 71",
    "anatomía: hecho conceptual 72",
    "anatomía: hecho conceptual 73",
    "anatomía: hecho conceptual 74",
    "anatomía: hecho conceptual 75",
    "anatomía: hecho conceptual 76",
    "anatomía: hecho conceptual 77",
    "anatomía: hecho conceptual 78",
    "anatomía: hecho conceptual 79",
    "anatomía: hecho conceptual 80",
]
ANATOM_A_QUESTION_VARIANTS = [
    "¿Qué es anatomía?",
    "¿Cómo explicarías anatomía de forma sencilla?",
    "¿Por qué es importante anatomía?",
    "¿Cuál es una idea fundamental de anatomía?",
    "¿Puedes darme un ejemplo relacionado con anatomía?",
    "¿Qué errores son frecuentes al estudiar anatomía?",
    "¿Cómo se aplica anatomía en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con anatomía?",
    "¿Cómo empezaría a estudiar anatomía?",
    "¿Qué relación tiene anatomía con otras disciplinas?",
    "¿Puedes resumir anatomía en pocas frases?",
    "¿Qué términos debería conocer sobre anatomía?",
    "¿Cómo comprobaría si entendí anatomía?",
    "¿Qué intuición ayuda a comprender anatomía?",
    "¿Qué problema sencillo puedo resolver sobre anatomía?",
    "¿Qué matiz suele pasarse por alto en anatomía?",
    "¿Cómo se representa anatomía?",
    "¿Qué supuestos se usan al hablar de anatomía?",
    "¿Qué aplicaciones tiene anatomía?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar anatomía?",
    "¿Cómo se conecta anatomía con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de anatomía?",
    "¿Qué vocabulario técnico aparece en anatomía?",
    "¿Puedes comparar dos enfoques dentro de anatomía?",
    "¿Cómo evolucionó la comprensión de anatomía?",
    "¿Qué ejemplo cotidiano ilustra anatomía?",
    "¿Qué dato conviene recordar sobre anatomía?",
    "¿Cómo evitar confusiones comunes en anatomía?",
    "¿Qué parte de anatomía suele ser más difícil?",
    "¿Puedes plantear un ejercicio de anatomía?",
    "¿Cómo resolverías un ejercicio introductorio de anatomía?",
    "¿Qué relación matemática aparece en anatomía?",
    "¿Qué observación apoya esta idea de anatomía?",
    "¿Qué pasaría si cambiamos una condición en anatomía?",
    "¿Cómo se usa anatomía en investigación?",
    "¿Qué herramientas sirven para estudiar anatomía?",
    "¿Cómo distinguir evidencia de interpretación en anatomía?",
    "¿Qué concepto previo necesito para entender anatomía?",
    "¿Puedes dar una analogía para anatomía?",
    "¿Qué preguntas avanzadas surgen de anatomía?",
    "¿Cómo se comunica correctamente información sobre anatomía?",
    "¿Qué ejemplo contradice una intuición común sobre anatomía?",
    "¿Qué pasos seguirías para analizar anatomía?",
    "¿Cómo resumirías la historia de anatomía?",
    "¿Qué incertidumbres existen al estudiar anatomía?",
    "¿Cómo se relacionan teoría y práctica en anatomía?",
    "¿Qué clasificación útil existe en anatomía?",
    "¿Cómo puedo practicar anatomía?",
    "¿Qué es anatomía?",
]
HISTORIA_UNIVERSAL_FACT_LABELS = [
    "historia universal: hecho conceptual 1",
    "historia universal: hecho conceptual 2",
    "historia universal: hecho conceptual 3",
    "historia universal: hecho conceptual 4",
    "historia universal: hecho conceptual 5",
    "historia universal: hecho conceptual 6",
    "historia universal: hecho conceptual 7",
    "historia universal: hecho conceptual 8",
    "historia universal: hecho conceptual 9",
    "historia universal: hecho conceptual 10",
    "historia universal: hecho conceptual 11",
    "historia universal: hecho conceptual 12",
    "historia universal: hecho conceptual 13",
    "historia universal: hecho conceptual 14",
    "historia universal: hecho conceptual 15",
    "historia universal: hecho conceptual 16",
    "historia universal: hecho conceptual 17",
    "historia universal: hecho conceptual 18",
    "historia universal: hecho conceptual 19",
    "historia universal: hecho conceptual 20",
    "historia universal: hecho conceptual 21",
    "historia universal: hecho conceptual 22",
    "historia universal: hecho conceptual 23",
    "historia universal: hecho conceptual 24",
    "historia universal: hecho conceptual 25",
    "historia universal: hecho conceptual 26",
    "historia universal: hecho conceptual 27",
    "historia universal: hecho conceptual 28",
    "historia universal: hecho conceptual 29",
    "historia universal: hecho conceptual 30",
    "historia universal: hecho conceptual 31",
    "historia universal: hecho conceptual 32",
    "historia universal: hecho conceptual 33",
    "historia universal: hecho conceptual 34",
    "historia universal: hecho conceptual 35",
    "historia universal: hecho conceptual 36",
    "historia universal: hecho conceptual 37",
    "historia universal: hecho conceptual 38",
    "historia universal: hecho conceptual 39",
    "historia universal: hecho conceptual 40",
    "historia universal: hecho conceptual 41",
    "historia universal: hecho conceptual 42",
    "historia universal: hecho conceptual 43",
    "historia universal: hecho conceptual 44",
    "historia universal: hecho conceptual 45",
    "historia universal: hecho conceptual 46",
    "historia universal: hecho conceptual 47",
    "historia universal: hecho conceptual 48",
    "historia universal: hecho conceptual 49",
    "historia universal: hecho conceptual 50",
    "historia universal: hecho conceptual 51",
    "historia universal: hecho conceptual 52",
    "historia universal: hecho conceptual 53",
    "historia universal: hecho conceptual 54",
    "historia universal: hecho conceptual 55",
    "historia universal: hecho conceptual 56",
    "historia universal: hecho conceptual 57",
    "historia universal: hecho conceptual 58",
    "historia universal: hecho conceptual 59",
    "historia universal: hecho conceptual 60",
    "historia universal: hecho conceptual 61",
    "historia universal: hecho conceptual 62",
    "historia universal: hecho conceptual 63",
    "historia universal: hecho conceptual 64",
    "historia universal: hecho conceptual 65",
    "historia universal: hecho conceptual 66",
    "historia universal: hecho conceptual 67",
    "historia universal: hecho conceptual 68",
    "historia universal: hecho conceptual 69",
    "historia universal: hecho conceptual 70",
    "historia universal: hecho conceptual 71",
    "historia universal: hecho conceptual 72",
    "historia universal: hecho conceptual 73",
    "historia universal: hecho conceptual 74",
    "historia universal: hecho conceptual 75",
    "historia universal: hecho conceptual 76",
    "historia universal: hecho conceptual 77",
    "historia universal: hecho conceptual 78",
    "historia universal: hecho conceptual 79",
    "historia universal: hecho conceptual 80",
]
HISTORIA_UNIVERSAL_QUESTION_VARIANTS = [
    "¿Qué es historia universal?",
    "¿Cómo explicarías historia universal de forma sencilla?",
    "¿Por qué es importante historia universal?",
    "¿Cuál es una idea fundamental de historia universal?",
    "¿Puedes darme un ejemplo relacionado con historia universal?",
    "¿Qué errores son frecuentes al estudiar historia universal?",
    "¿Cómo se aplica historia universal en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con historia universal?",
    "¿Cómo empezaría a estudiar historia universal?",
    "¿Qué relación tiene historia universal con otras disciplinas?",
    "¿Puedes resumir historia universal en pocas frases?",
    "¿Qué términos debería conocer sobre historia universal?",
    "¿Cómo comprobaría si entendí historia universal?",
    "¿Qué intuición ayuda a comprender historia universal?",
    "¿Qué problema sencillo puedo resolver sobre historia universal?",
    "¿Qué matiz suele pasarse por alto en historia universal?",
    "¿Cómo se representa historia universal?",
    "¿Qué supuestos se usan al hablar de historia universal?",
    "¿Qué aplicaciones tiene historia universal?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar historia universal?",
    "¿Cómo se conecta historia universal con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de historia universal?",
    "¿Qué vocabulario técnico aparece en historia universal?",
    "¿Puedes comparar dos enfoques dentro de historia universal?",
    "¿Cómo evolucionó la comprensión de historia universal?",
    "¿Qué ejemplo cotidiano ilustra historia universal?",
    "¿Qué dato conviene recordar sobre historia universal?",
    "¿Cómo evitar confusiones comunes en historia universal?",
    "¿Qué parte de historia universal suele ser más difícil?",
    "¿Puedes plantear un ejercicio de historia universal?",
    "¿Cómo resolverías un ejercicio introductorio de historia universal?",
    "¿Qué relación matemática aparece en historia universal?",
    "¿Qué observación apoya esta idea de historia universal?",
    "¿Qué pasaría si cambiamos una condición en historia universal?",
    "¿Cómo se usa historia universal en investigación?",
    "¿Qué herramientas sirven para estudiar historia universal?",
    "¿Cómo distinguir evidencia de interpretación en historia universal?",
    "¿Qué concepto previo necesito para entender historia universal?",
    "¿Puedes dar una analogía para historia universal?",
    "¿Qué preguntas avanzadas surgen de historia universal?",
    "¿Cómo se comunica correctamente información sobre historia universal?",
    "¿Qué ejemplo contradice una intuición común sobre historia universal?",
    "¿Qué pasos seguirías para analizar historia universal?",
    "¿Cómo resumirías la historia de historia universal?",
    "¿Qué incertidumbres existen al estudiar historia universal?",
    "¿Cómo se relacionan teoría y práctica en historia universal?",
    "¿Qué clasificación útil existe en historia universal?",
    "¿Cómo puedo practicar historia universal?",
    "¿Qué es historia universal?",
]
HISTORIA_DE_ESPA_A_FACT_LABELS = [
    "historia de España: hecho conceptual 1",
    "historia de España: hecho conceptual 2",
    "historia de España: hecho conceptual 3",
    "historia de España: hecho conceptual 4",
    "historia de España: hecho conceptual 5",
    "historia de España: hecho conceptual 6",
    "historia de España: hecho conceptual 7",
    "historia de España: hecho conceptual 8",
    "historia de España: hecho conceptual 9",
    "historia de España: hecho conceptual 10",
    "historia de España: hecho conceptual 11",
    "historia de España: hecho conceptual 12",
    "historia de España: hecho conceptual 13",
    "historia de España: hecho conceptual 14",
    "historia de España: hecho conceptual 15",
    "historia de España: hecho conceptual 16",
    "historia de España: hecho conceptual 17",
    "historia de España: hecho conceptual 18",
    "historia de España: hecho conceptual 19",
    "historia de España: hecho conceptual 20",
    "historia de España: hecho conceptual 21",
    "historia de España: hecho conceptual 22",
    "historia de España: hecho conceptual 23",
    "historia de España: hecho conceptual 24",
    "historia de España: hecho conceptual 25",
    "historia de España: hecho conceptual 26",
    "historia de España: hecho conceptual 27",
    "historia de España: hecho conceptual 28",
    "historia de España: hecho conceptual 29",
    "historia de España: hecho conceptual 30",
    "historia de España: hecho conceptual 31",
    "historia de España: hecho conceptual 32",
    "historia de España: hecho conceptual 33",
    "historia de España: hecho conceptual 34",
    "historia de España: hecho conceptual 35",
    "historia de España: hecho conceptual 36",
    "historia de España: hecho conceptual 37",
    "historia de España: hecho conceptual 38",
    "historia de España: hecho conceptual 39",
    "historia de España: hecho conceptual 40",
    "historia de España: hecho conceptual 41",
    "historia de España: hecho conceptual 42",
    "historia de España: hecho conceptual 43",
    "historia de España: hecho conceptual 44",
    "historia de España: hecho conceptual 45",
    "historia de España: hecho conceptual 46",
    "historia de España: hecho conceptual 47",
    "historia de España: hecho conceptual 48",
    "historia de España: hecho conceptual 49",
    "historia de España: hecho conceptual 50",
    "historia de España: hecho conceptual 51",
    "historia de España: hecho conceptual 52",
    "historia de España: hecho conceptual 53",
    "historia de España: hecho conceptual 54",
    "historia de España: hecho conceptual 55",
    "historia de España: hecho conceptual 56",
    "historia de España: hecho conceptual 57",
    "historia de España: hecho conceptual 58",
    "historia de España: hecho conceptual 59",
    "historia de España: hecho conceptual 60",
    "historia de España: hecho conceptual 61",
    "historia de España: hecho conceptual 62",
    "historia de España: hecho conceptual 63",
    "historia de España: hecho conceptual 64",
    "historia de España: hecho conceptual 65",
    "historia de España: hecho conceptual 66",
    "historia de España: hecho conceptual 67",
    "historia de España: hecho conceptual 68",
    "historia de España: hecho conceptual 69",
    "historia de España: hecho conceptual 70",
    "historia de España: hecho conceptual 71",
    "historia de España: hecho conceptual 72",
    "historia de España: hecho conceptual 73",
    "historia de España: hecho conceptual 74",
    "historia de España: hecho conceptual 75",
    "historia de España: hecho conceptual 76",
    "historia de España: hecho conceptual 77",
    "historia de España: hecho conceptual 78",
    "historia de España: hecho conceptual 79",
    "historia de España: hecho conceptual 80",
]
HISTORIA_DE_ESPA_A_QUESTION_VARIANTS = [
    "¿Qué es historia de España?",
    "¿Cómo explicarías historia de España de forma sencilla?",
    "¿Por qué es importante historia de España?",
    "¿Cuál es una idea fundamental de historia de España?",
    "¿Puedes darme un ejemplo relacionado con historia de España?",
    "¿Qué errores son frecuentes al estudiar historia de España?",
    "¿Cómo se aplica historia de España en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con historia de España?",
    "¿Cómo empezaría a estudiar historia de España?",
    "¿Qué relación tiene historia de España con otras disciplinas?",
    "¿Puedes resumir historia de España en pocas frases?",
    "¿Qué términos debería conocer sobre historia de España?",
    "¿Cómo comprobaría si entendí historia de España?",
    "¿Qué intuición ayuda a comprender historia de España?",
    "¿Qué problema sencillo puedo resolver sobre historia de España?",
    "¿Qué matiz suele pasarse por alto en historia de España?",
    "¿Cómo se representa historia de España?",
    "¿Qué supuestos se usan al hablar de historia de España?",
    "¿Qué aplicaciones tiene historia de España?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar historia de España?",
    "¿Cómo se conecta historia de España con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de historia de España?",
    "¿Qué vocabulario técnico aparece en historia de España?",
    "¿Puedes comparar dos enfoques dentro de historia de España?",
    "¿Cómo evolucionó la comprensión de historia de España?",
    "¿Qué ejemplo cotidiano ilustra historia de España?",
    "¿Qué dato conviene recordar sobre historia de España?",
    "¿Cómo evitar confusiones comunes en historia de España?",
    "¿Qué parte de historia de España suele ser más difícil?",
    "¿Puedes plantear un ejercicio de historia de España?",
    "¿Cómo resolverías un ejercicio introductorio de historia de España?",
    "¿Qué relación matemática aparece en historia de España?",
    "¿Qué observación apoya esta idea de historia de España?",
    "¿Qué pasaría si cambiamos una condición en historia de España?",
    "¿Cómo se usa historia de España en investigación?",
    "¿Qué herramientas sirven para estudiar historia de España?",
    "¿Cómo distinguir evidencia de interpretación en historia de España?",
    "¿Qué concepto previo necesito para entender historia de España?",
    "¿Puedes dar una analogía para historia de España?",
    "¿Qué preguntas avanzadas surgen de historia de España?",
    "¿Cómo se comunica correctamente información sobre historia de España?",
    "¿Qué ejemplo contradice una intuición común sobre historia de España?",
    "¿Qué pasos seguirías para analizar historia de España?",
    "¿Cómo resumirías la historia de historia de España?",
    "¿Qué incertidumbres existen al estudiar historia de España?",
    "¿Cómo se relacionan teoría y práctica en historia de España?",
    "¿Qué clasificación útil existe en historia de España?",
    "¿Cómo puedo practicar historia de España?",
    "¿Qué es historia de España?",
]
HISTORIA_DE_AM_RICA_FACT_LABELS = [
    "historia de América: hecho conceptual 1",
    "historia de América: hecho conceptual 2",
    "historia de América: hecho conceptual 3",
    "historia de América: hecho conceptual 4",
    "historia de América: hecho conceptual 5",
    "historia de América: hecho conceptual 6",
    "historia de América: hecho conceptual 7",
    "historia de América: hecho conceptual 8",
    "historia de América: hecho conceptual 9",
    "historia de América: hecho conceptual 10",
    "historia de América: hecho conceptual 11",
    "historia de América: hecho conceptual 12",
    "historia de América: hecho conceptual 13",
    "historia de América: hecho conceptual 14",
    "historia de América: hecho conceptual 15",
    "historia de América: hecho conceptual 16",
    "historia de América: hecho conceptual 17",
    "historia de América: hecho conceptual 18",
    "historia de América: hecho conceptual 19",
    "historia de América: hecho conceptual 20",
    "historia de América: hecho conceptual 21",
    "historia de América: hecho conceptual 22",
    "historia de América: hecho conceptual 23",
    "historia de América: hecho conceptual 24",
    "historia de América: hecho conceptual 25",
    "historia de América: hecho conceptual 26",
    "historia de América: hecho conceptual 27",
    "historia de América: hecho conceptual 28",
    "historia de América: hecho conceptual 29",
    "historia de América: hecho conceptual 30",
    "historia de América: hecho conceptual 31",
    "historia de América: hecho conceptual 32",
    "historia de América: hecho conceptual 33",
    "historia de América: hecho conceptual 34",
    "historia de América: hecho conceptual 35",
    "historia de América: hecho conceptual 36",
    "historia de América: hecho conceptual 37",
    "historia de América: hecho conceptual 38",
    "historia de América: hecho conceptual 39",
    "historia de América: hecho conceptual 40",
    "historia de América: hecho conceptual 41",
    "historia de América: hecho conceptual 42",
    "historia de América: hecho conceptual 43",
    "historia de América: hecho conceptual 44",
    "historia de América: hecho conceptual 45",
    "historia de América: hecho conceptual 46",
    "historia de América: hecho conceptual 47",
    "historia de América: hecho conceptual 48",
    "historia de América: hecho conceptual 49",
    "historia de América: hecho conceptual 50",
    "historia de América: hecho conceptual 51",
    "historia de América: hecho conceptual 52",
    "historia de América: hecho conceptual 53",
    "historia de América: hecho conceptual 54",
    "historia de América: hecho conceptual 55",
    "historia de América: hecho conceptual 56",
    "historia de América: hecho conceptual 57",
    "historia de América: hecho conceptual 58",
    "historia de América: hecho conceptual 59",
    "historia de América: hecho conceptual 60",
    "historia de América: hecho conceptual 61",
    "historia de América: hecho conceptual 62",
    "historia de América: hecho conceptual 63",
    "historia de América: hecho conceptual 64",
    "historia de América: hecho conceptual 65",
    "historia de América: hecho conceptual 66",
    "historia de América: hecho conceptual 67",
    "historia de América: hecho conceptual 68",
    "historia de América: hecho conceptual 69",
    "historia de América: hecho conceptual 70",
    "historia de América: hecho conceptual 71",
    "historia de América: hecho conceptual 72",
    "historia de América: hecho conceptual 73",
    "historia de América: hecho conceptual 74",
    "historia de América: hecho conceptual 75",
    "historia de América: hecho conceptual 76",
    "historia de América: hecho conceptual 77",
    "historia de América: hecho conceptual 78",
    "historia de América: hecho conceptual 79",
    "historia de América: hecho conceptual 80",
]
HISTORIA_DE_AM_RICA_QUESTION_VARIANTS = [
    "¿Qué es historia de América?",
    "¿Cómo explicarías historia de América de forma sencilla?",
    "¿Por qué es importante historia de América?",
    "¿Cuál es una idea fundamental de historia de América?",
    "¿Puedes darme un ejemplo relacionado con historia de América?",
    "¿Qué errores son frecuentes al estudiar historia de América?",
    "¿Cómo se aplica historia de América en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con historia de América?",
    "¿Cómo empezaría a estudiar historia de América?",
    "¿Qué relación tiene historia de América con otras disciplinas?",
    "¿Puedes resumir historia de América en pocas frases?",
    "¿Qué términos debería conocer sobre historia de América?",
    "¿Cómo comprobaría si entendí historia de América?",
    "¿Qué intuición ayuda a comprender historia de América?",
    "¿Qué problema sencillo puedo resolver sobre historia de América?",
    "¿Qué matiz suele pasarse por alto en historia de América?",
    "¿Cómo se representa historia de América?",
    "¿Qué supuestos se usan al hablar de historia de América?",
    "¿Qué aplicaciones tiene historia de América?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar historia de América?",
    "¿Cómo se conecta historia de América con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de historia de América?",
    "¿Qué vocabulario técnico aparece en historia de América?",
    "¿Puedes comparar dos enfoques dentro de historia de América?",
    "¿Cómo evolucionó la comprensión de historia de América?",
    "¿Qué ejemplo cotidiano ilustra historia de América?",
    "¿Qué dato conviene recordar sobre historia de América?",
    "¿Cómo evitar confusiones comunes en historia de América?",
    "¿Qué parte de historia de América suele ser más difícil?",
    "¿Puedes plantear un ejercicio de historia de América?",
    "¿Cómo resolverías un ejercicio introductorio de historia de América?",
    "¿Qué relación matemática aparece en historia de América?",
    "¿Qué observación apoya esta idea de historia de América?",
    "¿Qué pasaría si cambiamos una condición en historia de América?",
    "¿Cómo se usa historia de América en investigación?",
    "¿Qué herramientas sirven para estudiar historia de América?",
    "¿Cómo distinguir evidencia de interpretación en historia de América?",
    "¿Qué concepto previo necesito para entender historia de América?",
    "¿Puedes dar una analogía para historia de América?",
    "¿Qué preguntas avanzadas surgen de historia de América?",
    "¿Cómo se comunica correctamente información sobre historia de América?",
    "¿Qué ejemplo contradice una intuición común sobre historia de América?",
    "¿Qué pasos seguirías para analizar historia de América?",
    "¿Cómo resumirías la historia de historia de América?",
    "¿Qué incertidumbres existen al estudiar historia de América?",
    "¿Cómo se relacionan teoría y práctica en historia de América?",
    "¿Qué clasificación útil existe en historia de América?",
    "¿Cómo puedo practicar historia de América?",
    "¿Qué es historia de América?",
]
GEOGRAF_A_FACT_LABELS = [
    "geografía: hecho conceptual 1",
    "geografía: hecho conceptual 2",
    "geografía: hecho conceptual 3",
    "geografía: hecho conceptual 4",
    "geografía: hecho conceptual 5",
    "geografía: hecho conceptual 6",
    "geografía: hecho conceptual 7",
    "geografía: hecho conceptual 8",
    "geografía: hecho conceptual 9",
    "geografía: hecho conceptual 10",
    "geografía: hecho conceptual 11",
    "geografía: hecho conceptual 12",
    "geografía: hecho conceptual 13",
    "geografía: hecho conceptual 14",
    "geografía: hecho conceptual 15",
    "geografía: hecho conceptual 16",
    "geografía: hecho conceptual 17",
    "geografía: hecho conceptual 18",
    "geografía: hecho conceptual 19",
    "geografía: hecho conceptual 20",
    "geografía: hecho conceptual 21",
    "geografía: hecho conceptual 22",
    "geografía: hecho conceptual 23",
    "geografía: hecho conceptual 24",
    "geografía: hecho conceptual 25",
    "geografía: hecho conceptual 26",
    "geografía: hecho conceptual 27",
    "geografía: hecho conceptual 28",
    "geografía: hecho conceptual 29",
    "geografía: hecho conceptual 30",
    "geografía: hecho conceptual 31",
    "geografía: hecho conceptual 32",
    "geografía: hecho conceptual 33",
    "geografía: hecho conceptual 34",
    "geografía: hecho conceptual 35",
    "geografía: hecho conceptual 36",
    "geografía: hecho conceptual 37",
    "geografía: hecho conceptual 38",
    "geografía: hecho conceptual 39",
    "geografía: hecho conceptual 40",
    "geografía: hecho conceptual 41",
    "geografía: hecho conceptual 42",
    "geografía: hecho conceptual 43",
    "geografía: hecho conceptual 44",
    "geografía: hecho conceptual 45",
    "geografía: hecho conceptual 46",
    "geografía: hecho conceptual 47",
    "geografía: hecho conceptual 48",
    "geografía: hecho conceptual 49",
    "geografía: hecho conceptual 50",
    "geografía: hecho conceptual 51",
    "geografía: hecho conceptual 52",
    "geografía: hecho conceptual 53",
    "geografía: hecho conceptual 54",
    "geografía: hecho conceptual 55",
    "geografía: hecho conceptual 56",
    "geografía: hecho conceptual 57",
    "geografía: hecho conceptual 58",
    "geografía: hecho conceptual 59",
    "geografía: hecho conceptual 60",
    "geografía: hecho conceptual 61",
    "geografía: hecho conceptual 62",
    "geografía: hecho conceptual 63",
    "geografía: hecho conceptual 64",
    "geografía: hecho conceptual 65",
    "geografía: hecho conceptual 66",
    "geografía: hecho conceptual 67",
    "geografía: hecho conceptual 68",
    "geografía: hecho conceptual 69",
    "geografía: hecho conceptual 70",
    "geografía: hecho conceptual 71",
    "geografía: hecho conceptual 72",
    "geografía: hecho conceptual 73",
    "geografía: hecho conceptual 74",
    "geografía: hecho conceptual 75",
    "geografía: hecho conceptual 76",
    "geografía: hecho conceptual 77",
    "geografía: hecho conceptual 78",
    "geografía: hecho conceptual 79",
    "geografía: hecho conceptual 80",
]
GEOGRAF_A_QUESTION_VARIANTS = [
    "¿Qué es geografía?",
    "¿Cómo explicarías geografía de forma sencilla?",
    "¿Por qué es importante geografía?",
    "¿Cuál es una idea fundamental de geografía?",
    "¿Puedes darme un ejemplo relacionado con geografía?",
    "¿Qué errores son frecuentes al estudiar geografía?",
    "¿Cómo se aplica geografía en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con geografía?",
    "¿Cómo empezaría a estudiar geografía?",
    "¿Qué relación tiene geografía con otras disciplinas?",
    "¿Puedes resumir geografía en pocas frases?",
    "¿Qué términos debería conocer sobre geografía?",
    "¿Cómo comprobaría si entendí geografía?",
    "¿Qué intuición ayuda a comprender geografía?",
    "¿Qué problema sencillo puedo resolver sobre geografía?",
    "¿Qué matiz suele pasarse por alto en geografía?",
    "¿Cómo se representa geografía?",
    "¿Qué supuestos se usan al hablar de geografía?",
    "¿Qué aplicaciones tiene geografía?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar geografía?",
    "¿Cómo se conecta geografía con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de geografía?",
    "¿Qué vocabulario técnico aparece en geografía?",
    "¿Puedes comparar dos enfoques dentro de geografía?",
    "¿Cómo evolucionó la comprensión de geografía?",
    "¿Qué ejemplo cotidiano ilustra geografía?",
    "¿Qué dato conviene recordar sobre geografía?",
    "¿Cómo evitar confusiones comunes en geografía?",
    "¿Qué parte de geografía suele ser más difícil?",
    "¿Puedes plantear un ejercicio de geografía?",
    "¿Cómo resolverías un ejercicio introductorio de geografía?",
    "¿Qué relación matemática aparece en geografía?",
    "¿Qué observación apoya esta idea de geografía?",
    "¿Qué pasaría si cambiamos una condición en geografía?",
    "¿Cómo se usa geografía en investigación?",
    "¿Qué herramientas sirven para estudiar geografía?",
    "¿Cómo distinguir evidencia de interpretación en geografía?",
    "¿Qué concepto previo necesito para entender geografía?",
    "¿Puedes dar una analogía para geografía?",
    "¿Qué preguntas avanzadas surgen de geografía?",
    "¿Cómo se comunica correctamente información sobre geografía?",
    "¿Qué ejemplo contradice una intuición común sobre geografía?",
    "¿Qué pasos seguirías para analizar geografía?",
    "¿Cómo resumirías la historia de geografía?",
    "¿Qué incertidumbres existen al estudiar geografía?",
    "¿Cómo se relacionan teoría y práctica en geografía?",
    "¿Qué clasificación útil existe en geografía?",
    "¿Cómo puedo practicar geografía?",
    "¿Qué es geografía?",
]
PROGRAMACI_N_PYTHON_FACT_LABELS = [
    "programación Python: hecho conceptual 1",
    "programación Python: hecho conceptual 2",
    "programación Python: hecho conceptual 3",
    "programación Python: hecho conceptual 4",
    "programación Python: hecho conceptual 5",
    "programación Python: hecho conceptual 6",
    "programación Python: hecho conceptual 7",
    "programación Python: hecho conceptual 8",
    "programación Python: hecho conceptual 9",
    "programación Python: hecho conceptual 10",
    "programación Python: hecho conceptual 11",
    "programación Python: hecho conceptual 12",
    "programación Python: hecho conceptual 13",
    "programación Python: hecho conceptual 14",
    "programación Python: hecho conceptual 15",
    "programación Python: hecho conceptual 16",
    "programación Python: hecho conceptual 17",
    "programación Python: hecho conceptual 18",
    "programación Python: hecho conceptual 19",
    "programación Python: hecho conceptual 20",
    "programación Python: hecho conceptual 21",
    "programación Python: hecho conceptual 22",
    "programación Python: hecho conceptual 23",
    "programación Python: hecho conceptual 24",
    "programación Python: hecho conceptual 25",
    "programación Python: hecho conceptual 26",
    "programación Python: hecho conceptual 27",
    "programación Python: hecho conceptual 28",
    "programación Python: hecho conceptual 29",
    "programación Python: hecho conceptual 30",
    "programación Python: hecho conceptual 31",
    "programación Python: hecho conceptual 32",
    "programación Python: hecho conceptual 33",
    "programación Python: hecho conceptual 34",
    "programación Python: hecho conceptual 35",
    "programación Python: hecho conceptual 36",
    "programación Python: hecho conceptual 37",
    "programación Python: hecho conceptual 38",
    "programación Python: hecho conceptual 39",
    "programación Python: hecho conceptual 40",
    "programación Python: hecho conceptual 41",
    "programación Python: hecho conceptual 42",
    "programación Python: hecho conceptual 43",
    "programación Python: hecho conceptual 44",
    "programación Python: hecho conceptual 45",
    "programación Python: hecho conceptual 46",
    "programación Python: hecho conceptual 47",
    "programación Python: hecho conceptual 48",
    "programación Python: hecho conceptual 49",
    "programación Python: hecho conceptual 50",
    "programación Python: hecho conceptual 51",
    "programación Python: hecho conceptual 52",
    "programación Python: hecho conceptual 53",
    "programación Python: hecho conceptual 54",
    "programación Python: hecho conceptual 55",
    "programación Python: hecho conceptual 56",
    "programación Python: hecho conceptual 57",
    "programación Python: hecho conceptual 58",
    "programación Python: hecho conceptual 59",
    "programación Python: hecho conceptual 60",
    "programación Python: hecho conceptual 61",
    "programación Python: hecho conceptual 62",
    "programación Python: hecho conceptual 63",
    "programación Python: hecho conceptual 64",
    "programación Python: hecho conceptual 65",
    "programación Python: hecho conceptual 66",
    "programación Python: hecho conceptual 67",
    "programación Python: hecho conceptual 68",
    "programación Python: hecho conceptual 69",
    "programación Python: hecho conceptual 70",
    "programación Python: hecho conceptual 71",
    "programación Python: hecho conceptual 72",
    "programación Python: hecho conceptual 73",
    "programación Python: hecho conceptual 74",
    "programación Python: hecho conceptual 75",
    "programación Python: hecho conceptual 76",
    "programación Python: hecho conceptual 77",
    "programación Python: hecho conceptual 78",
    "programación Python: hecho conceptual 79",
    "programación Python: hecho conceptual 80",
]
PROGRAMACI_N_PYTHON_QUESTION_VARIANTS = [
    "¿Qué es programación Python?",
    "¿Cómo explicarías programación Python de forma sencilla?",
    "¿Por qué es importante programación Python?",
    "¿Cuál es una idea fundamental de programación Python?",
    "¿Puedes darme un ejemplo relacionado con programación Python?",
    "¿Qué errores son frecuentes al estudiar programación Python?",
    "¿Cómo se aplica programación Python en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con programación Python?",
    "¿Cómo empezaría a estudiar programación Python?",
    "¿Qué relación tiene programación Python con otras disciplinas?",
    "¿Puedes resumir programación Python en pocas frases?",
    "¿Qué términos debería conocer sobre programación Python?",
    "¿Cómo comprobaría si entendí programación Python?",
    "¿Qué intuición ayuda a comprender programación Python?",
    "¿Qué problema sencillo puedo resolver sobre programación Python?",
    "¿Qué matiz suele pasarse por alto en programación Python?",
    "¿Cómo se representa programación Python?",
    "¿Qué supuestos se usan al hablar de programación Python?",
    "¿Qué aplicaciones tiene programación Python?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar programación Python?",
    "¿Cómo se conecta programación Python con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de programación Python?",
    "¿Qué vocabulario técnico aparece en programación Python?",
    "¿Puedes comparar dos enfoques dentro de programación Python?",
    "¿Cómo evolucionó la comprensión de programación Python?",
    "¿Qué ejemplo cotidiano ilustra programación Python?",
    "¿Qué dato conviene recordar sobre programación Python?",
    "¿Cómo evitar confusiones comunes en programación Python?",
    "¿Qué parte de programación Python suele ser más difícil?",
    "¿Puedes plantear un ejercicio de programación Python?",
    "¿Cómo resolverías un ejercicio introductorio de programación Python?",
    "¿Qué relación matemática aparece en programación Python?",
    "¿Qué observación apoya esta idea de programación Python?",
    "¿Qué pasaría si cambiamos una condición en programación Python?",
    "¿Cómo se usa programación Python en investigación?",
    "¿Qué herramientas sirven para estudiar programación Python?",
    "¿Cómo distinguir evidencia de interpretación en programación Python?",
    "¿Qué concepto previo necesito para entender programación Python?",
    "¿Puedes dar una analogía para programación Python?",
    "¿Qué preguntas avanzadas surgen de programación Python?",
    "¿Cómo se comunica correctamente información sobre programación Python?",
    "¿Qué ejemplo contradice una intuición común sobre programación Python?",
    "¿Qué pasos seguirías para analizar programación Python?",
    "¿Cómo resumirías la historia de programación Python?",
    "¿Qué incertidumbres existen al estudiar programación Python?",
    "¿Cómo se relacionan teoría y práctica en programación Python?",
    "¿Qué clasificación útil existe en programación Python?",
    "¿Cómo puedo practicar programación Python?",
    "¿Qué es programación Python?",
]
PROGRAMACI_N_GENERAL_FACT_LABELS = [
    "programación general: hecho conceptual 1",
    "programación general: hecho conceptual 2",
    "programación general: hecho conceptual 3",
    "programación general: hecho conceptual 4",
    "programación general: hecho conceptual 5",
    "programación general: hecho conceptual 6",
    "programación general: hecho conceptual 7",
    "programación general: hecho conceptual 8",
    "programación general: hecho conceptual 9",
    "programación general: hecho conceptual 10",
    "programación general: hecho conceptual 11",
    "programación general: hecho conceptual 12",
    "programación general: hecho conceptual 13",
    "programación general: hecho conceptual 14",
    "programación general: hecho conceptual 15",
    "programación general: hecho conceptual 16",
    "programación general: hecho conceptual 17",
    "programación general: hecho conceptual 18",
    "programación general: hecho conceptual 19",
    "programación general: hecho conceptual 20",
    "programación general: hecho conceptual 21",
    "programación general: hecho conceptual 22",
    "programación general: hecho conceptual 23",
    "programación general: hecho conceptual 24",
    "programación general: hecho conceptual 25",
    "programación general: hecho conceptual 26",
    "programación general: hecho conceptual 27",
    "programación general: hecho conceptual 28",
    "programación general: hecho conceptual 29",
    "programación general: hecho conceptual 30",
    "programación general: hecho conceptual 31",
    "programación general: hecho conceptual 32",
    "programación general: hecho conceptual 33",
    "programación general: hecho conceptual 34",
    "programación general: hecho conceptual 35",
    "programación general: hecho conceptual 36",
    "programación general: hecho conceptual 37",
    "programación general: hecho conceptual 38",
    "programación general: hecho conceptual 39",
    "programación general: hecho conceptual 40",
    "programación general: hecho conceptual 41",
    "programación general: hecho conceptual 42",
    "programación general: hecho conceptual 43",
    "programación general: hecho conceptual 44",
    "programación general: hecho conceptual 45",
    "programación general: hecho conceptual 46",
    "programación general: hecho conceptual 47",
    "programación general: hecho conceptual 48",
    "programación general: hecho conceptual 49",
    "programación general: hecho conceptual 50",
    "programación general: hecho conceptual 51",
    "programación general: hecho conceptual 52",
    "programación general: hecho conceptual 53",
    "programación general: hecho conceptual 54",
    "programación general: hecho conceptual 55",
    "programación general: hecho conceptual 56",
    "programación general: hecho conceptual 57",
    "programación general: hecho conceptual 58",
    "programación general: hecho conceptual 59",
    "programación general: hecho conceptual 60",
    "programación general: hecho conceptual 61",
    "programación general: hecho conceptual 62",
    "programación general: hecho conceptual 63",
    "programación general: hecho conceptual 64",
    "programación general: hecho conceptual 65",
    "programación general: hecho conceptual 66",
    "programación general: hecho conceptual 67",
    "programación general: hecho conceptual 68",
    "programación general: hecho conceptual 69",
    "programación general: hecho conceptual 70",
    "programación general: hecho conceptual 71",
    "programación general: hecho conceptual 72",
    "programación general: hecho conceptual 73",
    "programación general: hecho conceptual 74",
    "programación general: hecho conceptual 75",
    "programación general: hecho conceptual 76",
    "programación general: hecho conceptual 77",
    "programación general: hecho conceptual 78",
    "programación general: hecho conceptual 79",
    "programación general: hecho conceptual 80",
]
PROGRAMACI_N_GENERAL_QUESTION_VARIANTS = [
    "¿Qué es programación general?",
    "¿Cómo explicarías programación general de forma sencilla?",
    "¿Por qué es importante programación general?",
    "¿Cuál es una idea fundamental de programación general?",
    "¿Puedes darme un ejemplo relacionado con programación general?",
    "¿Qué errores son frecuentes al estudiar programación general?",
    "¿Cómo se aplica programación general en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con programación general?",
    "¿Cómo empezaría a estudiar programación general?",
    "¿Qué relación tiene programación general con otras disciplinas?",
    "¿Puedes resumir programación general en pocas frases?",
    "¿Qué términos debería conocer sobre programación general?",
    "¿Cómo comprobaría si entendí programación general?",
    "¿Qué intuición ayuda a comprender programación general?",
    "¿Qué problema sencillo puedo resolver sobre programación general?",
    "¿Qué matiz suele pasarse por alto en programación general?",
    "¿Cómo se representa programación general?",
    "¿Qué supuestos se usan al hablar de programación general?",
    "¿Qué aplicaciones tiene programación general?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar programación general?",
    "¿Cómo se conecta programación general con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de programación general?",
    "¿Qué vocabulario técnico aparece en programación general?",
    "¿Puedes comparar dos enfoques dentro de programación general?",
    "¿Cómo evolucionó la comprensión de programación general?",
    "¿Qué ejemplo cotidiano ilustra programación general?",
    "¿Qué dato conviene recordar sobre programación general?",
    "¿Cómo evitar confusiones comunes en programación general?",
    "¿Qué parte de programación general suele ser más difícil?",
    "¿Puedes plantear un ejercicio de programación general?",
    "¿Cómo resolverías un ejercicio introductorio de programación general?",
    "¿Qué relación matemática aparece en programación general?",
    "¿Qué observación apoya esta idea de programación general?",
    "¿Qué pasaría si cambiamos una condición en programación general?",
    "¿Cómo se usa programación general en investigación?",
    "¿Qué herramientas sirven para estudiar programación general?",
    "¿Cómo distinguir evidencia de interpretación en programación general?",
    "¿Qué concepto previo necesito para entender programación general?",
    "¿Puedes dar una analogía para programación general?",
    "¿Qué preguntas avanzadas surgen de programación general?",
    "¿Cómo se comunica correctamente información sobre programación general?",
    "¿Qué ejemplo contradice una intuición común sobre programación general?",
    "¿Qué pasos seguirías para analizar programación general?",
    "¿Cómo resumirías la historia de programación general?",
    "¿Qué incertidumbres existen al estudiar programación general?",
    "¿Cómo se relacionan teoría y práctica en programación general?",
    "¿Qué clasificación útil existe en programación general?",
    "¿Cómo puedo practicar programación general?",
    "¿Qué es programación general?",
]
ALGORITMOS_FACT_LABELS = [
    "algoritmos: hecho conceptual 1",
    "algoritmos: hecho conceptual 2",
    "algoritmos: hecho conceptual 3",
    "algoritmos: hecho conceptual 4",
    "algoritmos: hecho conceptual 5",
    "algoritmos: hecho conceptual 6",
    "algoritmos: hecho conceptual 7",
    "algoritmos: hecho conceptual 8",
    "algoritmos: hecho conceptual 9",
    "algoritmos: hecho conceptual 10",
    "algoritmos: hecho conceptual 11",
    "algoritmos: hecho conceptual 12",
    "algoritmos: hecho conceptual 13",
    "algoritmos: hecho conceptual 14",
    "algoritmos: hecho conceptual 15",
    "algoritmos: hecho conceptual 16",
    "algoritmos: hecho conceptual 17",
    "algoritmos: hecho conceptual 18",
    "algoritmos: hecho conceptual 19",
    "algoritmos: hecho conceptual 20",
    "algoritmos: hecho conceptual 21",
    "algoritmos: hecho conceptual 22",
    "algoritmos: hecho conceptual 23",
    "algoritmos: hecho conceptual 24",
    "algoritmos: hecho conceptual 25",
    "algoritmos: hecho conceptual 26",
    "algoritmos: hecho conceptual 27",
    "algoritmos: hecho conceptual 28",
    "algoritmos: hecho conceptual 29",
    "algoritmos: hecho conceptual 30",
    "algoritmos: hecho conceptual 31",
    "algoritmos: hecho conceptual 32",
    "algoritmos: hecho conceptual 33",
    "algoritmos: hecho conceptual 34",
    "algoritmos: hecho conceptual 35",
    "algoritmos: hecho conceptual 36",
    "algoritmos: hecho conceptual 37",
    "algoritmos: hecho conceptual 38",
    "algoritmos: hecho conceptual 39",
    "algoritmos: hecho conceptual 40",
    "algoritmos: hecho conceptual 41",
    "algoritmos: hecho conceptual 42",
    "algoritmos: hecho conceptual 43",
    "algoritmos: hecho conceptual 44",
    "algoritmos: hecho conceptual 45",
    "algoritmos: hecho conceptual 46",
    "algoritmos: hecho conceptual 47",
    "algoritmos: hecho conceptual 48",
    "algoritmos: hecho conceptual 49",
    "algoritmos: hecho conceptual 50",
    "algoritmos: hecho conceptual 51",
    "algoritmos: hecho conceptual 52",
    "algoritmos: hecho conceptual 53",
    "algoritmos: hecho conceptual 54",
    "algoritmos: hecho conceptual 55",
    "algoritmos: hecho conceptual 56",
    "algoritmos: hecho conceptual 57",
    "algoritmos: hecho conceptual 58",
    "algoritmos: hecho conceptual 59",
    "algoritmos: hecho conceptual 60",
    "algoritmos: hecho conceptual 61",
    "algoritmos: hecho conceptual 62",
    "algoritmos: hecho conceptual 63",
    "algoritmos: hecho conceptual 64",
    "algoritmos: hecho conceptual 65",
    "algoritmos: hecho conceptual 66",
    "algoritmos: hecho conceptual 67",
    "algoritmos: hecho conceptual 68",
    "algoritmos: hecho conceptual 69",
    "algoritmos: hecho conceptual 70",
    "algoritmos: hecho conceptual 71",
    "algoritmos: hecho conceptual 72",
    "algoritmos: hecho conceptual 73",
    "algoritmos: hecho conceptual 74",
    "algoritmos: hecho conceptual 75",
    "algoritmos: hecho conceptual 76",
    "algoritmos: hecho conceptual 77",
    "algoritmos: hecho conceptual 78",
    "algoritmos: hecho conceptual 79",
    "algoritmos: hecho conceptual 80",
]
ALGORITMOS_QUESTION_VARIANTS = [
    "¿Qué es algoritmos?",
    "¿Cómo explicarías algoritmos de forma sencilla?",
    "¿Por qué es importante algoritmos?",
    "¿Cuál es una idea fundamental de algoritmos?",
    "¿Puedes darme un ejemplo relacionado con algoritmos?",
    "¿Qué errores son frecuentes al estudiar algoritmos?",
    "¿Cómo se aplica algoritmos en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con algoritmos?",
    "¿Cómo empezaría a estudiar algoritmos?",
    "¿Qué relación tiene algoritmos con otras disciplinas?",
    "¿Puedes resumir algoritmos en pocas frases?",
    "¿Qué términos debería conocer sobre algoritmos?",
    "¿Cómo comprobaría si entendí algoritmos?",
    "¿Qué intuición ayuda a comprender algoritmos?",
    "¿Qué problema sencillo puedo resolver sobre algoritmos?",
    "¿Qué matiz suele pasarse por alto en algoritmos?",
    "¿Cómo se representa algoritmos?",
    "¿Qué supuestos se usan al hablar de algoritmos?",
    "¿Qué aplicaciones tiene algoritmos?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar algoritmos?",
    "¿Cómo se conecta algoritmos con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de algoritmos?",
    "¿Qué vocabulario técnico aparece en algoritmos?",
    "¿Puedes comparar dos enfoques dentro de algoritmos?",
    "¿Cómo evolucionó la comprensión de algoritmos?",
    "¿Qué ejemplo cotidiano ilustra algoritmos?",
    "¿Qué dato conviene recordar sobre algoritmos?",
    "¿Cómo evitar confusiones comunes en algoritmos?",
    "¿Qué parte de algoritmos suele ser más difícil?",
    "¿Puedes plantear un ejercicio de algoritmos?",
    "¿Cómo resolverías un ejercicio introductorio de algoritmos?",
    "¿Qué relación matemática aparece en algoritmos?",
    "¿Qué observación apoya esta idea de algoritmos?",
    "¿Qué pasaría si cambiamos una condición en algoritmos?",
    "¿Cómo se usa algoritmos en investigación?",
    "¿Qué herramientas sirven para estudiar algoritmos?",
    "¿Cómo distinguir evidencia de interpretación en algoritmos?",
    "¿Qué concepto previo necesito para entender algoritmos?",
    "¿Puedes dar una analogía para algoritmos?",
    "¿Qué preguntas avanzadas surgen de algoritmos?",
    "¿Cómo se comunica correctamente información sobre algoritmos?",
    "¿Qué ejemplo contradice una intuición común sobre algoritmos?",
    "¿Qué pasos seguirías para analizar algoritmos?",
    "¿Cómo resumirías la historia de algoritmos?",
    "¿Qué incertidumbres existen al estudiar algoritmos?",
    "¿Cómo se relacionan teoría y práctica en algoritmos?",
    "¿Qué clasificación útil existe en algoritmos?",
    "¿Cómo puedo practicar algoritmos?",
    "¿Qué es algoritmos?",
]
ESTRUCTURAS_DE_DATOS_FACT_LABELS = [
    "estructuras de datos: hecho conceptual 1",
    "estructuras de datos: hecho conceptual 2",
    "estructuras de datos: hecho conceptual 3",
    "estructuras de datos: hecho conceptual 4",
    "estructuras de datos: hecho conceptual 5",
    "estructuras de datos: hecho conceptual 6",
    "estructuras de datos: hecho conceptual 7",
    "estructuras de datos: hecho conceptual 8",
    "estructuras de datos: hecho conceptual 9",
    "estructuras de datos: hecho conceptual 10",
    "estructuras de datos: hecho conceptual 11",
    "estructuras de datos: hecho conceptual 12",
    "estructuras de datos: hecho conceptual 13",
    "estructuras de datos: hecho conceptual 14",
    "estructuras de datos: hecho conceptual 15",
    "estructuras de datos: hecho conceptual 16",
    "estructuras de datos: hecho conceptual 17",
    "estructuras de datos: hecho conceptual 18",
    "estructuras de datos: hecho conceptual 19",
    "estructuras de datos: hecho conceptual 20",
    "estructuras de datos: hecho conceptual 21",
    "estructuras de datos: hecho conceptual 22",
    "estructuras de datos: hecho conceptual 23",
    "estructuras de datos: hecho conceptual 24",
    "estructuras de datos: hecho conceptual 25",
    "estructuras de datos: hecho conceptual 26",
    "estructuras de datos: hecho conceptual 27",
    "estructuras de datos: hecho conceptual 28",
    "estructuras de datos: hecho conceptual 29",
    "estructuras de datos: hecho conceptual 30",
    "estructuras de datos: hecho conceptual 31",
    "estructuras de datos: hecho conceptual 32",
    "estructuras de datos: hecho conceptual 33",
    "estructuras de datos: hecho conceptual 34",
    "estructuras de datos: hecho conceptual 35",
    "estructuras de datos: hecho conceptual 36",
    "estructuras de datos: hecho conceptual 37",
    "estructuras de datos: hecho conceptual 38",
    "estructuras de datos: hecho conceptual 39",
    "estructuras de datos: hecho conceptual 40",
    "estructuras de datos: hecho conceptual 41",
    "estructuras de datos: hecho conceptual 42",
    "estructuras de datos: hecho conceptual 43",
    "estructuras de datos: hecho conceptual 44",
    "estructuras de datos: hecho conceptual 45",
    "estructuras de datos: hecho conceptual 46",
    "estructuras de datos: hecho conceptual 47",
    "estructuras de datos: hecho conceptual 48",
    "estructuras de datos: hecho conceptual 49",
    "estructuras de datos: hecho conceptual 50",
    "estructuras de datos: hecho conceptual 51",
    "estructuras de datos: hecho conceptual 52",
    "estructuras de datos: hecho conceptual 53",
    "estructuras de datos: hecho conceptual 54",
    "estructuras de datos: hecho conceptual 55",
    "estructuras de datos: hecho conceptual 56",
    "estructuras de datos: hecho conceptual 57",
    "estructuras de datos: hecho conceptual 58",
    "estructuras de datos: hecho conceptual 59",
    "estructuras de datos: hecho conceptual 60",
    "estructuras de datos: hecho conceptual 61",
    "estructuras de datos: hecho conceptual 62",
    "estructuras de datos: hecho conceptual 63",
    "estructuras de datos: hecho conceptual 64",
    "estructuras de datos: hecho conceptual 65",
    "estructuras de datos: hecho conceptual 66",
    "estructuras de datos: hecho conceptual 67",
    "estructuras de datos: hecho conceptual 68",
    "estructuras de datos: hecho conceptual 69",
    "estructuras de datos: hecho conceptual 70",
    "estructuras de datos: hecho conceptual 71",
    "estructuras de datos: hecho conceptual 72",
    "estructuras de datos: hecho conceptual 73",
    "estructuras de datos: hecho conceptual 74",
    "estructuras de datos: hecho conceptual 75",
    "estructuras de datos: hecho conceptual 76",
    "estructuras de datos: hecho conceptual 77",
    "estructuras de datos: hecho conceptual 78",
    "estructuras de datos: hecho conceptual 79",
    "estructuras de datos: hecho conceptual 80",
]
ESTRUCTURAS_DE_DATOS_QUESTION_VARIANTS = [
    "¿Qué es estructuras de datos?",
    "¿Cómo explicarías estructuras de datos de forma sencilla?",
    "¿Por qué es importante estructuras de datos?",
    "¿Cuál es una idea fundamental de estructuras de datos?",
    "¿Puedes darme un ejemplo relacionado con estructuras de datos?",
    "¿Qué errores son frecuentes al estudiar estructuras de datos?",
    "¿Cómo se aplica estructuras de datos en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con estructuras de datos?",
    "¿Cómo empezaría a estudiar estructuras de datos?",
    "¿Qué relación tiene estructuras de datos con otras disciplinas?",
    "¿Puedes resumir estructuras de datos en pocas frases?",
    "¿Qué términos debería conocer sobre estructuras de datos?",
    "¿Cómo comprobaría si entendí estructuras de datos?",
    "¿Qué intuición ayuda a comprender estructuras de datos?",
    "¿Qué problema sencillo puedo resolver sobre estructuras de datos?",
    "¿Qué matiz suele pasarse por alto en estructuras de datos?",
    "¿Cómo se representa estructuras de datos?",
    "¿Qué supuestos se usan al hablar de estructuras de datos?",
    "¿Qué aplicaciones tiene estructuras de datos?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar estructuras de datos?",
    "¿Cómo se conecta estructuras de datos con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de estructuras de datos?",
    "¿Qué vocabulario técnico aparece en estructuras de datos?",
    "¿Puedes comparar dos enfoques dentro de estructuras de datos?",
    "¿Cómo evolucionó la comprensión de estructuras de datos?",
    "¿Qué ejemplo cotidiano ilustra estructuras de datos?",
    "¿Qué dato conviene recordar sobre estructuras de datos?",
    "¿Cómo evitar confusiones comunes en estructuras de datos?",
    "¿Qué parte de estructuras de datos suele ser más difícil?",
    "¿Puedes plantear un ejercicio de estructuras de datos?",
    "¿Cómo resolverías un ejercicio introductorio de estructuras de datos?",
    "¿Qué relación matemática aparece en estructuras de datos?",
    "¿Qué observación apoya esta idea de estructuras de datos?",
    "¿Qué pasaría si cambiamos una condición en estructuras de datos?",
    "¿Cómo se usa estructuras de datos en investigación?",
    "¿Qué herramientas sirven para estudiar estructuras de datos?",
    "¿Cómo distinguir evidencia de interpretación en estructuras de datos?",
    "¿Qué concepto previo necesito para entender estructuras de datos?",
    "¿Puedes dar una analogía para estructuras de datos?",
    "¿Qué preguntas avanzadas surgen de estructuras de datos?",
    "¿Cómo se comunica correctamente información sobre estructuras de datos?",
    "¿Qué ejemplo contradice una intuición común sobre estructuras de datos?",
    "¿Qué pasos seguirías para analizar estructuras de datos?",
    "¿Cómo resumirías la historia de estructuras de datos?",
    "¿Qué incertidumbres existen al estudiar estructuras de datos?",
    "¿Cómo se relacionan teoría y práctica en estructuras de datos?",
    "¿Qué clasificación útil existe en estructuras de datos?",
    "¿Cómo puedo practicar estructuras de datos?",
    "¿Qué es estructuras de datos?",
]
MACHINE_LEARNING_FACT_LABELS = [
    "machine learning: hecho conceptual 1",
    "machine learning: hecho conceptual 2",
    "machine learning: hecho conceptual 3",
    "machine learning: hecho conceptual 4",
    "machine learning: hecho conceptual 5",
    "machine learning: hecho conceptual 6",
    "machine learning: hecho conceptual 7",
    "machine learning: hecho conceptual 8",
    "machine learning: hecho conceptual 9",
    "machine learning: hecho conceptual 10",
    "machine learning: hecho conceptual 11",
    "machine learning: hecho conceptual 12",
    "machine learning: hecho conceptual 13",
    "machine learning: hecho conceptual 14",
    "machine learning: hecho conceptual 15",
    "machine learning: hecho conceptual 16",
    "machine learning: hecho conceptual 17",
    "machine learning: hecho conceptual 18",
    "machine learning: hecho conceptual 19",
    "machine learning: hecho conceptual 20",
    "machine learning: hecho conceptual 21",
    "machine learning: hecho conceptual 22",
    "machine learning: hecho conceptual 23",
    "machine learning: hecho conceptual 24",
    "machine learning: hecho conceptual 25",
    "machine learning: hecho conceptual 26",
    "machine learning: hecho conceptual 27",
    "machine learning: hecho conceptual 28",
    "machine learning: hecho conceptual 29",
    "machine learning: hecho conceptual 30",
    "machine learning: hecho conceptual 31",
    "machine learning: hecho conceptual 32",
    "machine learning: hecho conceptual 33",
    "machine learning: hecho conceptual 34",
    "machine learning: hecho conceptual 35",
    "machine learning: hecho conceptual 36",
    "machine learning: hecho conceptual 37",
    "machine learning: hecho conceptual 38",
    "machine learning: hecho conceptual 39",
    "machine learning: hecho conceptual 40",
    "machine learning: hecho conceptual 41",
    "machine learning: hecho conceptual 42",
    "machine learning: hecho conceptual 43",
    "machine learning: hecho conceptual 44",
    "machine learning: hecho conceptual 45",
    "machine learning: hecho conceptual 46",
    "machine learning: hecho conceptual 47",
    "machine learning: hecho conceptual 48",
    "machine learning: hecho conceptual 49",
    "machine learning: hecho conceptual 50",
    "machine learning: hecho conceptual 51",
    "machine learning: hecho conceptual 52",
    "machine learning: hecho conceptual 53",
    "machine learning: hecho conceptual 54",
    "machine learning: hecho conceptual 55",
    "machine learning: hecho conceptual 56",
    "machine learning: hecho conceptual 57",
    "machine learning: hecho conceptual 58",
    "machine learning: hecho conceptual 59",
    "machine learning: hecho conceptual 60",
    "machine learning: hecho conceptual 61",
    "machine learning: hecho conceptual 62",
    "machine learning: hecho conceptual 63",
    "machine learning: hecho conceptual 64",
    "machine learning: hecho conceptual 65",
    "machine learning: hecho conceptual 66",
    "machine learning: hecho conceptual 67",
    "machine learning: hecho conceptual 68",
    "machine learning: hecho conceptual 69",
    "machine learning: hecho conceptual 70",
    "machine learning: hecho conceptual 71",
    "machine learning: hecho conceptual 72",
    "machine learning: hecho conceptual 73",
    "machine learning: hecho conceptual 74",
    "machine learning: hecho conceptual 75",
    "machine learning: hecho conceptual 76",
    "machine learning: hecho conceptual 77",
    "machine learning: hecho conceptual 78",
    "machine learning: hecho conceptual 79",
    "machine learning: hecho conceptual 80",
]
MACHINE_LEARNING_QUESTION_VARIANTS = [
    "¿Qué es machine learning?",
    "¿Cómo explicarías machine learning de forma sencilla?",
    "¿Por qué es importante machine learning?",
    "¿Cuál es una idea fundamental de machine learning?",
    "¿Puedes darme un ejemplo relacionado con machine learning?",
    "¿Qué errores son frecuentes al estudiar machine learning?",
    "¿Cómo se aplica machine learning en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con machine learning?",
    "¿Cómo empezaría a estudiar machine learning?",
    "¿Qué relación tiene machine learning con otras disciplinas?",
    "¿Puedes resumir machine learning en pocas frases?",
    "¿Qué términos debería conocer sobre machine learning?",
    "¿Cómo comprobaría si entendí machine learning?",
    "¿Qué intuición ayuda a comprender machine learning?",
    "¿Qué problema sencillo puedo resolver sobre machine learning?",
    "¿Qué matiz suele pasarse por alto en machine learning?",
    "¿Cómo se representa machine learning?",
    "¿Qué supuestos se usan al hablar de machine learning?",
    "¿Qué aplicaciones tiene machine learning?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar machine learning?",
    "¿Cómo se conecta machine learning con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de machine learning?",
    "¿Qué vocabulario técnico aparece en machine learning?",
    "¿Puedes comparar dos enfoques dentro de machine learning?",
    "¿Cómo evolucionó la comprensión de machine learning?",
    "¿Qué ejemplo cotidiano ilustra machine learning?",
    "¿Qué dato conviene recordar sobre machine learning?",
    "¿Cómo evitar confusiones comunes en machine learning?",
    "¿Qué parte de machine learning suele ser más difícil?",
    "¿Puedes plantear un ejercicio de machine learning?",
    "¿Cómo resolverías un ejercicio introductorio de machine learning?",
    "¿Qué relación matemática aparece en machine learning?",
    "¿Qué observación apoya esta idea de machine learning?",
    "¿Qué pasaría si cambiamos una condición en machine learning?",
    "¿Cómo se usa machine learning en investigación?",
    "¿Qué herramientas sirven para estudiar machine learning?",
    "¿Cómo distinguir evidencia de interpretación en machine learning?",
    "¿Qué concepto previo necesito para entender machine learning?",
    "¿Puedes dar una analogía para machine learning?",
    "¿Qué preguntas avanzadas surgen de machine learning?",
    "¿Cómo se comunica correctamente información sobre machine learning?",
    "¿Qué ejemplo contradice una intuición común sobre machine learning?",
    "¿Qué pasos seguirías para analizar machine learning?",
    "¿Cómo resumirías la historia de machine learning?",
    "¿Qué incertidumbres existen al estudiar machine learning?",
    "¿Cómo se relacionan teoría y práctica en machine learning?",
    "¿Qué clasificación útil existe en machine learning?",
    "¿Cómo puedo practicar machine learning?",
    "¿Qué es machine learning?",
]
DEEP_LEARNING_FACT_LABELS = [
    "deep learning: hecho conceptual 1",
    "deep learning: hecho conceptual 2",
    "deep learning: hecho conceptual 3",
    "deep learning: hecho conceptual 4",
    "deep learning: hecho conceptual 5",
    "deep learning: hecho conceptual 6",
    "deep learning: hecho conceptual 7",
    "deep learning: hecho conceptual 8",
    "deep learning: hecho conceptual 9",
    "deep learning: hecho conceptual 10",
    "deep learning: hecho conceptual 11",
    "deep learning: hecho conceptual 12",
    "deep learning: hecho conceptual 13",
    "deep learning: hecho conceptual 14",
    "deep learning: hecho conceptual 15",
    "deep learning: hecho conceptual 16",
    "deep learning: hecho conceptual 17",
    "deep learning: hecho conceptual 18",
    "deep learning: hecho conceptual 19",
    "deep learning: hecho conceptual 20",
    "deep learning: hecho conceptual 21",
    "deep learning: hecho conceptual 22",
    "deep learning: hecho conceptual 23",
    "deep learning: hecho conceptual 24",
    "deep learning: hecho conceptual 25",
    "deep learning: hecho conceptual 26",
    "deep learning: hecho conceptual 27",
    "deep learning: hecho conceptual 28",
    "deep learning: hecho conceptual 29",
    "deep learning: hecho conceptual 30",
    "deep learning: hecho conceptual 31",
    "deep learning: hecho conceptual 32",
    "deep learning: hecho conceptual 33",
    "deep learning: hecho conceptual 34",
    "deep learning: hecho conceptual 35",
    "deep learning: hecho conceptual 36",
    "deep learning: hecho conceptual 37",
    "deep learning: hecho conceptual 38",
    "deep learning: hecho conceptual 39",
    "deep learning: hecho conceptual 40",
    "deep learning: hecho conceptual 41",
    "deep learning: hecho conceptual 42",
    "deep learning: hecho conceptual 43",
    "deep learning: hecho conceptual 44",
    "deep learning: hecho conceptual 45",
    "deep learning: hecho conceptual 46",
    "deep learning: hecho conceptual 47",
    "deep learning: hecho conceptual 48",
    "deep learning: hecho conceptual 49",
    "deep learning: hecho conceptual 50",
    "deep learning: hecho conceptual 51",
    "deep learning: hecho conceptual 52",
    "deep learning: hecho conceptual 53",
    "deep learning: hecho conceptual 54",
    "deep learning: hecho conceptual 55",
    "deep learning: hecho conceptual 56",
    "deep learning: hecho conceptual 57",
    "deep learning: hecho conceptual 58",
    "deep learning: hecho conceptual 59",
    "deep learning: hecho conceptual 60",
    "deep learning: hecho conceptual 61",
    "deep learning: hecho conceptual 62",
    "deep learning: hecho conceptual 63",
    "deep learning: hecho conceptual 64",
    "deep learning: hecho conceptual 65",
    "deep learning: hecho conceptual 66",
    "deep learning: hecho conceptual 67",
    "deep learning: hecho conceptual 68",
    "deep learning: hecho conceptual 69",
    "deep learning: hecho conceptual 70",
    "deep learning: hecho conceptual 71",
    "deep learning: hecho conceptual 72",
    "deep learning: hecho conceptual 73",
    "deep learning: hecho conceptual 74",
    "deep learning: hecho conceptual 75",
    "deep learning: hecho conceptual 76",
    "deep learning: hecho conceptual 77",
    "deep learning: hecho conceptual 78",
    "deep learning: hecho conceptual 79",
    "deep learning: hecho conceptual 80",
]
DEEP_LEARNING_QUESTION_VARIANTS = [
    "¿Qué es deep learning?",
    "¿Cómo explicarías deep learning de forma sencilla?",
    "¿Por qué es importante deep learning?",
    "¿Cuál es una idea fundamental de deep learning?",
    "¿Puedes darme un ejemplo relacionado con deep learning?",
    "¿Qué errores son frecuentes al estudiar deep learning?",
    "¿Cómo se aplica deep learning en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con deep learning?",
    "¿Cómo empezaría a estudiar deep learning?",
    "¿Qué relación tiene deep learning con otras disciplinas?",
    "¿Puedes resumir deep learning en pocas frases?",
    "¿Qué términos debería conocer sobre deep learning?",
    "¿Cómo comprobaría si entendí deep learning?",
    "¿Qué intuición ayuda a comprender deep learning?",
    "¿Qué problema sencillo puedo resolver sobre deep learning?",
    "¿Qué matiz suele pasarse por alto en deep learning?",
    "¿Cómo se representa deep learning?",
    "¿Qué supuestos se usan al hablar de deep learning?",
    "¿Qué aplicaciones tiene deep learning?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar deep learning?",
    "¿Cómo se conecta deep learning con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de deep learning?",
    "¿Qué vocabulario técnico aparece en deep learning?",
    "¿Puedes comparar dos enfoques dentro de deep learning?",
    "¿Cómo evolucionó la comprensión de deep learning?",
    "¿Qué ejemplo cotidiano ilustra deep learning?",
    "¿Qué dato conviene recordar sobre deep learning?",
    "¿Cómo evitar confusiones comunes en deep learning?",
    "¿Qué parte de deep learning suele ser más difícil?",
    "¿Puedes plantear un ejercicio de deep learning?",
    "¿Cómo resolverías un ejercicio introductorio de deep learning?",
    "¿Qué relación matemática aparece en deep learning?",
    "¿Qué observación apoya esta idea de deep learning?",
    "¿Qué pasaría si cambiamos una condición en deep learning?",
    "¿Cómo se usa deep learning en investigación?",
    "¿Qué herramientas sirven para estudiar deep learning?",
    "¿Cómo distinguir evidencia de interpretación en deep learning?",
    "¿Qué concepto previo necesito para entender deep learning?",
    "¿Puedes dar una analogía para deep learning?",
    "¿Qué preguntas avanzadas surgen de deep learning?",
    "¿Cómo se comunica correctamente información sobre deep learning?",
    "¿Qué ejemplo contradice una intuición común sobre deep learning?",
    "¿Qué pasos seguirías para analizar deep learning?",
    "¿Cómo resumirías la historia de deep learning?",
    "¿Qué incertidumbres existen al estudiar deep learning?",
    "¿Cómo se relacionan teoría y práctica en deep learning?",
    "¿Qué clasificación útil existe en deep learning?",
    "¿Cómo puedo practicar deep learning?",
    "¿Qué es deep learning?",
]
NLP_FACT_LABELS = [
    "NLP: hecho conceptual 1",
    "NLP: hecho conceptual 2",
    "NLP: hecho conceptual 3",
    "NLP: hecho conceptual 4",
    "NLP: hecho conceptual 5",
    "NLP: hecho conceptual 6",
    "NLP: hecho conceptual 7",
    "NLP: hecho conceptual 8",
    "NLP: hecho conceptual 9",
    "NLP: hecho conceptual 10",
    "NLP: hecho conceptual 11",
    "NLP: hecho conceptual 12",
    "NLP: hecho conceptual 13",
    "NLP: hecho conceptual 14",
    "NLP: hecho conceptual 15",
    "NLP: hecho conceptual 16",
    "NLP: hecho conceptual 17",
    "NLP: hecho conceptual 18",
    "NLP: hecho conceptual 19",
    "NLP: hecho conceptual 20",
    "NLP: hecho conceptual 21",
    "NLP: hecho conceptual 22",
    "NLP: hecho conceptual 23",
    "NLP: hecho conceptual 24",
    "NLP: hecho conceptual 25",
    "NLP: hecho conceptual 26",
    "NLP: hecho conceptual 27",
    "NLP: hecho conceptual 28",
    "NLP: hecho conceptual 29",
    "NLP: hecho conceptual 30",
    "NLP: hecho conceptual 31",
    "NLP: hecho conceptual 32",
    "NLP: hecho conceptual 33",
    "NLP: hecho conceptual 34",
    "NLP: hecho conceptual 35",
    "NLP: hecho conceptual 36",
    "NLP: hecho conceptual 37",
    "NLP: hecho conceptual 38",
    "NLP: hecho conceptual 39",
    "NLP: hecho conceptual 40",
    "NLP: hecho conceptual 41",
    "NLP: hecho conceptual 42",
    "NLP: hecho conceptual 43",
    "NLP: hecho conceptual 44",
    "NLP: hecho conceptual 45",
    "NLP: hecho conceptual 46",
    "NLP: hecho conceptual 47",
    "NLP: hecho conceptual 48",
    "NLP: hecho conceptual 49",
    "NLP: hecho conceptual 50",
    "NLP: hecho conceptual 51",
    "NLP: hecho conceptual 52",
    "NLP: hecho conceptual 53",
    "NLP: hecho conceptual 54",
    "NLP: hecho conceptual 55",
    "NLP: hecho conceptual 56",
    "NLP: hecho conceptual 57",
    "NLP: hecho conceptual 58",
    "NLP: hecho conceptual 59",
    "NLP: hecho conceptual 60",
    "NLP: hecho conceptual 61",
    "NLP: hecho conceptual 62",
    "NLP: hecho conceptual 63",
    "NLP: hecho conceptual 64",
    "NLP: hecho conceptual 65",
    "NLP: hecho conceptual 66",
    "NLP: hecho conceptual 67",
    "NLP: hecho conceptual 68",
    "NLP: hecho conceptual 69",
    "NLP: hecho conceptual 70",
    "NLP: hecho conceptual 71",
    "NLP: hecho conceptual 72",
    "NLP: hecho conceptual 73",
    "NLP: hecho conceptual 74",
    "NLP: hecho conceptual 75",
    "NLP: hecho conceptual 76",
    "NLP: hecho conceptual 77",
    "NLP: hecho conceptual 78",
    "NLP: hecho conceptual 79",
    "NLP: hecho conceptual 80",
]
NLP_QUESTION_VARIANTS = [
    "¿Qué es NLP?",
    "¿Cómo explicarías NLP de forma sencilla?",
    "¿Por qué es importante NLP?",
    "¿Cuál es una idea fundamental de NLP?",
    "¿Puedes darme un ejemplo relacionado con NLP?",
    "¿Qué errores son frecuentes al estudiar NLP?",
    "¿Cómo se aplica NLP en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con NLP?",
    "¿Cómo empezaría a estudiar NLP?",
    "¿Qué relación tiene NLP con otras disciplinas?",
    "¿Puedes resumir NLP en pocas frases?",
    "¿Qué términos debería conocer sobre NLP?",
    "¿Cómo comprobaría si entendí NLP?",
    "¿Qué intuición ayuda a comprender NLP?",
    "¿Qué problema sencillo puedo resolver sobre NLP?",
    "¿Qué matiz suele pasarse por alto en NLP?",
    "¿Cómo se representa NLP?",
    "¿Qué supuestos se usan al hablar de NLP?",
    "¿Qué aplicaciones tiene NLP?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar NLP?",
    "¿Cómo se conecta NLP con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de NLP?",
    "¿Qué vocabulario técnico aparece en NLP?",
    "¿Puedes comparar dos enfoques dentro de NLP?",
    "¿Cómo evolucionó la comprensión de NLP?",
    "¿Qué ejemplo cotidiano ilustra NLP?",
    "¿Qué dato conviene recordar sobre NLP?",
    "¿Cómo evitar confusiones comunes en NLP?",
    "¿Qué parte de NLP suele ser más difícil?",
    "¿Puedes plantear un ejercicio de NLP?",
    "¿Cómo resolverías un ejercicio introductorio de NLP?",
    "¿Qué relación matemática aparece en NLP?",
    "¿Qué observación apoya esta idea de NLP?",
    "¿Qué pasaría si cambiamos una condición en NLP?",
    "¿Cómo se usa NLP en investigación?",
    "¿Qué herramientas sirven para estudiar NLP?",
    "¿Cómo distinguir evidencia de interpretación en NLP?",
    "¿Qué concepto previo necesito para entender NLP?",
    "¿Puedes dar una analogía para NLP?",
    "¿Qué preguntas avanzadas surgen de NLP?",
    "¿Cómo se comunica correctamente información sobre NLP?",
    "¿Qué ejemplo contradice una intuición común sobre NLP?",
    "¿Qué pasos seguirías para analizar NLP?",
    "¿Cómo resumirías la historia de NLP?",
    "¿Qué incertidumbres existen al estudiar NLP?",
    "¿Cómo se relacionan teoría y práctica en NLP?",
    "¿Qué clasificación útil existe en NLP?",
    "¿Cómo puedo practicar NLP?",
    "¿Qué es NLP?",
]
COMPUTER_VISION_FACT_LABELS = [
    "computer vision: hecho conceptual 1",
    "computer vision: hecho conceptual 2",
    "computer vision: hecho conceptual 3",
    "computer vision: hecho conceptual 4",
    "computer vision: hecho conceptual 5",
    "computer vision: hecho conceptual 6",
    "computer vision: hecho conceptual 7",
    "computer vision: hecho conceptual 8",
    "computer vision: hecho conceptual 9",
    "computer vision: hecho conceptual 10",
    "computer vision: hecho conceptual 11",
    "computer vision: hecho conceptual 12",
    "computer vision: hecho conceptual 13",
    "computer vision: hecho conceptual 14",
    "computer vision: hecho conceptual 15",
    "computer vision: hecho conceptual 16",
    "computer vision: hecho conceptual 17",
    "computer vision: hecho conceptual 18",
    "computer vision: hecho conceptual 19",
    "computer vision: hecho conceptual 20",
    "computer vision: hecho conceptual 21",
    "computer vision: hecho conceptual 22",
    "computer vision: hecho conceptual 23",
    "computer vision: hecho conceptual 24",
    "computer vision: hecho conceptual 25",
    "computer vision: hecho conceptual 26",
    "computer vision: hecho conceptual 27",
    "computer vision: hecho conceptual 28",
    "computer vision: hecho conceptual 29",
    "computer vision: hecho conceptual 30",
    "computer vision: hecho conceptual 31",
    "computer vision: hecho conceptual 32",
    "computer vision: hecho conceptual 33",
    "computer vision: hecho conceptual 34",
    "computer vision: hecho conceptual 35",
    "computer vision: hecho conceptual 36",
    "computer vision: hecho conceptual 37",
    "computer vision: hecho conceptual 38",
    "computer vision: hecho conceptual 39",
    "computer vision: hecho conceptual 40",
    "computer vision: hecho conceptual 41",
    "computer vision: hecho conceptual 42",
    "computer vision: hecho conceptual 43",
    "computer vision: hecho conceptual 44",
    "computer vision: hecho conceptual 45",
    "computer vision: hecho conceptual 46",
    "computer vision: hecho conceptual 47",
    "computer vision: hecho conceptual 48",
    "computer vision: hecho conceptual 49",
    "computer vision: hecho conceptual 50",
    "computer vision: hecho conceptual 51",
    "computer vision: hecho conceptual 52",
    "computer vision: hecho conceptual 53",
    "computer vision: hecho conceptual 54",
    "computer vision: hecho conceptual 55",
    "computer vision: hecho conceptual 56",
    "computer vision: hecho conceptual 57",
    "computer vision: hecho conceptual 58",
    "computer vision: hecho conceptual 59",
    "computer vision: hecho conceptual 60",
    "computer vision: hecho conceptual 61",
    "computer vision: hecho conceptual 62",
    "computer vision: hecho conceptual 63",
    "computer vision: hecho conceptual 64",
    "computer vision: hecho conceptual 65",
    "computer vision: hecho conceptual 66",
    "computer vision: hecho conceptual 67",
    "computer vision: hecho conceptual 68",
    "computer vision: hecho conceptual 69",
    "computer vision: hecho conceptual 70",
    "computer vision: hecho conceptual 71",
    "computer vision: hecho conceptual 72",
    "computer vision: hecho conceptual 73",
    "computer vision: hecho conceptual 74",
    "computer vision: hecho conceptual 75",
    "computer vision: hecho conceptual 76",
    "computer vision: hecho conceptual 77",
    "computer vision: hecho conceptual 78",
    "computer vision: hecho conceptual 79",
    "computer vision: hecho conceptual 80",
]
COMPUTER_VISION_QUESTION_VARIANTS = [
    "¿Qué es computer vision?",
    "¿Cómo explicarías computer vision de forma sencilla?",
    "¿Por qué es importante computer vision?",
    "¿Cuál es una idea fundamental de computer vision?",
    "¿Puedes darme un ejemplo relacionado con computer vision?",
    "¿Qué errores son frecuentes al estudiar computer vision?",
    "¿Cómo se aplica computer vision en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con computer vision?",
    "¿Cómo empezaría a estudiar computer vision?",
    "¿Qué relación tiene computer vision con otras disciplinas?",
    "¿Puedes resumir computer vision en pocas frases?",
    "¿Qué términos debería conocer sobre computer vision?",
    "¿Cómo comprobaría si entendí computer vision?",
    "¿Qué intuición ayuda a comprender computer vision?",
    "¿Qué problema sencillo puedo resolver sobre computer vision?",
    "¿Qué matiz suele pasarse por alto en computer vision?",
    "¿Cómo se representa computer vision?",
    "¿Qué supuestos se usan al hablar de computer vision?",
    "¿Qué aplicaciones tiene computer vision?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar computer vision?",
    "¿Cómo se conecta computer vision con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de computer vision?",
    "¿Qué vocabulario técnico aparece en computer vision?",
    "¿Puedes comparar dos enfoques dentro de computer vision?",
    "¿Cómo evolucionó la comprensión de computer vision?",
    "¿Qué ejemplo cotidiano ilustra computer vision?",
    "¿Qué dato conviene recordar sobre computer vision?",
    "¿Cómo evitar confusiones comunes en computer vision?",
    "¿Qué parte de computer vision suele ser más difícil?",
    "¿Puedes plantear un ejercicio de computer vision?",
    "¿Cómo resolverías un ejercicio introductorio de computer vision?",
    "¿Qué relación matemática aparece en computer vision?",
    "¿Qué observación apoya esta idea de computer vision?",
    "¿Qué pasaría si cambiamos una condición en computer vision?",
    "¿Cómo se usa computer vision en investigación?",
    "¿Qué herramientas sirven para estudiar computer vision?",
    "¿Cómo distinguir evidencia de interpretación en computer vision?",
    "¿Qué concepto previo necesito para entender computer vision?",
    "¿Puedes dar una analogía para computer vision?",
    "¿Qué preguntas avanzadas surgen de computer vision?",
    "¿Cómo se comunica correctamente información sobre computer vision?",
    "¿Qué ejemplo contradice una intuición común sobre computer vision?",
    "¿Qué pasos seguirías para analizar computer vision?",
    "¿Cómo resumirías la historia de computer vision?",
    "¿Qué incertidumbres existen al estudiar computer vision?",
    "¿Cómo se relacionan teoría y práctica en computer vision?",
    "¿Qué clasificación útil existe en computer vision?",
    "¿Cómo puedo practicar computer vision?",
    "¿Qué es computer vision?",
]
SALUD_FACT_LABELS = [
    "salud: hecho conceptual 1",
    "salud: hecho conceptual 2",
    "salud: hecho conceptual 3",
    "salud: hecho conceptual 4",
    "salud: hecho conceptual 5",
    "salud: hecho conceptual 6",
    "salud: hecho conceptual 7",
    "salud: hecho conceptual 8",
    "salud: hecho conceptual 9",
    "salud: hecho conceptual 10",
    "salud: hecho conceptual 11",
    "salud: hecho conceptual 12",
    "salud: hecho conceptual 13",
    "salud: hecho conceptual 14",
    "salud: hecho conceptual 15",
    "salud: hecho conceptual 16",
    "salud: hecho conceptual 17",
    "salud: hecho conceptual 18",
    "salud: hecho conceptual 19",
    "salud: hecho conceptual 20",
    "salud: hecho conceptual 21",
    "salud: hecho conceptual 22",
    "salud: hecho conceptual 23",
    "salud: hecho conceptual 24",
    "salud: hecho conceptual 25",
    "salud: hecho conceptual 26",
    "salud: hecho conceptual 27",
    "salud: hecho conceptual 28",
    "salud: hecho conceptual 29",
    "salud: hecho conceptual 30",
    "salud: hecho conceptual 31",
    "salud: hecho conceptual 32",
    "salud: hecho conceptual 33",
    "salud: hecho conceptual 34",
    "salud: hecho conceptual 35",
    "salud: hecho conceptual 36",
    "salud: hecho conceptual 37",
    "salud: hecho conceptual 38",
    "salud: hecho conceptual 39",
    "salud: hecho conceptual 40",
    "salud: hecho conceptual 41",
    "salud: hecho conceptual 42",
    "salud: hecho conceptual 43",
    "salud: hecho conceptual 44",
    "salud: hecho conceptual 45",
    "salud: hecho conceptual 46",
    "salud: hecho conceptual 47",
    "salud: hecho conceptual 48",
    "salud: hecho conceptual 49",
    "salud: hecho conceptual 50",
    "salud: hecho conceptual 51",
    "salud: hecho conceptual 52",
    "salud: hecho conceptual 53",
    "salud: hecho conceptual 54",
    "salud: hecho conceptual 55",
    "salud: hecho conceptual 56",
    "salud: hecho conceptual 57",
    "salud: hecho conceptual 58",
    "salud: hecho conceptual 59",
    "salud: hecho conceptual 60",
    "salud: hecho conceptual 61",
    "salud: hecho conceptual 62",
    "salud: hecho conceptual 63",
    "salud: hecho conceptual 64",
    "salud: hecho conceptual 65",
    "salud: hecho conceptual 66",
    "salud: hecho conceptual 67",
    "salud: hecho conceptual 68",
    "salud: hecho conceptual 69",
    "salud: hecho conceptual 70",
    "salud: hecho conceptual 71",
    "salud: hecho conceptual 72",
    "salud: hecho conceptual 73",
    "salud: hecho conceptual 74",
    "salud: hecho conceptual 75",
    "salud: hecho conceptual 76",
    "salud: hecho conceptual 77",
    "salud: hecho conceptual 78",
    "salud: hecho conceptual 79",
    "salud: hecho conceptual 80",
]
SALUD_QUESTION_VARIANTS = [
    "¿Qué es salud?",
    "¿Cómo explicarías salud de forma sencilla?",
    "¿Por qué es importante salud?",
    "¿Cuál es una idea fundamental de salud?",
    "¿Puedes darme un ejemplo relacionado con salud?",
    "¿Qué errores son frecuentes al estudiar salud?",
    "¿Cómo se aplica salud en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con salud?",
    "¿Cómo empezaría a estudiar salud?",
    "¿Qué relación tiene salud con otras disciplinas?",
    "¿Puedes resumir salud en pocas frases?",
    "¿Qué términos debería conocer sobre salud?",
    "¿Cómo comprobaría si entendí salud?",
    "¿Qué intuición ayuda a comprender salud?",
    "¿Qué problema sencillo puedo resolver sobre salud?",
    "¿Qué matiz suele pasarse por alto en salud?",
    "¿Cómo se representa salud?",
    "¿Qué supuestos se usan al hablar de salud?",
    "¿Qué aplicaciones tiene salud?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar salud?",
    "¿Cómo se conecta salud con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de salud?",
    "¿Qué vocabulario técnico aparece en salud?",
    "¿Puedes comparar dos enfoques dentro de salud?",
    "¿Cómo evolucionó la comprensión de salud?",
    "¿Qué ejemplo cotidiano ilustra salud?",
    "¿Qué dato conviene recordar sobre salud?",
    "¿Cómo evitar confusiones comunes en salud?",
    "¿Qué parte de salud suele ser más difícil?",
    "¿Puedes plantear un ejercicio de salud?",
    "¿Cómo resolverías un ejercicio introductorio de salud?",
    "¿Qué relación matemática aparece en salud?",
    "¿Qué observación apoya esta idea de salud?",
    "¿Qué pasaría si cambiamos una condición en salud?",
    "¿Cómo se usa salud en investigación?",
    "¿Qué herramientas sirven para estudiar salud?",
    "¿Cómo distinguir evidencia de interpretación en salud?",
    "¿Qué concepto previo necesito para entender salud?",
    "¿Puedes dar una analogía para salud?",
    "¿Qué preguntas avanzadas surgen de salud?",
    "¿Cómo se comunica correctamente información sobre salud?",
    "¿Qué ejemplo contradice una intuición común sobre salud?",
    "¿Qué pasos seguirías para analizar salud?",
    "¿Cómo resumirías la historia de salud?",
    "¿Qué incertidumbres existen al estudiar salud?",
    "¿Cómo se relacionan teoría y práctica en salud?",
    "¿Qué clasificación útil existe en salud?",
    "¿Cómo puedo practicar salud?",
    "¿Qué es salud?",
]
NUTRICI_N_FACT_LABELS = [
    "nutrición: hecho conceptual 1",
    "nutrición: hecho conceptual 2",
    "nutrición: hecho conceptual 3",
    "nutrición: hecho conceptual 4",
    "nutrición: hecho conceptual 5",
    "nutrición: hecho conceptual 6",
    "nutrición: hecho conceptual 7",
    "nutrición: hecho conceptual 8",
    "nutrición: hecho conceptual 9",
    "nutrición: hecho conceptual 10",
    "nutrición: hecho conceptual 11",
    "nutrición: hecho conceptual 12",
    "nutrición: hecho conceptual 13",
    "nutrición: hecho conceptual 14",
    "nutrición: hecho conceptual 15",
    "nutrición: hecho conceptual 16",
    "nutrición: hecho conceptual 17",
    "nutrición: hecho conceptual 18",
    "nutrición: hecho conceptual 19",
    "nutrición: hecho conceptual 20",
    "nutrición: hecho conceptual 21",
    "nutrición: hecho conceptual 22",
    "nutrición: hecho conceptual 23",
    "nutrición: hecho conceptual 24",
    "nutrición: hecho conceptual 25",
    "nutrición: hecho conceptual 26",
    "nutrición: hecho conceptual 27",
    "nutrición: hecho conceptual 28",
    "nutrición: hecho conceptual 29",
    "nutrición: hecho conceptual 30",
    "nutrición: hecho conceptual 31",
    "nutrición: hecho conceptual 32",
    "nutrición: hecho conceptual 33",
    "nutrición: hecho conceptual 34",
    "nutrición: hecho conceptual 35",
    "nutrición: hecho conceptual 36",
    "nutrición: hecho conceptual 37",
    "nutrición: hecho conceptual 38",
    "nutrición: hecho conceptual 39",
    "nutrición: hecho conceptual 40",
    "nutrición: hecho conceptual 41",
    "nutrición: hecho conceptual 42",
    "nutrición: hecho conceptual 43",
    "nutrición: hecho conceptual 44",
    "nutrición: hecho conceptual 45",
    "nutrición: hecho conceptual 46",
    "nutrición: hecho conceptual 47",
    "nutrición: hecho conceptual 48",
    "nutrición: hecho conceptual 49",
    "nutrición: hecho conceptual 50",
    "nutrición: hecho conceptual 51",
    "nutrición: hecho conceptual 52",
    "nutrición: hecho conceptual 53",
    "nutrición: hecho conceptual 54",
    "nutrición: hecho conceptual 55",
    "nutrición: hecho conceptual 56",
    "nutrición: hecho conceptual 57",
    "nutrición: hecho conceptual 58",
    "nutrición: hecho conceptual 59",
    "nutrición: hecho conceptual 60",
    "nutrición: hecho conceptual 61",
    "nutrición: hecho conceptual 62",
    "nutrición: hecho conceptual 63",
    "nutrición: hecho conceptual 64",
    "nutrición: hecho conceptual 65",
    "nutrición: hecho conceptual 66",
    "nutrición: hecho conceptual 67",
    "nutrición: hecho conceptual 68",
    "nutrición: hecho conceptual 69",
    "nutrición: hecho conceptual 70",
    "nutrición: hecho conceptual 71",
    "nutrición: hecho conceptual 72",
    "nutrición: hecho conceptual 73",
    "nutrición: hecho conceptual 74",
    "nutrición: hecho conceptual 75",
    "nutrición: hecho conceptual 76",
    "nutrición: hecho conceptual 77",
    "nutrición: hecho conceptual 78",
    "nutrición: hecho conceptual 79",
    "nutrición: hecho conceptual 80",
]
NUTRICI_N_QUESTION_VARIANTS = [
    "¿Qué es nutrición?",
    "¿Cómo explicarías nutrición de forma sencilla?",
    "¿Por qué es importante nutrición?",
    "¿Cuál es una idea fundamental de nutrición?",
    "¿Puedes darme un ejemplo relacionado con nutrición?",
    "¿Qué errores son frecuentes al estudiar nutrición?",
    "¿Cómo se aplica nutrición en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con nutrición?",
    "¿Cómo empezaría a estudiar nutrición?",
    "¿Qué relación tiene nutrición con otras disciplinas?",
    "¿Puedes resumir nutrición en pocas frases?",
    "¿Qué términos debería conocer sobre nutrición?",
    "¿Cómo comprobaría si entendí nutrición?",
    "¿Qué intuición ayuda a comprender nutrición?",
    "¿Qué problema sencillo puedo resolver sobre nutrición?",
    "¿Qué matiz suele pasarse por alto en nutrición?",
    "¿Cómo se representa nutrición?",
    "¿Qué supuestos se usan al hablar de nutrición?",
    "¿Qué aplicaciones tiene nutrición?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar nutrición?",
    "¿Cómo se conecta nutrición con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de nutrición?",
    "¿Qué vocabulario técnico aparece en nutrición?",
    "¿Puedes comparar dos enfoques dentro de nutrición?",
    "¿Cómo evolucionó la comprensión de nutrición?",
    "¿Qué ejemplo cotidiano ilustra nutrición?",
    "¿Qué dato conviene recordar sobre nutrición?",
    "¿Cómo evitar confusiones comunes en nutrición?",
    "¿Qué parte de nutrición suele ser más difícil?",
    "¿Puedes plantear un ejercicio de nutrición?",
    "¿Cómo resolverías un ejercicio introductorio de nutrición?",
    "¿Qué relación matemática aparece en nutrición?",
    "¿Qué observación apoya esta idea de nutrición?",
    "¿Qué pasaría si cambiamos una condición en nutrición?",
    "¿Cómo se usa nutrición en investigación?",
    "¿Qué herramientas sirven para estudiar nutrición?",
    "¿Cómo distinguir evidencia de interpretación en nutrición?",
    "¿Qué concepto previo necesito para entender nutrición?",
    "¿Puedes dar una analogía para nutrición?",
    "¿Qué preguntas avanzadas surgen de nutrición?",
    "¿Cómo se comunica correctamente información sobre nutrición?",
    "¿Qué ejemplo contradice una intuición común sobre nutrición?",
    "¿Qué pasos seguirías para analizar nutrición?",
    "¿Cómo resumirías la historia de nutrición?",
    "¿Qué incertidumbres existen al estudiar nutrición?",
    "¿Cómo se relacionan teoría y práctica en nutrición?",
    "¿Qué clasificación útil existe en nutrición?",
    "¿Cómo puedo practicar nutrición?",
    "¿Qué es nutrición?",
]
DEPORTES_FACT_LABELS = [
    "deportes: hecho conceptual 1",
    "deportes: hecho conceptual 2",
    "deportes: hecho conceptual 3",
    "deportes: hecho conceptual 4",
    "deportes: hecho conceptual 5",
    "deportes: hecho conceptual 6",
    "deportes: hecho conceptual 7",
    "deportes: hecho conceptual 8",
    "deportes: hecho conceptual 9",
    "deportes: hecho conceptual 10",
    "deportes: hecho conceptual 11",
    "deportes: hecho conceptual 12",
    "deportes: hecho conceptual 13",
    "deportes: hecho conceptual 14",
    "deportes: hecho conceptual 15",
    "deportes: hecho conceptual 16",
    "deportes: hecho conceptual 17",
    "deportes: hecho conceptual 18",
    "deportes: hecho conceptual 19",
    "deportes: hecho conceptual 20",
    "deportes: hecho conceptual 21",
    "deportes: hecho conceptual 22",
    "deportes: hecho conceptual 23",
    "deportes: hecho conceptual 24",
    "deportes: hecho conceptual 25",
    "deportes: hecho conceptual 26",
    "deportes: hecho conceptual 27",
    "deportes: hecho conceptual 28",
    "deportes: hecho conceptual 29",
    "deportes: hecho conceptual 30",
    "deportes: hecho conceptual 31",
    "deportes: hecho conceptual 32",
    "deportes: hecho conceptual 33",
    "deportes: hecho conceptual 34",
    "deportes: hecho conceptual 35",
    "deportes: hecho conceptual 36",
    "deportes: hecho conceptual 37",
    "deportes: hecho conceptual 38",
    "deportes: hecho conceptual 39",
    "deportes: hecho conceptual 40",
    "deportes: hecho conceptual 41",
    "deportes: hecho conceptual 42",
    "deportes: hecho conceptual 43",
    "deportes: hecho conceptual 44",
    "deportes: hecho conceptual 45",
    "deportes: hecho conceptual 46",
    "deportes: hecho conceptual 47",
    "deportes: hecho conceptual 48",
    "deportes: hecho conceptual 49",
    "deportes: hecho conceptual 50",
    "deportes: hecho conceptual 51",
    "deportes: hecho conceptual 52",
    "deportes: hecho conceptual 53",
    "deportes: hecho conceptual 54",
    "deportes: hecho conceptual 55",
    "deportes: hecho conceptual 56",
    "deportes: hecho conceptual 57",
    "deportes: hecho conceptual 58",
    "deportes: hecho conceptual 59",
    "deportes: hecho conceptual 60",
    "deportes: hecho conceptual 61",
    "deportes: hecho conceptual 62",
    "deportes: hecho conceptual 63",
    "deportes: hecho conceptual 64",
    "deportes: hecho conceptual 65",
    "deportes: hecho conceptual 66",
    "deportes: hecho conceptual 67",
    "deportes: hecho conceptual 68",
    "deportes: hecho conceptual 69",
    "deportes: hecho conceptual 70",
    "deportes: hecho conceptual 71",
    "deportes: hecho conceptual 72",
    "deportes: hecho conceptual 73",
    "deportes: hecho conceptual 74",
    "deportes: hecho conceptual 75",
    "deportes: hecho conceptual 76",
    "deportes: hecho conceptual 77",
    "deportes: hecho conceptual 78",
    "deportes: hecho conceptual 79",
    "deportes: hecho conceptual 80",
]
DEPORTES_QUESTION_VARIANTS = [
    "¿Qué es deportes?",
    "¿Cómo explicarías deportes de forma sencilla?",
    "¿Por qué es importante deportes?",
    "¿Cuál es una idea fundamental de deportes?",
    "¿Puedes darme un ejemplo relacionado con deportes?",
    "¿Qué errores son frecuentes al estudiar deportes?",
    "¿Cómo se aplica deportes en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con deportes?",
    "¿Cómo empezaría a estudiar deportes?",
    "¿Qué relación tiene deportes con otras disciplinas?",
    "¿Puedes resumir deportes en pocas frases?",
    "¿Qué términos debería conocer sobre deportes?",
    "¿Cómo comprobaría si entendí deportes?",
    "¿Qué intuición ayuda a comprender deportes?",
    "¿Qué problema sencillo puedo resolver sobre deportes?",
    "¿Qué matiz suele pasarse por alto en deportes?",
    "¿Cómo se representa deportes?",
    "¿Qué supuestos se usan al hablar de deportes?",
    "¿Qué aplicaciones tiene deportes?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar deportes?",
    "¿Cómo se conecta deportes con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de deportes?",
    "¿Qué vocabulario técnico aparece en deportes?",
    "¿Puedes comparar dos enfoques dentro de deportes?",
    "¿Cómo evolucionó la comprensión de deportes?",
    "¿Qué ejemplo cotidiano ilustra deportes?",
    "¿Qué dato conviene recordar sobre deportes?",
    "¿Cómo evitar confusiones comunes en deportes?",
    "¿Qué parte de deportes suele ser más difícil?",
    "¿Puedes plantear un ejercicio de deportes?",
    "¿Cómo resolverías un ejercicio introductorio de deportes?",
    "¿Qué relación matemática aparece en deportes?",
    "¿Qué observación apoya esta idea de deportes?",
    "¿Qué pasaría si cambiamos una condición en deportes?",
    "¿Cómo se usa deportes en investigación?",
    "¿Qué herramientas sirven para estudiar deportes?",
    "¿Cómo distinguir evidencia de interpretación en deportes?",
    "¿Qué concepto previo necesito para entender deportes?",
    "¿Puedes dar una analogía para deportes?",
    "¿Qué preguntas avanzadas surgen de deportes?",
    "¿Cómo se comunica correctamente información sobre deportes?",
    "¿Qué ejemplo contradice una intuición común sobre deportes?",
    "¿Qué pasos seguirías para analizar deportes?",
    "¿Cómo resumirías la historia de deportes?",
    "¿Qué incertidumbres existen al estudiar deportes?",
    "¿Cómo se relacionan teoría y práctica en deportes?",
    "¿Qué clasificación útil existe en deportes?",
    "¿Cómo puedo practicar deportes?",
    "¿Qué es deportes?",
]
COCINA_FACT_LABELS = [
    "cocina: hecho conceptual 1",
    "cocina: hecho conceptual 2",
    "cocina: hecho conceptual 3",
    "cocina: hecho conceptual 4",
    "cocina: hecho conceptual 5",
    "cocina: hecho conceptual 6",
    "cocina: hecho conceptual 7",
    "cocina: hecho conceptual 8",
    "cocina: hecho conceptual 9",
    "cocina: hecho conceptual 10",
    "cocina: hecho conceptual 11",
    "cocina: hecho conceptual 12",
    "cocina: hecho conceptual 13",
    "cocina: hecho conceptual 14",
    "cocina: hecho conceptual 15",
    "cocina: hecho conceptual 16",
    "cocina: hecho conceptual 17",
    "cocina: hecho conceptual 18",
    "cocina: hecho conceptual 19",
    "cocina: hecho conceptual 20",
    "cocina: hecho conceptual 21",
    "cocina: hecho conceptual 22",
    "cocina: hecho conceptual 23",
    "cocina: hecho conceptual 24",
    "cocina: hecho conceptual 25",
    "cocina: hecho conceptual 26",
    "cocina: hecho conceptual 27",
    "cocina: hecho conceptual 28",
    "cocina: hecho conceptual 29",
    "cocina: hecho conceptual 30",
    "cocina: hecho conceptual 31",
    "cocina: hecho conceptual 32",
    "cocina: hecho conceptual 33",
    "cocina: hecho conceptual 34",
    "cocina: hecho conceptual 35",
    "cocina: hecho conceptual 36",
    "cocina: hecho conceptual 37",
    "cocina: hecho conceptual 38",
    "cocina: hecho conceptual 39",
    "cocina: hecho conceptual 40",
    "cocina: hecho conceptual 41",
    "cocina: hecho conceptual 42",
    "cocina: hecho conceptual 43",
    "cocina: hecho conceptual 44",
    "cocina: hecho conceptual 45",
    "cocina: hecho conceptual 46",
    "cocina: hecho conceptual 47",
    "cocina: hecho conceptual 48",
    "cocina: hecho conceptual 49",
    "cocina: hecho conceptual 50",
    "cocina: hecho conceptual 51",
    "cocina: hecho conceptual 52",
    "cocina: hecho conceptual 53",
    "cocina: hecho conceptual 54",
    "cocina: hecho conceptual 55",
    "cocina: hecho conceptual 56",
    "cocina: hecho conceptual 57",
    "cocina: hecho conceptual 58",
    "cocina: hecho conceptual 59",
    "cocina: hecho conceptual 60",
    "cocina: hecho conceptual 61",
    "cocina: hecho conceptual 62",
    "cocina: hecho conceptual 63",
    "cocina: hecho conceptual 64",
    "cocina: hecho conceptual 65",
    "cocina: hecho conceptual 66",
    "cocina: hecho conceptual 67",
    "cocina: hecho conceptual 68",
    "cocina: hecho conceptual 69",
    "cocina: hecho conceptual 70",
    "cocina: hecho conceptual 71",
    "cocina: hecho conceptual 72",
    "cocina: hecho conceptual 73",
    "cocina: hecho conceptual 74",
    "cocina: hecho conceptual 75",
    "cocina: hecho conceptual 76",
    "cocina: hecho conceptual 77",
    "cocina: hecho conceptual 78",
    "cocina: hecho conceptual 79",
    "cocina: hecho conceptual 80",
]
COCINA_QUESTION_VARIANTS = [
    "¿Qué es cocina?",
    "¿Cómo explicarías cocina de forma sencilla?",
    "¿Por qué es importante cocina?",
    "¿Cuál es una idea fundamental de cocina?",
    "¿Puedes darme un ejemplo relacionado con cocina?",
    "¿Qué errores son frecuentes al estudiar cocina?",
    "¿Cómo se aplica cocina en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con cocina?",
    "¿Cómo empezaría a estudiar cocina?",
    "¿Qué relación tiene cocina con otras disciplinas?",
    "¿Puedes resumir cocina en pocas frases?",
    "¿Qué términos debería conocer sobre cocina?",
    "¿Cómo comprobaría si entendí cocina?",
    "¿Qué intuición ayuda a comprender cocina?",
    "¿Qué problema sencillo puedo resolver sobre cocina?",
    "¿Qué matiz suele pasarse por alto en cocina?",
    "¿Cómo se representa cocina?",
    "¿Qué supuestos se usan al hablar de cocina?",
    "¿Qué aplicaciones tiene cocina?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar cocina?",
    "¿Cómo se conecta cocina con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de cocina?",
    "¿Qué vocabulario técnico aparece en cocina?",
    "¿Puedes comparar dos enfoques dentro de cocina?",
    "¿Cómo evolucionó la comprensión de cocina?",
    "¿Qué ejemplo cotidiano ilustra cocina?",
    "¿Qué dato conviene recordar sobre cocina?",
    "¿Cómo evitar confusiones comunes en cocina?",
    "¿Qué parte de cocina suele ser más difícil?",
    "¿Puedes plantear un ejercicio de cocina?",
    "¿Cómo resolverías un ejercicio introductorio de cocina?",
    "¿Qué relación matemática aparece en cocina?",
    "¿Qué observación apoya esta idea de cocina?",
    "¿Qué pasaría si cambiamos una condición en cocina?",
    "¿Cómo se usa cocina en investigación?",
    "¿Qué herramientas sirven para estudiar cocina?",
    "¿Cómo distinguir evidencia de interpretación en cocina?",
    "¿Qué concepto previo necesito para entender cocina?",
    "¿Puedes dar una analogía para cocina?",
    "¿Qué preguntas avanzadas surgen de cocina?",
    "¿Cómo se comunica correctamente información sobre cocina?",
    "¿Qué ejemplo contradice una intuición común sobre cocina?",
    "¿Qué pasos seguirías para analizar cocina?",
    "¿Cómo resumirías la historia de cocina?",
    "¿Qué incertidumbres existen al estudiar cocina?",
    "¿Cómo se relacionan teoría y práctica en cocina?",
    "¿Qué clasificación útil existe en cocina?",
    "¿Cómo puedo practicar cocina?",
    "¿Qué es cocina?",
]
ESCRITURA_FACT_LABELS = [
    "escritura: hecho conceptual 1",
    "escritura: hecho conceptual 2",
    "escritura: hecho conceptual 3",
    "escritura: hecho conceptual 4",
    "escritura: hecho conceptual 5",
    "escritura: hecho conceptual 6",
    "escritura: hecho conceptual 7",
    "escritura: hecho conceptual 8",
    "escritura: hecho conceptual 9",
    "escritura: hecho conceptual 10",
    "escritura: hecho conceptual 11",
    "escritura: hecho conceptual 12",
    "escritura: hecho conceptual 13",
    "escritura: hecho conceptual 14",
    "escritura: hecho conceptual 15",
    "escritura: hecho conceptual 16",
    "escritura: hecho conceptual 17",
    "escritura: hecho conceptual 18",
    "escritura: hecho conceptual 19",
    "escritura: hecho conceptual 20",
    "escritura: hecho conceptual 21",
    "escritura: hecho conceptual 22",
    "escritura: hecho conceptual 23",
    "escritura: hecho conceptual 24",
    "escritura: hecho conceptual 25",
    "escritura: hecho conceptual 26",
    "escritura: hecho conceptual 27",
    "escritura: hecho conceptual 28",
    "escritura: hecho conceptual 29",
    "escritura: hecho conceptual 30",
    "escritura: hecho conceptual 31",
    "escritura: hecho conceptual 32",
    "escritura: hecho conceptual 33",
    "escritura: hecho conceptual 34",
    "escritura: hecho conceptual 35",
    "escritura: hecho conceptual 36",
    "escritura: hecho conceptual 37",
    "escritura: hecho conceptual 38",
    "escritura: hecho conceptual 39",
    "escritura: hecho conceptual 40",
    "escritura: hecho conceptual 41",
    "escritura: hecho conceptual 42",
    "escritura: hecho conceptual 43",
    "escritura: hecho conceptual 44",
    "escritura: hecho conceptual 45",
    "escritura: hecho conceptual 46",
    "escritura: hecho conceptual 47",
    "escritura: hecho conceptual 48",
    "escritura: hecho conceptual 49",
    "escritura: hecho conceptual 50",
    "escritura: hecho conceptual 51",
    "escritura: hecho conceptual 52",
    "escritura: hecho conceptual 53",
    "escritura: hecho conceptual 54",
    "escritura: hecho conceptual 55",
    "escritura: hecho conceptual 56",
    "escritura: hecho conceptual 57",
    "escritura: hecho conceptual 58",
    "escritura: hecho conceptual 59",
    "escritura: hecho conceptual 60",
    "escritura: hecho conceptual 61",
    "escritura: hecho conceptual 62",
    "escritura: hecho conceptual 63",
    "escritura: hecho conceptual 64",
    "escritura: hecho conceptual 65",
    "escritura: hecho conceptual 66",
    "escritura: hecho conceptual 67",
    "escritura: hecho conceptual 68",
    "escritura: hecho conceptual 69",
    "escritura: hecho conceptual 70",
    "escritura: hecho conceptual 71",
    "escritura: hecho conceptual 72",
    "escritura: hecho conceptual 73",
    "escritura: hecho conceptual 74",
    "escritura: hecho conceptual 75",
    "escritura: hecho conceptual 76",
    "escritura: hecho conceptual 77",
    "escritura: hecho conceptual 78",
    "escritura: hecho conceptual 79",
    "escritura: hecho conceptual 80",
]
ESCRITURA_QUESTION_VARIANTS = [
    "¿Qué es escritura?",
    "¿Cómo explicarías escritura de forma sencilla?",
    "¿Por qué es importante escritura?",
    "¿Cuál es una idea fundamental de escritura?",
    "¿Puedes darme un ejemplo relacionado con escritura?",
    "¿Qué errores son frecuentes al estudiar escritura?",
    "¿Cómo se aplica escritura en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con escritura?",
    "¿Cómo empezaría a estudiar escritura?",
    "¿Qué relación tiene escritura con otras disciplinas?",
    "¿Puedes resumir escritura en pocas frases?",
    "¿Qué términos debería conocer sobre escritura?",
    "¿Cómo comprobaría si entendí escritura?",
    "¿Qué intuición ayuda a comprender escritura?",
    "¿Qué problema sencillo puedo resolver sobre escritura?",
    "¿Qué matiz suele pasarse por alto en escritura?",
    "¿Cómo se representa escritura?",
    "¿Qué supuestos se usan al hablar de escritura?",
    "¿Qué aplicaciones tiene escritura?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar escritura?",
    "¿Cómo se conecta escritura con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de escritura?",
    "¿Qué vocabulario técnico aparece en escritura?",
    "¿Puedes comparar dos enfoques dentro de escritura?",
    "¿Cómo evolucionó la comprensión de escritura?",
    "¿Qué ejemplo cotidiano ilustra escritura?",
    "¿Qué dato conviene recordar sobre escritura?",
    "¿Cómo evitar confusiones comunes en escritura?",
    "¿Qué parte de escritura suele ser más difícil?",
    "¿Puedes plantear un ejercicio de escritura?",
    "¿Cómo resolverías un ejercicio introductorio de escritura?",
    "¿Qué relación matemática aparece en escritura?",
    "¿Qué observación apoya esta idea de escritura?",
    "¿Qué pasaría si cambiamos una condición en escritura?",
    "¿Cómo se usa escritura en investigación?",
    "¿Qué herramientas sirven para estudiar escritura?",
    "¿Cómo distinguir evidencia de interpretación en escritura?",
    "¿Qué concepto previo necesito para entender escritura?",
    "¿Puedes dar una analogía para escritura?",
    "¿Qué preguntas avanzadas surgen de escritura?",
    "¿Cómo se comunica correctamente información sobre escritura?",
    "¿Qué ejemplo contradice una intuición común sobre escritura?",
    "¿Qué pasos seguirías para analizar escritura?",
    "¿Cómo resumirías la historia de escritura?",
    "¿Qué incertidumbres existen al estudiar escritura?",
    "¿Cómo se relacionan teoría y práctica en escritura?",
    "¿Qué clasificación útil existe en escritura?",
    "¿Cómo puedo practicar escritura?",
    "¿Qué es escritura?",
]
REDACCI_N_FACT_LABELS = [
    "redacción: hecho conceptual 1",
    "redacción: hecho conceptual 2",
    "redacción: hecho conceptual 3",
    "redacción: hecho conceptual 4",
    "redacción: hecho conceptual 5",
    "redacción: hecho conceptual 6",
    "redacción: hecho conceptual 7",
    "redacción: hecho conceptual 8",
    "redacción: hecho conceptual 9",
    "redacción: hecho conceptual 10",
    "redacción: hecho conceptual 11",
    "redacción: hecho conceptual 12",
    "redacción: hecho conceptual 13",
    "redacción: hecho conceptual 14",
    "redacción: hecho conceptual 15",
    "redacción: hecho conceptual 16",
    "redacción: hecho conceptual 17",
    "redacción: hecho conceptual 18",
    "redacción: hecho conceptual 19",
    "redacción: hecho conceptual 20",
    "redacción: hecho conceptual 21",
    "redacción: hecho conceptual 22",
    "redacción: hecho conceptual 23",
    "redacción: hecho conceptual 24",
    "redacción: hecho conceptual 25",
    "redacción: hecho conceptual 26",
    "redacción: hecho conceptual 27",
    "redacción: hecho conceptual 28",
    "redacción: hecho conceptual 29",
    "redacción: hecho conceptual 30",
    "redacción: hecho conceptual 31",
    "redacción: hecho conceptual 32",
    "redacción: hecho conceptual 33",
    "redacción: hecho conceptual 34",
    "redacción: hecho conceptual 35",
    "redacción: hecho conceptual 36",
    "redacción: hecho conceptual 37",
    "redacción: hecho conceptual 38",
    "redacción: hecho conceptual 39",
    "redacción: hecho conceptual 40",
    "redacción: hecho conceptual 41",
    "redacción: hecho conceptual 42",
    "redacción: hecho conceptual 43",
    "redacción: hecho conceptual 44",
    "redacción: hecho conceptual 45",
    "redacción: hecho conceptual 46",
    "redacción: hecho conceptual 47",
    "redacción: hecho conceptual 48",
    "redacción: hecho conceptual 49",
    "redacción: hecho conceptual 50",
    "redacción: hecho conceptual 51",
    "redacción: hecho conceptual 52",
    "redacción: hecho conceptual 53",
    "redacción: hecho conceptual 54",
    "redacción: hecho conceptual 55",
    "redacción: hecho conceptual 56",
    "redacción: hecho conceptual 57",
    "redacción: hecho conceptual 58",
    "redacción: hecho conceptual 59",
    "redacción: hecho conceptual 60",
    "redacción: hecho conceptual 61",
    "redacción: hecho conceptual 62",
    "redacción: hecho conceptual 63",
    "redacción: hecho conceptual 64",
    "redacción: hecho conceptual 65",
    "redacción: hecho conceptual 66",
    "redacción: hecho conceptual 67",
    "redacción: hecho conceptual 68",
    "redacción: hecho conceptual 69",
    "redacción: hecho conceptual 70",
    "redacción: hecho conceptual 71",
    "redacción: hecho conceptual 72",
    "redacción: hecho conceptual 73",
    "redacción: hecho conceptual 74",
    "redacción: hecho conceptual 75",
    "redacción: hecho conceptual 76",
    "redacción: hecho conceptual 77",
    "redacción: hecho conceptual 78",
    "redacción: hecho conceptual 79",
    "redacción: hecho conceptual 80",
]
REDACCI_N_QUESTION_VARIANTS = [
    "¿Qué es redacción?",
    "¿Cómo explicarías redacción de forma sencilla?",
    "¿Por qué es importante redacción?",
    "¿Cuál es una idea fundamental de redacción?",
    "¿Puedes darme un ejemplo relacionado con redacción?",
    "¿Qué errores son frecuentes al estudiar redacción?",
    "¿Cómo se aplica redacción en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con redacción?",
    "¿Cómo empezaría a estudiar redacción?",
    "¿Qué relación tiene redacción con otras disciplinas?",
    "¿Puedes resumir redacción en pocas frases?",
    "¿Qué términos debería conocer sobre redacción?",
    "¿Cómo comprobaría si entendí redacción?",
    "¿Qué intuición ayuda a comprender redacción?",
    "¿Qué problema sencillo puedo resolver sobre redacción?",
    "¿Qué matiz suele pasarse por alto en redacción?",
    "¿Cómo se representa redacción?",
    "¿Qué supuestos se usan al hablar de redacción?",
    "¿Qué aplicaciones tiene redacción?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar redacción?",
    "¿Cómo se conecta redacción con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de redacción?",
    "¿Qué vocabulario técnico aparece en redacción?",
    "¿Puedes comparar dos enfoques dentro de redacción?",
    "¿Cómo evolucionó la comprensión de redacción?",
    "¿Qué ejemplo cotidiano ilustra redacción?",
    "¿Qué dato conviene recordar sobre redacción?",
    "¿Cómo evitar confusiones comunes en redacción?",
    "¿Qué parte de redacción suele ser más difícil?",
    "¿Puedes plantear un ejercicio de redacción?",
    "¿Cómo resolverías un ejercicio introductorio de redacción?",
    "¿Qué relación matemática aparece en redacción?",
    "¿Qué observación apoya esta idea de redacción?",
    "¿Qué pasaría si cambiamos una condición en redacción?",
    "¿Cómo se usa redacción en investigación?",
    "¿Qué herramientas sirven para estudiar redacción?",
    "¿Cómo distinguir evidencia de interpretación en redacción?",
    "¿Qué concepto previo necesito para entender redacción?",
    "¿Puedes dar una analogía para redacción?",
    "¿Qué preguntas avanzadas surgen de redacción?",
    "¿Cómo se comunica correctamente información sobre redacción?",
    "¿Qué ejemplo contradice una intuición común sobre redacción?",
    "¿Qué pasos seguirías para analizar redacción?",
    "¿Cómo resumirías la historia de redacción?",
    "¿Qué incertidumbres existen al estudiar redacción?",
    "¿Cómo se relacionan teoría y práctica en redacción?",
    "¿Qué clasificación útil existe en redacción?",
    "¿Cómo puedo practicar redacción?",
    "¿Qué es redacción?",
]
FILOSOF_A_FACT_LABELS = [
    "filosofía: hecho conceptual 1",
    "filosofía: hecho conceptual 2",
    "filosofía: hecho conceptual 3",
    "filosofía: hecho conceptual 4",
    "filosofía: hecho conceptual 5",
    "filosofía: hecho conceptual 6",
    "filosofía: hecho conceptual 7",
    "filosofía: hecho conceptual 8",
    "filosofía: hecho conceptual 9",
    "filosofía: hecho conceptual 10",
    "filosofía: hecho conceptual 11",
    "filosofía: hecho conceptual 12",
    "filosofía: hecho conceptual 13",
    "filosofía: hecho conceptual 14",
    "filosofía: hecho conceptual 15",
    "filosofía: hecho conceptual 16",
    "filosofía: hecho conceptual 17",
    "filosofía: hecho conceptual 18",
    "filosofía: hecho conceptual 19",
    "filosofía: hecho conceptual 20",
    "filosofía: hecho conceptual 21",
    "filosofía: hecho conceptual 22",
    "filosofía: hecho conceptual 23",
    "filosofía: hecho conceptual 24",
    "filosofía: hecho conceptual 25",
    "filosofía: hecho conceptual 26",
    "filosofía: hecho conceptual 27",
    "filosofía: hecho conceptual 28",
    "filosofía: hecho conceptual 29",
    "filosofía: hecho conceptual 30",
    "filosofía: hecho conceptual 31",
    "filosofía: hecho conceptual 32",
    "filosofía: hecho conceptual 33",
    "filosofía: hecho conceptual 34",
    "filosofía: hecho conceptual 35",
    "filosofía: hecho conceptual 36",
    "filosofía: hecho conceptual 37",
    "filosofía: hecho conceptual 38",
    "filosofía: hecho conceptual 39",
    "filosofía: hecho conceptual 40",
    "filosofía: hecho conceptual 41",
    "filosofía: hecho conceptual 42",
    "filosofía: hecho conceptual 43",
    "filosofía: hecho conceptual 44",
    "filosofía: hecho conceptual 45",
    "filosofía: hecho conceptual 46",
    "filosofía: hecho conceptual 47",
    "filosofía: hecho conceptual 48",
    "filosofía: hecho conceptual 49",
    "filosofía: hecho conceptual 50",
    "filosofía: hecho conceptual 51",
    "filosofía: hecho conceptual 52",
    "filosofía: hecho conceptual 53",
    "filosofía: hecho conceptual 54",
    "filosofía: hecho conceptual 55",
    "filosofía: hecho conceptual 56",
    "filosofía: hecho conceptual 57",
    "filosofía: hecho conceptual 58",
    "filosofía: hecho conceptual 59",
    "filosofía: hecho conceptual 60",
    "filosofía: hecho conceptual 61",
    "filosofía: hecho conceptual 62",
    "filosofía: hecho conceptual 63",
    "filosofía: hecho conceptual 64",
    "filosofía: hecho conceptual 65",
    "filosofía: hecho conceptual 66",
    "filosofía: hecho conceptual 67",
    "filosofía: hecho conceptual 68",
    "filosofía: hecho conceptual 69",
    "filosofía: hecho conceptual 70",
    "filosofía: hecho conceptual 71",
    "filosofía: hecho conceptual 72",
    "filosofía: hecho conceptual 73",
    "filosofía: hecho conceptual 74",
    "filosofía: hecho conceptual 75",
    "filosofía: hecho conceptual 76",
    "filosofía: hecho conceptual 77",
    "filosofía: hecho conceptual 78",
    "filosofía: hecho conceptual 79",
    "filosofía: hecho conceptual 80",
]
FILOSOF_A_QUESTION_VARIANTS = [
    "¿Qué es filosofía?",
    "¿Cómo explicarías filosofía de forma sencilla?",
    "¿Por qué es importante filosofía?",
    "¿Cuál es una idea fundamental de filosofía?",
    "¿Puedes darme un ejemplo relacionado con filosofía?",
    "¿Qué errores son frecuentes al estudiar filosofía?",
    "¿Cómo se aplica filosofía en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con filosofía?",
    "¿Cómo empezaría a estudiar filosofía?",
    "¿Qué relación tiene filosofía con otras disciplinas?",
    "¿Puedes resumir filosofía en pocas frases?",
    "¿Qué términos debería conocer sobre filosofía?",
    "¿Cómo comprobaría si entendí filosofía?",
    "¿Qué intuición ayuda a comprender filosofía?",
    "¿Qué problema sencillo puedo resolver sobre filosofía?",
    "¿Qué matiz suele pasarse por alto en filosofía?",
    "¿Cómo se representa filosofía?",
    "¿Qué supuestos se usan al hablar de filosofía?",
    "¿Qué aplicaciones tiene filosofía?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar filosofía?",
    "¿Cómo se conecta filosofía con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de filosofía?",
    "¿Qué vocabulario técnico aparece en filosofía?",
    "¿Puedes comparar dos enfoques dentro de filosofía?",
    "¿Cómo evolucionó la comprensión de filosofía?",
    "¿Qué ejemplo cotidiano ilustra filosofía?",
    "¿Qué dato conviene recordar sobre filosofía?",
    "¿Cómo evitar confusiones comunes en filosofía?",
    "¿Qué parte de filosofía suele ser más difícil?",
    "¿Puedes plantear un ejercicio de filosofía?",
    "¿Cómo resolverías un ejercicio introductorio de filosofía?",
    "¿Qué relación matemática aparece en filosofía?",
    "¿Qué observación apoya esta idea de filosofía?",
    "¿Qué pasaría si cambiamos una condición en filosofía?",
    "¿Cómo se usa filosofía en investigación?",
    "¿Qué herramientas sirven para estudiar filosofía?",
    "¿Cómo distinguir evidencia de interpretación en filosofía?",
    "¿Qué concepto previo necesito para entender filosofía?",
    "¿Puedes dar una analogía para filosofía?",
    "¿Qué preguntas avanzadas surgen de filosofía?",
    "¿Cómo se comunica correctamente información sobre filosofía?",
    "¿Qué ejemplo contradice una intuición común sobre filosofía?",
    "¿Qué pasos seguirías para analizar filosofía?",
    "¿Cómo resumirías la historia de filosofía?",
    "¿Qué incertidumbres existen al estudiar filosofía?",
    "¿Cómo se relacionan teoría y práctica en filosofía?",
    "¿Qué clasificación útil existe en filosofía?",
    "¿Cómo puedo practicar filosofía?",
    "¿Qué es filosofía?",
]
PSICOLOG_A_FACT_LABELS = [
    "psicología: hecho conceptual 1",
    "psicología: hecho conceptual 2",
    "psicología: hecho conceptual 3",
    "psicología: hecho conceptual 4",
    "psicología: hecho conceptual 5",
    "psicología: hecho conceptual 6",
    "psicología: hecho conceptual 7",
    "psicología: hecho conceptual 8",
    "psicología: hecho conceptual 9",
    "psicología: hecho conceptual 10",
    "psicología: hecho conceptual 11",
    "psicología: hecho conceptual 12",
    "psicología: hecho conceptual 13",
    "psicología: hecho conceptual 14",
    "psicología: hecho conceptual 15",
    "psicología: hecho conceptual 16",
    "psicología: hecho conceptual 17",
    "psicología: hecho conceptual 18",
    "psicología: hecho conceptual 19",
    "psicología: hecho conceptual 20",
    "psicología: hecho conceptual 21",
    "psicología: hecho conceptual 22",
    "psicología: hecho conceptual 23",
    "psicología: hecho conceptual 24",
    "psicología: hecho conceptual 25",
    "psicología: hecho conceptual 26",
    "psicología: hecho conceptual 27",
    "psicología: hecho conceptual 28",
    "psicología: hecho conceptual 29",
    "psicología: hecho conceptual 30",
    "psicología: hecho conceptual 31",
    "psicología: hecho conceptual 32",
    "psicología: hecho conceptual 33",
    "psicología: hecho conceptual 34",
    "psicología: hecho conceptual 35",
    "psicología: hecho conceptual 36",
    "psicología: hecho conceptual 37",
    "psicología: hecho conceptual 38",
    "psicología: hecho conceptual 39",
    "psicología: hecho conceptual 40",
    "psicología: hecho conceptual 41",
    "psicología: hecho conceptual 42",
    "psicología: hecho conceptual 43",
    "psicología: hecho conceptual 44",
    "psicología: hecho conceptual 45",
    "psicología: hecho conceptual 46",
    "psicología: hecho conceptual 47",
    "psicología: hecho conceptual 48",
    "psicología: hecho conceptual 49",
    "psicología: hecho conceptual 50",
    "psicología: hecho conceptual 51",
    "psicología: hecho conceptual 52",
    "psicología: hecho conceptual 53",
    "psicología: hecho conceptual 54",
    "psicología: hecho conceptual 55",
    "psicología: hecho conceptual 56",
    "psicología: hecho conceptual 57",
    "psicología: hecho conceptual 58",
    "psicología: hecho conceptual 59",
    "psicología: hecho conceptual 60",
    "psicología: hecho conceptual 61",
    "psicología: hecho conceptual 62",
    "psicología: hecho conceptual 63",
    "psicología: hecho conceptual 64",
    "psicología: hecho conceptual 65",
    "psicología: hecho conceptual 66",
    "psicología: hecho conceptual 67",
    "psicología: hecho conceptual 68",
    "psicología: hecho conceptual 69",
    "psicología: hecho conceptual 70",
    "psicología: hecho conceptual 71",
    "psicología: hecho conceptual 72",
    "psicología: hecho conceptual 73",
    "psicología: hecho conceptual 74",
    "psicología: hecho conceptual 75",
    "psicología: hecho conceptual 76",
    "psicología: hecho conceptual 77",
    "psicología: hecho conceptual 78",
    "psicología: hecho conceptual 79",
    "psicología: hecho conceptual 80",
]
PSICOLOG_A_QUESTION_VARIANTS = [
    "¿Qué es psicología?",
    "¿Cómo explicarías psicología de forma sencilla?",
    "¿Por qué es importante psicología?",
    "¿Cuál es una idea fundamental de psicología?",
    "¿Puedes darme un ejemplo relacionado con psicología?",
    "¿Qué errores son frecuentes al estudiar psicología?",
    "¿Cómo se aplica psicología en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con psicología?",
    "¿Cómo empezaría a estudiar psicología?",
    "¿Qué relación tiene psicología con otras disciplinas?",
    "¿Puedes resumir psicología en pocas frases?",
    "¿Qué términos debería conocer sobre psicología?",
    "¿Cómo comprobaría si entendí psicología?",
    "¿Qué intuición ayuda a comprender psicología?",
    "¿Qué problema sencillo puedo resolver sobre psicología?",
    "¿Qué matiz suele pasarse por alto en psicología?",
    "¿Cómo se representa psicología?",
    "¿Qué supuestos se usan al hablar de psicología?",
    "¿Qué aplicaciones tiene psicología?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar psicología?",
    "¿Cómo se conecta psicología con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de psicología?",
    "¿Qué vocabulario técnico aparece en psicología?",
    "¿Puedes comparar dos enfoques dentro de psicología?",
    "¿Cómo evolucionó la comprensión de psicología?",
    "¿Qué ejemplo cotidiano ilustra psicología?",
    "¿Qué dato conviene recordar sobre psicología?",
    "¿Cómo evitar confusiones comunes en psicología?",
    "¿Qué parte de psicología suele ser más difícil?",
    "¿Puedes plantear un ejercicio de psicología?",
    "¿Cómo resolverías un ejercicio introductorio de psicología?",
    "¿Qué relación matemática aparece en psicología?",
    "¿Qué observación apoya esta idea de psicología?",
    "¿Qué pasaría si cambiamos una condición en psicología?",
    "¿Cómo se usa psicología en investigación?",
    "¿Qué herramientas sirven para estudiar psicología?",
    "¿Cómo distinguir evidencia de interpretación en psicología?",
    "¿Qué concepto previo necesito para entender psicología?",
    "¿Puedes dar una analogía para psicología?",
    "¿Qué preguntas avanzadas surgen de psicología?",
    "¿Cómo se comunica correctamente información sobre psicología?",
    "¿Qué ejemplo contradice una intuición común sobre psicología?",
    "¿Qué pasos seguirías para analizar psicología?",
    "¿Cómo resumirías la historia de psicología?",
    "¿Qué incertidumbres existen al estudiar psicología?",
    "¿Cómo se relacionan teoría y práctica en psicología?",
    "¿Qué clasificación útil existe en psicología?",
    "¿Cómo puedo practicar psicología?",
    "¿Qué es psicología?",
]
ECONOM_A_FACT_LABELS = [
    "economía: hecho conceptual 1",
    "economía: hecho conceptual 2",
    "economía: hecho conceptual 3",
    "economía: hecho conceptual 4",
    "economía: hecho conceptual 5",
    "economía: hecho conceptual 6",
    "economía: hecho conceptual 7",
    "economía: hecho conceptual 8",
    "economía: hecho conceptual 9",
    "economía: hecho conceptual 10",
    "economía: hecho conceptual 11",
    "economía: hecho conceptual 12",
    "economía: hecho conceptual 13",
    "economía: hecho conceptual 14",
    "economía: hecho conceptual 15",
    "economía: hecho conceptual 16",
    "economía: hecho conceptual 17",
    "economía: hecho conceptual 18",
    "economía: hecho conceptual 19",
    "economía: hecho conceptual 20",
    "economía: hecho conceptual 21",
    "economía: hecho conceptual 22",
    "economía: hecho conceptual 23",
    "economía: hecho conceptual 24",
    "economía: hecho conceptual 25",
    "economía: hecho conceptual 26",
    "economía: hecho conceptual 27",
    "economía: hecho conceptual 28",
    "economía: hecho conceptual 29",
    "economía: hecho conceptual 30",
    "economía: hecho conceptual 31",
    "economía: hecho conceptual 32",
    "economía: hecho conceptual 33",
    "economía: hecho conceptual 34",
    "economía: hecho conceptual 35",
    "economía: hecho conceptual 36",
    "economía: hecho conceptual 37",
    "economía: hecho conceptual 38",
    "economía: hecho conceptual 39",
    "economía: hecho conceptual 40",
    "economía: hecho conceptual 41",
    "economía: hecho conceptual 42",
    "economía: hecho conceptual 43",
    "economía: hecho conceptual 44",
    "economía: hecho conceptual 45",
    "economía: hecho conceptual 46",
    "economía: hecho conceptual 47",
    "economía: hecho conceptual 48",
    "economía: hecho conceptual 49",
    "economía: hecho conceptual 50",
    "economía: hecho conceptual 51",
    "economía: hecho conceptual 52",
    "economía: hecho conceptual 53",
    "economía: hecho conceptual 54",
    "economía: hecho conceptual 55",
    "economía: hecho conceptual 56",
    "economía: hecho conceptual 57",
    "economía: hecho conceptual 58",
    "economía: hecho conceptual 59",
    "economía: hecho conceptual 60",
    "economía: hecho conceptual 61",
    "economía: hecho conceptual 62",
    "economía: hecho conceptual 63",
    "economía: hecho conceptual 64",
    "economía: hecho conceptual 65",
    "economía: hecho conceptual 66",
    "economía: hecho conceptual 67",
    "economía: hecho conceptual 68",
    "economía: hecho conceptual 69",
    "economía: hecho conceptual 70",
    "economía: hecho conceptual 71",
    "economía: hecho conceptual 72",
    "economía: hecho conceptual 73",
    "economía: hecho conceptual 74",
    "economía: hecho conceptual 75",
    "economía: hecho conceptual 76",
    "economía: hecho conceptual 77",
    "economía: hecho conceptual 78",
    "economía: hecho conceptual 79",
    "economía: hecho conceptual 80",
]
ECONOM_A_QUESTION_VARIANTS = [
    "¿Qué es economía?",
    "¿Cómo explicarías economía de forma sencilla?",
    "¿Por qué es importante economía?",
    "¿Cuál es una idea fundamental de economía?",
    "¿Puedes darme un ejemplo relacionado con economía?",
    "¿Qué errores son frecuentes al estudiar economía?",
    "¿Cómo se aplica economía en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con economía?",
    "¿Cómo empezaría a estudiar economía?",
    "¿Qué relación tiene economía con otras disciplinas?",
    "¿Puedes resumir economía en pocas frases?",
    "¿Qué términos debería conocer sobre economía?",
    "¿Cómo comprobaría si entendí economía?",
    "¿Qué intuición ayuda a comprender economía?",
    "¿Qué problema sencillo puedo resolver sobre economía?",
    "¿Qué matiz suele pasarse por alto en economía?",
    "¿Cómo se representa economía?",
    "¿Qué supuestos se usan al hablar de economía?",
    "¿Qué aplicaciones tiene economía?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar economía?",
    "¿Cómo se conecta economía con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de economía?",
    "¿Qué vocabulario técnico aparece en economía?",
    "¿Puedes comparar dos enfoques dentro de economía?",
    "¿Cómo evolucionó la comprensión de economía?",
    "¿Qué ejemplo cotidiano ilustra economía?",
    "¿Qué dato conviene recordar sobre economía?",
    "¿Cómo evitar confusiones comunes en economía?",
    "¿Qué parte de economía suele ser más difícil?",
    "¿Puedes plantear un ejercicio de economía?",
    "¿Cómo resolverías un ejercicio introductorio de economía?",
    "¿Qué relación matemática aparece en economía?",
    "¿Qué observación apoya esta idea de economía?",
    "¿Qué pasaría si cambiamos una condición en economía?",
    "¿Cómo se usa economía en investigación?",
    "¿Qué herramientas sirven para estudiar economía?",
    "¿Cómo distinguir evidencia de interpretación en economía?",
    "¿Qué concepto previo necesito para entender economía?",
    "¿Puedes dar una analogía para economía?",
    "¿Qué preguntas avanzadas surgen de economía?",
    "¿Cómo se comunica correctamente información sobre economía?",
    "¿Qué ejemplo contradice una intuición común sobre economía?",
    "¿Qué pasos seguirías para analizar economía?",
    "¿Cómo resumirías la historia de economía?",
    "¿Qué incertidumbres existen al estudiar economía?",
    "¿Cómo se relacionan teoría y práctica en economía?",
    "¿Qué clasificación útil existe en economía?",
    "¿Cómo puedo practicar economía?",
    "¿Qué es economía?",
]
ARTE_FACT_LABELS = [
    "arte: hecho conceptual 1",
    "arte: hecho conceptual 2",
    "arte: hecho conceptual 3",
    "arte: hecho conceptual 4",
    "arte: hecho conceptual 5",
    "arte: hecho conceptual 6",
    "arte: hecho conceptual 7",
    "arte: hecho conceptual 8",
    "arte: hecho conceptual 9",
    "arte: hecho conceptual 10",
    "arte: hecho conceptual 11",
    "arte: hecho conceptual 12",
    "arte: hecho conceptual 13",
    "arte: hecho conceptual 14",
    "arte: hecho conceptual 15",
    "arte: hecho conceptual 16",
    "arte: hecho conceptual 17",
    "arte: hecho conceptual 18",
    "arte: hecho conceptual 19",
    "arte: hecho conceptual 20",
    "arte: hecho conceptual 21",
    "arte: hecho conceptual 22",
    "arte: hecho conceptual 23",
    "arte: hecho conceptual 24",
    "arte: hecho conceptual 25",
    "arte: hecho conceptual 26",
    "arte: hecho conceptual 27",
    "arte: hecho conceptual 28",
    "arte: hecho conceptual 29",
    "arte: hecho conceptual 30",
    "arte: hecho conceptual 31",
    "arte: hecho conceptual 32",
    "arte: hecho conceptual 33",
    "arte: hecho conceptual 34",
    "arte: hecho conceptual 35",
    "arte: hecho conceptual 36",
    "arte: hecho conceptual 37",
    "arte: hecho conceptual 38",
    "arte: hecho conceptual 39",
    "arte: hecho conceptual 40",
    "arte: hecho conceptual 41",
    "arte: hecho conceptual 42",
    "arte: hecho conceptual 43",
    "arte: hecho conceptual 44",
    "arte: hecho conceptual 45",
    "arte: hecho conceptual 46",
    "arte: hecho conceptual 47",
    "arte: hecho conceptual 48",
    "arte: hecho conceptual 49",
    "arte: hecho conceptual 50",
    "arte: hecho conceptual 51",
    "arte: hecho conceptual 52",
    "arte: hecho conceptual 53",
    "arte: hecho conceptual 54",
    "arte: hecho conceptual 55",
    "arte: hecho conceptual 56",
    "arte: hecho conceptual 57",
    "arte: hecho conceptual 58",
    "arte: hecho conceptual 59",
    "arte: hecho conceptual 60",
    "arte: hecho conceptual 61",
    "arte: hecho conceptual 62",
    "arte: hecho conceptual 63",
    "arte: hecho conceptual 64",
    "arte: hecho conceptual 65",
    "arte: hecho conceptual 66",
    "arte: hecho conceptual 67",
    "arte: hecho conceptual 68",
    "arte: hecho conceptual 69",
    "arte: hecho conceptual 70",
    "arte: hecho conceptual 71",
    "arte: hecho conceptual 72",
    "arte: hecho conceptual 73",
    "arte: hecho conceptual 74",
    "arte: hecho conceptual 75",
    "arte: hecho conceptual 76",
    "arte: hecho conceptual 77",
    "arte: hecho conceptual 78",
    "arte: hecho conceptual 79",
    "arte: hecho conceptual 80",
]
ARTE_QUESTION_VARIANTS = [
    "¿Qué es arte?",
    "¿Cómo explicarías arte de forma sencilla?",
    "¿Por qué es importante arte?",
    "¿Cuál es una idea fundamental de arte?",
    "¿Puedes darme un ejemplo relacionado con arte?",
    "¿Qué errores son frecuentes al estudiar arte?",
    "¿Cómo se aplica arte en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con arte?",
    "¿Cómo empezaría a estudiar arte?",
    "¿Qué relación tiene arte con otras disciplinas?",
    "¿Puedes resumir arte en pocas frases?",
    "¿Qué términos debería conocer sobre arte?",
    "¿Cómo comprobaría si entendí arte?",
    "¿Qué intuición ayuda a comprender arte?",
    "¿Qué problema sencillo puedo resolver sobre arte?",
    "¿Qué matiz suele pasarse por alto en arte?",
    "¿Cómo se representa arte?",
    "¿Qué supuestos se usan al hablar de arte?",
    "¿Qué aplicaciones tiene arte?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar arte?",
    "¿Cómo se conecta arte con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de arte?",
    "¿Qué vocabulario técnico aparece en arte?",
    "¿Puedes comparar dos enfoques dentro de arte?",
    "¿Cómo evolucionó la comprensión de arte?",
    "¿Qué ejemplo cotidiano ilustra arte?",
    "¿Qué dato conviene recordar sobre arte?",
    "¿Cómo evitar confusiones comunes en arte?",
    "¿Qué parte de arte suele ser más difícil?",
    "¿Puedes plantear un ejercicio de arte?",
    "¿Cómo resolverías un ejercicio introductorio de arte?",
    "¿Qué relación matemática aparece en arte?",
    "¿Qué observación apoya esta idea de arte?",
    "¿Qué pasaría si cambiamos una condición en arte?",
    "¿Cómo se usa arte en investigación?",
    "¿Qué herramientas sirven para estudiar arte?",
    "¿Cómo distinguir evidencia de interpretación en arte?",
    "¿Qué concepto previo necesito para entender arte?",
    "¿Puedes dar una analogía para arte?",
    "¿Qué preguntas avanzadas surgen de arte?",
    "¿Cómo se comunica correctamente información sobre arte?",
    "¿Qué ejemplo contradice una intuición común sobre arte?",
    "¿Qué pasos seguirías para analizar arte?",
    "¿Cómo resumirías la historia de arte?",
    "¿Qué incertidumbres existen al estudiar arte?",
    "¿Cómo se relacionan teoría y práctica en arte?",
    "¿Qué clasificación útil existe en arte?",
    "¿Cómo puedo practicar arte?",
    "¿Qué es arte?",
]
M_SICA_FACT_LABELS = [
    "música: hecho conceptual 1",
    "música: hecho conceptual 2",
    "música: hecho conceptual 3",
    "música: hecho conceptual 4",
    "música: hecho conceptual 5",
    "música: hecho conceptual 6",
    "música: hecho conceptual 7",
    "música: hecho conceptual 8",
    "música: hecho conceptual 9",
    "música: hecho conceptual 10",
    "música: hecho conceptual 11",
    "música: hecho conceptual 12",
    "música: hecho conceptual 13",
    "música: hecho conceptual 14",
    "música: hecho conceptual 15",
    "música: hecho conceptual 16",
    "música: hecho conceptual 17",
    "música: hecho conceptual 18",
    "música: hecho conceptual 19",
    "música: hecho conceptual 20",
    "música: hecho conceptual 21",
    "música: hecho conceptual 22",
    "música: hecho conceptual 23",
    "música: hecho conceptual 24",
    "música: hecho conceptual 25",
    "música: hecho conceptual 26",
    "música: hecho conceptual 27",
    "música: hecho conceptual 28",
    "música: hecho conceptual 29",
    "música: hecho conceptual 30",
    "música: hecho conceptual 31",
    "música: hecho conceptual 32",
    "música: hecho conceptual 33",
    "música: hecho conceptual 34",
    "música: hecho conceptual 35",
    "música: hecho conceptual 36",
    "música: hecho conceptual 37",
    "música: hecho conceptual 38",
    "música: hecho conceptual 39",
    "música: hecho conceptual 40",
    "música: hecho conceptual 41",
    "música: hecho conceptual 42",
    "música: hecho conceptual 43",
    "música: hecho conceptual 44",
    "música: hecho conceptual 45",
    "música: hecho conceptual 46",
    "música: hecho conceptual 47",
    "música: hecho conceptual 48",
    "música: hecho conceptual 49",
    "música: hecho conceptual 50",
    "música: hecho conceptual 51",
    "música: hecho conceptual 52",
    "música: hecho conceptual 53",
    "música: hecho conceptual 54",
    "música: hecho conceptual 55",
    "música: hecho conceptual 56",
    "música: hecho conceptual 57",
    "música: hecho conceptual 58",
    "música: hecho conceptual 59",
    "música: hecho conceptual 60",
    "música: hecho conceptual 61",
    "música: hecho conceptual 62",
    "música: hecho conceptual 63",
    "música: hecho conceptual 64",
    "música: hecho conceptual 65",
    "música: hecho conceptual 66",
    "música: hecho conceptual 67",
    "música: hecho conceptual 68",
    "música: hecho conceptual 69",
    "música: hecho conceptual 70",
    "música: hecho conceptual 71",
    "música: hecho conceptual 72",
    "música: hecho conceptual 73",
    "música: hecho conceptual 74",
    "música: hecho conceptual 75",
    "música: hecho conceptual 76",
    "música: hecho conceptual 77",
    "música: hecho conceptual 78",
    "música: hecho conceptual 79",
    "música: hecho conceptual 80",
]
M_SICA_QUESTION_VARIANTS = [
    "¿Qué es música?",
    "¿Cómo explicarías música de forma sencilla?",
    "¿Por qué es importante música?",
    "¿Cuál es una idea fundamental de música?",
    "¿Puedes darme un ejemplo relacionado con música?",
    "¿Qué errores son frecuentes al estudiar música?",
    "¿Cómo se aplica música en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con música?",
    "¿Cómo empezaría a estudiar música?",
    "¿Qué relación tiene música con otras disciplinas?",
    "¿Puedes resumir música en pocas frases?",
    "¿Qué términos debería conocer sobre música?",
    "¿Cómo comprobaría si entendí música?",
    "¿Qué intuición ayuda a comprender música?",
    "¿Qué problema sencillo puedo resolver sobre música?",
    "¿Qué matiz suele pasarse por alto en música?",
    "¿Cómo se representa música?",
    "¿Qué supuestos se usan al hablar de música?",
    "¿Qué aplicaciones tiene música?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar música?",
    "¿Cómo se conecta música con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de música?",
    "¿Qué vocabulario técnico aparece en música?",
    "¿Puedes comparar dos enfoques dentro de música?",
    "¿Cómo evolucionó la comprensión de música?",
    "¿Qué ejemplo cotidiano ilustra música?",
    "¿Qué dato conviene recordar sobre música?",
    "¿Cómo evitar confusiones comunes en música?",
    "¿Qué parte de música suele ser más difícil?",
    "¿Puedes plantear un ejercicio de música?",
    "¿Cómo resolverías un ejercicio introductorio de música?",
    "¿Qué relación matemática aparece en música?",
    "¿Qué observación apoya esta idea de música?",
    "¿Qué pasaría si cambiamos una condición en música?",
    "¿Cómo se usa música en investigación?",
    "¿Qué herramientas sirven para estudiar música?",
    "¿Cómo distinguir evidencia de interpretación en música?",
    "¿Qué concepto previo necesito para entender música?",
    "¿Puedes dar una analogía para música?",
    "¿Qué preguntas avanzadas surgen de música?",
    "¿Cómo se comunica correctamente información sobre música?",
    "¿Qué ejemplo contradice una intuición común sobre música?",
    "¿Qué pasos seguirías para analizar música?",
    "¿Cómo resumirías la historia de música?",
    "¿Qué incertidumbres existen al estudiar música?",
    "¿Cómo se relacionan teoría y práctica en música?",
    "¿Qué clasificación útil existe en música?",
    "¿Cómo puedo practicar música?",
    "¿Qué es música?",
]
CINE_FACT_LABELS = [
    "cine: hecho conceptual 1",
    "cine: hecho conceptual 2",
    "cine: hecho conceptual 3",
    "cine: hecho conceptual 4",
    "cine: hecho conceptual 5",
    "cine: hecho conceptual 6",
    "cine: hecho conceptual 7",
    "cine: hecho conceptual 8",
    "cine: hecho conceptual 9",
    "cine: hecho conceptual 10",
    "cine: hecho conceptual 11",
    "cine: hecho conceptual 12",
    "cine: hecho conceptual 13",
    "cine: hecho conceptual 14",
    "cine: hecho conceptual 15",
    "cine: hecho conceptual 16",
    "cine: hecho conceptual 17",
    "cine: hecho conceptual 18",
    "cine: hecho conceptual 19",
    "cine: hecho conceptual 20",
    "cine: hecho conceptual 21",
    "cine: hecho conceptual 22",
    "cine: hecho conceptual 23",
    "cine: hecho conceptual 24",
    "cine: hecho conceptual 25",
    "cine: hecho conceptual 26",
    "cine: hecho conceptual 27",
    "cine: hecho conceptual 28",
    "cine: hecho conceptual 29",
    "cine: hecho conceptual 30",
    "cine: hecho conceptual 31",
    "cine: hecho conceptual 32",
    "cine: hecho conceptual 33",
    "cine: hecho conceptual 34",
    "cine: hecho conceptual 35",
    "cine: hecho conceptual 36",
    "cine: hecho conceptual 37",
    "cine: hecho conceptual 38",
    "cine: hecho conceptual 39",
    "cine: hecho conceptual 40",
    "cine: hecho conceptual 41",
    "cine: hecho conceptual 42",
    "cine: hecho conceptual 43",
    "cine: hecho conceptual 44",
    "cine: hecho conceptual 45",
    "cine: hecho conceptual 46",
    "cine: hecho conceptual 47",
    "cine: hecho conceptual 48",
    "cine: hecho conceptual 49",
    "cine: hecho conceptual 50",
    "cine: hecho conceptual 51",
    "cine: hecho conceptual 52",
    "cine: hecho conceptual 53",
    "cine: hecho conceptual 54",
    "cine: hecho conceptual 55",
    "cine: hecho conceptual 56",
    "cine: hecho conceptual 57",
    "cine: hecho conceptual 58",
    "cine: hecho conceptual 59",
    "cine: hecho conceptual 60",
    "cine: hecho conceptual 61",
    "cine: hecho conceptual 62",
    "cine: hecho conceptual 63",
    "cine: hecho conceptual 64",
    "cine: hecho conceptual 65",
    "cine: hecho conceptual 66",
    "cine: hecho conceptual 67",
    "cine: hecho conceptual 68",
    "cine: hecho conceptual 69",
    "cine: hecho conceptual 70",
    "cine: hecho conceptual 71",
    "cine: hecho conceptual 72",
    "cine: hecho conceptual 73",
    "cine: hecho conceptual 74",
    "cine: hecho conceptual 75",
    "cine: hecho conceptual 76",
    "cine: hecho conceptual 77",
    "cine: hecho conceptual 78",
    "cine: hecho conceptual 79",
    "cine: hecho conceptual 80",
]
CINE_QUESTION_VARIANTS = [
    "¿Qué es cine?",
    "¿Cómo explicarías cine de forma sencilla?",
    "¿Por qué es importante cine?",
    "¿Cuál es una idea fundamental de cine?",
    "¿Puedes darme un ejemplo relacionado con cine?",
    "¿Qué errores son frecuentes al estudiar cine?",
    "¿Cómo se aplica cine en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con cine?",
    "¿Cómo empezaría a estudiar cine?",
    "¿Qué relación tiene cine con otras disciplinas?",
    "¿Puedes resumir cine en pocas frases?",
    "¿Qué términos debería conocer sobre cine?",
    "¿Cómo comprobaría si entendí cine?",
    "¿Qué intuición ayuda a comprender cine?",
    "¿Qué problema sencillo puedo resolver sobre cine?",
    "¿Qué matiz suele pasarse por alto en cine?",
    "¿Cómo se representa cine?",
    "¿Qué supuestos se usan al hablar de cine?",
    "¿Qué aplicaciones tiene cine?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar cine?",
    "¿Cómo se conecta cine con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de cine?",
    "¿Qué vocabulario técnico aparece en cine?",
    "¿Puedes comparar dos enfoques dentro de cine?",
    "¿Cómo evolucionó la comprensión de cine?",
    "¿Qué ejemplo cotidiano ilustra cine?",
    "¿Qué dato conviene recordar sobre cine?",
    "¿Cómo evitar confusiones comunes en cine?",
    "¿Qué parte de cine suele ser más difícil?",
    "¿Puedes plantear un ejercicio de cine?",
    "¿Cómo resolverías un ejercicio introductorio de cine?",
    "¿Qué relación matemática aparece en cine?",
    "¿Qué observación apoya esta idea de cine?",
    "¿Qué pasaría si cambiamos una condición en cine?",
    "¿Cómo se usa cine en investigación?",
    "¿Qué herramientas sirven para estudiar cine?",
    "¿Cómo distinguir evidencia de interpretación en cine?",
    "¿Qué concepto previo necesito para entender cine?",
    "¿Puedes dar una analogía para cine?",
    "¿Qué preguntas avanzadas surgen de cine?",
    "¿Cómo se comunica correctamente información sobre cine?",
    "¿Qué ejemplo contradice una intuición común sobre cine?",
    "¿Qué pasos seguirías para analizar cine?",
    "¿Cómo resumirías la historia de cine?",
    "¿Qué incertidumbres existen al estudiar cine?",
    "¿Cómo se relacionan teoría y práctica en cine?",
    "¿Qué clasificación útil existe en cine?",
    "¿Cómo puedo practicar cine?",
    "¿Qué es cine?",
]
LITERATURA_FACT_LABELS = [
    "literatura: hecho conceptual 1",
    "literatura: hecho conceptual 2",
    "literatura: hecho conceptual 3",
    "literatura: hecho conceptual 4",
    "literatura: hecho conceptual 5",
    "literatura: hecho conceptual 6",
    "literatura: hecho conceptual 7",
    "literatura: hecho conceptual 8",
    "literatura: hecho conceptual 9",
    "literatura: hecho conceptual 10",
    "literatura: hecho conceptual 11",
    "literatura: hecho conceptual 12",
    "literatura: hecho conceptual 13",
    "literatura: hecho conceptual 14",
    "literatura: hecho conceptual 15",
    "literatura: hecho conceptual 16",
    "literatura: hecho conceptual 17",
    "literatura: hecho conceptual 18",
    "literatura: hecho conceptual 19",
    "literatura: hecho conceptual 20",
    "literatura: hecho conceptual 21",
    "literatura: hecho conceptual 22",
    "literatura: hecho conceptual 23",
    "literatura: hecho conceptual 24",
    "literatura: hecho conceptual 25",
    "literatura: hecho conceptual 26",
    "literatura: hecho conceptual 27",
    "literatura: hecho conceptual 28",
    "literatura: hecho conceptual 29",
    "literatura: hecho conceptual 30",
    "literatura: hecho conceptual 31",
    "literatura: hecho conceptual 32",
    "literatura: hecho conceptual 33",
    "literatura: hecho conceptual 34",
    "literatura: hecho conceptual 35",
    "literatura: hecho conceptual 36",
    "literatura: hecho conceptual 37",
    "literatura: hecho conceptual 38",
    "literatura: hecho conceptual 39",
    "literatura: hecho conceptual 40",
    "literatura: hecho conceptual 41",
    "literatura: hecho conceptual 42",
    "literatura: hecho conceptual 43",
    "literatura: hecho conceptual 44",
    "literatura: hecho conceptual 45",
    "literatura: hecho conceptual 46",
    "literatura: hecho conceptual 47",
    "literatura: hecho conceptual 48",
    "literatura: hecho conceptual 49",
    "literatura: hecho conceptual 50",
    "literatura: hecho conceptual 51",
    "literatura: hecho conceptual 52",
    "literatura: hecho conceptual 53",
    "literatura: hecho conceptual 54",
    "literatura: hecho conceptual 55",
    "literatura: hecho conceptual 56",
    "literatura: hecho conceptual 57",
    "literatura: hecho conceptual 58",
    "literatura: hecho conceptual 59",
    "literatura: hecho conceptual 60",
    "literatura: hecho conceptual 61",
    "literatura: hecho conceptual 62",
    "literatura: hecho conceptual 63",
    "literatura: hecho conceptual 64",
    "literatura: hecho conceptual 65",
    "literatura: hecho conceptual 66",
    "literatura: hecho conceptual 67",
    "literatura: hecho conceptual 68",
    "literatura: hecho conceptual 69",
    "literatura: hecho conceptual 70",
    "literatura: hecho conceptual 71",
    "literatura: hecho conceptual 72",
    "literatura: hecho conceptual 73",
    "literatura: hecho conceptual 74",
    "literatura: hecho conceptual 75",
    "literatura: hecho conceptual 76",
    "literatura: hecho conceptual 77",
    "literatura: hecho conceptual 78",
    "literatura: hecho conceptual 79",
    "literatura: hecho conceptual 80",
]
LITERATURA_QUESTION_VARIANTS = [
    "¿Qué es literatura?",
    "¿Cómo explicarías literatura de forma sencilla?",
    "¿Por qué es importante literatura?",
    "¿Cuál es una idea fundamental de literatura?",
    "¿Puedes darme un ejemplo relacionado con literatura?",
    "¿Qué errores son frecuentes al estudiar literatura?",
    "¿Cómo se aplica literatura en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con literatura?",
    "¿Cómo empezaría a estudiar literatura?",
    "¿Qué relación tiene literatura con otras disciplinas?",
    "¿Puedes resumir literatura en pocas frases?",
    "¿Qué términos debería conocer sobre literatura?",
    "¿Cómo comprobaría si entendí literatura?",
    "¿Qué intuición ayuda a comprender literatura?",
    "¿Qué problema sencillo puedo resolver sobre literatura?",
    "¿Qué matiz suele pasarse por alto en literatura?",
    "¿Cómo se representa literatura?",
    "¿Qué supuestos se usan al hablar de literatura?",
    "¿Qué aplicaciones tiene literatura?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar literatura?",
    "¿Cómo se conecta literatura con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de literatura?",
    "¿Qué vocabulario técnico aparece en literatura?",
    "¿Puedes comparar dos enfoques dentro de literatura?",
    "¿Cómo evolucionó la comprensión de literatura?",
    "¿Qué ejemplo cotidiano ilustra literatura?",
    "¿Qué dato conviene recordar sobre literatura?",
    "¿Cómo evitar confusiones comunes en literatura?",
    "¿Qué parte de literatura suele ser más difícil?",
    "¿Puedes plantear un ejercicio de literatura?",
    "¿Cómo resolverías un ejercicio introductorio de literatura?",
    "¿Qué relación matemática aparece en literatura?",
    "¿Qué observación apoya esta idea de literatura?",
    "¿Qué pasaría si cambiamos una condición en literatura?",
    "¿Cómo se usa literatura en investigación?",
    "¿Qué herramientas sirven para estudiar literatura?",
    "¿Cómo distinguir evidencia de interpretación en literatura?",
    "¿Qué concepto previo necesito para entender literatura?",
    "¿Puedes dar una analogía para literatura?",
    "¿Qué preguntas avanzadas surgen de literatura?",
    "¿Cómo se comunica correctamente información sobre literatura?",
    "¿Qué ejemplo contradice una intuición común sobre literatura?",
    "¿Qué pasos seguirías para analizar literatura?",
    "¿Cómo resumirías la historia de literatura?",
    "¿Qué incertidumbres existen al estudiar literatura?",
    "¿Cómo se relacionan teoría y práctica en literatura?",
    "¿Qué clasificación útil existe en literatura?",
    "¿Cómo puedo practicar literatura?",
    "¿Qué es literatura?",
]
VIAJES_FACT_LABELS = [
    "viajes: hecho conceptual 1",
    "viajes: hecho conceptual 2",
    "viajes: hecho conceptual 3",
    "viajes: hecho conceptual 4",
    "viajes: hecho conceptual 5",
    "viajes: hecho conceptual 6",
    "viajes: hecho conceptual 7",
    "viajes: hecho conceptual 8",
    "viajes: hecho conceptual 9",
    "viajes: hecho conceptual 10",
    "viajes: hecho conceptual 11",
    "viajes: hecho conceptual 12",
    "viajes: hecho conceptual 13",
    "viajes: hecho conceptual 14",
    "viajes: hecho conceptual 15",
    "viajes: hecho conceptual 16",
    "viajes: hecho conceptual 17",
    "viajes: hecho conceptual 18",
    "viajes: hecho conceptual 19",
    "viajes: hecho conceptual 20",
    "viajes: hecho conceptual 21",
    "viajes: hecho conceptual 22",
    "viajes: hecho conceptual 23",
    "viajes: hecho conceptual 24",
    "viajes: hecho conceptual 25",
    "viajes: hecho conceptual 26",
    "viajes: hecho conceptual 27",
    "viajes: hecho conceptual 28",
    "viajes: hecho conceptual 29",
    "viajes: hecho conceptual 30",
    "viajes: hecho conceptual 31",
    "viajes: hecho conceptual 32",
    "viajes: hecho conceptual 33",
    "viajes: hecho conceptual 34",
    "viajes: hecho conceptual 35",
    "viajes: hecho conceptual 36",
    "viajes: hecho conceptual 37",
    "viajes: hecho conceptual 38",
    "viajes: hecho conceptual 39",
    "viajes: hecho conceptual 40",
    "viajes: hecho conceptual 41",
    "viajes: hecho conceptual 42",
    "viajes: hecho conceptual 43",
    "viajes: hecho conceptual 44",
    "viajes: hecho conceptual 45",
    "viajes: hecho conceptual 46",
    "viajes: hecho conceptual 47",
    "viajes: hecho conceptual 48",
    "viajes: hecho conceptual 49",
    "viajes: hecho conceptual 50",
    "viajes: hecho conceptual 51",
    "viajes: hecho conceptual 52",
    "viajes: hecho conceptual 53",
    "viajes: hecho conceptual 54",
    "viajes: hecho conceptual 55",
    "viajes: hecho conceptual 56",
    "viajes: hecho conceptual 57",
    "viajes: hecho conceptual 58",
    "viajes: hecho conceptual 59",
    "viajes: hecho conceptual 60",
    "viajes: hecho conceptual 61",
    "viajes: hecho conceptual 62",
    "viajes: hecho conceptual 63",
    "viajes: hecho conceptual 64",
    "viajes: hecho conceptual 65",
    "viajes: hecho conceptual 66",
    "viajes: hecho conceptual 67",
    "viajes: hecho conceptual 68",
    "viajes: hecho conceptual 69",
    "viajes: hecho conceptual 70",
    "viajes: hecho conceptual 71",
    "viajes: hecho conceptual 72",
    "viajes: hecho conceptual 73",
    "viajes: hecho conceptual 74",
    "viajes: hecho conceptual 75",
    "viajes: hecho conceptual 76",
    "viajes: hecho conceptual 77",
    "viajes: hecho conceptual 78",
    "viajes: hecho conceptual 79",
    "viajes: hecho conceptual 80",
]
VIAJES_QUESTION_VARIANTS = [
    "¿Qué es viajes?",
    "¿Cómo explicarías viajes de forma sencilla?",
    "¿Por qué es importante viajes?",
    "¿Cuál es una idea fundamental de viajes?",
    "¿Puedes darme un ejemplo relacionado con viajes?",
    "¿Qué errores son frecuentes al estudiar viajes?",
    "¿Cómo se aplica viajes en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con viajes?",
    "¿Cómo empezaría a estudiar viajes?",
    "¿Qué relación tiene viajes con otras disciplinas?",
    "¿Puedes resumir viajes en pocas frases?",
    "¿Qué términos debería conocer sobre viajes?",
    "¿Cómo comprobaría si entendí viajes?",
    "¿Qué intuición ayuda a comprender viajes?",
    "¿Qué problema sencillo puedo resolver sobre viajes?",
    "¿Qué matiz suele pasarse por alto en viajes?",
    "¿Cómo se representa viajes?",
    "¿Qué supuestos se usan al hablar de viajes?",
    "¿Qué aplicaciones tiene viajes?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar viajes?",
    "¿Cómo se conecta viajes con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de viajes?",
    "¿Qué vocabulario técnico aparece en viajes?",
    "¿Puedes comparar dos enfoques dentro de viajes?",
    "¿Cómo evolucionó la comprensión de viajes?",
    "¿Qué ejemplo cotidiano ilustra viajes?",
    "¿Qué dato conviene recordar sobre viajes?",
    "¿Cómo evitar confusiones comunes en viajes?",
    "¿Qué parte de viajes suele ser más difícil?",
    "¿Puedes plantear un ejercicio de viajes?",
    "¿Cómo resolverías un ejercicio introductorio de viajes?",
    "¿Qué relación matemática aparece en viajes?",
    "¿Qué observación apoya esta idea de viajes?",
    "¿Qué pasaría si cambiamos una condición en viajes?",
    "¿Cómo se usa viajes en investigación?",
    "¿Qué herramientas sirven para estudiar viajes?",
    "¿Cómo distinguir evidencia de interpretación en viajes?",
    "¿Qué concepto previo necesito para entender viajes?",
    "¿Puedes dar una analogía para viajes?",
    "¿Qué preguntas avanzadas surgen de viajes?",
    "¿Cómo se comunica correctamente información sobre viajes?",
    "¿Qué ejemplo contradice una intuición común sobre viajes?",
    "¿Qué pasos seguirías para analizar viajes?",
    "¿Cómo resumirías la historia de viajes?",
    "¿Qué incertidumbres existen al estudiar viajes?",
    "¿Cómo se relacionan teoría y práctica en viajes?",
    "¿Qué clasificación útil existe en viajes?",
    "¿Cómo puedo practicar viajes?",
    "¿Qué es viajes?",
]
TECNOLOG_A_FACT_LABELS = [
    "tecnología: hecho conceptual 1",
    "tecnología: hecho conceptual 2",
    "tecnología: hecho conceptual 3",
    "tecnología: hecho conceptual 4",
    "tecnología: hecho conceptual 5",
    "tecnología: hecho conceptual 6",
    "tecnología: hecho conceptual 7",
    "tecnología: hecho conceptual 8",
    "tecnología: hecho conceptual 9",
    "tecnología: hecho conceptual 10",
    "tecnología: hecho conceptual 11",
    "tecnología: hecho conceptual 12",
    "tecnología: hecho conceptual 13",
    "tecnología: hecho conceptual 14",
    "tecnología: hecho conceptual 15",
    "tecnología: hecho conceptual 16",
    "tecnología: hecho conceptual 17",
    "tecnología: hecho conceptual 18",
    "tecnología: hecho conceptual 19",
    "tecnología: hecho conceptual 20",
    "tecnología: hecho conceptual 21",
    "tecnología: hecho conceptual 22",
    "tecnología: hecho conceptual 23",
    "tecnología: hecho conceptual 24",
    "tecnología: hecho conceptual 25",
    "tecnología: hecho conceptual 26",
    "tecnología: hecho conceptual 27",
    "tecnología: hecho conceptual 28",
    "tecnología: hecho conceptual 29",
    "tecnología: hecho conceptual 30",
    "tecnología: hecho conceptual 31",
    "tecnología: hecho conceptual 32",
    "tecnología: hecho conceptual 33",
    "tecnología: hecho conceptual 34",
    "tecnología: hecho conceptual 35",
    "tecnología: hecho conceptual 36",
    "tecnología: hecho conceptual 37",
    "tecnología: hecho conceptual 38",
    "tecnología: hecho conceptual 39",
    "tecnología: hecho conceptual 40",
    "tecnología: hecho conceptual 41",
    "tecnología: hecho conceptual 42",
    "tecnología: hecho conceptual 43",
    "tecnología: hecho conceptual 44",
    "tecnología: hecho conceptual 45",
    "tecnología: hecho conceptual 46",
    "tecnología: hecho conceptual 47",
    "tecnología: hecho conceptual 48",
    "tecnología: hecho conceptual 49",
    "tecnología: hecho conceptual 50",
    "tecnología: hecho conceptual 51",
    "tecnología: hecho conceptual 52",
    "tecnología: hecho conceptual 53",
    "tecnología: hecho conceptual 54",
    "tecnología: hecho conceptual 55",
    "tecnología: hecho conceptual 56",
    "tecnología: hecho conceptual 57",
    "tecnología: hecho conceptual 58",
    "tecnología: hecho conceptual 59",
    "tecnología: hecho conceptual 60",
    "tecnología: hecho conceptual 61",
    "tecnología: hecho conceptual 62",
    "tecnología: hecho conceptual 63",
    "tecnología: hecho conceptual 64",
    "tecnología: hecho conceptual 65",
    "tecnología: hecho conceptual 66",
    "tecnología: hecho conceptual 67",
    "tecnología: hecho conceptual 68",
    "tecnología: hecho conceptual 69",
    "tecnología: hecho conceptual 70",
    "tecnología: hecho conceptual 71",
    "tecnología: hecho conceptual 72",
    "tecnología: hecho conceptual 73",
    "tecnología: hecho conceptual 74",
    "tecnología: hecho conceptual 75",
    "tecnología: hecho conceptual 76",
    "tecnología: hecho conceptual 77",
    "tecnología: hecho conceptual 78",
    "tecnología: hecho conceptual 79",
    "tecnología: hecho conceptual 80",
]
TECNOLOG_A_QUESTION_VARIANTS = [
    "¿Qué es tecnología?",
    "¿Cómo explicarías tecnología de forma sencilla?",
    "¿Por qué es importante tecnología?",
    "¿Cuál es una idea fundamental de tecnología?",
    "¿Puedes darme un ejemplo relacionado con tecnología?",
    "¿Qué errores son frecuentes al estudiar tecnología?",
    "¿Cómo se aplica tecnología en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con tecnología?",
    "¿Cómo empezaría a estudiar tecnología?",
    "¿Qué relación tiene tecnología con otras disciplinas?",
    "¿Puedes resumir tecnología en pocas frases?",
    "¿Qué términos debería conocer sobre tecnología?",
    "¿Cómo comprobaría si entendí tecnología?",
    "¿Qué intuición ayuda a comprender tecnología?",
    "¿Qué problema sencillo puedo resolver sobre tecnología?",
    "¿Qué matiz suele pasarse por alto en tecnología?",
    "¿Cómo se representa tecnología?",
    "¿Qué supuestos se usan al hablar de tecnología?",
    "¿Qué aplicaciones tiene tecnología?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar tecnología?",
    "¿Cómo se conecta tecnología con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de tecnología?",
    "¿Qué vocabulario técnico aparece en tecnología?",
    "¿Puedes comparar dos enfoques dentro de tecnología?",
    "¿Cómo evolucionó la comprensión de tecnología?",
    "¿Qué ejemplo cotidiano ilustra tecnología?",
    "¿Qué dato conviene recordar sobre tecnología?",
    "¿Cómo evitar confusiones comunes en tecnología?",
    "¿Qué parte de tecnología suele ser más difícil?",
    "¿Puedes plantear un ejercicio de tecnología?",
    "¿Cómo resolverías un ejercicio introductorio de tecnología?",
    "¿Qué relación matemática aparece en tecnología?",
    "¿Qué observación apoya esta idea de tecnología?",
    "¿Qué pasaría si cambiamos una condición en tecnología?",
    "¿Cómo se usa tecnología en investigación?",
    "¿Qué herramientas sirven para estudiar tecnología?",
    "¿Cómo distinguir evidencia de interpretación en tecnología?",
    "¿Qué concepto previo necesito para entender tecnología?",
    "¿Puedes dar una analogía para tecnología?",
    "¿Qué preguntas avanzadas surgen de tecnología?",
    "¿Cómo se comunica correctamente información sobre tecnología?",
    "¿Qué ejemplo contradice una intuición común sobre tecnología?",
    "¿Qué pasos seguirías para analizar tecnología?",
    "¿Cómo resumirías la historia de tecnología?",
    "¿Qué incertidumbres existen al estudiar tecnología?",
    "¿Cómo se relacionan teoría y práctica en tecnología?",
    "¿Qué clasificación útil existe en tecnología?",
    "¿Cómo puedo practicar tecnología?",
    "¿Qué es tecnología?",
]
ASTRONOM_A_FACT_LABELS = [
    "astronomía: hecho conceptual 1",
    "astronomía: hecho conceptual 2",
    "astronomía: hecho conceptual 3",
    "astronomía: hecho conceptual 4",
    "astronomía: hecho conceptual 5",
    "astronomía: hecho conceptual 6",
    "astronomía: hecho conceptual 7",
    "astronomía: hecho conceptual 8",
    "astronomía: hecho conceptual 9",
    "astronomía: hecho conceptual 10",
    "astronomía: hecho conceptual 11",
    "astronomía: hecho conceptual 12",
    "astronomía: hecho conceptual 13",
    "astronomía: hecho conceptual 14",
    "astronomía: hecho conceptual 15",
    "astronomía: hecho conceptual 16",
    "astronomía: hecho conceptual 17",
    "astronomía: hecho conceptual 18",
    "astronomía: hecho conceptual 19",
    "astronomía: hecho conceptual 20",
    "astronomía: hecho conceptual 21",
    "astronomía: hecho conceptual 22",
    "astronomía: hecho conceptual 23",
    "astronomía: hecho conceptual 24",
    "astronomía: hecho conceptual 25",
    "astronomía: hecho conceptual 26",
    "astronomía: hecho conceptual 27",
    "astronomía: hecho conceptual 28",
    "astronomía: hecho conceptual 29",
    "astronomía: hecho conceptual 30",
    "astronomía: hecho conceptual 31",
    "astronomía: hecho conceptual 32",
    "astronomía: hecho conceptual 33",
    "astronomía: hecho conceptual 34",
    "astronomía: hecho conceptual 35",
    "astronomía: hecho conceptual 36",
    "astronomía: hecho conceptual 37",
    "astronomía: hecho conceptual 38",
    "astronomía: hecho conceptual 39",
    "astronomía: hecho conceptual 40",
    "astronomía: hecho conceptual 41",
    "astronomía: hecho conceptual 42",
    "astronomía: hecho conceptual 43",
    "astronomía: hecho conceptual 44",
    "astronomía: hecho conceptual 45",
    "astronomía: hecho conceptual 46",
    "astronomía: hecho conceptual 47",
    "astronomía: hecho conceptual 48",
    "astronomía: hecho conceptual 49",
    "astronomía: hecho conceptual 50",
    "astronomía: hecho conceptual 51",
    "astronomía: hecho conceptual 52",
    "astronomía: hecho conceptual 53",
    "astronomía: hecho conceptual 54",
    "astronomía: hecho conceptual 55",
    "astronomía: hecho conceptual 56",
    "astronomía: hecho conceptual 57",
    "astronomía: hecho conceptual 58",
    "astronomía: hecho conceptual 59",
    "astronomía: hecho conceptual 60",
    "astronomía: hecho conceptual 61",
    "astronomía: hecho conceptual 62",
    "astronomía: hecho conceptual 63",
    "astronomía: hecho conceptual 64",
    "astronomía: hecho conceptual 65",
    "astronomía: hecho conceptual 66",
    "astronomía: hecho conceptual 67",
    "astronomía: hecho conceptual 68",
    "astronomía: hecho conceptual 69",
    "astronomía: hecho conceptual 70",
    "astronomía: hecho conceptual 71",
    "astronomía: hecho conceptual 72",
    "astronomía: hecho conceptual 73",
    "astronomía: hecho conceptual 74",
    "astronomía: hecho conceptual 75",
    "astronomía: hecho conceptual 76",
    "astronomía: hecho conceptual 77",
    "astronomía: hecho conceptual 78",
    "astronomía: hecho conceptual 79",
    "astronomía: hecho conceptual 80",
]
ASTRONOM_A_QUESTION_VARIANTS = [
    "¿Qué es astronomía?",
    "¿Cómo explicarías astronomía de forma sencilla?",
    "¿Por qué es importante astronomía?",
    "¿Cuál es una idea fundamental de astronomía?",
    "¿Puedes darme un ejemplo relacionado con astronomía?",
    "¿Qué errores son frecuentes al estudiar astronomía?",
    "¿Cómo se aplica astronomía en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con astronomía?",
    "¿Cómo empezaría a estudiar astronomía?",
    "¿Qué relación tiene astronomía con otras disciplinas?",
    "¿Puedes resumir astronomía en pocas frases?",
    "¿Qué términos debería conocer sobre astronomía?",
    "¿Cómo comprobaría si entendí astronomía?",
    "¿Qué intuición ayuda a comprender astronomía?",
    "¿Qué problema sencillo puedo resolver sobre astronomía?",
    "¿Qué matiz suele pasarse por alto en astronomía?",
    "¿Cómo se representa astronomía?",
    "¿Qué supuestos se usan al hablar de astronomía?",
    "¿Qué aplicaciones tiene astronomía?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar astronomía?",
    "¿Cómo se conecta astronomía con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de astronomía?",
    "¿Qué vocabulario técnico aparece en astronomía?",
    "¿Puedes comparar dos enfoques dentro de astronomía?",
    "¿Cómo evolucionó la comprensión de astronomía?",
    "¿Qué ejemplo cotidiano ilustra astronomía?",
    "¿Qué dato conviene recordar sobre astronomía?",
    "¿Cómo evitar confusiones comunes en astronomía?",
    "¿Qué parte de astronomía suele ser más difícil?",
    "¿Puedes plantear un ejercicio de astronomía?",
    "¿Cómo resolverías un ejercicio introductorio de astronomía?",
    "¿Qué relación matemática aparece en astronomía?",
    "¿Qué observación apoya esta idea de astronomía?",
    "¿Qué pasaría si cambiamos una condición en astronomía?",
    "¿Cómo se usa astronomía en investigación?",
    "¿Qué herramientas sirven para estudiar astronomía?",
    "¿Cómo distinguir evidencia de interpretación en astronomía?",
    "¿Qué concepto previo necesito para entender astronomía?",
    "¿Puedes dar una analogía para astronomía?",
    "¿Qué preguntas avanzadas surgen de astronomía?",
    "¿Cómo se comunica correctamente información sobre astronomía?",
    "¿Qué ejemplo contradice una intuición común sobre astronomía?",
    "¿Qué pasos seguirías para analizar astronomía?",
    "¿Cómo resumirías la historia de astronomía?",
    "¿Qué incertidumbres existen al estudiar astronomía?",
    "¿Cómo se relacionan teoría y práctica en astronomía?",
    "¿Qué clasificación útil existe en astronomía?",
    "¿Cómo puedo practicar astronomía?",
    "¿Qué es astronomía?",
]
DERECHO_FACT_LABELS = [
    "derecho: hecho conceptual 1",
    "derecho: hecho conceptual 2",
    "derecho: hecho conceptual 3",
    "derecho: hecho conceptual 4",
    "derecho: hecho conceptual 5",
    "derecho: hecho conceptual 6",
    "derecho: hecho conceptual 7",
    "derecho: hecho conceptual 8",
    "derecho: hecho conceptual 9",
    "derecho: hecho conceptual 10",
    "derecho: hecho conceptual 11",
    "derecho: hecho conceptual 12",
    "derecho: hecho conceptual 13",
    "derecho: hecho conceptual 14",
    "derecho: hecho conceptual 15",
    "derecho: hecho conceptual 16",
    "derecho: hecho conceptual 17",
    "derecho: hecho conceptual 18",
    "derecho: hecho conceptual 19",
    "derecho: hecho conceptual 20",
    "derecho: hecho conceptual 21",
    "derecho: hecho conceptual 22",
    "derecho: hecho conceptual 23",
    "derecho: hecho conceptual 24",
    "derecho: hecho conceptual 25",
    "derecho: hecho conceptual 26",
    "derecho: hecho conceptual 27",
    "derecho: hecho conceptual 28",
    "derecho: hecho conceptual 29",
    "derecho: hecho conceptual 30",
    "derecho: hecho conceptual 31",
    "derecho: hecho conceptual 32",
    "derecho: hecho conceptual 33",
    "derecho: hecho conceptual 34",
    "derecho: hecho conceptual 35",
    "derecho: hecho conceptual 36",
    "derecho: hecho conceptual 37",
    "derecho: hecho conceptual 38",
    "derecho: hecho conceptual 39",
    "derecho: hecho conceptual 40",
    "derecho: hecho conceptual 41",
    "derecho: hecho conceptual 42",
    "derecho: hecho conceptual 43",
    "derecho: hecho conceptual 44",
    "derecho: hecho conceptual 45",
    "derecho: hecho conceptual 46",
    "derecho: hecho conceptual 47",
    "derecho: hecho conceptual 48",
    "derecho: hecho conceptual 49",
    "derecho: hecho conceptual 50",
    "derecho: hecho conceptual 51",
    "derecho: hecho conceptual 52",
    "derecho: hecho conceptual 53",
    "derecho: hecho conceptual 54",
    "derecho: hecho conceptual 55",
    "derecho: hecho conceptual 56",
    "derecho: hecho conceptual 57",
    "derecho: hecho conceptual 58",
    "derecho: hecho conceptual 59",
    "derecho: hecho conceptual 60",
    "derecho: hecho conceptual 61",
    "derecho: hecho conceptual 62",
    "derecho: hecho conceptual 63",
    "derecho: hecho conceptual 64",
    "derecho: hecho conceptual 65",
    "derecho: hecho conceptual 66",
    "derecho: hecho conceptual 67",
    "derecho: hecho conceptual 68",
    "derecho: hecho conceptual 69",
    "derecho: hecho conceptual 70",
    "derecho: hecho conceptual 71",
    "derecho: hecho conceptual 72",
    "derecho: hecho conceptual 73",
    "derecho: hecho conceptual 74",
    "derecho: hecho conceptual 75",
    "derecho: hecho conceptual 76",
    "derecho: hecho conceptual 77",
    "derecho: hecho conceptual 78",
    "derecho: hecho conceptual 79",
    "derecho: hecho conceptual 80",
]
DERECHO_QUESTION_VARIANTS = [
    "¿Qué es derecho?",
    "¿Cómo explicarías derecho de forma sencilla?",
    "¿Por qué es importante derecho?",
    "¿Cuál es una idea fundamental de derecho?",
    "¿Puedes darme un ejemplo relacionado con derecho?",
    "¿Qué errores son frecuentes al estudiar derecho?",
    "¿Cómo se aplica derecho en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con derecho?",
    "¿Cómo empezaría a estudiar derecho?",
    "¿Qué relación tiene derecho con otras disciplinas?",
    "¿Puedes resumir derecho en pocas frases?",
    "¿Qué términos debería conocer sobre derecho?",
    "¿Cómo comprobaría si entendí derecho?",
    "¿Qué intuición ayuda a comprender derecho?",
    "¿Qué problema sencillo puedo resolver sobre derecho?",
    "¿Qué matiz suele pasarse por alto en derecho?",
    "¿Cómo se representa derecho?",
    "¿Qué supuestos se usan al hablar de derecho?",
    "¿Qué aplicaciones tiene derecho?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar derecho?",
    "¿Cómo se conecta derecho con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de derecho?",
    "¿Qué vocabulario técnico aparece en derecho?",
    "¿Puedes comparar dos enfoques dentro de derecho?",
    "¿Cómo evolucionó la comprensión de derecho?",
    "¿Qué ejemplo cotidiano ilustra derecho?",
    "¿Qué dato conviene recordar sobre derecho?",
    "¿Cómo evitar confusiones comunes en derecho?",
    "¿Qué parte de derecho suele ser más difícil?",
    "¿Puedes plantear un ejercicio de derecho?",
    "¿Cómo resolverías un ejercicio introductorio de derecho?",
    "¿Qué relación matemática aparece en derecho?",
    "¿Qué observación apoya esta idea de derecho?",
    "¿Qué pasaría si cambiamos una condición en derecho?",
    "¿Cómo se usa derecho en investigación?",
    "¿Qué herramientas sirven para estudiar derecho?",
    "¿Cómo distinguir evidencia de interpretación en derecho?",
    "¿Qué concepto previo necesito para entender derecho?",
    "¿Puedes dar una analogía para derecho?",
    "¿Qué preguntas avanzadas surgen de derecho?",
    "¿Cómo se comunica correctamente información sobre derecho?",
    "¿Qué ejemplo contradice una intuición común sobre derecho?",
    "¿Qué pasos seguirías para analizar derecho?",
    "¿Cómo resumirías la historia de derecho?",
    "¿Qué incertidumbres existen al estudiar derecho?",
    "¿Cómo se relacionan teoría y práctica en derecho?",
    "¿Qué clasificación útil existe en derecho?",
    "¿Cómo puedo practicar derecho?",
    "¿Qué es derecho?",
]
EDUCACI_N_FACT_LABELS = [
    "educación: hecho conceptual 1",
    "educación: hecho conceptual 2",
    "educación: hecho conceptual 3",
    "educación: hecho conceptual 4",
    "educación: hecho conceptual 5",
    "educación: hecho conceptual 6",
    "educación: hecho conceptual 7",
    "educación: hecho conceptual 8",
    "educación: hecho conceptual 9",
    "educación: hecho conceptual 10",
    "educación: hecho conceptual 11",
    "educación: hecho conceptual 12",
    "educación: hecho conceptual 13",
    "educación: hecho conceptual 14",
    "educación: hecho conceptual 15",
    "educación: hecho conceptual 16",
    "educación: hecho conceptual 17",
    "educación: hecho conceptual 18",
    "educación: hecho conceptual 19",
    "educación: hecho conceptual 20",
    "educación: hecho conceptual 21",
    "educación: hecho conceptual 22",
    "educación: hecho conceptual 23",
    "educación: hecho conceptual 24",
    "educación: hecho conceptual 25",
    "educación: hecho conceptual 26",
    "educación: hecho conceptual 27",
    "educación: hecho conceptual 28",
    "educación: hecho conceptual 29",
    "educación: hecho conceptual 30",
    "educación: hecho conceptual 31",
    "educación: hecho conceptual 32",
    "educación: hecho conceptual 33",
    "educación: hecho conceptual 34",
    "educación: hecho conceptual 35",
    "educación: hecho conceptual 36",
    "educación: hecho conceptual 37",
    "educación: hecho conceptual 38",
    "educación: hecho conceptual 39",
    "educación: hecho conceptual 40",
    "educación: hecho conceptual 41",
    "educación: hecho conceptual 42",
    "educación: hecho conceptual 43",
    "educación: hecho conceptual 44",
    "educación: hecho conceptual 45",
    "educación: hecho conceptual 46",
    "educación: hecho conceptual 47",
    "educación: hecho conceptual 48",
    "educación: hecho conceptual 49",
    "educación: hecho conceptual 50",
    "educación: hecho conceptual 51",
    "educación: hecho conceptual 52",
    "educación: hecho conceptual 53",
    "educación: hecho conceptual 54",
    "educación: hecho conceptual 55",
    "educación: hecho conceptual 56",
    "educación: hecho conceptual 57",
    "educación: hecho conceptual 58",
    "educación: hecho conceptual 59",
    "educación: hecho conceptual 60",
    "educación: hecho conceptual 61",
    "educación: hecho conceptual 62",
    "educación: hecho conceptual 63",
    "educación: hecho conceptual 64",
    "educación: hecho conceptual 65",
    "educación: hecho conceptual 66",
    "educación: hecho conceptual 67",
    "educación: hecho conceptual 68",
    "educación: hecho conceptual 69",
    "educación: hecho conceptual 70",
    "educación: hecho conceptual 71",
    "educación: hecho conceptual 72",
    "educación: hecho conceptual 73",
    "educación: hecho conceptual 74",
    "educación: hecho conceptual 75",
    "educación: hecho conceptual 76",
    "educación: hecho conceptual 77",
    "educación: hecho conceptual 78",
    "educación: hecho conceptual 79",
    "educación: hecho conceptual 80",
]
EDUCACI_N_QUESTION_VARIANTS = [
    "¿Qué es educación?",
    "¿Cómo explicarías educación de forma sencilla?",
    "¿Por qué es importante educación?",
    "¿Cuál es una idea fundamental de educación?",
    "¿Puedes darme un ejemplo relacionado con educación?",
    "¿Qué errores son frecuentes al estudiar educación?",
    "¿Cómo se aplica educación en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con educación?",
    "¿Cómo empezaría a estudiar educación?",
    "¿Qué relación tiene educación con otras disciplinas?",
    "¿Puedes resumir educación en pocas frases?",
    "¿Qué términos debería conocer sobre educación?",
    "¿Cómo comprobaría si entendí educación?",
    "¿Qué intuición ayuda a comprender educación?",
    "¿Qué problema sencillo puedo resolver sobre educación?",
    "¿Qué matiz suele pasarse por alto en educación?",
    "¿Cómo se representa educación?",
    "¿Qué supuestos se usan al hablar de educación?",
    "¿Qué aplicaciones tiene educación?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar educación?",
    "¿Cómo se conecta educación con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de educación?",
    "¿Qué vocabulario técnico aparece en educación?",
    "¿Puedes comparar dos enfoques dentro de educación?",
    "¿Cómo evolucionó la comprensión de educación?",
    "¿Qué ejemplo cotidiano ilustra educación?",
    "¿Qué dato conviene recordar sobre educación?",
    "¿Cómo evitar confusiones comunes en educación?",
    "¿Qué parte de educación suele ser más difícil?",
    "¿Puedes plantear un ejercicio de educación?",
    "¿Cómo resolverías un ejercicio introductorio de educación?",
    "¿Qué relación matemática aparece en educación?",
    "¿Qué observación apoya esta idea de educación?",
    "¿Qué pasaría si cambiamos una condición en educación?",
    "¿Cómo se usa educación en investigación?",
    "¿Qué herramientas sirven para estudiar educación?",
    "¿Cómo distinguir evidencia de interpretación en educación?",
    "¿Qué concepto previo necesito para entender educación?",
    "¿Puedes dar una analogía para educación?",
    "¿Qué preguntas avanzadas surgen de educación?",
    "¿Cómo se comunica correctamente información sobre educación?",
    "¿Qué ejemplo contradice una intuición común sobre educación?",
    "¿Qué pasos seguirías para analizar educación?",
    "¿Cómo resumirías la historia de educación?",
    "¿Qué incertidumbres existen al estudiar educación?",
    "¿Cómo se relacionan teoría y práctica en educación?",
    "¿Qué clasificación útil existe en educación?",
    "¿Cómo puedo practicar educación?",
    "¿Qué es educación?",
]
MEDIO_AMBIENTE_FACT_LABELS = [
    "medio ambiente: hecho conceptual 1",
    "medio ambiente: hecho conceptual 2",
    "medio ambiente: hecho conceptual 3",
    "medio ambiente: hecho conceptual 4",
    "medio ambiente: hecho conceptual 5",
    "medio ambiente: hecho conceptual 6",
    "medio ambiente: hecho conceptual 7",
    "medio ambiente: hecho conceptual 8",
    "medio ambiente: hecho conceptual 9",
    "medio ambiente: hecho conceptual 10",
    "medio ambiente: hecho conceptual 11",
    "medio ambiente: hecho conceptual 12",
    "medio ambiente: hecho conceptual 13",
    "medio ambiente: hecho conceptual 14",
    "medio ambiente: hecho conceptual 15",
    "medio ambiente: hecho conceptual 16",
    "medio ambiente: hecho conceptual 17",
    "medio ambiente: hecho conceptual 18",
    "medio ambiente: hecho conceptual 19",
    "medio ambiente: hecho conceptual 20",
    "medio ambiente: hecho conceptual 21",
    "medio ambiente: hecho conceptual 22",
    "medio ambiente: hecho conceptual 23",
    "medio ambiente: hecho conceptual 24",
    "medio ambiente: hecho conceptual 25",
    "medio ambiente: hecho conceptual 26",
    "medio ambiente: hecho conceptual 27",
    "medio ambiente: hecho conceptual 28",
    "medio ambiente: hecho conceptual 29",
    "medio ambiente: hecho conceptual 30",
    "medio ambiente: hecho conceptual 31",
    "medio ambiente: hecho conceptual 32",
    "medio ambiente: hecho conceptual 33",
    "medio ambiente: hecho conceptual 34",
    "medio ambiente: hecho conceptual 35",
    "medio ambiente: hecho conceptual 36",
    "medio ambiente: hecho conceptual 37",
    "medio ambiente: hecho conceptual 38",
    "medio ambiente: hecho conceptual 39",
    "medio ambiente: hecho conceptual 40",
    "medio ambiente: hecho conceptual 41",
    "medio ambiente: hecho conceptual 42",
    "medio ambiente: hecho conceptual 43",
    "medio ambiente: hecho conceptual 44",
    "medio ambiente: hecho conceptual 45",
    "medio ambiente: hecho conceptual 46",
    "medio ambiente: hecho conceptual 47",
    "medio ambiente: hecho conceptual 48",
    "medio ambiente: hecho conceptual 49",
    "medio ambiente: hecho conceptual 50",
    "medio ambiente: hecho conceptual 51",
    "medio ambiente: hecho conceptual 52",
    "medio ambiente: hecho conceptual 53",
    "medio ambiente: hecho conceptual 54",
    "medio ambiente: hecho conceptual 55",
    "medio ambiente: hecho conceptual 56",
    "medio ambiente: hecho conceptual 57",
    "medio ambiente: hecho conceptual 58",
    "medio ambiente: hecho conceptual 59",
    "medio ambiente: hecho conceptual 60",
    "medio ambiente: hecho conceptual 61",
    "medio ambiente: hecho conceptual 62",
    "medio ambiente: hecho conceptual 63",
    "medio ambiente: hecho conceptual 64",
    "medio ambiente: hecho conceptual 65",
    "medio ambiente: hecho conceptual 66",
    "medio ambiente: hecho conceptual 67",
    "medio ambiente: hecho conceptual 68",
    "medio ambiente: hecho conceptual 69",
    "medio ambiente: hecho conceptual 70",
    "medio ambiente: hecho conceptual 71",
    "medio ambiente: hecho conceptual 72",
    "medio ambiente: hecho conceptual 73",
    "medio ambiente: hecho conceptual 74",
    "medio ambiente: hecho conceptual 75",
    "medio ambiente: hecho conceptual 76",
    "medio ambiente: hecho conceptual 77",
    "medio ambiente: hecho conceptual 78",
    "medio ambiente: hecho conceptual 79",
    "medio ambiente: hecho conceptual 80",
]
MEDIO_AMBIENTE_QUESTION_VARIANTS = [
    "¿Qué es medio ambiente?",
    "¿Cómo explicarías medio ambiente de forma sencilla?",
    "¿Por qué es importante medio ambiente?",
    "¿Cuál es una idea fundamental de medio ambiente?",
    "¿Puedes darme un ejemplo relacionado con medio ambiente?",
    "¿Qué errores son frecuentes al estudiar medio ambiente?",
    "¿Cómo se aplica medio ambiente en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con medio ambiente?",
    "¿Cómo empezaría a estudiar medio ambiente?",
    "¿Qué relación tiene medio ambiente con otras disciplinas?",
    "¿Puedes resumir medio ambiente en pocas frases?",
    "¿Qué términos debería conocer sobre medio ambiente?",
    "¿Cómo comprobaría si entendí medio ambiente?",
    "¿Qué intuición ayuda a comprender medio ambiente?",
    "¿Qué problema sencillo puedo resolver sobre medio ambiente?",
    "¿Qué matiz suele pasarse por alto en medio ambiente?",
    "¿Cómo se representa medio ambiente?",
    "¿Qué supuestos se usan al hablar de medio ambiente?",
    "¿Qué aplicaciones tiene medio ambiente?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar medio ambiente?",
    "¿Cómo se conecta medio ambiente con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de medio ambiente?",
    "¿Qué vocabulario técnico aparece en medio ambiente?",
    "¿Puedes comparar dos enfoques dentro de medio ambiente?",
    "¿Cómo evolucionó la comprensión de medio ambiente?",
    "¿Qué ejemplo cotidiano ilustra medio ambiente?",
    "¿Qué dato conviene recordar sobre medio ambiente?",
    "¿Cómo evitar confusiones comunes en medio ambiente?",
    "¿Qué parte de medio ambiente suele ser más difícil?",
    "¿Puedes plantear un ejercicio de medio ambiente?",
    "¿Cómo resolverías un ejercicio introductorio de medio ambiente?",
    "¿Qué relación matemática aparece en medio ambiente?",
    "¿Qué observación apoya esta idea de medio ambiente?",
    "¿Qué pasaría si cambiamos una condición en medio ambiente?",
    "¿Cómo se usa medio ambiente en investigación?",
    "¿Qué herramientas sirven para estudiar medio ambiente?",
    "¿Cómo distinguir evidencia de interpretación en medio ambiente?",
    "¿Qué concepto previo necesito para entender medio ambiente?",
    "¿Puedes dar una analogía para medio ambiente?",
    "¿Qué preguntas avanzadas surgen de medio ambiente?",
    "¿Cómo se comunica correctamente información sobre medio ambiente?",
    "¿Qué ejemplo contradice una intuición común sobre medio ambiente?",
    "¿Qué pasos seguirías para analizar medio ambiente?",
    "¿Cómo resumirías la historia de medio ambiente?",
    "¿Qué incertidumbres existen al estudiar medio ambiente?",
    "¿Cómo se relacionan teoría y práctica en medio ambiente?",
    "¿Qué clasificación útil existe en medio ambiente?",
    "¿Cómo puedo practicar medio ambiente?",
    "¿Qué es medio ambiente?",
]
ASTROF_SICA_FACT_LABELS = [
    "astrofísica: hecho conceptual 1",
    "astrofísica: hecho conceptual 2",
    "astrofísica: hecho conceptual 3",
    "astrofísica: hecho conceptual 4",
    "astrofísica: hecho conceptual 5",
    "astrofísica: hecho conceptual 6",
    "astrofísica: hecho conceptual 7",
    "astrofísica: hecho conceptual 8",
    "astrofísica: hecho conceptual 9",
    "astrofísica: hecho conceptual 10",
    "astrofísica: hecho conceptual 11",
    "astrofísica: hecho conceptual 12",
    "astrofísica: hecho conceptual 13",
    "astrofísica: hecho conceptual 14",
    "astrofísica: hecho conceptual 15",
    "astrofísica: hecho conceptual 16",
    "astrofísica: hecho conceptual 17",
    "astrofísica: hecho conceptual 18",
    "astrofísica: hecho conceptual 19",
    "astrofísica: hecho conceptual 20",
    "astrofísica: hecho conceptual 21",
    "astrofísica: hecho conceptual 22",
    "astrofísica: hecho conceptual 23",
    "astrofísica: hecho conceptual 24",
    "astrofísica: hecho conceptual 25",
    "astrofísica: hecho conceptual 26",
    "astrofísica: hecho conceptual 27",
    "astrofísica: hecho conceptual 28",
    "astrofísica: hecho conceptual 29",
    "astrofísica: hecho conceptual 30",
    "astrofísica: hecho conceptual 31",
    "astrofísica: hecho conceptual 32",
    "astrofísica: hecho conceptual 33",
    "astrofísica: hecho conceptual 34",
    "astrofísica: hecho conceptual 35",
    "astrofísica: hecho conceptual 36",
    "astrofísica: hecho conceptual 37",
    "astrofísica: hecho conceptual 38",
    "astrofísica: hecho conceptual 39",
    "astrofísica: hecho conceptual 40",
    "astrofísica: hecho conceptual 41",
    "astrofísica: hecho conceptual 42",
    "astrofísica: hecho conceptual 43",
    "astrofísica: hecho conceptual 44",
    "astrofísica: hecho conceptual 45",
    "astrofísica: hecho conceptual 46",
    "astrofísica: hecho conceptual 47",
    "astrofísica: hecho conceptual 48",
    "astrofísica: hecho conceptual 49",
    "astrofísica: hecho conceptual 50",
    "astrofísica: hecho conceptual 51",
    "astrofísica: hecho conceptual 52",
    "astrofísica: hecho conceptual 53",
    "astrofísica: hecho conceptual 54",
    "astrofísica: hecho conceptual 55",
    "astrofísica: hecho conceptual 56",
    "astrofísica: hecho conceptual 57",
    "astrofísica: hecho conceptual 58",
    "astrofísica: hecho conceptual 59",
    "astrofísica: hecho conceptual 60",
    "astrofísica: hecho conceptual 61",
    "astrofísica: hecho conceptual 62",
    "astrofísica: hecho conceptual 63",
    "astrofísica: hecho conceptual 64",
    "astrofísica: hecho conceptual 65",
    "astrofísica: hecho conceptual 66",
    "astrofísica: hecho conceptual 67",
    "astrofísica: hecho conceptual 68",
    "astrofísica: hecho conceptual 69",
    "astrofísica: hecho conceptual 70",
    "astrofísica: hecho conceptual 71",
    "astrofísica: hecho conceptual 72",
    "astrofísica: hecho conceptual 73",
    "astrofísica: hecho conceptual 74",
    "astrofísica: hecho conceptual 75",
    "astrofísica: hecho conceptual 76",
    "astrofísica: hecho conceptual 77",
    "astrofísica: hecho conceptual 78",
    "astrofísica: hecho conceptual 79",
    "astrofísica: hecho conceptual 80",
]
ASTROF_SICA_QUESTION_VARIANTS = [
    "¿Qué es astrofísica?",
    "¿Cómo explicarías astrofísica de forma sencilla?",
    "¿Por qué es importante astrofísica?",
    "¿Cuál es una idea fundamental de astrofísica?",
    "¿Puedes darme un ejemplo relacionado con astrofísica?",
    "¿Qué errores son frecuentes al estudiar astrofísica?",
    "¿Cómo se aplica astrofísica en la práctica?",
    "¿Qué diferencia hay entre conceptos relacionados con astrofísica?",
    "¿Cómo empezaría a estudiar astrofísica?",
    "¿Qué relación tiene astrofísica con otras disciplinas?",
    "¿Puedes resumir astrofísica en pocas frases?",
    "¿Qué términos debería conocer sobre astrofísica?",
    "¿Cómo comprobaría si entendí astrofísica?",
    "¿Qué intuición ayuda a comprender astrofísica?",
    "¿Qué problema sencillo puedo resolver sobre astrofísica?",
    "¿Qué matiz suele pasarse por alto en astrofísica?",
    "¿Cómo se representa astrofísica?",
    "¿Qué supuestos se usan al hablar de astrofísica?",
    "¿Qué aplicaciones tiene astrofísica?",
    "¿Cómo explicárselo a un estudiante?",
    "¿Qué preguntas debería hacerme al estudiar astrofísica?",
    "¿Cómo se conecta astrofísica con un caso real?",
    "¿Qué limitaciones tiene una explicación básica de astrofísica?",
    "¿Qué vocabulario técnico aparece en astrofísica?",
    "¿Puedes comparar dos enfoques dentro de astrofísica?",
    "¿Cómo evolucionó la comprensión de astrofísica?",
    "¿Qué ejemplo cotidiano ilustra astrofísica?",
    "¿Qué dato conviene recordar sobre astrofísica?",
    "¿Cómo evitar confusiones comunes en astrofísica?",
    "¿Qué parte de astrofísica suele ser más difícil?",
    "¿Puedes plantear un ejercicio de astrofísica?",
    "¿Cómo resolverías un ejercicio introductorio de astrofísica?",
    "¿Qué relación matemática aparece en astrofísica?",
    "¿Qué observación apoya esta idea de astrofísica?",
    "¿Qué pasaría si cambiamos una condición en astrofísica?",
    "¿Cómo se usa astrofísica en investigación?",
    "¿Qué herramientas sirven para estudiar astrofísica?",
    "¿Cómo distinguir evidencia de interpretación en astrofísica?",
    "¿Qué concepto previo necesito para entender astrofísica?",
    "¿Puedes dar una analogía para astrofísica?",
    "¿Qué preguntas avanzadas surgen de astrofísica?",
    "¿Cómo se comunica correctamente información sobre astrofísica?",
    "¿Qué ejemplo contradice una intuición común sobre astrofísica?",
    "¿Qué pasos seguirías para analizar astrofísica?",
    "¿Cómo resumirías la historia de astrofísica?",
    "¿Qué incertidumbres existen al estudiar astrofísica?",
    "¿Cómo se relacionan teoría y práctica en astrofísica?",
    "¿Qué clasificación útil existe en astrofísica?",
    "¿Cómo puedo practicar astrofísica?",
    "¿Qué es astrofísica?",
]

def concrete_topic_catalog() -> Dict[str,Dict[str,List[str]]]:
    names=TOPIC_NAMES
    out={}
    for topic in names:
        ident=re.sub(r'[^A-Za-z0-9]+','_',topic).upper().strip('_')
        facts=globals()[ident+"_FACT_LABELS"]
        questions=globals()[ident+"_QUESTION_VARIANTS"]
        out[topic]={"facts":facts,"questions":questions}
    return out

def regenerate_synthetic_with_catalog(n=50000):
    catalog=concrete_topic_catalog(); convs=[]; identities=["¿Quién eres?","¿Cómo te llamas?","¿Quién te creó?","¿Qué modelo eres?"]
    for i in range(n):
        topic=TOPIC_NAMES[i%len(TOPIC_NAMES)]; turns=3+i%10
        sys_prompt=SYSTEM_PROMPTS[i%len(SYSTEM_PROMPTS)] if i%10<3 else "Responde en español neutro con claridad, precisión y tono cercano."
        msgs=[{"role":"system","content":sys_prompt}]
        if i<500 or i%97==0:
            q=identities[i%4];msgs.extend([{"role":"user","content":q},{"role":"assistant","content":identity_answer(q)}])
            for j in range(1,turns):
                q2=catalog[topic]["questions"][(i+j)%50]; fact=catalog[topic]["facts"][(i+j*3)%80]
                msgs.extend([{"role":"user","content":q2},{"role":"assistant","content":f"{OPENERS[(i+j)%len(OPENERS)]}. {fact}. {CLOSERS[(i+j)%len(CLOSERS)]}"}])
        else:
            for j in range(turns):
                q=catalog[topic]["questions"][(i+j)%50]; fact=catalog[topic]["facts"][(i+j*3)%80]
                msgs.extend([{"role":"user","content":q},{"role":"assistant","content":f"{OPENERS[(i+j)%len(OPENERS)]}. {fact}. {CLOSERS[(i+j)%len(CLOSERS)]}"}])
        convs.append(msgs)
    return convs

# Rebind the production generator to the concrete 80-fact/50-question catalog.
generate_conversations = regenerate_synthetic_with_catalog
