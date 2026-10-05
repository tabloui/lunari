# ============================================================
# AQUA 3.0.0 - FUSION (solo LLaMA, adios GPT-2)
# ~300M params | RoPE + SwiGLU + RMSNorm + GQA | chat-first | 12h Kaggle
# ============================================================
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, time, math, random, signal, glob, shutil, threading, queue, traceback, gc, re
from collections import Counter
import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoModelForCausalLM, LlamaConfig

try:
    import psutil
    # el limite de 12h de Kaggle cuenta desde que arranca la sesion, no desde esta celda
    T0 = psutil.Process(os.getpid()).create_time()
except Exception:
    T0 = time.time()

# ======================= CONFIG =======================
LIMITE_H = float(os.environ.get("AQUA_HOURS", 11.5))      # margen de 30 min para subir a HF
HORAS_SFT = float(os.environ.get("AQUA_SFT_HOURS", 1.5))  # ultima parte: fine-tuning de chat
USAR_COMPILE = os.environ.get("AQUA_COMPILE", "1") == "1"
SEED = 42
VOCAB = 32000
BLOCK = 1024               # contexto
TOK_PASO = 16384           # tokens por paso de optimizador
LR_PRE_MAX, LR_PRE_MIN, WARMUP = 4e-4, 4e-5, 400
CHAT_MIX0, CHAT_MIX1 = 0.15, 0.50   # % de bloques de chat: al inicio / al final del pre-entreno
SINTETICOS = 30000         # pares de aritmetica (0 para desactivar)
EXTRA_CHAT = True          # alpaca-spanish como pares de chat
SHAREGPT = True            # conversaciones reales (multi-turno)
SHAREGPT_REP = 3           # veces que se repite ShareGPT en train (es poco y es de lo mejor)
SFT_LEN, SFT_MAXB, SFT_EPOCAS_MAX = BLOCK, 64, 3
LR_SFT_MAX, LR_SFT_MIN, SFT_WARM = 4e-5, 4e-6, 40
EVAL_CADA = 300            # pasos de SFT entre validaciones
CKPT_MIN, SUBIR_MIN = 20, 60
RESUMIR_DESDE_HF = False   # True: si no hay ckpt local, baja ckpt.pt del repo
ROL_USER, ROL_BOT = "USUARIO", "ASISTENTE"
ARCH = "llama300m-v4"
REPO_ID = os.environ.get("AQUA_REPO", "aquas-modela/Aqua-3.0.0")   # repo NUEVO en HF
GEN = dict(temperature=0.78, top_p=0.92, top_k=50, repetition_penalty=1.08,
           no_repeat_ngram_size=3)                                   # sampling para chat

# Fuentes de texto web en espanol (las que fallen se ignoran solas)
FUENTES = [
    ("HuggingFaceFW/fineweb-2", "spa_Latn", 0.50),
    ("wikimedia/wikipedia", "20231101.es", 0.20),
    ("uonlp/CulturaX", "es", 0.15),
    ("oscar-corpus/OSCAR-2301", "es", 0.10),
    ("allenai/c4", "es", 0.05),
]

SALIDA = "/kaggle/working"
DATOS = os.path.join(SALIDA, "dataset_1m.json")
HF_DATA = os.path.join(SALIDA, "aqua_conversations.jsonl")
CONFIG_PY = os.path.join(SALIDA, "config.py")
TOK_DIR = os.path.join(SALIDA, "tokenizer")
CKPT = os.path.join(SALIDA, "ckpt.pt")
dev = "cuda" if torch.cuda.is_available() else "cpu"
usa_amp = dev == "cuda"
# bf16 solo en Ampere+ (en T4/P100 seria emulado y lentisimo): ahi se usa fp16 + GradScaler
AMP_DTYPE = torch.bfloat16 if usa_amp and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
LIMITE = LIMITE_H * 3600
random.seed(SEED)
torch.manual_seed(SEED)
if usa_amp:
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
print(f"dev={dev} | amp={str(AMP_DTYPE).split('.')[-1]} | sesion={(time.time() - T0) / 60:.0f} min "
      f"| limite={LIMITE_H}h | block={BLOCK}")

# ======================= HUGGING FACE =======================
REPO_HF, api, USER = None, None, None
try:
    from huggingface_hub import HfApi, login
    from kaggle_secrets import UserSecretsClient
    login(token=UserSecretsClient().get_secret("HF_TOKEN"))
    api = HfApi()
    USER = api.whoami()["name"]
    REPO_HF = REPO_ID
    print(f"HF OK: {USER} -> {REPO_HF}")
except Exception as e:
    print(f"HF aviso: {e}")
if api:
    try:
        api.create_repo(REPO_HF, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"HF aviso create_repo ({type(e).__name__}): sigo, el repo ya deberia existir")

def subir_pequeno(path, dest):
    """Sube un archivo chico (sincrono). Devuelve True/False."""
    if not REPO_HF or not os.path.exists(path):
        return False
    try:
        api.upload_file(path_or_fileobj=path, path_in_repo=dest, repo_id=REPO_HF, repo_type="model")
        print(f"[HF] subido: {dest}")
        return True
    except Exception as e:
        print(f"[HF] error {dest}: {e}")
        return False

lock_ckpt = threading.Lock()

def subir_ckpt_async():
    """Sube ckpt.pt en segundo plano (no frena el entrenamiento)."""
    if not REPO_HF or not os.path.exists(CKPT) or lock_ckpt.locked():
        return
    def _run():
        with lock_ckpt:
            try:
                api.upload_file(path_or_fileobj=CKPT, path_in_repo="ckpt.pt",
                                repo_id=REPO_HF, repo_type="model")
                print("[HF] ckpt subido")
            except Exception as e:
                print(f"[HF] error: {e}")
    threading.Thread(target=_run, daemon=True).start()

# ======================= RESUMIR =======================
if not os.path.exists(CKPT):
    previos = glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
    if previos:
        prev = os.path.dirname(previos[0])
        shutil.copy2(previos[0], CKPT)
        if os.path.isdir(os.path.join(prev, "tokenizer")):
            shutil.copytree(os.path.join(prev, "tokenizer"), TOK_DIR, dirs_exist_ok=True)
        print(f"Reanudando desde {prev}")
    elif RESUMIR_DESDE_HF and REPO_HF:
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(REPO_HF, "ckpt.pt", local_dir=SALIDA)
            os.makedirs(TOK_DIR, exist_ok=True)
            for f in ("vocab.json", "merges.txt"):
                hf_hub_download(REPO_HF, f, local_dir=TOK_DIR)
            print("Reanudando desde HF")
        except Exception as e:
            print(f"No pude reanudar desde HF: {e}")

# ======================= LIMPIEZA DE CONVERSACIONES =======================
RE_W = re.compile(r"\w+", re.UNICODE)

def normalize_text(x):
    """Limpia espacios pero CONSERVA saltos de linea (listas, codigo, parrafos)."""
    if not isinstance(x, str):
        return ""
    x = x.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"[ \t\u00a0]+", " ", x)
    x = re.sub(r" ?\n ?", "\n", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()

def words(x):
    return RE_W.findall(x.lower())

def repeated_ngram_ratio(ws, n=3):
    if len(ws) < n * 2:
        return 0.0
    grams = [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]
    counts = Counter(grams)
    return sum(c for c in counts.values() if c > 1) / max(1, len(grams))

def quality_pair(user, assistant):
    u, a = normalize_text(user), normalize_text(assistant)
    if len(u) < 2 or len(a) < 2 or len(u) > 3000 or len(a) > 8000:
        return None
    uw, aw = words(u), words(a)
    nu, na = len(uw), len(aw)
    if nu == 0 or na == 0 or nu + na < 4:           # deja pasar "hola" -> "¡Hola! ¿En qué ayudo?"
        return None
    ratio = na / nu
    if ratio > 100 or (nu >= 40 and ratio < 0.05):  # respuestas absurdamente desproporcionadas
        return None
    if repeated_ngram_ratio(uw) > 0.40 or repeated_ngram_ratio(aw) > 0.40:
        return None
    if len(re.findall(r"https?://", a)) > 8:
        return None
    if len(set(a)) < 8 and len(a) > 30:
        return None
    if "\ufffd" in a or a.count("Ã") > 2:            # texto corrupto / mojibake
        return None
    return u, a

def quality_conv(turns):
    """[u1,b1,u2,b2,...] -> conserva el prefijo de pares que pasan el filtro."""
    turns = [t if isinstance(t, str) else "" for t in turns]
    turns = turns[: len(turns) // 2 * 2]
    out = []
    for j in range(0, len(turns), 2):
        p = quality_pair(turns[j], turns[j + 1])
        if not p:
            break
        out += list(p)
    return out or None

def dedup(convs, seen):
    out = []
    for c in convs:
        k = hash("\x1f".join(c).lower())
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out

def load_local():
    if not os.path.exists(DATOS):
        print(f"No existe {DATOS}")
        return []
    try:
        raw = json.load(open(DATOS, encoding="utf-8"))
    except Exception as e:
        print(f"Error leyendo dataset local: {e}")
        return []
    out = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            turns = list(item)
        elif isinstance(item, dict):
            q = item.get("user") or item.get("instruction") or item.get("prompt") or ""
            a = item.get("assistant") or item.get("output") or item.get("response") or ""
            turns = [q, a]
        else:
            continue
        c = quality_conv(turns)
        if c:
            out.append(c)
    return out

def sinteticos(n, seed=5):
    rng, out = random.Random(seed), []
    for _ in range(n):
        op = rng.choice("+-*")
        if op == "+":
            a, b = rng.randint(0, 99), rng.randint(0, 99); r = a + b
            qs = [f"cuanto es {a} mas {b}", f"¿Cuánto es {a} + {b}?", f"suma {a} y {b}",
                  f"{a} mas {b}", f"cuanto da {a} + {b}"]
            rs = [f"{a} más {b} es {r}.", f"{a} + {b} = {r}", f"El resultado es {r}.",
                  f"La suma de {a} y {b} es {r}."]
        elif op == "-":
            a, b = sorted((rng.randint(0, 99), rng.randint(0, 99)), reverse=True); r = a - b
            qs = [f"cuanto es {a} menos {b}", f"¿Cuánto es {a} - {b}?", f"resta {b} a {a}",
                  f"{a} menos {b}"]
            rs = [f"{a} menos {b} es {r}.", f"{a} - {b} = {r}", f"El resultado es {r}.",
                  f"La resta de {a} menos {b} es {r}."]
        else:
            a, b = rng.randint(0, 12), rng.randint(0, 12); r = a * b
            qs = [f"cuanto es {a} por {b}", f"¿Cuánto es {a} x {b}?", f"multiplica {a} por {b}",
                  f"{a} por {b}"]
            rs = [f"{a} por {b} es {r}.", f"{a} x {b} = {r}", f"El resultado es {r}.",
                  f"La multiplicación de {a} por {b} es {r}."]
        out.append([rng.choice(qs), rng.choice(rs)])
    return out

def extra_chat():
    out = []
    try:
        from datasets import load_dataset
        ds = load_dataset("bertin-project/alpaca-spanish", split="train")
        for ex in ds:
            q = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            a = (ex.get("output") or "").strip()
            if q and a:
                c = quality_conv([q + ("\n" + inp if inp else ""), a])
                if c:
                    out.append(c)
        print(f"alpaca-spanish: {len(out):,} pares")
    except Exception as e:
        print(f"extra_chat omitido: {type(e).__name__}")
    return out

# ---------- ShareGPT real (conversaciones completas, multi-turno) ----------
USER_ROLES = {"human", "user", "prompter"}
BOT_ROLES = {"gpt", "assistant", "bot", "model"}

def parse_sharegpt_item(item):
    conv = item.get("conversations") if isinstance(item, dict) else None
    if not isinstance(conv, list):
        return None
    turns = []
    for t in conv:
        if not isinstance(t, dict):
            continue
        role = str(t.get("from") or t.get("role") or "").lower()
        val = normalize_text(t.get("value") or t.get("content") or "")
        if not val:
            continue
        r = 0 if role in USER_ROLES else 1 if role in BOT_ROLES else None
        if r is None:
            continue
        if r == len(turns) % 2:
            turns.append(val)
        elif turns:                      # mismo rol seguido: se fusiona con el anterior
            turns[-1] += "\n\n" + val
    return quality_conv(turns)

def parse_sharegpt_text(txt):
    objs = []
    try:
        data = json.loads(txt)
        objs = data if isinstance(data, list) else [data]
    except Exception:
        for line in txt.splitlines():
            try:
                objs.append(json.loads(line))
            except Exception:
                pass
    out = []
    for o in objs:
        c = parse_sharegpt_item(o)
        if c:
            out.append(c)
    return out

def load_sharegpt():
    repo, out = "FreedomIntelligence/sharegpt-spanish", []
    try:
        from huggingface_hub import HfApi as _Api, hf_hub_download
        files = [f for f in _Api().list_repo_files(repo, repo_type="dataset")
                 if f.endswith((".json", ".jsonl"))][:3]
        for f in files:
            p = hf_hub_download(repo, f, repo_type="dataset")
            out += parse_sharegpt_text(open(p, encoding="utf-8").read())
    except Exception as e:
        print(f"ShareGPT (hub) fallo: {type(e).__name__}")
    if not out:
        try:
            import requests
            base = f"https://huggingface.co/datasets/{repo}/resolve/main/"
            for name in ("data/train.json", "sharegpt_spanish.json", "data.json"):
                rr = requests.get(base + name, timeout=90)
                if rr.status_code == 200 and rr.content:
                    out += parse_sharegpt_text(rr.text)
                    if out:
                        break
        except Exception as e:
            print(f"ShareGPT (requests) fallo: {type(e).__name__}")
    print(f"ShareGPT: {len(out):,} conversaciones" if out
          else "AVISO: ShareGPT no aporto datos; sigo con el resto")
    return out

# ---------- datos locales + fallbacks ----------
if not os.path.exists(DATOS):
    cand = glob.glob("/kaggle/input/**/dataset_1m.json", recursive=True)
    if cand:
        shutil.copy2(cand[0], DATOS)
    elif api:
        from huggingface_hub import hf_hub_download
        for rid in (REPO_HF, f"{USER}/Aqua-2.0-300M"):     # el viejo tiene tu dataset_1m.json
            try:
                hf_hub_download(rid, "dataset_1m.json", local_dir=SALIDA)
                print(f"dataset_1m.json bajado de {rid}")
                break
            except Exception as e:
                print(f"dataset no esta en {rid}: {type(e).__name__}")

local = load_local()
print(f"Conversaciones locales validas: {len(local):,}")
if EXTRA_CHAT:
    local += extra_chat()
share = load_sharegpt() if SHAREGPT else []

seen = set()
local, share = dedup(local, seen), dedup(share, seen)
rg = random.Random(SEED)
rg.shuffle(local)
rg.shuffle(share)
if not local and not share:
    raise FileNotFoundError(f"Sin datos de chat: revisa {DATOS}")

with open(HF_DATA, "w", encoding="utf-8") as f:
    for c in local + share:
        rec = {"user": c[0], "assistant": c[1]} if len(c) == 2 else {"conversation": c}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

nv_l = max(200, min(len(local) // 50, 2000)) if len(local) > 400 else len(local) // 10
nv_s = min(300, len(share) // 20)
val_p = local[:nv_l] + share[:nv_s]
tr_p = local[nv_l:] + share[nv_s:] * SHAREGPT_REP + sinteticos(SINTETICOS)
random.Random(SEED + 1).shuffle(tr_p)
print(f"Chat: {len(tr_p):,} train | {len(val_p):,} val "
      f"(local {len(local):,} + sharegpt {len(share):,} x{SHAREGPT_REP})")

# ======================= TEXTO WEB (streaming) =======================
FUENTES_OK = []
try:
    from datasets import load_dataset
    for nombre, config, peso in FUENTES:
        try:
            if config:
                ds = load_dataset(nombre, config, split="train", streaming=True)
            else:
                ds = load_dataset(nombre, split="train", streaming=True)
            next(iter(ds))
            FUENTES_OK.append((nombre, ds, peso))
            print(f"PRE OK: {nombre} ({config})")
        except Exception as e:
            print(f"PRE aviso {nombre}: {type(e).__name__}")
except Exception as e:
    print(f"datasets no disponible: {e}")

def extraer(ex):
    for k in ("text", "content", "raw_content", "document"):
        v = ex.get(k)
        if isinstance(v, str) and len(v) >= 200:
            v = v.strip()[:30000]
            m = v[:1500]
            ok = sum(c.isalpha() or c in " \n" for c in m) / max(1, len(m)) > 0.70
            return v if ok else ""
    return ""

def textos_web(seed):
    rng = random.Random(seed)
    st = [(n, ds.shuffle(seed=seed, buffer_size=2000), p) for n, ds, p in FUENTES_OK]
    its = [iter(s[1]) for s in st]
    pesos = [s[2] for s in st]
    fallos = [0] * len(st)
    while any(p > 0 for p in pesos):
        i = rng.choices(range(len(st)), weights=pesos)[0]
        try:
            t = extraer(next(its[i]))
            fallos[i] = 0
        except StopIteration:
            its[i] = iter(st[i][1])
            continue
        except Exception:
            fallos[i] += 1
            time.sleep(1)
            if fallos[i] >= 20:
                pesos[i] = 0
                print(f"[datos] fuente descartada: {st[i][0]}")
            continue
        if t:
            yield t

# ======================= TOKENIZER =======================
def conv_texto(c):
    return "\n".join(f"{ROL_USER if j % 2 == 0 else ROL_BOT}: {t}" for j, t in enumerate(c))

if not os.path.exists(os.path.join(TOK_DIR, "vocab.json")):
    os.makedirs(TOK_DIR, exist_ok=True)
    def textos_tok():
        for c in tr_p[:200000]:          # conversaciones primero: vocabulario util para dialogo
            yield conv_texto(c)
        for i, t in enumerate(textos_web(0)):
            if i >= 80_000:
                break
            yield t[:3000]
    print("Entrenando tokenizer BPE 32k...")
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(textos_tok(), vocab_size=VOCAB, min_frequency=2,
                            special_tokens=["<pad>", "<eos>", "<bos>"])
    bpe.save_model(TOK_DIR)

tok = ByteLevelBPETokenizer(os.path.join(TOK_DIR, "vocab.json"),
                            os.path.join(TOK_DIR, "merges.txt"))
PAD, EOS = tok.token_to_id("<pad>"), tok.token_to_id("<eos>")
BOS = tok.token_to_id("<bos>")
HAY_BOS = BOS is not None
if not HAY_BOS:
    BOS = EOS
print(f"Vocab: {tok.get_vocab_size():,}")

def codificar_convs(convs):
    """'USUARIO: q\\nASISTENTE: a<eos>' (turnos extra: '\\nUSUARIO: q2\\nASISTENTE: a2<eos>').
    Devuelve [(ids, labels)] con labels=-100 en lo que no es respuesta del asistente."""
    segs = []
    for c in convs:
        for j in range(0, len(c) - 1, 2):
            segs.append(("" if j == 0 else "\n") + f"{ROL_USER}: {c[j]}\n{ROL_BOT}:")
            segs.append(" " + c[j + 1])
    encs = tok.encode_batch(segs)
    out, k = [], 0
    for c in convs:
        ids, lab = [], []
        for j in range(0, len(c) - 1, 2):
            p, r = encs[k].ids, encs[k + 1].ids + [EOS]
            k += 2
            ids += p + r
            lab += [-100] * len(p) + r
        out.append((ids, lab))
    return out

def ids_prompt(turnos):
    """turnos = [u1, b1, u2, ...] terminando en turno del usuario."""
    ids = []
    for j, t in enumerate(turnos):
        if j % 2 == 0:
            ids += tok.encode(("" if j == 0 else "\n") + f"{ROL_USER}: {t}\n{ROL_BOT}:").ids
        else:
            ids += tok.encode(" " + t).ids + [EOS]
    return ids

# ======================= MODELO (LLaMA) =======================
cfg = LlamaConfig(
    vocab_size=tok.get_vocab_size(),
    hidden_size=1024, intermediate_size=3072,
    num_hidden_layers=22, num_attention_heads=16, num_key_value_heads=4,
    max_position_embeddings=BLOCK, rms_norm_eps=1e-5, hidden_act="silu",
    tie_word_embeddings=True, attention_dropout=0.0,
    bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD, use_cache=False,
)
try:
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa")
except Exception:
    model = AutoModelForCausalLM.from_config(cfg)

with torch.no_grad():   # init residual escalada (estabiliza redes profundas)
    s_init = 0.02 / math.sqrt(2 * cfg.num_hidden_layers)
    for n_, p_ in model.named_parameters():
        if n_.endswith("o_proj.weight") or n_.endswith("down_proj.weight"):
            torch.nn.init.normal_(p_, mean=0.0, std=s_init)
model = model.to(dev)
model.train()
n_par = sum(p.numel() for p in model.parameters())
print(f"PARAMETROS: {n_par / 1e6:.1f}M")

dec = [p for p in model.parameters() if p.dim() >= 2]
nod = [p for p in model.parameters() if p.dim() < 2]
grupos = [{"params": dec, "weight_decay": 0.1}, {"params": nod, "weight_decay": 0.0}]
try:
    opt = torch.optim.AdamW(grupos, lr=LR_PRE_MAX, betas=(0.9, 0.95), eps=1e-8, fused=usa_amp)
except Exception:
    opt = torch.optim.AdamW(grupos, lr=LR_PRE_MAX, betas=(0.9, 0.95), eps=1e-8)
scaler = torch.amp.GradScaler("cuda", enabled=usa_amp and AMP_DTYPE == torch.float16)
estado = {"paso": 0, "fase": 1, "epoca_sft": 0, "mejor": float("inf"), "lr": LR_PRE_MAX}

# --- autoajuste de micro-batch (evita OOM) ---
MB = 1
if usa_amp:
    reserva = torch.empty(n_par * 8, dtype=torch.uint8, device=dev)  # simula estados de Adam
    for mb in (4, 2, 1):
        x = None
        try:
            x = torch.randint(0, tok.get_vocab_size(), (mb, BLOCK), device=dev)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                model(input_ids=x, labels=x).loss.backward()
            MB = mb
            break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"OOM con micro-batch {mb}, bajando...")
        finally:
            model.zero_grad(set_to_none=True)
            x = None
            gc.collect()
            torch.cuda.empty_cache()
    else:
        raise RuntimeError("OOM incluso con micro-batch 1")
    del reserva
    torch.cuda.empty_cache()
ACUM = max(1, TOK_PASO // (MB * BLOCK))
SFT_TOK, SFT_ACUM = MB * BLOCK, ACUM      # mismo perfil de memoria que ya se probo arriba
print(f"micro-batch {MB} x acumulacion {ACUM} = {MB * ACUM * BLOCK:,} tokens/paso")

# --- torch.compile (con fallback) ---
fwd = model
if USAR_COMPILE and usa_amp and hasattr(torch, "compile"):
    try:
        import torch._dynamo as _dyn
        _dyn.config.suppress_errors = True
        t_c = time.time()
        cm = torch.compile(model)
        x = torch.randint(0, tok.get_vocab_size(), (MB, BLOCK), device=dev)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            cm(input_ids=x, labels=x).loss.backward()
        model.zero_grad(set_to_none=True)
        fwd = cm
        print(f"torch.compile=ON ({time.time() - t_c:.0f}s)")
    except Exception as e:
        print(f"torch.compile=OFF: {type(e).__name__}")
        model.zero_grad(set_to_none=True)
        fwd = model

# --- cargar checkpoint si es compatible ---
if os.path.exists(CKPT):
    c = torch.load(CKPT, map_location="cpu", weights_only=False)
    if c.get("arch") != ARCH:
        print("Checkpoint de otra arquitectura (ej. GPT-2 viejo): lo ignoro y arranco limpio.")
    else:
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        scaler.load_state_dict(c["scaler"])
        estado.update(c["estado"])
        print(f"Checkpoint: paso={estado['paso']} fase={estado['fase']}")
    del c
    gc.collect()

def guardar_ckpt(bloquear=False):
    if not lock_ckpt.acquire(blocking=bloquear):
        print("[ckpt] subida en curso, salto este guardado")
        return
    try:
        tmp = CKPT + ".tmp"
        torch.save({"arch": ARCH, "model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "estado": estado}, tmp)
        os.replace(tmp, CKPT)
    finally:
        lock_ckpt.release()

# ======================= config.py (se sube al repo) =======================
import pprint

CONFIG_TXT = '''# Aqua - configuracion (autogenerada por el script de entrenamiento)
# Uso: from config import CONFIG, construir_ids, responder

CONFIG = __CONFIG__

ROL_USER = CONFIG["prompt"]["rol_usuario"]
ROL_BOT = CONFIG["prompt"]["rol_bot"]


def construir_ids(tok, turnos):
    """turnos = [u1, b1, u2, ...] terminando en turno del usuario -> ids EXACTOS del entrenamiento."""
    ids = []
    for j, t in enumerate(turnos):
        if j % 2 == 0:
            txt = ("" if j == 0 else "\\n") + f"{ROL_USER}: {t}\\n{ROL_BOT}:"
            ids += tok(txt, add_special_tokens=False)["input_ids"]
        else:
            ids += tok(" " + t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return ids


def responder(turnos, repo=CONFIG["repo"], max_new_tokens=200):
    """Carga el modelo desde HF y responde. turnos puede ser un str o [u1, b1, u2, ...]."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo).eval()
    if isinstance(turnos, str):
        turnos = [turnos]
    x = torch.tensor([construir_ids(tok, turnos)])
    gen = {k: v for k, v in CONFIG["generacion"].items() if k != "max_new_tokens"}
    with torch.no_grad():
        y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=True,
                           max_new_tokens=max_new_tokens, eos_token_id=tok.eos_token_id,
                           pad_token_id=tok.pad_token_id, **gen)
    txt = tok.decode(y[0][x.shape[1]:], skip_special_tokens=True).strip()
    return txt.split("\\n" + ROL_USER + ":")[0].strip()


if __name__ == "__main__":
    print(responder("hola"))
'''

def _limpio(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _limpio(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_limpio(v) for v in x]
    return x

def escribir_config(path):
    d = {
        "nombre": REPO_ID.split("/")[-1],
        "repo": REPO_ID,
        "arquitectura": {
            "tipo": "LlamaForCausalLM", "parametros": n_par, "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size, "intermediate_size": cfg.intermediate_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads, "contexto": BLOCK,
            "tie_word_embeddings": True, "rms_norm_eps": cfg.rms_norm_eps,
        },
        "tokens": {"pad": PAD, "eos": EOS, "bos": BOS if HAY_BOS else None},
        "prompt": {"rol_usuario": ROL_USER, "rol_bot": ROL_BOT},
        "generacion": dict(max_new_tokens=200, **GEN),
        "entrenamiento": {
            "amp": str(AMP_DTYPE).split(".")[-1], "micro_batch": MB, "acumulacion": ACUM,
            "tokens_por_paso": MB * ACUM * BLOCK, "lr_pre_max": LR_PRE_MAX, "lr_pre_min": LR_PRE_MIN,
            "warmup": WARMUP, "lr_sft_max": LR_SFT_MAX, "lr_sft_min": LR_SFT_MIN,
            "chat_mix": [CHAT_MIX0, CHAT_MIX1], "sft_epocas_max": SFT_EPOCAS_MAX,
            "limite_horas": LIMITE_H, "horas_sft": HORAS_SFT, "seed": SEED,
        },
        "datos": {
            "train": len(tr_p), "val": len(val_p), "locales": len(local), "sharegpt": len(share),
            "sharegpt_rep": SHAREGPT_REP, "sinteticos": SINTETICOS,
            "fuentes_web": [n for n, _, _ in FUENTES_OK],
        },
        "resultado": {
            "pasos": estado["paso"], "fase": estado["fase"], "epocas_sft": estado["epoca_sft"],
            "val_loss_mejor": estado["mejor"], "horas": round((time.time() - T0) / 3600, 2),
        },
    }
    texto = CONFIG_TXT.replace("__CONFIG__", pprint.pformat(_limpio(d), sort_dicts=False, width=100))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(texto)

escribir_config(CONFIG_PY)
if REPO_HF and not subir_pequeno(CONFIG_PY, "config.py"):
    print(f"OJO: no pude escribir en {REPO_HF}. Revisa que tu HF_TOKEN tenga permiso de "
          f"escritura en la organizacion (el entrenamiento sigue igual).")

# ======================= UTILIDADES =======================
def micro(ids, lab, div):
    ids = ids.to(dev, non_blocking=True)
    lab = ids if lab is None else lab.to(dev, non_blocking=True)
    with torch.autocast(device_type=dev, dtype=AMP_DTYPE, enabled=usa_amp):
        loss = fwd(input_ids=ids, labels=lab).loss
    scaler.scale(loss / div).backward()
    return loss.detach()

def aplicar(lr):
    for g in opt.param_groups:
        g["lr"] = lr
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    estado["paso"] += 1
    estado["lr"] = lr

@torch.no_grad()
def generar(turnos, max_new=120):
    if isinstance(turnos, str):
        turnos = [turnos]
    ids = ids_prompt(turnos)[-(BLOCK - max_new):]
    model.eval()
    x = torch.tensor([ids], device=dev)
    with torch.autocast(device_type=dev, dtype=AMP_DTYPE, enabled=usa_amp):
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                             max_new_tokens=max_new, do_sample=True, **GEN,
                             eos_token_id=EOS, pad_token_id=PAD, use_cache=True)
    model.train()
    txt = tok.decode(out[0][x.shape[1]:].tolist()).strip()
    return txt.split(f"\n{ROL_USER}:")[0].strip()

def prep(convs):
    """Codifica, trunca a SFT_LEN y descarta lo que no deja ninguna respuesta que aprender."""
    items = [(i[:SFT_LEN], l[:SFT_LEN]) for i, l in codificar_convs(convs)]
    return [(i, l) for i, l in items if any(v != -100 for v in l)]

def collate(items):
    n = (max(len(i) for i, _ in items) + 7) // 8 * 8
    ids = torch.full((len(items), n), PAD, dtype=torch.long)
    lab = torch.full((len(items), n), -100, dtype=torch.long)
    for k, (i, l) in enumerate(items):
        ids[k, :len(i)] = torch.tensor(i)
        lab[k, :len(l)] = torch.tensor(l)
    return ids, lab   # padding a la derecha + causal: no hace falta attention_mask

@torch.no_grad()
def evaluar(n=400):
    model.eval()
    tot, cnt, sub = 0.0, 0, val_p[:n]
    for s in range(0, len(sub), 8):
        items = prep(sub[s:s + 8])
        if not items:
            continue
        ids, lab = [t.to(dev) for t in collate(items)]
        with torch.autocast(device_type=dev, dtype=AMP_DTYPE, enabled=usa_amp):
            logits = model(input_ids=ids).logits
        tot += F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                               lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum").item()
        cnt += (lab[:, 1:] != -100).sum().item()
    model.train()
    return tot / max(cnt, 1)

def snapshot():
    return {k: v.detach().to("cpu", dtype=torch.float16, copy=True)
            for k, v in model.state_dict().items()}

class Prefetch:
    """Produce lotes en un hilo aparte para que la GPU nunca espere a la red/tokenizador."""
    def __init__(self, fn, n=32):
        self.q, self.stop, self.err = queue.Queue(maxsize=n), False, None
        threading.Thread(target=self._run, args=(fn,), daemon=True).start()
    def _put(self, x):
        while not self.stop:
            try:
                self.q.put(x, timeout=1)
                return True
            except queue.Full:
                pass
        return False
    def _run(self, fn):
        try:
            for x in fn():
                if not self._put(x):
                    return
        except Exception as e:
            self.err = e
        self._put(None)
    def __iter__(self):
        while True:
            x = self.q.get(timeout=1800)
            if x is None:
                if self.err:
                    raise self.err
                return
            yield x
    def close(self):
        self.stop = True

parar = False
def _s(*_):
    global parar
    parar = True
for sg in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(sg, _s)
    except Exception:
        pass

# ======================= FASE 1: PRE-ENTRENAMIENTO (web + chat) =======================
ctrl = {"frac": 0.0}

def p_chat():
    f = ctrl["frac"]
    if f < 0.7:
        return CHAT_MIX0
    return CHAT_MIX0 + (CHAT_MIX1 - CHAT_MIX0) * min(1.0, (f - 0.7) / 0.3)

def ids_web(seed):
    lote = []
    for t in textos_web(seed):
        lote.append(t)
        if len(lote) == 256:
            for e in tok.encode_batch(lote):
                yield e.ids + [EOS]
            lote = []

def ids_chat(seed):
    rng = random.Random(seed)
    idx = list(range(len(tr_p)))
    while True:
        rng.shuffle(idx)
        for s in range(0, len(idx), 256):
            for ids, _ in codificar_convs([tr_p[i] for i in idx[s:s + 256]]):
                yield ids

def bloques(gen):
    buf = []
    for ids in gen:
        buf.extend(ids)
        i = 0
        while len(buf) - i >= BLOCK:
            yield buf[i:i + BLOCK]
            i += BLOCK
        buf = buf[i:]

def mezcla(seed):
    rng = random.Random(seed)
    web, chat = bloques(ids_web(seed)), bloques(ids_chat(seed + 1))
    b = []
    while True:
        usar_chat = web is None or rng.random() < p_chat()
        g = chat if usar_chat else web
        try:
            b.append(next(g))
        except StopIteration:
            if usar_chat:
                return
            web = None
            continue
        if len(b) == MB:
            yield torch.tensor(b, dtype=torch.long)
            b = []

def lr_pre(paso, frac, lr_ini):
    if paso < WARMUP:
        return LR_PRE_MAX * max(0.05, (paso + 1) / WARMUP)
    f = min(max(frac, 0.0), 1.0)
    return LR_PRE_MIN + 0.5 * (lr_ini - LR_PRE_MIN) * (1 + math.cos(math.pi * f))

t_sft = T0 + LIMITE - HORAS_SFT * 3600
if estado["fase"] == 1 and time.time() < t_sft:
    print(f"\n=== FASE 1: PRETRAINING ({(t_sft - time.time()) / 3600:.1f}h disponibles) ===")
    t_ini = time.time()
    lr_ini = LR_PRE_MAX if estado["paso"] < WARMUP else min(LR_PRE_MAX, max(LR_PRE_MIN, estado["lr"]))
    seed = SEED + estado["paso"]
    pre = Prefetch(lambda: mezcla(seed))
    acum, n, k = torch.zeros((), device=dev), 0, 0
    t_log = t_ck = t_up = time.time()
    try:
        for ids in pre:
            now = time.time()
            frac = (now - t_ini) / max(1.0, t_sft - t_ini)
            ctrl["frac"] = frac
            acum += micro(ids, None, ACUM)
            n += 1
            k += 1
            if k % ACUM == 0:
                aplicar(lr_pre(estado["paso"], frac, lr_ini))
            if now - t_log > 180:
                tps = n * MB * BLOCK / (now - t_log)
                toks = estado["paso"] * MB * ACUM * BLOCK / 1e6
                print(f"[{(now - T0) / 3600:.2f}h] paso={estado['paso']} | {toks:.0f}M tok | "
                      f"loss={acum.item() / max(n, 1):.3f} | lr={estado['lr']:.1e} | "
                      f"{tps / 1e3:.1f}K tok/s | chat {p_chat():.0%}")
                acum.zero_()
                n, t_log = 0, now
            if now - t_ck > CKPT_MIN * 60:
                guardar_ckpt()
                t_ck = time.time()
                if time.time() - t_up > SUBIR_MIN * 60:
                    subir_ckpt_async()
                    t_up = time.time()
            if parar or now >= t_sft:
                break
    except Exception:
        traceback.print_exc()
        parar = True
    finally:
        pre.close()
    opt.zero_grad(set_to_none=True)
    if not parar:
        print(f"val (chat) tras pretraining: {evaluar():.3f}")
        print("PREVIEW:", generar("Explícame qué es la inteligencia artificial.", 80))
        estado["fase"] = 2
    guardar_ckpt(True)
elif estado["fase"] == 1:
    print("Sin tiempo para pre-entrenar: paso directo a SFT")
    estado["fase"] = 2

# ======================= FASE 2: SFT (conversacion) =======================
def gen_sft(epoca):
    """Lotes por presupuesto de tokens y agrupados por largo: casi sin padding."""
    rng = random.Random(SEED + 7 + epoca)
    idx = list(range(len(tr_p)))
    rng.shuffle(idx)
    for s in range(0, len(idx), 4096):
        enc = prep([tr_p[j] for j in idx[s:s + 4096]])
        enc.sort(key=lambda e: len(e[0]))
        lotes, cur = [], []
        for e in enc:
            if cur and ((len(cur) + 1) * len(e[0]) > SFT_TOK or len(cur) >= SFT_MAXB):
                lotes.append(cur)
                cur = []
            cur.append(e)
        if cur:
            lotes.append(cur)
        rng.shuffle(lotes)
        for b in lotes:
            yield collate(b)

def fase_sft():
    global parar
    fin_t = T0 + LIMITE
    t_s0 = time.time()
    muestra = random.Random(1).sample(tr_p, min(2000, len(tr_p)))
    lens = [len(i) for i, _ in prep(muestra)] or [SFT_LEN]
    pasos_ep = max(1, int(sum(lens) / len(lens) * len(tr_p) / (0.9 * SFT_TOK * SFT_ACUM)))
    total = pasos_ep * SFT_EPOCAS_MAX
    print(f"SFT: ~{pasos_ep:,} pasos/epoca | lotes de ~{SFT_TOK:,} tokens x {SFT_ACUM}")
    pasos, mejor, mejor_sd = 0, estado.get("mejor", float("inf")), None
    t_ck = time.time()
    print(f"val antes de SFT: {evaluar():.3f}")
    model.train()
    for epoca in range(estado["epoca_sft"], SFT_EPOCAS_MAX):
        pf = Prefetch(lambda e=epoca: gen_sft(e))
        acum, n, k, completo = torch.zeros((), device=dev), 0, 0, True
        try:
            for ids, lab in pf:
                acum += micro(ids, lab, SFT_ACUM)
                n += 1
                k += 1
                if k % SFT_ACUM:
                    continue
                now = time.time()
                frac = min(1.0, max((now - t_s0) / max(1.0, fin_t - t_s0), pasos / total))
                lr = LR_SFT_MIN + 0.5 * (LR_SFT_MAX - LR_SFT_MIN) * (1 + math.cos(math.pi * frac))
                if pasos < SFT_WARM:
                    lr *= (pasos + 1) / SFT_WARM
                aplicar(lr)
                pasos += 1
                if pasos % 100 == 0:
                    print(f"SFT paso {pasos} | ep {epoca + 1} | loss {acum.item() / max(n, 1):.4f} | lr {lr:.1e}")
                    acum.zero_()
                    n = 0
                if pasos % EVAL_CADA == 0:
                    v = evaluar()
                    print(f"  val {v:.4f} (ppl {math.exp(min(v, 20)):.1f})")
                    if v < mejor:
                        mejor, mejor_sd = v, snapshot()
                if now - t_ck > CKPT_MIN * 60:
                    guardar_ckpt()
                    t_ck = time.time()
                if parar or now >= fin_t:
                    completo = False
                    break
        except Exception:
            traceback.print_exc()
            parar, completo = True, False
        finally:
            pf.close()
        if completo:
            estado["epoca_sft"] = epoca + 1
            v = evaluar()
            print(f"SFT epoch={epoca + 1} completa | val {v:.4f}")
            if v < mejor:
                mejor, mejor_sd = v, snapshot()
            guardar_ckpt()
        else:
            break
    v = evaluar()
    if mejor_sd is not None and mejor < v:
        print(f"Restaurando mejor checkpoint de SFT (val {mejor:.4f} < final {v:.4f})")
        model.load_state_dict(mejor_sd)
    else:
        mejor = min(mejor, v)
    estado["mejor"] = mejor

if not parar and estado["fase"] == 2 and time.time() < T0 + LIMITE:
    print("\n=== FASE 2: SFT CONVERSACIONAL ===")
    fwd = model   # longitudes variables: sin compile
    fase_sft()

guardar_ckpt(True)

# ======================= EXPORTAR Y SUBIR =======================
FINAL = os.path.join(SALIDA, "modelo_final")
if os.path.exists(FINAL):
    shutil.rmtree(FINAL)
os.makedirs(FINAL)
if usa_amp:
    model.to(AMP_DTYPE)
model.config.use_cache = True
for k_, v_ in dict(eos_token_id=EOS, pad_token_id=PAD, bos_token_id=BOS, do_sample=True,
                   max_new_tokens=200, **GEN).items():
    setattr(model.generation_config, k_, v_)
model.save_pretrained(FINAL, safe_serialization=True)
try:
    from transformers import PreTrainedTokenizerFast
    kw = dict(eos_token="<eos>", pad_token="<pad>", model_input_names=["input_ids", "attention_mask"])
    if HAY_BOS:
        kw["bos_token"] = "<bos>"
    PreTrainedTokenizerFast(tokenizer_object=tok._tokenizer, **kw).save_pretrained(FINAL)
except Exception as e:
    print(f"tokenizer HF aviso: {e}")
for f in ("vocab.json", "merges.txt"):
    shutil.copy2(os.path.join(TOK_DIR, f), os.path.join(FINAL, f))
escribir_config(CONFIG_PY)                       # ahora con los resultados finales
shutil.copy2(CONFIG_PY, os.path.join(FINAL, "config.py"))

README = """---
language: es
license: apache-2.0
---
# Aqua 3.0.0 (~300M)
Modelo conversacional en espanol (arquitectura LLaMA: RoPE + SwiGLU + RMSNorm + GQA).
Pretraining mixto (web + chat) y SFT conversacional con dialogos multi-turno.

Formato de prompt:

    __U__: <pregunta>
    __B__:

Multi-turno y parametros de generacion: ver `config.py` (`responder([...])` arma los ids exactos del
entrenamiento; cada respuesta del asistente termina en el token <eos>).

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("__REPO__")
m = AutoModelForCausalLM.from_pretrained("__REPO__")
x = tok("__U__: hola\\n__B__:", return_tensors="pt")
y = m.generate(**x, max_new_tokens=100)
print(tok.decode(y[0][x["input_ids"].shape[1]:], skip_special_tokens=True))
```
"""
with open(os.path.join(FINAL, "README.md"), "w", encoding="utf-8") as fh:
    fh.write(README.replace("__REPO__", REPO_HF or REPO_ID)
                   .replace("__U__", ROL_USER).replace("__B__", ROL_BOT))

print(f"\nTERMINADO: {(time.time() - T0) / 3600:.2f}h | pasos={estado['paso']}")

if REPO_HF:
    try:
        print("Subiendo Aqua a Hugging Face...")
        api.upload_folder(folder_path=FINAL, repo_id=REPO_HF, repo_type="model")
        print(f"LISTO: https://huggingface.co/{REPO_HF}")
        with lock_ckpt:   # espera a que termine cualquier subida en curso
            api.upload_file(path_or_fileobj=CKPT, path_in_repo="ckpt.pt",
                            repo_id=REPO_HF, repo_type="model")
        api.upload_file(path_or_fileobj=HF_DATA, path_in_repo=os.path.basename(HF_DATA),
                        repo_id=REPO_HF, repo_type="model")
    except Exception as e:
        print(f"Error final HF: {e}")

# ======================= PRUEBAS =======================
print("\n=== PRUEBAS ===")
for q in ["hola", "cuéntame un chiste original", "¿cuánto es 5 + 3?", "¿qué es Python?",
          "¿cuál es la capital de Francia?", "explícame por qué el cielo es azul",
          "escribe una historia corta sobre un robot"]:
    print(f"\n{ROL_USER}: {q}")
    print(f"{ROL_BOT}: {generar(q, 100)}")
r1 = generar("hola", 40)
r2 = generar(["hola", r1, "¿cuánto es 7 por 6?"], 40)
print(f"\nMULTI-TURNO -> {r1} | {r2}")
