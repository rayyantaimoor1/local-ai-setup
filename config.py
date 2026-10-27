"""Central configuration for the local multi-agent AI system.

All paths are derived from this file's location, so the project folder can live
anywhere (and survive a Windows username change).
"""
import os
from pathlib import Path

import psutil

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH = Path(os.environ.get("LOCAL_AI_DB", BASE_DIR / "memory.db"))
OUTPUTS_DIR = Path(os.environ.get("LOCAL_AI_OUTPUTS", BASE_DIR / "outputs"))  # documents the chat creates
CONSTITUTION_TEXT = BASE_DIR / "constitution.md"  # read-only for the app
CONSTITUTION_SETTINGS = Path(os.environ.get("LOCAL_AI_CONSTITUTION", BASE_DIR / "constitution.json"))
AUDIT_LOG = Path(os.environ.get("LOCAL_AI_AUDIT", BASE_DIR / "audit.jsonl"))
VECTOR_DB_PATH = Path(os.environ.get("LOCAL_AI_VECTORS", BASE_DIR / "vectors.db"))  # semantic.py

# --- Server ----------------------------------------------------------------
HOST = "127.0.0.1"  # never bind anything else: no auth by design
PORT = 8420

# --- Ollama ----------------------------------------------------------------
OLLAMA_URL = os.environ.get("LOCAL_AI_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_URL}/api/tags"

# Ollama reads this ONLY at server startup; set it as a user environment
# variable and restart Ollama. Kept here as documentation / for setup_check.
OLLAMA_NUM_PARALLEL = 1
OLLAMA_ENV = {"OLLAMA_NUM_PARALLEL": "1", "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}

# One thread per physical core. Measured on this i7-1355U (2 P + 8 E cores, 12 threads), qwen2.5-coder:7b
# reading a 1,200-token prompt / writing: 4 -> 20.9 / 5.1, 6 -> 23.6 / 5.4, 8 -> 24.8 / 5.4,
# 10 -> 28.0 / 5.6, 12 -> 26.7 / 5.0 tokens per second. Hyper-threads only add contention.
NUM_THREAD = int(os.environ.get("LOCAL_AI_THREADS") or 0) or max(1, psutil.cpu_count(logical=False) or (os.cpu_count() or 2) // 2)
NUM_GPU = 0  # CPU-only machine
DEFAULT_NUM_CTX = 4096
ROUTER_NUM_CTX = 2048
KEEP_ALIVE = "0"  # unload a model right after each response (16 GB RAM)

ROUTER_TIMEOUT = 120  # seconds
SPECIALIST_TIMEOUT = 420  # seconds (per read gap while streaming)

# --- Models (all 4-bit via Ollama default tags) ----------------------------
ROUTER_MODEL = "qwen2.5:3b"
SPECIALISTS = {
    "CODE_TASK": "qwen2.5-coder:7b",
    # Benchmarked on this laptop (3 math problems with known answers): qwen3:8b with its thinking switched
    # off got 2/3 right in 1-3 min each; llama3.1:8b 1/3; deepseek-r1:8b 1/3, the other two still
    # reasoning after 11.5 min (3,000 tokens) with no answer; qwen3:8b WITH thinking: no answer in 12 min.
    # So math uses qwen3:8b without its reasoning phase, and deep thinking runs the app's own refine loop.
    "MATH_TASK": "qwen3:8b",
    "SECURITY_TASK": "llama3.1:8b",
    "WRITING_TASK": "gemma2:9b",
    "GENERAL_TASK": "llama3.2:3b",
}
CATEGORIES = tuple(SPECIALISTS)
# Models whose own reasoning phase is used when deep thinking is on (Ollama `think`); none by default
# since the benchmark above. Add "deepseek-r1:8b" here (and as MATH_TASK) to bring it back.
THINKING_MODELS = ()
# Models that accept Ollama's `think` field but always get think=false: their reasoning phase is far
# too slow on this CPU, and the refine loop gives better answers in less time.
THINK_OFF_MODELS = ("qwen3:8b",)
# With deep thinking OFF these categories use a different, faster model (none needed any more).
FAST_MODELS = {}
FALLBACK_CATEGORY = "GENERAL_TASK"
# Used when a model is not downloaded or does not fit in the free RAM: first pulled one that fits wins.
FALLBACK_MODELS = ("llama3.1:8b", "qwen2.5-coder:7b", "llama3.2:3b", "qwen2.5:3b")
FIT_RAM_FACTOR = 1.05  # a model "fits" when free RAM >= its size x this
SUMMARIZER_MODEL = "llama3.2:3b"
# Images: the vision model reads them first (text, code, errors, what is shown), then the normal pipeline
# answers with that description, so a screenshot of a Python error still reaches the coding model.
VISION_MODEL = "qwen2.5vl:3b"
VISION_MAX_SIDE = 1024  # pixels; work per image grows with its size on a CPU
VISION_MAX_IMAGES = 3
# ~9 GB at 4-bit: only for hard problems. Expect heavy trimming/paging on 16 GB.
ESCALATION_MODEL = "qwen2.5-coder:14b"

# --- Embeddings, vector database, semantic routing (semantic.py) ------------
# 0.27 GB; kept loaded (cheap) so routing and search take milliseconds
EMBED_MODEL = os.environ.get("LOCAL_AI_EMBED_MODEL", "nomic-embed-text")
EMBED_KEEP_ALIVE = "30m"
EMBED_BATCH = 16
EMBED_MAX_CHARS = 2000
INDEX_MAX_FILES = 3000
INDEX_MAX_DOC_BYTES = 25 * 1024 * 1024  # PDF / Word
INDEX_MAX_TEXT_BYTES = 2 * 1024 * 1024
RECALL_MIN_SCORE = 0.62  # how close an earlier chat must be to be recalled
ROUTER_EMBED_MARGIN = 0.05  # similarity gap between the best and second label that counts as certain
ROUTER_EMBED_MIN_CONFIDENCE = 0.6  # below this the router model decides, as before
ATTACHMENT_BUDGET_CHARS = 9000  # a bigger attached document is cut down to its most relevant passages

# --- RAM trimming ----------------------------------------------------------
MIN_FREE_RAM_MB_BEFORE_LOAD = 5000
MEMORY_TRIM_MIN_PROCESS_MB = 20
MEMORY_TRIM_EXCLUDE = (
    # "llama" covers llama-server / ollama_llama_server, where the model weights live: trimming it would
    # page out a model that is kept loaded (keep-warm, prewarm) right before it is used
    "system", "ollama", "llama", "python", "conda", "csrss",
    "wininit", "services", "lsass", "winlogon", "dwm", "audiodg",
)

# --- Tools -----------------------------------------------------------------
TOOL_RESULT_MAX_CHARS = 2000
# DEEP is excluded: in a real test the 14B called a PDF tool for a plain coding question.
TOOL_CATEGORIES = ("CODE_TASK", "SECURITY_TASK")
CHAT_MAX_TOOL_ROUNDS = 4  # e.g. read a PDF -> create the report; the last round must answer
TOOL_NUM_CTX = 8192  # tool chats carry results and document text (~+0.2 GB for the 7B)
TOOL_KEEP_ALIVE = "2m"  # keep the model loaded between the rounds of ONE answer
# After an answer the specialist stays loaded this long, so a follow-up skips loading (~10 s) and
# re-reading the instructions it already read (Ollama reuses the cached prompt: 45-55 s for the coder
# with tools). Safe on 16 GB because any other model is unloaded before a different one loads.
IDLE_KEEP_ALIVE = "3m"
WARM_KEEP_ALIVE = "15m"  # "keep model loaded" switch: the same, for longer
TOOL_RESULT_FULL_ROUNDS = 1  # only the newest round's tool results stay in full; older ones are shortened
TOOL_RESULT_OLD_CHARS = 400
ANSWER_RESERVE_TOKENS = 1500  # room kept for the answer; a longer prompt gets a bigger context window
CONTINUE_MAX_CHARS = 14000  # a conversation continued from the model's cache is capped at this size

# Deep thinking = a refine loop: draft -> automatic checks -> critique -> revise, at most this many times,
# stopping as soon as the checks pass and the critique finds nothing. Quality over speed.
REFINE_MAX_ROUNDS = 3
DOC_REVIEW_MIN_CHARS = 200  # in deep thinking, document content this long is reviewed once before saving

# Prewarm: load the model a message will need while you are still typing, only with this much RAM free.
PREWARM_ROUTER_FREE_GB = 3.0
PREWARM_SPECIALIST_FREE_GB = 7.0
PREWARM_KEEP_ALIVE = "2m"

# --- Conversation context ----------------------------------------------------
HISTORY_MESSAGES = 6  # last N real messages of the chat are re-injected into the prompt
HISTORY_MESSAGE_CHARS = 600
HISTORY_TOTAL_CHARS = 2500
FOLLOW_UP_MAX_CHARS = 120  # short follow-ups reuse the previous answer's route (no router call)

# --- Memory ----------------------------------------------------------------
MEMORY_CAP_CHARS = 800  # per tier
SESSION_HISTORY_TURNS = 8
PROJECT_TOPICS_KEPT = 6
LONG_MEMORIES_IN_CONTEXT = 3
DEFAULT_CHAT_TITLE = "New chat"
