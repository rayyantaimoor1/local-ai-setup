"""Central configuration for the local multi-agent AI system.

All paths are derived from this file's location, so the project folder can live
anywhere (and survive a Windows username change).
"""
import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH = Path(os.environ.get("LOCAL_AI_DB", BASE_DIR / "memory.db"))
DEFAULT_WORKSPACE = BASE_DIR / "agent-workspace"
CONSTITUTION_TEXT = BASE_DIR / "constitution.md"  # read-only for the app
CONSTITUTION_SETTINGS = Path(os.environ.get("LOCAL_AI_CONSTITUTION", BASE_DIR / "constitution.json"))
AUDIT_LOG = Path(os.environ.get("LOCAL_AI_AUDIT", BASE_DIR / "audit.jsonl"))

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

NUM_THREAD = max(1, (os.cpu_count() or 2) // 2)  # 12 logical cores -> 6
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
    "MATH_TASK": "deepseek-r1:8b",
    "SECURITY_TASK": "llama3.1:8b",
    "WRITING_TASK": "gemma2:9b",
    "GENERAL_TASK": "llama3.2:3b",
}
CATEGORIES = tuple(SPECIALISTS)
FALLBACK_CATEGORY = "GENERAL_TASK"
SUMMARIZER_MODEL = "llama3.2:3b"
# ~9 GB at 4-bit: only for hard problems. Expect heavy trimming/paging on 16 GB.
ESCALATION_MODEL = "qwen2.5-coder:14b"

# --- RAM trimming ----------------------------------------------------------
MIN_FREE_RAM_MB_BEFORE_LOAD = 5000
MEMORY_TRIM_MIN_PROCESS_MB = 20
MEMORY_TRIM_EXCLUDE = (
    "system", "ollama", "python", "conda", "csrss",
    "wininit", "services", "lsass", "winlogon",
)

# --- Tools -----------------------------------------------------------------
TOOL_RESULT_MAX_CHARS = 2000
# DEEP is excluded: in a real test the 14B called a PDF tool for a plain coding question.
TOOL_CATEGORIES = ("CODE_TASK", "SECURITY_TASK")

# --- Memory ----------------------------------------------------------------
MEMORY_CAP_CHARS = 800  # per tier
SESSION_HISTORY_TURNS = 8
PROJECT_TOPICS_KEPT = 6
LONG_MEMORIES_IN_CONTEXT = 3
DEFAULT_CHAT_TITLE = "New chat"

# --- Agent -----------------------------------------------------------------
ESCALATE_AFTER_FAILURES = 3  # consecutive failures on the primary coder
ESCALATION_MAX_ITERATIONS = 2  # attempts the 14B model gets
AGENT_NUM_CTX = 8192  # agents carry a step log, so they get twice the chat window (~+0.4 GB for the 7B)
AGENT_MAX_STEPS = 14  # hard stop for tool/write/run steps in one run
AGENT_KEEP_ALIVE = "10m"  # keep the model loaded between steps (faster); unloaded explicitly when the run ends
AGENT_RUN_TIMEOUT = 120  # seconds per script run (the first matplotlib/office import is slow)
AGENT_ERROR_CHARS = 1500
AGENT_STEP_LOG_CHARS = 900  # each earlier step is shown to the model at most this long
AGENT_APPROVAL_TIMEOUT = 900  # seconds to wait for the user to approve risky code
# The only packages the agent may pip-install (import name differs for some: python-docx -> docx).
AGENT_PIP_ALLOWLIST = (
    "numpy", "pandas", "matplotlib", "seaborn", "scipy", "sympy", "openpyxl", "xlsxwriter",
    "python-pptx", "python-docx", "reportlab", "fpdf2", "pillow", "requests", "beautifulsoup4",
    "lxml", "pyyaml", "tabulate", "markdown", "jinja2", "qrcode", "pypdf", "tqdm",
)
