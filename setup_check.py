"""Verify the machine is ready: Python packages, Git, Ollama, models, RAM.

Usage:  python setup_check.py
Exit code 0 when everything required is in place.
"""
import importlib
import os
import shutil
import sys

import psutil

import config

OK, BAD, WARN = "[ ok ]", "[FAIL]", "[warn]"
problems = 0


def report(flag: str, text: str) -> None:
    global problems
    if flag == BAD:
        problems += 1
    print(f"{flag} {text}")


def main() -> int:
    report(OK if sys.version_info >= (3, 10) else BAD, f"Python {sys.version.split()[0]}")

    for module in ("fastapi", "uvicorn", "httpx", "psutil", "pymupdf", "docx", "ruff", "pytest"):
        try:
            importlib.import_module(module)
            report(OK, f"package: {module}")
        except ImportError:
            report(BAD, f"package missing: {module}  (pip install -r requirements.txt)")
    try:
        importlib.import_module("ddgs")
        report(OK, "package: ddgs (web search)")
    except ImportError:
        try:
            importlib.import_module("duckduckgo_search")
            report(OK, "package: duckduckgo_search (web search)")
        except ImportError:
            report(WARN, "no web-search package (pip install duckduckgo-search)")

    report(OK if shutil.which("git") else WARN, "git " + ("found" if shutil.which("git") else "not on PATH"))

    vm = psutil.virtual_memory()
    report(OK, f"RAM: {vm.total / 2**30:.1f} GB total, {vm.available / 2**20:.0f} MB free")
    report(OK, f"threads for inference: {config.NUM_THREAD} (of {os.cpu_count()} logical cores)")
    if os.environ.get("OLLAMA_NUM_PARALLEL") != "1":
        report(WARN, "OLLAMA_NUM_PARALLEL is not 1 (set it as a user env var, then restart Ollama)")

    import ollama_client
    try:
        pulled = ollama_client.list_models()
    except ollama_client.OllamaError as exc:
        report(BAD, str(exc))
        return 1
    report(OK, f"Ollama reachable, {len(pulled)} models pulled")

    wanted = {config.ROUTER_MODEL: "router", **{m: c for c, m in config.SPECIALISTS.items()}}
    for model, role in wanted.items():
        report(OK if ollama_client.is_pulled(model, pulled) else BAD,
               f"{role}: {model}" + ("" if ollama_client.is_pulled(model, pulled) else f"  -> ollama pull {model}"))
    esc = ollama_client.is_pulled(config.ESCALATION_MODEL, pulled)
    report(OK if esc else WARN,
           f"escalation: {config.ESCALATION_MODEL}" + ("" if esc else f"  (optional) -> ollama pull {config.ESCALATION_MODEL}"))

    print("\nAll required checks passed." if not problems else f"\n{problems} problem(s) found.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
