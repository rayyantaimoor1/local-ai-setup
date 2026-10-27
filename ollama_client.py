"""Thin Ollama HTTP client: blocking and streaming generation."""
import json
import re
from typing import Iterator, Optional

import httpx

import config


class OllamaError(RuntimeError):
    """Anything that stops Ollama from answering, with a human-readable message."""


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning (deepseek-r1) from a finished reply."""
    text = _THINK_RE.sub("", text)
    if "<think>" in text:  # unterminated block: drop everything after the tag
        text = text.split("<think>", 1)[0]
    return text.strip()


class ThinkFilter:
    """Streaming version of strip_think; copes with tags split across chunks."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.inside = False
        self.buf = ""

    @staticmethod
    def _partial(buf: str, tag: str) -> int:
        """Length of the longest tail of buf that is a proper prefix of tag."""
        for k in range(min(len(tag) - 1, len(buf)), 0, -1):
            if buf.endswith(tag[:k]):
                return k
        return 0

    def feed(self, text: str) -> str:
        self.buf += text
        out = []
        while True:
            if self.inside:
                i = self.buf.find(self.CLOSE)
                if i == -1:
                    keep = self._partial(self.buf, self.CLOSE)
                    self.buf = self.buf[len(self.buf) - keep:]
                    break
                self.buf = self.buf[i + len(self.CLOSE):]
                self.inside = False
            else:
                i = self.buf.find(self.OPEN)
                if i == -1:
                    keep = self._partial(self.buf, self.OPEN)
                    out.append(self.buf[:len(self.buf) - keep])
                    self.buf = self.buf[len(self.buf) - keep:]
                    break
                out.append(self.buf[:i])
                self.buf = self.buf[i + len(self.OPEN):]
                self.inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.inside else self.buf
        self.buf = ""
        return rest


def _payload(model, prompt, system, num_ctx, stream, temperature, num_predict, keep_alive=None, think=None,
             images=None) -> dict:
    options = {
        "num_thread": config.NUM_THREAD,
        "num_gpu": config.NUM_GPU,
        "num_ctx": num_ctx,
    }
    if temperature is not None:
        options["temperature"] = temperature
    if num_predict is not None:
        options["num_predict"] = num_predict
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "keep_alive": keep_alive if keep_alive is not None else config.KEEP_ALIVE,
        "options": options,
    }
    if system:
        payload["system"] = system
    if images:
        payload["images"] = list(images)  # base64 JPEG/PNG, for vision models
    if think is not None and model in config.THINKING_MODELS:
        payload["think"] = bool(think)  # only reasoning models accept it; others would reject the field
    elif model in config.THINK_OFF_MODELS:
        payload["think"] = False  # never its slow reasoning phase
    return payload


def _timeout(read: float) -> httpx.Timeout:
    return httpx.Timeout(connect=5.0, read=read, write=30.0, pool=30.0)


def _unreachable() -> OllamaError:
    return OllamaError(
        f"Cannot reach Ollama at {config.OLLAMA_URL}. "
        "Start the Ollama app (or run `ollama serve`) and try again."
    )


def _raise_for_status(status: int, body: str, model: str) -> None:
    try:
        detail = json.loads(body).get("error", body)
    except ValueError:
        detail = body
    if status == 404:
        raise OllamaError(f"Model '{model}' is not pulled. Run: ollama pull {model}")
    raise OllamaError(f"Ollama error {status} for '{model}': {detail}")


def generate(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    num_ctx: int = config.DEFAULT_NUM_CTX,
    timeout: float = config.SPECIALIST_TIMEOUT,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
) -> str:
    """Blocking generation; returns the reply with reasoning blocks removed."""
    payload = _payload(model, prompt, system, num_ctx, False, temperature, num_predict)
    try:
        resp = httpx.post(config.OLLAMA_GENERATE_URL, json=payload, timeout=_timeout(timeout))
    except httpx.ConnectError:
        raise _unreachable() from None
    except httpx.TimeoutException:
        raise OllamaError(f"Model '{model}' timed out after {timeout:.0f}s.") from None
    if resp.status_code != 200:
        _raise_for_status(resp.status_code, resp.text, model)
    return strip_think(resp.json().get("response", ""))


def stream_generate(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    num_ctx: int = config.DEFAULT_NUM_CTX,
    timeout: float = config.SPECIALIST_TIMEOUT,
    temperature: Optional[float] = None,
    cancel=None,
    keep_alive: Optional[str] = None,
    think: Optional[bool] = None,
    images: Optional[list] = None,
) -> Iterator[tuple]:
    """Yield ("thinking" | "response", text) pieces as Ollama produces them.

    `cancel` is an optional threading.Event; when set, the stream is closed.
    `keep_alive` overrides how long the model stays loaded afterwards (default: unload at once).
    `think` switches a reasoning model's (deepseek-r1) thinking on or off; ignored for other models.
    """
    payload = _payload(model, prompt, system, num_ctx, True, temperature, None, keep_alive, think, images)
    try:
        with httpx.stream(
            "POST", config.OLLAMA_GENERATE_URL, json=payload, timeout=_timeout(timeout)
        ) as resp:
            if resp.status_code != 200:
                resp.read()
                _raise_for_status(resp.status_code, resp.text, model)
            for line in resp.iter_lines():
                if cancel is not None and cancel.is_set():
                    return
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                if chunk.get("error"):
                    raise OllamaError(str(chunk["error"]))
                if chunk.get("thinking"):
                    yield ("thinking", chunk["thinking"])
                if chunk.get("response"):
                    yield ("response", chunk["response"])
                if chunk.get("done"):
                    yield ("stats", {k: chunk.get(k, 0) for k in (
                        "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")})
                    return
    except httpx.ConnectError:
        raise _unreachable() from None
    except httpx.TimeoutException:
        raise OllamaError(f"Model '{model}' stopped responding for {timeout:.0f}s.") from None


def stream_chat(
    model: str,
    messages: list,
    num_ctx: int = config.DEFAULT_NUM_CTX,
    timeout: float = config.SPECIALIST_TIMEOUT,
    temperature: Optional[float] = None,
    cancel=None,
    keep_alive: Optional[str] = None,
    think: Optional[bool] = None,
) -> Iterator[tuple]:
    """Like stream_generate, but for a list of {"role", "content"} messages (Ollama /api/chat).

    Used to continue a conversation with a model that is still loaded: the earlier messages render to
    exactly the tokens Ollama already holds in its cache, so only the new message has to be read."""
    payload = _payload(model, "", None, num_ctx, True, temperature, None, keep_alive, think)
    payload.pop("prompt", None)
    payload["messages"] = messages
    try:
        with httpx.stream("POST", config.OLLAMA_URL + "/api/chat", json=payload, timeout=_timeout(timeout)) as resp:
            if resp.status_code != 200:
                resp.read()
                _raise_for_status(resp.status_code, resp.text, model)
            for line in resp.iter_lines():
                if cancel is not None and cancel.is_set():
                    return
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                if chunk.get("error"):
                    raise OllamaError(str(chunk["error"]))
                message = chunk.get("message") or {}
                if message.get("thinking"):
                    yield ("thinking", message["thinking"])
                if message.get("content"):
                    yield ("response", message["content"])
                if chunk.get("done"):
                    yield ("stats", {k: chunk.get(k, 0) for k in (
                        "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")})
                    return
    except httpx.ConnectError:
        raise _unreachable() from None
    except httpx.TimeoutException:
        raise OllamaError(f"Model '{model}' stopped responding for {timeout:.0f}s.") from None


def model_sizes() -> dict:
    """{name: size in bytes} for every pulled model."""
    try:
        resp = httpx.get(config.OLLAMA_TAGS_URL, timeout=5.0)
    except (httpx.ConnectError, httpx.TimeoutException):
        raise _unreachable() from None
    if resp.status_code != 200:
        raise OllamaError(f"Ollama error {resp.status_code} listing models.")
    return {m.get("name", ""): int(m.get("size", 0)) for m in resp.json().get("models", [])}


def loaded_models() -> list:
    """Names of the models Ollama has in RAM right now. Best effort: [] on any problem."""
    try:
        resp = httpx.get(config.OLLAMA_URL.rstrip("/") + "/api/ps", timeout=5.0)
        return [m.get("name", "") for m in resp.json().get("models", [])] if resp.status_code == 200 else []
    except (httpx.HTTPError, ValueError):
        return []


def preload(model: str, num_ctx: int, keep_alive: str) -> bool:
    """Load a model into RAM without generating anything, with the same options a real request
    will use (a different num_ctx would make Ollama load it again). Best effort: never raises."""
    payload = _payload(model, "", None, num_ctx, False, None, None, keep_alive)
    try:
        return httpx.post(config.OLLAMA_GENERATE_URL, json=payload, timeout=180.0).status_code == 200
    except httpx.HTTPError:
        return False


def evict_except(keep: list) -> list:
    """Unload every model in RAM that is not in `keep` (one big model at a time on 16 GB). Returns the
    names unloaded. Best effort: never raises."""
    wanted = {k.removesuffix(":latest") for k in keep}
    gone = []
    for name in loaded_models():
        if name.removesuffix(":latest") not in wanted:
            unload(name)
            gone.append(name)
    return gone


def unload(model: str) -> None:
    """Ask Ollama to drop a model from RAM now. Best effort: never raises."""
    try:
        httpx.post(config.OLLAMA_GENERATE_URL, json={"model": model, "keep_alive": 0}, timeout=15.0)
    except httpx.HTTPError:
        pass


def list_models() -> list:
    """Names of models that are pulled locally (e.g. 'qwen2.5:3b')."""
    try:
        resp = httpx.get(config.OLLAMA_TAGS_URL, timeout=5.0)
    except (httpx.ConnectError, httpx.TimeoutException):
        raise _unreachable() from None
    if resp.status_code != 200:
        raise OllamaError(f"Ollama error {resp.status_code} listing models.")
    return [m.get("name", "") for m in resp.json().get("models", [])]


def ensure_running(wait: float = 60.0) -> bool:
    """Make sure the Ollama server answers, starting it if it is installed but stopped."""
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    def up() -> bool:
        try:
            list_models()
            return True
        except OllamaError:
            return False

    if up():
        return True
    exe = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if sys.platform != "win32" or not exe.exists():
        return False
    flags = 0x08000000 | 0x00000008  # CREATE_NO_WINDOW | DETACHED_PROCESS
    subprocess.Popen([str(exe), "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, creationflags=flags)
    deadline = time.time() + wait
    while time.time() < deadline:
        if up():
            return True
        time.sleep(1)
    return False


def is_pulled(model: str, pulled: list) -> bool:
    return model in pulled or f"{model}:latest" in pulled
