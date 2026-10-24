"""FastAPI backend. Bound to 127.0.0.1 only; no auth by design."""
import json
import threading
from pathlib import Path
from typing import Optional

import anyio
import psutil
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agent_engine
import config
import constitution
import ollama_client
import router_engine
import tools
from graph_memory import MemoryManager
from ollama_client import OllamaError

GENERAL_FACTS = {
    "machine": "HP EliteBook 840 G10 laptop, CPU only (no dedicated GPU), 16 GB RAM, Windows 11",
    "setup": "Fully offline local multi-agent AI assistant using Ollama; one specialist model loaded at a time",
    "primary_use": "Coding help; slower answers are accepted in exchange for quality",
    "owner": "Scorpion (Muhammad Rayyan Taimoor Khan)",
}

app = FastAPI(title="Local AI")
# Defeats DNS-rebinding: only loopback Host headers are served.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

_LOCAL_ORIGINS = {"127.0.0.1", "localhost", "::1"}


@app.middleware("http")
async def block_cross_site_writes(request: Request, call_next):
    """A website open in your browser must not be able to click this app's buttons for you.

    Browsers label cross-site requests (Sec-Fetch-Site, Origin); anything state-changing that is
    not from this app's own page is refused. Matters most for pushes and overrides.
    """
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        cross = request.headers.get("sec-fetch-site") == "cross-site"
        if cross or (origin is not None and urlparse(origin).hostname not in _LOCAL_ORIGINS):
            constitution.audit("cross_site_blocked", path=request.url.path, origin=origin or "")
            return JSONResponse({"detail": "Cross-site requests are not allowed."}, status_code=403)
    return await call_next(request)


memory = MemoryManager(config.DB_PATH)
memory.seed_general(GENERAL_FACTS)

# Ollama runs one request at a time (OLLAMA_NUM_PARALLEL=1, 16 GB RAM).
GEN_LOCK = threading.Lock()

# Folders the agent has used; only these can be browsed through /api/workspace/file.
KNOWN_WORKSPACES = {str(config.DEFAULT_WORKSPACE.resolve())}
MAX_VIEW_BYTES = 200_000


# --- request models ------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    session_key: Optional[str] = None  # legacy alias for chat_id
    project_key: Optional[str] = None
    include_long_memory: bool = False
    deep: bool = False


class AgentRequest(BaseModel):
    goal: str
    workspace: Optional[str] = None
    allow_escalation: bool = True
    auto_approve: bool = False  # skip the "may I run this?" question for risky generated code


class ApproveRequest(BaseModel):
    id: str
    approved: bool
    overrides: list[str] = []  # rules the user overrides for this one action
    confirm_text: str = ""  # must be OVERRIDE when a core rule is overridden


class RepoRequest(BaseModel):
    repo_path: str = "."
    remote: str = "origin"
    branch: str = ""


class OverrideGrant(BaseModel):
    rule: str
    minutes: float


class PushRequest(BaseModel):
    repo_path: str = "."
    confirmed: bool = False
    remote: str = "origin"
    branch: str = ""
    force: bool = False
    overrides: list[str] = []
    confirm_text: str = ""


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


class FileSearchRequest(BaseModel):
    query: str
    path: str = "."
    max_results: int = 40


class FetchRequest(BaseModel):
    url: str
    max_chars: int = 20000


class ChatCreate(BaseModel):
    project_key: Optional[str] = None
    title: Optional[str] = None


class ChatUpdate(BaseModel):
    title: str


class ChatMove(BaseModel):
    project_key: Optional[str] = None


class ProjectCreate(BaseModel):
    name: str
    notes: str = ""


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None


# --- streaming plumbing ---------------------------------------------------------

_END = object()


def _next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return _END


async def _sse(generator, cancel: threading.Event):
    """Drive a blocking event generator from async code, as Server-Sent Events.

    If the client goes away, `cancel` is set so the model stream stops and the
    generation lock is released.
    """
    iterator = iter(generator)
    try:
        while True:
            event = await anyio.to_thread.run_sync(_next, iterator)
            if event is _END:
                break
            yield f"data: {json.dumps(event)}\n\n"
    finally:
        cancel.set()
        try:
            iterator.close()
        except ValueError:  # generator still mid-step in its worker thread
            pass


def _sse_response(generator, cancel: threading.Event) -> StreamingResponse:
    return StreamingResponse(
        _sse(generator, cancel), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _locked(factory, cancel: threading.Event):
    """Run factory()'s events while holding the generation lock (one model at a time)."""
    queued = False
    while not GEN_LOCK.acquire(timeout=0.5):
        if cancel.is_set():
            return
        if not queued:
            queued = True
            yield {"type": "phase", "phase": "queued"}
    try:
        yield from factory()
    finally:
        GEN_LOCK.release()


def _chat_events(req: ChatRequest, cancel: threading.Event):
    """Chat pipeline: persist messages + memory around the router's events."""
    try:
        chat = memory.ensure_session(req.chat_id or req.session_key, req.project_key)
    except KeyError as exc:
        yield {"type": "error", "message": f"Not found: {exc}"}
        return
    chat_id = chat["id"]
    yield {"type": "chat", "chat": chat}
    memory.add_message(chat_id, "user", req.message)
    context = memory.build_context(chat_id, include_long_memory=req.include_long_memory)

    partial: list = []
    finished = False
    try:
        for event in _locked(lambda: router_engine.process_request(
                req.message, context, req.deep, cancel), cancel):
            kind = event["type"]
            if kind == "token":
                partial.append(event["text"])
            elif kind == "reset":
                partial.clear()
            elif kind == "done":
                finished = True
                memory.add_message(chat_id, "assistant", event["reply"], event["meta"])
                memory.append_session_turn(chat_id, req.message, event["meta"]["category"], event["reply"])
                event["chat"] = memory.autotitle(chat_id, req.message)
            elif kind == "error":
                finished = True
                memory.add_message(chat_id, "error", event["message"])
            yield event
    finally:
        if not finished and partial:
            memory.add_message(chat_id, "assistant", "".join(partial), {"interrupted": True})


# --- chat -------------------------------------------------------------------------

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    cancel = threading.Event()
    return _sse_response(_chat_events(req, cancel), cancel)


@app.post("/api/chat")
def chat(req: ChatRequest):
    cancel = threading.Event()
    for event in _chat_events(req, cancel):
        if event["type"] == "error":
            raise HTTPException(503, event["message"])
        if event["type"] == "done":
            return {"reply": event["reply"], **event["meta"], "chat": event["chat"]}
    raise HTTPException(500, "No reply produced.")


# --- agent --------------------------------------------------------------------------

def _agent_events(req: AgentRequest, cancel: threading.Event):
    workspace = Path(req.workspace).expanduser() if req.workspace else config.DEFAULT_WORKSPACE
    KNOWN_WORKSPACES.add(str(workspace.resolve()))
    yield from _locked(lambda: agent_engine.run_agent(
        req.goal, workspace, req.allow_escalation, cancel, req.auto_approve), cancel)


@app.post("/api/agent/stream")
async def agent_stream(req: AgentRequest):
    cancel = threading.Event()
    return _sse_response(_agent_events(req, cancel), cancel)


@app.post("/api/agent/run")
def agent_run(req: AgentRequest):
    cancel = threading.Event()
    for event in _agent_events(req, cancel):
        if event["type"] in ("approval", "push_approval"):  # no UI to ask: declined (pushes are never automatic)
            agent_engine.approvals.decide(event["id"], False)
        elif event["type"] == "error":
            raise HTTPException(503, event["message"])
        elif event["type"] == "done":
            return event["result"]
    raise HTTPException(500, "The agent produced no result.")


def _workspace_target(workspace: str, path: str) -> Path:
    """A file inside a workspace the agent has used, or an HTTP error."""
    root = Path(workspace).expanduser()
    if str(root.resolve()) not in KNOWN_WORKSPACES:
        raise HTTPException(403, "That folder is not an agent workspace.")
    try:
        target = agent_engine._resolve_in_workspace(root, path)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not target.is_file():
        raise HTTPException(404, f"Not found: {path}")
    return target


@app.get("/api/workspace/file")
def workspace_file(workspace: str, path: str):
    """Read back a text file the agent wrote (confined to a workspace the agent has used)."""
    data = _workspace_target(workspace, path).read_bytes()[:MAX_VIEW_BYTES]
    return {"path": path, "content": data.decode("utf-8", errors="replace")}


@app.get("/api/workspace/download")
def workspace_download(workspace: str, path: str):
    """Download any file the agent made (Word, Excel, PDF, images...), confined the same way."""
    target = _workspace_target(workspace, path)
    return FileResponse(target, filename=target.name)


@app.post("/api/agent/approve")
def agent_approve(body: ApproveRequest):
    """Answer the agent's "may I run this code / push this branch?" question."""
    overrides = []
    if body.approved and body.overrides:
        try:
            overrides = constitution.check_overrides(body.overrides, body.confirm_text)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    if not agent_engine.approvals.decide(body.id, body.approved, overrides):
        raise HTTPException(404, "That approval request is no longer pending.")
    if overrides:
        constitution.audit("override_used", rules=overrides, via="agent approval", approval=body.id)
    return {"id": body.id, "approved": body.approved, "overrides": overrides}


# --- chats and projects (sidebar) -------------------------------------------------------

def _not_found(exc: KeyError):
    return HTTPException(404, f"Not found: {exc.args[0]}")


@app.get("/api/chats")
def list_chats():
    return {"chats": memory.list_chats(), "projects": memory.list_projects()}


@app.post("/api/chats")
def create_chat(body: ChatCreate):
    try:
        return memory.create_chat(body.project_key, body.title or config.DEFAULT_CHAT_TITLE)
    except KeyError as exc:
        raise _not_found(exc)


@app.get("/api/chats/{chat_id}/messages")
def chat_messages(chat_id: str):
    try:
        return {"chat": memory.chat_summary(chat_id), "messages": memory.get_messages(chat_id)}
    except KeyError as exc:
        raise _not_found(exc)


@app.patch("/api/chats/{chat_id}")
def rename_chat(chat_id: str, body: ChatUpdate):
    try:
        return memory.rename_chat(chat_id, body.title)
    except KeyError as exc:
        raise _not_found(exc)


@app.post("/api/chats/{chat_id}/move")
def move_chat(chat_id: str, body: ChatMove):
    try:
        return memory.move_chat(chat_id, body.project_key or None)
    except KeyError as exc:
        raise _not_found(exc)


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    try:
        memory.delete_chat(chat_id)
    except KeyError as exc:
        raise _not_found(exc)
    return {"deleted": chat_id}


@app.post("/api/projects")
def create_project(body: ProjectCreate):
    return memory.create_project(body.name, body.notes)


@app.patch("/api/projects/{key}")
def update_project(key: str, body: ProjectUpdate):
    try:
        return memory.update_project(key, body.name, body.notes)
    except KeyError as exc:
        raise _not_found(exc)


@app.delete("/api/projects/{key}")
def delete_project(key: str, delete_chats: bool = False):
    try:
        memory.delete_project(key, delete_chats)
    except KeyError as exc:
        raise _not_found(exc)
    return {"deleted": key}


# --- git / web / repo --------------------------------------------------------------------

@app.post("/api/git/status")
def git_status(body: RepoRequest):
    return {"output": tools.git_status(body.repo_path)}


@app.post("/api/git/pull")
def git_pull(body: RepoRequest):
    return {"output": tools.git_pull(body.repo_path)}


@app.post("/api/git/log")
def git_log(body: RepoRequest):
    return {"output": tools.git_log(body.repo_path, 15)}


@app.post("/api/git/diff")
def git_diff(body: RepoRequest):
    return {"output": tools.git_diff(body.repo_path)}


@app.post("/api/git/diff-for-push-review")
def git_diff_for_push_review(body: RepoRequest):
    """What a push would send plus the constitution's verdict. Never pushes."""
    review = tools.git_push_review(body.repo_path, body.remote, body.branch)
    constitution.audit("push_reviewed", repo=review["repo"], branch=review["branch"], ok=review["ok"],
                       rules=[v["rule"] for v in review["violations"]])
    return {"output": tools.git_diff_for_push_review(body.repo_path, body.remote, body.branch), "review": review}


@app.post("/api/git/push")
def git_push(body: PushRequest):
    """The only way the UI pushes: your confirmation + a clean constitution check (or your overrides)."""
    if body.confirmed is not True:
        raise HTTPException(403, "git push requires explicit user confirmation (confirmed=true).")
    try:
        overrides = constitution.check_overrides(body.overrides, body.confirm_text)
        result = tools.guarded_push(body.repo_path, body.remote, body.branch, body.force, overrides, True)
    except tools.PolicyBlocked as exc:
        return JSONResponse(status_code=409, content={"detail": f"Blocked by the constitution: {exc}",
                                                      "violations": exc.violations})
    except PermissionError as exc:
        raise HTTPException(403, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if overrides:
        constitution.audit("override_used", rules=overrides, via="push button", repo=result["review"]["repo"])
    return {"output": result["output"], "pushed": result["pushed"], "review": result["review"]}


# --- constitution ---------------------------------------------------------------------------

@app.get("/api/constitution")
def get_constitution():
    return {
        "text": constitution.full_text(),
        "rules": constitution.rules_view(),
        "settings": constitution.load_settings(),
        "overrides": constitution.overrides.active(),
        "max_override_minutes": constitution.MAX_OVERRIDE_MINUTES,
        "confirm_phrase": constitution.CONFIRM_PHRASE,
    }


@app.put("/api/constitution/settings")
def put_constitution_settings(body: dict):
    try:
        return {"settings": constitution.save_settings(body)}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/constitution/overrides")
def grant_override(body: OverrideGrant):
    try:
        constitution.overrides.grant(body.rule, body.minutes)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"overrides": constitution.overrides.active()}


@app.delete("/api/constitution/overrides")
def revoke_override(rule: Optional[str] = None):
    constitution.overrides.revoke(rule)
    return {"overrides": constitution.overrides.active()}


@app.get("/api/audit")
def get_audit(limit: int = 60):
    return {"entries": constitution.read_audit(min(max(limit, 1), 500))}


@app.post("/api/web-search")
def web_search(body: SearchRequest):
    try:
        results = tools.web_search_results(body.query, body.max_results)
    except Exception as exc:  # offline, rate-limited...
        raise HTTPException(502, f"Web search failed: {type(exc).__name__}: {exc}")
    return {"results": results}


@app.get("/api/repo-tree")
def repo_tree(path: str = "."):
    return {"output": tools.get_directory_tree(path)}


# --- documents, search, web page, system ----------------------------------------------------

async def _read_upload(request: Request) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > tools.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (limit {tools.MAX_UPLOAD_BYTES // 2**20} MB).")
    data = await request.body()
    if not data:
        raise HTTPException(400, "The file is empty.")
    if len(data) > tools.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (limit {tools.MAX_UPLOAD_BYTES // 2**20} MB).")
    return data


@app.post("/api/docs/extract")
async def docs_extract(request: Request, filename: str):
    """Raw file bytes in the request body; returns the extracted text."""
    data = await _read_upload(request)
    try:
        return await anyio.to_thread.run_sync(tools.extract_document, filename, data)
    except ValueError as exc:
        raise HTTPException(415, str(exc))
    except Exception as exc:  # corrupt / encrypted file
        raise HTTPException(422, f"Could not read {filename}: {type(exc).__name__}: {exc}")


@app.post("/api/docs/pdf-to-docx")
async def docs_pdf_to_docx(request: Request, filename: str = "document.pdf"):
    data = await _read_upload(request)
    try:
        docx_bytes = await anyio.to_thread.run_sync(tools.pdf_bytes_to_docx_bytes, data)
    except ValueError as exc:
        raise HTTPException(415, str(exc))
    except Exception as exc:
        raise HTTPException(422, f"Could not convert {filename}: {type(exc).__name__}: {exc}")
    return Response(
        docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


@app.post("/api/files/search")
def files_search(body: FileSearchRequest):
    return {"output": tools.search_files(body.query, body.path, body.max_results)}


@app.post("/api/web/fetch")
def web_fetch(body: FetchRequest):
    try:
        text = tools.fetch_webpage_text(body.url, body.max_chars)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"url": body.url, "text": text, "chars": len(text)}


@app.get("/api/system")
def system_info():
    return {"output": tools.get_system_info()}


# --- memory / status -------------------------------------------------------------------------

@app.get("/api/memory")
def get_memory(session_key: Optional[str] = None, project_key: Optional[str] = None,
               include_long_memory: bool = False):
    return {
        "context": memory.build_context(session_key, project_key, include_long_memory),
        "session": memory.graph.get_node("session:" + session_key) if session_key else None,
        "project": memory.graph.get_node("project:" + project_key) if project_key else None,
    }


@app.get("/api/status")
def status():
    vm = psutil.virtual_memory()
    try:
        pulled, ollama_up = ollama_client.list_models(), True
    except OllamaError:
        pulled, ollama_up = [], False
    roster = [{"role": "ROUTER", "model": config.ROUTER_MODEL}]
    roster += [{"role": cat, "model": model} for cat, model in config.SPECIALISTS.items()]
    roster.append({"role": "DEEP", "model": config.ESCALATION_MODEL})
    for entry in roster:
        entry["pulled"] = ollama_client.is_pulled(entry["model"], pulled)
    return {
        "ollama_up": ollama_up,
        "ram_total_mb": round(vm.total / 2**20),
        "ram_free_mb": round(vm.available / 2**20),
        "threads": config.NUM_THREAD,
        "models": roster,
        "default_workspace": str(config.DEFAULT_WORKSPACE),
        "project_dir": str(config.BASE_DIR),
        "busy": GEN_LOCK.locked(),
    }


# --- UI --------------------------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


config.STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
