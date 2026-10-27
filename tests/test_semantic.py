"""Vector database, document library (RAG), chat recall, embedding router, context sizing, RAM admission.

A deterministic fake embedder (hashed bag of words) stands in for nomic-embed-text: texts that share
words are close, which is all these tests need.
"""
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import ollama_client  # noqa: E402
import router_engine  # noqa: E402
import semantic  # noqa: E402

AWS = "AKIA" + "ABCDEFGHIJKLMNOP"


def fake_vectors(texts, kind="document"):
    mat = np.zeros((len(texts), 512), dtype=np.float32)
    for i, t in enumerate(texts):
        for word in re.findall(r"[a-z0-9]+", t.lower()):
            if len(word) > 2 and word not in {"search_query", "search_document", "classification", "the", "and"}:
                mat[i, zlib.crc32(word.encode()) % 512] += 1
    mat[:, 511] += 0.01  # never an all-zero vector
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


@pytest.fixture
def emb(monkeypatch, tmp_path):
    """Fake embeddings and a fresh vector database for each test."""
    calls = []
    monkeypatch.setattr(config, "VECTOR_DB_PATH", tmp_path / "vectors.db")
    monkeypatch.setattr(semantic, "_store", None)

    def embed(texts, kind="document"):
        calls.append((kind, list(texts)))
        return fake_vectors(texts, kind)
    monkeypatch.setattr(semantic, "embed", embed)
    return calls


# --- chunking and the store ------------------------------------------------------------------------------

def test_chunks_respect_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(12))
    chunks = semantic.chunk_text(text, size=600, overlap=100)
    assert len(chunks) > 3 and all(len(c) <= 700 for c in chunks)
    assert all(c.strip() for c in chunks) and "Paragraph 0" in chunks[0] and "Paragraph 11" in chunks[-1]
    huge = semantic.chunk_text("x" * 5000, size=900)
    assert len(huge) == 6 and "".join(huge) == "x" * 5000  # an unbroken block is still cut
    assert semantic.chunk_text("   ") == []


def test_store_search_delete_and_model_change(emb, monkeypatch):
    st = semantic.store()
    texts = ["the cat sat on the mat", "stock prices fell sharply", "a kitten sleeps on a rug"]
    st.add("docs", "a.txt", texts, fake_vectors(texts))
    hits = st.search("docs", fake_vectors(["stock prices"])[0], k=2)
    assert hits[0]["text"] == "stock prices fell sharply" and hits[0]["score"] > hits[1]["score"]
    assert st.count("docs") == 3 and st.delete("docs", "a.txt") == 3 and st.count("docs") == 0
    st.add("docs", "b.txt", ["x"], fake_vectors(["x"]))
    monkeypatch.setattr(config, "EMBED_MODEL", "another-model")
    assert semantic.VectorStore(st.path).count("docs") == 0  # vectors of another model are useless: reset


# --- document library (RAG) -------------------------------------------------------------------------------

def test_index_search_reindex_and_forget(emb, tmp_path):
    lib = tmp_path / "notes"
    lib.mkdir()
    (lib / "cooking.md").write_text("# Pasta\nBoil water, add salt, cook spaghetti for nine minutes.")
    (lib / "finance.txt").write_text("Quarterly revenue grew because subscription renewals increased.")
    (lib / "keys.txt").write_text(f"deploy key {AWS} for the server")
    (lib / ".env").write_text("TOKEN=abc")
    (lib / "image.png").write_bytes(b"\x89PNG\x00\x00")
    first = semantic.index_path(str(lib))
    assert first.startswith("Indexed 3 file(s)") and "skipped" in first
    found = semantic.search_documents("how long to cook spaghetti")
    assert found.startswith("[1] cooking.md") and "nine minutes" in found
    everything = " ".join(row[3] for row in semantic.store()._matrix("docs")[1])
    assert AWS not in everything and "[REDACTED]" in everything  # secrets never enter the index
    assert ".env" not in str(semantic.store().files())
    assert "Indexed 0 file(s)" in semantic.index_path(str(lib)) and "3 unchanged" in semantic.index_path(str(lib))
    (lib / "finance.txt").unlink()
    assert "1 removed" in semantic.index_path(str(lib))
    assert "finance" not in semantic.search_documents("revenue subscription renewals")


def test_search_without_a_library_says_how_to_start(emb):
    assert "library is empty" in semantic.search_documents("anything")


def test_retrieve_keeps_the_start_and_the_relevant_part(emb):
    filler = "\n\n".join(f"Section {i}. " + "Unrelated filler sentence about weather. " * 15 for i in range(30))
    text = "Annual report 2025.\n\n" + filler + "\n\nThe warranty lasts exactly three years from purchase.\n\n" + filler
    kept = semantic.retrieve("how long is the warranty", text, budget=2500)
    assert len(kept) <= 2700 and kept.startswith("Annual report 2025.") and "three years from purchase" in kept


def test_big_attachments_are_cut_to_relevant_passages(emb):
    body = "Intro line.\n\n" + "\n\n".join("Filler paragraph about nothing much. " * 12 for _ in range(60)) + \
        "\n\nThe refund deadline is 30 days.\n\n" + "More filler text. " * 300
    message = f"--- Attached: terms.pdf ---\n{body}\n--- end of terms.pdf ---\n\nWhat is the refund deadline?"
    small, picked = router_engine.compact_attachments(message)
    assert picked and picked[0][0] == "terms.pdf" and len(small) < config.ATTACHMENT_BUDGET_CHARS + 600
    assert "refund deadline is 30 days" in small and small.rstrip().endswith("What is the refund deadline?")
    assert router_engine.compact_attachments("short question") == ("short question", [])


def test_big_attachments_without_embeddings_keep_start_and_end():
    body = "START " + "x" * 30000 + " END"
    small, picked = router_engine.compact_attachments(f"--- Attached: a.txt ---\n{body}\n--- end of a.txt ---\nsum up")
    assert "START" in small and "END" in small and len(small) < 12000 and picked[0][2] == len(body)


# --- chat recall ---------------------------------------------------------------------------------------------------

def test_recall_finds_other_chats_only(emb):
    semantic.remember_turn("chatA", "What is the wifi password at the office?", "It is on the fridge note.")
    semantic.remember_turn("chatB", "Plan a birthday party for Sara", "Balloons, cake and music.")
    hits = semantic.recall("birthday party plans for Sara", exclude_chat="chatX")
    assert hits and hits[0]["ref"] == "chatB"
    assert semantic.recall("birthday party plans for Sara", exclude_chat="chatB") == [] or \
        all(h["ref"] != "chatB" for h in semantic.recall("birthday party plans for Sara", exclude_chat="chatB"))
    semantic.forget_chat("chatB")
    assert all(h["ref"] != "chatB" for h in semantic.recall("birthday party plans for Sara"))


def test_remember_turn_never_raises_without_a_model():
    semantic.remember_turn("c", "hi", "hello")  # the default test embedder raises EmbedError: swallowed
    assert semantic.recall("hi") == []


# --- embedding router ------------------------------------------------------------------------------------------------

def test_embedding_router_decides_clear_cases_and_defers_unclear_ones(emb, monkeypatch):
    label, conf, ranking = semantic.route("Solve for x: 5x + 3 = 18", router_engine.ROUTE_EXAMPLES)
    assert label == "MATH_TASK" and conf > 0 and ranking[0][0] == "MATH_TASK"
    monkeypatch.setattr(router_engine.ollama_client, "generate", lambda *a, **k: "GENERAL_TASK")
    monkeypatch.setattr(config, "ROUTER_EMBED_MIN_CONFIDENCE", 0.0)
    assert router_engine.classify("Solve for x: 5x + 3 = 18")[0::2] == ("MATH_TASK", "embedding")
    monkeypatch.setattr(config, "ROUTER_EMBED_MIN_CONFIDENCE", 1.01)  # never confident enough
    assert router_engine.classify("Solve for x: 5x + 3 = 18")[0::2] == ("GENERAL_TASK", "router")
    n = len(emb)
    semantic.route("another message", router_engine.ROUTE_EXAMPLES)
    assert len(emb) == n + 1  # the example bank is embedded once and cached in the database


@pytest.mark.parametrize("message,core", [
    ("Why do leaves change colour in autumn? Two sentences.", "Why do leaves change colour in autumn?"),
    ("Answer in one sentence: who built the pyramids?", "who built the pyramids?"),
    ("Write a poem in 4 lines", "Write a poem"),
    ("Explain recursion briefly", "Explain recursion"),
    ("Be brief.", "Be brief."),  # nothing left: keep the original
])
def test_routing_ignores_length_instructions(message, core):
    assert router_engine.routing_text(message) == core


def test_classify_falls_back_to_the_router_model_without_embeddings(monkeypatch):
    monkeypatch.setattr(router_engine.ollama_client, "generate", lambda *a, **k: "WRITING_TASK")
    assert router_engine.classify("Write a poem")[0::2] == ("WRITING_TASK", "router")
    assert router_engine.classify("Create a Word report")[0::2] == ("CODE_TASK", "rule")


def test_library_requests_route_to_the_tools():
    for m in ("search my notes for the meeting date", "according to my documents, when is the deadline?",
              "index my documents folder"):
        assert router_engine.rule_category(m) == "CODE_TASK", m
    assert "search_documents" in router_engine.select_tools("what do my notes say about pricing?")


# --- context window and RAM admission ----------------------------------------------------------------------------

def test_context_window_grows_with_the_prompt():
    assert router_engine.fit_ctx("s", "short", 4096, 1500) == 4096
    assert router_engine.fit_ctx("s", "x" * 12000, 4096, 1500) == 8192
    assert router_engine.fit_ctx("s", "x" * 30000, 8192, 1500) == 12288
    assert router_engine.fit_ctx("s", "x" * 400000, 4096, 1500) == 16384  # capped


def test_a_long_prompt_gets_a_bigger_window_instead_of_being_cut(monkeypatch):
    seen = []

    def stream(model, prompt, system=None, num_ctx=0, **kw):
        seen.append(num_ctx)
        yield ("response", "ok")
    monkeypatch.setattr(ollama_client, "stream_generate", stream)
    gen = router_engine._stream_model("m", "x" * 20000, "sys", None, False, 4096)
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration:
        pass
    assert seen == [8192] and "context window 4096 → 8192" in events[0]["detail"]


@pytest.fixture
def ram(monkeypatch):
    state = {"free": 12 * 2**30, "loaded": [],
             "sizes": {"qwen3:8b": 5_200_000_000, "llama3.1:8b": 4_900_000_000, "llama3.2:3b": 2_000_000_000,
                       "qwen2.5-coder:7b": 4_700_000_000, "qwen2.5:3b": 1_900_000_000}}
    monkeypatch.setattr(router_engine, "_MODEL_INFO", {"at": 0.0, "sizes": {}})
    monkeypatch.setattr(ollama_client, "model_sizes", lambda: dict(state["sizes"]))
    monkeypatch.setattr(ollama_client, "loaded_models", lambda: list(state["loaded"]))
    monkeypatch.setattr(router_engine.psutil, "virtual_memory", lambda: type("VM", (), {"available": state["free"]})())
    return state


def test_fit_model_admission_control(ram):
    assert router_engine.fit_model("qwen3:8b") == ("qwen3:8b", "")
    ram["free"] = 3 * 2**30
    model, why = router_engine.fit_model("qwen3:8b")
    assert model == "llama3.2:3b" and "3.0 GB RAM free" in why
    assert router_engine.fit_model("qwen3:8b", allow_smaller=False) == ("qwen3:8b", "")  # 14B / explicit choice
    ram["loaded"] = ["qwen3:8b"]
    assert router_engine.fit_model("qwen3:8b") == ("qwen3:8b", "")  # already in RAM: nothing to load
    ram["free"], ram["loaded"] = 12 * 2**30, []
    del ram["sizes"]["qwen3:8b"]
    router_engine._MODEL_INFO["at"] = 0.0
    model, why = router_engine.fit_model("qwen3:8b")
    assert model == "llama3.1:8b" and "not downloaded" in why


# --- images ------------------------------------------------------------------------------------------------------------

def _png(w=1600, h=900):
    import base64
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_images_are_validated_and_shrunk():
    import base64
    import io
    import vision
    from PIL import Image
    out = vision.prepare_image("data:image/png;base64," + _png())
    with Image.open(io.BytesIO(base64.b64decode(out))) as img:
        assert img.format == "JPEG" and max(img.size) == config.VISION_MAX_SIDE
    for bad in ("not base64!!", base64.b64encode(b"just text, not an image").decode()):
        with pytest.raises(ValueError):
            vision.prepare_image(bad)


def test_an_image_is_read_first_then_the_specialist_answers(ram, monkeypatch):
    ram["sizes"][config.VISION_MODEL] = 3_200_000_000
    calls = []

    def stream(model, prompt, system=None, num_ctx=0, timeout=0, temperature=None, cancel=None, keep_alive=None,
               think=None, images=None):
        calls.append((model, bool(images), prompt))
        text = "Traceback: NameError: name 'pd' is not defined" if images else "Add `import pandas as pd`."
        yield ("response", text)
    monkeypatch.setattr(ollama_client, "stream_generate", stream)
    monkeypatch.setattr(ollama_client, "generate", lambda *a, **k: "CODE_TASK")
    events = list(router_engine.process_request("why does my script fail?", images=[_png(200, 100)]))
    done = events[-1]
    assert calls[0][0] == config.VISION_MODEL and calls[0][1] and "why does my script fail?" in calls[0][2]
    assert calls[1][0] == config.SPECIALISTS["CODE_TASK"] and not calls[1][1]
    assert "NameError: name 'pd' is not defined" in calls[1][2]  # the specialist sees what the image says
    assert done["reply"] == "Add `import pandas as pd`." and any(e.get("label") == "Reading the image" for e in events)


def test_a_missing_vision_model_is_explained(ram, monkeypatch):
    monkeypatch.setattr(ollama_client, "stream_generate", lambda *a, **k: iter([("response", "ok")]))
    monkeypatch.setattr(ollama_client, "generate", lambda *a, **k: "GENERAL_TASK")
    events = list(router_engine.process_request("what is this?", images=[_png(50, 50)]))
    step = next(e for e in events if e.get("label") == "Cannot read images")
    assert "ollama pull" in step["detail"] and events[-1]["type"] == "done"


def test_api_rejects_a_fake_image(client):
    events = [line for line in client.post("/api/chat/stream", json={"message": "hi", "images": ["bm90IGFuIGltYWdl"]}).text.splitlines()
              if line.startswith("data: ")]
    assert "not an image" in events[-1]


def test_the_trimmer_never_touches_the_model_runner():
    import memory_manager
    for name in ("llama-server.exe", "ollama_llama_server.exe", "ollama.exe"):
        assert memory_manager._excluded(name), name
    assert not memory_manager._excluded("chrome.exe")


# --- API -------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app, base_url="http://127.0.0.1:8420") as c:
        yield c


def test_api_library_index_search_forget(client, emb, tmp_path):
    folder = tmp_path / "lib"
    folder.mkdir()
    (folder / "guide.md").write_text("To reset the router, hold the button for ten seconds.")
    assert client.post("/api/library/index", json={"path": str(folder)}).json()["result"].startswith("Indexed 1")
    lib = client.get("/api/library").json()
    assert lib["passages"] == 1 and lib["files"][0]["path"].endswith("guide.md")
    assert "ten seconds" in client.post("/api/library/search", json={"query": "reset the router"}).json()["result"]
    assert client.post("/api/library/search", json={"query": "  "}).status_code == 400
    assert client.delete("/api/library", params={"path": str(folder)}).json() == {"removed": 1}
    assert client.get("/api/library").json()["passages"] == 0


def test_api_library_reports_a_missing_embedding_model(client):
    r = client.post("/api/library/index", json={"path": str(ROOT / "README.md")})
    assert r.status_code == 503 and "embedding" in r.json()["detail"]


def test_deleting_a_chat_forgets_it_for_recall(client, emb):
    chat = client.post("/api/chats", json={}).json()
    semantic.remember_turn(chat["id"], "favourite colour question", "Blue.")
    assert semantic.store().count("chats") == 1
    client.delete(f"/api/chats/{chat['id']}")
    assert semantic.store().count("chats") == 0


def test_the_wrong_reader_still_reads_the_file(tmp_path):
    import tools
    md = tmp_path / "notes.md"
    md.write_text("# Notes\nhello")
    assert tools.read_docx_text(str(md)).startswith("# Notes")  # a real run used read_docx_text on README.md
    assert tools.extract_pdf_text(str(md)).startswith("# Notes")
    env = tmp_path / ".env"
    env.write_text("TOKEN=x")
    assert "refusing" in tools.read_docx_text(str(env))  # the secret-file rule still applies


def test_library_search_returns_enough_passages(emb, tmp_path):
    folder = tmp_path / "d"
    folder.mkdir()
    for i in range(6):
        (folder / f"f{i}.txt").write_text(f"apple orchard harvest number {i}")
    semantic.index_path(str(folder))
    assert semantic.search_documents("apple harvest", k=1).count("relevance") == 4


def test_bm25_finds_exact_terms_and_fusion_rewards_agreement():
    docs = ["The printer shows error ERR_4471 when the tray is empty.", "Printers are useful office machines.",
            "Error messages appear on screens.", "ERR_4471 err_4471 and more ERR_4471 here"]
    scores = semantic.BM25(docs).scores("what does ERR_4471 mean")
    assert scores[0] > 0 and scores[3] > scores[0] and scores[1] == 0 and scores[2] == 0
    fused = semantic.fuse([0, 1, 2], [2, 0])
    assert max(fused, key=fused.get) == 0  # second in one list and first in the other beats first in one only


def test_hybrid_search_marks_exact_word_hits(emb):
    st = semantic.store()
    texts = ["Reset code ZX81Q unlocks the admin panel.", "General notes about the office kitchen."]
    st.add("docs", "a.txt", texts, fake_vectors(texts))
    hits = st.hybrid_search("docs", fake_vectors(["ZX81Q"])[0], "ZX81Q", k=2, min_score=0.99)
    assert len(hits) == 1 and hits[0]["keyword"] and "ZX81Q" in hits[0]["text"]  # kept for its exact word


def test_library_questions_are_told_to_search_first(emb, monkeypatch):
    texts = ["Office hours are 9 to 5."]
    semantic.store().add("docs", "h.txt", texts, fake_vectors(texts))
    seen = []

    def stream(model, prompt, system=None, **kw):
        seen.append(prompt)
        yield ("response", "ok")
    monkeypatch.setattr(ollama_client, "stream_generate", stream)
    list(router_engine.process_request("According to my documents, what are the office hours?"))
    assert router_engine.LIBRARY_HINT in seen[0]
    seen.clear()
    monkeypatch.setattr(ollama_client, "generate", lambda *a, **k: "CODE_TASK")
    list(router_engine.process_request("Show me the files in my project"))
    assert router_engine.LIBRARY_HINT not in seen[0]


def test_daily_summary_keeps_the_newest_material_and_no_secrets():
    import summarize_memory
    items = [("new", "Session 'new': " + "N" * 5000), ("old", "Session 'old': " + "O" * 5000), ("k", f"key {AWS}")]
    notes = summarize_memory.build_notes(items)
    assert notes.startswith("Session 'new'") and "O" * 1100 not in notes and len(notes) == summarize_memory.MAX_INPUT_CHARS
    assert AWS not in summarize_memory.build_notes([("k", f"key {AWS}")])
