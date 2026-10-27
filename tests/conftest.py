"""Runs before any test module is imported: point every on-disk setting at a temp folder so the
tests can never touch your real chat database, constitution settings or audit log."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="localai-tests-"))
os.environ["LOCAL_AI_DB"] = str(_TMP / "test-memory.db")
os.environ["LOCAL_AI_CONSTITUTION"] = str(_TMP / "constitution.json")
os.environ["LOCAL_AI_AUDIT"] = str(_TMP / "audit.jsonl")
os.environ["LOCAL_AI_OUTPUTS"] = str(_TMP / "outputs")
os.environ["LOCAL_AI_VECTORS"] = str(_TMP / "vectors.db")
os.environ["LOCAL_AI_EMBED_MODEL"] = "no-such-embedding-model-in-tests"  # even a stray real call cannot embed

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def no_real_embeddings(monkeypatch):
    """Tests never reach a real embedding model: by default it is 'not pulled', so the router-model
    path is used; tests of semantic features install a deterministic fake (fake_embed below)."""
    import semantic

    def missing(texts, kind="document"):
        raise semantic.EmbedError("embedding model not available in tests")
    monkeypatch.setattr(semantic, "embed", missing)
    yield


@pytest.fixture(autouse=True)
def no_real_model_listing(monkeypatch):
    """RAM admission control must not depend on this machine's real Ollama and free RAM: by default the
    model list is unknown (fit_model keeps the chosen model). Tests that need it install their own."""
    import ollama_client
    import router_engine

    def unknown():
        raise ollama_client.OllamaError("no model listing in tests")
    monkeypatch.setattr(ollama_client, "model_sizes", unknown)
    monkeypatch.setattr(ollama_client, "loaded_models", lambda: [])
    monkeypatch.setattr(router_engine, "_MODEL_INFO", {"at": 0.0, "sizes": {}})
    yield


@pytest.fixture(autouse=True)
def fresh_approvals():
    """Tests that fake `wait` leave pending approvals behind; never let one answer another test's question."""
    from approvals import approvals
    approvals._items.clear()
    approvals._taken.clear()
    yield
