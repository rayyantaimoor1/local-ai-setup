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

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_approvals():
    """Tests that fake `wait` leave pending approvals behind; never let one answer another test's question."""
    import agent_engine
    agent_engine.approvals._items.clear()
    agent_engine.approvals._taken.clear()
    yield
