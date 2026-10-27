"""Questions that need the user's click before anything happens (today: git pushes).

The chat pipeline yields a `push_approval` event; the browser shows the push dialog and answers
through POST /api/agent/approve, which lands in `approvals.decide`. Nothing is pushed unless the
user clicks, and a closed tab or a timeout counts as "no".
"""
import threading
import time
import uuid
from collections.abc import Iterator

import constitution
import tools

APPROVAL_TIMEOUT = 900  # seconds


class _Approvals:
    """Pending questions, answered from the UI. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict = {}
        self._taken: dict = {}

    def create(self) -> str:
        aid = uuid.uuid4().hex[:10]
        with self._lock:
            self._items[aid] = {"event": threading.Event(), "approved": False, "overrides": []}
        return aid

    def decide(self, aid: str, approved: bool, overrides=()) -> bool:
        """Record the user's answer; `overrides` are rule ids they overrode for this one action."""
        with self._lock:
            item = self._items.get(aid)
        if item is None:
            return False
        item["approved"] = bool(approved)
        item["overrides"] = [r for r in overrides if r in constitution.RULE_BY_ID] if approved else []
        item["event"].set()
        return True

    def take_overrides(self, aid: str) -> list:
        """The overrides that came with an approved answer (read once, after wait())."""
        with self._lock:
            return self._taken.pop(aid, [])

    def pending(self) -> list:
        with self._lock:
            return list(self._items)

    def wait(self, aid: str, cancel=None, timeout: float = APPROVAL_TIMEOUT) -> bool:
        """True only if the user approved; a timeout or a closed browser tab counts as no."""
        with self._lock:
            item = self._items.get(aid)
        if item is None:
            return False
        deadline = time.time() + timeout
        try:
            while not item["event"].wait(0.5):
                if (cancel is not None and cancel.is_set()) or time.time() > deadline:
                    return False
            with self._lock:
                self._taken[aid] = item["overrides"]
            return item["approved"]
        finally:
            with self._lock:
                self._items.pop(aid, None)


approvals = _Approvals()


def request_push(args=(), kwargs=None, cancel=None) -> Iterator[dict]:
    """Ask the user to approve a push. Yields events; returns the text the model gets back.

    Relative repo paths mean the project folder. The push happens only through
    tools.guarded_push, so the constitution check runs again at the moment of pushing.
    """
    params = dict(zip(("repo_path", "remote", "branch"), args))
    params.update({k: v for k, v in (kwargs or {}).items() if k in ("repo_path", "remote", "branch")})
    repo = str(tools._resolve(params.get("repo_path") or "."))
    remote, branch = str(params.get("remote") or "origin"), str(params.get("branch") or "")
    review = tools.git_push_review(repo, remote, branch)
    if not review["ok"]:
        return f"Error: {review['error']}"
    if review["nothing_to_push"]:
        return "Nothing to push: the remote already has every commit on this branch."
    aid = approvals.create()
    yield {"type": "push_approval", "id": aid, "review": review["text"], "violations": review["violations"],
           "repo": review["repo"], "remote": remote, "branch": review["branch"], "host": review["host"]}
    yield {"type": "phase", "phase": "waiting_for_approval"}
    approved = approvals.wait(aid, cancel)
    offered = {v["rule"] for v in review["violations"]}
    granted = [r for r in approvals.take_overrides(aid) if r in offered]
    yield {"type": "approval_result", "id": aid, "approved": approved, "kind": "push"}
    if not approved:
        constitution.audit("push_declined", repo=review["repo"], branch=review["branch"], via="chat")
        return "The user declined the push. Nothing was pushed. Do not ask again unless the user requests it."
    try:
        result = tools.guarded_push(repo, remote, review["branch"], overrides=granted, confirmed=True)
    except tools.PolicyBlocked as exc:
        return f"Blocked by the constitution: {exc}. Tell the user what to fix; only they can override a rule."
    except (ValueError, PermissionError) as exc:
        return f"Error: {exc}"
    return "Pushed. " + result["output"][-500:]
