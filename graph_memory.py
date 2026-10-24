"""SQLite-as-graph memory with four tiers, plus chat/project management.

Tiers
  General  `general`            always included (hardware / use-case facts)
  Project  `project:<key>`      linked to General via BELONGS_TO
  Session  `session:<key>`      a chat; linked under its Project via BELONGS_TO
  Long     `longmem:<date>`     daily summaries linked via SUMMARIZES; opt-in

A chat's memory is only its own Session node (plus its Project and General),
so chats never leak into each other. Full message text lives in the
`messages` table; the Session node keeps only a short rolling `history`.
"""
import copy
import json
import sqlite3
import threading
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

import config

GENERAL_ID = "general"
SESSION_PREFIX = "session:"
PROJECT_PREFIX = "project:"
LONG_PREFIX = "longmem:"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    props TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    dst TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    rel TEXT NOT NULL,
    UNIQUE (src, dst, rel)
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);
"""


class GraphMemory:
    """Nodes + edges on SQLite. One shared connection guarded by an RLock."""

    def __init__(self, db_path) -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._cache: dict = {}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- nodes ------------------------------------------------------------
    def get_node(self, node_id: str) -> Optional[dict]:
        with self._lock:
            hit = self._cache.get(node_id)
            if hit is not None:
                return copy.deepcopy(hit)
            row = self._conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
            if row is None:
                return None
            node = {
                "id": row["id"], "kind": row["kind"], "props": json.loads(row["props"]),
                "created": row["created"], "updated": row["updated"],
            }
            self._cache[node_id] = node
            return copy.deepcopy(node)

    def put_node(self, node_id: str, kind: str, props: Optional[dict] = None) -> dict:
        """Create the node, or merge props into it. Skips the write if nothing changed."""
        props = props or {}
        with self._lock:
            existing = self.get_node(node_id)
            now = time.time()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO nodes (id, kind, props, created, updated) VALUES (?,?,?,?,?)",
                    (node_id, kind, json.dumps(props), now, now),
                )
            else:
                merged = {**existing["props"], **props}
                if merged == existing["props"]:
                    return existing
                self._conn.execute(
                    "UPDATE nodes SET props=?, updated=? WHERE id=?",
                    (json.dumps(merged), now, node_id),
                )
            self._conn.commit()
            self._cache.pop(node_id, None)
            return self.get_node(node_id)

    def set_props(self, node_id: str, **props) -> dict:
        node = self.get_node(node_id)
        if node is None:
            raise KeyError(node_id)
        return self.put_node(node_id, node["kind"], props)

    def delete_node(self, node_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
            self._conn.commit()
            self._cache.pop(node_id, None)

    def nodes_of_kind(self, kind: str) -> list:
        with self._lock:
            ids = [r["id"] for r in self._conn.execute(
                "SELECT id FROM nodes WHERE kind=? ORDER BY updated DESC", (kind,))]
            return [self.get_node(i) for i in ids]

    # --- edges ------------------------------------------------------------
    def add_edge(self, src: str, dst: str, rel: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO edges (src, dst, rel) VALUES (?,?,?)", (src, dst, rel))
            self._conn.commit()

    def remove_edges(self, src: Optional[str] = None, dst: Optional[str] = None,
                     rel: Optional[str] = None) -> None:
        clauses, args = [], []
        for col, val in (("src", src), ("dst", dst), ("rel", rel)):
            if val is not None:
                clauses.append(f"{col}=?")
                args.append(val)
        if not clauses:
            raise ValueError("remove_edges needs at least one filter")
        with self._lock:
            self._conn.execute(f"DELETE FROM edges WHERE {' AND '.join(clauses)}", args)
            self._conn.commit()

    def neighbors(self, node_id: str, rel: Optional[str] = None, incoming: bool = False) -> list:
        col_from, col_to = ("dst", "src") if incoming else ("src", "dst")
        sql = f"SELECT {col_to} AS other FROM edges WHERE {col_from}=?"
        args = [node_id]
        if rel:
            sql += " AND rel=?"
            args.append(rel)
        with self._lock:
            return [r["other"] for r in self._conn.execute(sql, args)]

    # --- messages ---------------------------------------------------------
    def add_message(self, chat_id: str, role: str, content: str, meta: Optional[dict] = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (chat_id, role, content, meta, ts) VALUES (?,?,?,?,?)",
                (chat_id, role, content, json.dumps(meta or {}), time.time()))
            self._conn.commit()

    def get_messages(self, chat_id: str, limit: Optional[int] = None) -> list:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, meta, ts FROM messages WHERE chat_id=? ORDER BY id", (chat_id,)
            ).fetchall()
        msgs = [{"role": r["role"], "content": r["content"],
                 "meta": json.loads(r["meta"]), "ts": r["ts"]} for r in rows]
        return msgs[-limit:] if limit else msgs


# --- rendering helpers -------------------------------------------------------

def _one_line(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _cap(text: str, cap: int, keep_tail: bool = False) -> str:
    if len(text) <= cap:
        return text
    return "…" + text[-(cap - 1):] if keep_tail else text[: cap - 1].rstrip() + "…"


def _render(props: dict) -> str:
    """key: value lines; list values become `· item` lines."""
    lines = []
    for key, value in props.items():
        if isinstance(value, (list, tuple)):
            if value:
                lines.append(f"{key}:")
                lines.extend(f"· {item}" for item in value)
        elif value not in (None, ""):
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


class MemoryManager:
    """The four tiers plus chat and project management."""

    def __init__(self, db_path=config.DB_PATH) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.graph = GraphMemory(db_path)
        self.graph.put_node(GENERAL_ID, "General", {})

    # --- General ----------------------------------------------------------
    def seed_general(self, facts: dict) -> None:
        self.graph.put_node(GENERAL_ID, "General", facts)

    # --- Projects ---------------------------------------------------------
    def create_project(self, name: str, notes: str = "") -> dict:
        key = uuid.uuid4().hex[:8]
        self.graph.put_node(PROJECT_PREFIX + key, "Project",
                            {"name": name.strip() or "Untitled project", "notes": notes, "topics": []})
        self.graph.add_edge(PROJECT_PREFIX + key, GENERAL_ID, "BELONGS_TO")
        return self.project_summary(key)

    def project_summary(self, key: str) -> dict:
        node = self.graph.get_node(PROJECT_PREFIX + key)
        if node is None:
            raise KeyError(key)
        return {"id": key, "name": node["props"].get("name", ""),
                "notes": node["props"].get("notes", ""), "updated": node["updated"]}

    def list_projects(self) -> list:
        nodes = self.graph.nodes_of_kind("Project")
        return [{"id": n["id"][len(PROJECT_PREFIX):], "name": n["props"].get("name", ""),
                 "notes": n["props"].get("notes", ""), "updated": n["updated"]} for n in nodes]

    def update_project(self, key: str, name: Optional[str] = None, notes: Optional[str] = None) -> dict:
        changes = {}
        if name is not None:
            changes["name"] = name.strip() or "Untitled project"
        if notes is not None:
            changes["notes"] = notes
        self.graph.set_props(PROJECT_PREFIX + key, **changes)
        return self.project_summary(key)

    def delete_project(self, key: str, delete_chats: bool = False) -> None:
        pid = PROJECT_PREFIX + key
        if self.graph.get_node(pid) is None:
            raise KeyError(key)
        for chat_id in self.graph.neighbors(pid, "BELONGS_TO", incoming=True):
            if not chat_id.startswith(SESSION_PREFIX):
                continue
            if delete_chats:
                self.graph.delete_node(chat_id)
            else:
                self.graph.set_props(chat_id, project_key=None)
        self.graph.delete_node(pid)

    # --- Chats (Session nodes) -------------------------------------------
    def create_chat(self, project_key: Optional[str] = None, title: str = config.DEFAULT_CHAT_TITLE,
                    chat_id: Optional[str] = None) -> dict:
        if project_key and self.graph.get_node(PROJECT_PREFIX + project_key) is None:
            raise KeyError(f"project {project_key}")
        chat_id = chat_id or uuid.uuid4().hex[:12]
        sid = SESSION_PREFIX + chat_id
        self.graph.put_node(sid, "Session", {"title": title, "project_key": project_key, "history": []})
        if project_key:
            self.graph.add_edge(sid, PROJECT_PREFIX + project_key, "BELONGS_TO")
        return self.chat_summary(chat_id)

    def ensure_session(self, chat_id: Optional[str] = None, project_key: Optional[str] = None) -> dict:
        """Return the chat, creating it (optionally inside a project) if it doesn't exist."""
        if chat_id and self.graph.get_node(SESSION_PREFIX + chat_id) is not None:
            return self.chat_summary(chat_id)
        if project_key and self.graph.get_node(PROJECT_PREFIX + project_key) is None:
            project_key = None
        return self.create_chat(project_key=project_key, chat_id=chat_id)

    def chat_summary(self, chat_id: str) -> dict:
        node = self.graph.get_node(SESSION_PREFIX + chat_id)
        if node is None:
            raise KeyError(chat_id)
        return {"id": chat_id, "title": node["props"].get("title", config.DEFAULT_CHAT_TITLE),
                "project_key": node["props"].get("project_key"),
                "created": node["created"], "updated": node["updated"]}

    def list_chats(self) -> list:
        return [self.chat_summary(n["id"][len(SESSION_PREFIX):])
                for n in self.graph.nodes_of_kind("Session")]

    def rename_chat(self, chat_id: str, title: str) -> dict:
        self.graph.set_props(SESSION_PREFIX + chat_id, title=_one_line(title, 80) or config.DEFAULT_CHAT_TITLE)
        return self.chat_summary(chat_id)

    def move_chat(self, chat_id: str, project_key: Optional[str]) -> dict:
        sid = SESSION_PREFIX + chat_id
        if self.graph.get_node(sid) is None:
            raise KeyError(chat_id)
        if project_key and self.graph.get_node(PROJECT_PREFIX + project_key) is None:
            raise KeyError(f"project {project_key}")
        self.graph.remove_edges(src=sid, rel="BELONGS_TO")
        if project_key:
            self.graph.add_edge(sid, PROJECT_PREFIX + project_key, "BELONGS_TO")
        self.graph.set_props(sid, project_key=project_key or None)
        return self.chat_summary(chat_id)

    def delete_chat(self, chat_id: str) -> None:
        if self.graph.get_node(SESSION_PREFIX + chat_id) is None:
            raise KeyError(chat_id)
        self.graph.delete_node(SESSION_PREFIX + chat_id)

    def autotitle(self, chat_id: str, first_message: str) -> dict:
        """Title a fresh chat from its first message."""
        summary = self.chat_summary(chat_id)
        if summary["title"] == config.DEFAULT_CHAT_TITLE:
            return self.rename_chat(chat_id, _one_line(first_message, 48))
        return summary

    # --- messages (full text) ---------------------------------------------
    def add_message(self, chat_id: str, role: str, content: str, meta: Optional[dict] = None) -> None:
        self.graph.add_message(SESSION_PREFIX + chat_id, role, content, meta)

    def get_messages(self, chat_id: str) -> list:
        if self.graph.get_node(SESSION_PREFIX + chat_id) is None:
            raise KeyError(chat_id)
        return self.graph.get_messages(SESSION_PREFIX + chat_id)

    # --- rolling session memory ---------------------------------------------
    def append_session_turn(self, chat_id: str, user_message: str, category: str, reply: str) -> None:
        """Keep a bounded rolling history (last N turns) instead of overwriting."""
        sid = SESSION_PREFIX + chat_id
        node = self.graph.get_node(sid)
        if node is None:
            raise KeyError(chat_id)
        note = (f"User ({category}): {_one_line(user_message, 120)} "
                f"→ Assistant: {_one_line(reply, 140)}")
        history = list(node["props"].get("history") or [])[-(config.SESSION_HISTORY_TURNS - 1):] + [note]
        self.graph.set_props(sid, history=history)
        project_key = node["props"].get("project_key")
        if project_key and self.graph.get_node(PROJECT_PREFIX + project_key):
            pid = PROJECT_PREFIX + project_key
            topics = list(self.graph.get_node(pid)["props"].get("topics") or [])
            topics = (topics + [_one_line(user_message, 80)])[-config.PROJECT_TOPICS_KEPT:]
            self.graph.set_props(pid, topics=topics)

    # --- context for a request ---------------------------------------------
    def build_context(self, session_key: Optional[str] = None, project_key: Optional[str] = None,
                      include_long_memory: bool = False) -> str:
        cap = config.MEMORY_CAP_CHARS
        parts = []
        general = self.graph.get_node(GENERAL_ID)
        if general and general["props"]:
            parts.append("[General]\n" + _cap(_render(general["props"]), cap))

        session = self.graph.get_node(SESSION_PREFIX + session_key) if session_key else None
        if not project_key and session:
            project_key = session["props"].get("project_key")
        project = self.graph.get_node(PROJECT_PREFIX + project_key) if project_key else None
        if project:
            props = {k: v for k, v in project["props"].items() if k in ("name", "notes", "topics")}
            parts.append("[Project]\n" + _cap(_render(props), cap))
        if session and session["props"].get("history"):
            parts.append("[This chat so far]\n" + _cap(
                _render({"recent": session["props"]["history"]}), cap, keep_tail=True))
        if include_long_memory:
            longs = self.latest_long_memories(config.LONG_MEMORIES_IN_CONTEXT)
            if longs:
                text = "\n".join(f"· {m['date']}: {m['summary']}" for m in longs)
                parts.append("[Long-term summaries]\n" + _cap(text, cap))
        return "\n\n".join(parts)

    # --- Long memory ---------------------------------------------------------
    def add_long_memory(self, summary: str, covers: list, day: Optional[str] = None) -> str:
        day = day or date.today().isoformat()
        lid = LONG_PREFIX + day
        self.graph.put_node(lid, "LongMemory", {"date": day, "summary": summary.strip()})
        for target in covers:
            if self.graph.get_node(target) is not None:
                self.graph.add_edge(lid, target, "SUMMARIZES")
        return lid

    def latest_long_memories(self, n: int = 3) -> list:
        nodes = sorted(self.graph.nodes_of_kind("LongMemory"),
                       key=lambda x: x["props"].get("date", ""), reverse=True)[:n]
        return [{"date": x["props"].get("date", ""), "summary": x["props"].get("summary", "")}
                for x in nodes]

    def gather_for_summary(self) -> list:
        """(node_id, text) for every Project and Session node that has content."""
        items = []
        for node in self.graph.nodes_of_kind("Project") + self.graph.nodes_of_kind("Session"):
            p = node["props"]
            label = p.get("name") or p.get("title") or node["id"]
            body = _render({k: v for k, v in p.items() if k in ("notes", "topics", "history")})
            if body:
                items.append((node["id"], f"{node['kind']} '{label}':\n{body}"))
        return items
