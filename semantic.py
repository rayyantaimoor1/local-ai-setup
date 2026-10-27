"""Embeddings and a small local vector database: semantic routing, document search (RAG) and chat recall.

The embedding model (nomic-embed-text, ~0.3 GB) runs in Ollama like the chat models. Vectors live in
SQLite (`vectors.db`, next to the chat database) and are searched with numpy: exact cosine similarity
over a normalized matrix, a few milliseconds for tens of thousands of chunks, with no extra server or
heavy dependency. Everything stays on this machine.

Collections
  docs      chunks of files you indexed (PDF, Word, text, code) -> search_documents tool, Tools tab
  chats     one entry per answered message -> "related from earlier chats" (long-term memory switch)
  router    labelled example requests -> embedding router (replaces most slow router-model calls)
"""
import hashlib
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

import config
import constitution

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection TEXT NOT NULL,
    source TEXT NOT NULL,
    ref TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    vec BLOB NOT NULL,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(collection, source);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    chunks INTEGER NOT NULL,
    indexed REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

# nomic-embed-text is trained with task prefixes; they measurably improve retrieval
_PREFIX = {"query": "search_query: ", "document": "search_document: ", "classify": "classification: "}


class EmbedError(RuntimeError):
    """The embedding model is missing or Ollama is not answering; callers fall back gracefully."""


def embed(texts: list, kind: str = "document") -> np.ndarray:
    """Normalized float32 vectors, one row per text."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    rows = []
    for start in range(0, len(texts), config.EMBED_BATCH):
        batch = [_PREFIX[kind] + t[: config.EMBED_MAX_CHARS] for t in texts[start:start + config.EMBED_BATCH]]
        try:
            r = httpx.post(config.OLLAMA_URL + "/api/embed", timeout=300, json={
                "model": config.EMBED_MODEL, "input": batch, "keep_alive": config.EMBED_KEEP_ALIVE,
                "options": {"num_thread": config.NUM_THREAD, "num_gpu": 0}})
        except httpx.HTTPError as exc:
            raise EmbedError(f"Ollama is not answering ({type(exc).__name__}).") from None
        if r.status_code == 404:
            raise EmbedError(f"The embedding model is not pulled. Run: ollama pull {config.EMBED_MODEL}")
        if r.status_code != 200:
            raise EmbedError(f"Ollama embedding error {r.status_code}: {r.text[:200]}")
        rows.extend(r.json()["embeddings"])
    mat = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list:
    """Paragraph-aware chunks of about `size` characters; the tail of each chunk repeats at the start
    of the next one so a sentence on a boundary is still found."""
    text = re.sub(r"\r\n?", "\n", text or "").strip()
    if not text:
        return []
    pieces = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for piece in pieces:
        while len(piece) > size:  # a huge paragraph or a long code block: cut it on line breaks
            cut = piece.rfind("\n", 0, size)
            cut = cut if cut > size // 3 else size
            pieces_head, piece = piece[:cut], piece[cut:].lstrip("\n")
            if current:
                chunks.append(current)
                current = ""
            chunks.append(pieces_head)
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            current = current[-overlap:].split(" ", 1)[-1] + "\n\n" + piece if overlap else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


_WORD = re.compile(r"[a-z0-9_]{2,}")
_STOP = frozenset("""a an and are as at be by for from has have how i in is it its of on or that the this to was
were what when where which who why will with you your do does did can my me we our not""".split())


def tokens(text: str) -> list:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class BM25:
    """Keyword relevance (Okapi BM25) over a fixed list of texts: finds exact terms such as error codes,
    names and IDs that a meaning-based search can rank too low."""

    def __init__(self, texts: list, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b, self.n = k1, b, len(texts)
        self.lengths = np.zeros(self.n, dtype=np.float32)
        self.postings: dict = {}
        for i, text in enumerate(texts):
            toks = tokens(text)
            self.lengths[i] = len(toks)
            counts: dict = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                self.postings.setdefault(t, []).append((i, c))
        self.avg = float(self.lengths.mean()) if self.n else 1.0

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for t in set(tokens(query)):
            plist = self.postings.get(t)
            if not plist:
                continue
            idf = np.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avg, 1.0))
                out[i] += idf * tf * (self.k1 + 1) / norm
        return out


def fuse(*rankings, k: int = 60) -> dict:
    """Reciprocal rank fusion: {index: score} from several best-first rankings (a robust, tuning-free
    way to combine meaning-based and keyword-based results)."""
    fused: dict = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (k + rank)
    return fused


class VectorStore:
    """Vectors in SQLite, searched with numpy. One connection, guarded by a lock; matrices cached per
    collection and rebuilt after a write."""

    def __init__(self, path=None) -> None:
        self.path = Path(path or config.VECTOR_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._cache: dict = {}
        self._keywords: dict = {}
        self._check_model()

    def _check_model(self) -> None:
        """Vectors from a different embedding model are meaningless: start over if it changed."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key='model'").fetchone()
            if row and row[0] != config.EMBED_MODEL:
                self._conn.execute("DELETE FROM chunks")
                self._conn.execute("DELETE FROM files")
            self._conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('model', ?)", (config.EMBED_MODEL,))
            self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
            self._conn.commit()

    def add(self, collection: str, source: str, texts: list, vecs: np.ndarray, ref: str = "") -> None:
        with self._lock:
            now = time.time()
            self._conn.executemany(
                "INSERT INTO chunks (collection, source, ref, text, vec, created) VALUES (?,?,?,?,?,?)",
                [(collection, source, ref, t, v.astype(np.float32).tobytes(), now) for t, v in zip(texts, vecs)])
            self._conn.commit()
            self._cache.pop(collection, None)
            self._keywords.pop(collection, None)

    def delete(self, collection: str, source: Optional[str] = None) -> int:
        with self._lock:
            if source is None:
                cur = self._conn.execute("DELETE FROM chunks WHERE collection=?", (collection,))
            else:
                cur = self._conn.execute("DELETE FROM chunks WHERE collection=? AND source=?", (collection, source))
            self._conn.commit()
            self._cache.pop(collection, None)
            self._keywords.pop(collection, None)
            return cur.rowcount

    def count(self, collection: str) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM chunks WHERE collection=?", (collection,)).fetchone()[0]

    def _matrix(self, collection: str) -> tuple:
        with self._lock:
            hit = self._cache.get(collection)
            if hit is None:
                rows = self._conn.execute(
                    "SELECT id, source, ref, text, vec FROM chunks WHERE collection=? ORDER BY id", (collection,)).fetchall()
                if rows:
                    mat = np.vstack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
                    meta = [(r[0], r[1], r[2], r[3]) for r in rows]
                else:
                    mat, meta = np.zeros((0, 1), dtype=np.float32), []
                hit = self._cache[collection] = (mat, meta)
            return hit

    def search(self, collection: str, qvec: np.ndarray, k: int = 6, source_prefix: Optional[str] = None,
               exclude_ref: Optional[str] = None, min_score: float = 0.0) -> list:
        mat, meta = self._matrix(collection)
        if not meta or mat.shape[1] != qvec.shape[-1]:
            return []
        scores = mat @ qvec.reshape(-1)
        order = np.argsort(-scores)
        out = []
        prefix = source_prefix.replace("\\", "/").lower() if source_prefix else None
        for i in order:
            if scores[i] < min_score or len(out) >= k:
                break
            cid, source, ref, text = meta[i]
            if prefix and not source.replace("\\", "/").lower().startswith(prefix):
                continue
            if exclude_ref and ref == exclude_ref:
                continue
            out.append({"score": float(scores[i]), "source": source, "ref": ref, "text": text})
        return out

    def hybrid_search(self, collection: str, qvec: np.ndarray, query: str, k: int = 6,
                      source_prefix: Optional[str] = None, min_score: float = 0.3, pool: int = 40) -> list:
        """Meaning (vectors) and exact words (BM25) together, fused by rank. A passage qualifies when it is
        close in meaning (cosine >= min_score) or contains the query's words."""
        mat, meta = self._matrix(collection)
        if not meta:
            return []
        with self._lock:
            kw_index = self._keywords.get(collection)
            if kw_index is None:
                kw_index = self._keywords[collection] = BM25([m[3] for m in meta])
        cos = mat @ qvec.reshape(-1) if mat.shape[1] == qvec.shape[-1] else np.zeros(len(meta), dtype=np.float32)
        kw = kw_index.scores(query)
        allowed = np.ones(len(meta), dtype=bool)
        if source_prefix:
            prefix = source_prefix.replace("\\", "/").lower()
            allowed = np.array([m[1].replace("\\", "/").lower().startswith(prefix) for m in meta])
        by_vec = [i for i in np.argsort(-cos) if allowed[i]][:pool]
        by_kw = [i for i in np.argsort(-kw) if allowed[i] and kw[i] > 0][:pool]
        fused = fuse(by_vec, by_kw)
        out = []
        for i in sorted(fused, key=fused.get, reverse=True):
            if cos[i] < min_score and kw[i] <= 0:
                continue
            cid, source, ref, text = meta[i]
            out.append({"score": float(cos[i]), "keyword": bool(kw[i] > 0), "source": source, "ref": ref, "text": text})
            if len(out) >= k:
                break
        return out

    # --- indexed files ---------------------------------------------------------------------------------
    def file_state(self, path: str) -> Optional[tuple]:
        with self._lock:
            row = self._conn.execute("SELECT mtime, size FROM files WHERE path=?", (path,)).fetchone()
            return (row[0], row[1]) if row else None

    def set_file(self, path: str, mtime: float, size: int, chunks: int) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO files (path, mtime, size, chunks, indexed) VALUES (?,?,?,?,?)",
                               (path, mtime, size, chunks, time.time()))
            self._conn.commit()

    def forget_file(self, path: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM files WHERE path=?", (path,))
            self._conn.commit()
        self.delete("docs", path)

    def files(self) -> list:
        with self._lock:
            return [{"path": r[0], "chunks": r[1], "indexed": r[2]} for r in
                    self._conn.execute("SELECT path, chunks, indexed FROM files ORDER BY path").fetchall()]


_store: Optional[VectorStore] = None
_store_lock = threading.Lock()


def store() -> VectorStore:
    global _store
    with _store_lock:
        if _store is None or _store.path != Path(config.VECTOR_DB_PATH):
            _store = VectorStore()
        return _store


# --- document library (RAG) -----------------------------------------------------------------------------

_DOC_EXTS = {".pdf", ".docx"}


def _file_text(path: Path) -> Optional[str]:
    import tools  # late import: tools imports this module for its search tools

    ext = path.suffix.lower()
    if ext not in _DOC_EXTS and ext not in tools.TEXT_EXTENSIONS:
        return None
    limit = config.INDEX_MAX_DOC_BYTES if ext in _DOC_EXTS else config.INDEX_MAX_TEXT_BYTES
    if path.stat().st_size > limit:
        return None
    try:
        return tools.extract_document(path.name, path.read_bytes())["text"]
    except (ValueError, OSError):
        return None


def index_path(path: str = ".") -> str:
    """Add a folder's (or one file's) documents to the searchable library: PDF, Word, text and code.

    Unchanged files are skipped, changed ones re-indexed, deleted ones forgotten. Secret-looking files
    are never indexed and secrets inside files are blanked out (constitution 1.3)."""
    import tools

    root = tools._resolve(path)
    if not root.exists():
        return f"Error: not found: {root}"
    started, st = time.time(), store()
    files = [root] if root.is_file() else []
    if root.is_dir():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in tools.IGNORED_DIRS and not d.startswith(".")]
            files.extend(Path(dirpath) / f for f in filenames)
            if len(files) > config.INDEX_MAX_FILES:
                break
    added = unchanged = skipped = chunks_total = 0
    seen = set()
    for fp in files[:config.INDEX_MAX_FILES]:
        key = str(fp.resolve())
        seen.add(key)
        try:
            stat = fp.stat()
        except OSError:
            continue
        if tools._is_sensitive(fp):
            skipped += 1
            continue
        if st.file_state(key) == (stat.st_mtime, stat.st_size):
            unchanged += 1
            continue
        text = _file_text(fp)
        if not text or text.startswith("(no extractable text"):
            skipped += 1
            continue
        if constitution.enforced("secret_files"):
            text = constitution.redact(text)
        pieces = chunk_text(text)
        st.delete("docs", key)
        if pieces:
            labelled = [f"{fp.name}\n{p}" for p in pieces]  # the file name helps retrieval
            st.add("docs", key, pieces, embed(labelled, "document"), ref=fp.name)
        st.set_file(key, stat.st_mtime, stat.st_size, len(pieces))
        added += 1
        chunks_total += len(pieces)
    gone = 0
    if root.is_dir():
        prefix = str(root.resolve()).lower()
        for f in st.files():
            if f["path"].lower().startswith(prefix) and f["path"] not in seen and not Path(f["path"]).exists():
                st.forget_file(f["path"])
                gone += 1
    return (f"Indexed {added} file(s) ({chunks_total} passages) in {time.time() - started:.0f}s; "
            f"{unchanged} unchanged, {skipped} skipped (unsupported, empty or private), {gone} removed. "
            f"The library now holds {st.count('docs')} passages.")


def search_documents(query: str, path: str = "", k: int = 5) -> str:
    """Search the indexed document library by meaning (not exact words); path limits it to one folder."""
    import tools

    st = store()
    if not st.count("docs"):
        return "The document library is empty. Index a folder first with index_documents(path=...)."
    prefix = str(tools._resolve(path).resolve()) if path else None
    k = min(max(int(k), 4), 10)  # small models ask for k=1 and then miss the passage that answers
    hits = st.hybrid_search("docs", embed([query], "query")[0], query, k=k, source_prefix=prefix, min_score=0.3)
    if not hits:
        return f"No passages matching '{query}'" + (f" under {path}" if path else "") + "."
    return "\n\n".join(f"[{i}] {Path(h['source']).name} (relevance {h['score']:.2f}):\n{h['text'][:700]}"
                       for i, h in enumerate(hits, 1))


def retrieve(query: str, text: str, budget: int) -> str:
    """The parts of a long text most relevant to the query, in their original order, within `budget`
    characters (RAG for a large attached document instead of cutting it off at the end)."""
    pieces = chunk_text(text, size=800, overlap=100)
    if sum(len(p) for p in pieces) <= budget:
        return text
    vecs = embed(pieces, "document")
    q = embed([query], "query")[0]
    kw = BM25(pieces).scores(query)
    fused = fuse(np.argsort(-(vecs @ q)), [i for i in np.argsort(-kw) if kw[i] > 0])
    order = sorted(fused, key=fused.get, reverse=True)
    keep, used = {0}, len(pieces[0])  # always keep the start: titles and introductions matter
    for i in order:
        if used + len(pieces[i]) > budget:
            continue
        keep.add(int(i))
        used += len(pieces[i])
    return "\n[...]\n".join(pieces[i] for i in sorted(keep))


# --- chat recall --------------------------------------------------------------------------------------------

def remember_turn(chat_id: str, user_message: str, reply: str) -> None:
    """Store one answered exchange for recall from other chats. Best effort: never raises."""
    try:
        text = f"User: {user_message.strip()[:600]}\nAssistant: {reply.strip()[:600]}"
        if constitution.enforced("secret_files"):
            text = constitution.redact(text)
        store().add("chats", chat_id, [text], embed([text], "document"), ref=chat_id)
    except (EmbedError, sqlite3.Error, ValueError):
        pass


def recall(query: str, exclude_chat: Optional[str] = None, k: int = 3) -> list:
    """Earlier exchanges from OTHER chats that are about the same thing (empty if none is close)."""
    try:
        if not store().count("chats"):
            return []
        return store().search("chats", embed([query], "query")[0], k=k, exclude_ref=exclude_chat,
                              min_score=config.RECALL_MIN_SCORE)
    except EmbedError:
        return []


def forget_chat(chat_id: str) -> None:
    store().delete("chats", chat_id)


# --- embedding router -----------------------------------------------------------------------------------------

def _examples_hash(examples: dict) -> str:
    return hashlib.sha256(repr(sorted((k, tuple(v)) for k, v in examples.items())).encode()).hexdigest()[:16]


def _router_bank(examples: dict) -> None:
    st = store()
    digest = _examples_hash(examples) + config.EMBED_MODEL
    if st.get_meta("router_examples") == digest and st.count("router"):
        return
    st.delete("router")
    for label, texts in examples.items():
        st.add("router", label, list(texts), embed(list(texts), "classify"))
    st.set_meta("router_examples", digest)


def route(message: str, examples: dict) -> tuple:
    """(label, confidence 0-1, ranking) by nearest labelled examples; raises EmbedError when unavailable.

    Score per label = mean similarity of its 3 closest examples; confidence = gap to the runner-up,
    scaled so 0.5+ is a clear decision."""
    _router_bank(examples)
    q = embed([message[:1500]], "classify")[0]
    mat, meta = store()._matrix("router")
    sims = mat @ q
    per = {}
    for (cid, label, ref, text), s in zip(meta, sims):
        per.setdefault(label, []).append(float(s))
    scored = sorted(((float(np.mean(sorted(v, reverse=True)[:3])), lab) for lab, v in per.items()), reverse=True)
    best, second = scored[0], scored[1] if len(scored) > 1 else (0.0, "")
    confidence = max(0.0, min(1.0, (best[0] - second[0]) / config.ROUTER_EMBED_MARGIN))
    return best[1], confidence, [(lab, round(s, 3)) for s, lab in scored]
