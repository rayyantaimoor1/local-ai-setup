"""The constitution in code: rules, settings, overrides, audit log, secret detection.

`constitution.md` is the human text. This module is what actually enforces it, so
a model cannot argue its way past a rule. Only the user (through the app's buttons,
which call the API) can change settings or grant overrides; there is no tool that
lets a model do either.

Tiers
  core      data-security rules. Cannot be switched off. Overridable one action at a time.
  standard  workflow rules. Can be switched off in settings, or lifted for up to 60 minutes.
"""
import fnmatch
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import config

MAX_OVERRIDE_MINUTES = 60

RULES = [
    {"id": "secrets_in_push", "tier": "core", "title": "Never push secrets",
     "description": "Blocks a push whose commits contain keys, tokens, passwords or private keys."},
    {"id": "sensitive_files_in_push", "tier": "core", "title": "Never push sensitive files",
     "description": "Blocks .env files, keys, databases, credentials and similar files."},
    {"id": "no_force_push", "tier": "core", "title": "Never rewrite remote history",
     "description": "Blocks force-pushes. The assistant has no way to force-push at all."},
    {"id": "secrets_in_documents", "tier": "core", "title": "Never copy secrets into documents",
     "description": "Word, PDF, Excel and other files the assistant creates are refused if the content looks like a secret."},
    {"id": "secrets_outbound", "tier": "core", "title": "No secrets in searches or URLs",
     "description": "Web searches and page fetches that contain something that looks like a secret are refused."},
    {"id": "secret_files", "tier": "core", "title": "Don't read secret files",
     "description": "The assistant's file tools skip .env files, keys, credentials and the chat database."},
    {"id": "protected_branches", "tier": "standard", "title": "No direct pushes to protected branches",
     "description": "Pushes to branches such as main or master are blocked unless you allow them."},
    {"id": "approved_remotes", "tier": "standard", "title": "Push only to approved hosts",
     "description": "Pushes to a host that is not on your approved list are blocked."},
    {"id": "large_files", "tier": "standard", "title": "No huge files in a push",
     "description": "Blocks pushes containing files above the size limit."},
]
RULE_BY_ID = {r["id"]: r for r in RULES}
STANDARD_IDS = [r["id"] for r in RULES if r["tier"] == "standard"]

DEFAULT_SETTINGS = {
    "protected_branches": ["main", "master"],
    "allowed_remote_hosts": ["github.com"],  # add "local" to allow plain folder remotes
    "max_push_file_mb": 20,
    "extra_sensitive_patterns": [],
    "disabled_rules": [],  # standard rules only
}


# --- settings --------------------------------------------------------------------------------

_lock = threading.RLock()


def load_settings() -> dict:
    settings = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT_SETTINGS.items()}
    with _lock:
        try:
            data = json.loads(Path(config.CONSTITUTION_SETTINGS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return settings
    if isinstance(data, dict):
        for key in settings:
            value = data.get(key)
            if key == "max_push_file_mb":
                valid = isinstance(value, (int, float)) and not isinstance(value, bool) and 1 <= value <= 500
            else:
                valid = isinstance(value, list) and all(isinstance(item, str) for item in value)
            if valid:
                settings[key] = value
    settings["disabled_rules"] = [r for r in settings["disabled_rules"] if r in STANDARD_IDS]  # core can't be disabled
    return settings


def _clean_list(value, name: str, max_items: int = 30, max_len: int = 120) -> list:
    if not isinstance(value, list) or len(value) > max_items:
        raise ValueError(f"{name} must be a list of at most {max_items} items.")
    out = []
    for item in value:
        text = str(item).strip()
        if not text or len(text) > max_len:
            raise ValueError(f"{name} contains an empty or too-long entry.")
        if text not in out:
            out.append(text)
    return out


def save_settings(changes: dict) -> dict:
    """Validate and persist changes made by the user. Raises ValueError on bad input."""
    current = load_settings()
    new = dict(current)
    if "protected_branches" in changes:
        new["protected_branches"] = _clean_list(changes["protected_branches"], "protected_branches")
    if "allowed_remote_hosts" in changes:
        new["allowed_remote_hosts"] = [h.lower() for h in _clean_list(changes["allowed_remote_hosts"], "allowed_remote_hosts")]
    if "extra_sensitive_patterns" in changes:
        new["extra_sensitive_patterns"] = _clean_list(changes["extra_sensitive_patterns"], "extra_sensitive_patterns")
    if "max_push_file_mb" in changes:
        mb = changes["max_push_file_mb"]
        if isinstance(mb, bool) or not isinstance(mb, (int, float)) or not 1 <= mb <= 500:
            raise ValueError("max_push_file_mb must be a number between 1 and 500.")
        new["max_push_file_mb"] = mb
    if "disabled_rules" in changes:
        disabled = _clean_list(changes["disabled_rules"], "disabled_rules")
        bad = [r for r in disabled if r not in STANDARD_IDS]
        if bad:
            raise ValueError(f"Core rules cannot be switched off: {', '.join(bad)}. Override them one action at a time instead.")
        new["disabled_rules"] = disabled
    with _lock:
        path = Path(config.CONSTITUTION_SETTINGS)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(new, indent=2), encoding="utf-8")
        tmp.replace(path)
    diff = {k: new[k] for k in new if new[k] != current[k]}
    if diff:
        audit("settings_changed", changed=diff)
    return new


def rules_view() -> list:
    disabled = set(load_settings()["disabled_rules"])
    return [{**r, "locked": r["tier"] == "core", "enabled": r["id"] not in disabled} for r in RULES]


def rule_enabled(rule_id: str) -> bool:
    return rule_id not in load_settings()["disabled_rules"]


# --- overrides (the user's authority) -------------------------------------------------------------

class _Overrides:
    """Timed lifts: one standard rule, or the WHOLE constitution (every rule, core ones too).

    In memory only: gone when the app restarts, and every lift is logged.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._until: dict = {}
        self._all_until = 0.0

    def grant_all(self, minutes: float) -> float:
        """Lift the whole constitution for a while. The user's call; logged."""
        if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not 1 <= minutes <= MAX_OVERRIDE_MINUTES:
            raise ValueError(f"Minutes must be between 1 and {MAX_OVERRIDE_MINUTES}.")
        with self._lock:
            self._all_until = time.time() + minutes * 60
        audit("constitution_lifted", minutes=minutes)
        return self._all_until

    def lifted_seconds(self) -> int:
        """Seconds left on a whole-constitution lift (0 = the rules apply)."""
        with self._lock:
            left = self._all_until - time.time()
            if left <= 0:
                self._all_until = 0.0
                return 0
            return int(left)

    def grant(self, rule_id: str, minutes: float) -> float:
        rule = RULE_BY_ID.get(rule_id)
        if rule is None:
            raise ValueError(f"Unknown rule '{rule_id}'.")
        if rule["tier"] != "standard":
            raise ValueError("Core rules can only be overridden one action at a time, never for a time window.")
        if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not 1 <= minutes <= MAX_OVERRIDE_MINUTES:
            raise ValueError(f"Minutes must be between 1 and {MAX_OVERRIDE_MINUTES}.")
        until = time.time() + minutes * 60
        with self._lock:
            self._until[rule_id] = until
        audit("override_granted", rule=rule_id, minutes=minutes)
        return until

    def revoke(self, rule_id=None) -> None:
        """Revoke one rule's override, or all of them (and a whole-constitution lift) when rule_id is None."""
        with self._lock:
            if rule_id is None:
                removed = list(self._until)
                restored = self._all_until > time.time()
                self._all_until = 0.0
            else:
                removed = [rule_id] if rule_id in self._until else []
                restored = False
            for r in removed:
                del self._until[r]
        for r in removed:
            audit("override_revoked", rule=r)
        if restored:
            audit("constitution_restored")

    def is_active(self, rule_id: str) -> bool:
        if self.lifted_seconds():
            return True  # the whole constitution is lifted
        with self._lock:
            until = self._until.get(rule_id)
            if until is not None and until < time.time():
                del self._until[rule_id]
                until = None
        return until is not None

    def active(self) -> list:
        now = time.time()
        with self._lock:
            self._until = {r: u for r, u in self._until.items() if u > now}
            return [{"rule": r, "seconds_left": int(u - now)} for r, u in sorted(self._until.items())]


overrides = _Overrides()


def enforced(rule_id: str) -> bool:
    """Should this rule be applied right now? False while it (or the whole constitution) is lifted
    or while a standard rule is switched off in settings."""
    if overrides.is_active(rule_id):
        return False
    return not (RULE_BY_ID.get(rule_id, {}).get("tier") == "standard" and not rule_enabled(rule_id))


# --- audit log ---------------------------------------------------------------------------------------------

def audit(event: str, **fields) -> None:
    """Append one JSON line. Strings are scrubbed of anything that looks like a secret."""
    def clean(value):
        if isinstance(value, str):
            return redact(value)[:500]
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [clean(v) for v in list(value)[:50]]
        return value

    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **clean(fields)}
    path = Path(config.AUDIT_LOG)
    with _lock:
        try:
            if path.exists() and path.stat().st_size > 2_000_000:
                path.replace(path.with_suffix(".1.jsonl"))
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass  # auditing must never break the action it records


def read_audit(limit: int = 50) -> list:
    with _lock:
        try:
            lines = Path(config.AUDIT_LOG).read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
    out = []
    for line in lines[-int(limit):]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return list(reversed(out))  # newest first


# --- secrets and sensitive files -----------------------------------------------------------------------------

SECRET_PATTERNS = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP |ENCRYPTED )?PRIVATE KEY")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("Stripe live key", re.compile(r"\b[sr]k_live_[0-9A-Za-z]{20,}")),
    ("API key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{32,}")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("password in a URL", re.compile(r"://[^/\s:@]+:[^@/\s]{3,}@")),
    ("hard-coded secret", re.compile(
        r"""(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|token)\b[\w\s"']{0,12}[:=]\s*["'][^"'\s]{10,}["']""")),
]
_PLACEHOLDER = re.compile(r"(?i)your[_-]|example|placeholder|changeme|xxxx|<[^>]+>|\*{4,}|\.\.\.")


def find_secrets(text: str) -> list:
    """Names of the kinds of secret found in text (empty list = looks clean)."""
    found = []
    for name, pattern in SECRET_PATTERNS:
        for m in pattern.finditer(text):
            if name == "hard-coded secret" and _PLACEHOLDER.search(m.group(0)):
                continue
            if name not in found:
                found.append(name)
            break
    return found


def redact(text: str) -> str:
    for name, pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


SENSITIVE_DIRS = {".ssh", ".aws", ".gnupg", ".azure", ".kube", ".docker"}
SENSITIVE_NAMES = {"id_rsa", "id_ed25519", "id_ecdsa", "id_dsa", ".netrc", "credentials", "credentials.json",
                   "secrets.json", "secret.json", "token.json", "memory.db", "audit.jsonl", ".npmrc", ".pypirc"}
SENSITIVE_SUFFIXES = {".pem", ".key", ".pfx", ".p12", ".kdbx", ".db", ".sqlite", ".sqlite3", ".db-wal",
                      ".db-shm", ".keystore", ".jks", ".ppk"}


def is_sensitive_path(path) -> bool:
    """True for files that probably hold secrets or private data (by name, never by content)."""
    p = str(path).replace("\\", "/")
    parts = [x.lower() for x in p.split("/") if x]
    if not parts:
        return False
    name = parts[-1]
    if set(parts[:-1]) & SENSITIVE_DIRS or name in SENSITIVE_NAMES or name.startswith(".env"):
        return True
    if any(name.endswith(s) for s in SENSITIVE_SUFFIXES):
        return True
    db = Path(config.DB_PATH).name.lower()
    if name in {db, db + "-wal", db + "-shm"}:
        return True
    return any(fnmatch.fnmatch(name, pat.lower()) or fnmatch.fnmatch(p.lower(), pat.lower())
               for pat in load_settings()["extra_sensitive_patterns"])


def scan_patch(patch: str) -> list:
    """[(file, kind)] for secrets in the ADDED lines of a unified diff / git log -p output."""
    hits, current, seen = [], "?", set()
    for line in patch.splitlines():
        if line.startswith("+++ "):
            current = line[4:].removeprefix("b/").strip()
        elif line.startswith("+") and not line.startswith("+++"):
            for kind in find_secrets(line[1:]):
                if (current, kind) not in seen:
                    seen.add((current, kind))
                    hits.append((current, kind))
    return hits


# --- push policy ----------------------------------------------------------------------------------------------------

def _violation(rule_id: str, message: str, details=()) -> dict:
    rule = RULE_BY_ID[rule_id]
    return {"rule": rule_id, "title": rule["title"], "tier": rule["tier"], "message": message,
            "details": list(details)[:30], "overridden": overrides.is_active(rule_id)}


def evaluate_push(info: dict) -> list:
    """Rule violations for a prospective push.

    info: branch, host, force, files (names), sizes {name: bytes}, secrets [(file, kind)]
    """
    s = load_settings()
    disabled = set(s["disabled_rules"])
    out = []
    if info.get("force"):
        out.append(_violation("no_force_push", "This would rewrite history on the remote."))
    if info.get("secrets"):
        out.append(_violation("secrets_in_push", "The commits being pushed appear to contain secrets.",
                              [f"{f}: {kind}" for f, kind in info["secrets"]]))
    sensitive = [f for f in info.get("files", []) if is_sensitive_path(f)]
    if sensitive:
        out.append(_violation("sensitive_files_in_push", "The commits include files that likely hold secrets or private data.", sensitive))
    branch = info.get("branch", "")
    if "protected_branches" not in disabled and branch in s["protected_branches"]:
        out.append(_violation("protected_branches", f"'{branch}' is a protected branch."))
    host = (info.get("host") or "local").lower()
    if "approved_remotes" not in disabled and host not in [h.lower() for h in s["allowed_remote_hosts"]]:
        out.append(_violation("approved_remotes", f"'{host}' is not on your approved hosts list ({', '.join(s['allowed_remote_hosts']) or 'empty'})."))
    limit = s["max_push_file_mb"] * 1024 * 1024
    big = [f"{f}: {size / 2**20:.1f} MB" for f, size in info.get("sizes", {}).items() if size > limit]
    if "large_files" not in disabled and big:
        out.append(_violation("large_files", f"Files above {s['max_push_file_mb']} MB are included.", big))
    return out


def unresolved(violations: list, request_overrides=()) -> list:
    """Violations still blocking the action: not timed-overridden and not overridden in this request."""
    granted = set(request_overrides or ())
    return [v for v in violations if not v["overridden"] and v["rule"] not in granted]


CONFIRM_PHRASE = "OVERRIDE"


def check_overrides(rule_ids, confirm_text: str = "") -> list:
    """Validate a one-action override request from the user.

    Core rules need the typed confirmation word on top of the click. Raises ValueError.
    """
    rule_ids = list(dict.fromkeys(rule_ids or []))
    unknown = [r for r in rule_ids if r not in RULE_BY_ID]
    if unknown:
        raise ValueError(f"Unknown rule(s): {', '.join(unknown)}")
    core = [r for r in rule_ids if RULE_BY_ID[r]["tier"] == "core"]
    if core and (confirm_text or "").strip() != CONFIRM_PHRASE:
        raise ValueError(f"Overriding core rule(s) ({', '.join(core)}) needs you to type {CONFIRM_PHRASE} to confirm.")
    return rule_ids


def full_text() -> str:
    try:
        return Path(config.CONSTITUTION_TEXT).read_text(encoding="utf-8")
    except OSError:
        return "(constitution.md is missing)"
