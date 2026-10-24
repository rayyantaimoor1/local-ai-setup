"""Soft RAM trimmer (Windows only).

Calls EmptyWorkingSet on eligible processes when free RAM is low. Nothing is
closed; Windows simply re-faults pages on demand. Raises on other platforms.
"""
import ctypes
import sys
import time
from ctypes import wintypes

import psutil

import config

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_SET_QUOTA = 0x0100
_ACCESS = PROCESS_QUERY_INFORMATION | PROCESS_SET_QUOTA


def _load_winapi():
    if sys.platform != "win32":
        raise RuntimeError("memory_manager is Windows-only (EmptyWorkingSet).")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    psapi.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
    psapi.EmptyWorkingSet.restype = wintypes.BOOL
    return kernel32, psapi


def free_ram_mb() -> float:
    return psutil.virtual_memory().available / (1024 * 1024)


def _excluded(name: str) -> bool:
    name = name.lower()
    return any(token in name for token in config.MEMORY_TRIM_EXCLUDE)


def trim_working_sets() -> dict:
    """Trim every eligible process unconditionally; returns a result summary."""
    kernel32, psapi = _load_winapi()
    before = free_ram_mb()
    trimmed = skipped = 0
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            info = proc.info
            mem = info.get("memory_info")
            if _excluded(info.get("name") or "") or mem is None:
                skipped += 1
                continue
            if mem.rss < config.MEMORY_TRIM_MIN_PROCESS_MB * 1024 * 1024:
                continue
            handle = kernel32.OpenProcess(_ACCESS, False, info["pid"])
            if not handle:
                skipped += 1
                continue
            try:
                if psapi.EmptyWorkingSet(handle):
                    trimmed += 1
                else:
                    skipped += 1
            finally:
                kernel32.CloseHandle(handle)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            skipped += 1
    time.sleep(0.5)  # let Windows update its counters
    after = free_ram_mb()
    return {
        "triggered": True,
        "before_mb": round(before),
        "after_mb": round(after),
        "freed_mb": round(after - before),
        "trimmed": trimmed,
        "skipped": skipped,
    }


def trim_if_needed(threshold_mb: int = config.MIN_FREE_RAM_MB_BEFORE_LOAD) -> dict:
    """Trim only when free RAM is below the threshold."""
    free = free_ram_mb()
    if free >= threshold_mb:
        return {"triggered": False, "before_mb": round(free), "after_mb": round(free),
                "freed_mb": 0, "trimmed": 0, "skipped": 0}
    return trim_working_sets()


if __name__ == "__main__":
    print(trim_working_sets())
