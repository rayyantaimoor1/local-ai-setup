"""Live system meters for the sidebar: CPU load, RAM use, model RAM and CPU temperature.

Temperature comes from Windows' "Thermal Zone Information" performance counters (readable without
admin rights) through the PDH API via ctypes, so there is nothing to install and no PowerShell
process is started on each poll. On machines without those counters the temperature is None.
"""
import ctypes
import sys
import threading
from ctypes import wintypes
from typing import Optional

import psutil

_PDH_FMT_DOUBLE = 0x00000200
_PDH_MORE_DATA = 0x800007D2
_MODEL_PROCESSES = ("ollama", "llama-server", "ollama_llama_server")


class _FmtValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]


class _FmtItem(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _FmtValue)]


class _Thermal:
    """One PDH query on '\\Thermal Zone Information(*)\\Temperature', opened on first use."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.query = self.counter = None
        self.failed = sys.platform != "win32"

    def _open(self) -> bool:
        pdh = ctypes.windll.pdh
        query, counter = wintypes.HANDLE(), wintypes.HANDLE()
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
            return False
        path = "\\Thermal Zone Information(*)\\Temperature"
        if pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(counter)) != 0:
            pdh.PdhCloseQuery(query)
            return False
        self.query, self.counter = query, counter
        return True

    def read(self) -> dict:
        """{zone name: degrees C} for every thermal zone, or {} when unavailable."""
        if self.failed:
            return {}
        with self.lock:
            try:
                if self.query is None and not self._open():
                    self.failed = True
                    return {}
                pdh = ctypes.windll.pdh
                if pdh.PdhCollectQueryData(self.query) != 0:
                    return {}
                size, count = wintypes.DWORD(0), wintypes.DWORD(0)
                status = pdh.PdhGetFormattedCounterArrayW(self.counter, _PDH_FMT_DOUBLE, ctypes.byref(size),
                                                          ctypes.byref(count), None)
                if status & 0xFFFFFFFF != _PDH_MORE_DATA or not size.value:
                    return {}
                buf = ctypes.create_string_buffer(size.value)
                if pdh.PdhGetFormattedCounterArrayW(self.counter, _PDH_FMT_DOUBLE, ctypes.byref(size),
                                                    ctypes.byref(count), buf) != 0:
                    return {}
                items = ctypes.cast(buf, ctypes.POINTER(_FmtItem))
                zones = {}
                for i in range(count.value):
                    kelvin = items[i].FmtValue.doubleValue
                    if items[i].FmtValue.CStatus in (0, 1) and 200 < kelvin < 400:  # valid data, sane range
                        zones[str(items[i].szName).lstrip("\\")] = round(kelvin - 273.15, 1)
                return zones
            except (OSError, AttributeError, ValueError):
                self.failed = True
                return {}


_thermal = _Thermal()


def cpu_temperature(zones: Optional[dict] = None) -> Optional[float]:
    """The CPU zone's temperature in C (the hottest zone if none is named after the CPU), or None."""
    zones = _thermal.read() if zones is None else zones
    if not zones:
        return None
    cpu = [t for name, t in zones.items() if "cpu" in name.lower()]
    return max(cpu) if cpu else max(zones.values())


def model_ram_gb() -> float:
    """RAM held by Ollama and the model runners it starts."""
    total = 0
    for proc in psutil.process_iter(["name", "memory_info"]):
        name = (proc.info.get("name") or "").lower()
        mem = proc.info.get("memory_info")
        if mem and name.startswith(_MODEL_PROCESSES):
            total += mem.rss
    return round(total / 2**30, 1)


def snapshot() -> dict:
    vm = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),  # since the previous call: no waiting
        "ram_used_gb": round((vm.total - vm.available) / 2**30, 1),
        "ram_total_gb": round(vm.total / 2**30, 1),
        "ram_percent": vm.percent,
        "model_ram_gb": model_ram_gb(),
        "cpu_temp_c": cpu_temperature(),
    }
