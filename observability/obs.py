# observability/obs.py

import time
import threading
from typing import Dict, Any

# -------------------------------------------------------
# GLOBAL METRICS STORE (thread-safe)
# -------------------------------------------------------
_METRICS_LOCK = threading.Lock()
_METRICS: Dict[str, Any] = {
    "counters": {},
    "latencies": {},
    "timings": {},
}


# -------------------------------------------------------
# COUNTERS (for calls, iterations, frames, etc.)
# -------------------------------------------------------
def inc(key: str, amount: int = 1) -> None:
    """Increment a numeric counter by amount."""
    with _METRICS_LOCK:
        _METRICS["counters"][key] = _METRICS["counters"].get(key, 0) + amount


# -------------------------------------------------------
# LATENCY STORAGE
# -------------------------------------------------------
def record_latency(key: str, value: float) -> None:
    """Record a latency value in seconds."""
    with _METRICS_LOCK:
        _METRICS["latencies"][key] = float(value)


# -------------------------------------------------------
# TIMER CONTEXT MANAGER
# -------------------------------------------------------
class timer:
    """
    Usage:
        with timer("frame_extraction"):
            ...
    """

    def __init__(self, key: str):
        self.key = key
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        end = time.perf_counter()
        duration = end - self.start  # type: ignore # seconds

        with _METRICS_LOCK:
            # store raw timing
            _METRICS["timings"][self.key] = duration
        return False  # don't suppress exceptions


# -------------------------------------------------------
# GET ALL METRICS
# -------------------------------------------------------
def get_metrics() -> Dict[str, Any]:
    """Return a deep copy of all metrics collected."""
    with _METRICS_LOCK:
        return {
            "counters": dict(_METRICS["counters"]),
            "latencies": dict(_METRICS["latencies"]),
            "timings": dict(_METRICS["timings"]),
        }
