"""Dependency-free structured JSONL telemetry for PS2 runs.

Telemetry is diagnostic only. It records lifecycle events and elapsed time,
not performance claims. Each line is a self-contained JSON object.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

class EventLogger:
    """Append structured events to a JSONL file."""
    def __init__(self, path: Optional[str | Path], *, best_effort: bool = True):
        self.path = Path(path) if path else None
        self.best_effort = best_effort
        self._started = time.monotonic()

    def emit(self, event: str, **fields: Any) -> None:
        if self.path is None:
            return
        record: Dict[str, Any] = {
            "schema_version": 1,
            "event": str(event),
            "elapsed_s": round(time.monotonic() - self._started, 6),
            **fields,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\\n")
        except Exception:
            if not self.best_effort:
                raise

    def start(self, **fields: Any) -> None:
        self.emit("run_started", **fields)

    def finish(self, **fields: Any) -> None:
        self.emit("run_finished", **fields)

    def error(self, error: BaseException, **fields: Any) -> None:
        self.emit("run_error", error_type=type(error).__name__, error=str(error), **fields)
