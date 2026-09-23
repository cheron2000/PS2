"""Tests for dependency-free structured telemetry."""
import json
import tempfile
from pathlib import Path
from src.telemetry import EventLogger

def test_jsonl_lifecycle_and_best_effort():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run.jsonl"
        logger = EventLogger(path)
        logger.start(mode="test", seed=3)
        logger.emit("tile_completed", tile=1)
        logger.finish(status="ok")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["event"] for row in rows] == ["run_started", "tile_completed", "run_finished"]
        assert all(row["schema_version"] == 1 for row in rows)
        assert all(row["elapsed_s"] >= 0 for row in rows)
        assert rows[0]["seed"] == 3

def test_best_effort_does_not_abort_on_io_failure():
    logger = EventLogger("/proc/does-not-exist/ps2.jsonl", best_effort=True)
    logger.emit("run_started")

if __name__ == "__main__":
    test_jsonl_lifecycle_and_best_effort()
    test_best_effort_does_not_abort_on_io_failure()
    print("telemetry tests passed")
