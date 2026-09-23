#!/usr/bin/env python3
"""Run every repository test script from a clean checkout.

The project intentionally uses executable test scripts rather than requiring
pytest. This runner discovers both root-level and nested tests, executes each
in its own subprocess from the repository root, and exits non-zero if any
test fails.

Usage:
    python3 run_tests.py
    python3 run_tests.py --pattern test_scene_protocol.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="test_*.py", help="filename glob to run (default: test_*.py)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    tests = sorted(
        path for path in root.rglob(args.pattern)
        if path.is_file() and ".venv" not in path.parts and ".git" not in path.parts
    )
    if not tests:
        print(f"No tests matched {args.pattern!r}.")
        return 1

    failures = 0
    print(f"Discovered {len(tests)} test script(s).")
    for test in tests:
        rel = test.relative_to(root)
        print(f"\n=== {rel} ===")
        result = subprocess.run([sys.executable, str(test)], cwd=root)
        if result.returncode != 0:
            failures += 1
            print(f"[FAIL] {rel} (exit {result.returncode})")
        else:
            print(f"[PASS] {rel}")

    print(f"\nCompleted {len(tests)} test script(s): {len(tests) - failures} passed, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
