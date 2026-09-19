#!/usr/bin/env bash
# setup.sh — macOS/Linux entry point. Checks Python exists, then hands off to
# the real cross-platform logic in setup.py. See SETUP_GUIDE.md for details.
set -e

PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Python 3.9+ not found."
    echo "  macOS:  brew install python3   (or https://www.python.org/downloads/)"
    echo "  Linux:  sudo apt-get install python3   (Debian/Ubuntu, or your distro's equivalent)"
    exit 1
fi

# If setup.py isn't next to this script yet, we were invoked standalone
# (e.g. via curl before cloning) — fetch it before handing off.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/setup.py" ]; then
    curl -sO https://raw.githubusercontent.com/cheron2000/PS2/main/setup.py
    exec "$PYTHON" setup.py "$@"
fi

exec "$PYTHON" "$SCRIPT_DIR/setup.py" "$@"
