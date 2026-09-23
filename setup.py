#!/usr/bin/env python3
"""
setup.py — automated environment setup + verification for the SIH26142
Sentinel-2 super-resolution prototype.

What this does, in order:
  1. Detects your OS, architecture, Python version, and GPU (informational).
  2. Checks git is available; clones the repo if you're not already inside it.
  3. Creates an isolated virtual environment (.venv) so this doesn't touch
     your system Python.
  4. Installs requirements.txt inside that environment.
  5. Runs the full test suite (every test_*.py) to verify the install
     actually works, and prints a clear pass/fail summary.

This is standard-library only by design — it has to run BEFORE anything in
requirements.txt is installed, so it can't depend on anything from that file.

Usage:
    python3 setup.py                 # clone (if needed) + install + verify
    python3 setup.py --skip-clone    # run from inside an existing checkout
    python3 setup.py --skip-tests    # install only, skip verification run
    python3 setup.py --dir PATH      # clone/use a specific directory

See SETUP_GUIDE.md for a plain-language walkthrough of what this script does
and what to do if a step fails.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

REPO_URL = "https://github.com/cheron2000/PS2.git"
REPO_MARKER_FILE = "BUILD_AGENTS.md"  # used to detect "am I already inside the repo?"
MIN_PYTHON = (3, 9)


def header(text: str) -> None:
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, streaming output, raising on failure unless check=False
    is passed explicitly."""
    print(f"  $ {' '.join(cmd)}")
    kwargs.setdefault("check", True)
    return subprocess.run(cmd, **kwargs)


# ---------------------------------------------------------------------------
# Step 1: detect the system
# ---------------------------------------------------------------------------

def detect_system() -> dict:
    info = {
        "os": platform.system(),          # 'Linux', 'Darwin', 'Windows'
        "machine": platform.machine(),    # 'x86_64', 'arm64', 'AMD64', ...
        "python_version": sys.version_info,
        "python_exe": sys.executable,
    }

    # GPU detection is informational only -- requirements.txt installs the
    # same torch either way. This just tells you what to expect once torch
    # is installed (torch.cuda.is_available() / MPS availability).
    gpu = "none detected"
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                gpu = f"NVIDIA: {out.stdout.strip().splitlines()[0]}"
        except Exception:
            gpu = "NVIDIA GPU present (nvidia-smi query failed, driver may need attention)"
    elif info["os"] == "Darwin" and info["machine"] == "arm64":
        gpu = "Apple Silicon (torch MPS backend — not CUDA, but accelerated)"
    info["gpu"] = gpu

    header("System detected")
    print(f"  OS:              {info['os']} ({info['machine']})")
    print(f"  Python:          {'.'.join(map(str, info['python_version'][:3]))} at {info['python_exe']}")
    print(f"  GPU:             {info['gpu']}")
    return info


def check_python_version(info: dict) -> None:
    if info["python_version"][:2] < MIN_PYTHON:
        got = ".".join(map(str, info["python_version"][:2]))
        need = ".".join(map(str, MIN_PYTHON))
        print(f"\n[FAIL] Python {need}+ required, found {got}.")
        print("  Install a newer Python first, then re-run this script with it:")
        print("    python3.11 setup.py   (or whichever newer version you install)")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 2: git + repo
# ---------------------------------------------------------------------------

def check_git(info: dict) -> None:
    if shutil.which("git"):
        return
    header("git not found")
    print("  This script needs git to fetch the repository. Install it, then re-run:")
    if info["os"] == "Darwin":
        print("    macOS:   brew install git         (or install Xcode Command Line Tools)")
    elif info["os"] == "Linux":
        print("    Debian/Ubuntu:  sudo apt-get install git")
        print("    Fedora/RHEL:    sudo dnf install git")
        print("    Arch:           sudo pacman -S git")
    elif info["os"] == "Windows":
        print("    Windows: https://git-scm.com/download/win")
    else:
        print("    See https://git-scm.com/downloads")
    sys.exit(1)


def get_repo_dir(target_dir: str | None, skip_clone: bool) -> Path:
    cwd = Path.cwd()

    if skip_clone:
        if not (cwd / REPO_MARKER_FILE).exists():
            print(f"\n[FAIL] --skip-clone was given but {REPO_MARKER_FILE} isn't in the "
                  f"current directory ({cwd}). Run this from inside the repo checkout, "
                  f"or drop --skip-clone to let this script clone it for you.")
            sys.exit(1)
        return cwd

    if (cwd / REPO_MARKER_FILE).exists():
        header("Already inside the repo")
        print(f"  Found {REPO_MARKER_FILE} in {cwd} — using this directory, skipping clone.")
        return cwd

    dest = Path(target_dir) if target_dir else cwd / "PS2"
    header(f"Cloning repository into {dest}")
    if dest.exists():
        if (dest / REPO_MARKER_FILE).exists():
            print(f"  {dest} already looks like a clone of this repo — reusing it.")
            return dest
        print(f"\n[FAIL] {dest} already exists and doesn't look like this repo.")
        print(f"  Remove it, or pass --dir to choose a different location.")
        sys.exit(1)

    run(["git", "clone", REPO_URL, str(dest)])
    return dest


# ---------------------------------------------------------------------------
# Step 3: virtual environment
# ---------------------------------------------------------------------------

def venv_paths(venv_dir: Path, os_name: str) -> tuple[Path, Path]:
    """Returns (python_exe, pip_exe) inside the venv, correct per OS."""
    if os_name == "Windows":
        return venv_dir / "Scripts" / "python.exe", venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "python3", venv_dir / "bin" / "pip3"


def create_venv(repo_dir: Path, os_name: str) -> tuple[Path, Path]:
    venv_dir = repo_dir / ".venv"
    py_exe, pip_exe = venv_paths(venv_dir, os_name)

    header("Setting up virtual environment (.venv)")
    if py_exe.exists():
        print(f"  .venv already exists at {venv_dir} — reusing it.")
    else:
        print(f"  Creating at {venv_dir} ...")
        venv.EnvBuilder(with_pip=True).create(venv_dir)
    return py_exe, pip_exe


# ---------------------------------------------------------------------------
# Step 4: install
# ---------------------------------------------------------------------------

def install_requirements(repo_dir: Path, pip_exe: Path) -> None:
    header("Installing requirements.txt")
    req_file = repo_dir / "requirements.txt"
    if not req_file.exists():
        print(f"[FAIL] {req_file} not found — is {repo_dir} really the repo root?")
        sys.exit(1)
    run([str(pip_exe), "install", "--upgrade", "pip"])
    run([str(pip_exe), "install", "-r", str(req_file)])


# ---------------------------------------------------------------------------
# Step 5: verify (this is the closest thing this project has to "starting" —
# see SETUP_GUIDE.md for why there's no server/daemon to launch)
# ---------------------------------------------------------------------------

def run_tests(repo_dir: Path, py_exe: Path) -> bool:
    """Run the repository's canonical test runner, including nested tests."""
    header("Verifying the install: running the canonical test suite")
    runner = repo_dir / "run_tests.py"
    if not runner.exists():
        print(f"  [FAIL] {runner} not found.")
        return False
    result = subprocess.run([str(py_exe), str(runner)], cwd=repo_dir)
    return result.returncode == 0

# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=None, help="directory to clone into / use (default: ./PS2)")
    parser.add_argument("--skip-clone", action="store_true", help="run from the current directory, don't clone")
    parser.add_argument("--skip-tests", action="store_true", help="install only, skip the verification run")
    args = parser.parse_args()

    info = detect_system()
    check_python_version(info)
    check_git(info)

    repo_dir = get_repo_dir(args.dir, args.skip_clone)
    py_exe, pip_exe = create_venv(repo_dir, info["os"])
    install_requirements(repo_dir, pip_exe)

    if args.skip_tests:
        header("Skipping verification (--skip-tests given)")
        all_passed = None
    else:
        all_passed = run_tests(repo_dir, py_exe)

    header("Done")
    activate = (
        f"{repo_dir}\\.venv\\Scripts\\activate" if info["os"] == "Windows"
        else f"source {repo_dir}/.venv/bin/activate"
    )
    print(f"  Repository:  {repo_dir}")
    print(f"  Activate the environment before running anything else:")
    print(f"    {activate}")
    if all_passed is False:
        print("\n  [WARNING] One or more tests failed above — see SETUP_GUIDE.md's")
        print("  troubleshooting section before assuming the install is good.")
        sys.exit(1)
    elif all_passed is True:
        print("\n  All tests passed. See SETUP_GUIDE.md or README.md for next steps")
        print("  (training/inference need your own data — nothing here trains")
        print("  or predicts automatically, see 'What this script does NOT do').")


if __name__ == "__main__":
    main()
