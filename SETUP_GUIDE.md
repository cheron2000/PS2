# SETUP_GUIDE.md — Getting This Running on Your Machine

This is the plain-language walkthrough. If you're comfortable with the command
line and just want the commands, see `README.md` instead — this guide is for
"I want to understand what's happening at each step, and what to do if
something goes wrong."

---

## What you'll end up with

A working copy of this project on your computer, with everything it needs
installed in its own isolated space (so it can't break anything else on your
machine), and a test run confirming it actually works.

**One thing to set expectations on: there's no app or server to "launch" here.**
This is a machine learning prototype, not a website — there's no button to
press that shows you a running system. What `setup.py` does at the end is run
the project's full test suite as its way of confirming "yes, this actually
works on your machine" — that's the closest thing to "starting the system"
that exists at this stage. Actually using the model (training it on real
data, or running a trained model on a real image) is a separate, later step
covered in `README.md`, because it needs data you'll need to get yourself
(see "What this doesn't do" at the bottom).

---

## Before you start: one thing you need

**Python, version 3.9 or newer.** Everything else (git, the virtual
environment, all the packages) either gets checked automatically or installed
by the script.

Check if you already have it:
```bash
python3 --version
```
If that prints `Python 3.9` or higher, you're set. If it says "command not
found" or shows an older version:
- **Windows:** download from [python.org/downloads](https://www.python.org/downloads/) — during install, tick "Add Python to PATH"
- **macOS:** `brew install python3` (or from [python.org](https://www.python.org/downloads/))
- **Linux:** `sudo apt-get install python3` (Debian/Ubuntu) or your distro's equivalent

---

## Running it

Open a terminal (Command Prompt or PowerShell on Windows, Terminal on
macOS/Linux), navigate to the folder where you want this project to live, and
run one line:

**macOS / Linux:**
```bash
curl -O https://raw.githubusercontent.com/cheron2000/PS2/main/setup.sh && bash setup.sh
```

**Windows (PowerShell):**
```powershell
curl.exe -O https://raw.githubusercontent.com/cheron2000/PS2/main/setup.bat && .\setup.bat
```

**Or, if you'd rather grab the whole project first and run it from inside:**
```bash
git clone https://github.com/cheron2000/PS2.git
cd PS2
python3 setup.py --skip-clone
```

---

## What happens, step by step

1. **"System detected"** — it prints your operating system, Python version,
   and whether it found a GPU. This is just informational; nothing here
   changes what gets installed. (If you have an NVIDIA GPU and it's not
   detected, that usually means the NVIDIA driver isn't installed — not
   something this script can fix for you.)

2. **Checks for git** — the tool used to download the project's code. If it's
   missing, the script tells you exactly what command to run for your OS,
   then stops so you can install it and try again.

3. **Downloads the project** (unless you already have it) into a folder
   called `PS2`.

4. **Creates a `.venv` folder** — this is a self-contained space for this
   project's Python packages, completely separate from anything else on your
   computer. Deleting the `PS2` folder later removes everything cleanly; it
   won't leave anything behind on your system.

5. **Installs the required packages** (`torch`, `numpy`, `rasterio`) inside
   that `.venv`. This is the slowest step — `torch` alone is several hundred
   megabytes — and needs a decent internet connection and a few GB of free
   disk space.

6. **Runs every test file** and prints a pass/fail summary. If everything
   shows `PASS`, your setup is confirmed working.

---

## Troubleshooting

**"No space left on device" during install.**
`torch` (with GPU support bundled in) needs several gigabytes of free disk
space to download and install. Free up space and re-run
`python3 setup.py --skip-clone` from inside the `PS2` folder — it'll reuse
what's already downloaded and pick up where it left off.

**A test fails.**
Re-run just that one test for more detail:
```bash
# macOS/Linux
source PS2/.venv/bin/activate
# Windows
PS2\.venv\Scripts\activate

python3 test_<name>.py
```
Read the error message — every test in this project prints what it expected
vs. what it got. If you're not sure what it means, share the exact output
with whoever's maintaining this project (or with an AI agent following
`BUILD_AGENTS.md` — fixing a broken test is a reasonable task to hand one).

**"git not found" even after installing it.**
Close and reopen your terminal — installers usually need a fresh terminal
session to update your PATH.

**It's very slow.**
The install step downloads several hundred MB; this depends entirely on your
internet connection, not the script.

---

## What this script does NOT do

- It does not download any real satellite imagery (Sentinel-2, NAIP,
  Cartosat, or WorldCover) — none of that data ships with this repo or gets
  fetched automatically. Every test uses data the code generates itself.
- It does not train a model or run predictions — see `README.md`'s "Running
  training" / "Running inference" sections once you have real data to point
  it at.
- It does not set up access to ISRO's Bhoonidhi portal or any other data
  source — that's a manual, human step (account registration, an end-user
  agreement, and in some cases payment), not something any script or AI
  agent can do on your behalf.

---

## Re-running later

Once installed, you don't need `setup.py` again for normal use — just
activate the environment each time:
```bash
# macOS/Linux
source PS2/.venv/bin/activate
# Windows
PS2\.venv\Scripts\activate
```
Re-run `python3 setup.py --skip-clone` only if you want to reinstall or
re-verify (e.g. after pulling new code changes).
