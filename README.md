# SIH26142 — Uncertainty-Aware Sentinel-2 Super-Resolution

Deep learning super-resolution for Sentinel-2 imagery (10m → 2.5m), with a
per-pixel uncertainty output and validation against real Indian ground truth
(ISRO Cartosat-2S/3). Built for SIH26142 (NTRO — Deep Learning Based Super
Resolution Mapping from Medium Resolution Satellite Imageries).

- **What and why:** see `SUMMARY.md` (short) or `solution-draft.md` (full research, v11, 8 rounds of multi-agent review).
- **How this codebase was built:** `BUILD_AGENTS.md` (protocol) and `build-status.md` (full task-by-task log) — a dynamic set of AI agents built this collaboratively; that log is the real build history and is worth reading if something here seems under-explained.
- **This file:** how to actually set up and run what's here.

---

## Setup

```bash
pip install -r requirements.txt
```

`rasterio` (georeferenced I/O — GeoTIFF read/write, CRS/reprojection) is a real
dependency, not optional, used by `src/infer.py` and `src/datasets/worldcover.py`.
`torch` is required for the model, losses, and training/inference; most of the
data-side modules (`src/preprocessing.py`, `src/metrics.py`,
`src/eval_downstream.py`, `src/datasets/sen2naip.py`,
`src/datasets/sen2venus.py`, `src/datasets/cartosat_pairing.py`) also work with
just `numpy` if you only need those pieces.

---

## Repository structure

```
src/
  model.py                    Core architecture: CNN attention + transformer fusion
                               + sub-pixel upsampling + uncertainty head (T1)
  losses.py                   Reconstruction (L1/L2) + heteroscedastic NLL loss (T2)
  preprocessing.py            Sentinel-2 L2A: SCL cloud masking, normalization, tiling (T4)
  metrics.py                  PSNR, SSIM, spectral-angle error, alignment check (T7)
  eval_downstream.py          Downstream-task (land-cover) comparison: SR vs. bicubic vs. HR (T8)
  train.py                    Training loop: wires model + loss + data together (T9)
  infer.py                    Inference CLI: checkpoint -> SR image + uncertainty map,
                               georeferenced GeoTIFF output (T11)
  datasets/
    sen2naip.py                SEN2NAIP loader + NCC pair co-registration/QC (T5)
    sen2venus.py                SEN2Vénus loader, secondary 5m scale (T6)
    cartosat_pairing.py         Cartosat<->Sentinel-2 pairing for India validation (T10)
    worldcover.py               Real ESA WorldCover land-cover loader (T13)

test_*.py / tests/test_*.py  Executable regression scripts. `run_tests.py` discovers
                               both root-level and nested tests without pytest.

tasks.md                      Full task board and build history
```

---

## Running tests

Tests are executable Python scripts, not pytest-only. The canonical command
discovers both root-level and nested `tests/` scripts and returns a non-zero exit
code if any test fails:

```bash
python3 run_tests.py
```

To run one specific test directly:

```bash
python3 test_preprocessing.py
python3 tests/test_scene_protocol.py
```

The setup script uses the same `run_tests.py` runner, so setup verification and
manual verification exercise the same test inventory. This avoids the previous
failure mode where a nested test could be silently omitted by a root-only glob.

All of these generate their own synthetic data — **none require real Sentinel-2,
NAIP, Cartosat, or WorldCover downloads**. That's deliberate: every data-access
task in this build hit the same wall (no internet access to the actual data
portals from inside an agent's sandbox — see the "Known gaps" section below),
so every loader was built and verified against synthetic data matching the
real format's documented shape, with the real-download step left as an
explicit, flagged gap rather than silently faked.

---

## Running training (on your own data)

```bash
python -m src.train --data-root data/sen2naip --epochs 5 --checkpoint-dir runs/demo
```

`--data-root` must follow the layout `src/datasets/sen2naip.py` expects:
```
data/sen2naip/
  lr/<id>.npy   # (C, H, W) low-res Sentinel-2 patch
  hr/<id>.npy   # (C, H*scale, W*scale) high-res reference patch, matching <id>
```
This is **not** the official SEN2NAIP HuggingFace release format (that's a TACO
archive) — see "Known gaps" below for what's needed to bridge the two.

## Running inference

```bash
python -m src.infer --checkpoint runs/demo/best.pt \
    --input tile.tif --output sr.tif --uncertainty-output uncertainty.tif
```

Accepts a GeoTIFF (via rasterio, CRS/transform preserved in the output) or a
`.npy` array with a `--metadata-json` sidecar. Outputs the super-resolved
image and a separate per-band variance map (`exp(log_variance)` from the
model's uncertainty head).

---

## Known gaps (read before assuming something works end-to-end with real data)

Nothing here is hidden — each is flagged in its own file's header and in
`build-status.md`'s log, this just collects them in one place:

- **SEN2NAIP / SEN2Vénus / Cartosat**: loaders target a documented plain-`.npy`
  convention, not the official release formats (TACO archives / undocumented
  portal schema). A converter from the real format to this convention is not
  built. **ESA WorldCover is the one exception** — its format is well-documented
  and stable, so `worldcover.py` reads the real Cloud-Optimized GeoTIFF format
  directly.
- **No agent in this build had network access to the actual data portals**
  (Copernicus, HuggingFace, Bhoonidhi, AWS S3 for WorldCover) — every test
  uses synthetic data shaped like the real thing, not real downloads.
- **Cartosat via Bhoonidhi** additionally needs a real NSIL account, a placed
  order, and — per `solution-draft.md`'s research — normal-priority turnaround
  time that ISRO/NRSC doesn't publish; budget for asking directly, not just
  downloading.
- **GPU-scale training has not been run.** All execution verification in this
  build was CPU-only (see `build-status.md` for which agent verified which
  module and how) — training-time and memory behavior at real batch sizes on
  a GPU is unverified.
- **The windowed transformer block in `model.py`** uses fixed (non-shifted)
  windows, a simplification flagged in T1's own notes — worth revisiting if
  window-boundary artifacts show up once real training starts.
