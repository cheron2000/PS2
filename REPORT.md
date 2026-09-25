# Project Achievement Report — SIH26142
## Deep Learning Based Super Resolution Mapping from Medium Resolution Satellite Imageries
**Team:** cheron2000 | **Date:** 2026-09-25

---

## 1. Problem Statement Summary

**SIH26142 (NTRO):** Build a robust AI-based super-resolution framework that transforms 10m Sentinel-2 satellite imagery into sharper, information-rich products (<4m GSD) while preserving geospatial and spectral consistency. The solution must include preprocessing, model training with paired datasets, accuracy assessment, validation against high-resolution references, and explicit uncertainty management.

---

## 2. What We Built

We developed an **end-to-end, uncertainty-aware super-resolution framework** that takes 10m Sentinel-2 imagery and produces 2.5m enhanced output (4× upscaling) with per-pixel confidence maps. The system is fully functional — from data ingestion to interactive web-based inference.

### Core Architecture
| Component | Description |
|-----------|-------------|
| **Model** (`src/model.py`) | CNN feature stem → Multi-Head Attention Network (MHAN) for local high-frequency features → Cross-spatial transformer fusion → Sub-pixel reconstruction head → Heteroscedastic uncertainty head (mean + variance) |
| **Loss Function** (`src/losses.py`) | Combined reconstruction loss + uncertainty-aware NLL loss, enabling the model to learn *where* its predictions are uncertain |
| **Training Pipeline** (`src/train.py`) | Full training loop with train/val split, reproducible seeding, checkpoint saving (best + latest), resume support, and JSONL telemetry |
| **Inference Engine** (`src/infer.py`) | Supports single-pass and tiled inference for large scenes, geospatial I/O (GeoTIFF with CRS/transform preservation), input validation, and atomic file writes |
| **Evaluation** (`src/metrics.py`, `src/benchmark.py`) | PSNR, SSIM, spectral angle error, hallucination/omission metrics, uncertainty calibration |
| **Downstream Task Eval** (`src/eval_downstream.py`) | Land-cover classification comparison (SR vs. bicubic vs. HR ground truth) using ESA WorldCover labels |
| **Interactive Prototype** (`app.py`) | Streamlit web application for live inference and visualization |

### Module Count
- **11 core source modules** in `src/`
- **3 dataset loaders** (SEN2NAIP, SEN2Vénus, TACO converter)
- **2 utility scripts** in `scripts/`
- **1 interactive web UI** (`app.py`)
- **Comprehensive test suite** (`run_tests.py`, `tests/`)

---

## 3. Key Achievements

### ✅ 3.1 Real-World Dataset Integration
- Successfully downloaded **100 real Sentinel-2/NAIP paired images** (50 LR at 130×130, 50 HR at 520×520) from the SEN2NAIP cross-sensor dataset
- Each pair is a genuine cross-sensor match: Sentinel-2 (10m) ↔ NAIP (2.5m), not synthetic degradation
- Downloaded ESA WorldCover (10m land-cover map) for downstream task validation

### ✅ 3.2 Model Training Completed
Trained the model for 5 epochs on the downloaded dataset with a 80/20 train/val split:

| Epoch | Train Loss | Val Loss | Train Recon | Val Recon |
|-------|-----------|----------|-------------|-----------|
| 1 | 0.4156 | -0.1298 | 0.4391 | 0.0557 |
| 2 | -0.1199 | -0.1989 | 0.0569 | 0.0294 |
| 3 | -0.1996 | -0.2654 | 0.0361 | 0.0256 |
| 4 | -0.2017 | -0.2423 | 0.0343 | 0.0218 |
| 5 | -0.2052 | -0.2335 | 0.0384 | 0.0221 |

- Validation reconstruction error dropped from **0.0557 → 0.0221** (60% improvement)
- Best checkpoint saved at epoch 3 (lowest val_loss: -0.2654)
- The negative total loss reflects the uncertainty NLL component — the model is simultaneously learning to predict accurate pixel values *and* calibrate its per-pixel confidence

### ✅ 3.3 Interactive Web Prototype
Built a **Streamlit-based web application** (`app.py`) with:
- Image sample selection from the test dataset via dropdown
- Side-by-side comparison: Low-Res Input → High-Res Ground Truth
- One-click inference execution producing:
  - **Super-resolved output** (2.5m from 10m input)
  - **Uncertainty heatmap** (per-pixel variance, magma colormap)
- Model checkpoint auto-loading with status indicator

### ✅ 3.4 Security & Compatibility Fixes
- Fixed PyTorch 2.6 `weights_only=True` checkpoint loading compatibility
- Maintained secure loading practices with documented audit exceptions
- Fixed Streamlit API deprecation warnings (`use_container_width` → `width`)

### ✅ 3.5 Production-Grade Engineering
- **Atomic file writes** — crash-safe checkpoint and output saving via temp-file + `os.replace()`
- **Input validation** — preflight checks reject malformed/oversized/non-finite input before GPU allocation
- **Tiled inference** — bounded overlapping tiles for processing arbitrarily large scenes within fixed VRAM
- **Geospatial I/O** — full CRS, affine transform, and band metadata preservation in GeoTIFF outputs
- **Structured telemetry** — JSONL event logging for training and inference lifecycle tracking
- **Reproducibility** — seeded RNG state (Python, NumPy, PyTorch CPU/CUDA) saved in checkpoints for exact resume

---

## 4. Research Deliberation Process

The solution was developed through an **8-round multi-model research deliberation** (documented in `AGENTS.md`, `status.md`, and `/log/`). Three independent AI agents reviewed and critiqued each draft version:

| Agent | Model | Rounds Active |
|-------|-------|---------------|
| `claude` | Claude (Anthropic) | Rounds 1–8 |
| `sonnet5` | Claude Sonnet 5 | Rounds 1, 2, 4 |
| `astrasr` | Custom agent | Multiple rounds |

Key research contributions:
- Identified **DiffFuSR** and **SEN2SR/SEN2SRLite** as closest prior art
- Discovered a competing team (DrishtiSR) already executing the compute-tier approach
- Converged on **India-geography validation via Cartosat-2S/3** as the unique differentiator
- Added downstream-task evaluation (land-cover IoU) based on GeoSR-Bench findings

---

## 5. Repository Structure

```
PS2/
├── app.py                          # Streamlit prototype UI
├── visualize_pair.py               # Quick LR/HR comparison script
├── problem-statement.md            # Official SIH26142 text
├── solution-draft.md               # Research-backed solution (v11)
├── AGENTS.md                       # Multi-model deliberation protocol
├── src/
│   ├── model.py                    # SR model (CNN + attention + uncertainty)
│   ├── train.py                    # Training loop with val split + checkpointing
│   ├── infer.py                    # Inference engine (single + tiled)
│   ├── losses.py                   # Reconstruction + NLL uncertainty loss
│   ├── metrics.py                  # PSNR, SSIM, SAM, hallucination metrics
│   ├── benchmark.py                # Benchmark runner
│   ├── eval_downstream.py          # Downstream task evaluation
│   ├── preprocessing.py            # L2A ingestion, cloud masking, normalization
│   ├── evaluation_contract.py      # Machine-readable eval policy
│   ├── scene_protocol.py           # Scene-disjoint train/test splitting
│   ├── telemetry.py                # JSONL structured telemetry
│   └── datasets/
│       ├── sen2naip.py             # SEN2NAIP dataset loader
│       ├── sen2venus.py            # SEN2Vénus dataset loader
│       ├── convert_taco.py         # TACO format conversion
│       └── download_worldcover.py  # ESA WorldCover downloader
├── configs/
│   └── evaluation_contract.json    # Frozen evaluation policy
├── scripts/
│   ├── run_evaluation.py           # Evaluation runner
│   └── visualize_results.py        # Results visualization
├── data/sen2naip/                  # 100 real LR/HR pairs (local only)
├── runs/demo/                      # Trained checkpoint (local only)
└── tests/                          # Test suite
```

---

## 6. How to Run

```bash
# Train the model
python -m src.train --data-root data/sen2naip --epochs 5 --checkpoint-dir runs/demo

# Run inference on a single tile
python -m src.infer --checkpoint runs/demo/best.pt --input tile.npy --output sr.npy

# Launch the interactive prototype
streamlit run app.py
# → Open http://localhost:8501
```

---

## 7. Next Steps

| Priority | Task | Status |
|----------|------|--------|
| **P0** | Secure Cartosat-2S/3 data via ISRO Bhoonidhi portal for India-geography validation | Pending |
| **P1** | Train for more epochs (50–100) on full SEN2NAIP dataset for production-quality results | Ready |
| **P2** | Run OpenSR-test benchmark suite for standardized comparison | Ready |
| **P3** | Add downstream land-cover IoU evaluation with ESA WorldCover labels | Code ready |
| **P4** | Deploy as a cloud service for judge demonstration | Planned |

---

*Generated on 2026-09-25. All code is committed to [github.com/cheron2000/PS2](https://github.com/cheron2000/PS2).*
