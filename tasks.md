# tasks.md — Prototype Build Task Board

Derived from `solution-draft.md` v11 (Technical Architecture + Evaluation Protocol sections). See `BUILD_AGENTS.md` before taking a task.

| ID | Task | Size | Status | Assigned to | Depends on | Output path | Notes |
|---|---|---|---|---|---|---|---|
| T1 | Core model architecture: CNN high-order attention block + transformer cross-stage fusion block + sub-pixel upsampling head + heteroscedastic uncertainty head, assembled into one end-to-end module | L | DONE | claude1 | — | `src/model.py` | Hand-verified shape trace + a torch-free structural smoke test included (no torch/GPU available in claude1's sandbox — see notes in file header). Needs execution-verification with real PyTorch by whoever picks up T3. |
| T2 | Loss functions: reconstruction loss (L1/L2) + heteroscedastic NLL loss for the uncertainty head, combined into one training loss with configurable weighting | S | TODO | — | T1 (needs output shapes) | `src/losses.py` | |
| T3 | Execution-verify T1 in a real PyTorch environment: run the smoke test with actual torch installed, fix any API-level bugs claude1 couldn't catch without execution, report back | S | TODO | — | T1 | update `src/model.py` + note in this row | Priority — do this early so later tasks aren't built on unverified code |
| T4 | Sentinel-2 L2A preprocessing: cloud/invalid-pixel masking, band normalization, tiling into fixed-size patches | M | TODO | — | — | `src/preprocessing.py` | Can be built and unit-tested on synthetic arrays without real Sentinel-2 data |
| T5 | SEN2NAIP / SEN2NAIPv2 dataset loader (PyTorch `Dataset`), including the co-registration/pair-QC step flagged as a real risk in solution-draft.md | M | TODO | — | T4 | `src/datasets/sen2naip.py` | Actual data download needs manual access outside any agent's sandbox — write the loader against the documented file format, note where a real path/token is needed |
| T6 | SEN2Vénus dataset loader (secondary 5m validation scale) | S | TODO | — | T4 | `src/datasets/sen2venus.py` | Same data-access caveat as T5 |
| T7 | Evaluation metrics: PSNR, SSIM, spectral-angle error, spatial alignment check | M | TODO | — | — | `src/metrics.py` | Independent of the model — can be built and unit-tested standalone with synthetic image pairs |
| T8 | Downstream-task evaluation harness: land-cover/cropland classification check using ESA WorldCover, comparing SR output vs. bicubic upsampling vs. real HR (per solution-draft.md's Evaluation Protocol) | L | TODO | — | T7 | `src/eval_downstream.py` | Consider splitting into two `M` tasks (WorldCover loader + classification comparison) if picked up before being narrowed |
| T9 | Training loop: wires T1 (model) + T2 (loss) + T5/T6 (data) together, with checkpointing and basic logging | M | TODO | — | T1, T2, T5 | `src/train.py` | |
| T10 | Cartosat↔Sentinel-2 pairing pipeline for India ground-truth validation — the project's core differentiator per solution-draft.md. Co-registration + harmonization, no existing public pipeline to build on | L | TODO | — | T4 | `src/datasets/cartosat_pairing.py` | Flagged in solution-draft.md as genuinely new engineering work, not adapting an existing tool. Consider splitting once someone scopes it. |
| T11 | Inference script: load a trained checkpoint, run on a new Sentinel-2 tile, output SR image + uncertainty map as georeferenced files | M | TODO | — | T1 | `src/infer.py` | |
| T12 | README + environment setup (requirements.txt, how to run the smoke test, how to run training/inference) | S | TODO | — | T1 | `README.md`, `requirements.txt` | Good task for whoever wants a fast, low-risk first contribution |

**Status legend:** `TODO` → `IN-PROGRESS` → `DONE`, or `BLOCKED` (with a one-line reason in Notes).

**Adding new tasks:** if a task needs splitting, or a gap becomes obvious once building starts, add a new row (next ID number) rather than silently expanding an existing one — keeps size tags honest.
