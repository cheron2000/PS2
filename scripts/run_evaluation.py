"""
Batch evaluation of SR model vs bicubic baseline across all test pairs.

Usage:
    python scripts/run_evaluation.py \
        --checkpoint runs/real/best.pt \
        --data-root data/sen2naip \
        --out results/
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.model import SRModel


# ── Metrics ─────────────────────────────────────────────────────────────────

def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred - target) ** 2)
    return float(20.0 * math.log10(1.0 / math.sqrt(mse + 1e-10)))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    scores = []
    for b in range(pred.shape[0]):
        p = pred[b].astype(np.float64)
        t = target[b].astype(np.float64)
        mu_p, mu_t = p.mean(), t.mean()
        sig_p, sig_t = p.std(), t.std()
        sig_pt = ((p - mu_p) * (t - mu_t)).mean()
        num = (2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)
        den = (mu_p ** 2 + mu_t ** 2 + C1) * (sig_p ** 2 + sig_t ** 2 + C2)
        scores.append(num / (den + 1e-10))
    return float(np.mean(scores))


def sam(pred: np.ndarray, target: np.ndarray) -> float:
    """Spectral Angle Mapper (radians, lower is better)."""
    p = pred.reshape(pred.shape[0], -1).T
    t = target.reshape(target.shape[0], -1).T
    dot = (p * t).sum(axis=1)
    norm = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1) + 1e-10
    cos = np.clip(dot / norm, -1, 1)
    return float(np.arccos(cos).mean())


def ndvi_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute NDVI error (bands: B,G,R,NIR → NIR=3, R=2)."""
    def _ndvi(arr):
        nir = arr[3] if arr.shape[0] > 3 else arr[-1]
        red = arr[2] if arr.shape[0] > 2 else arr[0]
        return (nir - red) / (nir + red + 1e-6)
    return float(np.abs(_ndvi(pred) - _ndvi(target)).mean())


def bicubic_upsample(lr: np.ndarray, scale: int = 4) -> np.ndarray:
    from scipy.ndimage import zoom
    return np.clip(zoom(lr, (1, scale, scale), order=3), 0.0, 1.0).astype(np.float32)


# ── Model runner ─────────────────────────────────────────────────────────────

def load_model(checkpoint: str, device: str) -> SRModel:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = SRModel(**cfg).to(device)
    state_key = "model_state" if "model_state" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model


@torch.no_grad()
def infer(model: SRModel, lr_arr: np.ndarray, device: str) -> np.ndarray:
    t = torch.from_numpy(lr_arr).unsqueeze(0).to(device)
    out = model(t)
    mean = out[0] if isinstance(out, (tuple, list)) else out
    return mean.squeeze(0).cpu().numpy().clip(0, 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="data/sen2naip")
    ap.add_argument("--out", default="results")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="test",
                    help="test uses last 20pct of pairs; all uses everything")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lr_dir = Path(args.data_root) / "lr"
    hr_dir = Path(args.data_root) / "hr"
    lr_files = sorted(lr_dir.glob("*.npy"))
    hr_files = sorted(hr_dir.glob("*.npy"))

    assert len(lr_files) == len(hr_files), "LR/HR count mismatch"
    n = len(lr_files)

    if args.split == "test":
        split_start = int(n * 0.8)
        lr_files = lr_files[split_start:]
        hr_files = hr_files[split_start:]

    print(f"Evaluating {len(lr_files)} pairs from {args.data_root}...")

    model = load_model(args.checkpoint, args.device)

    results = []
    bic_psnrs, bic_ssims, bic_sams, bic_ndvis = [], [], [], []
    sr_psnrs,  sr_ssims,  sr_sams,  sr_ndvis  = [], [], [], []

    for lr_path, hr_path in zip(lr_files, hr_files):
        lr = np.load(lr_path).astype(np.float32)
        hr = np.load(hr_path).astype(np.float32)
        if lr.ndim == 2: lr = lr[None]
        if hr.ndim == 2: hr = hr[None]

        bic = bicubic_upsample(lr, args.scale)
        sr  = infer(model, lr, args.device)

        # Align shapes
        h = min(bic.shape[1], hr.shape[1], sr.shape[1])
        w = min(bic.shape[2], hr.shape[2], sr.shape[2])
        bic = bic[:, :h, :w]
        sr  = sr[:, :h, :w]
        hr  = hr[:, :h, :w]

        r = {
            "name": lr_path.stem,
            "bicubic_psnr": psnr(bic, hr),
            "bicubic_ssim": ssim(bic, hr),
            "bicubic_sam":  sam(bic, hr),
            "bicubic_ndvi_mae": ndvi_mae(bic, hr),
            "sr_psnr":     psnr(sr, hr),
            "sr_ssim":     ssim(sr, hr),
            "sr_sam":      sam(sr, hr),
            "sr_ndvi_mae": ndvi_mae(sr, hr),
        }
        results.append(r)

        bic_psnrs.append(r["bicubic_psnr"])
        bic_ssims.append(r["bicubic_ssim"])
        bic_sams.append(r["bicubic_sam"])
        bic_ndvis.append(r["bicubic_ndvi_mae"])
        sr_psnrs.append(r["sr_psnr"])
        sr_ssims.append(r["sr_ssim"])
        sr_sams.append(r["sr_sam"])
        sr_ndvis.append(r["sr_ndvi_mae"])

    summary = {
        "n_pairs": len(results),
        "checkpoint": str(args.checkpoint),
        "bicubic": {
            "psnr_mean": float(np.mean(bic_psnrs)),
            "psnr_std":  float(np.std(bic_psnrs)),
            "ssim_mean": float(np.mean(bic_ssims)),
            "ssim_std":  float(np.std(bic_ssims)),
            "sam_mean":  float(np.mean(bic_sams)),
            "ndvi_mae_mean": float(np.mean(bic_ndvis)),
        },
        "sr_model": {
            "psnr_mean": float(np.mean(sr_psnrs)),
            "psnr_std":  float(np.std(sr_psnrs)),
            "ssim_mean": float(np.mean(sr_ssims)),
            "ssim_std":  float(np.std(sr_ssims)),
            "sam_mean":  float(np.mean(sr_sams)),
            "ndvi_mae_mean": float(np.mean(sr_ndvis)),
        },
        "delta": {
            "psnr_db": float(np.mean(sr_psnrs) - np.mean(bic_psnrs)),
            "ssim":    float(np.mean(sr_ssims) - np.mean(bic_ssims)),
            "sam_rad": float(np.mean(sr_sams)  - np.mean(bic_sams)),
            "ndvi_mae_pct_reduction": float(
                100 * (np.mean(bic_ndvis) - np.mean(sr_ndvis)) / (np.mean(bic_ndvis) + 1e-8)
            ),
        },
        "per_pair": results,
    }

    # Save JSON
    json_path = out_dir / "metrics_report.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save markdown table
    md_path = out_dir / "metrics_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Super-Resolution Evaluation Results\n\n")
        f.write(f"**Pairs evaluated:** {summary['n_pairs']}  \n")
        f.write(f"**Checkpoint:** `{summary['checkpoint']}`\n\n")
        f.write("## Summary\n\n")
        f.write("| Metric | Bicubic | SR Model | Delta |\n")
        f.write("|--------|---------|----------|---|\n")
        d = summary["delta"]
        b = summary["bicubic"]
        s = summary["sr_model"]
        f.write(f"| PSNR (dB) | {b['psnr_mean']:.2f} ± {b['psnr_std']:.2f} | **{s['psnr_mean']:.2f} ± {s['psnr_std']:.2f}** | **{d['psnr_db']:+.2f}** |\n")
        f.write(f"| SSIM | {b['ssim_mean']:.4f} ± {b['ssim_std']:.4f} | **{s['ssim_mean']:.4f} ± {s['ssim_std']:.4f}** | **{d['ssim']:+.4f}** |\n")
        f.write(f"| SAM (rad) | {b['sam_mean']:.4f} | **{s['sam_mean']:.4f}** | {d['sam_rad']:+.4f} |\n")
        f.write(f"| NDVI MAE | {b['ndvi_mae_mean']:.4f} | **{s['ndvi_mae_mean']:.4f}** | {d['ndvi_mae_pct_reduction']:+.1f}% reduction |\n")
        f.write("\n## Per-Pair Results\n\n")
        f.write("| Pair | Bic PSNR | SR PSNR | Bic SSIM | SR SSIM |\n")
        f.write("|------|----------|---------|----------|--------|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['bicubic_psnr']:.2f} | {r['sr_psnr']:.2f} | {r['bicubic_ssim']:.4f} | {r['sr_ssim']:.4f} |\n")

    # Console summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Bicubic':>12} {'SR Model':>12} {'Delta':>10}")
    print("-" * 60)
    print(f"{'PSNR (dB)':<20} {b['psnr_mean']:>12.2f} {s['psnr_mean']:>12.2f} {d['psnr_db']:>+10.2f}")
    print(f"{'SSIM':<20} {b['ssim_mean']:>12.4f} {s['ssim_mean']:>12.4f} {d['ssim']:>+10.4f}")
    print(f"{'SAM (rad)':<20} {b['sam_mean']:>12.4f} {s['sam_mean']:>12.4f} {d['sam_rad']:>+10.4f}")
    print(f"{'NDVI MAE':<20} {b['ndvi_mae_mean']:>12.4f} {s['ndvi_mae_mean']:>12.4f} {d['ndvi_mae_pct_reduction']:>+9.1f}%")
    print("=" * 60)
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
