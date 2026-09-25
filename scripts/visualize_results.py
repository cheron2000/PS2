"""
Visualize super-resolution results: 3-panel comparison plot.

Usage:
    python scripts/visualize_results.py \
        --checkpoint runs/real/best.pt \
        --lr-npy data/sen2naip/lr/real_synth_050.npy \
        --hr-npy data/sen2naip/hr/real_synth_050.npy \
        --out results/comparison.png
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from src.model import SRModel


# ── Metrics ─────────────────────────────────────────────────────────────────

def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def ssim(pred: np.ndarray, target: np.ndarray, C1=0.01**2, C2=0.03**2) -> float:
    """Per-image SSIM averaged across bands."""
    scores = []
    for b in range(pred.shape[0]):
        p, t = pred[b].astype(np.float64), target[b].astype(np.float64)
        mu_p, mu_t = p.mean(), t.mean()
        sig_p = p.std()
        sig_t = t.std()
        sig_pt = ((p - mu_p) * (t - mu_t)).mean()
        num = (2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)
        den = (mu_p**2 + mu_t**2 + C1) * (sig_p**2 + sig_t**2 + C2)
        scores.append(num / den if den != 0 else 0.0)
    return float(np.mean(scores))


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(checkpoint: str, lr_arr: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Load model from checkpoint and run SR. Returns (sr_mean, sr_var)."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = SRModel(**cfg).to(device)
    state_key = "model_state" if "model_state" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    tensor = torch.from_numpy(lr_arr).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    if isinstance(out, (tuple, list)):
        mean, log_var = out
    else:
        mean, log_var = out, torch.zeros_like(out)

    sr_mean = mean.squeeze(0).cpu().numpy().clip(0, 1)
    sr_var  = np.exp(log_var.squeeze(0).cpu().numpy())
    return sr_mean, sr_var


# ── Bicubic baseline ─────────────────────────────────────────────────────────

def bicubic_upsample(lr: np.ndarray, scale: int = 4) -> np.ndarray:
    from scipy.ndimage import zoom
    return np.clip(zoom(lr, (1, scale, scale), order=3), 0.0, 1.0).astype(np.float32)


# ── Plotting ─────────────────────────────────────────────────────────────────

def to_rgb(arr: np.ndarray, band_order=(2, 1, 0)) -> np.ndarray:
    """Convert (C, H, W) float32 -> (H, W, 3) uint8 RGB for display."""
    # Use R, G, B bands (bands 2, 1, 0 for BGRNIR layout)
    bands = [arr[b] for b in band_order if b < arr.shape[0]]
    rgb = np.stack(bands[:3], axis=-1)
    # Percentile stretch for display
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def make_comparison_figure(
    lr: np.ndarray,
    bicubic: np.ndarray,
    sr_mean: np.ndarray,
    sr_var: np.ndarray,
    hr: np.ndarray,
    psnr_bicubic: float,
    ssim_bicubic: float,
    psnr_sr: float,
    ssim_sr: float,
    save_path: str,
) -> None:
    fig = plt.figure(figsize=(20, 10), facecolor="#0d1117")
    fig.suptitle(
        "Satellite Super-Resolution: PS2 SRModel vs Bicubic Baseline",
        fontsize=16, fontweight="bold", color="white", y=0.98,
    )

    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.05)

    panels = [
        (0, 0, to_rgb(lr, (2, 1, 0)),      f"LR Input\n(Sentinel-2 10m, 64×64)",     "#6e7681"),
        (0, 1, to_rgb(bicubic, (2, 1, 0)), f"Bicubic Upsample (×4)\nPSNR {psnr_bicubic:.2f} dB | SSIM {ssim_bicubic:.4f}", "#f0883e"),
        (0, 2, to_rgb(sr_mean, (2, 1, 0)), f"SR Output (ours)\nPSNR {psnr_sr:.2f} dB | SSIM {ssim_sr:.4f}", "#3fb950"),
        (0, 3, to_rgb(hr, (2, 1, 0)),      f"HR Ground Truth\n(NAIP 2.5m, 256×256)",  "#58a6ff"),
    ]

    for col, (row, c, img, title, color) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, color=color, fontsize=9, pad=6, fontweight="bold")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    # 5th panel: uncertainty map
    ax_unc = fig.add_subplot(gs[0, 4])
    unc_disp = sr_var.mean(axis=0)
    im = ax_unc.imshow(unc_disp, cmap="magma", interpolation="nearest")
    ax_unc.set_title("Uncertainty Map\n(mean predictive variance)", color="#d2a8ff", fontsize=9, pad=6, fontweight="bold")
    ax_unc.axis("off")
    plt.colorbar(im, ax=ax_unc, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")

    # Bottom row: NDVI comparison
    def ndvi(arr):
        nir = arr[3] if arr.shape[0] > 3 else arr[0]
        red = arr[2] if arr.shape[0] > 2 else arr[0]
        denom = nir + red + 1e-6
        return (nir - red) / denom

    ndvi_bicubic = ndvi(bicubic)
    ndvi_sr      = ndvi(sr_mean)
    ndvi_hr      = ndvi(hr)
    err_bicubic  = np.abs(ndvi_bicubic - ndvi_hr)
    err_sr       = np.abs(ndvi_sr - ndvi_hr)

    bottom_panels = [
        (gs[1, 0], ndvi_hr,      "NDVI — HR Ground Truth",   "RdYlGn", (-1, 1)),
        (gs[1, 1], ndvi_bicubic, "NDVI — Bicubic",           "RdYlGn", (-1, 1)),
        (gs[1, 2], ndvi_sr,      "NDVI — SR (ours)",         "RdYlGn", (-1, 1)),
        (gs[1, 3], err_bicubic,  "|NDVI Error| — Bicubic",   "Reds",   (0, None)),
        (gs[1, 4], err_sr,       "|NDVI Error| — SR (ours)", "Reds",   (0, None)),
    ]

    for spec, data, title, cmap, vrange in bottom_panels:
        ax = fig.add_subplot(spec)
        vmin, vmax = vrange
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax if vmax else data.max(), interpolation="nearest")
        ax.set_title(title, color="white", fontsize=8, pad=4)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Metrics text box
    delta_psnr = psnr_sr - psnr_bicubic
    delta_ssim = ssim_sr - ssim_bicubic
    err_improvement_ndvi = ((err_bicubic.mean() - err_sr.mean()) / (err_bicubic.mean() + 1e-8)) * 100
    stats_text = (
        f"PSNR:  Bicubic={psnr_bicubic:.2f} dB  ->  SR={psnr_sr:.2f} dB  (Δ{delta_psnr:+.2f} dB)\n"
        f"SSIM:  Bicubic={ssim_bicubic:.4f}   ->  SR={ssim_sr:.4f}   (Δ{delta_ssim:+.4f})\n"
        f"NDVI MAE improvement: {err_improvement_ndvi:+.1f}%"
    )
    fig.text(
        0.5, 0.01, stats_text,
        ha="center", va="bottom", fontsize=10, color="#e6edf3",
        fontfamily="monospace",
        bbox=dict(facecolor="#161b22", edgecolor="#30363d", pad=6),
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved comparison figure -> {save_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lr-npy", required=True)
    ap.add_argument("--hr-npy", required=True)
    ap.add_argument("--out", default="results/comparison.png")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    lr_arr = np.load(args.lr_npy).astype(np.float32)
    hr_arr = np.load(args.hr_npy).astype(np.float32)

    # Ensure 3D CHW
    if lr_arr.ndim == 2: lr_arr = lr_arr[None]
    if hr_arr.ndim == 2: hr_arr = hr_arr[None]

    print(f"LR: {lr_arr.shape}, HR: {hr_arr.shape}")

    bicubic = bicubic_upsample(lr_arr, args.scale)

    print("Running model inference...")
    sr_mean, sr_var = run_inference(args.checkpoint, lr_arr, args.device)
    print(f"SR output: {sr_mean.shape}")

    # Align shapes
    min_h = min(sr_mean.shape[1], hr_arr.shape[1], bicubic.shape[1])
    min_w = min(sr_mean.shape[2], hr_arr.shape[2], bicubic.shape[2])
    sr_mean = sr_mean[:, :min_h, :min_w]
    hr_crop = hr_arr[:, :min_h, :min_w]
    bicubic = bicubic[:, :min_h, :min_w]

    p_bic = psnr(bicubic, hr_crop)
    s_bic = ssim(bicubic, hr_crop)
    p_sr  = psnr(sr_mean, hr_crop)
    s_sr  = ssim(sr_mean, hr_crop)

    print(f"Bicubic — PSNR: {p_bic:.2f} dB  SSIM: {s_bic:.4f}")
    print(f"SR Model — PSNR: {p_sr:.2f} dB  SSIM: {s_sr:.4f}")
    print(f"Gain — PSNR: {p_sr-p_bic:+.2f} dB  SSIM: {s_sr-s_bic:+.4f}")

    make_comparison_figure(
        lr=lr_arr, bicubic=bicubic, sr_mean=sr_mean, sr_var=sr_var, hr=hr_crop,
        psnr_bicubic=p_bic, ssim_bicubic=s_bic,
        psnr_sr=p_sr, ssim_sr=s_sr,
        save_path=args.out,
    )


if __name__ == "__main__":
    main()
