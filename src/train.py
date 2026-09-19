"""
Training loop for the Sentinel-2 super-resolution prototype (T9).

The loop intentionally keeps the dataset contract small: each sample must provide
``lr`` and ``hr`` tensors/arrays in CHW format, and may optionally provide a
broadcastable ``valid_mask``. The existing SEN2NAIP/SEN2Vénus loaders satisfy this
contract. Batch size defaults to 1 because co-registration crops can produce
slightly different spatial sizes across pairs.

Example:
    python -m src.train --data-root data/sen2naip --epochs 5 --checkpoint-dir runs/demo
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.losses import SRLoss
from src.model import SRModel
from src.datasets.sen2naip import SEN2NAIPDataset


def set_seed(seed: int) -> None:
    """Make Python, NumPy, and CPU/GPU torch operations as repeatable as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_batch_tensor(value, device: torch.device) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def _sample_tensors(sample: Dict, device: torch.device):
    lr = _as_batch_tensor(sample["lr"], device)
    hr = _as_batch_tensor(sample["hr"], device)
    mask = sample.get("valid_mask")
    if mask is not None:
        mask = _as_batch_tensor(mask, device)
    return lr, hr, mask


def run_epoch(
    model: torch.nn.Module,
    loader: Iterable[Dict],
    criterion: SRLoss,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, Dict[str, float]]:
    """Run one train or validation epoch and return mean loss/components."""
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "reconstruction": 0.0, "uncertainty_nll": 0.0}
    count = 0

    for sample in loader:
        lr, target, valid_mask = _sample_tensors(sample, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction, log_var = model(lr)
            if prediction.shape != target.shape:
                raise ValueError(
                    "model output and target shapes differ: "
                    f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}"
                )
            loss, parts = criterion(
                prediction,
                target,
                log_var,
                valid_mask=valid_mask,
                return_components=True,
            )
            if training:
                loss.backward()
                optimizer.step()

        for name, value in parts.items():
            totals[name] += float(value.item())
        count += 1

    if count == 0:
        raise ValueError("cannot run an epoch over an empty loader")
    means = {name: value / count for name, value in totals.items()}
    return means["total"], means


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    model_config: Optional[Dict] = None,
) -> None:
    """Save a self-contained checkpoint suitable for later inference/resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "model_config": model_config or {},
        },
        path,
    )


def fit(
    model: torch.nn.Module,
    train_loader: Iterable[Dict],
    criterion: SRLoss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    val_loader: Optional[Iterable[Dict]] = None,
    checkpoint_dir: Optional[str | Path] = None,
    model_config: Optional[Dict] = None,
) -> list[Dict[str, float]]:
    """Train, optionally validate, checkpoint latest/best, and return history."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    model.to(device)
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
    history = []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_parts = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_reconstruction": train_parts["reconstruction"],
            "train_uncertainty_nll": train_parts["uncertainty_nll"],
        }
        if val_loader is not None:
            val_loss, val_parts = run_epoch(model, val_loader, criterion, device)
            record.update(
                {
                    "val_loss": val_loss,
                    "val_reconstruction": val_parts["reconstruction"],
                    "val_uncertainty_nll": val_parts["uncertainty_nll"],
                }
            )
            score = val_loss
        else:
            score = train_loss

        history.append(record)
        if checkpoint_path:
            save_checkpoint(
                checkpoint_path / "last.pt", model, optimizer, epoch, record, model_config
            )
            if score <= best_val:
                best_val = score
                save_checkpoint(
                    checkpoint_path / "best.pt", model, optimizer, epoch, record, model_config
                )
            with (checkpoint_path / "history.json").open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f}"
            + (f" val_loss={record['val_loss']:.6f}" if val_loader is not None else "")
        )
    return history


def build_loader(dataset: Dataset, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    """Build a loader; batch size 1 avoids stacking variable post-QC crop sizes."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def split_dataset(dataset: Dataset, val_fraction: float, seed: int):
    """Deterministic train/val split by sample index (audit finding #1 fix,
    2026-09-19). fit() already accepted a val_loader and used it correctly
    for best-checkpoint selection -- this CLI just never built one, so
    "best.pt" was always selected by *training* loss, the exact failure mode
    the audit flagged. random_split was already imported and unused; this is
    the wiring, not new machinery.

    NOTE on scope: this gives a real, reproducible-under-a-fixed-seed index
    split, which is enough to stop training loss silently driving checkpoint
    selection. It does NOT give a persisted, scene/AOI-aware split with a
    manifest of which source files went where -- that's a larger piece of
    work (the audit's proposed reproducibility/scene-split task) genuinely
    out of scope for this fix. Two dataset instances share the same list of
    pairs in the same order (assuming the same directory contents), so a
    fixed seed reproduces the same index split across runs, but does not
    yet record file-level provenance -- flagged as a follow-up, not silently
    treated as solved.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    if n_train < 1:
        raise ValueError(
            f"val_fraction={val_fraction} leaves no training samples "
            f"(dataset size={len(dataset)}) -- use a smaller val_fraction or more data"
        )
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", default="runs/sr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="fraction of samples held out for validation-driven checkpoint "
             "selection (default 0.2). Pass 0 to disable and fall back to "
             "training-loss-based selection explicitly, rather than by omission.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    dataset = SEN2NAIPDataset(args.data_root)
    if len(dataset) == 0:
        raise ValueError("dataset contains no trusted pairs after alignment/QC")
    model_config = {"in_channels": 4, "out_channels": 4, "scale": dataset.scale}
    model = SRModel(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = SRLoss()

    if args.val_fraction > 0:
        train_subset, val_subset = split_dataset(dataset, args.val_fraction, args.seed)
        train_loader = build_loader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = build_loader(val_subset, batch_size=args.batch_size, shuffle=False)
        print(f"train/val split: {len(train_subset)} train, {len(val_subset)} val "
              f"(val_fraction={args.val_fraction}, seed={args.seed})")
    else:
        train_loader = build_loader(dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = None
        print("WARNING: --val-fraction 0 -- best-checkpoint selection will use "
              "TRAINING loss, not validation loss. Only do this deliberately.")

    fit(
        model,
        train_loader,
        criterion,
        optimizer,
        args.epochs,
        device,
        val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
        model_config=model_config,
    )


if __name__ == "__main__":
    main()
