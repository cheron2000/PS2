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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", default="runs/sr")
    parser.add_argument("--seed", type=int, default=0)
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
    fit(
        model,
        build_loader(dataset, batch_size=args.batch_size),
        criterion,
        optimizer,
        args.epochs,
        device,
        checkpoint_dir=args.checkpoint_dir,
        model_config=model_config,
    )


if __name__ == "__main__":
    main()
