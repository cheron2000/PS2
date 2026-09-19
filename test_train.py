"""Synthetic T9 integration test. Run: python3 test_train.py"""
import os
import shutil
import tempfile

import numpy as np
import torch

from src.datasets.sen2naip import upsample_nearest
from src.losses import SRLoss
from src.model import SRModel
from src.train import build_loader, fit, set_seed


def main():
    set_seed(7)
    root = tempfile.mkdtemp()
    run_dir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "lr"))
        os.makedirs(os.path.join(root, "hr"))
        rng = np.random.default_rng(7)
        for i in range(2):
            lr = (rng.random((4, 8, 8)) * 0.8 + 0.1).astype(np.float32)
            hr = upsample_nearest(lr, 4).astype(np.float32)
            np.save(os.path.join(root, "lr", f"scene{i}.npy"), lr)
            np.save(os.path.join(root, "hr", f"scene{i}.npy"), hr)

        from src.datasets.sen2naip import SEN2NAIPDataset
        dataset = SEN2NAIPDataset(root, min_ncc_score=0.3)
        assert len(dataset) == 2, dataset.dropped_pairs
        loader = build_loader(dataset, batch_size=1, shuffle=False)
        model_config = {
            "in_channels": 4,
            "out_channels": 4,
            "base_channels": 8,
            "num_attn_blocks": 1,
            "window_size": 4,
            "num_heads": 2,
            "scale": 4,
        }
        model = SRModel(**model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        history = fit(
            model, loader, SRLoss(), optimizer, epochs=1,
            device=torch.device("cpu"), val_loader=loader,
            checkpoint_dir=run_dir, model_config=model_config,
        )
        assert len(history) == 1
        assert np.isfinite(history[0]["train_loss"])
        assert np.isfinite(history[0]["val_loss"])
        assert os.path.exists(os.path.join(run_dir, "last.pt"))
        assert os.path.exists(os.path.join(run_dir, "best.pt"))
        assert os.path.exists(os.path.join(run_dir, "history.json"))
        checkpoint = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
        assert checkpoint["model_config"] == model_config
        print("[PASS] T9 training, validation, checkpoint, and history integration")
    finally:
        shutil.rmtree(root)
        shutil.rmtree(run_dir)


if __name__ == "__main__":
    main()
