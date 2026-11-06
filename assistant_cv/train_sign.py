"""Train SignNet GRU classifier with stratified train/val split and sequence augmentation."""

from __future__ import annotations

import argparse
import copy
import pathlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .config import DATA_DIR, DEVICE, SIGN_MODEL_PATH
    from .model import SignNet
except ImportError:
    from config import DATA_DIR, DEVICE, SIGN_MODEL_PATH
    from model import SignNet


def train(
    data_dir: str | pathlib.Path = DATA_DIR,
    out_path: str | pathlib.Path = SIGN_MODEL_PATH,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Train SignNet on .npy sequences under data_dir and save best checkpoint to out_path."""
    data_root = pathlib.Path(data_dir)
    if not data_root.exists():
        raise FileNotFoundError(
            f"Data directory '{data_root}' does not exist. "
            "Collect sequences first with: python assistant_cv/collect.py <label>"
        )

    candidate_dirs = sorted(
        p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )

    labels: list[str] = []
    class_sequences: dict[str, list[np.ndarray]] = {}
    for d in candidate_dirs:
        files = sorted(d.glob("*.npy"))
        if files:
            seqs = [np.load(f).astype(np.float32) for f in files]
            labels.append(d.name)
            class_sequences[d.name] = seqs

    if len(labels) < 2:
        raise ValueError(
            f"Need at least 2 non-empty sign classes in '{data_root}', found {len(labels)} "
            f"({labels}). Run `python assistant_cv/collect.py <label>` for at least 2 classes."
        )

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X_tr_list: list[np.ndarray] = []
    y_tr_list: list[int] = []
    X_va_list: list[np.ndarray] = []
    y_va_list: list[int] = []

    for cls_idx, label in enumerate(labels):
        seqs = class_sequences[label]
        n = len(seqs)
        perm = rng.permutation(n)
        if n >= 2:
            n_val = max(1, int(round(0.2 * n)))
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]
        else:
            tr_idx = perm
            val_idx = perm

        for idx in tr_idx:
            X_tr_list.append(seqs[int(idx)])
            y_tr_list.append(cls_idx)
        for idx in val_idx:
            X_va_list.append(seqs[int(idx)])
            y_va_list.append(cls_idx)

    X_tr = torch.tensor(np.stack(X_tr_list, axis=0), dtype=torch.float32)
    y_tr = torch.tensor(y_tr_list, dtype=torch.long)
    X_va = torch.tensor(np.stack(X_va_list, axis=0), dtype=torch.float32, device=DEVICE)
    y_va = torch.tensor(y_va_list, dtype=torch.long, device=DEVICE)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=min(batch_size, len(X_tr)),
        shuffle=True,
    )

    net = SignNet(len(labels)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(net.state_dict())

    for ep in range(1, int(epochs) + 1):
        net.train()
        total_loss = 0.0
        total_samples = 0

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            # Sequence augmentation: random scale jitter + Gaussian noise
            scale = torch.empty((xb.size(0), 1, 1), device=xb.device, dtype=xb.dtype).uniform_(
                0.92, 1.08
            )
            xb_aug = xb * scale + 0.01 * torch.randn_like(xb)

            opt.zero_grad()
            logits = net(xb_aug)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * xb.size(0)
            total_samples += xb.size(0)

        avg_loss = total_loss / max(1, total_samples)

        net.eval()
        with torch.no_grad():
            va_logits = net(X_va)
            va_loss = float(crit(va_logits, y_va).item())
            acc = float((va_logits.argmax(dim=1) == y_va).float().mean().item())

        if acc > best_acc or (acc == best_acc and va_loss < best_val_loss):
            best_acc = acc
            best_val_loss = va_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if ep % 10 == 0 or ep == 1 or ep == int(epochs):
            print(f"{ep:03d}  loss {avg_loss:.3f}  val_acc {acc:.3f}")

    out_file = pathlib.Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"m": best_state_dict, "labels": labels}
    torch.save(ckpt, out_file)

    return {
        "best_val_acc": best_acc,
        "best_val_loss": best_val_loss,
        "labels": labels,
        "out_path": str(out_file),
        "m": best_state_dict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SignNet GRU sign language classifier.")
    parser.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR, help="Dataset directory")
    parser.add_argument(
        "--out", type=pathlib.Path, default=SIGN_MODEL_PATH, help="Output checkpoint path"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()
    train(data_dir=args.data_dir, out_path=args.out, epochs=args.epochs)


if __name__ == "__main__":
    main()
