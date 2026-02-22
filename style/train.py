"""
Training loop for StyleNet.

Trains the model to predict which move the target player would make,
using cross-entropy loss over the 4096-dimensional move vocabulary.

Top-1 accuracy here means "the model's single best guess was the exact move
played." In practice 15-30% top-1 accuracy is quite good — chess has many
reasonable moves and the network isn't expected to perfectly clone the player,
just capture their stylistic tendencies.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from style.model import StyleNet
from style.dataset import PlayerGamesDataset


def train(
    dataset: PlayerGamesDataset,
    model_path: str,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    channels: int = 128,
    num_res_blocks: int = 6,
) -> StyleNet:
    """
    Train a StyleNet on the given dataset and save the best checkpoint.

    Args:
        dataset:        PlayerGamesDataset of (board_tensor, move_index) pairs.
        model_path:     Where to save the best model weights (.pt file).
        epochs:         Number of full passes over the training data.
        batch_size:     Mini-batch size.
        lr:             Initial learning rate (Adam).
        val_split:      Fraction of data held out for validation.
        channels:       StyleNet channel width.
        num_res_blocks: StyleNet depth.

    Returns:
        The trained StyleNet (best checkpoint loaded).
    """
    device = torch.device(
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Training on {device}  |  {len(dataset):,} positions")

    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    pin = device.type == "cuda"  # pin_memory only benefits CUDA
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin)

    model     = StyleNet(channels=channels, num_res_blocks=num_res_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = train_correct = 0

        for tensors, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            tensors, labels = tensors.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(tensors)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()

        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = val_correct = 0

        with torch.no_grad():
            for tensors, labels in val_loader:
                tensors, labels = tensors.to(device), labels.to(device)
                logits  = model(tensors)
                loss    = criterion(logits, labels)
                val_loss    += loss.item() * len(labels)
                val_correct += (logits.argmax(1) == labels).sum().item()

        avg_train_loss = train_loss / n_train
        avg_val_loss   = val_loss   / n_val
        train_acc      = train_correct / n_train
        val_acc        = val_correct   / n_val

        print(
            f"Epoch {epoch:>2}/{epochs}  "
            f"train_loss={avg_train_loss:.4f}  train_acc={train_acc:.3f}  |  "
            f"val_loss={avg_val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"           → saved best model  (val_loss={best_val_loss:.4f})")

    # Load best weights before returning
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"\nDone. Best val_loss={best_val_loss:.4f}  model saved to {model_path}")
    return model
