"""
Training script for the Chest X-Ray Pneumonia CNN classifier.

Usage:
    python src/train.py --config config/config.yaml
"""

import argparse
import csv
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm

# Allow running from repo root or from src/
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_class_distribution, get_data_loaders
from model import get_model
from utils import load_config, plot_training_curves, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a CNN on the Chest X-Ray Pneumonia dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file (default: config/config.yaml)",
    )
    return parser.parse_args()


def sanity_check(config: dict, logger: logging.Logger) -> None:
    """Verify that the data directory exists and log class distribution.

    Raises:
        FileNotFoundError: If the data directory or any required split is missing.
    """
    data_dir = Path(config["data"]["data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found.\n"
            "Download the dataset with:\n"
            "  kaggle datasets download -d paultimothymooney/chest-xray-pneumonia\n"
            "  unzip chest-xray-pneumonia.zip -d data/"
        )

    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Expected split directory '{split_dir}' not found. "
                "Ensure the dataset was extracted correctly."
            )

    distribution = get_class_distribution(data_dir)
    logger.info("Class distribution across splits:")
    for split, counts in distribution.items():
        total = sum(counts.values())
        parts = ", ".join(f"{cls}: {n}" for cls, n in counts.items())
        logger.info(f"  {split:5s}: {parts}  (total={total})")


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_train: bool,
    gradient_clip: float = 0.0,
) -> Tuple[float, float]:
    """Run one full pass over a DataLoader.

    Args:
        model:         The neural network.
        loader:        DataLoader for the current split.
        criterion:     Loss function.
        optimizer:     Optimiser (only used when is_train=True).
        device:        Computation device.
        is_train:      If True, performs backward pass and parameter updates.
        gradient_clip: Max gradient norm for clipping; 0 disables clipping.

    Returns:
        Tuple of (average_loss, accuracy_percent).
    """
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    desc = "train" if is_train else "val  "
    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, leave=False, desc=desc):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                if gradient_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train(config: dict, logger: logging.Logger) -> None:
    """Execute the full training loop.

    Loads data, builds the model, runs epochs, saves the best checkpoint,
    logs metrics per epoch to CSV, and generates training curve plots.

    Args:
        config: Parsed YAML config dict.
        logger: Logger instance for console and file output.
    """
    set_seed(config["training"]["seed"])

    output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")

    sanity_check(config, logger)

    train_loader, val_loader, _ = get_data_loaders(config)

    model = get_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model  : {config['model']['name']} | "
        f"Total params: {n_params:,} | Trainable: {n_trainable:,}"
    )

    epochs = config["training"]["epochs"]

    label_smoothing = config["training"].get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler_name = config["training"]["scheduler"]
    if scheduler_name == "step_lr":
        scheduler = StepLR(
            optimizer,
            step_size=config["training"]["step_lr_step_size"],
            gamma=config["training"]["step_lr_gamma"],
        )
    elif scheduler_name == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config["training"]["reduce_lr_patience"],
            factor=config["training"]["reduce_lr_factor"],
        )
    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config["training"].get("cosine_t_max", epochs),
            eta_min=config["training"].get("cosine_eta_min", 1e-6),
        )
    else:
        raise ValueError(
            f"Unknown scheduler: '{scheduler_name}'. Use step_lr, reduce_on_plateau, or cosine."
        )

    gradient_clip: float = config["training"].get("gradient_clip", 0.0)

    model_name = config["model"]["name"]
    date_tag = date.today().strftime("%Y-%m-%d")
    log_csv = output_dir / f"training_log_{model_name}_{date_tag}.csv"
    best_model_path = output_dir / f"best_model_{model_name}_{date_tag}.pth"
    best_val_loss = float("inf")

    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    logger.info(f"Starting training for {epochs} epoch(s)...")
    logger.info("-" * 75)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True,
            gradient_clip=gradient_clip,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, is_train=False,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train  loss={train_loss:.4f} acc={train_acc:.2f}% | "
            f"Val    loss={val_loss:.4f} acc={val_acc:.2f}% | "
            f"LR={current_lr:.2e}"
        )

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()  # StepLR and CosineAnnealingLR step every epoch

        with open(log_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_acc:.4f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.4f}",
                    f"{current_lr:.2e}",
                ]
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": config,
                },
                best_model_path,
            )
            logger.info(f"           -> Best model saved  (val_loss={val_loss:.4f})")

    logger.info("-" * 75)
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoint : {best_model_path}")
    logger.info(f"Log CSV    : {log_csv}")

    plot_training_curves(log_csv, output_dir, f"{model_name}_{date_tag}")


def main() -> None:
    """Entry point."""
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config["data"]["output_dir"])
    model_name = config["model"]["name"]
    date_tag = date.today().strftime("%Y-%m-%d")
    logger = setup_logging(output_dir, log_filename=f"train_{model_name}_{date_tag}.log")
    train(config, logger)


if __name__ == "__main__":
    main()
