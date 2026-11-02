"""
Shared utility functions: config loading, reproducibility, logging, and plotting.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml


def load_config(config_path: str) -> dict:
    """Load and parse a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir: Path, log_filename: str = "run.log") -> logging.Logger:
    """Configure the root logger to write to both stdout and a log file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_filename

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    """Serialise a metrics dict to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Render a confusion-matrix heatmap and save it to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved -> {output_path}")


def plot_training_curves(log_csv: Path, output_dir: Path, model_name: str = "") -> None:
    """Plot loss and accuracy curves from the training log CSV and save as PNG."""
    import pandas as pd

    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    suffix = f" ({model_name})" if model_name else ""

    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss",   marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss Curves{suffix}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["train_acc"], label="Train Acc", marker="o", markersize=3)
    axes[1].plot(df["epoch"], df["val_acc"],   label="Val Acc",   marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"Accuracy Curves{suffix}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    stem    = f"training_curves_{model_name}" if model_name else "training_curves"
    out_path = output_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved -> {out_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str = "ROC Curve",
) -> None:
    """Plot the Receiver Operating Characteristic curve and save as PNG.

    Args:
        y_true:      Ground-truth binary labels (0 = NORMAL, 1 = PNEUMONIA).
        y_prob:      Predicted probability for the positive class.
        output_path: Where to save the PNG.
        title:       Plot title.
    """
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2.5,
            label=f"AUC = {roc_auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.12, color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved        -> {output_path}")


def plot_threshold_sweep(
    sweep: List[Dict],
    optimal_threshold: float,
    output_path: Path,
    title: str = "Threshold Analysis",
) -> None:
    """Plot sensitivity/specificity and F1 score as a function of decision threshold.

    Args:
        sweep:             List of dicts with keys threshold, sensitivity, specificity, f1.
        optimal_threshold: The Youden's-J-optimal threshold (shown as vertical line).
        output_path:       Where to save the PNG.
        title:             Figure suptitle.
    """
    thresholds   = [s["threshold"]   for s in sweep]
    sensitivity  = [s["sensitivity"] for s in sweep]
    specificity  = [s["specificity"] for s in sweep]
    f1_scores    = [s["f1"]          for s in sweep]

    max_f1   = max(f1_scores)
    max_f1_t = thresholds[f1_scores.index(max_f1)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Left: Sensitivity & Specificity vs Threshold
    ax = axes[0]
    ax.plot(thresholds, sensitivity, color="tomato",    linewidth=2.2, label="Sensitivity (TPR)")
    ax.plot(thresholds, specificity, color="steelblue", linewidth=2.2, label="Specificity (TNR)")
    ax.axvline(optimal_threshold, color="green", linestyle="--", linewidth=1.8,
               label=f"Youden's J  (t = {optimal_threshold:.2f})")
    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Sensitivity & Specificity vs Threshold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Right: F1 vs Threshold
    ax = axes[1]
    ax.plot(thresholds, f1_scores, color="darkorchid", linewidth=2.2, label="F1 Score")
    ax.axvline(max_f1_t, color="green", linestyle="--", linewidth=1.8,
               label=f"Max F1 = {max_f1:.3f}  (t = {max_f1_t:.2f})")
    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("F1 Score vs Threshold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Threshold sweep saved  -> {output_path}")
