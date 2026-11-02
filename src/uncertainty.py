"""
MC Dropout uncertainty estimation for the Chest X-Ray Pneumonia classifier.

Runs N stochastic forward passes (dropout active, BatchNorm frozen) to estimate
per-sample prediction uncertainty.  High uncertainty = model is unsure = the
case should be flagged for human review.

Outputs:
  outputs/uncertainty/uncertainty_{model}_{date}.png  — three-panel plot
  outputs/uncertainty/uncertainty_{model}_{date}.csv  — per-sample predictions

Usage:
    python src/uncertainty.py --config config/config.yaml
    python src/uncertainty.py --config config/config.yaml \\
        --checkpoint outputs/best_model_improved_cnn_2026-05-13.pth \\
        --n_passes 30
"""

import argparse
import csv
import sys
from datetime import date
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Patch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import CLASS_NAMES, get_data_loaders
from model import get_model
from utils import load_config


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------

def _enable_dropout(model: nn.Module) -> None:
    """Keep Dropout layers in train mode (stochastic) while BatchNorm stays eval."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_inference(
    model:    nn.Module,
    loader:   torch.utils.data.DataLoader,
    device:   torch.device,
    n_passes: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run N stochastic forward passes and compute mean/std of predictions.

    Dropout is active at inference; BatchNorm uses its running statistics so
    batch-size artefacts don't pollute the uncertainty estimate.

    Args:
        model:    Model whose dropout layers will be made stochastic.
        loader:   Test DataLoader with shuffle=False (consistent ordering).
        device:   Computation device.
        n_passes: Number of forward passes.

    Returns:
        y_true:     Ground-truth labels              (N,)
        mean_probs: Mean softmax probability         (N, 2)
        std_probs:  Std-dev of softmax probability   (N, 2)
        y_pred:     Final predictions (argmax mean)  (N,)
    """
    _enable_dropout(model)

    all_pass_probs: list = []
    y_true: list = []

    for pass_idx in tqdm(range(n_passes), desc="MC Dropout passes"):
        pass_probs: list = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                probs  = torch.softmax(model(images), dim=1).cpu().numpy()
                pass_probs.append(probs)
                if pass_idx == 0:
                    y_true.extend(labels.numpy())
        all_pass_probs.append(np.vstack(pass_probs))

    stacked    = np.stack(all_pass_probs)   # (n_passes, N, 2)
    mean_probs = stacked.mean(axis=0)       # (N, 2)
    std_probs  = stacked.std(axis=0)        # (N, 2)
    y_pred     = mean_probs.argmax(axis=1)

    return np.array(y_true), mean_probs, std_probs, y_pred


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(
    y_true:     np.ndarray,
    mean_probs: np.ndarray,
    std_probs:  np.ndarray,
    y_pred:     np.ndarray,
) -> None:
    """Print accuracy, uncertainty statistics, and the 10 most uncertain samples."""
    uncertainty = std_probs[:, 1]
    correct     = y_pred == y_true
    accuracy    = correct.mean()

    print(f"\n{'=' * 62}")
    print(f"  MC Dropout Uncertainty Report")
    print(f"  Accuracy             : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  Uncertainty (all)    : {uncertainty.mean():.4f} ± {uncertainty.std():.4f}")
    if correct.any():
        print(f"  Uncertainty (correct): {uncertainty[correct].mean():.4f}")
    if (~correct).any():
        print(f"  Uncertainty (wrong)  : {uncertainty[~correct].mean():.4f}")

    q1, q2, q3 = np.percentile(uncertainty, [25, 50, 75])
    print(f"\n  Accuracy by uncertainty quartile:")
    for label, mask in [
        (f"Q1 (< {q1:.4f})",    uncertainty < q1),
        (f"Q2 ({q1:.3f}–{q2:.3f})", (uncertainty >= q1) & (uncertainty < q2)),
        (f"Q3 ({q2:.3f}–{q3:.3f})", (uncertainty >= q2) & (uncertainty < q3)),
        (f"Q4 (> {q3:.4f})",    uncertainty > q3),
    ]:
        acc_q = correct[mask].mean() if mask.any() else float("nan")
        print(f"    {label:<22}: {acc_q:.3f}  (n={mask.sum()})")

    print(f"\n  Top 10 most uncertain predictions:")
    print(f"  {'Idx':>5}  {'True':>12}  {'Pred':>12}  {'P(Pneu)':>9}  {'Uncert':>8}  Correct")
    for idx in np.argsort(uncertainty)[-10:][::-1]:
        print(
            f"  {idx:>5}  {CLASS_NAMES[int(y_true[idx])]:>12}  "
            f"{CLASS_NAMES[int(y_pred[idx])]:>12}  "
            f"{mean_probs[idx, 1]:>9.4f}  "
            f"{uncertainty[idx]:>8.4f}  "
            f"{'Yes' if correct[idx] else 'No'}"
        )
    print("=" * 62)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_uncertainty_analysis(
    y_true:     np.ndarray,
    mean_probs: np.ndarray,
    std_probs:  np.ndarray,
    y_pred:     np.ndarray,
    output_dir: Path,
    model_name: str,
) -> None:
    """Save a three-panel uncertainty visualisation.

    Panel 1: Uncertainty histograms — correct vs incorrect predictions.
    Panel 2: Confidence vs uncertainty scatter — coloured by correctness.
    Panel 3: Reliability diagram — calibration of the mean probabilities.
    """
    uncertainty = std_probs[:, 1]
    confidence  = mean_probs[:, 1]
    correct     = y_pred == y_true

    output_dir.mkdir(parents=True, exist_ok=True)
    date_tag = date.today().strftime("%Y-%m-%d")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"MC Dropout Uncertainty Analysis — {model_name}", fontsize=13, fontweight="bold")

    # --- Panel 1: Uncertainty histogram ---
    ax = axes[0]
    ax.hist(uncertainty[correct],  bins=30, alpha=0.75, color="steelblue", label=f"Correct ({correct.sum()})")
    ax.hist(uncertainty[~correct], bins=30, alpha=0.75, color="tomato",    label=f"Incorrect ({(~correct).sum()})")
    ax.set_xlabel("Prediction Uncertainty (std of P(Pneumonia))", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Uncertainty Distribution", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 2: Confidence vs Uncertainty scatter ---
    ax = axes[1]
    colors = np.where(correct, "steelblue", "tomato")
    ax.scatter(confidence, uncertainty, c=colors, alpha=0.4, s=12, linewidths=0)
    ax.set_xlabel("Mean P(Pneumonia)", fontsize=10)
    ax.set_ylabel("Uncertainty (std)", fontsize=10)
    ax.set_title("Confidence vs Uncertainty", fontsize=11)
    handles = [
        Patch(color="steelblue", label=f"Correct ({correct.sum()})"),
        Patch(color="tomato",    label=f"Incorrect ({(~correct).sum()})"),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 3: Reliability diagram ---
    ax = axes[2]
    bins = np.linspace(0.0, 1.0, 11)
    bin_acc  = []
    bin_conf = []
    bin_cnt  = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() > 0:
            bin_acc.append(float(correct[mask].mean()))
            bin_conf.append(float(confidence[mask].mean()))
            bin_cnt.append(int(mask.sum()))
        else:
            bin_acc.append(0.0)
            bin_conf.append(float((lo + hi) / 2))
            bin_cnt.append(0)

    bin_centers = [(lo + hi) / 2 for lo, hi in zip(bins[:-1], bins[1:])]
    ax.bar(bin_centers, bin_acc, width=0.09, alpha=0.7, color="steelblue", label="Accuracy per bin")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
    ax.set_xlabel("Mean Confidence", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("Reliability Diagram (Calibration)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out_path = output_dir / f"uncertainty_{model_name}_{date_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Uncertainty plots saved -> {out_path}")


def save_uncertainty_csv(
    y_true:     np.ndarray,
    mean_probs: np.ndarray,
    std_probs:  np.ndarray,
    y_pred:     np.ndarray,
    output_dir: Path,
    model_name: str,
) -> None:
    """Save per-sample predictions and uncertainty scores to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    date_tag = date.today().strftime("%Y-%m-%d")
    out_path = output_dir / f"uncertainty_{model_name}_{date_tag}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_idx", "true_label", "pred_label",
            "p_normal_mean", "p_pneumonia_mean",
            "p_normal_std",  "p_pneumonia_std",
            "uncertainty", "correct",
        ])
        for i in range(len(y_true)):
            writer.writerow([
                i,
                CLASS_NAMES[int(y_true[i])],
                CLASS_NAMES[int(y_pred[i])],
                f"{mean_probs[i, 0]:.4f}",
                f"{mean_probs[i, 1]:.4f}",
                f"{std_probs[i, 0]:.4f}",
                f"{std_probs[i, 1]:.4f}",
                f"{std_probs[i, 1]:.4f}",
                int(y_true[i] == y_pred[i]),
            ])
    print(f"Uncertainty CSV saved   -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MC Dropout uncertainty estimation")
    parser.add_argument("--config",     type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_passes",   type=int, default=30,
                        help="Number of MC Dropout forward passes (default: 30)")
    return parser.parse_args()


def main() -> None:
    args       = parse_args()
    config     = load_config(args.config)
    output_dir = Path(config["data"]["output_dir"])
    model_name = config["model"]["name"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        candidates = sorted(output_dir.glob(f"best_model_{model_name}_*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint found for '{model_name}' in '{output_dir}'.\n"
                "Run training first:  python src/train.py --config config/config.yaml"
            )
        checkpoint_path = candidates[-1]

    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"MC passes  : {args.n_passes}")

    checkpoint   = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config", config)
    model        = get_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, test_loader = get_data_loaders(config)

    y_true, mean_probs, std_probs, y_pred = mc_dropout_inference(
        model, test_loader, device, n_passes=args.n_passes
    )

    print_summary(y_true, mean_probs, std_probs, y_pred)

    uncertainty_dir = output_dir / "uncertainty"
    plot_uncertainty_analysis(y_true, mean_probs, std_probs, y_pred, uncertainty_dir, model_name)
    save_uncertainty_csv(y_true, mean_probs, std_probs, y_pred, uncertainty_dir, model_name)


if __name__ == "__main__":
    main()
