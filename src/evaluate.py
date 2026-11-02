"""
Evaluation script for the Chest X-Ray Pneumonia CNN classifier.

Loads the best saved checkpoint, runs inference on the test set, and reports:
  - Accuracy, ROC-AUC, weighted Precision / Recall / F1
  - Clinical metrics: Sensitivity, Specificity, PPV, NPV
  - Optimal decision threshold via Youden's J index
  - Confusion matrix (text + PNG)
  - ROC curve PNG
  - Threshold sweep PNG (Sensitivity, Specificity, F1 vs threshold)
  - Full sklearn classification report

Usage:
    python src/evaluate.py --config config/config.yaml
    python src/evaluate.py --config config/config.yaml --checkpoint outputs/best_model.pth
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import CLASS_NAMES, get_data_loaders
from model import get_model
from utils import (
    load_config,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_threshold_sweep,
    save_metrics,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained CNN on the chest X-ray test set"
    )
    parser.add_argument("--config",     type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint. Defaults to latest in outputs/.")
    return parser.parse_args()


def load_model(
    config: dict,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load model architecture and weights from a checkpoint."""
    checkpoint   = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config", config)
    model        = get_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch   = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", float("nan"))
    val_acc  = checkpoint.get("val_acc",  float("nan"))
    print(f"Loaded checkpoint — epoch {epoch}  (val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%)")
    return model


def run_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over a DataLoader and collect labels, predictions, probabilities.

    Returns:
        y_true: ground-truth labels
        y_pred: argmax predictions
        y_prob: softmax probability for PNEUMONIA (index 1), used for ROC-AUC
    """
    all_labels: list = []
    all_preds:  list = []
    all_probs:  list = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images  = images.to(device, non_blocking=True)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = probs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def _threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_steps: int = 99,
) -> Tuple[list, float]:
    """Compute sensitivity, specificity, and F1 at evenly-spaced thresholds.

    Returns:
        sweep:             List of dicts {threshold, sensitivity, specificity, f1}.
        optimal_threshold: Threshold that maximises Youden's J (sens + spec − 1).
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    sweep = []

    for t in thresholds:
        y_t  = (y_prob >= t).astype(int)
        cm_t = confusion_matrix(y_true, y_t, labels=[0, 1])
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

        sens = float(tp_t / (tp_t + fn_t)) if (tp_t + fn_t) > 0 else 0.0
        spec = float(tn_t / (tn_t + fp_t)) if (tn_t + fp_t) > 0 else 0.0
        f1   = float(f1_score(y_true, y_t, pos_label=1, zero_division=0))

        sweep.append({
            "threshold":   float(t),
            "sensitivity": sens,
            "specificity": spec,
            "f1":          f1,
        })

    youden_j = [s["sensitivity"] + s["specificity"] - 1.0 for s in sweep]
    optimal_threshold = float(sweep[int(np.argmax(youden_j))]["threshold"])
    return sweep, optimal_threshold


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list,
) -> dict:
    """Compute all classification metrics and return as a serialisable dict.

    Includes standard ML metrics, clinical metrics (sensitivity, specificity,
    PPV, NPV), a threshold sweep, and the Youden's-J-optimal threshold.
    """
    # --- Standard metrics ---
    accuracy = float(accuracy_score(y_true, y_pred))
    roc_auc  = float(roc_auc_score(y_true, y_prob))

    precision_per = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    recall_per    = recall_score(y_true, y_pred,    average=None, zero_division=0).tolist()
    f1_per        = f1_score(y_true, y_pred,        average=None, zero_division=0).tolist()

    precision_w = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_w    = float(recall_score(y_true, y_pred,    average="weighted", zero_division=0))
    f1_w        = float(f1_score(y_true, y_pred,        average="weighted", zero_division=0))

    # --- Clinical metrics at default threshold (0.5) ---
    cm_arr             = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp     = cm_arr.ravel()
    sensitivity        = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0   # recall for PNEUMONIA
    specificity        = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0   # recall for NORMAL
    ppv                = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0   # precision for PNEUMONIA
    npv                = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0   # precision for NORMAL

    # --- Threshold sweep ---
    sweep, optimal_threshold = _threshold_sweep(y_true, y_prob)
    opt = next(s for s in sweep if s["threshold"] == optimal_threshold)

    return {
        "accuracy": accuracy,
        "roc_auc":  roc_auc,
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "f1": float(f)}
            for name, p, r, f in zip(class_names, precision_per, recall_per, f1_per)
        },
        "weighted_average": {
            "precision": precision_w,
            "recall":    recall_w,
            "f1":        f1_w,
        },
        "confusion_matrix": cm_arr.tolist(),
        "clinical": {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv":         ppv,
            "npv":         npv,
        },
        "threshold_analysis": {
            "optimal_threshold":   optimal_threshold,
            "optimal_sensitivity": opt["sensitivity"],
            "optimal_specificity": opt["specificity"],
            "optimal_f1":          opt["f1"],
            "sweep":               sweep,
        },
    }


def print_results(
    metrics: dict,
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    logger:  logging.Logger,
) -> None:
    """Log a human-readable summary of all evaluation results."""
    sep = "=" * 65
    logger.info("\n" + sep)
    logger.info(f"  Test Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    logger.info(f"  Weighted F1    : {metrics['weighted_average']['f1']:.4f}")
    logger.info(f"  Weighted Prec  : {metrics['weighted_average']['precision']:.4f}")
    logger.info(f"  Weighted Recall: {metrics['weighted_average']['recall']:.4f}")

    logger.info("\n  Per-class metrics:")
    for cls_name, m in metrics["per_class"].items():
        logger.info(
            f"    {cls_name:<12}: "
            f"Precision={m['precision']:.4f}  "
            f"Recall={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}"
        )

    clin = metrics["clinical"]
    logger.info("\n  Clinical Metrics (threshold = 0.50):")
    logger.info(f"    Sensitivity (recall for PNEUMONIA) : {clin['sensitivity']:.4f}")
    logger.info(f"    Specificity (recall for NORMAL)    : {clin['specificity']:.4f}")
    logger.info(f"    PPV (precision for PNEUMONIA)      : {clin['ppv']:.4f}")
    logger.info(f"    NPV (precision for NORMAL)         : {clin['npv']:.4f}")

    ta = metrics["threshold_analysis"]
    logger.info(f"\n  Optimal Threshold (Youden's J) : {ta['optimal_threshold']:.3f}")
    logger.info(f"    Sensitivity @ optimal : {ta['optimal_sensitivity']:.4f}")
    logger.info(f"    Specificity @ optimal : {ta['optimal_specificity']:.4f}")
    logger.info(f"    F1          @ optimal : {ta['optimal_f1']:.4f}")
    logger.info(sep)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    logger.info("\nClassification Report:\n" + report)

    cm_array = np.array(metrics["confusion_matrix"])
    logger.info("Confusion Matrix (rows = True label, cols = Predicted label):")
    header = " " * 12 + "  ".join(f"{n:>12}" for n in CLASS_NAMES)
    logger.info(header)
    for i, row in enumerate(cm_array):
        row_str = f"{CLASS_NAMES[i]:>12}  " + "  ".join(f"{v:>12}" for v in row)
        logger.info(row_str)


def evaluate(
    config:          dict,
    checkpoint_path: Path,
    logger:          logging.Logger,
    run_tag:         str,
) -> None:
    """Full evaluation pipeline: inference → metrics → all output files."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_path}'.\n"
            "Run training first:  python src/train.py --config config/config.yaml"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device     : {device}")
    logger.info(f"Checkpoint : {checkpoint_path}")
    logger.info(f"Run tag    : {run_tag}")

    model = load_model(config, checkpoint_path, device)
    _, _, test_loader = get_data_loaders(config)

    y_true, y_pred, y_prob = run_inference(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob, CLASS_NAMES)

    print_results(metrics, y_true, y_pred, logger)

    output_dir = Path(config["data"]["output_dir"])

    # Confusion matrix
    cm_path = output_dir / f"confusion_matrix_{run_tag}.png"
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        CLASS_NAMES,
        cm_path,
        title=f"Confusion Matrix — {run_tag}",
    )

    # ROC curve
    roc_path = output_dir / f"roc_curve_{run_tag}.png"
    plot_roc_curve(y_true, y_prob, roc_path, title=f"ROC Curve — {run_tag}")

    # Threshold sweep
    sweep_path = output_dir / f"threshold_sweep_{run_tag}.png"
    plot_threshold_sweep(
        metrics["threshold_analysis"]["sweep"],
        metrics["threshold_analysis"]["optimal_threshold"],
        sweep_path,
        title=f"Threshold Analysis — {run_tag}",
    )

    # JSON metrics (excludes verbose sweep list to keep file small)
    compact_metrics = {k: v for k, v in metrics.items() if k != "threshold_analysis"}
    compact_metrics["threshold_analysis"] = {
        k: v for k, v in metrics["threshold_analysis"].items() if k != "sweep"
    }
    metrics_path = output_dir / f"metrics_{run_tag}.json"
    save_metrics(compact_metrics, metrics_path)
    logger.info(f"\nMetrics saved          -> {metrics_path}")


def main() -> None:
    args       = parse_args()
    config     = load_config(args.config)
    output_dir = Path(config["data"]["output_dir"])
    model_name = config["model"]["name"]

    logger = setup_logging(output_dir, log_filename=f"evaluate_{model_name}.log")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        candidates = sorted(output_dir.glob(f"best_model_{model_name}_*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint found matching 'best_model_{model_name}_*.pth' in '{output_dir}'.\n"
                "Run training first:  python src/train.py --config config/config.yaml"
            )
        checkpoint_path = candidates[-1]

    evaluate(config, checkpoint_path, logger, model_name)


if __name__ == "__main__":
    main()
