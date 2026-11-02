"""
Grad-CAM visualization for the Chest X-Ray Pneumonia classifier.

Generates class activation heatmaps that highlight which regions of a chest
X-ray drive each prediction.  Saves a grid of (original X-ray, Grad-CAM
overlay) pairs grouped by prediction outcome: TP, TN, FP, FN.

Usage:
    python src/gradcam.py --config config/config.yaml
    python src/gradcam.py --config config/config.yaml \\
        --checkpoint outputs/best_model_improved_cnn_2026-05-13.pth \\
        --n_per_category 2
"""

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import CLASS_NAMES, get_data_loaders
from model import CustomCNN, ImprovedCustomCNN, ResNet18Model, get_model
from utils import load_config

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Grad-CAM core
# ---------------------------------------------------------------------------

class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Registers hooks on a target layer to capture activations and gradients
    during a forward-backward pass.  The heatmap highlights spatial regions
    most responsible for the model's predicted class score.

    Example:
        cam = GradCAM(model, get_target_layer(model))
        heatmap, pred = cam(image_tensor)
        cam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations:    Optional[torch.Tensor] = None
        self._gradients:      Optional[torch.Tensor] = None
        self._activation_ref: Optional[torch.Tensor] = None
        self._hooks = [
            target_layer.register_forward_hook(self._save_activation),
        ]

    def _save_activation(self, module, input, output) -> None:
        # retain_grad() lets us read .grad after backward without inserting
        # a custom autograd Function — avoids the inplace-view conflict that
        # register_full_backward_hook triggers on DenseNet / any model whose
        # forward applies relu_(…) immediately after the hooked layer.
        self._activations    = output.detach().clone()
        self._activation_ref = output
        if output.requires_grad:
            output.retain_grad()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Compute the Grad-CAM heatmap for a single image.

        Args:
            input_tensor: Shape (1, C, H, W), on the model's device.
            class_idx:    Class to explain; defaults to the predicted class.

        Returns:
            heatmap:  Float32 array (H, W) in [0, 1], resized to input dims.
            pred_idx: Predicted class index.
        """
        self.model.eval()
        output   = self.model(input_tensor)
        pred_idx = int(output.argmax(dim=1).item())
        target   = pred_idx if class_idx is None else class_idx

        self.model.zero_grad()
        output[0, target].backward()

        self._gradients = self._activation_ref.grad.detach()

        # Global-average-pool gradients → per-channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)    # (1, C, 1, 1)
        cam     = torch.relu((weights * self._activations).sum(dim=1))  # (1, h, w)
        cam     = cam.squeeze().cpu().numpy()                           # (h, w)

        if cam.max() > 0:
            cam /= cam.max()

        # Bilinear upsample to input spatial size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        ).astype(np.float32) / 255.0

        return cam, pred_idx

    def remove_hooks(self) -> None:
        """Clean up forward/backward hooks."""
        for hook in self._hooks:
            hook.remove()


def get_target_layer(model: nn.Module) -> nn.Module:
    """Return the Grad-CAM target layer for a supported model architecture."""
    if isinstance(model, ImprovedCustomCNN):
        # Last ResidualBlock in stage4 — output is (512, 7, 7)
        return model.stage4[-1]
    if isinstance(model, CustomCNN):
        # ReLU after the last conv in the feature extractor — output is (256, 14, 14)
        return model.features[18]
    if isinstance(model, ResNet18Model):
        # Last BasicBlock in layer4 — output is (512, 7, 7)
        return model.model.layer4[-1]
    if isinstance(model, DenseNet121Model):
        # Final batch norm after denseblock4 — output is (1024, 7, 7)
        return model.model.features.norm5
    if isinstance(model, EfficientNetModel):
        # Last Conv2dNormActivation block in features — output is (1536, 7, 7)
        return model.model.features[-1]
    raise ValueError(f"Grad-CAM target layer not defined for {type(model).__name__}")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation. Returns HWC float32 in [0, 1]."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)   # CHW → HWC
    return np.clip(img * _IMAGENET_STD + _IMAGENET_MEAN, 0.0, 1.0)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a Jet-colourmap heatmap over an RGB image.

    Args:
        image:   HWC float32 in [0, 1].
        heatmap: HW float32 in [0, 1].
        alpha:   Heatmap opacity (0 = original only, 1 = heatmap only).

    Returns:
        HWC float32 in [0, 1].
    """
    colormap = plt.cm.jet(heatmap)[:, :, :3]   # drop alpha channel
    return np.clip((1.0 - alpha) * image + alpha * colormap, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Inference + example collection
# ---------------------------------------------------------------------------

_CATEGORY_META = {
    "TP": ("True: PNEUMONIA", "Pred: PNEUMONIA", "✓", "green"),
    "TN": ("True: NORMAL",    "Pred: NORMAL",    "✓", "green"),
    "FP": ("True: NORMAL",    "Pred: PNEUMONIA", "✗", "crimson"),
    "FN": ("True: PNEUMONIA", "Pred: NORMAL",    "✗", "crimson"),
}

_Example = Tuple[torch.Tensor, int, int, float]  # (img, true, pred, confidence)


def collect_examples(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_per_category: int,
) -> Dict[str, List[_Example]]:
    """Run inference on the test set and bucket examples by prediction outcome.

    Iterates until each category (TP, TN, FP, FN) has n_per_category items,
    or the DataLoader is exhausted.

    Returns:
        Dict mapping category key → list of (image_tensor_cpu, true, pred, conf).
    """
    buckets: Dict[str, List[_Example]] = {"TP": [], "TN": [], "FP": [], "FN": []}

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Scanning test set"):
            if all(len(v) >= n_per_category for v in buckets.values()):
                break

            images = images.to(device, non_blocking=True)
            probs  = torch.softmax(model(images), dim=1)
            preds  = probs.argmax(dim=1)

            for i in range(images.size(0)):
                true = int(labels[i].item())
                pred = int(preds[i].item())
                conf = float(probs[i, pred].item())
                img  = images[i].cpu()

                if   true == 1 and pred == 1 and len(buckets["TP"]) < n_per_category:
                    buckets["TP"].append((img, true, pred, conf))
                elif true == 0 and pred == 0 and len(buckets["TN"]) < n_per_category:
                    buckets["TN"].append((img, true, pred, conf))
                elif true == 0 and pred == 1 and len(buckets["FP"]) < n_per_category:
                    buckets["FP"].append((img, true, pred, conf))
                elif true == 1 and pred == 0 and len(buckets["FN"]) < n_per_category:
                    buckets["FN"].append((img, true, pred, conf))

    return buckets


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_gradcam_grid(
    examples: Dict[str, List[_Example]],
    gradcam: GradCAM,
    device: torch.device,
    output_path: Path,
    model_name: str,
) -> None:
    """Generate and save a grid of (original X-ray, Grad-CAM overlay) pairs.

    Rows are ordered TP → TN → FP → FN.  Each row shows the raw image on
    the left and the heatmap overlay on the right.  Row titles are colour-coded
    green (correct) or red (incorrect).
    """
    rows: List[Tuple[str, _Example]] = []
    for cat in ("TP", "TN", "FP", "FN"):
        for ex in examples.get(cat, []):
            rows.append((cat, ex))

    if not rows:
        print("No examples collected — Grad-CAM grid not saved.")
        return

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 3.6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"Grad-CAM Visualisation — {model_name}", fontsize=14, fontweight="bold")

    for row_idx, (cat, (img_t, true_l, pred_l, conf)) in enumerate(rows):
        true_str, pred_str, outcome, color = _CATEGORY_META[cat]

        orig_np = denormalize(img_t)
        heatmap, _ = gradcam(img_t.unsqueeze(0).to(device))
        overlay = overlay_heatmap(orig_np, heatmap)

        ax_orig = axes[row_idx, 0]
        ax_cam  = axes[row_idx, 1]

        ax_orig.imshow(orig_np, cmap="gray" if orig_np.std() < 0.05 else None)
        ax_orig.axis("off")
        ax_orig.set_title(
            f"{true_str}  /  {pred_str}  ({conf * 100:.1f}%)  {outcome}",
            fontsize=9, color=color, pad=4,
        )

        ax_cam.imshow(overlay)
        ax_cam.axis("off")
        ax_cam.set_title("Grad-CAM Overlay", fontsize=9, pad=4)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grad-CAM grid saved -> {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for the pneumonia classifier")
    parser.add_argument("--config",         type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint",     type=str, default=None,
                        help="Path to .pth checkpoint. Defaults to latest in outputs/.")
    parser.add_argument("--n_per_category", type=int, default=2,
                        help="Examples per outcome category (TP/TN/FP/FN). Default: 2")
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

    checkpoint   = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config", config)
    model        = get_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_layer = get_target_layer(model)
    gradcam      = GradCAM(model, target_layer)

    _, _, test_loader = get_data_loaders(config)
    examples = collect_examples(model, test_loader, device, args.n_per_category)

    for cat, items in examples.items():
        print(f"  {cat}: {len(items)} example(s) collected")

    date_tag    = date.today().strftime("%Y-%m-%d")
    output_path = output_dir / "gradcam" / f"gradcam_{model_name}_{date_tag}.png"

    save_gradcam_grid(examples, gradcam, device, output_path, model_name)
    gradcam.remove_hooks()


if __name__ == "__main__":
    main()
