"""
Chest X-Ray Pneumonia Classifier — Interactive Streamlit Dashboard

Three pages:
  1. Model Dashboard  — architecture comparison, metrics, training curves
  2. GradCAM Explorer — upload an X-ray and see what the model focuses on
  3. Diagnosis Game   — compete against the AI: can you out-diagnose it?

Models: CustomCNN · ImprovedCNN · ResNet18
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm_plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup — allow importing from src/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from model import get_model, CustomCNN, ImprovedCustomCNN, ResNet18Model  # noqa: E402
from gradcam import GradCAM, denormalize, overlay_heatmap  # noqa: E402
from utils import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = ROOT / "config" / "config.yaml"
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR    = ROOT / "data" / "chest_xray"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Per-model display metadata
MODEL_META = {
    "custom_cnn": {
        "display":      "CustomCNN",
        "badge":        "Baseline",
        "type":         "Vanilla CNN",
        "params":       "~2.5M",
        "pretrained":   "No",
        "gradcam_tgt":  "features[18]  (ReLU, 256 × 14 × 14)",
        "features":     "5-block CNN · BatchNorm · Dropout",
        "strengths":    "Fast inference, low memory, easy to interpret",
        "weaknesses":   "Limited capacity, no residual connections",
        "color":        "#636EFA",
    },
    "improved_cnn": {
        "display":      "ImprovedCNN",
        "badge":        "Best from-scratch",
        "type":         "Residual CNN + SE",
        "params":       "~11M",
        "pretrained":   "No",
        "gradcam_tgt":  "stage4[-1]  (ResidualBlock, 512 × 7 × 7)",
        "features":     "Residual connections · Squeeze-&-Excitation blocks · Channel attention",
        "strengths":    "Highest sensitivity (98.2 %), strong gradient flow, no ImageNet dependency",
        "weaknesses":   "Lower specificity, larger parameter count than CustomCNN",
        "color":        "#EF553B",
    },
    "resnet18": {
        "display":      "ResNet18",
        "badge":        "Best overall",
        "type":         "Transfer Learning",
        "params":       "~11.2M",
        "pretrained":   "ImageNet (1.2M images)",
        "gradcam_tgt":  "model.layer4[-1]  (BasicBlock, 512 × 7 × 7)",
        "features":     "Fine-tuned ResNet18 backbone · Custom 2-class head",
        "strengths":    "Best accuracy (88.9 %) and AUC (0.969), benefits from ImageNet pretraining",
        "weaknesses":   "Depends on pretrained weights, slightly lower NPV",
        "color":        "#00CC96",
    },
}

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_cached(model_name: str):
    """Load and cache a model from its latest checkpoint."""
    checkpoints = sorted(OUTPUTS_DIR.glob(f"best_model_{model_name}_*.pth"))
    if not checkpoints:
        return None, None
    ckpt_path = checkpoints[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # weights_only=False required because checkpoints include config dicts
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", load_config(str(CONFIG_PATH)))
    cfg["model"]["name"] = model_name

    model = get_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device


@st.cache_data
def load_all_metrics() -> dict:
    """Read metrics_*.json for every model and add derived clinical values."""
    out = {}
    for name in MODEL_META:
        path = OUTPUTS_DIR / f"metrics_{name}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        cm = data["confusion_matrix"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        data["derived"] = {
            "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "ppv":         tp / (tp + fp) if (tp + fp) else 0.0,
            "npv":         tn / (tn + fn) if (tn + fn) else 0.0,
        }
        out[name] = data
    return out


@st.cache_data
def load_training_logs() -> dict:
    """Load training_log_*.csv for every model."""
    logs = {}
    for name in MODEL_META:
        csvs = sorted(OUTPUTS_DIR.glob(f"training_log_{name}_*.csv"))
        if csvs:
            logs[name] = pd.read_csv(csvs[-1])
    return logs


@st.cache_data
def get_test_image_paths() -> list:
    """Return [(path_str, label_idx, label_name)] for all test images."""
    test_dir = DATA_DIR / "test"
    items = []
    for idx, name in enumerate(CLASS_NAMES):
        class_dir = test_dir / name
        if not class_dir.exists():
            continue
        for ext in ("*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"):
            for p in class_dir.glob(ext):
                items.append((str(p), idx, name))
    return items


# ---------------------------------------------------------------------------
# Image / model utilities
# ---------------------------------------------------------------------------

def _val_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])


def preprocess_pil(pil_img: Image.Image) -> torch.Tensor:
    """PIL → normalised tensor (1, 3, 224, 224)."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return _val_transform()(pil_img).unsqueeze(0)


def compute_gradcam(
    model,
    device,
    img_tensor: torch.Tensor,
    class_idx: Optional[int] = None,
    alpha: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Run GradCAM on a single image.
    Returns (orig_np, heatmap, overlay, pred_idx, probs).
    """
    img_tensor = img_tensor.to(device)

    # In-place ReLU (relu_) conflicts with PyTorch backward hooks because the
    # hook captures an activation tensor that gets overwritten before the backward
    # pass reads it. Temporarily disable inplace on every ReLU, then restore.
    relu_states = {
        m: m.inplace
        for m in model.modules()
        if isinstance(m, torch.nn.ReLU)
    }
    for m in relu_states:
        m.inplace = False

    # Use type name rather than isinstance so this works even when Streamlit
    # hot-reloads model.py and the cached model's class has a different identity
    # than the freshly-imported one.
    _type = type(model).__name__
    if _type == "ImprovedCustomCNN":
        target_layer = model.stage4[-1]
    elif _type == "CustomCNN":
        target_layer = model.features[18]
    elif _type == "ResNet18Model":
        target_layer = model.model.layer4[-1]
    else:
        raise ValueError(f"No GradCAM target layer defined for {_type}")
    cam = GradCAM(model, target_layer)

    heatmap, pred_idx = cam(img_tensor, class_idx=class_idx)
    cam.remove_hooks()

    for m, orig in relu_states.items():
        m.inplace = orig

    orig_np = denormalize(img_tensor.squeeze(0).cpu())
    overlay = overlay_heatmap(orig_np, heatmap, alpha=alpha)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    return orig_np, heatmap, overlay, pred_idx, probs


def mc_dropout_predict(
    model,
    device,
    img_tensor: torch.Tensor,
    n_passes: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MC Dropout uncertainty: enable dropout at inference, run n_passes forward
    passes, return (mean_probs, std_probs) each of shape (num_classes,).
    """
    model.train()   # activates dropout layers
    img_tensor = img_tensor.to(device)

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            probs = F.softmax(model(img_tensor), dim=1)[0].cpu().numpy()
            all_probs.append(probs)

    model.eval()
    arr = np.array(all_probs)
    return arr.mean(axis=0), arr.std(axis=0)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Float [0,1] HWC → uint8 HWC for st.image."""
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Page 1: Model Dashboard
# ---------------------------------------------------------------------------

def show_model_dashboard():
    st.title("Model Comparison Dashboard")
    st.markdown(
        "Side-by-side look at all three pneumonia classifiers: "
        "architecture, training dynamics, and evaluation metrics."
    )

    metrics = load_all_metrics()
    logs    = load_training_logs()

    if not metrics:
        st.error("No metrics JSON files found in outputs/. Run `python src/evaluate.py` first.")
        return

    # ── Architecture overview ──────────────────────────────────────────────
    st.header("Architecture Overview")

    arch_rows = []
    for name, meta in MODEL_META.items():
        arch_rows.append({
            "Model":        f"{meta['display']} ({meta['badge']})",
            "Type":         meta["type"],
            "Parameters":   meta["params"],
            "Pretrained":   meta["pretrained"],
            "Key Features": meta["features"],
        })
    st.dataframe(pd.DataFrame(arch_rows), use_container_width=True, hide_index=True)

    # ── Performance tabs ───────────────────────────────────────────────────
    st.header("Performance Metrics")
    tab_bar, tab_radar, tab_clinical = st.tabs(["Bar Charts", "Radar Chart", "Clinical Deep Dive"])

    # Build a tidy summary frame used across tabs
    summary_rows = []
    for name, meta in MODEL_META.items():
        if name not in metrics:
            continue
        m = metrics[name]
        d = m["derived"]
        summary_rows.append({
            "Model":       meta["display"],
            "_name":       name,
            "_color":      meta["color"],
            "Accuracy":    m["accuracy"],
            "Weighted F1": m["weighted_average"]["f1"],
            "ROC AUC":     m["roc_auc"],
            "Sensitivity": d["sensitivity"],
            "Specificity": d["specificity"],
            "PPV":         d["ppv"],
            "NPV":         d["npv"],
        })
    summary_df = pd.DataFrame(summary_rows)

    with tab_bar:
        col_l, col_r = st.columns(2)

        with col_l:
            fig = go.Figure()
            for _, row in summary_df.iterrows():
                fig.add_trace(go.Bar(
                    name=row["Model"],
                    x=["Accuracy", "Weighted F1", "ROC AUC"],
                    y=[row["Accuracy"], row["Weighted F1"], row["ROC AUC"]],
                    marker_color=row["_color"],
                ))
            fig.update_layout(
                barmode="group",
                title="Overall Performance",
                yaxis=dict(range=[0.75, 1.0], title="Score"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            fig2 = go.Figure()
            for _, row in summary_df.iterrows():
                fig2.add_trace(go.Bar(
                    name=row["Model"],
                    x=["Sensitivity", "Specificity", "PPV", "NPV"],
                    y=[row["Sensitivity"], row["Specificity"], row["PPV"], row["NPV"]],
                    marker_color=row["_color"],
                ))
            fig2.update_layout(
                barmode="group",
                title="Clinical Metrics",
                yaxis=dict(range=[0, 1.1], title="Score"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Numeric table
        display_cols = ["Model", "Accuracy", "Weighted F1", "ROC AUC", "Sensitivity", "Specificity", "PPV", "NPV"]
        st.dataframe(
            summary_df[display_cols].set_index("Model").style.format("{:.4f}"),
            use_container_width=True,
        )

    with tab_radar:
        categories = ["Accuracy", "Weighted F1", "ROC AUC", "Sensitivity", "Specificity", "PPV", "NPV"]
        fig = go.Figure()
        for _, row in summary_df.iterrows():
            vals = [row[c] for c in categories]
            vals.append(vals[0])   # close polygon
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill="toself",
                name=row["Model"],
                opacity=0.55,
                line=dict(color=row["_color"], width=2),
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1], tickformat=".0%")),
            title="Model Comparison — Radar View",
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_clinical:
        for name, meta in MODEL_META.items():
            if name not in metrics:
                continue
            m = metrics[name]
            d = m["derived"]
            cm = m["confusion_matrix"]
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

            with st.expander(
                f"**{meta['display']}** — {meta['badge']}",
                expanded=(name == "improved_cnn"),
            ):
                c = st.columns(4)
                c[0].metric("Sensitivity",  f"{d['sensitivity']:.1%}", help="Pneumonia recall (TPR)")
                c[1].metric("Specificity",  f"{d['specificity']:.1%}", help="Normal recall (TNR)")
                c[2].metric("PPV",          f"{d['ppv']:.1%}",         help="Pneumonia precision")
                c[3].metric("NPV",          f"{d['npv']:.1%}",         help="Normal precision")

                # Confusion matrix as plotly heatmap
                z    = [[tn, fp], [fn, tp]]
                xlbl = ["Pred: NORMAL", "Pred: PNEUMONIA"]
                ylbl = ["True: NORMAL", "True: PNEUMONIA"]
                ann  = [[f"TN={tn}", f"FP={fp}"], [f"FN={fn}", f"TP={tp}"]]

                hfig = go.Figure(go.Heatmap(
                    z=z, x=xlbl, y=ylbl,
                    colorscale="Blues",
                    text=ann,
                    texttemplate="%{text}",
                    showscale=False,
                ))
                hfig.update_layout(
                    title="Confusion Matrix",
                    height=260,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(hfig, use_container_width=True)

                if "threshold_analysis" in m:
                    ta = m["threshold_analysis"]
                    st.info(
                        f"**Optimal threshold (Youden's J):** {ta['optimal_threshold']:.2f}  →  "
                        f"Sensitivity {ta['optimal_sensitivity']:.1%} · "
                        f"Specificity {ta['optimal_specificity']:.1%} · "
                        f"F1 {ta['optimal_f1']:.3f}"
                    )

    # ── Training curves ────────────────────────────────────────────────────
    st.header("Training History")

    if not logs:
        st.warning("No training logs found in outputs/.")
    else:
        t_loss, t_acc, t_all = st.tabs(["Loss", "Accuracy", "Side-by-Side"])

        def _add_model_traces(fig, col, show_legend=True):
            for name, meta in MODEL_META.items():
                if name not in logs:
                    continue
                df = logs[name]
                kw = dict(line_color=meta["color"])
                fig.add_trace(go.Scatter(
                    x=df["epoch"], y=df["train_loss" if col == "loss" else "train_acc"],
                    name=f"{meta['display']} train",
                    showlegend=show_legend, **kw,
                ), **({} if col in ("loss", "acc") else {"row": 1, "col": 1 if col == "loss" else 2}))
                fig.add_trace(go.Scatter(
                    x=df["epoch"], y=df["val_loss" if col == "loss" else "val_acc"],
                    name=f"{meta['display']} val",
                    line=dict(color=meta["color"], dash="dash"),
                    showlegend=show_legend,
                ), **({} if col in ("loss", "acc") else {"row": 1, "col": 1 if col == "loss" else 2}))

        with t_loss:
            fig = go.Figure()
            for name, meta in MODEL_META.items():
                if name not in logs:
                    continue
                df = logs[name]
                fig.add_trace(go.Scatter(
                    x=df["epoch"], y=df["train_loss"],
                    name=f"{meta['display']} train", line_color=meta["color"],
                ))
                fig.add_trace(go.Scatter(
                    x=df["epoch"], y=df["val_loss"],
                    name=f"{meta['display']} val",
                    line=dict(color=meta["color"], dash="dash"),
                ))
            fig.update_layout(
                title="Training & Validation Loss",
                xaxis_title="Epoch", yaxis_title="Loss", height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        with t_acc:
            fig = go.Figure()
            for name, meta in MODEL_META.items():
                if name not in logs:
                    continue
                df = logs[name]
                fig.add_trace(go.Scatter(
                    x=df["epoch"], y=df["train_acc"],
                    name=f"{meta['display']} train", line_color=meta["color"],
                ))
                fig.add_trace(go.Scatter(
                    x=df["epoch"], y=df["val_acc"],
                    name=f"{meta['display']} val",
                    line=dict(color=meta["color"], dash="dash"),
                ))
            fig.update_layout(
                title="Training & Validation Accuracy (%)",
                xaxis_title="Epoch", yaxis_title="Accuracy (%)", height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        with t_all:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Loss", "Accuracy (%)"],
            )
            for name, meta in MODEL_META.items():
                if name not in logs:
                    continue
                df = logs[name]
                kw = dict(line_color=meta["color"])
                fig.add_trace(
                    go.Scatter(x=df["epoch"], y=df["train_loss"],
                               name=f"{meta['display']}", showlegend=True, **kw),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(x=df["epoch"], y=df["val_loss"],
                               name=f"{meta['display']} val",
                               showlegend=False,
                               line=dict(color=meta["color"], dash="dash")),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(x=df["epoch"], y=df["train_acc"],
                               name=f"{meta['display']}",
                               showlegend=False, **kw),
                    row=1, col=2,
                )
                fig.add_trace(
                    go.Scatter(x=df["epoch"], y=df["val_acc"],
                               name=f"{meta['display']} val",
                               showlegend=False,
                               line=dict(color=meta["color"], dash="dash")),
                    row=1, col=2,
                )
            fig.update_layout(height=420, title_text="Training Overview (solid = train, dashed = val)")
            st.plotly_chart(fig, use_container_width=True)

    # ── Architecture deep-dive ─────────────────────────────────────────────
    st.header("Architecture Deep Dive")
    chosen = st.selectbox(
        "Select a model to inspect",
        list(MODEL_META.keys()),
        format_func=lambda k: MODEL_META[k]["display"],
    )
    meta = MODEL_META[chosen]
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"**Type:** {meta['type']}")
        st.markdown(f"**Parameters:** {meta['params']}")
        st.markdown(f"**Pretrained on:** {meta['pretrained']}")
        st.markdown(f"**Key Features:** {meta['features']}")
        st.markdown(f"**GradCAM Target:** `{meta['gradcam_tgt']}`")
    with col_r:
        st.markdown(f"**Strengths:** {meta['strengths']}")
        st.markdown(f"**Weaknesses:** {meta['weaknesses']}")


# ---------------------------------------------------------------------------
# Page 2: GradCAM Explorer
# ---------------------------------------------------------------------------

def show_gradcam_explorer():
    st.title("GradCAM Explorer")
    st.markdown(
        "Pick a model, load a chest X-ray, and see exactly **which regions drive the prediction**. "
        "Adjust opacity and target class in real time."
    )

    # ── Sidebar settings ───────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("GradCAM Settings")

    model_name = st.sidebar.selectbox(
        "Model",
        list(MODEL_META.keys()),
        format_func=lambda k: MODEL_META[k]["display"],
        key="gc_model",
    )

    target_raw = st.sidebar.radio(
        "Explain for class",
        ["Predicted (auto)", "NORMAL", "PNEUMONIA"],
        key="gc_class",
    )
    class_idx: Optional[int] = (
        None if target_raw == "Predicted (auto)"
        else CLASS_NAMES.index(target_raw)
    )

    alpha     = st.sidebar.slider("Heatmap opacity", 0.1, 0.9, 0.45, 0.05, key="gc_alpha")
    n_passes  = st.sidebar.slider("MC Dropout passes", 5, 50, 20, 5, key="gc_passes")
    st.sidebar.caption("MC Dropout runs the model multiple times with dropout active to measure prediction uncertainty.")

    # ── Image source ───────────────────────────────────────────────────────
    src = st.radio("Image source", ["Upload my own X-ray", "Pick from test set"], horizontal=True)

    pil_img    = None
    true_label = None

    if src == "Upload my own X-ray":
        up = st.file_uploader("Upload chest X-ray (JPG / PNG)", type=["jpg", "jpeg", "png"])
        if up:
            pil_img = Image.open(up).convert("RGB")
    else:
        test_paths = get_test_image_paths()
        if not test_paths:
            st.error("Test images not found at data/chest_xray/test/. Check your data directory.")
            return

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            if st.button("Random image"):
                st.session_state["gc_img"] = random.choice(test_paths)

        if "gc_img" not in st.session_state:
            st.session_state["gc_img"] = random.choice(test_paths)

        path, lbl_idx, lbl_name = st.session_state["gc_img"]
        pil_img    = Image.open(path).convert("RGB")
        true_label = (lbl_idx, lbl_name)

    if pil_img is None:
        st.info("Upload an image or click **Random image** to get started.")
        return

    # Preview
    col_prev, col_info = st.columns([1, 2])
    with col_prev:
        st.image(pil_img, caption="Input X-ray", width=280)
    with col_info:
        if true_label:
            st.markdown(f"**Ground truth:** `{true_label[1]}`")
        st.markdown(f"**Model:** {MODEL_META[model_name]['display']}")
        st.markdown(f"**GradCAM target:** `{MODEL_META[model_name]['gradcam_tgt']}`")
        st.markdown(f"**Explain for:** {target_raw}")
        st.markdown(f"**Heatmap opacity:** {alpha}")

    if not st.button("Run GradCAM Analysis", type="primary", use_container_width=True):
        return

    model, device = load_model_cached(model_name)
    if model is None:
        st.error(f"No checkpoint found for **{model_name}**. Train the model first.")
        return

    img_tensor = preprocess_pil(pil_img)

    with st.spinner("Computing GradCAM heatmap…"):
        orig_np, heatmap, overlay, pred_idx, probs = compute_gradcam(
            model, device, img_tensor, class_idx=class_idx, alpha=alpha
        )

    pred_class = CLASS_NAMES[pred_idx]
    pred_conf  = probs[pred_idx]

    # ── Verdict banner ─────────────────────────────────────────────────────
    st.markdown("---")
    if true_label is not None:
        correct = pred_idx == true_label[0]
        if correct:
            st.success(
                f"Correct — Ground truth: `{true_label[1]}` → "
                f"Predicted: `{pred_class}` ({pred_conf:.1%})"
            )
        else:
            st.error(
                f"Incorrect — Ground truth: `{true_label[1]}` → "
                f"Predicted: `{pred_class}` ({pred_conf:.1%})"
            )
    else:
        st.info(f"**Prediction:** `{pred_class}` ({pred_conf:.1%})")

    # Probability bars
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.metric("P(NORMAL)",    f"{probs[0]:.1%}")
        st.progress(float(probs[0]))
    with pcol2:
        st.metric("P(PNEUMONIA)", f"{probs[1]:.1%}")
        st.progress(float(probs[1]))

    # ── Three-panel visualisation ──────────────────────────────────────────
    st.subheader("Visualisation")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Original X-ray**")
        st.image(to_uint8(orig_np), use_container_width=True)

    with c2:
        st.markdown("**GradCAM Heatmap**")
        hmap_colored = cm_plt.jet(heatmap)[:, :, :3]
        st.image(to_uint8(hmap_colored), use_container_width=True)
        st.caption("Hot = high importance · Cool = low importance")

    with c3:
        st.markdown("**Overlay**")
        st.image(to_uint8(overlay), use_container_width=True)

    # ── MC Dropout uncertainty ─────────────────────────────────────────────
    st.subheader("Prediction Uncertainty (MC Dropout)")

    with st.spinner(f"Running {n_passes} stochastic forward passes…"):
        mean_probs, std_probs = mc_dropout_predict(model, device, img_tensor, n_passes=n_passes)

    fig = go.Figure(go.Bar(
        x=CLASS_NAMES,
        y=mean_probs.tolist(),
        error_y=dict(type="data", array=std_probs.tolist(), visible=True),
        marker_color=["#636EFA", "#EF553B"],
        text=[f"{p:.1%} ± {s:.1%}" for p, s in zip(mean_probs, std_probs)],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"MC Dropout — {n_passes} passes  (bar = mean, whisker = std)",
        yaxis=dict(range=[0, 1.15], title="Probability"),
        xaxis_title="Class",
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)

    entropy     = float(-np.sum(mean_probs * np.log(mean_probs + 1e-9)))
    max_entropy = float(np.log(len(CLASS_NAMES)))
    ecol1, ecol2 = st.columns(2)
    ecol1.metric(
        "Prediction Entropy",
        f"{entropy:.3f}",
        help=f"Max possible entropy = {max_entropy:.3f} (uniform). Low = confident.",
    )
    ecol2.metric(
        "Normalised Uncertainty",
        f"{entropy / max_entropy:.1%}",
        help="0 % = perfectly certain · 100 % = maximally uncertain",
    )


# ---------------------------------------------------------------------------
# Page 3: Diagnosis Game
# ---------------------------------------------------------------------------

_GAME_ROUNDS = 10


def _init_game():
    """Populate session_state['game'] with a fresh 10-round game."""
    test_paths = get_test_image_paths()
    normals    = [(p, l, n) for p, l, n in test_paths if n == "NORMAL"]
    pneumonias = [(p, l, n) for p, l, n in test_paths if n == "PNEUMONIA"]

    n_each = _GAME_ROUNDS // 2
    sample = (
        random.sample(normals,    min(n_each, len(normals)))
        + random.sample(pneumonias, min(n_each, len(pneumonias)))
    )
    # If we couldn't get 5+5 just fill from the full pool
    while len(sample) < _GAME_ROUNDS and test_paths:
        sample.append(random.choice(test_paths))

    random.shuffle(sample)

    st.session_state["game"] = {
        "images":         sample[:_GAME_ROUNDS],
        "current":        0,
        "player_answers": [],
        "ai_answers":     [],
        "ai_probs":       [],
        "revealed":       False,
        "done":           False,
    }


def _ai_predict(pil_img: Image.Image):
    """Run ImprovedCNN inference, return (pred_idx, probs_list)."""
    model, device = load_model_cached("improved_cnn")
    if model is None:
        return 0, [0.5, 0.5]
    img_tensor = preprocess_pil(pil_img)
    with torch.no_grad():
        probs = F.softmax(model(img_tensor.to(device)), dim=1)[0].cpu().numpy()
    return int(probs.argmax()), probs.tolist()


def _show_final_score():
    game  = st.session_state["game"]
    imgs  = game["images"]
    n     = len(game["player_answers"])

    player_score = sum(
        game["player_answers"][i] == imgs[i][1] for i in range(n)
    )
    ai_score = sum(
        game["ai_answers"][i] == imgs[i][1] for i in range(n)
    )

    st.markdown("---")
    st.header("Final Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Your Score", f"{player_score} / {n}", f"{player_score / n:.0%}")
    c2.metric("AI Score",   f"{ai_score} / {n}",   f"{ai_score / n:.0%}")

    diff = player_score - ai_score
    if diff > 0:
        c3.metric("Winner", "You", f"+{diff} over the AI")
    elif diff < 0:
        c3.metric("Winner", "AI", f"+{-diff} over you")
    else:
        c3.metric("Winner", "Tie", "Dead even")

    # Round-by-round table
    st.subheader("Round-by-Round Breakdown")
    rows = []
    for i in range(n):
        _, true_idx, true_name = imgs[i]
        p_ans  = game["player_answers"][i]
        ai_ans = game["ai_answers"][i]
        ai_p   = game["ai_probs"][i]
        rows.append({
            "Round":        i + 1,
            "Ground Truth": true_name,
            "You":          f"{'✅' if p_ans == true_idx else '❌'} {CLASS_NAMES[p_ans]}",
            "AI":           f"{'✅' if ai_ans == true_idx else '❌'} {CLASS_NAMES[ai_ans]} ({max(ai_p):.0%})",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Round"), use_container_width=True)

    # Agreement chart
    both_right  = sum(game["player_answers"][i] == imgs[i][1] and game["ai_answers"][i] == imgs[i][1] for i in range(n))
    you_only    = sum(game["player_answers"][i] == imgs[i][1] and game["ai_answers"][i] != imgs[i][1] for i in range(n))
    ai_only     = sum(game["player_answers"][i] != imgs[i][1] and game["ai_answers"][i] == imgs[i][1] for i in range(n))
    both_wrong  = sum(game["player_answers"][i] != imgs[i][1] and game["ai_answers"][i] != imgs[i][1] for i in range(n))

    fig = go.Figure(go.Bar(
        x=["Both correct", "Only you", "Only AI", "Both wrong"],
        y=[both_right, you_only, ai_only, both_wrong],
        marker_color=["#00CC96", "#636EFA", "#EF553B", "#aaa"],
        text=[both_right, you_only, ai_only, both_wrong],
        textposition="outside",
    ))
    fig.update_layout(
        title="Diagnosis Agreement",
        yaxis=dict(range=[0, n + 1]),
        height=310,
    )
    st.plotly_chart(fig, use_container_width=True)

    avg_conf = np.mean([max(p) for p in game["ai_probs"]])
    st.caption(f"AI average confidence this game: **{avg_conf:.1%}**")


def show_diagnosis_game():
    st.title("Diagnose Like a Radiologist")
    st.markdown(
        "Think you can spot pneumonia? Go head-to-head with the AI — "
        f"{_GAME_ROUNDS} chest X-rays, no labels, no hints. "
        "The AI uses **ImprovedCNN** (our best from-scratch model)."
    )

    game = st.session_state.get("game")

    # ── Start / Restart screen ─────────────────────────────────────────────
    if game is None:
        st.markdown("""
        **How it works:**
        1. A chest X-ray appears — no label visible
        2. Click **Normal** or **Pneumonia**
        3. The AI reveals its verdict + confidence + GradCAM
        4. After 10 rounds, see who won!
        """)
        if st.button("Start Game", type="primary", use_container_width=True):
            _init_game()
            st.rerun()
        return

    # ── Final score screen ─────────────────────────────────────────────────
    if game["done"]:
        _show_final_score()
        if st.button("Play Again", type="primary", use_container_width=True):
            _init_game()
            st.rerun()
        return

    # ── Active game ────────────────────────────────────────────────────────
    total   = len(game["images"])
    current = game["current"]

    st.progress(current / total, text=f"Round {current + 1} of {total}")

    img_path, true_idx, true_name = game["images"][current]
    pil_img = Image.open(img_path).convert("RGB")

    if not game["revealed"]:
        # ── Await player input ─────────────────────────────────────────────
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.image(pil_img, caption="What's your diagnosis?", use_container_width=True)

        st.markdown("**Your diagnosis:**")
        b1, b2 = st.columns(2)

        def _submit(player_ans: int):
            ai_pred, ai_p = _ai_predict(pil_img)
            game["player_answers"].append(player_ans)
            game["ai_answers"].append(ai_pred)
            game["ai_probs"].append(ai_p)
            game["revealed"] = True

        with b1:
            if st.button("Normal", use_container_width=True, key=f"btn_normal_{current}"):
                _submit(0)
                st.rerun()
        with b2:
            if st.button("Pneumonia", use_container_width=True, key=f"btn_pneu_{current}"):
                _submit(1)
                st.rerun()

    else:
        # ── Reveal result ──────────────────────────────────────────────────
        player_ans = game["player_answers"][-1]
        ai_ans     = game["ai_answers"][-1]
        ai_prob    = game["ai_probs"][-1]

        p_correct  = player_ans == true_idx
        ai_correct = ai_ans == true_idx

        if p_correct and ai_correct:
            st.success("Both correct.")
        elif p_correct:
            st.success("You beat the AI on this one.")
        elif ai_correct:
            st.info("AI correct, you weren't.")
        else:
            st.warning("Both missed this one.")

        # Verdict columns
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            st.markdown("**Ground Truth**")
            if true_name == "PNEUMONIA":
                st.error(f"{true_name}")
            else:
                st.success(f"{true_name}")
        with vc2:
            icon = "✅" if p_correct else "❌"
            st.markdown("**Your Answer**")
            st.markdown(f"{icon} `{CLASS_NAMES[player_ans]}`")
        with vc3:
            icon = "✅" if ai_correct else "❌"
            st.markdown("**AI Answer**")
            st.markdown(f"{icon} `{CLASS_NAMES[ai_ans]}` ({max(ai_prob):.0%})")

        # GradCAM reveal
        with st.expander("AI Attention — GradCAM", expanded=True):
            model, device = load_model_cached("improved_cnn")
            if model:
                img_tensor = preprocess_pil(pil_img)
                with st.spinner("Rendering GradCAM…"):
                    orig_np, _, overlay, _, _ = compute_gradcam(model, device, img_tensor)
                gc1, gc2 = st.columns(2)
                with gc1:
                    st.image(to_uint8(orig_np), caption="Original", use_container_width=True)
                with gc2:
                    st.image(to_uint8(overlay), caption="AI Focus (GradCAM)", use_container_width=True)
                st.caption(
                    "Red/yellow = high model attention · "
                    "Pneumonia typically shows bilateral opacities in lower lung fields."
                )

        # Running score
        n_done       = len(game["player_answers"])
        player_score = sum(game["player_answers"][i] == game["images"][i][1] for i in range(n_done))
        ai_score     = sum(game["ai_answers"][i]     == game["images"][i][1] for i in range(n_done))

        sc1, sc2 = st.columns(2)
        sc1.metric("Your score", f"{player_score} / {n_done}")
        sc2.metric("AI score",   f"{ai_score} / {n_done}")

        # Advance
        if current + 1 < total:
            if st.button("Next X-ray", type="primary", use_container_width=True):
                game["current"] += 1
                game["revealed"] = False
                st.rerun()
        else:
            if st.button("See Final Results", type="primary", use_container_width=True):
                game["done"] = True
                st.rerun()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Pneumonia Classifier — AI Dashboard",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-size: 1.65rem !important;
        font-weight: 650 !important;
        margin-bottom: 0.2rem !important;
    }
    h2 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
    }
    h3 {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }
    [data-testid="metric-container"] {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
    }
    .stAlert {
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
    }
    div[data-testid="stSidebar"] {
        background: #f9fafb;
    }
    .streamlit-expanderHeader {
        font-size: 0.9rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("Pneumonia AI")
    st.sidebar.markdown("Chest X-Ray Classification Explorer")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["Model Dashboard", "GradCAM Explorer", "Diagnosis Game"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.caption(f"Running on: **{device_name}**")
    st.sidebar.caption("Models: CustomCNN · ImprovedCNN · ResNet18")
    st.sidebar.caption("Dataset: Kaggle Chest X-Ray (5,863 images)")

    if page == "Model Dashboard":
        show_model_dashboard()
    elif page == "GradCAM Explorer":
        show_gradcam_explorer()
    elif page == "Diagnosis Game":
        show_diagnosis_game()


if __name__ == "__main__":
    main()
