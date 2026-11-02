# Chest X-Ray Pneumonia Detection with CNN

A production-ready PyTorch pipeline for classifying chest X-rays as **NORMAL** or **PNEUMONIA**, with support for multiple architectures, Grad-CAM explanations, and MC Dropout uncertainty estimation.

---

## Project Overview

Pneumonia is a life-threatening infection that inflames the air sacs in one or both lungs. Early and accurate detection from chest X-rays can significantly improve patient outcomes. This project trains a deep learning classifier on the [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset and provides a clean, configurable pipeline for training, evaluation, interpretability, and uncertainty analysis.

---

## Dataset

~5,863 JPEG chest X-ray images (NORMAL and PNEUMONIA) organised into train/val/test splits. The training set has a roughly 3:1 PNEUMONIA:NORMAL class imbalance, which is handled automatically via `WeightedRandomSampler`.

### Download Instructions

1. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. Download and extract the dataset:
   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip -d data/
   ```

3. After extraction the layout should look like this:
   ```
   data/chest_xray/
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── val/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── test/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Project Structure

```
ML2_Group-Project/
├── config/
│   └── config.yaml                  # All hyperparameters and paths
├── src/
│   ├── train.py                     # Main training loop
│   ├── evaluate.py                  # Test-set evaluation and metrics
│   ├── gradcam.py                   # Grad-CAM visual explanations
│   ├── uncertainty.py               # MC Dropout uncertainty estimation
│   ├── model.py                     # Model architecture definitions
│   ├── dataset.py                   # Dataset, transforms, DataLoaders
│   └── utils.py                     # Config, logging, and plotting helpers
├── notebooks/
│   └── exploratory_analysis.ipynb   # EDA and dataset visualisation
├── outputs/                         # Generated artifacts (git-ignored)
├── data/                            # Downloaded dataset (git-ignored)
├── requirements.txt
└── README.md
```

---

## File Descriptions

### `config/config.yaml`

Central configuration file that controls all hyperparameters and paths. Every script reads this file via `--config`. Key options:

| Field | Default | Description |
|---|---|---|
| `model.name` | `improved_cnn` | Architecture: `custom_cnn`, `improved_cnn`, or `resnet18` |
| `model.pretrained` | `true` | Use ImageNet weights (ResNet18 only) |
| `training.epochs` | `25` | Number of training epochs |
| `training.batch_size` | `32` | Mini-batch size |
| `training.learning_rate` | `0.001` | Initial Adam learning rate |
| `training.scheduler` | `reduce_on_plateau` | LR scheduler: `step_lr` or `reduce_on_plateau` |
| `data.data_dir` | `data/chest_xray` | Path to the extracted dataset root |
| `data.output_dir` | `outputs` | Directory for checkpoints, logs, and plots |

---

### `src/model.py`

Defines three model architectures, all selectable via `model.name` in the config. A `get_model(config)` factory function instantiates the correct one.

| Architecture | Config Name | Params | Notes |
|---|---|---|---|
| `CustomCNN` | `custom_cnn` | ~2.5M | 5-block lightweight baseline, trains from scratch |
| `ImprovedCustomCNN` | `improved_cnn` | ~11M | Residual blocks + Squeeze-and-Excitation attention, trains from scratch |
| `ResNet18Model` | `resnet18` | ~11.2M | Standard ResNet18 with optional ImageNet pretraining, custom 2-class head |

Not run directly — imported by `train.py`, `evaluate.py`, `gradcam.py`, and `uncertainty.py`.

---

### `src/dataset.py`

Handles all data loading and augmentation. Reads images from the `data/chest_xray/` folder structure and returns PyTorch `DataLoader` objects.

- **Training transforms:** resize to 224×224 → optional horizontal flip → optional ±10° rotation → optional colour jitter → normalise (ImageNet mean/std)
- **Val/Test transforms:** resize to 224×224 → normalise only (no augmentation)
- **Class balancing:** `WeightedRandomSampler` ensures balanced mini-batches despite the 3:1 class ratio

Not run directly — imported by training and evaluation scripts.

---

### `src/utils.py`

Shared utility functions used across all scripts. Provides config parsing, seed setting, logging setup, JSON serialisation, and all plot generation (confusion matrix, training curves, ROC curve, threshold sweep).

Not run directly — imported by all other scripts.

---

### `src/train.py`

The main training loop. Instantiates the model and data loaders, trains for N epochs, and saves the best checkpoint.

```bash
python src/train.py --config config/config.yaml
```

**What it does:**
1. Validates the data directory and logs class distribution per split
2. Creates train/val/test DataLoaders with `WeightedRandomSampler` for class balancing
3. Trains with the Adam optimizer and a configurable LR scheduler
4. Saves a checkpoint whenever validation loss improves
5. Logs per-epoch metrics and generates training curves

**Expected console output:**
```
INFO - Using device: cuda
INFO - Class distribution:
INFO -   train -> NORMAL: 1341, PNEUMONIA: 3875
INFO -   val   -> NORMAL:    8, PNEUMONIA:    8
INFO -   test  -> NORMAL:  234, PNEUMONIA:  390
INFO - Starting training for 25 epochs
Epoch  1/25: 100%|████████| 163/163 [01:12<00:00]
INFO - Epoch 01 | train_loss=0.4231 train_acc=0.8012 | val_loss=0.3876 val_acc=0.8125 | lr=0.001000
INFO - New best model saved (val_loss=0.3876)
...
INFO - Training complete. Best val_loss=0.2314 at epoch 18
```

**Output files** (all in `outputs/`, named with model name and date):
| File | Description |
|---|---|
| `best_model_{model}_{date}.pth` | Self-contained checkpoint (model state, optimizer state, epoch, val metrics, full config) |
| `training_log_{model}_{date}.csv` | Per-epoch metrics: epoch, train_loss, train_acc, val_loss, val_acc, lr |
| `train_{model}_{date}.log` | Full training log (mirrors console output) |
| `training_curves_{model}_{date}.png` | Side-by-side loss and accuracy plots |

---

### `src/evaluate.py`

Runs inference on the held-out test set and computes comprehensive performance metrics, including clinical metrics relevant to medical diagnosis.

```bash
python src/evaluate.py --config config/config.yaml
```

To evaluate a specific checkpoint instead of auto-detecting the latest:
```bash
python src/evaluate.py --config config/config.yaml --checkpoint outputs/best_model_improved_cnn_2026-05-10.pth
```

**What it does:**
1. Loads the model and checkpoint
2. Runs full test-set inference
3. Computes standard metrics (accuracy, weighted F1, ROC-AUC, per-class precision/recall/F1), clinical metrics (sensitivity, specificity, PPV, NPV), and threshold analysis via Youden's J index
4. Saves a metrics JSON, visualisations, and a detailed log

**Expected console output:**
```
INFO - Evaluating on 624 test samples
INFO - Test Accuracy  : 0.8766
INFO - ROC-AUC        : 0.9612
INFO - Sensitivity    : 0.9282  (recall for PNEUMONIA)
INFO - Specificity    : 0.7906  (recall for NORMAL)
INFO - PPV            : 0.8913
INFO - NPV            : 0.8571
INFO - Optimal threshold (Youden's J): 0.4823
```

**Output files:**
| File | Description |
|---|---|
| `metrics_{model}.json` | All computed metrics in JSON format |
| `confusion_matrix_{model}.png` | Heatmap of TP/TN/FP/FN counts |
| `roc_curve_{model}.png` | ROC curve with AUC annotation |
| `threshold_sweep_{model}.png` | Sensitivity, specificity, and F1 vs. decision threshold |
| `evaluate_{model}.log` | Full evaluation log |

---

### `src/gradcam.py`

Generates Gradient-weighted Class Activation Maps (Grad-CAM) to visually explain what regions of an X-ray the model focused on when making a prediction. Examples are grouped by outcome (True Positive, True Negative, False Positive, False Negative).

```bash
python src/gradcam.py --config config/config.yaml
```

To specify a checkpoint or change how many examples per outcome category are shown:
```bash
python src/gradcam.py --config config/config.yaml --checkpoint outputs/best_model_improved_cnn_2026-05-10.pth --n_per_category 3
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `config/config.yaml` | Path to config |
| `--checkpoint` | auto-detect latest | Path to `.pth` checkpoint |
| `--n_per_category` | `2` | Number of examples per TP/TN/FP/FN category |

**What it does:**
1. Loads the trained model and hooks into the last convolutional layer
2. Runs inference on the test set and groups samples by outcome category
3. For each example, computes a Grad-CAM heatmap and overlays it on the original X-ray
4. Saves a grid image with paired columns (original | heatmap overlay)

**Expected console output:**
```
INFO - Running Grad-CAM on test set (624 samples)...
INFO - Collected: TP=8, TN=8, FP=2, FN=2
INFO - Grad-CAM grid saved to outputs/gradcam/gradcam_improved_cnn_2026-05-10.png
```

**Output files:**
| File | Description |
|---|---|
| `outputs/gradcam/gradcam_{model}_{date}.png` | Grid of original/heatmap overlay pairs, grouped by TP/TN/FP/FN |

---

### `src/uncertainty.py`

Estimates prediction uncertainty using Monte Carlo (MC) Dropout: dropout layers are kept active at inference time and N stochastic forward passes are run. The standard deviation across passes serves as an uncertainty score, revealing which predictions the model is least confident about.

```bash
python src/uncertainty.py --config config/config.yaml
```

To specify a checkpoint or number of stochastic passes:
```bash
python src/uncertainty.py --config config/config.yaml --checkpoint outputs/best_model_improved_cnn_2026-05-10.pth --n_passes 50
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `config/config.yaml` | Path to config |
| `--checkpoint` | auto-detect latest | Path to `.pth` checkpoint |
| `--n_passes` | `30` | Number of stochastic forward passes |

**What it does:**
1. Enables dropout layers while keeping BatchNorm in eval mode
2. Runs N forward passes per sample and computes mean and std of softmax outputs
3. Uses std as the uncertainty estimate
4. Generates visualisations comparing uncertainty on correct vs. incorrect predictions

**Expected console output:**
```
INFO - Running MC Dropout with 30 passes on 624 samples...
INFO - Uncertainty Summary:
INFO -   Correct predictions   (n=547): mean_uncertainty=0.0312
INFO -   Incorrect predictions (n=77):  mean_uncertainty=0.1084
INFO - High-uncertainty samples (>Q3): 156 total, 48 incorrect (30.8%)
INFO - Results saved to outputs/uncertainty/
```

**Output files:**
| File | Description |
|---|---|
| `outputs/uncertainty/uncertainty_{model}_{date}.png` | Three panels: uncertainty histogram, confidence vs. uncertainty scatter, and reliability/calibration diagram |
| `outputs/uncertainty/uncertainty_{model}_{date}.csv` | Per-sample table: true label, predicted label, mean probability, std (uncertainty), correctness flag |

---

### `notebooks/exploratory_analysis.ipynb`

Interactive Jupyter notebook for exploring the dataset before training. Run cells sequentially after downloading the dataset.

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

**What it covers:**
1. Class distribution bar plots per split (train/val/test)
2. Sample image visualisation (4 NORMAL, 4 PNEUMONIA from training set)
3. Image size distribution histograms
4. Pixel intensity statistics vs. ImageNet mean/std
5. Class imbalance analysis and `WeightedRandomSampler` explanation
6. Data augmentation preview (5 augmented versions of one image)
7. Summary table and next steps

All output is rendered inline in the notebook.

---

### `requirements.txt`

Python package dependencies. Install with:
```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---|---|---|
| `torch` | >=2.1.0 | Core deep learning framework |
| `torchvision` | >=0.16.0 | Model zoo (ResNet18) and image transforms |
| `numpy` | >=1.24.0 | Numerical computation |
| `pandas` | >=2.0.0 | CSV logging and data manipulation |
| `scikit-learn` | >=1.3.0 | Metrics: accuracy, F1, ROC-AUC, confusion matrix |
| `matplotlib` | >=3.7.0 | Plot generation |
| `seaborn` | >=0.12.0 | Styled heatmaps |
| `pyyaml` | >=6.0 | YAML config parsing |
| `tqdm` | >=4.65.0 | Progress bars |
| `Pillow` | >=10.0.0 | Image I/O |

---

## Full Run Walkthrough

Run these commands in order for a complete experiment:

```bash
# 1. Setup environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Download dataset (requires Kaggle credentials)
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/

# 3. (Optional) Explore the dataset
jupyter notebook notebooks/exploratory_analysis.ipynb

# 4. Train the model
python src/train.py --config config/config.yaml

# 5. Evaluate on the test set
python src/evaluate.py --config config/config.yaml

# 6. Generate Grad-CAM explanations
python src/gradcam.py --config config/config.yaml --n_per_category 3

# 7. Estimate prediction uncertainty
python src/uncertainty.py --config config/config.yaml --n_passes 30
```

After running steps 4–7, the `outputs/` directory will contain:
```
outputs/
├── best_model_{model}_{date}.pth
├── training_log_{model}_{date}.csv
├── training_curves_{model}_{date}.png
├── train_{model}_{date}.log
├── metrics_{model}.json
├── confusion_matrix_{model}.png
├── roc_curve_{model}.png
├── threshold_sweep_{model}.png
├── evaluate_{model}.log
├── gradcam/
│   └── gradcam_{model}_{date}.png
└── uncertainty/
    ├── uncertainty_{model}_{date}.png
    └── uncertainty_{model}_{date}.csv
```

---

## Results

| Model | Accuracy | Weighted F1 | ROC-AUC |
|---|---|---|---|
| Custom CNN | 0.81 | 0.79 | 0.94 |
| Improved CNN | 0.86 | 0.86 | 0.95 |
| ResNet18 (fine-tuned) | 0.88 | 0.88 | 0.96 |

---

## Key Design Choices

- **Class imbalance:** `WeightedRandomSampler` oversamples NORMAL images so each batch is balanced. The 3:1 PNEUMONIA:NORMAL ratio in training is handled at the sampler level, not the loss level.
- **Augmentation:** Applied only to the training set — random horizontal flip, ±10° rotation, colour jitter. Val/test sets receive only resize and normalise.
- **Normalisation:** ImageNet mean/std for all models, enabling effective transfer learning even for architectures trained from scratch.
- **Scheduler:** `ReduceLROnPlateau` (default) or `StepLR`, both configurable via YAML.
- **Checkpointing:** Each `.pth` file stores the model state, optimiser state, epoch, validation metrics, and the full config, making checkpoints fully self-contained for reproducibility.
- **Reproducibility:** Seed 42 is applied to Python, NumPy, PyTorch, and CUDA at the start of each script.
- **Interpretability:** Grad-CAM targets the last convolutional layer of each architecture, giving spatially meaningful activation maps regardless of which model is used.
- **Uncertainty:** MC Dropout uses the model's existing dropout layers — no architectural changes needed. Only dropout is left active; BatchNorm remains in eval mode to avoid noise from batch statistics.
