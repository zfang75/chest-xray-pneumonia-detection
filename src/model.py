"""
CNN model definitions for Chest X-Ray Pneumonia classification.

Architectures available:
  - CustomCNN         : lightweight 5-block baseline trained from scratch
  - ImprovedCustomCNN : residual CNN with SE blocks, trained from scratch
  - ResNet18Model     : ResNet18 backbone with a fine-tuned classification head

Use get_model(config) to instantiate any of these from the YAML config.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Globally pools each feature map, runs a small FC bottleneck, then scales
    every channel by its learned importance weight.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    """Basic residual block: two 3×3 convs, optional projection shortcut, SE attention.

    When in_channels != out_channels or stride > 1, the shortcut uses a 1×1
    conv to match dimensions.  SE attention is applied before the residual add.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.se    = SEBlock(out_channels) if use_se else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.relu(out + residual)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class ImprovedCustomCNN(nn.Module):
    """Residual CNN with SE blocks, trained from scratch.

    Mirrors ResNet18's spatial progression (224→112→56→28→14→7→1) and
    parameter budget (~11M) using residual connections and channel attention,
    without relying on ImageNet pre-training.

    Stage layout:
        stem   : 7×7 conv stride-2 + MaxPool  → 64 ch, 56×56
        stage1 : 2× ResidualBlock(64→64)       → 56×56
        stage2 : 2× ResidualBlock(64→128, s=2) → 28×28
        stage3 : 2× ResidualBlock(128→256, s=2)→ 14×14
        stage4 : 2× ResidualBlock(256→512, s=2)→  7×7
        pool   : AdaptiveAvgPool → 1×1
        head   : Dropout → FC(512→256) → ReLU → Dropout → FC(256→classes)
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5) -> None:
        super().__init__()
        # 224×224 → 56×56
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # 56×56
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        # 56×56 → 28×28
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        # 28×28 → 14×14
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
        )
        # 14×14 → 7×7
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.classifier(x)


class CustomCNN(nn.Module):
    """Lightweight custom CNN baseline for binary chest X-ray classification.

    Five convolutional blocks with batch normalisation and max-pooling,
    followed by a fully-connected classifier with dropout regularisation.
    Accepts 224×224 RGB images and outputs raw logits for two classes.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — 224×224 → 112×112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2 — 112×112 → 56×56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3 — 56×56 → 28×28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 4 — 28×28 → 14×14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 5 — 14×14, adaptive pool → 4×4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # index 16
            nn.BatchNorm2d(256),
            nn.ReLU(),                                      # index 18 — Grad-CAM target
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ResNet18Model(nn.Module):
    """ResNet18 with a fine-tuned classification head for binary classification.

    Replaces the original 1000-class FC layer with a two-layer head that
    includes dropout.  When pretrained=True the backbone starts from ImageNet
    weights; when freeze_backbone=True only the head parameters are trained.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(config: dict) -> nn.Module:
    """Instantiate and return the model specified in config.

    Args:
        config: Parsed YAML config dict. Uses config['model']['name'],
                config['model']['num_classes'], and config['model']['pretrained'].

    Returns:
        An uninitialised (but potentially pre-trained) nn.Module ready for
        training.  Call .to(device) on the returned model before use.

    Raises:
        ValueError: If config['model']['name'] is not recognised.
    """
    model_name  = config["model"]["name"]
    num_classes = config["model"]["num_classes"]
    pretrained  = config["model"].get("pretrained", True)

    if model_name == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
    elif model_name == "improved_cnn":
        return ImprovedCustomCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        return ResNet18Model(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            "Valid options are: custom_cnn, improved_cnn, resnet18"
        )
