"""
src/models/architecture.py
===========================
Model definitions for Cats vs Dogs binary classification.

Includes:
  - SimpleCNN    : Baseline from-scratch CNN
  - MobileNetV2  : Transfer learning model (recommended)
  - ResNet18     : Alternative transfer learning model
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights


# ─── Baseline: Simple CNN ─────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    Baseline simple CNN for binary (cat/dog) classification.
    Input: (B, 3, 224, 224)
    Output: (B, 2) logits
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 224 → 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 112 → 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 56 → 28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)), # → 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ─── Transfer Learning: MobileNetV2 ──────────────────────────────────────────

class MobileNetV2Classifier(nn.Module):
    """
    Fine-tuned MobileNetV2 for binary classification.
    Lightweight and fast — recommended for production inference.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # Replace final classifier
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze feature extractor (useful for initial fine-tuning)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ─── Transfer Learning: ResNet18 ─────────────────────────────────────────────

class ResNet18Classifier(nn.Module):
    """Fine-tuned ResNet18 for binary classification."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


# ─── Model Factory ───────────────────────────────────────────────────────────

ARCHITECTURES = {
    "simple_cnn": SimpleCNN,
    "mobilenet_v2": MobileNetV2Classifier,
    "resnet18": ResNet18Classifier,
}


def build_model(
    architecture: str = "mobilenet_v2",
    num_classes: int = 2,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function to build a model by name.

    Args:
        architecture: One of 'simple_cnn', 'mobilenet_v2', 'resnet18'.
        num_classes: Number of output classes.
        dropout: Dropout probability.
        pretrained: Use ImageNet pretrained weights (transfer learning).

    Returns:
        Instantiated nn.Module.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: {list(ARCHITECTURES.keys())}"
        )

    model_cls = ARCHITECTURES[architecture]

    if architecture == "simple_cnn":
        return model_cls(num_classes=num_classes, dropout=dropout)
    else:
        return model_cls(num_classes=num_classes, dropout=dropout, pretrained=pretrained)


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
