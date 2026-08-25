"""Transfer-learning model factory for ResNet50 and EfficientNet variants."""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights, EfficientNet_B3_Weights, ResNet50_Weights,
    efficientnet_b0, efficientnet_b3, resnet50,
)


def build_model(model_name: str, num_classes: int = 3, freeze_backbone: bool = False) -> nn.Module:
    """Construct a pretrained CNN with GAP and a 3-logit classification head.

    CrossEntropyLoss consumes logits; softmax is applied only for displayed probabilities.
    Both torchvision backbones already include AdaptiveAvgPool2d (global average pooling).
    """
    name = model_name.lower().replace("-", "_")
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head = model.fc
    elif name in {"efficientnet", "efficientnet_b0"}:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        head = model.classifier[-1]
    elif name == "efficientnet_b3":
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        head = model.classifier[-1]
    else:
        raise ValueError("model_name must be resnet50, efficientnet_b0, or efficientnet_b3")
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in head.parameters():
            parameter.requires_grad = True
    return model
