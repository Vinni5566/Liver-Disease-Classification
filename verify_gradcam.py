"""Read-only Grad-CAM smoke test for final checkpoints using one fixed validation image."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data import LiverUltrasoundDataset, build_transforms, load_split_selection, resolve_raw_root
from models import build_model
from utils.gradcam import GradCAM, find_last_conv_layer


def module_name(model, module) -> str:
    """Return a human-readable module path for the selected target layer."""
    return next(name for name, candidate in model.named_modules() if candidate is module)


def verify_checkpoint(root: Path, model_name: str, checkpoint_relative_path: str) -> None:
    checkpoint_path = root / checkpoint_relative_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(model_name, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    selection = load_split_selection(root / "dataset" / "development_split.json", "validation")
    validation_set = LiverUltrasoundDataset(
        root / "dataset" / "train",
        resolve_raw_root(root.parent / "7272660"),
        build_transforms(224, training=False),
        selected_filenames=selection,
    )
    image, _label = validation_set[0]  # Fixed development validation image; never dataset/test.
    target_layer = find_last_conv_layer(model)
    heatmap, target_class, confidence = GradCAM(model, target_layer)(image.unsqueeze(0))
    forced_heatmap, forced_target_class, forced_confidence = GradCAM(model, target_layer)(
        image.unsqueeze(0), target_class=0
    )

    valid_heatmaps = (heatmap, forced_heatmap)
    if any(
        item.shape != (224, 224)
        or not np.isfinite(item).all()
        or not (0.0 <= item.min() <= item.max() <= 1.0)
        for item in valid_heatmaps
    ):
        raise RuntimeError("Grad-CAM heatmap failed shape, finite-value, or normalization checks.")
    if forced_target_class != 0:
        raise RuntimeError("Grad-CAM did not retain the explicitly requested target class.")
    print(
        f"{model_name}: target_layer={module_name(model, target_layer)}, "
        f"predicted_target={target_class}, confidence={confidence:.6f}, "
        f"explicit_target={forced_target_class}, explicit_confidence={forced_confidence:.6f}, "
        f"heatmap_shape={heatmap.shape}"
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    verify_checkpoint(project_root, "resnet50", "models/resnet50/staged_transfer/stage2_finetune/best_model.pt")
    verify_checkpoint(project_root, "efficientnet_b0", "models/efficientnet_b0/staged_transfer/stage2_finetune/best_model.pt")
