"""Create reproducible, validation-only Grad-CAM example figures for final checkpoints.

This script intentionally reads only the fixed development validation selection from
``dataset/train``.  It never reads the held-out test directory, retrains a model, or
updates checkpoints.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import CLASS_NAMES, LiverUltrasoundDataset, build_transforms, load_split_selection, resolve_raw_root  # noqa: E402
from models import build_model  # noqa: E402
from utils.gradcam import GradCAM, find_last_conv_layer, overlay_heatmap  # noqa: E402


@dataclass(frozen=True)
class Category:
    """A reproducible requested true/predicted-label condition."""

    slug: str
    true_index: int
    predicted_index: int
    correct: bool


@dataclass
class Prediction:
    """One validation prediction retained for deterministic example selection."""

    index: int
    image_path: Path
    true_index: int
    predicted_index: int
    confidence: float


CATEGORIES = (
    Category("correct_normal", 0, 0, True),
    Category("correct_benign", 1, 1, True),
    Category("correct_malignant", 2, 2, True),
    Category("benign_to_malignant", 1, 2, False),
    Category("malignant_to_benign", 2, 1, False),
    Category("normal_to_malignant", 0, 2, False),
)
CHECKPOINTS = {
    "resnet50": "models/resnet50/staged_transfer/stage2_finetune/best_model.pt",
    "efficientnet_b0": "models/efficientnet_b0/staged_transfer/stage2_finetune/best_model.pt",
}
CSV_FIELDS = (
    "model", "image_path", "true_class", "predicted_class", "confidence", "correct",
    "error_type", "target_class", "target_layer",
)


def module_name(model: torch.nn.Module, target_layer: torch.nn.Module) -> str:
    """Find the architecture-independent module path of a selected layer."""
    return next(name for name, module in model.named_modules() if module is target_layer)


def load_validation_dataset(root: Path, checkpoint_config: dict) -> LiverUltrasoundDataset:
    """Load only the already-fixed validation filenames from dataset/train.

    The manifest, rather than a new random split, guarantees that every rerun uses the
    same 118 validation candidates and preserves the original experimental split.
    """
    selection = load_split_selection(root / "dataset" / "development_split.json", "validation")
    return LiverUltrasoundDataset(
        root / "dataset" / "train",  # Deliberately not dataset/test.
        resolve_raw_root(root.parent / "7272660"),
        build_transforms(224, training=False),
        use_roi_annotations=checkpoint_config["use_roi_annotations"],
        roi_mode=checkpoint_config["roi_mode"],
        selected_filenames=selection,
    )


def collect_predictions(model: torch.nn.Module, dataset: LiverUltrasoundDataset, device: torch.device) -> list[Prediction]:
    """Run deterministic eval-mode inference across the fixed validation set only."""
    predictions: list[Prediction] = []
    model.eval()
    with torch.no_grad():
        for index, (image, true_index) in enumerate(dataset):
            probabilities = torch.softmax(model(image.unsqueeze(0).to(device)), dim=1)[0].cpu()
            confidence, predicted_index = probabilities.max(dim=0)
            image_path, sample_label = dataset.samples[index]
            if sample_label != true_index:
                raise RuntimeError("Validation dataset label order changed during inference.")
            predictions.append(Prediction(index, image_path, true_index, int(predicted_index), float(confidence)))
    return predictions


def select_representative(predictions: Iterable[Prediction], category: Category) -> Prediction | None:
    """Select the highest-confidence category match; filename breaks any exact tie.

    Correct examples therefore show a strong correct prediction and error examples show
    a strong incorrect prediction. Sorting by the stable source path makes selection
    reproducible even if equal confidence values occur.
    """
    matches = [
        item for item in predictions
        if item.true_index == category.true_index and item.predicted_index == category.predicted_index
    ]
    return min(matches, key=lambda item: (-item.confidence, item.image_path.as_posix())) if matches else None


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Render a normalized Grad-CAM heatmap as an RGB image."""
    colored_bgr = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def save_example_assets(category_dir: Path, source_path: Path, heatmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Save original, heatmap, and overlay while preserving the source image pixels."""
    category_dir.mkdir(parents=True, exist_ok=True)
    original = np.asarray(Image.open(source_path).convert("RGB"))
    heatmap_rgb = colorize_heatmap(heatmap)
    overlay = overlay_heatmap(original, heatmap)
    Image.fromarray(original).save(category_dir / "original.png")
    Image.fromarray(heatmap_rgb).save(category_dir / "heatmap.png")
    Image.fromarray(overlay).save(category_dir / "overlay.png")
    return original, overlay


def render_composite(output_path: Path, model_name: str, rows: list[dict]) -> None:
    """Render a consistent publication-ready six-category Original | Overlay layout."""
    figure, axes = plt.subplots(len(CATEGORIES), 2, figsize=(10, 4.0 * len(CATEGORIES)), squeeze=False)
    figure.suptitle(f"{model_name.replace('_', '-').title()} Stage-2 Grad-CAM examples (validation only)", fontsize=16, y=0.995)
    for row_index, category in enumerate(CATEGORIES):
        original_axis, overlay_axis = axes[row_index]
        record = next((item for item in rows if item["category"] == category.slug), None)
        if record is None:
            for axis, title in ((original_axis, "Original"), (overlay_axis, "Grad-CAM overlay")):
                axis.set_title(title, fontsize=12, fontweight="bold")
                axis.text(0.5, 0.5, f"Unavailable: {category.slug.replace('_', ' ')}", ha="center", va="center", fontsize=11)
                axis.axis("off")
            continue
        original_axis.imshow(record["original"])
        overlay_axis.imshow(record["overlay"])
        label = (
            f"Original\n{category.slug.replace('_', ' ').title()}\n"
            f"True: {record['true_class']} | Predicted: {record['predicted_class']}\n"
            f"Confidence: {record['confidence'] * 100:.1f}%"
        )
        original_axis.set_title(label, fontsize=9, fontweight="bold", loc="left", pad=6)
        overlay_axis.set_title("Grad-CAM Overlay", fontsize=10, fontweight="bold")
        original_axis.axis("off")
        overlay_axis.axis("off")
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_readme(output_root: Path, selected_rows: list[dict], unavailable_rows: list[dict]) -> None:
    """Document inputs and interpretation boundaries for reproducible research use."""
    selected_categories = ", ".join(sorted({row["category"] for row in selected_rows})) or "none"
    unavailable = ", ".join(f"{row['model']}:{row['category']}" for row in unavailable_rows) or "none"
    text = f"""# Validation-only Grad-CAM examples

This directory was generated by `scripts/generate_gradcam_examples.py` using only the
fixed validation selection in `dataset/development_split.json`. It does not read or
modify `dataset/test`, train models, or alter checkpoints.

## Models and checkpoints

- ResNet50 Stage 2: `models/resnet50/staged_transfer/stage2_finetune/best_model.pt`
- EfficientNet-B0 Stage 2: `models/efficientnet_b0/staged_transfer/stage2_finetune/best_model.pt`

## Selection

Requested categories are correct Normal, correct Benign, correct Malignant,
Benign → Malignant, Malignant → Benign, and Normal → Malignant. For every available
category, the script deterministically selects the highest-confidence matching
validation prediction; filename is the stable tie-breaker. Selected categories: {selected_categories}.
Unavailable categories: {unavailable}.

## Grad-CAM

`utils/gradcam.py` automatically selects the final `Conv2d` feature layer for each
architecture, computes a Grad-CAM map for the predicted class, and removes its hooks
after each image. Grad-CAM highlights image regions that contributed to that model
prediction. It is an explanation of model behaviour, not proof of a medically causal
lesion or a clinical diagnosis.

## Reproduction

From the project root, run:

```powershell
python scripts/generate_gradcam_examples.py
```
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    output_root = PROJECT_ROOT / "results" / "gradcam"
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_rows: list[dict] = []
    unavailable_rows: list[dict] = []

    for model_name, checkpoint_relative_path in CHECKPOINTS.items():
        checkpoint_path = PROJECT_ROOT / checkpoint_relative_path
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("class_names") != CLASS_NAMES:
            raise ValueError(f"Class mapping mismatch in {checkpoint_path}.")
        model = build_model(checkpoint["config"]["model_name"], freeze_backbone=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        target_layer = find_last_conv_layer(model)
        target_layer_name = module_name(model, target_layer)
        validation_set = load_validation_dataset(PROJECT_ROOT, checkpoint["config"])
        predictions = collect_predictions(model, validation_set, device)
        composite_rows: list[dict] = []

        for category in CATEGORIES:
            selected = select_representative(predictions, category)
            if selected is None:
                unavailable_rows.append({"model": model_name, "category": category.slug, "status": "unavailable"})
                continue
            # No target is passed: Grad-CAM defaults to the model prediction, the evidence
            # we want to explain for a correct or incorrect classification example.
            image_tensor, _ = validation_set[selected.index]
            heatmap, target_class, confidence = GradCAM(model, target_layer)(image_tensor.unsqueeze(0).to(device))
            if target_class != selected.predicted_index or not np.isclose(confidence, selected.confidence, rtol=0, atol=1e-6):
                raise RuntimeError("Grad-CAM prediction differs from deterministic inference result.")
            category_dir = output_root / model_name / category.slug
            original, overlay = save_example_assets(category_dir, selected.image_path, heatmap)
            error_type = "correct" if category.correct else category.slug
            row = {
                "model": model_name,
                "image_path": selected.image_path.relative_to(PROJECT_ROOT).as_posix(),
                "true_class": CLASS_NAMES[selected.true_index],
                "predicted_class": CLASS_NAMES[selected.predicted_index],
                "confidence": confidence,
                "correct": str(category.correct).lower(),
                "error_type": error_type,
                "target_class": CLASS_NAMES[target_class],
                "target_layer": target_layer_name,
                "category": category.slug,
                "original": original,
                "overlay": overlay,
            }
            csv_rows.append(row)
            composite_rows.append(row)

        render_composite(output_root / f"{model_name}_gradcam_examples.png", model_name, composite_rows)
        print(f"{model_name}: generated {len(composite_rows)} Grad-CAM examples; target layer={target_layer_name}")

    with (output_root / "gradcam_examples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in CSV_FIELDS} for row in csv_rows)
    with (output_root / "unavailable_categories.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("model", "category", "status"))
        writer.writeheader()
        writer.writerows(unavailable_rows)
    write_readme(output_root, csv_rows, unavailable_rows)


if __name__ == "__main__":
    main()
