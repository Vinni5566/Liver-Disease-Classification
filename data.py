"""Datasets and transforms for AUL ultrasound classification."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["Normal", "Benign", "Malignant"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AddGaussianNoise:
    def __init__(self, std: float = 0.03):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0, 1)


def build_transforms(image_size: int, training: bool) -> transforms.Compose:
    stages = [transforms.Resize((image_size, image_size))]
    if training:
        stages += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.18, contrast=0.18),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # zoom
        ]
    stages += [transforms.ToTensor()]
    if training:
        stages += [AddGaussianNoise()]
    stages += [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(stages)


def liver_annotation_path(raw_root: Path, class_name: str, stem: str) -> Path:
    return raw_root / class_name / class_name / "segmentation" / "liver" / f"{stem}.json"


def resolve_raw_root(raw_root: Path) -> Path:
    """Support the supplied archive when it contains a redundant 7272660 wrapper."""
    raw_root = Path(raw_root)
    if (raw_root / "Normal" / "Normal" / "image").exists():
        return raw_root
    nested = raw_root / raw_root.name
    return nested if (nested / "Normal" / "Normal" / "image").exists() else raw_root


def apply_liver_roi(image: Image.Image, annotation_path: Path, mode: str) -> Image.Image:
    """Use native-pixel polygon coordinates to crop or mask an RGB ultrasound image."""
    if not annotation_path.exists():
        return image  # Missing annotations (known malignant gap) gracefully use full frame.
    with annotation_path.open(encoding="utf-8") as file:
        points = np.asarray(json.load(file), dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return image
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    polygon = np.round(points).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    if mode == "crop":
        x, y, crop_width, crop_height = cv2.boundingRect(polygon)
        return Image.fromarray(rgb[y:y + crop_height, x:x + crop_width])
    if mode == "mask":
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        return Image.fromarray(cv2.bitwise_and(rgb, rgb, mask=mask))
    raise ValueError("roi_mode must be 'crop' or 'mask'.")


class LiverUltrasoundDataset(Dataset):
    """Read the prepared split while resolving optional annotations back to raw AUL data."""
    def __init__(self, split_dir: Path, raw_root: Path, transform, use_roi_annotations=False, roi_mode="crop",
                 selected_filenames: Optional[Dict[str, set[str]]] = None):
        self.raw_root, self.transform = Path(raw_root), transform
        self.use_roi_annotations, self.roi_mode = use_roi_annotations, roi_mode
        self.samples: List[Tuple[Path, int]] = []
        for class_name in CLASS_NAMES:
            class_dir = Path(split_dir) / class_name
            for path in sorted(class_dir.glob("*")) if class_dir.exists() else []:
                is_selected = selected_filenames is None or path.name in selected_filenames.get(class_name, set())
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and is_selected:
                    self.samples.append((path, CLASS_TO_INDEX[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.use_roi_annotations:
            class_name = CLASS_NAMES[label]
            image = apply_liver_roi(image, liver_annotation_path(self.raw_root, class_name, image_path.stem), self.roi_mode)
        return self.transform(image), label


def class_weights(dataset: LiverUltrasoundDataset) -> torch.Tensor:
    labels = np.array([label for _, label in dataset.samples])
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return torch.tensor(len(labels) / (len(CLASS_NAMES) * counts), dtype=torch.float32)


def weighted_sampler(dataset: LiverUltrasoundDataset) -> WeightedRandomSampler:
    weights = class_weights(dataset)
    sample_weights = [float(weights[label]) for _, label in dataset.samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def create_development_split_manifest(train_dir: Path, test_dir: Path, manifest_path: Path,
                                      seed: int = 42, validation_fraction: float = 0.20) -> Dict:
    """Stratify only the fixed development split; never re-split or read from test for selection.

    The manifest stores filenames, so train and validation reuse the original dataset/train files
    without duplicating images. It also records a zero-overlap check against the held-out test set.
    """
    development_items = []
    test_keys = set()
    for class_name in CLASS_NAMES:
        for path in sorted((Path(train_dir) / class_name).glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                development_items.append((class_name, path.name))
        for path in sorted((Path(test_dir) / class_name).glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                test_keys.add((class_name, path.name))
    names = [f"{class_name}/{filename}" for class_name, filename in development_items]
    labels = [class_name for class_name, _ in development_items]
    train_names, validation_names = train_test_split(
        names, test_size=validation_fraction, random_state=seed, stratify=labels
    )
    train_keys = {(name.split("/", 1)[0], name.split("/", 1)[1]) for name in train_names}
    validation_keys = {(name.split("/", 1)[0], name.split("/", 1)[1]) for name in validation_names}
    overlaps = {
        "train_validation": len(train_keys & validation_keys),
        "train_test": len(train_keys & test_keys),
        "validation_test": len(validation_keys & test_keys),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Split overlap detected; refusing to write manifest: {overlaps}")
    manifest = {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "source_development_directory": "dataset/train",
        "held_out_test_directory": "dataset/test",
        "splits": {
            "train": {class_name: sorted(filename for label, filename in train_keys if label == class_name) for class_name in CLASS_NAMES},
            "validation": {class_name: sorted(filename for label, filename in validation_keys if label == class_name) for class_name in CLASS_NAMES},
            "test": {class_name: sorted(filename for label, filename in test_keys if label == class_name) for class_name in CLASS_NAMES},
        },
        "overlap_check": overlaps,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split_selection(manifest_path: Path, split_name: str) -> Dict[str, set[str]]:
    """Load a filename selection from the reproducible development-split manifest."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if split_name not in {"train", "validation"}:
        raise ValueError("split_name must be 'train' or 'validation'.")
    return {class_name: set(manifest["splits"][split_name][class_name]) for class_name in CLASS_NAMES}
