
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


# Define the three classification categories and their numerical labels.
CLASS_NAMES = ["Normal", "Benign", "Malignant"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

# ImageNet mean and standard deviation used for image normalization.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# Gaussian Noise Addition
class AddGaussianNoise:
    def __init__(self, std: float = 0.03):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Add random Gaussian noise to introduce variations in training images.
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0, 1)


# Image Preprocessing and Data Augmentation
def build_transforms(image_size: int, training: bool) -> transforms.Compose:
    # Resize images to the required input dimensions.
    stages = [transforms.Resize((image_size, image_size))]

    if training:
        stages += [
            # Randomly flip images horizontally.
            transforms.RandomHorizontalFlip(),

            # Apply random rotations up to 15 degrees.
            transforms.RandomRotation(15),

            # Introduce brightness and contrast variations.
            transforms.ColorJitter(brightness=0.18, contrast=0.18),

            # Apply random zooming to images.
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        ]

    # Convert images into PyTorch tensors.
    stages += [transforms.ToTensor()]

    if training:
        # Apply Gaussian noise only during training.
        stages += [AddGaussianNoise()]

    # Normalize images using ImageNet mean and standard deviation.
    stages += [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    return transforms.Compose(stages)


# Locate the ROI Annotation File
def liver_annotation_path(raw_root: Path, class_name: str, stem: str) -> Path:
    return raw_root / class_name / class_name / "segmentation" / "liver" / f"{stem}.json"


# Resolve the Correct Dataset Directory
def resolve_raw_root(raw_root: Path) -> Path:
    """Support the supplied archive when it contains a redundant 7272660 wrapper."""
    raw_root = Path(raw_root)

    if (raw_root / "Normal" / "Normal" / "image").exists():
        return raw_root

    nested = raw_root / raw_root.name
    return nested if (nested / "Normal" / "Normal" / "image").exists() else raw_root


# ROI Annotation Validation, Cropping and Masking
def apply_liver_roi(image: Image.Image, annotation_path: Path, mode: str) -> Image.Image:
    """Use native-pixel polygon coordinates to crop or mask an RGB ultrasound image."""

    # Use the full image if its ROI annotation is missing.
    if not annotation_path.exists():
        return image

    # Load polygon coordinates from the ROI annotation file.
    with annotation_path.open(encoding="utf-8") as file:
        points = np.asarray(json.load(file), dtype=np.float32)

    # Validate the polygon coordinates before processing.
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return image

    # Convert the image into RGB format.
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]

    # Round and restrict polygon coordinates to image boundaries.
    polygon = np.round(points).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)

    if mode == "crop":
        # Crop the image around the liver ROI bounding rectangle.
        x, y, crop_width, crop_height = cv2.boundingRect(polygon)
        return Image.fromarray(rgb[y:y + crop_height, x:x + crop_width])

    if mode == "mask":
        # Create a mask to retain only the liver ROI.
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        return Image.fromarray(cv2.bitwise_and(rgb, rgb, mask=mask))

    raise ValueError("roi_mode must be 'crop' or 'mask'.")


# AUL Dataset Loading and Image Preparation
class LiverUltrasoundDataset(Dataset):
    """Read the prepared split while resolving optional annotations back to raw AUL data."""

    def __init__(self, split_dir: Path, raw_root: Path, transform, use_roi_annotations=False, roi_mode="crop",
                 selected_filenames: Optional[Dict[str, set[str]]] = None):

        self.raw_root, self.transform = Path(raw_root), transform
        self.use_roi_annotations, self.roi_mode = use_roi_annotations, roi_mode
        self.samples: List[Tuple[Path, int]] = []

        # Collect image paths and assign their corresponding class labels.
        for class_name in CLASS_NAMES:
            class_dir = Path(split_dir) / class_name
            for path in sorted(class_dir.glob("*")) if class_dir.exists() else []:
                is_selected = selected_filenames is None or path.name in selected_filenames.get(class_name, set())

                if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and is_selected:
                    self.samples.append((path, CLASS_TO_INDEX[class_name]))

    def __len__(self):
        # Return the total number of images in the dataset.
        return len(self.samples)

    def __getitem__(self, index: int):
        # Load the image and its corresponding class label.
        image_path, label = self.samples[index]

        # Convert the image into three-channel RGB format.
        image = Image.open(image_path).convert("RGB")

        if self.use_roi_annotations:
            # Apply the corresponding liver ROI annotation when enabled.
            class_name = CLASS_NAMES[label]
            image = apply_liver_roi(
                image,
                liver_annotation_path(self.raw_root, class_name, image_path.stem),
                self.roi_mode
            )

        # Apply preprocessing and return the image tensor with its label.
        return self.transform(image), label


# Class Weight Calculation for Imbalanced Data
def class_weights(dataset: LiverUltrasoundDataset) -> torch.Tensor:
    # Extract labels and count the number of images in each class.
    labels = np.array([label for _, label in dataset.samples])
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))

    # Assign higher weights to classes with fewer images.
    return torch.tensor(len(labels) / (len(CLASS_NAMES) * counts), dtype=torch.float32)


# Handling Class Imbalance using WeightedRandomSampler
def weighted_sampler(dataset: LiverUltrasoundDataset) -> WeightedRandomSampler:
    # Calculate weights for each class.
    weights = class_weights(dataset)

    # Assign the corresponding class weight to every image.
    sample_weights = [float(weights[label]) for _, label in dataset.samples]

    # Sample images with replacement, giving minority classes a higher selection probability.
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# Creating Reproducible Training, Validation and Test Split Manifest
def create_development_split_manifest(train_dir: Path, test_dir: Path, manifest_path: Path,
                                     seed: int = 42, validation_fraction: float = 0.20) -> Dict:
    """Stratify only the fixed development split; never re-split or read from test for selection.

    The manifest stores filenames, so train and validation reuse the original dataset/train files
    without duplicating images. It also records a zero-overlap check against the held-out test set.
    """

    development_items = []
    test_keys = set()

    # Collect image filenames from the development training directory.
    for class_name in CLASS_NAMES:
        for path in sorted((Path(train_dir) / class_name).glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                development_items.append((class_name, path.name))

        # Collect filenames from the separate held-out test directory.
        for path in sorted((Path(test_dir) / class_name).glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                test_keys.add((class_name, path.name))

    names = [f"{class_name}/{filename}" for class_name, filename in development_items]
    labels = [class_name for class_name, _ in development_items]

    # Split the development data into training and validation sets
    # while maintaining the class distribution.
    train_names, validation_names = train_test_split(
        names, test_size=validation_fraction, random_state=seed, stratify=labels
    )

    train_keys = {(name.split("/", 1)[0], name.split("/", 1)[1]) for name in train_names}
    validation_keys = {(name.split("/", 1)[0], name.split("/", 1)[1]) for name in validation_names}

    # Check for filename overlap between training, validation and test sets.
    overlaps = {
        "train_validation": len(train_keys & validation_keys),
        "train_test": len(train_keys & test_keys),
        "validation_test": len(validation_keys & test_keys),
    }

    # Stop execution if any filename overlap is detected.
    if any(overlaps.values()):
        raise RuntimeError(f"Split overlap detected; refusing to write manifest: {overlaps}")

    # Store split filenames and reproducibility details in the manifest.
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

    # Create the output directory and save the split manifest.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


# Loading Training and Validation Split Selections
def load_split_selection(manifest_path: Path, split_name: str) -> Dict[str, set[str]]:
    """Load a filename selection from the reproducible development-split manifest."""

    # Read the saved dataset split manifest.
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    # Allow only training or validation split selection.
    if split_name not in {"train", "validation"}:
        raise ValueError("split_name must be 'train' or 'validation'.")

    # Return the selected filenames grouped by class.
    return {
        class_name: set(manifest["splits"][split_name][class_name])
        for class_name in CLASS_NAMES
    }
