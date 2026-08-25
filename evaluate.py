"""Evaluate a trained model, create plots, and maintain a cross-model comparison table."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from data import CLASS_NAMES, LiverUltrasoundDataset, build_transforms, resolve_raw_root
from models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "efficientnet", "efficientnet_b0", "efficientnet_b3"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    root, raw_root = Path(__file__).resolve().parent, resolve_raw_root(Path(__file__).resolve().parent.parent / "7272660")
    checkpoint_path = args.checkpoint or root / "models" / args.model / "best_model.pt"
    if not checkpoint_path.exists(): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config["model_name"], freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    test_set = LiverUltrasoundDataset(root / "dataset/test", raw_root, build_transforms(224, False), config["use_roi_annotations"], config["roi_mode"])
    loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    targets, predictions, probabilities, elapsed = [], [], [], 0.0
    with torch.no_grad():
        for images, labels in loader:
            started = time.perf_counter(); scores = model(images.to(device)); elapsed += time.perf_counter() - started
            probs = torch.softmax(scores, 1).cpu().numpy()
            targets.extend(labels.numpy()); predictions.extend(probs.argmax(1)); probabilities.extend(probs)
    targets, predictions, probabilities = np.array(targets), np.array(predictions), np.array(probabilities)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, labels=range(3), zero_division=0)
    macro = precision_recall_fscore_support(targets, predictions, average="macro", zero_division=0)
    aucs = {name: float(roc_auc_score((targets == i).astype(int), probabilities[:, i])) for i, name in enumerate(CLASS_NAMES)}
    result = {"accuracy": float(accuracy_score(targets, predictions)), "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]), "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "auc": aucs[name]} for i, name in enumerate(CLASS_NAMES)}, "inference_ms_per_image": elapsed / len(targets) * 1000, "roi": {"enabled": config["use_roi_annotations"], "mode": config["roi_mode"]}}
    output_dir = checkpoint_path.parent
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); print("\nClassification report:\n", classification_report(targets, predictions, target_names=CLASS_NAMES, zero_division=0))
    fig, ax = plt.subplots(figsize=(5, 5)); ConfusionMatrixDisplay.from_predictions(targets, predictions, display_labels=CLASS_NAMES, cmap="Blues", ax=ax); fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(targets == i, probabilities[:, i]); ax.plot(fpr, tpr, label=f"{class_name} (AUC={aucs[class_name]:.3f})")
    ax.plot([0, 1], [0, 1], "k--"); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC curves"); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "roc_curves.png", dpi=160); plt.close(fig)
    comparison = root / "models" / "model_comparison.csv"; rows = []
    if comparison.exists(): rows = list(csv.DictReader(comparison.open(encoding="utf-8")))
    row = {"model": config["model_name"], "roi_mode": config["roi_mode"] if config["use_roi_annotations"] else "full_frame", "accuracy": result["accuracy"], "macro_f1": result["macro_f1"], "macro_recall": result["macro_recall"], "malignant_recall": result["per_class"]["Malignant"]["recall"], "macro_auc": float(np.mean(list(aucs.values()))), "inference_ms_per_image": result["inference_ms_per_image"]}
    rows = [old for old in rows if not (old["model"] == row["model"] and old["roi_mode"] == row["roi_mode"])] + [row]
    with comparison.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys()); writer.writeheader(); writer.writerows(rows)
    # A compact visual comparison is updated each time either architecture/ROI variant is evaluated.
    labels = [f"{item['model']}\n{item['roi_mode']}" for item in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, [float(item["macro_f1"]) for item in rows], label="Macro F1")
    axes[0].bar(labels, [float(item["malignant_recall"]) for item in rows], alpha=.65, label="Malignant recall")
    axes[0].set_ylim(0, 1); axes[0].set_title("Classification quality"); axes[0].legend()
    axes[1].bar(labels, [float(item["inference_ms_per_image"]) for item in rows], color="tab:orange")
    axes[1].set_title("Inference time per image (ms)")
    for axis in axes: axis.tick_params(axis="x", labelrotation=20)
    fig.tight_layout(); fig.savefig(root / "models" / "model_comparison.png", dpi=160); plt.close(fig)
    print(f"Saved metrics, confusion matrix, ROC curves, and comparison row to {output_dir}")


if __name__ == "__main__": main()
