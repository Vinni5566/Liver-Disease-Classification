"""Create final comparison tables and figures from saved evaluation artifacts only.

This script never imports datasets, data loaders, checkpoints, models, or evaluate.py.
It reads saved metric JSON files and already-recorded confusion matrices only.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "final_analysis"
FIGURES = OUTPUT / "figures"
CLASSES = ("Normal", "Benign", "Malignant")
MODELS = (
    {
        "name": "ResNet50",
        "metrics_path": ROOT / "models" / "resnet50" / "staged_transfer" / "stage2_finetune" / "evaluation_metrics.json",
        # Rows=true, columns=predicted; recorded during the completed held-out evaluation.
        "matrix": np.array([[9, 7, 4], [3, 20, 17], [2, 13, 72]], dtype=int),
    },
    {
        "name": "EfficientNet-B0",
        "metrics_path": ROOT / "models" / "efficientnet_b0" / "staged_transfer" / "stage2_finetune" / "evaluation_metrics.json",
        # Rows=true, columns=predicted; recorded during the completed held-out evaluation.
        "matrix": np.array([[16, 4, 0], [11, 10, 19], [6, 9, 72]], dtype=int),
    },
)


def load_results() -> list[dict]:
    """Read saved metrics only; never access test images or run inference."""
    return [{**spec, "metrics": json.loads(spec["metrics_path"].read_text(encoding="utf-8"))} for spec in MODELS]


def write_model_comparison(results: list[dict]) -> None:
    fields = ["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    fields += [f"{label} {metric}" for label in CLASSES for metric in ("Precision", "Recall", "F1", "AUC")]
    fields += ["Inference ms/image"]
    with (OUTPUT / "model_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in results:
            metric, per_class = item["metrics"], item["metrics"]["per_class"]
            row = {
                "Model": item["name"], "Accuracy": metric["accuracy"],
                "Macro Precision": metric["macro_precision"], "Macro Recall": metric["macro_recall"],
                "Macro F1": metric["macro_f1"], "Inference ms/image": metric["inference_ms_per_image"],
            }
            for label in CLASSES:
                for key, title in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("auc", "AUC")):
                    row[f"{label} {title}"] = per_class[label][key]
            writer.writerow(row)


def write_per_class_metrics(results: list[dict]) -> None:
    fields = ("Model", "Class", "Precision", "Recall", "F1", "AUC")
    with (OUTPUT / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in results:
            for label in CLASSES:
                values = item["metrics"]["per_class"][label]
                writer.writerow({"Model": item["name"], "Class": label, "Precision": values["precision"],
                                 "Recall": values["recall"], "F1": values["f1"], "AUC": values["auc"]})


def write_error_analysis(results: list[dict]) -> None:
    """Write each off-diagonal cell, pairwise totals, and malignant false negatives."""
    fields = ("Model", "Analysis", "True Class", "Predicted Class", "Count")
    with (OUTPUT / "confusion_error_analysis.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in results:
            matrix = item["matrix"]
            for true_idx, true_label in enumerate(CLASSES):
                for pred_idx, pred_label in enumerate(CLASSES):
                    if true_idx != pred_idx:
                        writer.writerow({"Model": item["name"], "Analysis": "Misclassification",
                                         "True Class": true_label, "Predicted Class": pred_label,
                                         "Count": int(matrix[true_idx, pred_idx])})
            for pair, first, second in (("Normal <-> Benign", 0, 1), ("Normal <-> Malignant", 0, 2),
                                        ("Benign <-> Malignant", 1, 2)):
                writer.writerow({"Model": item["name"], "Analysis": "Pairwise error total",
                                 "True Class": pair, "Predicted Class": "Both directions",
                                 "Count": int(matrix[first, second] + matrix[second, first])})
            for pred_idx, pred_label in ((0, "Normal"), (1, "Benign")):
                writer.writerow({"Model": item["name"], "Analysis": "Malignant false negative",
                                 "True Class": "Malignant", "Predicted Class": pred_label,
                                 "Count": int(matrix[2, pred_idx])})
            writer.writerow({"Model": item["name"], "Analysis": "Malignant false negative total",
                             "True Class": "Malignant", "Predicted Class": "non-Malignant",
                             "Count": int(matrix[2, 0] + matrix[2, 1])})


def style_axis(axis, ylabel: str) -> None:
    axis.set_ylabel(ylabel, fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)


def annotate(axis, bars, percent: bool = True) -> None:
    for bar in bars:
        value = bar.get_height()
        label = f"{value * 100:.1f}%" if percent else f"{value:.1f}"
        axis.annotate(label, (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 4),
                      textcoords="offset points", ha="center", va="bottom", fontsize=10)


def model_bar(results: list[dict], title: str, values: list[float], ylabel: str, filename: str,
              percent: bool = True, ymax: float | None = 1.0) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 5.0))
    bars = axis.bar([item["name"] for item in results], values, color=("#2A6FBB", "#E07A3F"), width=0.58)
    axis.set_title(title, fontweight="bold", fontsize=14)
    style_axis(axis, ylabel)
    if ymax is not None:
        axis.set_ylim(0, ymax)
    annotate(axis, bars, percent)
    figure.tight_layout()
    figure.savefig(FIGURES / filename, dpi=250, bbox_inches="tight")
    plt.close(figure)


def auc_figure(results: list[dict]) -> None:
    x, width = np.arange(3), 0.36
    figure, axis = plt.subplots(figsize=(7.6, 5.0))
    for offset, item, color in ((-width / 2, results[0], "#2A6FBB"), (width / 2, results[1], "#E07A3F")):
        bars = axis.bar(x + offset, [item["metrics"]["per_class"][label]["auc"] for label in CLASSES],
                        width, label=item["name"], color=color)
        annotate(axis, bars)
    axis.set_xticks(x, CLASSES)
    axis.set_ylim(0, 1)
    axis.set_title("One-vs-rest AUC by class", fontweight="bold", fontsize=14)
    style_axis(axis, "AUC")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES / "auc_comparison.png", dpi=250, bbox_inches="tight")
    plt.close(figure)


def error_figure(results: list[dict]) -> None:
    """Show pairwise error burden and clinically important malignant false negatives."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    width = 0.36
    specifications = (
        (axes[0], ("Normal <->\nBenign", "Normal <->\nMalignant", "Benign <->\nMalignant"),
         "Pairwise misclassification totals", lambda m: [m[0, 1] + m[1, 0], m[0, 2] + m[2, 0], m[1, 2] + m[2, 1]]),
        (axes[1], ("Malignant ->\nNormal", "Malignant ->\nBenign", "Malignant ->\nnon-Malignant"),
         "Malignant false negatives", lambda m: [m[2, 0], m[2, 1], m[2, 0] + m[2, 1]]),
    )
    for axis, labels, title, value_fn in specifications:
        x = np.arange(len(labels))
        for offset, item, color in ((-width / 2, results[0], "#2A6FBB"), (width / 2, results[1], "#E07A3F")):
            bars = axis.bar(x + offset, value_fn(item["matrix"]), width, label=item["name"], color=color)
            annotate(axis, bars, percent=False)
        axis.set_xticks(x, labels)
        axis.set_title(title, fontweight="bold", fontsize=13)
        style_axis(axis, "Images")
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES / "error_distribution.png", dpi=250, bbox_inches="tight")
    plt.close(figure)


def create_figures(results: list[dict]) -> None:
    metric = lambda key: [item["metrics"][key] for item in results]
    model_bar(results, "Held-out accuracy", metric("accuracy"), "Accuracy", "accuracy_comparison.png")
    model_bar(results, "Held-out macro F1", metric("macro_f1"), "Macro F1", "macro_f1_comparison.png")
    model_bar(results, "Held-out macro recall", metric("macro_recall"), "Macro recall", "macro_recall_comparison.png")
    model_bar(results, "Malignant recall",
              [item["metrics"]["per_class"]["Malignant"]["recall"] for item in results],
              "Recall", "malignant_recall_comparison.png")
    model_bar(results, "Malignant F1",
              [item["metrics"]["per_class"]["Malignant"]["f1"] for item in results],
              "F1", "malignant_f1_comparison.png")
    auc_figure(results)
    model_bar(results, "Inference time", metric("inference_ms_per_image"), "Milliseconds per image",
              "inference_time_comparison.png", percent=False, ymax=None)
    error_figure(results)


def write_summary(results: list[dict]) -> None:
    resnet, efficient = results
    r, e = resnet["metrics"], efficient["metrics"]
    summary = f"""# Final model comparison and error analysis

## Scope

This package was created from the two saved evaluation metrics JSON artifacts and their
already-recorded confusion matrices. No dataset images, checkpoints, models, training,
or evaluation code were accessed or run.

## Results-section-ready summary

On the held-out evaluation, ResNet50 Stage 2 achieved higher accuracy ({r['accuracy']:.3f} vs
{e['accuracy']:.3f}) and macro F1 ({r['macro_f1']:.3f} vs {e['macro_f1']:.3f}) than
EfficientNet-B0 under the final argmax decision rule. Both models had identical Malignant
recall (82.76%, 72/87). EfficientNet-B0 had slightly higher Malignant precision/F1
({e['per_class']['Malignant']['precision']:.3f}/{e['per_class']['Malignant']['f1']:.3f} vs
{r['per_class']['Malignant']['precision']:.3f}/{r['per_class']['Malignant']['f1']:.3f}) and
faster measured inference ({e['inference_ms_per_image']:.1f} vs {r['inference_ms_per_image']:.1f}
ms/image). The dominant error pattern for both models was Benign <-> Malignant confusion.

## Overall performance and selection rationale

ResNet50 has better accuracy ({r['accuracy']:.4f} vs {e['accuracy']:.4f}) and macro F1
({r['macro_f1']:.4f} vs {e['macro_f1']:.4f}), making it the preferred option when balanced
performance under this fixed argmax operating rule is primary. This does not make
ResNet50 universally superior: EfficientNet-B0 has higher macro recall
({e['macro_recall']:.4f} vs {r['macro_recall']:.4f}), higher Malignant precision/F1, and
substantially faster measured inference. Choice should remain dependent on intended
operating constraints and further validation.

## Per-class behaviour

- **Malignant:** Both models recalled 72 of 87 images (82.76%). EfficientNet-B0 had higher
  precision ({e['per_class']['Malignant']['precision']:.4f} vs {r['per_class']['Malignant']['precision']:.4f})
  and F1 ({e['per_class']['Malignant']['f1']:.4f} vs {r['per_class']['Malignant']['f1']:.4f}).
- **Benign:** Benign classification was difficult, especially for EfficientNet-B0: recall
  {e['per_class']['Benign']['recall']:.2f} vs {r['per_class']['Benign']['recall']:.2f} and F1
  {e['per_class']['Benign']['f1']:.3f} vs {r['per_class']['Benign']['f1']:.3f}.
- **Normal:** EfficientNet-B0 had much higher Normal recall ({e['per_class']['Normal']['recall']:.2f} vs
  {r['per_class']['Normal']['recall']:.2f}) but lower precision ({e['per_class']['Normal']['precision']:.3f} vs
  {r['per_class']['Normal']['precision']:.3f}).

## AUC, speed, and error patterns

EfficientNet-B0 has higher one-vs-rest AUC for Normal ({e['per_class']['Normal']['auc']:.3f} vs
{r['per_class']['Normal']['auc']:.3f}), Benign ({e['per_class']['Benign']['auc']:.3f} vs
{r['per_class']['Benign']['auc']:.3f}), and Malignant ({e['per_class']['Malignant']['auc']:.3f} vs
{r['per_class']['Malignant']['auc']:.3f}). AUC measures ranking/discrimination across
thresholds, whereas accuracy and F1 reflect the selected argmax decision rule; higher AUC
alone therefore does not establish EfficientNet-B0 as superior for the reported rule.

The largest pairwise error was Benign <-> Malignant for both models (30 ResNet50; 28
EfficientNet-B0). Normal <-> Benign totals were 10 and 15; Normal <-> Malignant totals
were 6 for each. Malignant false negatives totalled 15/87 for both models. ResNet50 had
2 Malignant -> Normal and 13 Malignant -> Benign errors; EfficientNet-B0 had 6 and 9.
These clinically important missed-Malignant patterns should be examined explicitly.

Saved inference measurements were {r['inference_ms_per_image']:.1f} ms/image for ResNet50
and {e['inference_ms_per_image']:.1f} ms/image for EfficientNet-B0. Timing is
environment-dependent and should be re-measured for deployment.

## Limitations

This single held-out comparison has no confidence intervals, external cohort, calibration
assessment, reader study, or statistical significance testing. Neither model is clinically
validated, and these descriptive results do not support diagnostic or causal claims.
"""
    (OUTPUT / "results_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = load_results()
    write_model_comparison(results)
    write_per_class_metrics(results)
    write_error_analysis(results)
    create_figures(results)
    write_summary(results)
    print(f"Created final analysis package in {OUTPUT}")


if __name__ == "__main__":
    main()

