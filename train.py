"""Train ResNet50 or EfficientNet on prepared AUL ultrasound splits."""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import TrainingConfig
from data import (CLASS_NAMES, LiverUltrasoundDataset, build_transforms, class_weights,
                  load_split_selection, resolve_raw_root, weighted_sampler)
from models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    """Capture RNG state needed to continue the next epoch reproducibly."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore RNG state when it is available in a full training-state checkpoint."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Avoid replacing the last valid checkpoint if a save is interrupted."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def is_training_state_checkpoint(checkpoint: dict) -> bool:
    """Distinguish new resumable checkpoints from legacy weights-only checkpoints."""
    required = {
        "epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "best_val_loss", "early_stopping_counter", "history", "rng_state", "config", "class_names",
    }
    return required.issubset(checkpoint)


def verify_resume_config(saved_config: dict, current_config: TrainingConfig) -> None:
    """Refuse a full-state resume if any scientific training setting changed."""
    current = asdict(current_config)
    for key, value in current.items():
        if key == "resume_checkpoint":
            continue  # This necessarily changes to point at checkpoint_last.pt.
        if saved_config.get(key) != value:
            raise ValueError(
                f"Full-state checkpoint configuration differs for {key!r}: "
                f"saved={saved_config.get(key)!r}, requested={value!r}."
            )


def freeze_batchnorm_running_stats(model) -> None:
    """Keep BatchNorm running mean/variance fixed while allowing affine weights if unfrozen."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def run_epoch(model, loader, criterion, optimizer, device, freeze_batchnorm=False):
    training = optimizer is not None
    model.train(training)
    # model.train() enables BatchNorm updates, so restore eval mode for every training epoch.
    if training and freeze_batchnorm:
        freeze_batchnorm_running_stats(model)
    total_loss = total_correct = total_samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward(); optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples


def save_curves(history, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in (("loss", "Loss"), ("accuracy", "Accuracy")):
        axes[0 if key == "loss" else 1].plot(history[f"train_{key}"], label="Train")
        axes[0 if key == "loss" else 1].plot(history[f"val_{key}"], label="Validation")
        axes[0 if key == "loss" else 1].set(title=label, xlabel="Epoch", ylabel=label)
        axes[0 if key == "loss" else 1].legend()
    fig.tight_layout(); fig.savefig(output_dir / "training_curves.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "efficientnet", "efficientnet_b0", "efficientnet_b3"])
    parser.add_argument("--batch-size", type=int, default=32, choices=[16, 32])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-batchnorm", action="store_true")
    parser.add_argument("--imbalance-strategy", choices=["sampler", "weighted_loss", "both"], default="sampler")
    parser.add_argument("--use-roi-annotations", action="store_true")
    parser.add_argument("--roi-mode", choices=["crop", "mask"], default="crop")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--experiment-name", default=None,
                        help="Relative artifact directory below models/<model>.")
    args = parser.parse_args()
    if args.experiment_name and (Path(args.experiment_name).is_absolute() or ".." in Path(args.experiment_name).parts):
        raise ValueError("experiment-name must be a relative path below models/<model>.")
    learning_rate = args.learning_rate if args.learning_rate is not None else TrainingConfig.learning_rate
    config = TrainingConfig(model_name=args.model, batch_size=args.batch_size, epochs=args.epochs,
                            learning_rate=learning_rate, freeze_backbone=args.freeze_backbone,
                            freeze_batchnorm=args.freeze_batchnorm, imbalance_strategy=args.imbalance_strategy,
                            use_roi_annotations=args.use_roi_annotations, roi_mode=args.roi_mode,
                            experiment_name=args.experiment_name,
                            resume_checkpoint=str(args.resume_checkpoint) if args.resume_checkpoint else None)
    root, raw_root = Path(__file__).resolve().parent, resolve_raw_root(Path(__file__).resolve().parent.parent / "7272660")
    output_dir = root / "models" / config.model_name
    if config.experiment_name:
        output_dir = output_dir / config.experiment_name
    completed_history_path = output_dir / "training_history.json"
    if completed_history_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed experiment artifacts in {output_dir}. "
            "Use a new experiment name."
        )
    manifest_path = root / "dataset" / "development_split.json"
    if not (root / "dataset" / "train").exists():
        raise FileNotFoundError("Prepared split missing. Run `python prepare_dataset.py` first.")
    if not manifest_path.exists():
        raise FileNotFoundError("Development split missing. Run `python create_validation_split.py` first.")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Only fixed development images are used here; dataset/test stays fully held out.
    train_set = LiverUltrasoundDataset(root / "dataset/train", raw_root, build_transforms(224, True), config.use_roi_annotations, config.roi_mode, load_split_selection(manifest_path, "train"))
    val_set = LiverUltrasoundDataset(root / "dataset/train", raw_root, build_transforms(224, False), config.use_roi_annotations, config.roi_mode, load_split_selection(manifest_path, "validation"))
    sampler = weighted_sampler(train_set) if config.imbalance_strategy in {"sampler", "both"} else None
    train_loader = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = build_model(config.model_name, freeze_backbone=config.freeze_backbone).to(device)
    resume_state = None
    if args.resume_checkpoint:
        if not args.resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        resumed = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if resumed.get("class_names") != CLASS_NAMES:
            raise ValueError("Resume checkpoint class mapping does not match this project.")
        if resumed.get("config", {}).get("model_name") != config.model_name:
            raise ValueError("Resume checkpoint model does not match --model.")
        model.load_state_dict(resumed["model_state_dict"])
        if is_training_state_checkpoint(resumed):
            if args.resume_checkpoint.resolve().parent != output_dir.resolve():
                raise ValueError("A full-state checkpoint must resume into its original experiment directory.")
            verify_resume_config(resumed["config"], config)
            resume_state = resumed
            print(f"Loaded resumable training state: {args.resume_checkpoint}")
        else:
            print(f"Loaded initialization checkpoint: {args.resume_checkpoint}")
    weights = class_weights(train_set).to(device) if config.imbalance_strategy in {"weighted_loss", "both"} else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    if output_dir.exists() and any(output_dir.iterdir()) and resume_state is None:
        raise FileExistsError(
            f"Refusing to overwrite existing experiment artifacts in {output_dir}. "
            "Use a new experiment name or a full-state checkpoint from this directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    history = {key: [] for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy")}
    best_loss, waiting = float("inf"), 0
    start_epoch, elapsed_before = 1, 0.0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        history = resume_state["history"]
        best_loss = float(resume_state["best_val_loss"])
        waiting = int(resume_state["early_stopping_counter"])
        start_epoch = int(resume_state["epoch"]) + 1
        elapsed_before = float(resume_state.get("elapsed_seconds", 0.0))
        restore_rng_state(resume_state["rng_state"])
    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, config.freeze_batchnorm)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)
        for key, value in (("train_loss", train_loss), ("train_accuracy", train_acc), ("val_loss", val_loss), ("val_accuracy", val_acc)):
            history[key].append(value)
        print(f"Epoch {epoch:02d}/{config.epochs} | train loss {train_loss:.4f}, acc {train_acc:.3f} | val loss {val_loss:.4f}, acc {val_acc:.3f}")
        if val_loss < best_loss:
            best_loss, waiting = val_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "config": asdict(config), "class_names": CLASS_NAMES}, output_dir / "best_model.pt")
        else:
            waiting += 1
        atomic_torch_save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_loss,
            "early_stopping_counter": waiting,
            "history": history,
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            "rng_state": capture_rng_state(),
            "config": asdict(config),
            "class_names": CLASS_NAMES,
        }, output_dir / "checkpoint_last.pt")
        if waiting >= config.patience:
            print(f"Early stopping after {epoch} epochs."); break
    history["elapsed_seconds"] = elapsed_before + time.perf_counter() - started
    history["best_val_loss"] = best_loss
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_curves(history, output_dir)
    print(f"Saved best checkpoint and training artifacts to {output_dir}")


if __name__ == "__main__":
    main()
