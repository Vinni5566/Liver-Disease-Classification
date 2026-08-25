"""Create a reproducible stratified train/test split from the supplied AUL data."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

CLASS_NAMES = ("Normal", "Benign", "Malignant")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def project_root() -> Path:
    return Path(__file__).resolve().parent


def default_raw_root() -> Path:
    return project_root().parent / "7272660"


def resolve_raw_root(raw_root: Path) -> Path:
    """Accept either the documented root or a common extra archive-wrapper folder."""
    raw_root = Path(raw_root)
    if all((raw_root / name / name / "image").exists() for name in CLASS_NAMES):
        return raw_root
    nested = raw_root / raw_root.name
    if all((nested / name / name / "image").exists() for name in CLASS_NAMES):
        print(f"Detected archive wrapper; using raw data root: {nested}")
        return nested
    return raw_root


def scan_dataset(raw_root: Path) -> Tuple[List[Tuple[Path, str]], Dict[str, int], List[Path]]:
    """Scan the double-nested class folders and identify missing liver annotations."""
    records: List[Tuple[Path, str]] = []
    counts: Dict[str, int] = {class_name: 0 for class_name in CLASS_NAMES}
    missing_liver: List[Path] = []
    for class_name in CLASS_NAMES:
        class_root = raw_root / class_name / class_name
        image_dir = class_root / "image"
        liver_dir = class_root / "segmentation" / "liver"
        if not image_dir.exists():
            raise FileNotFoundError(f"Expected image directory was not found: {image_dir}")
        for image_path in sorted(image_dir.iterdir(), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            counts[class_name] += 1
            records.append((image_path, class_name))
            if not (liver_dir / f"{image_path.stem}.json").exists():
                missing_liver.append(image_path)
    return records, counts, missing_liver


def csv_image_counts(csv_path: Path) -> Dict[str, int]:
    """Read only the summary image rows; this CSV is never joined to image files."""
    expected: Dict[str, int] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            # The provided summary currently names this field sub_category_level_1;
            # accept the documented sub_category spelling as well.
            if row.get("sub_category_level_1", row.get("sub_category")) == "image":
                expected[row["main_category"]] = int(row["number_of_files"])
    return expected


def copy_split(items: List[Tuple[Path, str]], split_name: str, output_root: Path) -> None:
    for image_path, class_name in items:
        destination = output_root / split_name / class_name / image_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the AUL ultrasound dataset.")
    parser.add_argument("--raw-root", type=Path, default=default_raw_root())
    parser.add_argument("--csv", type=Path, default=project_root().parent / "liverdataset.csv")
    parser.add_argument("--output", type=Path, default=project_root() / "dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of using hard links.")
    args = parser.parse_args()

    args.raw_root = resolve_raw_root(args.raw_root)
    records, found_counts, missing_liver = scan_dataset(args.raw_root)
    summary_counts = csv_image_counts(args.csv)
    print("\nFolder scan and CSV sanity check")
    for class_name in CLASS_NAMES:
        found, expected = found_counts[class_name], summary_counts.get(class_name)
        status = "OK" if found == expected else f"MISMATCH (CSV: {expected})"
        print(f"  {class_name:10s}: found={found:3d} | {status}")
    print(f"\nImages without segmentation/liver JSON: {len(missing_liver)}")
    for image_path in missing_liver:
        print(f"  WARNING: {image_path}")
    print("These remain in the baseline split; ROI mode skips its annotation and uses the full image.")

    paths, labels = zip(*records)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=0.20, random_state=args.seed, stratify=labels
    )
    train_items = list(zip(train_paths, train_labels))
    test_items = list(zip(test_paths, test_labels))

    if args.output.exists():
        shutil.rmtree(args.output)
    for split_name, items in (("train", train_items), ("test", test_items)):
        for image_path, class_name in items:
            destination = args.output / split_name / class_name / image_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if args.copy:
                shutil.copy2(image_path, destination)
            else:
                # Hard links avoid duplicating the image data; transparently fall back to copy.
                try:
                    destination.hardlink_to(image_path)
                except OSError:
                    shutil.copy2(image_path, destination)
        distribution = {name: sum(label == name for _, label in items) for name in CLASS_NAMES}
        print(f"{split_name.title()} distribution: {distribution} (total={len(items)})")
    print(f"\nCreated stratified split at: {args.output}")


if __name__ == "__main__":
    main()
