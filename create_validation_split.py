"""Create the fixed, stratified validation subset from dataset/train only."""
from __future__ import annotations

import argparse
from pathlib import Path

from data import CLASS_NAMES, create_development_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible train/validation manifest from development data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = create_development_split_manifest(
        root / "dataset" / "train", root / "dataset" / "test",
        root / "dataset" / "development_split.json", args.seed, args.validation_fraction,
    )
    print("Created reproducible development split manifest:")
    for split_name in ("train", "validation", "test"):
        counts = {class_name: len(manifest["splits"][split_name][class_name]) for class_name in CLASS_NAMES}
        print(f"  {split_name:10s}: {counts} (total={sum(counts.values())})")
    print(f"Overlap check: {manifest['overlap_check']}")


if __name__ == "__main__":
    main()
