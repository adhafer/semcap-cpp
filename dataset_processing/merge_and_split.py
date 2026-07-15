"""Merge converted sources and create a deterministic 80/10/10 dataset split."""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from common import prepare_output_directory
from mappings import NUM_CLASSES, TARGET_NAMES


SOURCES = ("floodnet", "satellite", "rescuenet")
Pair = Tuple[str, Path, Path]


def count_records(label_path: Path) -> int:
    with label_path.open(encoding="utf-8") as label_file:
        return sum(1 for line in label_file if line.strip())


def split_counts(total: int, validation_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    """Assign rounding remainders to training and keep held-out splits equal."""
    validation = int(total * validation_ratio)
    test = int(total * test_ratio)
    train = total - validation - test
    return train, validation, test


def collect_pairs(processed_root: Path, min_instances: int) -> List[Pair]:
    pairs: List[Pair] = []
    for source in SOURCES:
        images_dir = processed_root / source / "images"
        labels_dir = processed_root / source / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Converted {source} data not found under {processed_root / source}"
            )

        source_count = 0
        for image_path in sorted(images_dir.glob("*.jpg")):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            if min_instances > 0 and count_records(label_path) <= min_instances:
                continue
            pairs.append((source, image_path, label_path))
            source_count += 1
        print(f"[{source}] retained {source_count} image/label pairs")
    return pairs


def write_dataset_yaml(merged_dir: Path) -> None:
    names = "\n".join(
        f"  {class_id}: {json.dumps(TARGET_NAMES[class_id])}"
        for class_id in range(NUM_CLASSES)
    )
    content = (
        f"path: {json.dumps(str(merged_dir.resolve()))}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        f"nc: {NUM_CLASSES}\n"
        f"names:\n{names}\n"
    )
    (merged_dir / "data.yaml").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        "--out_root",
        type=Path,
        default=Path("processed"),
    )
    parser.add_argument(
        "--merged-dir",
        "--merged_dir",
        type=Path,
        default=Path("merged"),
    )
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--val", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-instances",
        "--min_labels",
        type=int,
        default=0,
        help="drop images with this many or fewer polygon records; 0 keeps all images",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        help="fail if the retained image count differs from this value",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="replace an existing merged directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = (args.train, args.val, args.test)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios cannot be negative")
    if abs(sum(ratios) - 1.0) >= 1e-9:
        raise ValueError("Split ratios must sum to 1")
    if args.min_instances < 0:
        raise ValueError("--min-instances cannot be negative")

    pairs = collect_pairs(args.processed_root, args.min_instances)
    if not pairs:
        raise RuntimeError("No converted image/label pairs were found")
    if args.expected_total is not None and len(pairs) != args.expected_total:
        raise RuntimeError(
            f"Expected {args.expected_total} retained pairs, found {len(pairs)}"
        )

    output_names = [image_path.name for _, image_path, _ in pairs]
    if len(output_names) != len(set(output_names)):
        raise RuntimeError("Duplicate converted image names would overwrite merged data")

    random.Random(args.seed).shuffle(pairs)
    train_count, validation_count, test_count = split_counts(
        len(pairs),
        args.val,
        args.test,
    )
    split_items: Dict[str, Sequence[Pair]] = {
        "train": pairs[:train_count],
        "val": pairs[train_count : train_count + validation_count],
        "test": pairs[train_count + validation_count :],
    }

    prepare_output_directory(args.merged_dir, args.clean)
    manifest_rows = []
    for split, items in split_items.items():
        output_images = args.merged_dir / split / "images"
        output_labels = args.merged_dir / split / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        for source, image_path, label_path in items:
            shutil.copy2(image_path, output_images / image_path.name)
            shutil.copy2(label_path, output_labels / label_path.name)
            manifest_rows.append(
                {
                    "split": split,
                    "source": source,
                    "image": image_path.name,
                    "label": label_path.name,
                }
            )
        print(f"[{split}] wrote {len(items)} image/label pairs")

    with (args.merged_dir / "split_manifest.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=("split", "source", "image", "label"),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    write_dataset_yaml(args.merged_dir)
    print(
        f"Wrote {len(pairs)} pairs to {args.merged_dir} with seed {args.seed} "
        f"and split counts {train_count}/{validation_count}/{test_count}"
    )


if __name__ == "__main__":
    main()
