"""Convert RescueNet masks to the common YOLO segmentation format."""

import argparse
from pathlib import Path
from typing import List, Tuple

from common import convert_dataset, prepare_output_directory
from mappings import RESCUENET_COLOR_MAP, RESCUENET_MAP


SPLITS = ("train", "val", "test")
COLOUR_MASK_DIRS = {
    "train": "ColorMasks-TrainSet",
    "val": "ColorMasks-ValSet",
    "test": "ColorMasks-TestSet",
}


def _find_rescuenet_root(dataset_root: Path) -> Path:
    candidates = (
        dataset_root / "rescuenet" / "RescueNet",
        dataset_root / "rescuenet",
        dataset_root / "RescueNet",
        dataset_root,
    )
    for candidate in candidates:
        if (candidate / "train" / "train-org-img").is_dir():
            return candidate
    return candidates[0]


def _find_colour_root(dataset_root: Path) -> Path:
    candidates = (
        dataset_root / "rescuenet" / "ColorMasks-RescueNet",
        dataset_root / "ColorMasks-RescueNet",
        dataset_root,
    )
    for candidate in candidates:
        if (candidate / COLOUR_MASK_DIRS["train"]).is_dir():
            return candidate
    return candidates[0]


def source_directories(
    dataset_root: Path,
    colour_labels: bool,
) -> List[Tuple[str, Path, Path]]:
    rescuenet_root = _find_rescuenet_root(dataset_root)
    colour_root = _find_colour_root(dataset_root) if colour_labels else None
    directories = []
    for split in SPLITS:
        images_dir = rescuenet_root / split / f"{split}-org-img"
        labels_dir = (
            colour_root / COLOUR_MASK_DIRS[split]
            if colour_root is not None
            else rescuenet_root / split / f"{split}-label-img"
        )
        directories.append((split, images_dir, labels_dir))
    return directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", "--dataset_root", type=Path, default=Path("dataset"))
    parser.add_argument("--out-root", "--out_root", type=Path, default=Path("processed"))
    parser.add_argument("--images-dir", "--images_dir", type=Path)
    parser.add_argument("--labels-dir", "--labels_dir", type=Path)
    parser.add_argument("--label-suffix", default="_lab.png")
    parser.add_argument("--colour-labels", "--color", action="store_true")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="replace an existing RescueNet output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.images_dir) != bool(args.labels_dir):
        raise ValueError("--images-dir and --labels-dir must be supplied together")

    output_dir = args.out_root / "rescuenet"
    prepare_output_directory(output_dir, args.clean)

    if args.images_dir:
        directories = [("all", args.images_dir, args.labels_dir)]
    else:
        directories = source_directories(args.dataset_root, args.colour_labels)

    total = 0
    for split, images_dir, labels_dir in directories:
        total += convert_dataset(
            f"rescuenet/{split}",
            images_dir,
            labels_dir,
            output_dir,
            RESCUENET_MAP,
            label_suffix=args.label_suffix,
            filename_prefix=f"rescuenet_{split}",
            colour_labels=args.colour_labels,
            colour_map=RESCUENET_COLOR_MAP,
        )
    print(f"[rescuenet] total converted image/mask pairs: {total}")


if __name__ == "__main__":
    main()
