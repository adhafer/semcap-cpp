"""Convert FloodNet semantic masks to the common YOLO segmentation format."""

import argparse
from pathlib import Path
from typing import Tuple

from common import convert_dataset, prepare_output_directory
from mappings import FLOODNET_MAP


def default_directories(dataset_root: Path) -> Tuple[Path, Path]:
    candidates = (
        (
            dataset_root / "floodnet" / "data" / "image",
            dataset_root / "floodnet" / "data" / "mask",
        ),
        (dataset_root / "floodnet" / "images", dataset_root / "floodnet" / "masks"),
        (dataset_root / "data" / "image", dataset_root / "data" / "mask"),
        (dataset_root / "image", dataset_root / "mask"),
    )
    for images_dir, labels_dir in candidates:
        if images_dir.is_dir() and labels_dir.is_dir():
            return images_dir, labels_dir
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", "--dataset_root", type=Path, default=Path("dataset"))
    parser.add_argument("--out-root", "--out_root", type=Path, default=Path("processed"))
    parser.add_argument("--images-dir", "--images_dir", type=Path)
    parser.add_argument("--labels-dir", "--labels_dir", type=Path)
    parser.add_argument("--label-suffix", default="_lab.png")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="replace an existing FloodNet output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.images_dir) != bool(args.labels_dir):
        raise ValueError("--images-dir and --labels-dir must be supplied together")

    if args.images_dir:
        images_dir, labels_dir = args.images_dir, args.labels_dir
    else:
        images_dir, labels_dir = default_directories(args.dataset_root)

    output_dir = args.out_root / "floodnet"
    prepare_output_directory(output_dir, args.clean)
    convert_dataset(
        "floodnet",
        images_dir,
        labels_dir,
        output_dir,
        FLOODNET_MAP,
        label_suffix=args.label_suffix,
    )


if __name__ == "__main__":
    main()
