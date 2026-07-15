"""Shared utilities for converting semantic masks to YOLO segmentation labels."""

import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import cv2 as cv
import numpy as np

from mappings import IGNORE, NUM_CLASSES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def prepare_output_directory(output_dir: Path, clean: bool = False) -> None:
    """Create an empty derived-data directory.

    Existing output is retained only when the directory is empty. Pass
    ``clean=True`` to replace a previous conversion explicitly.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not clean:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Choose another directory or pass --clean."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def remap_mask(native_mask: np.ndarray, mapping: Mapping[int, int]) -> np.ndarray:
    """Map native class identifiers to the common taxonomy."""
    remapped = np.full(native_mask.shape, IGNORE, dtype=np.int32)
    for source_id, target_id in mapping.items():
        remapped[native_mask == source_id] = target_id
    return remapped


def colour_mask_to_ids(
    mask_bgr: np.ndarray,
    colour_map: Mapping[int, Sequence[int]],
) -> np.ndarray:
    """Decode a BGR colour mask into native class identifiers."""
    if mask_bgr is None or mask_bgr.ndim != 3 or mask_bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR colour mask")

    native_mask = np.full(mask_bgr.shape[:2], IGNORE, dtype=np.int32)
    for class_id, bgr in colour_map.items():
        colour = np.asarray(bgr, dtype=np.uint8)
        native_mask[cv.inRange(mask_bgr, colour, colour) > 0] = class_id
    return native_mask


def mask_to_yolo_lines(
    target_mask: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> List[str]:
    """Convert a target-class mask to normalised YOLO polygon records."""
    if target_mask.ndim != 2:
        raise ValueError("Expected a two-dimensional target mask")

    height, width = target_mask.shape
    if height == 0 or width == 0:
        raise ValueError("Mask dimensions must be non-zero")

    records: List[str] = []
    for class_id in range(num_classes):
        binary = (target_mask == class_id).astype(np.uint8) * 255
        contours, _ = cv.findContours(
            binary,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            polygon = contour.reshape(-1, 2).astype(np.float64)
            polygon[:, 0] /= width
            polygon[:, 1] /= height
            coordinates = " ".join(f"{value:.6f}" for value in polygon.ravel())
            records.append(f"{class_id} {coordinates}")
    return records


def _image_files(images_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def convert_dataset(
    dataset_name: str,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    mapping: Mapping[int, int],
    label_suffix: str,
    filename_prefix: Optional[str] = None,
    colour_labels: bool = False,
    colour_map: Optional[Dict[int, Sequence[int]]] = None,
    num_classes: int = NUM_CLASSES,
) -> int:
    """Convert one image/mask directory pair to the common YOLO format."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels_dir}")
    if colour_labels and not colour_map:
        raise ValueError("A colour map is required for colour-coded labels")

    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    prefix = filename_prefix or dataset_name
    converted = 0
    skipped = 0
    unmapped_ids = set()

    for image_path in _image_files(images_dir):
        label_path = labels_dir / f"{image_path.stem}{label_suffix}"
        if not label_path.is_file():
            skipped += 1
            continue

        image = cv.imread(str(image_path), cv.IMREAD_COLOR)
        label_mode = cv.IMREAD_COLOR if colour_labels else cv.IMREAD_GRAYSCALE
        source_mask = cv.imread(str(label_path), label_mode)
        if image is None or source_mask is None:
            skipped += 1
            continue

        if colour_labels:
            native_mask = colour_mask_to_ids(source_mask, colour_map or {})
        else:
            native_mask = source_mask.astype(np.int32)
            unmapped_ids.update(set(np.unique(native_mask)) - set(mapping))

        if image.shape[:2] != native_mask.shape:
            raise ValueError(
                f"Image and mask dimensions differ for {image_path.name}: "
                f"{image.shape[:2]} != {native_mask.shape}"
            )

        target_mask = remap_mask(native_mask, mapping)
        records = mask_to_yolo_lines(target_mask, num_classes)
        output_stem = f"{prefix}_{image_path.stem}"

        if not cv.imwrite(str(output_images / f"{output_stem}.jpg"), image):
            raise OSError(f"Could not write converted image: {output_stem}.jpg")
        with (output_labels / f"{output_stem}.txt").open(
            "w",
            encoding="utf-8",
        ) as label_file:
            if records:
                label_file.write("\n".join(records) + "\n")
        converted += 1

    print(
        f"[{dataset_name}] converted {converted} image/mask pairs; "
        f"skipped {skipped} missing or unreadable pairs"
    )
    if unmapped_ids:
        values = ", ".join(str(value) for value in sorted(unmapped_ids))
        print(f"[{dataset_name}] ignored unmapped native class IDs: {values}")
    return converted
