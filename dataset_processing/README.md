# Semantic coverage dataset preprocessing

This directory implements the preprocessing pipeline used to construct the
10-class semantic-segmentation dataset described in the paper. The pipeline:

1. maps the native labels of RescueNet, FloodNet, and the Semantic
   Segmentation of Satellite Imagery dataset to a common taxonomy;
2. converts the semantic masks to YOLO segmentation polygons; and
3. pools the converted samples into a deterministic 80/10/10 split.

The source images and annotations are not redistributed. Download each dataset
from its original provider and follow its licence terms.

## Requirements

Python 3.9 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

## Source data layout

Download the source datasets from their original repositories:

- [RescueNet](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation)
- [FloodNet](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021#floodnet-dataset)
- [Semantic Segmentation with PyTorch using Satellite Imagery](https://github.com/JenAlchimowicz/Semantic-segmentation-with-PyTorch-Satellite-Imagery)

Place the downloaded and extracted datasets under a `dataset` directory using
the following structure:

```text
dataset/
├── rescuenet/
│   ├── RescueNet/
│   │   ├── train/{train-org-img,train-label-img}/
│   │   ├── val/{val-org-img,val-label-img}/
│   │   └── test/{test-org-img,test-label-img}/
│   └── ColorMasks-RescueNet/                 # optional colour masks
├── floodnet/
│   └── data/{image,mask}/
└── satellite/
    ├── data/
    └── labels/
```

If a downloaded archive uses different directory names, pass
`--images-dir` and `--labels-dir` to the relevant converter. Run a converter
with `--help` for all available options.

## Convert the source datasets

```bash
python convert_rescuenet.py --dataset-root dataset
python convert_floodnet.py --dataset-root dataset
python convert_satellite.py --dataset-root dataset
```

The converted files are written to
`processed/<source>/{images,labels}`. Output filenames include their source
name to prevent collisions during merging. The official grayscale RescueNet
masks are used by default; pass `--colour-labels` to use the colour-coded mask
distribution.

The scripts refuse to mix new output with an existing conversion. Pass
`--clean` to replace the corresponding derived-data directory explicitly.

## Merge and split

```bash
python merge_and_split.py \
  --processed-root processed \
  --merged-dir merged \
  --expected-total 5234
```

The default seed is 42. For 5,234 retained image/label pairs, the split contains
4,188 training, 523 validation, and 523 test images. Rounding remainders are
assigned to training. The generated `split_manifest.csv` records the source and
split of every image, and `data.yaml` can be passed to an Ultralytics training
command.

`--expected-total` is a consistency check for reproducing the dataset reported
in the paper. A different source release or a filtered input directory may
produce another total. The optional `--min-instances` argument removes images
with few polygon records and should only be used when intentionally constructing
a filtered variant.

## Common taxonomy

| ID | Class |
|---:|---|
| 0 | Water |
| 1 | Building No Damage |
| 2 | Building Medium Damage |
| 3 | Building Major Damage |
| 4 | Building Total Destruction |
| 5 | Vehicle |
| 6 | Clear Road |
| 7 | Blocked Road |
| 8 | Tree |
| 9 | Pool |

Pixels mapped to 255 are ignored when polygon labels are generated. The full
native-to-common mappings are defined in `mappings.py`. The main consolidation
rules are:

- FloodNet flooded roads map to Blocked Road, while non-flooded roads and grass
  map to Clear Road.
- Satellite roofs and structural objects map to Building No Damage;
  under-construction areas map to Building Medium Damage; and road, grass,
  parking, and dense-vegetation labels follow the Clear Road assignment used in
  the project preprocessing notebook.
- RescueNet debris maps to Blocked Road, minor building damage maps to Building
  Medium Damage, and its road and sand labels map to Clear Road.
