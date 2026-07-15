"""Common taxonomy and source-specific class mappings.

Pixels mapped to ``IGNORE`` are excluded when YOLO segmentation labels are
generated. The class identifiers match Table I of the paper.
"""

IGNORE = 255

TARGET_NAMES = {
    0: "Water",
    1: "Building No Damage",
    2: "Building Medium Damage",
    3: "Building Major Damage",
    4: "Building Total Destruction",
    5: "Vehicle",
    6: "Clear Road",
    7: "Blocked Road",
    8: "Tree",
    9: "Pool",
}
NUM_CLASSES = 10

# FloodNet native labels: 0 Background, 1 Building-flooded,
# 2 Building-non-flooded, 3 Road-flooded, 4 Road-non-flooded, 5 Water,
# 6 Tree, 7 Vehicle, 8 Pool, and 9 Grass.
FLOODNET_MAP = {
    0: IGNORE,
    1: 1,
    2: 1,
    3: 7,
    4: 6,
    5: 0,
    6: 8,
    7: 5,
    8: 9,
    9: 6,
}

# Semantic Segmentation of Satellite Imagery native labels. The assignments
# preserve the mapping used in the project preprocessing notebook.
SATELLITE_MAP = {
    0: IGNORE,
    1: 1,
    2: 1,
    3: 9,
    4: 5,
    5: 6,
    6: 8,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: IGNORE,
    12: IGNORE,
    13: IGNORE,
    14: 6,
    15: 2,
    16: IGNORE,
    17: IGNORE,
    18: 6,
    19: 1,
    20: 1,
    21: 6,
    22: 0,
    23: 0,
    24: IGNORE,
}

# RescueNet v1.0 native labels:
# 0 Background, 1 Debris, 2 Water, 3 Building_No_Damage,
# 4 Building_Minor_Damage, 5 Building_Major_Damage,
# 6 Building_Total_Destruction, 7 Vehicle, 8 Road, 9 Tree, 10 Pool,
# and 11 Sand.
RESCUENET_MAP = {
    0: IGNORE,
    1: 7,
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 8,
    10: 9,
    11: 6,
}

# BGR colours used by the colour-coded RescueNet masks distributed alongside
# the original project data. The official grayscale masks are preferred.
RESCUENET_COLOR_MAP = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    2: [6, 184, 255],
    3: [7, 250, 4],
    4: [7, 255, 235],
    5: [20, 150, 160],
    6: [120, 120, 180],
    7: [140, 101, 0],
    8: [140, 140, 140],
    9: [245, 0, 255],
    10: [250, 230, 61],
}
