"""Global configuration and hyperparameters for V.I.T.A.L.
"""

from pathlib import Path
from dataclasses import dataclass

# Physical voxel scale in micrometers (µm / voxel)
SCALE_Z: float = 1.625
SCALE_Y: float = 0.40625
SCALE_X: float = 0.40625
VOXEL_SCALE = (SCALE_Z, SCALE_Y, SCALE_X)

# Tracking parameters
MAX_MATCH_DISTANCE_UM: float = 7.0  # Hungarian assignment cutoff in micrometers

# Division detection parameters (rule-based geometric bounds)
DIVISION_MIN_DIST_UM: float = 1.5   # Minimum parent-to-daughter distance in µm
DIVISION_MAX_DIST_UM: float = 16.0  # Maximum parent-to-daughter distance in µm

# Segmentation model configurations
# Config flag for backend: "cellpose" | "monai_unet"
SEGMENTER_BACKEND: str = "cellpose"
CELLPOSE_MODEL_TYPE: str = "cyto3"
CELLPOSE_DO_3D: bool = True
CELLPOSE_DIAMETER_VOXELS: float = 15.0  # Approximate cell diameter in voxels

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "biohub-cell-tracking-during-development"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
DEMO_CACHE_DIR = BASE_DIR / "vital" / "app" / "demo_cache"

# Selected Demo Recordings (rich in division events)
DEMO_RECORDINGS = [
    "6bba_48816121",  # 5 division events, 17 root lineages, 935 nodes
    "6bba_df673a83",  # 4 division events, 14 root lineages, 668 nodes
    "6bba_4ffd3da3",  # 3 division events, 17 root lineages, 706 nodes
]
