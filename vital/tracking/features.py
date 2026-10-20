"""Cell Feature Extraction from 3D Segmented Volumes.

Uses skimage.measure.regionprops_table to measure 3D centroids (z, y, x), volume, and mean intensity.
Generates Stage 1 detection records adhering to CZ Biohub light-sheet specification.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from vital.config import SCALE_Z, SCALE_Y, SCALE_X


def extract_frame_features(
    mask_3d: np.ndarray,
    intensity_3d: Optional[np.ndarray] = None,
    t: int = 0,
    start_detection_id: int = 1,
) -> Tuple[List[Dict[str, Any]], int]:
    """Extract object features from a 3D segmented mask for a single timepoint.

    Args:
        mask_3d: 3D integer array of shape (Z, Y, X) with labeled objects.
        intensity_3d: Optional 3D intensity array of shape (Z, Y, X) for mean intensity.
        t: Integer timepoint frame index.
        start_detection_id: Counter offset for globally unique per-instance detection IDs.

    Returns:
        detections: List of dictionaries representing detected cell instances.
        next_detection_id: Updated counter for consecutive frames.
    """
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask_3d.shape}")

    # If all background, return empty list
    if not np.any(mask_3d > 0):
        return [], start_detection_id

    properties = ["label", "centroid", "area"]
    if intensity_3d is not None:
        props_dict = regionprops_table(
            mask_3d,
            intensity_image=intensity_3d,
            properties=properties + ["intensity_mean"],
        )
    else:
        props_dict = regionprops_table(
            mask_3d,
            properties=properties,
        )

    labels = props_dict["label"]
    centroids_z = props_dict["centroid-0"]
    centroids_y = props_dict["centroid-1"]
    centroids_x = props_dict["centroid-2"]
    volumes = props_dict["area"]  # In 3D skimage, 'area' reports voxel count (volume)

    if intensity_3d is not None and "intensity_mean" in props_dict:
        mean_intensities = props_dict["intensity_mean"]
    else:
        mean_intensities = np.ones_like(volumes, dtype=np.float32)

    detections = []
    current_id = start_detection_id

    for i in range(len(labels)):
        z = float(centroids_z[i])
        y = float(centroids_y[i])
        x = float(centroids_x[i])
        vol = int(volumes[i])
        mean_int = float(mean_intensities[i])

        detections.append({
            "detection_id": current_id,
            "t": int(t),
            "z": z,
            "y": y,
            "x": x,
            "z_um": z * SCALE_Z,
            "y_um": y * SCALE_Y,
            "x_um": x * SCALE_X,
            "volume": vol,
            "mean_intensity": mean_int,
        })
        current_id += 1

    return detections, current_id


def detections_to_dataframe(detections: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format detection records into Stage 1 DataFrame with standard columns.

    Standard columns: [unique_id, cell_id, detection_id, t, z, y, x, volume, mean_intensity]
    """
    columns = [
        "unique_id",
        "cell_id",
        "detection_id",
        "t",
        "z",
        "y",
        "x",
        "volume",
        "mean_intensity",
        "z_um",
        "y_um",
        "x_um",
    ]
    if not detections:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(detections)
    if "unique_id" not in df.columns:
        if "detection_id" in df.columns:
            df["unique_id"] = df["detection_id"]
        elif "cell_id" in df.columns:
            df["unique_id"] = df["cell_id"]
    if "cell_id" not in df.columns:
        df["cell_id"] = df["unique_id"]

    # Ensure standard ordering
    desired_cols = [c for c in columns if c in df.columns] + [
        c for c in df.columns if c not in columns
    ]
    df = df[desired_cols]
    df.sort_values(by=["t", "detection_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
