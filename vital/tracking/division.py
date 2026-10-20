"""Rule-Based Geometric Cell Division Detector.

Identifies cell division (mitosis/cytokinesis) events without machine learning:
When a parent track does not have a 1-to-1 migration match in frame t+1, but two unmatched
detections in t+1 both fall within the physical distance threshold (1.5 µm to 16.0 µm)
of the parent's terminal position, the parent track is closed and two child tracks are spawned.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from vital.config import DIVISION_MIN_DIST_UM, DIVISION_MAX_DIST_UM


class DivisionDetector:
    """Detects cell division events between an unmatched parent track and unmatched detections."""

    def __init__(
        self,
        min_dist_um: float = DIVISION_MIN_DIST_UM,
        max_dist_um: float = DIVISION_MAX_DIST_UM,
    ):
        self.min_dist_um = min_dist_um
        self.max_dist_um = max_dist_um

    def compute_distance_um(
        self,
        pos1_um: Tuple[float, float, float],
        pos2_um: Tuple[float, float, float],
    ) -> float:
        """Compute Euclidean distance between two 3D points in micrometers."""
        dz = pos1_um[0] - pos2_um[0]
        dy = pos1_um[1] - pos2_um[1]
        dx = pos1_um[2] - pos2_um[2]
        return float(np.sqrt(dz * dz + dy * dy + dx * dx))

    def find_daughter_candidates(
        self,
        parent_pos_um: Tuple[float, float, float],
        unmatched_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify if any unmatched detections in frame t+1 satisfy daughter criteria.

        Args:
            parent_pos_um: (z_um, y_um, x_um) coordinate tuple of the parent's last position.
            unmatched_candidates: List of candidate detection dictionaries at t+1.

        Returns:
            List of 2 daughter detection dictionaries if a valid division is detected, else [].
        """
        if len(unmatched_candidates) < 2:
            return []

        qualified = []
        for det in unmatched_candidates:
            det_pos_um = (det["z_um"], det["y_um"], det["x_um"])
            dist = self.compute_distance_um(parent_pos_um, det_pos_um)
            if self.min_dist_um <= dist <= self.max_dist_um:
                qualified.append((dist, det))

        # We require at least 2 daughters within the geometric radius
        if len(qualified) < 2:
            return []

        # Sort by distance to parent and select the 2 closest daughters
        qualified.sort(key=lambda x: x[0])
        daughter_1 = qualified[0][1]
        daughter_2 = qualified[1][1]

        return [daughter_1, daughter_2]
