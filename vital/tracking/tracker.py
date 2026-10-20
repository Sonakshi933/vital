"""Hungarian Algorithm 3D Spatiotemporal Cell Tracker with Integrated Division Detection.

Performs frame-to-frame cell linking using scipy.optimize.linear_sum_assignment
on a physically scaled Euclidean distance cost matrix. Matches within max cutoff (7.0 µm)
continue as migration. Unmatched tracks or bifurcating tracks are evaluated by the
rule-based DivisionDetector for mitosis events (1.5 - 16.0 µm).
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from vital.config import (
    MAX_MATCH_DISTANCE_UM,
    DIVISION_MIN_DIST_UM,
    DIVISION_MAX_DIST_UM,
    SCALE_Z,
    SCALE_Y,
    SCALE_X,
)
from vital.tracking.division import DivisionDetector


class HungarianTracker:
    """Spatiotemporal cell tracker with Hungarian matching and geometric division detection."""

    def __init__(
        self,
        max_match_dist_um: float = MAX_MATCH_DISTANCE_UM,
        division_detector: Optional[DivisionDetector] = None,
    ):
        self.max_match_dist_um = max_match_dist_um
        self.division_detector = division_detector or DivisionDetector()

        # State tracking
        self.next_track_id: int = 1
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.records: List[Dict[str, Any]] = []
        self.division_events: List[Dict[str, Any]] = []

    def _compute_physical_distance(
        self,
        pos1_um: Tuple[float, float, float],
        pos2_um: Tuple[float, float, float],
    ) -> float:
        dz = pos1_um[0] - pos2_um[0]
        dy = pos1_um[1] - pos2_um[1]
        dx = pos1_um[2] - pos2_um[2]
        return float(np.sqrt(dz * dz + dy * dy + dx * dx))

    def reset(self) -> None:
        """Reset tracker state."""
        self.next_track_id = 1
        self.active_tracks.clear()
        self.records.clear()
        self.division_events.clear()

    def track_all_frames(
        self,
        detections_by_frame: Dict[int, List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """Track cells across all available frames.

        Args:
            detections_by_frame: Dictionary mapping frame index t to list of detection dicts.

        Returns:
            pd.DataFrame: Formatted Stage 2 tracking results table.
        """
        self.reset()
        frames = sorted(detections_by_frame.keys())
        if not frames:
            return self._build_dataframe([])

        for t in frames:
            current_detections = detections_by_frame[t]
            self.step_frame(t, current_detections)

        return self._build_dataframe(self.records)

    def step_frame(self, t: int, current_detections: List[Dict[str, Any]]) -> None:
        """Process a single timepoint frame."""
        # Initial frame or no active tracks: all detections initialize new root tracks
        if not self.active_tracks:
            for det in current_detections:
                track_id = self.next_track_id
                self.next_track_id += 1

                track_state = {
                    "track_id": track_id,
                    "parent_track_id": -1,
                    "last_t": t,
                    "z": det["z"],
                    "y": det["y"],
                    "x": det["x"],
                    "z_um": det["z_um"],
                    "y_um": det["y_um"],
                    "x_um": det["x_um"],
                    "detection_id": det["detection_id"],
                }
                self.active_tracks[track_id] = track_state

                self.records.append({
                    "track_id": track_id,
                    "parent_track_id": -1,
                    "t": t,
                    "z": det["z"],
                    "y": det["y"],
                    "x": det["x"],
                    "z_um": det["z_um"],
                    "y_um": det["y_um"],
                    "x_um": det["x_um"],
                    "detection_id": det["detection_id"],
                    "event": "track_start",
                    "displacement_um": 0.0,
                    "qc_flag": "NORMAL",
                })
            return

        active_track_ids = list(self.active_tracks.keys())
        n_active = len(active_track_ids)
        n_dets = len(current_detections)

        if n_dets == 0:
            for tid in active_track_ids:
                for r in reversed(self.records):
                    if r["track_id"] == tid:
                        if r["event"] not in ("division", "track_end"):
                            r["event"] = "track_end"
                        break
                del self.active_tracks[tid]
            return

        # 1. Build cost matrix (n_active x n_dets) of physical Euclidean distances
        cost_matrix = np.zeros((n_active, n_dets), dtype=np.float64)
        for i, tid in enumerate(active_track_ids):
            t_state = self.active_tracks[tid]
            pos1_um = (t_state["z_um"], t_state["y_um"], t_state["x_um"])
            for j, det in enumerate(current_detections):
                pos2_um = (det["z_um"], det["y_um"], det["x_um"])
                cost_matrix[i, j] = self._compute_physical_distance(pos1_um, pos2_um)

        # 2. Solve Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()
        provisional_matches: Dict[int, Dict[str, Any]] = {}

        for r, c in zip(row_ind, col_ind):
            dist = cost_matrix[r, c]
            if dist <= self.max_match_dist_um:
                tid = active_track_ids[r]
                det = current_detections[c]
                matched_tracks.add(tid)
                matched_dets.add(c)
                provisional_matches[tid] = {
                    "detection": det,
                    "det_index": c,
                    "distance": dist,
                }

        # Remaining unmatched detections
        unmatched_det_indices = [c for c in range(n_dets) if c not in matched_dets]
        unmatched_dets = [current_detections[c] for c in unmatched_det_indices]

        # 3. Check for division events
        # A track divides if:
        # Case A: An unmatched active track has 2 unmatched detections in [1.5, 16.0] µm
        # Case B: A matched active track has another unmatched detection in [1.5, 16.0] µm
        #         (and its matched detection is also >= 1.5 µm from parent)
        dividing_parents = {}

        # First check Case A: Unmatched active tracks
        unmatched_active_ids = [tid for tid in active_track_ids if tid not in matched_tracks]
        for tid in unmatched_active_ids:
            parent_pos_um = (
                self.active_tracks[tid]["z_um"],
                self.active_tracks[tid]["y_um"],
                self.active_tracks[tid]["x_um"],
            )
            daughters = self.division_detector.find_daughter_candidates(
                parent_pos_um, unmatched_dets
            )
            if len(daughters) == 2:
                dividing_parents[tid] = daughters
                # Remove daughters from pool
                d_ids = {d["detection_id"] for d in daughters}
                unmatched_dets = [d for d in unmatched_dets if d["detection_id"] not in d_ids]

        # Next check Case B: Matched tracks that also have an unmatched nearby detection
        for tid in list(provisional_matches.keys()):
            if not unmatched_dets:
                break
            p_state = self.active_tracks[tid]
            parent_pos_um = (p_state["z_um"], p_state["y_um"], p_state["x_um"])
            d1 = provisional_matches[tid]["detection"]
            d1_dist = provisional_matches[tid]["distance"]

            if d1_dist >= self.division_detector.min_dist_um:
                # Check if any unmatched detection is also within division bounds
                for d2 in unmatched_dets:
                    d2_pos_um = (d2["z_um"], d2["y_um"], d2["x_um"])
                    d2_dist = self._compute_physical_distance(parent_pos_um, d2_pos_um)
                    if self.division_detector.min_dist_um <= d2_dist <= self.division_detector.max_dist_um:
                        # Division detected!
                        dividing_parents[tid] = [d1, d2]
                        matched_tracks.remove(tid)
                        del provisional_matches[tid]
                        unmatched_dets = [d for d in unmatched_dets if d["detection_id"] != d2["detection_id"]]
                        break

        # Process all confirmed divisions
        for tid, daughters in dividing_parents.items():
            d1, d2 = daughters[0], daughters[1]
            parent_state = self.active_tracks[tid]
            parent_pos_um = (parent_state["z_um"], parent_state["y_um"], parent_state["x_um"])

            # Update parent's last record to 'division'
            for r in reversed(self.records):
                if r["track_id"] == tid:
                    r["event"] = "division"
                    break

            # Spawn 2 new child tracks
            child1_id = self.next_track_id
            self.next_track_id += 1
            child2_id = self.next_track_id
            self.next_track_id += 1

            # Record child 1
            self.active_tracks[child1_id] = {
                "track_id": child1_id,
                "parent_track_id": tid,
                "last_t": t,
                "z": d1["z"],
                "y": d1["y"],
                "x": d1["x"],
                "z_um": d1["z_um"],
                "y_um": d1["y_um"],
                "x_um": d1["x_um"],
                "detection_id": d1["detection_id"],
            }
            self.records.append({
                "track_id": child1_id,
                "parent_track_id": tid,
                "t": t,
                "z": d1["z"],
                "y": d1["y"],
                "x": d1["x"],
                "z_um": d1["z_um"],
                "y_um": d1["y_um"],
                "x_um": d1["x_um"],
                "detection_id": d1["detection_id"],
                "event": "track_start",
                "displacement_um": self._compute_physical_distance(
                    parent_pos_um, (d1["z_um"], d1["y_um"], d1["x_um"])
                ),
                "qc_flag": "NORMAL",
            })

            # Record child 2
            self.active_tracks[child2_id] = {
                "track_id": child2_id,
                "parent_track_id": tid,
                "last_t": t,
                "z": d2["z"],
                "y": d2["y"],
                "x": d2["x"],
                "z_um": d2["z_um"],
                "y_um": d2["y_um"],
                "x_um": d2["x_um"],
                "detection_id": d2["detection_id"],
            }
            self.records.append({
                "track_id": child2_id,
                "parent_track_id": tid,
                "t": t,
                "z": d2["z"],
                "y": d2["y"],
                "x": d2["x"],
                "z_um": d2["z_um"],
                "y_um": d2["y_um"],
                "x_um": d2["x_um"],
                "detection_id": d2["detection_id"],
                "event": "track_start",
                "displacement_um": self._compute_physical_distance(
                    parent_pos_um, (d2["z_um"], d2["y_um"], d2["x_um"])
                ),
                "qc_flag": "NORMAL",
            })

            self.division_events.append({
                "parent_track_id": tid,
                "t": t,
                "parent_pos_um": parent_pos_um,
                "daughter_1_id": child1_id,
                "daughter_2_id": child2_id,
            })

            del self.active_tracks[tid]

        # Process remaining confirmed 1-to-1 migration matches
        for tid, match_info in provisional_matches.items():
            det = match_info["detection"]
            dist = match_info["distance"]
            prev_state = self.active_tracks[tid]

            self.active_tracks[tid] = {
                "track_id": tid,
                "parent_track_id": prev_state["parent_track_id"],
                "last_t": t,
                "z": det["z"],
                "y": det["y"],
                "x": det["x"],
                "z_um": det["z_um"],
                "y_um": det["y_um"],
                "x_um": det["x_um"],
                "detection_id": det["detection_id"],
            }

            qc_flag = "LARGE_JUMP" if dist > (self.max_match_dist_um * 0.8) else "NORMAL"
            self.records.append({
                "track_id": tid,
                "parent_track_id": prev_state["parent_track_id"],
                "t": t,
                "z": det["z"],
                "y": det["y"],
                "x": det["x"],
                "z_um": det["z_um"],
                "y_um": det["y_um"],
                "x_um": det["x_um"],
                "detection_id": det["detection_id"],
                "event": "migration",
                "displacement_um": dist,
                "qc_flag": qc_flag,
            })

        # Process remaining unmatched active tracks (terminated)
        for tid in list(self.active_tracks.keys()):
            if tid not in matched_tracks and tid not in dividing_parents and tid not in provisional_matches:
                for r in reversed(self.records):
                    if r["track_id"] == tid:
                        if r["event"] == "track_start":
                            r["event"] = "isolated_cell"
                        else:
                            r["event"] = "track_end"
                        break
                del self.active_tracks[tid]

        # Remaining unmatched detections initialize new tracks
        for det in unmatched_dets:
            track_id = self.next_track_id
            self.next_track_id += 1

            self.active_tracks[track_id] = {
                "track_id": track_id,
                "parent_track_id": -1,
                "last_t": t,
                "z": det["z"],
                "y": det["y"],
                "x": det["x"],
                "z_um": det["z_um"],
                "y_um": det["y_um"],
                "x_um": det["x_um"],
                "detection_id": det["detection_id"],
            }
            self.records.append({
                "track_id": track_id,
                "parent_track_id": -1,
                "t": t,
                "z": det["z"],
                "y": det["y"],
                "x": det["x"],
                "z_um": det["z_um"],
                "y_um": det["y_um"],
                "x_um": det["x_um"],
                "detection_id": det["detection_id"],
                "event": "track_start",
                "displacement_um": 0.0,
                "qc_flag": "NORMAL",
            })

    def _build_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        columns = [
            "track_id",
            "parent_track_id",
            "t",
            "z",
            "y",
            "x",
            "event",
            "detection_id",
            "z_um",
            "y_um",
            "x_um",
            "displacement_um",
            "qc_flag",
        ]
        if not records:
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(records)
        df.sort_values(by=["t", "track_id"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
