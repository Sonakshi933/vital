"""Lineage Graph Ground-Truth Evaluator.

Implements the official CZ Biohub challenge evaluation metrics:
1. Edge Jaccard: Matching spatio-temporal directed edges within 7.0 µm physical cutoff,
   with over-prediction penalty: Jaccard = TP / (TP + FP + FN) * min(1, N_gt / N_pred).
2. Division Jaccard: Micro-averaged evaluation of mitotic division events where parent
   and both daughter cells match within the geometric tolerance.
"""

from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment

from vital.config import (
    SCALE_Z,
    SCALE_Y,
    SCALE_X,
    MAX_MATCH_DISTANCE_UM,
    DIVISION_MIN_DIST_UM,
    DIVISION_MAX_DIST_UM,
)


class LineageEvaluator:
    """Evaluates predicted lineage graphs against ground truth GEFF graphs."""

    def __init__(
        self,
        edge_tolerance_um: float = MAX_MATCH_DISTANCE_UM,
        daughter_min_dist_um: float = DIVISION_MIN_DIST_UM,
        daughter_max_dist_um: float = DIVISION_MAX_DIST_UM,
    ):
        self.edge_tolerance_um = edge_tolerance_um
        self.daughter_min_dist_um = daughter_min_dist_um
        self.daughter_max_dist_um = daughter_max_dist_um

    @staticmethod
    def _get_node_pos_um(data: Dict[str, Any]) -> Tuple[float, float, float]:
        """Convert node coordinates to micrometers."""
        z = float(data.get("z", 0.0)) * SCALE_Z
        y = float(data.get("y", 0.0)) * SCALE_Y
        x = float(data.get("x", 0.0)) * SCALE_X
        return (z, y, x)

    @staticmethod
    def _dist_um(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        dz = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dx = p1[2] - p2[2]
        return float(np.sqrt(dz * dz + dy * dy + dx * dx))

    def evaluate(self, gt_graph: nx.DiGraph, pred_graph: nx.DiGraph) -> Dict[str, Any]:
        """Compute Edge Jaccard and Division Jaccard between GT and Predicted graphs."""
        n_gt_nodes = gt_graph.number_of_nodes()
        n_pred_nodes = pred_graph.number_of_nodes()
        n_gt_edges = gt_graph.number_of_edges()
        n_pred_edges = pred_graph.number_of_edges()

        if n_gt_edges == 0 and n_pred_edges == 0:
            return {
                "edge_jaccard": 1.0,
                "division_jaccard": 1.0,
                "edge_tp": 0,
                "edge_fp": 0,
                "edge_fn": 0,
                "division_tp": 0,
                "division_fp": 0,
                "division_fn": 0,
                "n_gt_nodes": n_gt_nodes,
                "n_pred_nodes": n_pred_nodes,
                "n_gt_edges": n_gt_edges,
                "n_pred_edges": n_pred_edges,
            }

        # 1. Edge Matching grouped by transition (t_src -> t_dst)
        gt_edges_by_transition: Dict[Tuple[int, int], List[Tuple[Any, Any]]] = {}
        for u, v in gt_graph.edges():
            t_u = int(gt_graph.nodes[u].get("t", 0))
            t_v = int(gt_graph.nodes[v].get("t", 0))
            gt_edges_by_transition.setdefault((t_u, t_v), []).append((u, v))

        pred_edges_by_transition: Dict[Tuple[int, int], List[Tuple[Any, Any]]] = {}
        for u, v in pred_graph.edges():
            t_u = int(pred_graph.nodes[u].get("t", 0))
            t_v = int(pred_graph.nodes[v].get("t", 0))
            pred_edges_by_transition.setdefault((t_u, t_v), []).append((u, v))

        total_edge_tp = 0
        all_transitions = set(gt_edges_by_transition.keys()).union(
            pred_edges_by_transition.keys()
        )

        for trans in all_transitions:
            gt_edges = gt_edges_by_transition.get(trans, [])
            pred_edges = pred_edges_by_transition.get(trans, [])

            if not gt_edges or not pred_edges:
                continue

            # Build cost matrix for bipartite edge matching
            # Cost = average endpoint distance; infinity if either endpoint exceeds threshold
            n_gt_e = len(gt_edges)
            n_pr_e = len(pred_edges)
            cost_matrix = np.full((n_gt_e, n_pr_e), 1e9, dtype=np.float64)

            for i, (g_u, g_v) in enumerate(gt_edges):
                gu_pos = self._get_node_pos_um(gt_graph.nodes[g_u])
                gv_pos = self._get_node_pos_um(gt_graph.nodes[g_v])

                for j, (p_u, p_v) in enumerate(pred_edges):
                    pu_pos = self._get_node_pos_um(pred_graph.nodes[p_u])
                    pv_pos = self._get_node_pos_um(pred_graph.nodes[p_v])

                    d_src = self._dist_um(gu_pos, pu_pos)
                    d_dst = self._dist_um(gv_pos, pv_pos)

                    if d_src <= self.edge_tolerance_um and d_dst <= self.edge_tolerance_um:
                        cost_matrix[i, j] = (d_src + d_dst) / 2.0

            # Solve Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e8:
                    total_edge_tp += 1

        edge_fp = max(0, n_pred_edges - total_edge_tp)
        edge_fn = max(0, n_gt_edges - total_edge_tp)
        denom = total_edge_tp + edge_fp + edge_fn
        raw_edge_jaccard = (total_edge_tp / denom) if denom > 0 else 0.0

        # Penalty factor for node count over-prediction
        node_penalty = min(1.0, n_gt_nodes / n_pred_nodes) if n_pred_nodes > 0 else 0.0
        edge_jaccard = raw_edge_jaccard * node_penalty

        # 2. Division Jaccard
        gt_divisions = self._extract_divisions(gt_graph)
        pred_divisions = self._extract_divisions(pred_graph)

        div_tp = 0
        matched_gt_div = set()
        matched_pred_div = set()

        for p_idx, p_div in enumerate(pred_divisions):
            p_parent_pos = p_div["parent_pos"]
            p_t = p_div["t"]
            p_daughters = p_div["daughter_positions"]

            best_match_idx = None
            best_cost = 1e9

            for g_idx, g_div in enumerate(gt_divisions):
                if g_idx in matched_gt_div:
                    continue
                if g_div["t"] != p_t:
                    continue

                d_parent = self._dist_um(p_parent_pos, g_div["parent_pos"])
                if d_parent > self.edge_tolerance_um:
                    continue

                # Match daughters (2 daughters)
                g_daughters = g_div["daughter_positions"]
                if len(p_daughters) >= 2 and len(g_daughters) >= 2:
                    # Check both permutations
                    cost_perm1 = (
                        self._dist_um(p_daughters[0], g_daughters[0])
                        + self._dist_um(p_daughters[1], g_daughters[1])
                    )
                    cost_perm2 = (
                        self._dist_um(p_daughters[0], g_daughters[1])
                        + self._dist_um(p_daughters[1], g_daughters[0])
                    )
                    min_cost = min(cost_perm1, cost_perm2)
                    max_single_daughter_dist = max(
                        self._dist_um(p_daughters[0], g_daughters[0]),
                        self._dist_um(p_daughters[1], g_daughters[1]),
                    ) if cost_perm1 < cost_perm2 else max(
                        self._dist_um(p_daughters[0], g_daughters[1]),
                        self._dist_um(p_daughters[1], g_daughters[0]),
                    )

                    if max_single_daughter_dist <= self.edge_tolerance_um:
                        total_div_cost = d_parent + min_cost
                        if total_div_cost < best_cost:
                            best_cost = total_div_cost
                            best_match_idx = g_idx

            if best_match_idx is not None:
                div_tp += 1
                matched_gt_div.add(best_match_idx)
                matched_pred_div.add(p_idx)

        n_gt_div = len(gt_divisions)
        n_pr_div = len(pred_divisions)
        div_fp = max(0, n_pr_div - div_tp)
        div_fn = max(0, n_gt_div - div_tp)
        div_denom = div_tp + div_fp + div_fn
        division_jaccard = (div_tp / div_denom) if div_denom > 0 else (1.0 if (n_gt_div == 0 and n_pr_div == 0) else 0.0)

        return {
            "edge_jaccard": float(edge_jaccard),
            "raw_edge_jaccard": float(raw_edge_jaccard),
            "node_penalty": float(node_penalty),
            "division_jaccard": float(division_jaccard),
            "edge_tp": total_edge_tp,
            "edge_fp": edge_fp,
            "edge_fn": edge_fn,
            "n_gt_edges": n_gt_edges,
            "n_pred_edges": n_pred_edges,
            "n_gt_nodes": n_gt_nodes,
            "n_pred_nodes": n_pred_nodes,
            "division_tp": div_tp,
            "division_fp": div_fp,
            "division_fn": div_fn,
            "n_gt_divisions": n_gt_div,
            "n_pred_divisions": n_pr_div,
        }

    def _extract_divisions(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        divisions = []
        for node_id in graph.nodes():
            successors = list(graph.successors(node_id))
            if len(successors) >= 2:
                parent_data = graph.nodes[node_id]
                parent_pos = self._get_node_pos_um(parent_data)
                daughter_positions = [
                    self._get_node_pos_um(graph.nodes[c]) for c in successors[:2]
                ]
                divisions.append({
                    "parent_id": node_id,
                    "t": int(parent_data.get("t", 0)),
                    "parent_pos": parent_pos,
                    "daughter_positions": daughter_positions,
                })
        return divisions
