"""Optional Kaggle Submission Formatter.

Generates submission.csv in Kaggle CZ Biohub schema:
[id, dataset, row_type, node_id, t, z, y, x, source_id, target_id]
"""

from typing import Union, Optional
from pathlib import Path
import pandas as pd
import networkx as nx


class SubmissionWriter:
    """Formats predicted lineage graphs into Kaggle submission.csv format."""

    @staticmethod
    def format_submission(
        dataset_name: str,
        spatiotemporal_graph: nx.DiGraph,
        start_row_id: int = 0,
    ) -> pd.DataFrame:
        """Converts graph into Kaggle submission rows."""
        rows = []
        current_id = start_row_id

        # 1. Node rows
        for node_id, data in spatiotemporal_graph.nodes(data=True):
            rows.append({
                "id": current_id,
                "dataset": dataset_name,
                "row_type": "node",
                "node_id": node_id,
                "t": int(data.get("t", 0)),
                "z": int(round(data.get("z", 0.0))),
                "y": int(round(data.get("y", 0.0))),
                "x": int(round(data.get("x", 0.0))),
                "source_id": -1,
                "target_id": -1,
            })
            current_id += 1

        # 2. Edge rows
        for u, v in spatiotemporal_graph.edges():
            rows.append({
                "id": current_id,
                "dataset": dataset_name,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": u,
                "target_id": v,
            })
            current_id += 1

        return pd.DataFrame(rows)
