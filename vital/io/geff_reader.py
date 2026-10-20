"""GEFF v1.1 Lineage Graph Reader and Writer.

Reads and writes zebrafish cell lineage graphs adhering to the GEFF v1.1 specification.
Converts GEFF stores to/from NetworkX DiGraph objects and structured Pandas DataFrames.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import networkx as nx
import pandas as pd
import geff

from vital.config import SCALE_Z, SCALE_Y, SCALE_X


class GeffReader:
    """Reader for GEFF v1.1 lineage graph stores."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"GEFF store does not exist at: {self.path}")

        # Read using official geff library
        self._graph, self._metadata = geff.read(str(self.path))

    @property
    def graph(self) -> nx.DiGraph:
        """Returns the underlying NetworkX directed graph."""
        return self._graph

    @property
    def metadata(self) -> Any:
        """Returns the GEFF metadata object."""
        return self._metadata

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    def get_nodes_dataframe(self) -> pd.DataFrame:
        """Extract all nodes and their spatial/temporal attributes as a DataFrame."""
        rows = []
        for node_id, data in self._graph.nodes(data=True):
            t = int(data.get("t", 0))
            z = float(data.get("z", 0.0))
            y = float(data.get("y", 0.0))
            x = float(data.get("x", 0.0))
            in_deg = self._graph.in_degree(node_id)
            out_deg = self._graph.out_degree(node_id)

            if in_deg == 0:
                role = "root"
            elif out_deg >= 2:
                role = "division_parent"
            elif out_deg == 0:
                role = "terminal"
            else:
                role = "migration"

            rows.append({
                "node_id": node_id,
                "t": t,
                "z": z,
                "y": y,
                "x": x,
                "z_um": z * SCALE_Z,
                "y_um": y * SCALE_Y,
                "x_um": x * SCALE_X,
                "in_degree": in_deg,
                "out_degree": out_deg,
                "role": role,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values(by=["t", "node_id"], inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def get_edges_dataframe(self) -> pd.DataFrame:
        """Extract all edges with source and target coordinate information."""
        rows = []
        for src, dst in self._graph.edges():
            src_data = self._graph.nodes[src]
            dst_data = self._graph.nodes[dst]
            out_deg = self._graph.out_degree(src)
            edge_type = "division" if out_deg >= 2 else "migration"

            rows.append({
                "source_id": src,
                "target_id": dst,
                "source_t": int(src_data.get("t", 0)),
                "target_t": int(dst_data.get("t", 0)),
                "edge_type": edge_type,
                "source_z": float(src_data.get("z", 0.0)),
                "source_y": float(src_data.get("y", 0.0)),
                "source_x": float(src_data.get("x", 0.0)),
                "target_z": float(dst_data.get("z", 0.0)),
                "target_y": float(dst_data.get("y", 0.0)),
                "target_x": float(dst_data.get("x", 0.0)),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values(by=["source_t", "source_id"], inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def get_division_events(self) -> List[Dict[str, Any]]:
        """Identify all division events where a parent node branches into multiple children."""
        divisions = []
        for node_id, data in self._graph.nodes(data=True):
            successors = list(self._graph.successors(node_id))
            if len(successors) >= 2:
                children_info = [
                    {
                        "child_id": c,
                        "t": self._graph.nodes[c].get("t"),
                        "z": self._graph.nodes[c].get("z"),
                        "y": self._graph.nodes[c].get("y"),
                        "x": self._graph.nodes[c].get("x"),
                    }
                    for c in successors
                ]
                divisions.append({
                    "parent_id": node_id,
                    "t": data.get("t"),
                    "z": data.get("z"),
                    "y": data.get("y"),
                    "x": data.get("x"),
                    "num_daughters": len(successors),
                    "daughters": children_info,
                })
        return sorted(divisions, key=lambda d: d["t"])

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics for this lineage graph."""
        nodes_df = self.get_nodes_dataframe()
        divisions = self.get_division_events()
        roots = [n for n in self._graph.nodes if self._graph.in_degree(n) == 0]

        return {
            "total_nodes": self.num_nodes,
            "total_edges": self.num_edges,
            "total_divisions": len(divisions),
            "total_roots": len(roots),
            "min_t": int(nodes_df["t"].min()) if not nodes_df.empty else 0,
            "max_t": int(nodes_df["t"].max()) if not nodes_df.empty else 0,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"GeffReader(path='{self.path.name}', nodes={s['total_nodes']}, "
            f"edges={s['total_edges']}, divisions={s['total_divisions']})"
        )


class GeffWriter:
    """Writer for GEFF v1.1 lineage graphs."""

    @staticmethod
    def write(graph: nx.DiGraph, output_path: Union[str, Path], overwrite: bool = True) -> None:
        """Serializes a NetworkX DiGraph into GEFF v1.1 format."""
        out_path = Path(output_path)
        if out_path.exists() and overwrite:
            import shutil
            shutil.rmtree(out_path)

        geff.write(graph, str(out_path), overwrite=overwrite)
