"""Lineage Graph Construction and Ancestral Tree Reconstruction.

Builds dual graph representations from Stage 2 tracking results:
1. Fine-grained Spatiotemporal DiGraph: GEFF v1.1 compatible graph where each node
   is a cell instance at time t with coordinates (t, z, y, x) and edges connect
   consecutive migration steps and mitotic divisions.
2. Track-Hierarchy Family Tree: Progenitor-to-daughter lineage tree where each node
   represents an entire cell lifetime and edges represent lineage divisions.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go


class LineageGraphBuilder:
    """Constructs spatiotemporal and hierarchical lineage graphs from Stage 2 tracks."""

    def __init__(self):
        pass

    def build_spatiotemporal_graph(self, stage2_df: pd.DataFrame) -> nx.DiGraph:
        """Build GEFF-compliant DiGraph where nodes represent cell instances at time t.

        Node ID format: int(t * 1_000_000 + track_id) to ensure unique, predictable IDs.
        Attributes: t, z, y, x, track_id, detection_id, event.
        """
        G = nx.DiGraph()
        if stage2_df.empty:
            return G

        # 1. Add all nodes
        node_id_map = {}  # (t, track_id) -> node_id
        for _, row in stage2_df.iterrows():
            t = int(row["t"])
            track_id = int(row["track_id"])
            node_id = int(t * 1_000_000 + track_id)
            node_id_map[(t, track_id)] = node_id

            G.add_node(
                node_id,
                t=t,
                z=float(row["z"]),
                y=float(row["y"]),
                x=float(row["x"]),
                track_id=track_id,
                detection_id=int(row.get("detection_id", -1)),
                event=str(row["event"]),
            )

        # 2. Add sequential migration edges within each track
        for track_id, group in stage2_df.groupby("track_id"):
            sorted_group = group.sort_values(by="t")
            rows = sorted_group.to_dict("records")
            for i in range(len(rows) - 1):
                u_t = int(rows[i]["t"])
                v_t = int(rows[i + 1]["t"])
                u = node_id_map[(u_t, track_id)]
                v = node_id_map[(v_t, track_id)]
                G.add_edge(u, v, edge_type="migration")

        # 3. Add division edges from parent terminal node to daughter initial nodes
        # Find daughter tracks with parent_track_id != -1
        daughter_tracks = stage2_df[stage2_df["parent_track_id"] != -1]
        for parent_id, daughter_group in daughter_tracks.groupby("parent_track_id"):
            parent_id = int(parent_id)
            parent_rows = stage2_df[stage2_df["track_id"] == parent_id].sort_values(by="t")
            if parent_rows.empty:
                continue

            last_parent_row = parent_rows.iloc[-1]
            parent_node_id = node_id_map[(int(last_parent_row["t"]), parent_id)]

            # Connect to initial node of each daughter track
            for daughter_id, d_group in daughter_group.groupby("track_id"):
                daughter_id = int(daughter_id)
                first_d_row = d_group.sort_values(by="t").iloc[0]
                daughter_node_id = node_id_map[(int(first_d_row["t"]), daughter_id)]
                G.add_edge(parent_node_id, daughter_node_id, edge_type="division")

        return G

    def build_track_hierarchy_graph(self, stage2_df: pd.DataFrame) -> nx.DiGraph:
        """Build macro family-tree DiGraph where each node represents a full cell track lifetime.

        Nodes: track_id
        Edges: parent_track_id -> daughter_track_id
        """
        tree = nx.DiGraph()
        if stage2_df.empty:
            return tree

        df = stage2_df.copy()
        if "z_um" not in df.columns:
            df["z_um"] = df["z"] * 1.625
            df["y_um"] = df["y"] * 0.40625
            df["x_um"] = df["x"] * 0.40625
        if "displacement_um" not in df.columns:
            df["displacement_um"] = 0.0
        if "event" not in df.columns:
            df["event"] = "migration"
        if "parent_track_id" not in df.columns:
            df["parent_track_id"] = -1

        for track_id, group in df.groupby("track_id"):
            track_id = int(track_id)
            parent_id = int(group["parent_track_id"].iloc[0])
            start_t = int(group["t"].min())
            end_t = int(group["t"].max())
            lifespan = end_t - start_t + 1
            displacements = group["displacement_um"].values
            total_disp = float(np.sum(displacements))
            mean_speed = float(np.mean(displacements)) if len(displacements) > 0 else 0.0
            terminal_event = str(group.sort_values(by="t")["event"].iloc[-1])

            tree.add_node(
                track_id,
                track_id=track_id,
                parent_track_id=parent_id,
                start_t=start_t,
                end_t=end_t,
                lifespan=lifespan,
                total_displacement_um=total_disp,
                mean_speed_um_per_frame=mean_speed,
                terminal_event=terminal_event,
                initial_pos_um=(
                    float(group["z_um"].iloc[0]),
                    float(group["y_um"].iloc[0]),
                    float(group["x_um"].iloc[0]),
                ),
                final_pos_um=(
                    float(group["z_um"].iloc[-1]),
                    float(group["y_um"].iloc[-1]),
                    float(group["x_um"].iloc[-1]),
                ),
            )

        # Add edges for division parentage
        for node, data in tree.nodes(data=True):
            parent = data["parent_track_id"]
            if parent != -1 and parent in tree:
                tree.add_edge(parent, node, edge_type="division")

        return tree

    def compute_lineage_metrics(
        self,
        tree: nx.DiGraph,
        spatiotemporal_graph: Optional[nx.DiGraph] = None,
    ) -> Dict[str, Any]:
        """Compute key biological and structural summary statistics."""
        if tree.number_of_nodes() == 0:
            return {
                "total_tracks": 0,
                "total_roots": 0,
                "total_divisions": 0,
                "max_lineage_depth": 0,
                "avg_lifespan_frames": 0.0,
                "avg_speed_um": 0.0,
            }

        roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
        divisions = [n for n in tree.nodes if tree.out_degree(n) >= 2]

        # Calculate max depth from any root to leaf
        max_depth = 1
        for root in roots:
            lengths = nx.single_source_shortest_path_length(tree, root)
            if lengths:
                max_depth = max(max_depth, max(lengths.values()) + 1)

        lifespans = [data["lifespan"] for _, data in tree.nodes(data=True)]
        speeds = [data["mean_speed_um_per_frame"] for _, data in tree.nodes(data=True)]

        return {
            "total_tracks": tree.number_of_nodes(),
            "total_roots": len(roots),
            "total_divisions": len(divisions),
            "max_lineage_depth": max_depth,
            "avg_lifespan_frames": float(np.mean(lifespans)) if lifespans else 0.0,
            "avg_speed_um": float(np.mean(speeds)) if speeds else 0.0,
        }

    def generate_plotly_lineage_tree(self, tree: nx.DiGraph) -> go.Figure:
        """Create an interactive family-tree visualization using Plotly.

        X-axis: Lineage branch index
        Y-axis: Time (t) progression
        Nodes: Cells, sized by lifespan and colored by track hierarchy
        Edges: Orthogonal branching lines indicating divisions
        """
        fig = go.Figure()
        if tree.number_of_nodes() == 0:
            fig.update_layout(title="No Lineage Graph Data")
            return fig

        # Assign X positions to roots and recursively position descendants
        roots = sorted([n for n in tree.nodes if tree.in_degree(n) == 0])
        x_pos: Dict[int, float] = {}
        current_x = 0.0

        def layout_subtree(node: int, left_x: float) -> float:
            children = sorted(list(tree.successors(node)))
            if not children:
                x_pos[node] = left_x
                return left_x + 1.0

            child_x = left_x
            for child in children:
                child_x = layout_subtree(child, child_x)
            # Parent centered over children
            children_positions = [x_pos[c] for c in children]
            x_pos[node] = (min(children_positions) + max(children_positions)) / 2.0
            return child_x

        for r in roots:
            current_x = layout_subtree(r, current_x) + 0.5

        # Draw Lifespan Bars & Branching Edges
        edge_x, edge_y = [], []
        for u, v in tree.edges():
            u_data = tree.nodes[u]
            v_data = tree.nodes[v]
            # Orthogonal connector from parent end_t to daughter start_t
            p_x = x_pos[u]
            p_y = u_data["end_t"]
            c_x = x_pos[v]
            c_y = v_data["start_t"]

            edge_x.extend([p_x, c_x, c_x, None])
            edge_y.extend([p_y, p_y, c_y, None])

        # Add Branching Lines Trace
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="#00f2fe", width=2.5),
                hoverinfo="none",
                name="Lineage Division",
            )
        )

        # Draw Vertical Lifetime Spans for each cell
        span_x, span_y = [], []
        for node, data in tree.nodes(data=True):
            nx_val = x_pos[node]
            span_x.extend([nx_val, nx_val, None])
            span_y.extend([data["start_t"], data["end_t"], None])

        fig.add_trace(
            go.Scatter(
                x=span_x,
                y=span_y,
                mode="lines",
                line=dict(color="rgba(16, 185, 129, 0.7)", width=4),
                hoverinfo="none",
                name="Cell Lifetime",
            )
        )

        # Add Nodes Trace (Start, Divisions, Terminations)
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for node, data in tree.nodes(data=True):
            nx_val = x_pos[node]
            is_root = data["parent_track_id"] == -1
            num_children = tree.out_degree(node)

            # Mark start point
            node_x.append(nx_val)
            node_y.append(data["start_t"])
            node_size.append(12 if is_root else 8)
            node_color.append("#10b981" if is_root else "#38bdf8")
            node_text.append(
                f"<b>Track {node}</b> (Origin)<br>"
                f"Role: {'Progenitor Root' if is_root else 'Daughter Cell'}<br>"
                f"Parent: {data['parent_track_id']}<br>"
                f"Start frame: t={data['start_t']}<br>"
                f"Lifespan: {data['lifespan']} frames<br>"
                f"Mean speed: {data['mean_speed_um_per_frame']:.2f} µm/frame"
            )

            # If division occurs, mark terminal branching point
            if num_children >= 2:
                node_x.append(nx_val)
                node_y.append(data["end_t"])
                node_size.append(14)
                node_color.append("#f43f5e")  # Bright rose/magenta for division
                node_text.append(
                    f"<b>Track {node}</b> (Mitosis / Division)<br>"
                    f"Frame: t={data['end_t']}<br>"
                    f"Daughters: {list(tree.successors(node))}<br>"
                    f"Lifespan: {data['lifespan']} frames"
                )

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=dict(
                    color=node_color,
                    size=node_size,
                    line=dict(color="#ffffff", width=1.5),
                ),
                text=node_text,
                hoverinfo="text",
                name="Cell Events",
            )
        )

        fig.update_layout(
            title="<b>Ancestral Lineage Family Tree (Branching Mitosis Progression)</b>",
            xaxis=dict(
                title="Lineage Branch Hierarchy",
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                title="Developmental Time (Frame t)",
                autorange="reversed",  # 0 at the top (roots at top, time flows downward)
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.08)",
            ),
            plot_bgcolor="#0b0f19",
            paper_bgcolor="#0b0f19",
            font=dict(color="#e2e8f0"),
            hoverlabel=dict(bgcolor="#1e293b", font_color="#ffffff"),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            height=650,
            margin=dict(l=40, r=40, t=60, b=40),
        )

        return fig

    # Alias for dendrogram rendering compatibility
    render_plotly_dendrogram = generate_plotly_lineage_tree

    def generate_lineage_stage3_dataframe(
        self,
        stage2_df: pd.DataFrame,
        tree: Optional[nx.DiGraph] = None,
    ) -> pd.DataFrame:
        """Constructs Stage 3 lineage CSV with columns:
        [cell_id, t, z, y, x, daughter_ids, new_cell_id, new_position, parent_id, sibling_ids].
        """
        if stage2_df.empty:
            return pd.DataFrame(columns=[
                "cell_id", "t", "z", "y", "x",
                "daughter_ids", "new_cell_id", "new_position", "parent_id", "sibling_ids",
            ])

        if tree is None:
            tree = self.build_track_hierarchy_graph(stage2_df)

        parent_map = {}
        daughters_map = {}
        siblings_map = {}
        first_pos_map = {}

        for tid, grp in stage2_df.groupby("track_id"):
            sorted_grp = grp.sort_values(by="t")
            first_row = sorted_grp.iloc[0]
            first_pos_map[tid] = f"({first_row['x']:.1f}, {first_row['y']:.1f}, {first_row['z']:.1f}, {int(first_row['t'])})"

        for node in tree.nodes():
            parents = list(tree.predecessors(node))
            p_id = str(parents[0]) if parents else ""
            parent_map[node] = p_id

            children = list(tree.successors(node))
            daughters_map[node] = str(children) if children else ""

            if parents:
                sibs = [s for s in tree.successors(parents[0]) if s != node]
                siblings_map[node] = str(sibs) if sibs else ""
            else:
                siblings_map[node] = ""

        rows = []
        for _, row in stage2_df.iterrows():
            tid = int(row["track_id"])
            p_id = str(parent_map.get(tid, ""))
            s_ids = str(siblings_map.get(tid, ""))
            d_ids = str(daughters_map.get(tid, ""))

            new_cid = ""
            new_pos = ""
            children = list(tree.successors(tid)) if tree.has_node(tid) else []
            if children and row.get("event") == "division":
                new_cid = str(children)
                new_pos = "; ".join([f"Track #{c}: {first_pos_map.get(c, '')}" for c in children])

            rows.append({
                "cell_id": tid,
                "t": int(row["t"]),
                "z": round(float(row["z"]), 2),
                "y": round(float(row["y"]), 2),
                "x": round(float(row["x"]), 2),
                "daughter_ids": d_ids,
                "new_cell_id": new_cid,
                "new_position": new_pos,
                "parent_id": p_id,
                "sibling_ids": s_ids,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            for col in ["daughter_ids", "new_cell_id", "new_position", "parent_id", "sibling_ids"]:
                df[col] = df[col].astype(str)
        return df

