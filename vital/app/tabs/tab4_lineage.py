"""Tab 4: Ancestral Lineage Tree (Division & Mitosis).

Reconstructs the hierarchical ancestral tree and mitotic divisions:
- Rule-based division detection without ML (1.5 - 16.0 µm geometric proximity for bifurcating daughters).
- Prominently displays the full global specimen lineage tree graph (Plotly dendrogram).
- Dropdown to inspect relationships for any cell ID (parent, siblings, daughters, granddaughters)
  without rendering individual mini-graphs.
- Downloadable output files:
  - lineage_stage3.csv with columns [cell_id, t, z, y, x, daughter_ids, new_cell_id, new_position, parent_id, sibling_ids].
  - lineage_output.geff (standard GEFF v1.1 directed graph zipped for streaming).
"""

import io
import zipfile
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import networkx as nx
import streamlit as st
import plotly.graph_objects as go

from vital.lineage.graph_builder import LineageGraphBuilder


def render_tab4_lineage(
    selected_rec: str,
    stage2_df: pd.DataFrame,
    rec_dir: Path,
):
    """Renders Tab 4: Ancestral Lineage Tree."""
    # Specimen ID pill badge header
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 8px;">
            <h2 style="color: #ffffff; margin: 0; font-weight: 800; font-size: 1.6rem;">
                🌳 Ancestral Lineage Tree:
            </h2>
            <span style="background: #e2e8f0; color: #10b981; font-family: 'JetBrains Mono', monospace;
                         font-weight: 800; font-size: 1.15rem; padding: 4px 14px; border-radius: 9999px;
                         box-shadow: 0 2px 8px rgba(16, 185, 129, 0.25);">
                Specimen {selected_rec}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: rgba(15, 23, 42, 0.85); border-left: 4px solid #10b981;
                    border: 1px solid rgba(16, 185, 129, 0.35); border-left-width: 4px;
                    padding: 12px 18px; border-radius: 10px; margin-bottom: 20px;">
            <p style="color: #ffffff; font-size: 1.0rem; margin: 0; line-height: 1.6; font-weight: 500;">
                🌳 <b>Cell Family Tree & Mitosis:</b> When a parent cell divides, it splits into two daughter cells. This tree reconstructs all developmental generations from initial founder cells to their descendants.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if stage2_df.empty:
        st.warning(f"No tracking data available to construct lineage tree for `{selected_rec}`.")
        return

    builder = LineageGraphBuilder()
    tree = builder.build_track_hierarchy_graph(stage2_df)
    metrics = builder.compute_lineage_metrics(tree)

    # 1. Summary Metrics
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    with col_l1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Progenitor Roots</div>"
            f"<div class='metric-value'>{metrics['total_roots']}</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>Founder ancestral cells</div></div>",
            unsafe_allow_html=True,
        )
    with col_l2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Mitotic Divisions</div>"
            f"<div class='metric-value-rose'>{metrics['total_divisions']}</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>Parent &rarr; 2 Daughters</div></div>",
            unsafe_allow_html=True,
        )
    with col_l3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Lineage Depth</div>"
            f"<div class='metric-value'>{metrics['max_lineage_depth']} Generations</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>Maximum tree depth</div></div>",
            unsafe_allow_html=True,
        )
    with col_l4:
        total_tracks_count = metrics.get('total_tracks', len(tree.nodes()))
        total_divs_count = metrics.get('total_divisions', 0)
        div_ratio_pct = (total_divs_count / total_tracks_count * 100.0) if total_tracks_count > 0 else 0.0
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Dividing Cell Ratio</div>"
            f"<div class='metric-value-accent'>{div_ratio_pct:.1f}%</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>Active mitosis frequency</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =========================================================================
    # 2. MAIN VISUALIZATION: GLOBAL SPECIMEN ANCESTRAL LINEAGE TREE
    # =========================================================================
    st.markdown(
        f"<h3 style='color: #ffffff; font-weight: 800; margin-bottom: 6px;'>"
        f"🌳 Specimen Ancestral Lineage Tree ({len(tree.nodes())} Cells & {metrics['total_divisions']} Mitotic Divisions)"
        f"</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="background: rgba(16, 185, 129, 0.08); border-left: 4px solid #10b981;
                    border: 1px solid rgba(16, 185, 129, 0.25); border-left-width: 4px;
                    padding: 10px 16px; border-radius: 8px; margin-bottom: 14px;">
            <p style="color: #ffffff; font-size: 0.95rem; margin: 0; font-weight: 500;">
                🌳 <b>Specimen Family Tree:</b> Time flows downward from top to bottom. Splitting branches show where cells underwent division into two daughter cells.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_global = builder.generate_plotly_lineage_tree(tree)
    st.plotly_chart(fig_global, use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # 3. INTERACTIVE CELL LINEAGE & RELATIONSHIP INSPECTOR (NO INDIVIDUAL GRAPH)
    # =========================================================================
    st.markdown("<h3 style='color: #ffffff; font-weight: 800; margin-bottom: 6px;'>🔍 Cell Lineage & Relationship Inspector</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: rgba(0, 242, 254, 0.08); border-left: 4px solid #00f2fe;
                    border: 1px solid rgba(0, 242, 254, 0.25); border-left-width: 4px;
                    padding: 10px 16px; border-radius: 8px; margin-bottom: 14px;">
            <p style="color: #ffffff; font-size: 0.95rem; margin: 0; font-weight: 500;">
                🔍 <b>Inspect Cell Lineage:</b> Choose any cell ID below to view its parent, siblings, daughter cells, active lifespan, and speed.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_cell_ids = sorted(list(tree.nodes()))
    dividing_nodes = [n for n in tree.nodes() if tree.out_degree(n) >= 2 or tree.in_degree(n) > 0]

    col_sel_cell, _ = st.columns([1.5, 1.0])
    with col_sel_cell:
        sel_cell_id = st.selectbox(
            "Select Cell ID to Inspect Lineage Relationships:",
            options=all_cell_ids,
            format_func=lambda cid: f"Cell ID #{cid} {'(🌿 Has Lineage / Mitosis)' if cid in dividing_nodes else '(Simple Migration)'}",
            index=0,
            key=f"tab4_cell_select_{selected_rec}",
        )

    has_children = tree.out_degree(sel_cell_id) >= 2
    has_parent = tree.in_degree(sel_cell_id) > 0

    if not has_children and not has_parent:
        st.info(f"ℹ️ **No division/mitosis occurs for this cell**: Cell ID #{sel_cell_id} migrates persistently through the embryonic tissue without undergoing mitosis.")
    else:
        # Details relationship cards
        cell_node_data = tree.nodes[sel_cell_id]
        parents = list(tree.predecessors(sel_cell_id))
        daughters = list(tree.successors(sel_cell_id))
        siblings = [sib for sib in tree.successors(parents[0]) if sib != sel_cell_id] if parents else []
        grandchildren = []
        for d in daughters:
            grandchildren.extend(list(tree.successors(d)))

        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.9); border: 1.5px solid rgba(0, 242, 254, 0.4);
                        border-radius: 12px; padding: 18px 24px; margin-bottom: 14px;">
                <h4 style="color:#00f2fe; margin-top:0; margin-bottom:10px;">
                    🧬 Lineage Record for Cell #{sel_cell_id}
                </h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; font-size: 1.0rem;">
                    <div>
                        <p style="color:#e2e8f0; margin-bottom:6px;">&bull; <b>Cell Role:</b> <span style="color:#10b981; font-weight:700;">{'Progenitor Root' if not parents else 'Daughter Branch'}</span></p>
                        <p style="color:#e2e8f0; margin-bottom:6px;">&bull; <b>Parent Cell:</b> <code style="color:#38bdf8; background:rgba(0,0,0,0.4); padding:2px 8px; border-radius:4px;">{f'Cell #{parents[0]}' if parents else 'None (Progenitor Origin)'}</code></p>
                        <p style="color:#e2e8f0; margin-bottom:0px;">&bull; <b>Sibling Cells:</b> <code style="color:#38bdf8; background:rgba(0,0,0,0.4); padding:2px 8px; border-radius:4px;">{siblings if siblings else 'None'}</code></p>
                    </div>
                    <div>
                        <p style="color:#e2e8f0; margin-bottom:6px;">&bull; <b>Mitotic Divisions:</b> <span style="color:#f43f5e; font-weight:700;">{'Yes (Bifurcates)' if has_children else 'No'}</span></p>
                        <p style="color:#e2e8f0; margin-bottom:6px;">&bull; <b>Daughter Cells:</b> <code style="color:#f43f5e; background:rgba(0,0,0,0.4); padding:2px 8px; border-radius:4px;">{daughters if daughters else 'None'}</code></p>
                        <p style="color:#e2e8f0; margin-bottom:0px;">&bull; <b>Granddaughter Cells:</b> <code style="color:#f43f5e; background:rgba(0,0,0,0.4); padding:2px 8px; border-radius:4px;">{grandchildren if grandchildren else 'None'}</code></p>
                    </div>
                    <div>
                        <p style="color:#e2e8f0; margin-bottom:6px;">&bull; <b>Active Lifetime:</b> <span style="color:#ffffff; font-weight:700;">{cell_node_data.get('lifespan', 0)} frames</span> (t={cell_node_data.get('start_t', 0)} &rarr; t={cell_node_data.get('end_t', 0)})</p>
                        <p style="color:#e2e8f0; margin-bottom:0px;">&bull; <b>Mean Speed:</b> <span style="color:#10b981; font-weight:700;">{cell_node_data.get('mean_speed_um_per_frame', 0.0):.2f} µm/frame</span></p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =========================================================================
    # 4. DOWNLOADABLE FILES: lineage_stage3.csv & lineage_output.geff
    # =========================================================================
    st.markdown("<h4 style='color: #ffffff; margin-bottom: 8px;'>📁 Download Lineage Files</h4>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: rgba(0, 242, 254, 0.08); border-left: 4px solid #00f2fe;
                    border: 1px solid rgba(0, 242, 254, 0.25); border-left-width: 4px;
                    padding: 10px 16px; border-radius: 8px; margin-bottom: 12px;">
            <p style="color: #ffffff; font-size: 0.95rem; margin: 0; font-weight: 500;">
                Download the complete cell lineage table (<code>lineage_stage3.csv</code>) and the standard graph file (<code>lineage_output.geff</code>).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Build lineage_stage3.csv DataFrame
    stage3_df = builder.generate_lineage_stage3_dataframe(stage2_df, tree)
    for col in ["parent_id", "daughter_ids", "new_cell_id", "new_position", "sibling_ids"]:
        if col in stage3_df.columns:
            stage3_df[col] = stage3_df[col].astype(str)

    col_dl1, col_dl2 = st.columns([1.5, 1.2])

    with col_dl1:
        st.markdown("<b style='color:#ffffff; font-size:1.05rem;'>lineage_stage3.csv Table Preview:</b>", unsafe_allow_html=True)
        st.dataframe(stage3_df.head(60), use_container_width=True, height=220, hide_index=True)
        csv_stage3 = stage3_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download lineage_stage3.csv ({len(stage3_df)} records)",
            data=csv_stage3,
            file_name=f"{selected_rec}_lineage_stage3.csv",
            mime="text/csv",
            key=f"tab4_dl_stage3_{selected_rec}",
        )

    with col_dl2:
        st.markdown("<b style='color:#ffffff; font-size:1.05rem;'>GEFF v1.1 Lineage Graph File:</b>", unsafe_allow_html=True)
        geff_path = rec_dir / "lineage_output.geff"
        if geff_path.exists():
            if geff_path.is_dir():
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in geff_path.rglob("*"):
                        if f.is_file():
                            zf.write(f, arcname=f.relative_to(geff_path))
                geff_data = buf.getvalue()
                dl_name = f"{selected_rec}_lineage_output.geff.zip"
                mime_type = "application/zip"
                dl_label = "📥 Download lineage_output.geff (ZIP)"
            else:
                with open(geff_path, "rb") as f:
                    geff_data = f.read()
                dl_name = f"{selected_rec}_lineage_output.geff"
                mime_type = "application/octet-stream"
                dl_label = "📥 Download lineage_output.geff"

            st.download_button(
                label=dl_label,
                data=geff_data,
                file_name=dl_name,
                mime=mime_type,
                key=f"tab4_dl_geff_{selected_rec}",
            )
            st.markdown(f"<p style='color:#cbd5e1; font-size:0.88rem; margin-top:6px;'>File size: <b>{len(geff_data)/1024:.1f} KB</b> &bull; Standard GEFF v1.1 format.</p>", unsafe_allow_html=True)
        else:
            st.info("lineage_output.geff will be written upon running the full pipeline.")
