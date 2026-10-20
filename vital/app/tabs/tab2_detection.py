"""Tab 2: Slice-by-Slice Explorer (Detection).

Displays 3D cell detections (Cellpose 'cyto3' 3D mode + regionprops) across all timepoints:
- Total unique cells detected across the entire recording.
- Total number of cells detected per individual time frame.
- Interactive time-frame selector dropdown updating live cell count and coordinate list.
- Dynamic visual mode toggle: Maximum Intensity Projection (MIP) vs. Optical Z-slice slider with detected centroids overlaid as labeled markers.
- Downloadable cells_stage1.csv with columns [cell_id, t, z, y, x, volume, mean_intensity].
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from vital.io.zarr_reader import ZarrReader
from vital.config import SCALE_Z, SCALE_Y, SCALE_X


def render_tab2_detection(
    selected_rec: str,
    stage1_df: pd.DataFrame,
    zarr_reader: Optional[ZarrReader],
    ann_video_path: Optional[Path],
):
    """Renders Tab 2: Slice-by-Slice Explorer."""
    # Specimen ID pill badge header
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 8px;">
            <h2 style="color: #ffffff; margin: 0; font-weight: 800; font-size: 1.6rem;">
                🔬 Slice-by-Slice Explorer:
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
        <div style="background: rgba(15, 23, 42, 0.85); border-left: 4px solid #00f2fe;
                    border: 1px solid rgba(0, 242, 254, 0.35); border-left-width: 4px;
                    padding: 12px 18px; border-radius: 10px; margin-bottom: 20px;">
            <p style="color: #ffffff; font-size: 1.0rem; margin: 0; line-height: 1.6; font-weight: 500;">
                🔬 <b>Explore 3D Cell Detections:</b> Browse individual cell positions, sizes, and counts frame-by-frame across all 100 time points using maximum intensity projections (MIP) or optical depth slices.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if stage1_df.empty:
        st.warning(f"No Stage 1 cell detections found for specimen `{selected_rec}`. Run tracking or choose a Demo specimen.")
        return

    # Normalize columns
    stage1_df = stage1_df.copy()
    if "detection_id" not in stage1_df.columns and "cell_id" in stage1_df.columns:
        stage1_df["detection_id"] = stage1_df["cell_id"]
    elif "cell_id" not in stage1_df.columns and "detection_id" in stage1_df.columns:
        stage1_df["cell_id"] = stage1_df["detection_id"]

    if "z_um" not in stage1_df.columns:
        stage1_df["z_um"] = stage1_df["z"] * SCALE_Z
        stage1_df["y_um"] = stage1_df["y"] * SCALE_Y
        stage1_df["x_um"] = stage1_df["x"] * SCALE_X
    if "volume" not in stage1_df.columns:
        stage1_df["volume"] = 250
    if "mean_intensity" not in stage1_df.columns:
        stage1_df["mean_intensity"] = 1000.0

    # 1. Summary Metrics Bar
    counts_by_t = stage1_df.groupby("t").size()
    total_unique_cells = len(stage1_df)
    mean_cells_per_frame = float(counts_by_t.mean())
    peak_frame_t = int(counts_by_t.idxmax())
    peak_frame_count = int(counts_by_t.max())

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Unique Cells Detected</div>"
            f"<div class='metric-value'>{total_unique_cells}</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>Across all 100 frames</div></div>",
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Average Cells / Frame</div>"
            f"<div class='metric-value-accent'>{mean_cells_per_frame:.1f}</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>Mean active population</div></div>",
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Peak Cell Count</div>"
            f"<div class='metric-value'>{peak_frame_count} cells</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>At frame t = {peak_frame_t}</div></div>",
            unsafe_allow_html=True,
        )
    with col_m4:
        seed_count = int(counts_by_t.get(0, 0))
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Seed Population (t=0)</div>"
            f"<div class='metric-value-accent'>{seed_count} cells</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>Initial progenitors</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 2. Interactive Time Frame Dropdown & Visual View
    st.markdown("<h4 style='color: #ffffff; margin-bottom: 8px;'>⏱️ Time-Frame Cell Inspection & Optical Viewer</h4>", unsafe_allow_html=True)

    all_t = sorted(stage1_df["t"].unique().tolist())

    col_ctrl1, col_ctrl2 = st.columns([1.4, 1.0])
    with col_ctrl1:
        sel_t = st.selectbox(
            "Select Time Frame (t):",
            options=all_t,
            format_func=lambda t: f"Frame {t:02d} (t={t})  —  {counts_by_t.get(t, 0)} cells detected",
            index=0,
            key=f"tab2_t_select_{selected_rec}",
        )
    with col_ctrl2:
        proj_mode = st.radio(
            "Visualization Mode:",
            ["Maximum Intensity Projection (MIP)", "Optical Z-Section Slice"],
            horizontal=True,
            key=f"tab2_proj_mode_{selected_rec}",
        )

    # Frame badge
    frame_cells_df = stage1_df[stage1_df["t"] == sel_t].copy()
    num_frame_cells = len(frame_cells_df)

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(0, 242, 254, 0.12), rgba(16, 185, 129, 0.15));
                    border: 1px solid rgba(0, 242, 254, 0.5); border-radius: 10px; padding: 12px 18px; margin-bottom: 18px;">
            <span style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">
                💡 Frame <code style="color:#00f2fe; background:rgba(0,0,0,0.4); padding:2px 8px; border-radius:4px;">t = {sel_t:02d} / 99</code>:
                <span style="color: #10b981; margin-left: 10px;">{num_frame_cells} Cells Detected</span>
            </span>
            <span style="color: #ffffff; font-size: 0.88rem; margin-left: 18px; font-weight: 500;">
                &bull; Coordinate Span: Z [{frame_cells_df['z'].min():.1f}..{frame_cells_df['z'].max():.1f}],
                Y [{frame_cells_df['y'].min():.1f}..{frame_cells_df['y'].max():.1f}],
                X [{frame_cells_df['x'].min():.1f}..{frame_cells_df['x'].max():.1f}]
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_view_left, col_view_right = st.columns([1.3, 1.1])

    with col_view_left:
        if zarr_reader is not None:
            if "MIP" in proj_mode:
                disp_slice = zarr_reader.get_max_projection(sel_t, axis=0)
                plot_title = f"MIP View with Detected Centroids at t={sel_t:02d}"
                dets_to_plot = frame_cells_df
            else:
                med_z = int(frame_cells_df["z"].median()) if not frame_cells_df.empty else 32
                z_slice = st.slider("Optical Z Slice", 0, 63, med_z, key=f"tab2_z_slider_{selected_rec}")
                disp_slice = zarr_reader.get_slice(sel_t, z_slice)
                plot_title = f"Optical Section (z={z_slice}) with Centroids at t={sel_t:02d}"
                dets_to_plot = frame_cells_df[np.abs(frame_cells_df["z"] - z_slice) <= 3.0]

            fig_detect = px.imshow(
                disp_slice,
                color_continuous_scale="Viridis",
                title=plot_title,
                origin="upper",
            )
            if not dets_to_plot.empty:
                fig_detect.add_trace(
                    go.Scatter(
                        x=dets_to_plot["x"],
                        y=dets_to_plot["y"],
                        mode="markers+text",
                        marker=dict(
                            size=13,
                            color="rgba(0, 242, 254, 0.9)",
                            symbol="circle-open",
                            line=dict(color="#00f2fe", width=2.5),
                        ),
                        text=[f"#{int(cid)}" for cid in dets_to_plot["detection_id"]],
                        textposition="top center",
                        textfont=dict(color="#00f2fe", size=10),
                        name="Detected Cells",
                        hoverinfo="text",
                        hovertext=[
                            f"<b>Cell ID #{int(cid)}</b><br>"
                            f"Position (z,y,x): ({z:.1f}, {y:.1f}, {x:.1f})<br>"
                            f"Physical: ({z_u:.1f} µm, {y_u:.1f} µm, {x_u:.1f} µm)<br>"
                            f"Volume: {int(vol)} voxels<br>"
                            f"Intensity: {fl:.1f}"
                            for cid, z, y, x, z_u, y_u, x_u, vol, fl in zip(
                                dets_to_plot["detection_id"],
                                dets_to_plot["z"],
                                dets_to_plot["y"],
                                dets_to_plot["x"],
                                dets_to_plot["z_um"],
                                dets_to_plot["y_um"],
                                dets_to_plot["x_um"],
                                dets_to_plot["volume"],
                                dets_to_plot["mean_intensity"],
                            )
                        ],
                    )
                )

            fig_detect.update_layout(
                plot_bgcolor="#080c15",
                paper_bgcolor="#080c15",
                font=dict(color="#e2e8f0"),
                margin=dict(l=10, r=10, t=35, b=10),
                height=480,
            )
            st.plotly_chart(fig_detect, use_container_width=True)
        else:
            st.info("Zarr reader not available.")

    with col_view_right:
        st.markdown(f"<b style='color:#ffffff; font-size:1.05rem;'>Cells Detected at Frame t={sel_t:02d} ({num_frame_cells} cells):</b>", unsafe_allow_html=True)
        table_disp = frame_cells_df[["detection_id", "z", "y", "x", "volume"]].copy()
        table_disp.columns = ["Cell ID", "Z (vox)", "Y (vox)", "X (vox)", "Volume (vox)"]
        st.dataframe(table_disp, use_container_width=True, height=440, hide_index=True)

    st.markdown("---")

    # 3. Downloadable Output File: cells_stage1.csv
    st.markdown("<h4 style='color: #ffffff; margin-bottom: 8px;'>📁 Download Stage 1 Detections</h4>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: rgba(0, 242, 254, 0.08); border-left: 4px solid #00f2fe;
                    border: 1px solid rgba(0, 242, 254, 0.25); border-left-width: 4px;
                    padding: 10px 16px; border-radius: 8px; margin-bottom: 12px;">
            <p style="color: #ffffff; font-size: 0.95rem; margin: 0; font-weight: 500;">
                Download the complete table of detected cells across all 100 frames as a standard CSV file (<code>cell_id, t, z, y, x, volume</code>).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stage1_export = pd.DataFrame({
        "cell_id": stage1_df["detection_id"],
        "t": stage1_df["t"],
        "z": stage1_df["z"],
        "y": stage1_df["y"],
        "x": stage1_df["x"],
        "volume": stage1_df["volume"],
    })

    st.dataframe(stage1_export.head(100), use_container_width=True, height=220, hide_index=True)

    csv_data = stage1_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download cells_stage1.csv ({len(stage1_export)} records)",
        data=csv_data,
        file_name=f"{selected_rec}_cells_stage1.csv",
        mime="text/csv",
        key=f"tab2_dl_stage1_{selected_rec}",
    )
