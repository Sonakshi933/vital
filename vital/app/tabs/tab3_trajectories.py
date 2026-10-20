"""Tab 3: Trajectories & Dynamics (Tracking).

Links Stage 1 detections across consecutive frames in full 3D+time using Hungarian bipartite matching
(scipy.optimize.linear_sum_assignment, max match cutoff 7.0 µm on physical anisotropic distance).

Features 3 distinct trajectory graphs:
1. Global Trajectory Ensemble: All cell tracks displayed together in a single 3D/2D visualization.
2. Individual Cell Trajectory: Deep-dive into an individual cell path with velocity, displacement, and time steps.
3. Multi-Cell Cohort Trajectories: Select multiple cells (e.g. 3-4 cells of choice) to compare their migration together in a single graph.
Includes downloadable cells_stage2.csv with columns [track_id, t, z, y, x].
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from vital.config import SCALE_Z, SCALE_Y, SCALE_X

# Distinct vibrant color palette for multi-track plotting
COHORT_PALETTE = [
    "#00f2fe", "#10b981", "#f43f5e", "#fbbf24", "#a855f7", 
    "#ec4899", "#38bdf8", "#34d399", "#f97316", "#06b6d4"
]


def render_tab3_trajectories(
    selected_rec: str,
    stage2_df: pd.DataFrame,
):
    """Renders Tab 3: Trajectories & Dynamics with 3 distinct trajectory graphs."""
    # Specimen ID pill badge header
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 8px;">
            <h2 style="color: #ffffff; margin: 0; font-weight: 800; font-size: 1.6rem;">
                🚀 Trajectories & Dynamics:
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
                🚀 <b>Track 3D Cell Migration Over Time:</b> Follow how cells move through the developing embryo across all 100 time points. Inspect individual cell speeds and paths, compare custom cell groups together, or view the complete tissue flow.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if stage2_df.empty:
        st.warning(f"No Stage 2 tracking trajectories found for specimen `{selected_rec}`.")
        return

    # Ensure physical coordinate columns exist
    stage2_df = stage2_df.copy()
    if "z_um" not in stage2_df.columns:
        stage2_df["z_um"] = stage2_df["z"] * SCALE_Z
        stage2_df["y_um"] = stage2_df["y"] * SCALE_Y
        stage2_df["x_um"] = stage2_df["x"] * SCALE_X

    all_tracks = sorted(stage2_df["track_id"].unique().tolist())
    total_tracks = len(all_tracks)
    track_lengths = stage2_df.groupby("track_id").size()
    mean_track_len = float(track_lengths.mean())

    # 1. Overall Trajectory Statistics Cards
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Total Track Lines</div>"
            f"<div class='metric-value'>{total_tracks}</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>Persistent cell paths</div></div>",
            unsafe_allow_html=True,
        )
    with col_t2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Mean Track Lifetime</div>"
            f"<div class='metric-value-accent'>{mean_track_len:.1f} frames</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>Temporal persistence</div></div>",
            unsafe_allow_html=True,
        )
    with col_t3:
        max_len = int(track_lengths.max())
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Longest Track</div>"
            f"<div class='metric-value'>{max_len} frames</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>Max continuous duration</div></div>",
            unsafe_allow_html=True,
        )
    with col_t4:
        total_mig_points = len(stage2_df)
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>3D Linked Points</div>"
            f"<div class='metric-value-accent'>{total_mig_points}</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>Full spatiotemporal points</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =========================================================================
    # GRAPH 1: ALL CELL TRAJECTORIES IN A SINGLE GRAPH (GLOBAL ENSEMBLE)
    # =========================================================================
    st.markdown("<h3 style='color: #00f2fe; font-weight: 800; margin-bottom: 6px;'>🌐 1. Global Trajectory Ensemble (All Cells in a Single Graph)</h3>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background: rgba(0, 242, 254, 0.08); border-left: 4px solid #00f2fe;
                    border: 1px solid rgba(0, 242, 254, 0.3); border-left-width: 4px;
                    padding: 12px 18px; border-radius: 10px; margin-bottom: 14px;">
            <p style="color: #ffffff; font-size: 1.0rem; font-weight: 500; margin: 0; line-height: 1.6;">
                🌐 <b>Global Tissue Flow ({total_tracks} Cells):</b> Displays all cell pathways combined in a single graph, revealing global tissue drift and collective cell migration across developmental time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_all_ctrl1, col_all_ctrl2 = st.columns([1.5, 1.0])
    with col_all_ctrl1:
        all_proj_mode = st.radio(
            "Global Ensemble Projection:",
            ["3D Spatial Line Tracks (All Cells)", "2D Multi-Track Projection (All Cells)"],
            horizontal=True,
            key=f"tab3_all_proj_{selected_rec}",
        )

    if "3D" in all_proj_mode:
        fig_all_3d = go.Figure()
        # Add tracks with bold, luminous lines
        for idx, (tid, grp) in enumerate(stage2_df.groupby("track_id")):
            grp_sorted = grp.sort_values(by="t")
            color_choice = COHORT_PALETTE[idx % len(COHORT_PALETTE)]
            fig_all_3d.add_trace(
                go.Scatter3d(
                    x=grp_sorted["x_um"],
                    y=grp_sorted["y_um"],
                    z=grp_sorted["z_um"],
                    mode="lines+markers",
                    line=dict(color=color_choice, width=5.5),
                    marker=dict(size=3, color=color_choice),
                    name=f"Cell #{tid}",
                    hoverinfo="text",
                    hovertext=[
                        f"<b>Cell #{int(tid)}</b><br>Frame t={int(t)}<br>Position: ({x:.1f}, {y:.1f}, {z:.1f}) µm"
                        for t, x, y, z in zip(grp_sorted["t"], grp_sorted["x_um"], grp_sorted["y_um"], grp_sorted["z_um"])
                    ],
                    showlegend=False,
                )
            )
        fig_all_3d.update_layout(
            title=dict(
                text="<b>3D Spatial Migratory Landscape (All Cells Combined)</b>",
                font=dict(color="#ffffff", size=18),
            ),
            scene=dict(
                xaxis=dict(title=dict(text="X Position (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.22)", showbackground=True, backgroundcolor="rgba(15,23,42,0.65)"),
                yaxis=dict(title=dict(text="Y Position (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.22)", showbackground=True, backgroundcolor="rgba(15,23,42,0.65)"),
                zaxis=dict(title=dict(text="Z Depth (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.22)", showbackground=True, backgroundcolor="rgba(15,23,42,0.65)"),
                bgcolor="#080c15",
            ),
            plot_bgcolor="#080c15",
            paper_bgcolor="#080c15",
            font=dict(color="#ffffff"),
            height=540,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_all_3d, use_container_width=True)

    else:
        # 2D All-Cells Projection
        fig_all_2d = px.line(
            stage2_df.sort_values(by=["track_id", "t"]),
            x="x_um",
            y="y_um",
            color="track_id",
            title="<b>2D Projection of All Cell Tracks (Lateral Y vs X in µm)</b>",
            labels={"x_um": "X Position (µm)", "y_um": "Y Position (µm)", "track_id": "Cell ID"},
        )
        fig_all_2d.update_traces(line=dict(width=3.5))
        fig_all_2d.update_layout(
            title=dict(
                text="<b>2D Projection of All Cell Tracks (Lateral Y vs X in µm)</b>",
                font=dict(color="#ffffff", size=18),
            ),
            xaxis=dict(title=dict(text="X Position (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.18)"),
            yaxis=dict(title=dict(text="Y Position (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.18)"),
            plot_bgcolor="#080c15",
            paper_bgcolor="#080c15",
            font=dict(color="#ffffff"),
            height=540,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_all_2d, use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # GRAPH 2: INDIVIDUAL CELL TRAJECTORY DEEP-DIVE
    # =========================================================================
    st.markdown("<h3 style='color: #10b981; font-weight: 800; margin-bottom: 6px;'>🎯 2. Individual Cell Trajectory Deep-Dive</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: rgba(16, 185, 129, 0.08); border-left: 4px solid #10b981;
                    border: 1px solid rgba(16, 185, 129, 0.35); border-left-width: 4px;
                    padding: 12px 18px; border-radius: 10px; margin-bottom: 14px;">
            <p style="color: #ffffff; font-size: 1.0rem; font-weight: 500; margin: 0; line-height: 1.6;">
                🎯 <b>Single Cell Deep-Dive:</b> Select any cell ID below to view its dedicated 3D migration pathway, total displacement, velocity, and time-resolved steps.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_sel_track, col_view_mode = st.columns([1.2, 1.2])
    with col_sel_track:
        sel_track_id = st.selectbox(
            "Select Individual Cell (Track) ID:",
            options=all_tracks,
            format_func=lambda tid: f"Cell Track #{tid} ({track_lengths.get(tid, 0)} frames tracked)",
            index=0,
            key=f"tab3_track_select_{selected_rec}",
        )

    with col_view_mode:
        traj_mode = st.radio(
            "Individual Chart Projection:",
            ["3D Spatial Line Track (Z vs Y vs X)", "2D Projection Chart (Y vs X)"],
            horizontal=True,
            key=f"tab3_proj_mode_{selected_rec}",
        )

    # Filter data for selected track
    track_data = stage2_df[stage2_df["track_id"] == sel_track_id].sort_values(by="t").copy()

    if not track_data.empty:
        # Displacement & Velocity
        dz = np.diff(track_data["z_um"])
        dy = np.diff(track_data["y_um"])
        dx = np.diff(track_data["x_um"])
        step_dists = np.sqrt(dz**2 + dy**2 + dx**2)
        total_displacement = float(np.sum(step_dists)) if len(step_dists) > 0 else 0.0
        num_frames_tracked = len(track_data)
        avg_velocity = (total_displacement / (num_frames_tracked - 1)) if num_frames_tracked > 1 else 0.0

        net_dz = track_data["z_um"].iloc[-1] - track_data["z_um"].iloc[0]
        net_dy = track_data["y_um"].iloc[-1] - track_data["y_um"].iloc[0]
        net_dx = track_data["x_um"].iloc[-1] - track_data["x_um"].iloc[0]
        net_displacement = float(np.sqrt(net_dz**2 + net_dy**2 + net_dx**2))

        # Metrics banner (Requirement 4: Start->End voxels removed!)
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Displacement", f"{total_displacement:.1f} µm", f"Net: {net_displacement:.1f} µm")
        with col_stat2:
            st.metric("Average Velocity", f"{avg_velocity:.2f} µm/frame")
        with col_stat3:
            st.metric("Frames Tracked", f"{num_frames_tracked} frames", f"t={int(track_data['t'].iloc[0])} to t={int(track_data['t'].iloc[-1])}")

        col_chart_left, col_chart_right = st.columns([1.5, 1.0])

        with col_chart_left:
            if "3D" in traj_mode:
                fig_traj = go.Figure()
                fig_traj.add_trace(
                    go.Scatter3d(
                        x=track_data["x_um"],
                        y=track_data["y_um"],
                        z=track_data["z_um"],
                        mode="lines+markers",
                        line=dict(color="#00f2fe", width=6),
                        marker=dict(
                            size=5,
                            color=track_data["t"],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(
                                title=dict(text="Time (t)", font=dict(color="#ffffff", size=13)),
                                tickfont=dict(color="#ffffff"),
                                len=0.7,
                            ),
                        ),
                        text=[f"t={t}<br>({x:.1f}, {y:.1f}, {z:.1f}) µm" for t, x, y, z in zip(track_data["t"], track_data["x_um"], track_data["y_um"], track_data["z_um"])],
                        hoverinfo="text",
                        name=f"Track #{sel_track_id}",
                    )
                )
                fig_traj.add_trace(
                    go.Scatter3d(
                        x=[track_data["x_um"].iloc[0]],
                        y=[track_data["y_um"].iloc[0]],
                        z=[track_data["z_um"].iloc[0]],
                        mode="markers+text",
                        marker=dict(size=12, color="#10b981", symbol="diamond", line=dict(color="#ffffff", width=2.5)),
                        text=["🚀 START (t=0)"],
                        textposition="top center",
                        textfont=dict(color="#ffffff", size=13, family="Outfit, sans-serif"),
                        name="Start Point",
                    )
                )
                fig_traj.add_trace(
                    go.Scatter3d(
                        x=[track_data["x_um"].iloc[-1]],
                        y=[track_data["y_um"].iloc[-1]],
                        z=[track_data["z_um"].iloc[-1]],
                        mode="markers+text",
                        marker=dict(size=12, color="#f43f5e", symbol="circle", line=dict(color="#ffffff", width=2.5)),
                        text=["🏁 END"],
                        textposition="top center",
                        textfont=dict(color="#ffffff", size=13, family="Outfit, sans-serif"),
                        name="End Point",
                    )
                )
                fig_traj.update_layout(
                    title=dict(
                        text=f"<b>3D Path for Cell #{sel_track_id}</b>",
                        font=dict(color="#ffffff", size=18),
                    ),
                    scene=dict(
                        xaxis=dict(title=dict(text="X (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.22)", showbackground=True, backgroundcolor="rgba(15,23,42,0.65)"),
                        yaxis=dict(title=dict(text="Y (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.22)", showbackground=True, backgroundcolor="rgba(15,23,42,0.65)"),
                        zaxis=dict(title=dict(text="Z Depth (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.22)", showbackground=True, backgroundcolor="rgba(15,23,42,0.65)"),
                        bgcolor="#080c15",
                    ),
                    plot_bgcolor="#080c15",
                    paper_bgcolor="#080c15",
                    font=dict(color="#ffffff"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=460,
                )
                st.plotly_chart(fig_traj, use_container_width=True)

            else:
                fig_2d = px.scatter(
                    track_data,
                    x="x_um",
                    y="y_um",
                    color="t",
                    color_continuous_scale="Plasma",
                    title=f"<b>2D Projection Trajectory for Cell #{sel_track_id}</b>",
                    labels={"x_um": "X Position (µm)", "y_um": "Y Position (µm)", "t": "Frame (t)"},
                )
                fig_2d.add_trace(
                    go.Scatter(
                        x=track_data["x_um"],
                        y=track_data["y_um"],
                        mode="lines",
                        line=dict(color="#00f2fe", width=4),
                        showlegend=False,
                    )
                )
                fig_2d.update_layout(
                    title=dict(
                        text=f"<b>2D Projection Trajectory for Cell #{sel_track_id}</b>",
                        font=dict(color="#ffffff", size=18),
                    ),
                    xaxis=dict(title=dict(text="X Position (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.18)"),
                    yaxis=dict(title=dict(text="Y Position (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.18)"),
                    plot_bgcolor="#080c15",
                    paper_bgcolor="#080c15",
                    font=dict(color="#ffffff"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=460,
                )
                st.plotly_chart(fig_2d, use_container_width=True)

        with col_chart_right:
            st.markdown(f"<b style='color:#ffffff; font-size:1.02rem;'>Time-Resolved Steps for Cell #{sel_track_id}:</b>", unsafe_allow_html=True)
            track_disp = track_data[["t", "z", "y", "x", "z_um", "y_um", "x_um"]].copy()
            track_disp.columns = ["t", "Z (vox)", "Y (vox)", "X (vox)", "Z (µm)", "Y (µm)", "X (µm)"]
            st.dataframe(track_disp, use_container_width=True, height=420, hide_index=True)

    st.markdown("---")

    # =========================================================================
    # GRAPH 3: MULTI-CELL COHORT TRAJECTORIES (SELECT 3-4 SPECIFIC CELLS TOGETHER)
    # =========================================================================
    st.markdown("<h3 style='color: #38bdf8; font-weight: 800; margin-bottom: 6px;'>🔬 3. Multi-Cell Cohort Trajectories (Select Specific Cells Together)</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: rgba(56, 189, 248, 0.08); border-left: 4px solid #38bdf8;
                    border: 1px solid rgba(56, 189, 248, 0.35); border-left-width: 4px;
                    padding: 12px 18px; border-radius: 10px; margin-bottom: 14px;">
            <p style="color: #ffffff; font-size: 1.0rem; font-weight: 500; margin: 0; line-height: 1.6;">
                🔬 <b>Multi-Cell Trajectory Comparison:</b> Choose 3 to 4 cells from the dropdown below to compare their migration pathways side-by-side in a single 3D or 2D chart.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_cohort_sel, col_cohort_proj = st.columns([1.5, 1.0])
    with col_cohort_sel:
        default_cohort = all_tracks[:min(4, len(all_tracks))]
        selected_cohort = st.multiselect(
            "Choose Cells to Compare Together in Single Graph:",
            options=all_tracks,
            default=default_cohort,
            format_func=lambda tid: f"Cell #{tid} ({track_lengths.get(tid, 0)} frames)",
            key=f"tab3_cohort_multiselect_{selected_rec}",
        )

    with col_cohort_proj:
        cohort_proj_mode = st.radio(
            "Cohort Chart Projection:",
            ["3D Spatial Line Tracks", "2D Multi-Cell Projection"],
            horizontal=True,
            key=f"tab3_cohort_proj_{selected_rec}",
        )

    if not selected_cohort:
        st.info("ℹ️ Please select at least one cell ID from the dropdown above to display the comparison graph.")
    else:
        cohort_df = stage2_df[stage2_df["track_id"].isin(selected_cohort)]

        if "3D" in cohort_proj_mode:
            fig_cohort_3d = go.Figure()
            for idx, tid in enumerate(selected_cohort):
                c_data = cohort_df[cohort_df["track_id"] == tid].sort_values(by="t")
                if c_data.empty:
                    continue
                c_color = COHORT_PALETTE[idx % len(COHORT_PALETTE)]

                # Path line
                fig_cohort_3d.add_trace(
                    go.Scatter3d(
                        x=c_data["x_um"],
                        y=c_data["y_um"],
                        z=c_data["z_um"],
                        mode="lines+markers",
                        line=dict(color=c_color, width=6.5),
                        marker=dict(size=4.5, color=c_color, line=dict(color="#ffffff", width=1)),
                        name=f"Cell #{tid}",
                        hoverinfo="text",
                        hovertext=[
                            f"<b>Cell #{tid}</b><br>Frame t={int(t)}<br>Position: ({x:.1f}, {y:.1f}, {z:.1f}) µm"
                            for t, x, y, z in zip(c_data["t"], c_data["x_um"], c_data["y_um"], c_data["z_um"])
                        ],
                    )
                )
                # Origin diamond
                fig_cohort_3d.add_trace(
                    go.Scatter3d(
                        x=[c_data["x_um"].iloc[0]],
                        y=[c_data["y_um"].iloc[0]],
                        z=[c_data["z_um"].iloc[0]],
                        mode="markers+text",
                        marker=dict(size=12, color=c_color, symbol="diamond", line=dict(color="#ffffff", width=2.5)),
                        text=[f"🚀 Cell #{tid} Start"],
                        textposition="top center",
                        textfont=dict(color="#ffffff", size=13, family="Outfit, sans-serif"),
                        showlegend=False,
                    )
                )
                # Terminal circle
                fig_cohort_3d.add_trace(
                    go.Scatter3d(
                        x=[c_data["x_um"].iloc[-1]],
                        y=[c_data["y_um"].iloc[-1]],
                        z=[c_data["z_um"].iloc[-1]],
                        mode="markers+text",
                        marker=dict(size=12, color=c_color, symbol="circle", line=dict(color="#ffffff", width=2.5)),
                        text=[f"🏁 Cell #{tid} End"],
                        textposition="top center",
                        textfont=dict(color="#ffffff", size=13, family="Outfit, sans-serif"),
                        showlegend=False,
                    )
                )

            fig_cohort_3d.update_layout(
                title=dict(
                    text=f"<b>3D Trajectories for Selected Cohort ({len(selected_cohort)} Cells)</b>",
                    font=dict(color="#ffffff", size=18),
                ),
                scene=dict(
                    xaxis=dict(title=dict(text="X (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.22)", showbackground=True, backgroundcolor="rgba(15,23,42,0.65)"),
                    yaxis=dict(title=dict(text="Y (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.22)", showbackground=True, backgroundcolor="rgba(15,23,42,0.65)"),
                    zaxis=dict(title=dict(text="Z Depth (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.22)", showbackground=True, backgroundcolor="rgba(15,23,42,0.65)"),
                    bgcolor="#080c15",
                ),
                plot_bgcolor="#080c15",
                paper_bgcolor="#080c15",
                font=dict(color="#ffffff"),
                legend=dict(font=dict(color="#ffffff", size=12)),
                height=540,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_cohort_3d, use_container_width=True)

        else:
            # 2D Cohort Projection
            fig_cohort_2d = go.Figure()
            for idx, tid in enumerate(selected_cohort):
                c_data = cohort_df[cohort_df["track_id"] == tid].sort_values(by="t")
                if c_data.empty:
                    continue
                c_color = COHORT_PALETTE[idx % len(COHORT_PALETTE)]

                fig_cohort_2d.add_trace(
                    go.Scatter(
                        x=c_data["x_um"],
                        y=c_data["y_um"],
                        mode="lines+markers",
                        line=dict(color=c_color, width=4.5),
                        marker=dict(size=7, color=c_color, line=dict(color="#ffffff", width=1.5)),
                        name=f"Cell #{tid}",
                        hoverinfo="text",
                        hovertext=[
                            f"<b>Cell #{tid}</b><br>Frame t={int(t)}<br>({x:.1f}, {y:.1f}) µm"
                            for t, x, y in zip(c_data["t"], c_data["x_um"], c_data["y_um"])
                        ],
                    )
                )

            fig_cohort_2d.update_layout(
                title=dict(
                    text=f"<b>2D Projection for Selected Cohort ({len(selected_cohort)} Cells)</b>",
                    font=dict(color="#ffffff", size=18),
                ),
                xaxis=dict(title=dict(text="X Position (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.18)"),
                yaxis=dict(title=dict(text="Y Position (µm)", font=dict(color="#ffffff", size=13)), color="#ffffff", gridcolor="rgba(255,255,255,0.18)"),
                plot_bgcolor="#080c15",
                paper_bgcolor="#080c15",
                font=dict(color="#ffffff"),
                legend=dict(font=dict(color="#ffffff", size=12)),
                height=520,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_cohort_2d, use_container_width=True)

        # Comparative Metrics Table for Chosen Cohort
        st.markdown("<b style='color:#ffffff; font-size:1.05rem;'>Cohort Comparison Metrics:</b>", unsafe_allow_html=True)
        cohort_summary_rows = []
        for tid in selected_cohort:
            c_data = cohort_df[cohort_df["track_id"] == tid].sort_values(by="t")
            if c_data.empty:
                continue
            dz = np.diff(c_data["z_um"])
            dy = np.diff(c_data["y_um"])
            dx = np.diff(c_data["x_um"])
            disp = float(np.sum(np.sqrt(dz**2 + dy**2 + dx**2))) if len(dz) > 0 else 0.0
            net_d = float(np.sqrt((c_data["z_um"].iloc[-1] - c_data["z_um"].iloc[0])**2 +
                                  (c_data["y_um"].iloc[-1] - c_data["y_um"].iloc[0])**2 +
                                  (c_data["x_um"].iloc[-1] - c_data["x_um"].iloc[0])**2))
            lifetime = len(c_data)
            avg_spd = (disp / (lifetime - 1)) if lifetime > 1 else 0.0
            cohort_summary_rows.append({
                "Cell ID": f"Cell #{tid}",
                "Start Frame": int(c_data["t"].iloc[0]),
                "End Frame": int(c_data["t"].iloc[-1]),
                "Lifetime (Frames)": lifetime,
                "Total Displacement (µm)": f"{disp:.1f}",
                "Net Displacement (µm)": f"{net_d:.1f}",
                "Average Speed (µm/frame)": f"{avg_spd:.2f}",
            })
        if cohort_summary_rows:
            st.dataframe(pd.DataFrame(cohort_summary_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # =========================================================================
    # 4. DOWNLOADABLE OUTPUT FILE: cells_stage2.csv
    # =========================================================================
    st.markdown("<h4 style='color: #ffffff; margin-bottom: 8px;'>📁 Downloadable Output File: <code>cells_stage2.csv</code></h4>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #e2e8f0; font-size: 0.98rem;'>"
        "Export the complete trajectory records linked across all frames matching the CZ Biohub specification: "
        "<code>[track_id, t, z, y, x]</code>."
        "</p>",
        unsafe_allow_html=True,
    )

    stage2_export = stage2_df[["track_id", "t", "z", "y", "x"]].copy()

    # Track Preview Filter (Cleaned up, no start->end voxels display)
    col_filt_track, col_filt_btn = st.columns([1.5, 1.0])
    with col_filt_track:
        filt_tid = st.selectbox(
            "Filter cells_stage2.csv preview by Track ID:",
            options=["(All Tracks)"] + all_tracks,
            index=0,
            key=f"tab3_preview_filter_{selected_rec}",
        )

    if filt_tid != "(All Tracks)":
        preview_df = stage2_export[stage2_export["track_id"] == filt_tid]
        if not preview_df.empty:
            start_row = preview_df.iloc[0]
            end_row = preview_df.iloc[-1]
            sum_table = pd.DataFrame([{
                "Track ID": f"Cell #{filt_tid}",
                "Start Frame": int(start_row["t"]),
                "End Frame": int(end_row["t"]),
                "Track Duration": f"{len(preview_df)} frames",
            }])
            st.markdown("<b style='color:#10b981;'>Filtered Track Lifetime:</b>", unsafe_allow_html=True)
            st.dataframe(sum_table, use_container_width=True, hide_index=True)
    else:
        preview_df = stage2_export

    st.dataframe(preview_df.head(100), use_container_width=True, height=200, hide_index=True)

    csv_data = stage2_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download cells_stage2.csv ({len(stage2_export)} records)",
        data=csv_data,
        file_name=f"{selected_rec}_cells_stage2.csv",
        mime="text/csv",
        key=f"tab3_dl_stage2_{selected_rec}",
    )
