"""Tab 1: Embryo Timelapse Cinema.

Dual-stream biological and AI timelapse playback:
1. Raw Fluorescence Microscopy Movie: pure, unannotated biological light-sheet recording
   collapsed via Maximum Intensity Projection (MIP) across all 64 optical depth slices.
2. AI Tracked & Annotated Lineage Movie: overlaid with persistent cell IDs, Hungarian trajectories,
   and mitotic division markers.
Includes glowing download buttons for both .mp4 videos and an interactive frame-by-frame
freeze scrubber with live specimen diagnostics and active cell coordinates.
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


def render_tab1_cinema(
    selected_rec: str,
    zarr_reader: Optional[ZarrReader],
    raw_video_path: Optional[Path],
    ann_video_path: Optional[Path] = None,
    stage2_df: Optional[pd.DataFrame] = None,
):
    """Renders Tab 1: Embryo Timelapse Cinema with both Raw & Annotated movies."""
    # Specimen ID pill badge header
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 8px;">
            <h2 style="color: #ffffff; margin: 0; font-weight: 800; font-size: 1.6rem;">
                🎬 Embryo Timelapse Cinema:
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
                🎬 <b>Live Embryo Cinema:</b> Compare the raw microscope recording against the AI-tracked cell lineages side-by-side across all 100 time frames (t = 0 to 99).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Side-by-Side Video Players
    col_vid_left, col_vid_right = st.columns([1.2, 1.2])

    with col_vid_left:
        st.markdown("<h4 style='color: #10b981; margin-bottom: 4px;'>🟢 1. Raw Microscopy Video</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color: #ffffff; font-size: 0.95rem; margin-bottom: 10px; font-weight: 500;'>Original, unannotated fluorescence light-sheet recording.</p>", unsafe_allow_html=True)
        if raw_video_path and raw_video_path.exists():
            st.video(str(raw_video_path), loop=True, autoplay=True)
            with open(raw_video_path, "rb") as f:
                st.download_button(
                    "📥 Download Raw Video (.mp4)",
                    data=f.read(),
                    file_name=f"{selected_rec}_raw_timelapse.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    key=f"tab1_dl_raw_{selected_rec}",
                )
        else:
            st.warning("Raw video is currently rendering or unavailable for this specimen.")

    with col_vid_right:
        st.markdown("<h4 style='color: #00f2fe; margin-bottom: 4px;'>🧬 2. AI Tracked & Annotated Video</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color: #ffffff; font-size: 0.95rem; margin-bottom: 10px; font-weight: 500;'>Overlaid with cell IDs, movement tracks, and cell division markers.</p>", unsafe_allow_html=True)
        if ann_video_path and ann_video_path.exists():
            st.video(str(ann_video_path), loop=True, autoplay=True)
            with open(ann_video_path, "rb") as f:
                st.download_button(
                    "📥 Download Annotated Video (.mp4)",
                    data=f.read(),
                    file_name=f"{selected_rec}_annotated_timelapse.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    key=f"tab1_dl_ann_{selected_rec}",
                )
        else:
            st.info("Annotated video will appear once tracking is finalized.")

    st.markdown("---")

    # 2. Interactive Frame Freeze & Scrubber
    st.markdown("<h4 style='color: #ffffff; margin-bottom: 6px;'>🔍 Frame Scrubber & Inspector</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color: #ffffff; font-size: 0.95rem; margin-bottom: 14px; font-weight: 500;'>Freeze playback at any frame (t = 0 to 99) to inspect optical projections and cell locations.</p>", unsafe_allow_html=True)

    col_scrub1, col_scrub2 = st.columns([1.2, 1.0])

    with col_scrub1:
        freeze_t = st.slider("Freeze Frame at Timepoint (t)", 0, 99, 0, key=f"tab1_freeze_slider_{selected_rec}")
        if zarr_reader is not None:
            mip_freeze = zarr_reader.get_max_projection(freeze_t, axis=0)
            fig_freeze = px.imshow(
                mip_freeze,
                color_continuous_scale="Viridis",
                title=f"MIP Projection Frozen at Frame t={freeze_t:02d} / 99",
                origin="upper",
            )
            # Overlay current tracked cell detections if stage2_df is provided
            if stage2_df is not None and not stage2_df.empty:
                cur_dets = stage2_df[stage2_df["t"] == freeze_t]
                if not cur_dets.empty:
                    fig_freeze.add_trace(
                        go.Scatter(
                            x=cur_dets["x"],
                            y=cur_dets["y"],
                            mode="markers+text",
                            marker=dict(
                                size=13,
                                color="#00f2fe",
                                symbol="circle-open",
                                line=dict(color="#00f2fe", width=2.5),
                            ),
                            text=[f"#{int(d)}" for d in cur_dets["track_id"]],
                            textposition="top center",
                            textfont=dict(color="#00f2fe", size=10),
                            name="Tracked Cells",
                            hoverinfo="text",
                            hovertext=[
                                f"<b>Cell #{int(tid)}</b><br>"
                                f"Voxel (z,y,x): ({z:.1f}, {y:.1f}, {x:.1f})<br>"
                                f"Physical (z,y,x): ({z*SCALE_Z:.1f}, {y*SCALE_Y:.1f}, {x*SCALE_X:.1f}) µm<br>"
                                f"Event: {ev}"
                                for tid, z, y, x, ev in zip(
                                    cur_dets["track_id"],
                                    cur_dets["z"],
                                    cur_dets["y"],
                                    cur_dets["x"],
                                    cur_dets["event"],
                                )
                            ],
                        )
                    )
            fig_freeze.update_layout(
                plot_bgcolor="#080c15",
                paper_bgcolor="#080c15",
                font=dict(color="#e2e8f0"),
                margin=dict(l=10, r=10, t=35, b=10),
                height=480,
            )
            st.plotly_chart(fig_freeze, use_container_width=True)

    with col_scrub2:
        st.markdown(f"<b style='color:#ffffff; font-size:1.05rem;'>Specimen Diagnostics at Frame t={freeze_t:02d}:</b>", unsafe_allow_html=True)
        if stage2_df is not None and not stage2_df.empty:
            active_now = stage2_df[stage2_df["t"] == freeze_t].copy()
            div_now = active_now[active_now["event"] == "division"]

            if "z_um" not in active_now.columns:
                active_now["z_um"] = active_now["z"] * SCALE_Z
                active_now["y_um"] = active_now["y"] * SCALE_Y
                active_now["x_um"] = active_now["x"] * SCALE_X

            mean_z_um = float(active_now['z_um'].mean()) if not active_now.empty else 0.0
            mean_z_vox = int(round(float(active_now['z'].mean()))) if not active_now.empty else 0

            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(0, 242, 254, 0.4);
                            border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;">
                    <p style="color:#ffffff; font-size:1.02rem; margin-bottom:6px;">
                        &bull; <b>Active Cell Population:</b> <span style="color:#10b981; font-weight:800;">{len(active_now)} cells</span>
                    </p>
                    <p style="color:#ffffff; font-size:1.02rem; margin-bottom:6px;">
                        &bull; <b>Mitotic Divisions in this frame:</b> <span style="color:#f43f5e; font-weight:800;">{len(div_now)}</span>
                    </p>
                    <p style="color:#ffffff; font-size:1.02rem; margin-bottom:0px;">
                        &bull; <b>Mean Coordinate Depth:</b> <span style="color:#00f2fe; font-weight:700;">Z = {mean_z_um:.1f} µm</span> (Slice ~{mean_z_vox})
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not div_now.empty:
                st.markdown(
                    f"""
                    <div style="background: rgba(244, 63, 94, 0.25); border: 1.5px solid #f43f5e;
                                border-radius: 8px; padding: 8px 14px; margin-bottom: 10px; color: #ffffff; font-weight: 700;">
                        🚨 Mitotic Division in progress at frame t={freeze_t}: Track #{div_now['track_id'].iloc[0]}!
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<b style='color:#ffffff; font-size:0.92rem;'>Cells Active in This Frame:</b>", unsafe_allow_html=True)
            if not active_now.empty:
                active_disp = active_now[["track_id", "z_um", "y_um", "x_um", "event"]].copy()
                active_disp.columns = ["Track ID", "Z (µm)", "Y (µm)", "X (µm)", "Event Status"]
                st.dataframe(active_disp, use_container_width=True, height=280, hide_index=True)
            else:
                st.info(f"No cell trajectories active at frame t={freeze_t}.")
        else:
            st.info("Tracking diagnostics will populate when Stage 2 tracks are computed.")
