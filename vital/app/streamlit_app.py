"""V.I.T.A.L: Visual Identification and Tracking of Ancestral Lineages.

Interactive Full-Stack Scientific Application for 3D+time Zebrafish Light-Sheet Microscopy
(CZ Biohub Kaggle Challenge: "Cell Tracking During Development").

Architectural Design:
- Persistent sidebar with Mode Toggle ("Demo Mode" / "Process New Recording") and Specimen Selector.
- Top Navigation with 5 distinct tabs in exact order:
  1. 🎬 Embryo Timelapse Cinema
  2. 🔬 Slice-by-Slice Explorer (Detection)
  3. 🚀 Trajectories & Dynamics (Tracking)
  4. 🌳 Ancestral Lineage Tree (Division)
  5. 🎯 Ground Truth & Accuracy (Evaluation)
- Dark theme with near-black background, high-contrast white text, green accent states,
  and light-background pill with green monospace font for specimen IDs.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

# Setup Workspace Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from vital.config import (
    TRAIN_DIR,
    TEST_DIR,
    DEMO_CACHE_DIR,
    DEMO_RECORDINGS,
    SCALE_Z,
    SCALE_Y,
    SCALE_X,
    VOXEL_SCALE,
)
from vital.io.zarr_reader import ZarrReader
from vital.io.geff_reader import GeffReader
from vital.io.video_generator import generate_embryo_timelapse_videos
from vital.lineage.graph_builder import LineageGraphBuilder
from scripts.run_pipeline import run_vital_pipeline

# Import Modular 5-Tab Renderers (with dynamic reload to guarantee live code updates)
import importlib
import vital.app.tabs.tab1_cinema as tab1_mod
import vital.app.tabs.tab2_detection as tab2_mod
import vital.app.tabs.tab3_trajectories as tab3_mod
import vital.app.tabs.tab4_lineage as tab4_mod
import vital.app.tabs.tab5_accuracy as tab5_mod

importlib.reload(tab1_mod)
importlib.reload(tab2_mod)
importlib.reload(tab3_mod)
importlib.reload(tab4_mod)
importlib.reload(tab5_mod)

render_tab1_cinema = tab1_mod.render_tab1_cinema
render_tab2_detection = tab2_mod.render_tab2_detection
render_tab3_trajectories = tab3_mod.render_tab3_trajectories
render_tab4_lineage = tab4_mod.render_tab4_lineage
render_tab5_accuracy = tab5_mod.render_tab5_accuracy

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CUSTOM SCIENTIFIC THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="V.I.T.A.L — Zebrafish 3D Cell Lineage Tracker",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Theme with Bioluminescent Accents & High-Contrast White Text
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background-color: #080c15;
            color: #f1f5f9;
        }

        /* Glassmorphic Metric Cards */
        .metric-card {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            margin-bottom: 12px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(16, 185, 129, 0.5);
        }

        .metric-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #cbd5e1;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .metric-value {
            font-size: 1.75rem;
            font-weight: 800;
            color: #ffffff;
        }

        .metric-value-accent {
            font-size: 1.75rem;
            font-weight: 800;
            background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .metric-value-rose {
            font-size: 1.75rem;
            font-weight: 800;
            background: linear-gradient(135deg, #f43f5e 0%, #fb7185 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Top Navigation 5-Tab Bar */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: #0b1120 !important;
            padding: 10px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            margin-bottom: 20px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            background-color: rgba(255, 255, 255, 0.08) !important;
            border: 1.5px solid rgba(255, 255, 255, 0.18) !important;
            padding: 11px 20px !important;
            font-size: 1.02rem !important;
            border-bottom: 3.5px solid transparent !important;
            transition: all 0.2s ease-in-out !important;
        }

        /* Tab names before clicking - High-contrast readable white */
        .stTabs [data-baseweb="tab"],
        .stTabs [data-baseweb="tab"] p,
        .stTabs [data-baseweb="tab"] span,
        .stTabs [data-baseweb="tab"] div {
            color: #f8fafc !important;
            font-weight: 700 !important;
        }

        .stTabs [data-baseweb="tab"]:hover,
        .stTabs [data-baseweb="tab"]:hover p,
        .stTabs [data-baseweb="tab"]:hover span,
        .stTabs [data-baseweb="tab"]:hover div {
            color: #00f2fe !important;
            background-color: rgba(0, 242, 254, 0.16) !important;
            border-color: rgba(0, 242, 254, 0.5) !important;
        }

        .stTabs [aria-selected="true"],
        .stTabs [aria-selected="true"] p,
        .stTabs [aria-selected="true"] span,
        .stTabs [aria-selected="true"] div {
            background-color: rgba(16, 185, 129, 0.25) !important;
            color: #ffffff !important;
            font-weight: 800 !important;
            border: 1.5px solid rgba(16, 185, 129, 0.7) !important;
            border-bottom: 3.5px solid #10b981 !important;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
        }

        /* High-visibility typography for labels, radio buttons, selectboxes, and summaries */
        label[data-testid="stWidgetLabel"],
        .stSelectbox label,
        .stMultiSelect label,
        .stRadio label,
        .stSlider label {
            color: #ffffff !important;
            font-weight: 800 !important;
            font-size: 1.03rem !important;
            letter-spacing: 0.01em !important;
        }

        /* Radio button option labels (e.g. Chart projection) */
        div[data-testid="stRadio"] [role="radiogroup"] label div p,
        div[data-testid="stRadio"] label p,
        div[data-testid="stRadio"] label span,
        div[data-testid="stRadio"] label div {
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 0.98rem !important;
        }

        /* Selectbox and Multiselect text & container */
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] input {
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 0.98rem !important;
        }

        /* High-visibility input container for selecting cells */
        [data-baseweb="select"] > div,
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {
            background-color: #0b1329 !important;
            border: 2px solid #00f2fe !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 14px rgba(0, 242, 254, 0.25) !important;
            min-height: 44px !important;
            transition: all 0.2s ease-in-out !important;
        }

        [data-baseweb="select"] > div:hover,
        [data-baseweb="select"] > div:focus-within {
            border-color: #10b981 !important;
            box-shadow: 0 0 18px rgba(16, 185, 129, 0.5) !important;
        }

        /* Multiselect Tags (Selected Cell Badges) */
        [data-baseweb="tag"],
        span[data-baseweb="tag"],
        div[data-baseweb="tag"] {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.45) 0%, rgba(0, 242, 254, 0.4) 100%) !important;
            border: 1.5px solid #00f2fe !important;
            border-radius: 8px !important;
            padding: 4px 10px !important;
            margin: 3px !important;
            box-shadow: 0 2px 8px rgba(0, 242, 254, 0.35) !important;
        }

        [data-baseweb="tag"] *,
        [data-baseweb="tag"] span,
        [data-baseweb="tag"] div,
        [data-baseweb="tag"] p {
            color: #ffffff !important;
            font-weight: 800 !important;
            font-size: 0.95rem !important;
        }

        [data-baseweb="tag"] svg {
            fill: #ffffff !important;
            color: #ffffff !important;
        }

        [data-baseweb="tag"]:hover {
            border-color: #10b981 !important;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.65) 0%, rgba(0, 242, 254, 0.55) 100%) !important;
            box-shadow: 0 3px 14px rgba(16, 185, 129, 0.6) !important;
        }

        /* Dropdown popover list options when selecting cell IDs */
        ul[role="listbox"],
        div[data-baseweb="popover"],
        div[data-baseweb="menu"] {
            background-color: #080f21 !important;
            border: 2px solid #00f2fe !important;
            border-radius: 10px !important;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.85) !important;
        }

        li[role="option"],
        ul[role="listbox"] li,
        div[data-baseweb="menu"] li {
            background-color: #080f21 !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 0.98rem !important;
            padding: 10px 16px !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        }

        li[role="option"] *,
        ul[role="listbox"] li *,
        div[data-baseweb="menu"] li * {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        li[role="option"]:hover,
        li[role="option"]:hover *,
        ul[role="listbox"] li:hover,
        ul[role="listbox"] li:hover * {
            background-color: rgba(0, 242, 254, 0.25) !important;
            color: #00f2fe !important;
        }

        li[role="option"][aria-selected="true"],
        ul[role="listbox"] li[aria-selected="true"] {
            background-color: rgba(16, 185, 129, 0.3) !important;
            border-left: 4px solid #10b981 !important;
        }

        /* Subtitles, summaries, captions, and descriptions */
        p, .stMarkdown p, .stCaption, [data-testid="stCaptionContainer"] p {
            color: #f1f5f9;
            font-size: 0.98rem;
        }

        /* Gradient Main Header */
        .main-header {
            background: linear-gradient(90deg, #10b981 0%, #00f2fe 50%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0px;
        }

        /* Specimen ID Pill Badge (Light pill with bold green monospace font) */
        .specimen-pill {
            background: #e2e8f0;
            color: #10b981;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 800;
            font-size: 1.05rem;
            padding: 3px 12px;
            border-radius: 9999px;
            box-shadow: 0 2px 6px rgba(16, 185, 129, 0.2);
            display: inline-block;
        }

        /* High-Visibility Download Buttons */
        div[data-testid="stDownloadButton"] > button,
        .stDownloadButton > button {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.45) 0%, rgba(0, 242, 254, 0.45) 100%) !important;
            color: #ffffff !important;
            border: 1.5px solid #10b981 !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            padding: 10px 18px !important;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25) !important;
            transition: all 0.2s ease-in-out !important;
        }

        div[data-testid="stDownloadButton"] > button:hover,
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #10b981 0%, #00f2fe 100%) !important;
            color: #080c15 !important;
            border-color: #ffffff !important;
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6) !important;
            transform: translateY(-2px) !important;
        }

        /* Dataframe styling */
        div[data-testid="stDataFrame"] {
            border: 1.5px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 10px !important;
            background: #0f172a !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# CACHED DATA LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def get_zarr_reader(recording_id: str):
    """Cached loader for 4D microscopy volumes."""
    for root in [TRAIN_DIR, TEST_DIR]:
        path = root / f"{recording_id}.zarr"
        if path.exists():
            return ZarrReader(path)
    return None

@st.cache_resource
def get_gt_reader(recording_id: str):
    """Cached loader for ground-truth GEFF graphs (train set only)."""
    path = TRAIN_DIR / f"{recording_id}.geff"
    if path.exists():
        return GeffReader(path)
    return None

def load_cached_run(recording_dir: Path):
    """Loads stage1, stage2 CSVs and summary reports from disk."""
    s1_path = recording_dir / "cells_stage1.csv"
    s2_path = recording_dir / "cells_stage2.csv"
    rep_path = recording_dir / "summary_report.json"

    stage1_df = pd.read_csv(s1_path) if s1_path.exists() else pd.DataFrame()
    stage2_df = pd.read_csv(s2_path) if s2_path.exists() else pd.DataFrame()
    summary = {}
    if rep_path.exists():
        with open(rep_path, "r") as f:
            summary = json.load(f)

    return stage1_df, stage2_df, summary

def get_or_generate_videos(recording_id: str, rec_dir: Path, stage2_df: pd.DataFrame):
    """Fetch cached timelapse videos or render on-demand."""
    raw_path = rec_dir / "embryo_timelapse_raw.mp4"
    ann_path = rec_dir / "embryo_timelapse_annotated.mp4"

    if raw_path.exists() and ann_path.exists():
        return raw_path, ann_path

    # Generate if missing
    return generate_embryo_timelapse_videos(
        recording_id=recording_id,
        output_dir=rec_dir,
        stage2_df=stage2_df,
        fps=12,
    )


# -----------------------------------------------------------------------------
# PERSISTENT SIDEBAR CONTROLS (SHARED ACROSS ALL 5 TABS)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#10b981; margin-bottom:0px;'>🧬 V.I.T.A.L</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#cbd5e1; font-size:0.88rem; font-weight:500;'>Visual Identification & Tracking of Ancestral Lineages</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Mode Toggle: Demo Mode vs Process New Recording
    app_mode = st.radio(
        "Application Mode",
        options=["⚡ Demo Mode", "🔬 Process New Recording"],
        index=0,
        help="Demo Mode loads pre-processed recordings instantly with zero latency. Process New Recording runs the pipeline on any specimen.",
    )

    st.markdown("---")

    target_run_dir = None

    if "Demo Mode" in app_mode:
        st.markdown("### 🔬 Select Embryo Specimen")
        selected_rec = st.selectbox(
            "Precomputed Demo Recording",
            options=DEMO_RECORDINGS,
            format_func=lambda x: {
                "6bba_48816121": "Specimen 6bba_48816121 (5 Divisions, 17 Roots)",
                "6bba_df673a83": "Specimen 6bba_df673a83 (4 Divisions, 14 Roots)",
                "6bba_4ffd3da3": "Specimen 6bba_4ffd3da3 (3 Divisions, 17 Roots)",
            }.get(x, x),
            help="Precomputed train-set recordings chosen for rich division events.",
        )
        rec_dir = DEMO_CACHE_DIR / selected_rec
        stage1_df, stage2_df, summary = load_cached_run(rec_dir)
        zarr_reader = get_zarr_reader(selected_rec)
        gt_reader = get_gt_reader(selected_rec)
        raw_video_path, ann_video_path = get_or_generate_videos(selected_rec, rec_dir, stage2_df)
        st.success(f"Loaded Demo: **{selected_rec}**")

    else:
        # Process New Recording Mode
        train_zarrs = sorted([p.stem for p in TRAIN_DIR.glob("*.zarr")])
        test_zarrs = sorted([p.stem for p in TEST_DIR.glob("*.zarr")])
        all_options = [f"train: {z}" for z in train_zarrs] + [f"test: {z}" for z in test_zarrs]

        rec_selection = st.selectbox("Select Target Embryo Specimen", options=all_options, index=0)
        split, rec_name = rec_selection.split(": ")
        selected_rec = rec_name

        st.markdown("### Pipeline Execution Controls")
        frame_limit = st.slider("Timepoint Range (t)", 0, 99, (0, 20), help="Frames to process live")
        use_gt_dets = st.checkbox(
            "Benchmark with Ground-Truth Detections (Fast)",
            value=(split == "train"),
            help="Bypasses Cellpose 3D inference for rapid pipeline verification.",
        )

        run_btn = st.button("🚀 Run Live Pipeline", type="primary", use_container_width=True)

        target_run_dir = WORKSPACE_ROOT / "vital" / "runs" / selected_rec
        if run_btn:
            with st.status(f"Processing volume for {selected_rec}...", expanded=True) as status_box:
                def update_prog(msg, pct):
                    st.write(f"[{pct*100:.0f}%] {msg}")

                summary = run_vital_pipeline(
                    recording_id=selected_rec,
                    output_dir=target_run_dir,
                    time_start=frame_limit[0],
                    time_end=frame_limit[1],
                    use_gt_detections=use_gt_dets,
                    progress_callback=update_prog,
                )
                status_box.update(label=f"Pipeline finished for {selected_rec}!", state="complete")
                # Generate videos on demand after pipeline run
                stage2_path = target_run_dir / "cells_stage2.csv"
                s2_df_temp = pd.read_csv(stage2_path) if stage2_path.exists() else None
                get_or_generate_videos(selected_rec, target_run_dir, s2_df_temp)

        rec_dir = target_run_dir
        stage1_df, stage2_df, summary = load_cached_run(rec_dir)
        zarr_reader = get_zarr_reader(selected_rec)
        gt_reader = get_gt_reader(selected_rec)
        raw_path = rec_dir / "embryo_timelapse_raw.mp4"
        ann_path = rec_dir / "embryo_timelapse_annotated.mp4"
        raw_video_path = raw_path if raw_path.exists() else None
        ann_video_path = ann_path if ann_path.exists() else None

    # Physical voxel scale reference
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.5;">
            <b style="color:#10b981;">PHYSICAL VOXEL SCALE:</b><br>
            &bull; Z = 1.625 µm / voxel<br>
            &bull; Y = 0.40625 µm / voxel<br>
            &bull; X = 0.40625 µm / voxel<br>
            &bull; Box: 104 µm &times; 104 µm &times; 104 µm
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# MAIN APP HEADER & TOP METRIC BAR
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style='display: flex; align-items: baseline; flex-wrap: wrap; gap: 12px; margin-bottom: 2px;'>
        <h1 class='main-header' style='font-size: 2.35rem; margin: 0; display: inline-block;'>V.I.T.A.L:</h1>
        <span style='font-size: 1.05rem; font-weight: 600; color: #cbd5e1; letter-spacing: 0.01em;'>
            3D + Time Cell Tracking & Lineage Reconstruction
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='color:#e2e8f0; margin-top:-6px; margin-bottom:20px; font-size:1.02rem;'>"
    f"Active Embryo Specimen: <span class='specimen-pill'>{selected_rec}</span> "
    f"&bull; <b>Volume:</b> <code style='color:#00f2fe; background:rgba(0,0,0,0.5); padding:2px 8px; border-radius:4px;'>104 µm &times; 104 µm &times; 104 µm</code> &bull; 100 Time-Lapse Frames"
    f"</p>",
    unsafe_allow_html=True,
)

# Top Metrics Row
builder = LineageGraphBuilder()
tree = builder.build_track_hierarchy_graph(stage2_df)
metrics = builder.compute_lineage_metrics(tree)

col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
with col_m1:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Total Cells (S1)</div>"
        f"<div class='metric-value'>{len(stage1_df)}</div></div>",
        unsafe_allow_html=True,
    )
with col_m2:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Track Lines</div>"
        f"<div class='metric-value'>{metrics['total_tracks']}</div></div>",
        unsafe_allow_html=True,
    )
with col_m3:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Progenitor Roots</div>"
        f"<div class='metric-value-accent'>{metrics['total_roots']}</div></div>",
        unsafe_allow_html=True,
    )
with col_m4:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Mitotic Divisions</div>"
        f"<div class='metric-value-rose'>{metrics['total_divisions']}</div></div>",
        unsafe_allow_html=True,
    )
with col_m5:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Lineage Depth</div>"
        f"<div class='metric-value'>{metrics['max_lineage_depth']} Gen</div></div>",
        unsafe_allow_html=True,
    )
with col_m6:
    eval_info = summary.get("evaluation", {})
    edge_jaccard_val = eval_info.get("edge_jaccard", None)
    display_jaccard = f"{edge_jaccard_val:.3f}" if edge_jaccard_val is not None else "N/A"
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Edge Jaccard</div>"
        f"<div class='metric-value-accent'>{display_jaccard}</div></div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# MAIN INTERACTIVE TABS (EXACT SPECIFICATION: 5 TABS IN EXACT ORDER)
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "🎬 Embryo Timelapse Cinema",
    "🔬 Slice-by-Slice Explorer",
    "🚀 Trajectories & Dynamics",
    "🌳 Ancestral Lineage Tree",
    "🎯 Ground Truth & Accuracy",
])

with tabs[0]:
    render_tab1_cinema(
        selected_rec=selected_rec,
        zarr_reader=zarr_reader,
        raw_video_path=raw_video_path,
        ann_video_path=ann_video_path,
        stage2_df=stage2_df,
    )

with tabs[1]:
    render_tab2_detection(selected_rec, stage1_df, zarr_reader, ann_video_path)

with tabs[2]:
    render_tab3_trajectories(selected_rec, stage2_df)

with tabs[3]:
    current_run_dir = DEMO_CACHE_DIR / selected_rec if "Demo Mode" in app_mode else (target_run_dir or DEMO_CACHE_DIR / selected_rec)
    render_tab4_lineage(selected_rec, stage2_df, current_run_dir)

with tabs[4]:
    eval_info = summary.get("evaluation", {})
    render_tab5_accuracy(selected_rec, gt_reader, eval_info)
