"""Tab 5: Ground Truth & Accuracy (Evaluation).

Evaluates predicted lineage graphs against ground-truth GEFF v1.1 annotations:
- Only meaningful for train-set specimens with ground truth available; shows clear message for test-set specimens.
- Computes Edge Jaccard: TP / (TP + FP + FN) * min(1.0, N_gt / N_pred) via Hungarian node matching on physical distance (max 7.0 µm).
- Computes Division Jaccard: micro-averaged across all mitotic divisions (1.5 - 16.0 µm daughter tolerance).
- Displays large metric cards and TP/FP/FN breakdown table for both edges and divisions.
"""

from typing import Optional, Dict, Any
import pandas as pd
import streamlit as st

from vital.io.geff_reader import GeffReader


def render_tab5_accuracy(
    selected_rec: str,
    gt_reader: Optional[GeffReader],
    summary_eval: Optional[Dict[str, Any]],
):
    """Renders Tab 5: Ground Truth & Accuracy."""
    # Specimen ID pill badge header
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 8px;">
            <h2 style="color: #ffffff; margin: 0; font-weight: 800; font-size: 1.6rem;">
                🎯 Ground Truth & Accuracy:
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
                🎯 <b>Benchmark Ground-Truth Accuracy:</b> Compare AI predictions against expert biologist annotations using official CZ Biohub competition metrics (Edge Jaccard for track movement and Division Jaccard for mitosis).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Empty-state check: test-set specimen or no ground truth
    if gt_reader is None:
        st.warning(
            f"ℹ️ **No Ground Truth Available for Specimen `{selected_rec}`.**\n\n"
            "This recording is from the unannotated test set (or custom upload) where ground-truth GEFF graphs "
            "are held out for competition scoring. Ground-truth benchmarking is only available for train-set specimens "
            "(e.g., `6bba_48816121`, `6bba_df673a83`, `6bba_4ffd3da3`)."
        )
        return

    eval_data = summary_eval or {}
    edge_jaccard = eval_data.get("edge_jaccard", 0.0)
    div_jaccard = eval_data.get("division_jaccard", 0.0)
    edge_tp = eval_data.get("edge_tp", 0)
    edge_fp = eval_data.get("edge_fp", 0)
    edge_fn = eval_data.get("edge_fn", 0)
    div_tp = eval_data.get("division_tp", 0)
    div_fp = eval_data.get("division_fp", 0)
    div_fn = eval_data.get("division_fn", 0)
    n_gt_nodes = eval_data.get("n_gt_nodes", (gt_reader.num_nodes if gt_reader is not None else 0))
    n_pred_nodes = eval_data.get("n_pred_nodes", 0)

    # 1. Primary Metric Cards
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Edge Jaccard</div>"
            f"<div class='metric-value'>{edge_jaccard:.4f}</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>Primary competition score</div></div>",
            unsafe_allow_html=True,
        )
    with col_e2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Division Jaccard</div>"
            f"<div class='metric-value-accent'>{div_jaccard:.4f}</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>Mitosis event accuracy</div></div>",
            unsafe_allow_html=True,
        )
    with col_e3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Matched Mitoses</div>"
            f"<div class='metric-value'>{div_tp} / {div_tp + div_fn}</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>True Positive divisions</div></div>",
            unsafe_allow_html=True,
        )
    with col_e4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Ground Truth Nodes</div>"
            f"<div class='metric-value-accent'>{n_gt_nodes}</div>"
            f"<div style='color:#ffffff; font-size:0.8rem; margin-top:3px; font-weight:600;'>Expert manual annotations</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 2. Detailed TP / FP / FN Breakdown Table
    st.markdown("<h4 style='color: #ffffff; margin-bottom: 8px;'>📊 Confusion Matrix & Breakdown Table</h4>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: rgba(0, 242, 254, 0.08); border-left: 4px solid #00f2fe;
                    border: 1px solid rgba(0, 242, 254, 0.25); border-left-width: 4px;
                    padding: 10px 16px; border-radius: 8px; margin-bottom: 12px;">
            <p style="color: #ffffff; font-size: 0.95rem; margin: 0; font-weight: 500;">
                📊 <b>Accuracy Breakdown:</b> Counts of true matches (TP), false additions (FP), and missed annotations (FN) for cell movement edges and cell division events.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    edge_precision = (edge_tp / (edge_tp + edge_fp)) if (edge_tp + edge_fp) > 0 else 0.0
    edge_recall = (edge_tp / (edge_tp + edge_fn)) if (edge_tp + edge_fn) > 0 else 0.0
    div_precision = (div_tp / (div_tp + div_fp)) if (div_tp + div_fp) > 0 else 0.0
    div_recall = (div_tp / (div_tp + div_fn)) if (div_tp + div_fn) > 0 else 0.0

    breakdown_df = pd.DataFrame([
        {
            "Metric Category": "Directed Migration Edges",
            "True Positives (TP)": edge_tp,
            "False Positives (FP)": edge_fp,
            "False Negatives (FN)": edge_fn,
            "Precision": f"{edge_precision:.3f}",
            "Recall": f"{edge_recall:.3f}",
            "Jaccard Score": f"{edge_jaccard:.4f}",
        },
        {
            "Metric Category": "Mitotic Division Events",
            "True Positives (TP)": div_tp,
            "False Positives (FP)": div_fp,
            "False Negatives (FN)": div_fn,
            "Precision": f"{div_precision:.3f}",
            "Recall": f"{div_recall:.3f}",
            "Jaccard Score": f"{div_jaccard:.4f}",
        },
    ])

    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 3. Metric Explanation Cards
    st.markdown("<h4 style='color: #ffffff; margin-bottom: 8px;'>📐 How Accuracy Is Measured</h4>", unsafe_allow_html=True)
    col_form1, col_form2 = st.columns(2)

    with col_form1:
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(0, 242, 254, 0.3);
                        border-left: 4px solid #00f2fe; border-radius: 12px; padding: 18px;">
                <h5 style="color: #00f2fe; margin-top: 0; font-size: 1.1rem; font-weight: 700;">1. Edge Jaccard (Tracking Score)</h5>
                <p style="color: #ffffff; font-size: 0.92rem; line-height: 1.6; font-weight: 500;">
                    Measures how accurately cell movements between frames are connected, with a penalty for predicting phantom cells:
                </p>
                <div style="background: rgba(0, 0, 0, 0.4); padding: 10px 14px; border-radius: 6px; margin: 10px 0;">
                    <span style="color: #10b981; font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 700;">
                        Edge Jaccard = [TP / (TP + FP + FN)] &times; min(1.0, N_ground_truth / N_predicted)
                    </span>
                </div>
                <p style="color: #cbd5e1; font-size: 0.88rem; margin: 0; font-weight: 500;">
                    Cells are matched to ground truth if their 3D physical position is within <b>7.0 µm</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_form2:
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(16, 185, 129, 0.3);
                        border-left: 4px solid #10b981; border-radius: 12px; padding: 18px;">
                <h5 style="color: #10b981; margin-top: 0; font-size: 1.1rem; font-weight: 700;">2. Division Jaccard (Mitosis Score)</h5>
                <p style="color: #ffffff; font-size: 0.92rem; line-height: 1.6; font-weight: 500;">
                    Measures how accurately cell splitting events (mitosis) are discovered:
                </p>
                <div style="background: rgba(0, 0, 0, 0.4); padding: 10px 14px; border-radius: 6px; margin: 10px 0;">
                    <span style="color: #10b981; font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 700;">
                        Division Jaccard = TP_div / (TP_div + FP_div + FN_div)
                    </span>
                </div>
                <p style="color: #cbd5e1; font-size: 0.88rem; margin: 0; font-weight: 500;">
                    Counted as a match (TP) only when the parent cell and <b>both daughter cells</b> are correctly identified within <b>1.5 to 16.0 µm</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
