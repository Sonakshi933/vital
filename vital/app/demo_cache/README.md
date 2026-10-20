# V.I.T.A.L Precomputed Demo Cache

This directory contains pre-processed demo recordings optimized for instant, zero-latency exploration
in V.I.T.A.L Demo Mode.

## Selected Recordings & Biological Rationale

1. **`6bba_48816121`** (Primary Showcase)
   - **Mitotic Divisions**: 5 distinct division events across developmental time
   - **Root Lineages**: 17 progenitor cell lines
   - **Total Nodes**: 935 spatio-temporal cell instances
   - **Rationale**: Highest number of cell divisions in the CZ Biohub training dataset,
     demonstrating rich tree branching and multi-generational lineages.

2. **`6bba_df673a83`** (Clean Developmental Bifurcations)
   - **Mitotic Divisions**: 4 distinct division events
   - **Root Lineages**: 14 progenitor cell lines
   - **Total Nodes**: 668 spatio-temporal cell instances
   - **Rationale**: Compact, clean volume with high SNR and distinct cell daughter bifurcations.

3. **`6bba_4ffd3da3`** (Robust Branching Dynamics)
   - **Mitotic Divisions**: 3 distinct division events
   - **Root Lineages**: 17 progenitor cell lines
   - **Total Nodes**: 706 spatio-temporal cell instances
   - **Rationale**: Shows diverse cell speeds and migration trajectories alongside mitosis.

Each subfolder contains:
- `cells_stage1.csv`: Detected 3D cell centroids, volume, intensity
- `cells_stage2.csv`: Persistent track IDs, parent IDs, migration & division events, velocities
- `lineage_output.geff`: GEFF v1.1 lineage graph format
- `summary_report.json`: Lineage summary statistics and ground-truth comparison metrics
