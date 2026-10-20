"""V.I.T.A.L Dashboard Tabs Module.

Exports modular tab renderers for the 5-tab application workflow:
- Tab 1: Embryo Timelapse Cinema
- Tab 2: Slice-by-Slice Explorer (Detection)
- Tab 3: Trajectories & Dynamics (Tracking)
- Tab 4: Ancestral Lineage Tree (Division)
- Tab 5: Ground Truth & Accuracy (Evaluation)
"""

from vital.app.tabs.tab1_cinema import render_tab1_cinema
from vital.app.tabs.tab2_detection import render_tab2_detection
from vital.app.tabs.tab3_trajectories import render_tab3_trajectories
from vital.app.tabs.tab4_lineage import render_tab4_lineage
from vital.app.tabs.tab5_accuracy import render_tab5_accuracy

__all__ = [
    "render_tab1_cinema",
    "render_tab2_detection",
    "render_tab3_trajectories",
    "render_tab4_lineage",
    "render_tab5_accuracy",
]
