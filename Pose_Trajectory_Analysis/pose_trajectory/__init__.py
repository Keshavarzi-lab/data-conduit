'''
pose_trajectory: data pull and formatting for the movement-based pose analysis.
===============================================================================

Description:
    This package does ONE job: pull raw Bonsai/HARP + DeepLabCut session data out
    of disk and reshape it into the tidy objects that the analysis notebooks feed
    into the ``movement`` library. It deliberately performs NO kinematic analysis
    itself. Every ``movement`` call (path length, straightness, head direction,
    speed, and so on) and every plot lives in the notebooks under
    ``Pose_Trajectory_Analysis/notebooks/``, so that the notebooks read as a clear,
    self-contained demonstration of how to use ``movement``.

    The split is intentional and worth keeping:
      * package  -> "get the data, shape it for movement" (this code).
      * notebooks -> "run movement on it, make the figures" (the showcase).

Contents:
--------------------------------
- catalog:              build_catalog() -> a from-scratch DataStructureCatalog of
                        which data objects to extract per session.
- loading:              list_sessions / build_session_group / combine_all /
                        session_movement_dataset -> select sessions and load pose.
- trials:               parse_trials (the lab's canonical parser) + build_summary_table
                        -> the per-trial reference table with inbound/outbound windows.
- infer_poke_positions: infer the 18 nosepoke pixel positions per session, with an
                        accuracy/spread report.
- workspace:            paths, output/figure directories, and a figure saver.

Notes:
    Importing this package adds the repo's ``q_c_data_analysis`` folder to
    ``sys.path`` so the lab's canonical ``parse_trials`` (in
    ``q_c_data_analysis/firstdata/behavioural_metrics/event_parsing.py``) can be
    imported and reused rather than re-implemented. ``data_conduit`` itself is a
    normal installed import (editable in the repo ``.venv``) and needs no path
    juggling.
'''

from __future__ import annotations

import sys
from pathlib import Path

# --- repo location, derived from this file -----------------------------------
# This file lives at <repo>/Pose_Trajectory_Analysis/pose_trajectory/__init__.py,
# so the repository root is three parents up. We resolve it once here and reuse it
# (workspace.py imports REPO_ROOT from here) to avoid every module re-deriving it.
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- make the lab's canonical trial parser importable ------------------------
# parse_trials is the agreed reference implementation and lives in the sibling
# q_c_data_analysis tree. We add that folder to sys.path (once) so trials.py can do
# `from firstdata.behavioural_metrics.event_parsing import parse_trials` and we keep
# a single source of truth instead of copying the parser here.
_QC_ANALYSIS = REPO_ROOT / 'q_c_data_analysis'
if _QC_ANALYSIS.is_dir() and str(_QC_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_QC_ANALYSIS))
