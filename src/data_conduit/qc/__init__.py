'''
Q_C experiment helpers for data-conduit.
----------------------------------------

Description:
    Experiment-specific glue for the Q_C nosepoke / pose task: a trial-table
    parser and a configured DataStructure (with trial-table and DLC-alignment
    configurators). The logic is ported from the Q_C_Analysis_Workflow analysis
    repo so it can run on data-conduit's DataStructure / Catalog front end.

Contents:
--------------------------------
- parse_events_to_trials: ExperimentEvents log -> one row per trial.
- build_qc_catalog:       Assemble the Q_C Catalog (readers + configurators).
- qc_datastructure:       Configured DataStructure for a given root + layout.
- QC_STREAMS:             The stream names the Q_C catalog can read.
- slice_pose / ...:       Slice DLC pose by a trial's time windows.
'''

from data_conduit.qc.catalog import QC_STREAMS, build_qc_catalog, qc_datastructure
from data_conduit.qc.slicing import slice_pose, slice_pose_for_trial, slice_pose_per_trial
from data_conduit.qc.trials import parse_events_to_trials

__all__ = [
    'parse_events_to_trials',
    'build_qc_catalog',
    'qc_datastructure',
    'QC_STREAMS',
    'slice_pose',
    'slice_pose_for_trial',
    'slice_pose_per_trial',
]
