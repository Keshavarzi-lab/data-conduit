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

from importlib import import_module
from typing import TYPE_CHECKING

from data_conduit.qc.catalog import QC_STREAMS, build_qc_catalog, qc_datastructure
from data_conduit.qc.slicing import slice_pose, slice_pose_for_trial, slice_pose_per_trial
from data_conduit.qc.trials import parse_events_to_trials

if TYPE_CHECKING:
    from data_conduit.qc.path_plots import DEFAULT_CENTROID_POINTS, add_group_column, plot_path_grid

_PLOT_EXPORTS = {'DEFAULT_CENTROID_POINTS', 'add_group_column', 'plot_path_grid'}

__all__ = [
    'parse_events_to_trials',
    'build_qc_catalog',
    'qc_datastructure',
    'QC_STREAMS',
    'slice_pose',
    'slice_pose_for_trial',
    'slice_pose_per_trial',
    'plot_path_grid',
    'add_group_column',
    'DEFAULT_CENTROID_POINTS',
]


def __getattr__(name):
    if name in _PLOT_EXPORTS:
        module = import_module('data_conduit.qc.path_plots')
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
