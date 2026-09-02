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
- parse_events_to_trials: Compatibility name backed by the generic trial parser.
- QC_TRIALS / qc_trial_spec / qc_trials_reader:
                          The same task as a TrialSpec for the general parser.
- build_qc_catalog:       Assemble the Q_C Catalog (readers + configurators).
- qc_datastructure:       Configured DataStructure for a given root + layout.
- QC_STREAMS:             The stream names the Q_C catalog can read.
- slice_pose / ...:       Slice DLC pose by a trial's time windows.
'''

from importlib import import_module
from typing import TYPE_CHECKING

from data_conduit.refactor_qc.catalog import QC_STREAMS, build_qc_catalog, qc_datastructure
from data_conduit.refactor_qc.slicing import slice_pose, slice_pose_for_trial, slice_pose_per_trial
from data_conduit.refactor_qc.training_filter import (
    TRAINING_DETAIL,
    TRAINING_FIRST_MID_LAST_DETAIL,
    TRAINING_PHASE_LABELS,
    filter_trials,
    summarise_training_filter,
    training_session_names,
    training_spec,
)
from data_conduit.refactor_qc.trial_spec import QC_TRIALS, qc_trial_spec, qc_trials_reader
from data_conduit.refactor_qc.trials import parse_events_to_trials

if TYPE_CHECKING:
    from data_conduit.refactor_qc.path_plots import DEFAULT_CENTROID_POINTS, add_group_column, diagnose_path_grid, plot_path_grid

_PLOT_EXPORTS = {'DEFAULT_CENTROID_POINTS', 'add_group_column', 'diagnose_path_grid', 'plot_path_grid'}

__all__ = [
    'parse_events_to_trials',
    'QC_TRIALS',
    'qc_trial_spec',
    'qc_trials_reader',
    'build_qc_catalog',
    'qc_datastructure',
    'QC_STREAMS',
    'slice_pose',
    'slice_pose_for_trial',
    'slice_pose_per_trial',
    'TRAINING_DETAIL',
    'TRAINING_FIRST_MID_LAST_DETAIL',
    'TRAINING_PHASE_LABELS',
    'training_spec',
    'training_session_names',
    'summarise_training_filter',
    'filter_trials',
    'plot_path_grid',
    'diagnose_path_grid',
    'add_group_column',
    'DEFAULT_CENTROID_POINTS',
]


def __getattr__(name):
    if name in _PLOT_EXPORTS:
        module = import_module('data_conduit.refactor_qc.path_plots')
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
