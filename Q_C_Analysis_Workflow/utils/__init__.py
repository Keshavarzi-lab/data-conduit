'''
Miscellaneous utility functions designed for the Q_C_Analysis_Workflow setup. These are not necessarily specific to the workflow, and may be useful in other contexts too.

Contents
--------
- interactive_tables
    Helper functions for easily making interactive tables in notebooks. These are used in the design space exploration notebook, but could be used elsewhere too.
'''

#===== Imports
from .bonsai_parsers import parse_poke_outcome
from .collect_session_settings import (
    collect_session_settings_directory,
    make_usid,
    make_utid,
)
from .interactive_tables import display_table
from .inspect_sessions_trials import compare_trial_settings, group_trials, trial_column_summary
from .trial_table import (
    align_trials_with_bonsai,
    build_master_trial_table,
    build_trial_table,
    build_trial_table_for_directory,
    compare_trial_counts,
    count_events_per_session,
    count_premature_pokes_per_directory,
)

__all__ = [
    'align_trials_with_bonsai',
    'build_master_trial_table',
    'build_trial_table',
    'build_trial_table_for_directory',
    'collect_session_settings_directory',
    'compare_trial_counts',
    'compare_trial_settings',
    'count_events_per_session',
    'count_premature_pokes_per_directory',
    'display_table',
    'group_trials',
    'make_usid',
    'make_utid',
    'parse_poke_outcome',
    'trial_column_summary',
]
