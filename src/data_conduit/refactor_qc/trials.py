"""Backward-compatible Q_C trial parsing name backed by the generic parser.

The notebook-facing ``parse_events_to_trials`` name is retained so existing Q_C
code does not need to change. Its implementation now builds the Q_C-specific
``TrialSpec`` and delegates all trial parsing to
``data_conduit.datastructures.parse_trials``.
"""

################################################################################
# Imports
################################################################################

import pandas as pd

from data_conduit.datastructures import parse_trials
from data_conduit.refactor_qc.trial_spec import qc_trial_spec

################################################################################


__all__ = ["parse_events_to_trials"]


################################################################################
# Public API
################################################################################


def parse_events_to_trials(
    events_df: pd.DataFrame,
    *,
    nosepoke_count: int = 18,
    trial_start_buffer: float = 1.0,
) -> pd.DataFrame:
    """Parse a Q_C event log with the generic ``TrialSpec`` machinery.

    This compatibility function preserves the established Q_C function name and
    arguments while routing the work through the refactored generic parser.

    Parameters
    ----------
    events_df : pandas.DataFrame
        One session's time-indexed ExperimentEvents table with an ``Event``
        column.
    nosepoke_count : int
        Number of arena ports used when calculating angular offsets. Default 18.
    trial_start_buffer : float
        Seconds added between the preceding trial end and the next trial start.
        Default 1.0; pass 0.0 for contiguous trial times.

    Returns
    -------
    pandas.DataFrame
        One row per Q_C trial, produced by
        ``data_conduit.datastructures.parse_trials``.

    Notes
    -----
    The generic parser deliberately assigns a closing poke to only the trial it
    closes when adjacent trials share a zero-buffer boundary. This can differ
    from the former Q_C parser, which included that poke in both event windows.
    """

    spec = qc_trial_spec(
        nosepoke_count=nosepoke_count,
        trial_start_buffer=trial_start_buffer,
    )
    return parse_trials(events_df, spec)


################################################################################
